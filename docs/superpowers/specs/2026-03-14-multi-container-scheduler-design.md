# Phase 6: Multi-Container Scheduler

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Priority queue with fairness (Phase B), designed for future extensibility toward Phase C (dynamic lifecycle, work stealing, distributed graph)

## Overview

Add a multi-container scheduler to applegpu_runtime that accepts evaluation requests from multiple containers, schedules them with priority-based fair queuing on a single GPU device, and enforces per-container resource quotas. The scheduler is always present inside `LazyRuntime` with a default container for backward compatibility.

## Architecture

The scheduler embeds inside `LazyRuntime`, protected by the existing `Mutex<LazyRuntime>` in the Python layer. No second mutex. A default container (`ContainerId(0)`) is auto-created at scheduler construction with the global resource limits, so all existing APIs work unchanged.

```
Python API  ──►  Mutex<LazyRuntime>
                    ├── tensors: HashMap<u64, Tensor>
                    ├── graph: Graph
                    └── scheduler: Scheduler
                          ├── global_limits + global_tracker
                          ├── containers: HashMap<ContainerId, ContainerState>
                          ├── queues: [VecDeque<Job>; 3]  (High, Normal, Low)
                          ├── jobs: HashMap<JobId, JobStatus>
                          └── tensor_owners: HashMap<u64, ContainerId>
```

## Core Types

### ContainerId and JobId

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContainerId(pub(crate) u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub(crate) u64);
```

Both implement `Display`. Each uses a separate `AtomicU64` counter. `ContainerId(0)` is reserved for the default container.

### Priority

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    High = 0,
    Normal = 1,
    Low = 2,
}
```

Used as index into the three-`VecDeque` queue array.

### ContainerConfig

```rust
pub struct ContainerConfig {
    pub priority: Priority,
    pub max_memory_bytes: usize,
    pub max_tensor_count: usize,
    pub max_tensor_size_bytes: usize,  // 0 = inherit from global limits
    pub max_pending_jobs: usize,
}
```

**Mapping to ResourceLimits**: At registration, `ContainerConfig` is converted to a `ResourceLimits`:
- `max_total_memory_bytes` = `config.max_memory_bytes`
- `max_tensor_count` = `config.max_tensor_count`
- `max_tensor_size_bytes` = if `config.max_tensor_size_bytes == 0` then `global_limits.max_tensor_size_bytes` else `config.max_tensor_size_bytes`

### ContainerState

```rust
pub struct ContainerState {
    pub id: ContainerId,
    pub config: ContainerConfig,
    pub limits: ResourceLimits,        // derived from config at registration
    pub tracker: MemoryTracker,        // per-container usage
    pub tensor_ids: HashSet<u64>,      // owned tensor IDs
    pub pending_jobs: usize,
    pub total_jobs_completed: u64,
    pub cumulative_exec_time_ns: u64,  // wall-clock eval time for deficit-based fair queuing
    pub last_scheduled_at: Option<Instant>,
    pub created_at: Instant,
    pub paused: bool,
}
```

### Job

```rust
pub struct Job {
    pub id: JobId,
    pub container_id: ContainerId,
    pub target_tensor_id: u64,
    pub submitted_at: Instant,
    pub priority: Priority,
}
```

### JobStatus

```rust
pub enum JobStatus {
    Queued,
    Running { started_at: Instant },
    Completed { tensor_id: u64, exec_time_ns: u64 },
    Failed { error: String },
}
```

## Scheduler Struct

```rust
pub struct Scheduler {
    global_limits: ResourceLimits,
    global_tracker: MemoryTracker,
    containers: HashMap<ContainerId, ContainerState>,
    queues: [HashMap<ContainerId, VecDeque<Job>>; 3],  // per-container sub-queues, indexed by Priority
    jobs: HashMap<JobId, JobStatus>,
    tensor_owners: HashMap<u64, ContainerId>,  // reverse lookup
    next_container_id: u64,              // counter (not AtomicU64 — behind Mutex)
    next_job_id: u64,                    // counter
    starvation_threshold_ns: u64,       // configurable, default 10 seconds
}
```

Since the scheduler lives inside `Mutex<LazyRuntime>`, counters do not need to be atomic.

**Queue structure**: Each priority tier holds a `HashMap<ContainerId, VecDeque<Job>>` rather than a flat `VecDeque<Job>`. This makes the fairness selection O(C) (number of containers with queued jobs in the tier) rather than O(J) (total jobs). Within each container's sub-queue, jobs are FIFO.

## Public API

### Construction

```rust
impl Scheduler {
    pub fn new(global_limits: ResourceLimits) -> Self;
}
```

Creates the scheduler and auto-registers the default container (`ContainerId(0)`) with `global_limits` as its quota and `Priority::Normal`.

### Container Lifecycle

```rust
pub fn register_container(&mut self, config: ContainerConfig) -> Result<ContainerId>;
pub fn deregister_container(&mut self, id: ContainerId) -> Result<Vec<u64>>;
pub fn pause_container(&mut self, id: ContainerId) -> Result<()>;
pub fn resume_container(&mut self, id: ContainerId) -> Result<()>;
```

- **register**: Validates that the requested quota does not exceed remaining global capacity. "Remaining capacity" is defined conservatively: `global_limits - sum(all registered container quotas)`. This prevents overcommit — one container cannot starve another even if both are at their quota. Returns new `ContainerId`.
- **deregister**: Rejects if any job for this container is in `Running` state. Drains all queued jobs for this container. Returns the list of owned tensor IDs (caller is responsible for freeing them from `LazyRuntime`). Cannot deregister `ContainerId(0)`.
- **pause**: Sets `paused = true`. Running job completes normally. `next_job()` skips paused containers.
- **resume**: Sets `paused = false`.

### Job Submission and Execution

```rust
pub fn submit(&mut self, container_id: ContainerId, target_tensor_id: u64) -> Result<JobId>;
pub fn next_job(&mut self) -> Option<Job>;
pub fn complete_job(&mut self, job_id: JobId, exec_time_ns: u64) -> Result<()>;
pub fn fail_job(&mut self, job_id: JobId, error: String) -> Result<()>;
pub fn job_status(&self, job_id: JobId) -> Option<&JobStatus>;
```

- **submit**: Checks container exists, is not paused, and `pending_jobs < max_pending_jobs`. Creates `Job` with container's current priority. Enqueues in appropriate priority `VecDeque`. Returns `JobId`.
- **next_job**: Scans tiers High -> Normal -> Low. Within a tier, selects the container with the lowest `cumulative_exec_time_ns` (fairness). Dequeues that container's oldest job (FIFO). Updates `last_scheduled_at`. Applies starvation boost (see below). Transitions job to `Running`.
- **complete_job**: Transitions from `Running` to `Completed`. Updates container's `cumulative_exec_time_ns`, `total_jobs_completed`, decrements `pending_jobs`. Returns `JobNotFound` if job is not in `Running` state.
- **fail_job**: Transitions from `Running` to `Failed`. Decrements `pending_jobs`.

### Execution Loop (drives eval)

```rust
pub fn run_next(
    lazy: &mut LazyRuntimeInner,
    device: &Device,
) -> Result<Option<JobId>>;
```

This is a method on `LazyRuntime` (not `Scheduler`) because it needs access to both the scheduler and the tensor/graph state. Flow:

1. `scheduler.next_job()` -> dequeue job (or return `None`)
2. `lazy.eval(device, job.target_tensor_id)` -> execute the graph
3. Measure wall-clock GPU time (`Instant::now()` before/after eval)
4. `scheduler.complete_job(job_id, exec_time_ns)` or `scheduler.fail_job(job_id, err)` on error
5. Return `Ok(Some(job_id))`

### Resource Tracking (Single Authority)

```rust
pub fn allocate_tensor(&mut self, container_id: ContainerId, tensor_id: u64, size_bytes: usize) -> Result<()>;
pub fn free_tensor(&mut self, tensor_id: u64, size_bytes: usize);
```

- **allocate_tensor**: Atomic check-and-register. Checks per-container limits AND global limits, then updates both trackers and records ownership in `tensor_ids` and `tensor_owners`. If the limit check fails, nothing is modified (no partial state). Fails with `ContainerQuotaExceeded` or `ResourceLimitExceeded`.
- **free_tensor**: Looks up container from `tensor_owners`. If found, decrements both trackers (using `saturating_sub`), removes from `tensor_ids` and `tensor_owners`. If the tensor is not in `tensor_owners`, this is a no-op (defensive — handles tensors created before scheduler integration or double-free).

### Queries

```rust
pub fn container_usage(&self, id: ContainerId) -> Option<(usize, usize)>;
pub fn global_usage(&self) -> (usize, usize);
pub fn pending_job_count(&self, id: ContainerId) -> usize;
pub fn queue_depth(&self) -> usize;
pub fn container_count(&self) -> usize;
```

## Fairness Algorithm

### Deficit-Based Fair Queuing

Within each priority tier:
1. Group queued jobs by container
2. Select the container with the lowest `cumulative_exec_time_ns`
3. Dequeue that container's oldest job (FIFO within a container)
4. On completion, `cumulative_exec_time_ns += exec_time_ns`

This ensures containers that have used less GPU time get scheduled first, providing proportional fairness.

### Starvation Prevention

On each `next_job()` call, before selecting from the highest non-empty tier:
- Check all containers with queued jobs in lower tiers
- If any container's `last_scheduled_at` is `None` or older than a configurable threshold (default: 10 seconds), **permanently move** its oldest queued job from the current tier queue to the next higher tier queue
- `last_scheduled_at` is only reset when the job actually transitions to `Running` state (not on promotion), so a promoted-but-not-yet-running job does not suppress further starvation detection
- A job can only be promoted one tier per `next_job()` call (Low -> Normal or Normal -> High, not Low -> High in one step)
- If a container has multiple starved jobs, only the oldest is promoted per call

## Integration with LazyRuntime

### Modified LazyRuntime

```rust
pub struct LazyRuntime {
    tensors: HashMap<u64, Tensor>,
    graph: Graph,
    scheduler: Scheduler,
}
```

`limits` and `tracker` are removed from `LazyRuntime`. The `Scheduler` is the single authority for resource tracking.

### Modified insert_tensor

```rust
pub fn insert_tensor(&mut self, tensor: Tensor, container_id: ContainerId) -> Result<u64> {
    let size = tensor.meta.size_bytes();
    let id = tensor.meta.id;
    self.scheduler.allocate_tensor(container_id, id, size)?;
    self.tensors.insert(id, tensor);
    Ok(id)
}
```

For backward compatibility, a convenience method `insert_tensor_default` uses `ContainerId(0)`.

### Container Resolution During eval()

When `eval()` executes a graph for a target tensor, it needs to attribute intermediate tensor allocations to the correct container. Resolution strategy:

1. Look up the target tensor's `container_id` from `scheduler.tensor_owners` (if the target is already materialized)
2. Otherwise, look up the target's `OpNode.container_id` from the graph (if the target is pending)
3. All intermediate tensors created during this eval are attributed to the same container
4. This `container_id` is passed to `allocate_tensor` for each intermediate result

The `eval()` method signature becomes:
```rust
pub fn eval(&mut self, device: &Device, id: u64) -> Result<()>
```
It resolves the container internally — no caller change needed.

### eval_remote Migration

`eval_remote()` also allocates tensors when receiving results from the GPU service. The same container resolution applies: look up the container from the target tensor ID's owner, then attribute the result tensor to that container. The `eval_remote()` method resolves the container internally, same as `eval()`.

### Modified Graph: OpNode Container Tracking

```rust
pub struct OpNode {
    pub id: u64,
    pub op: OpKind,
    pub inputs: Vec<u64>,
    pub out_shape: Shape,
    pub out_dtype: DType,
    pub container_id: ContainerId,  // NEW — always set, defaults to ContainerId(0)
}
```

New method: `Graph::remove_nodes_for_container(container_id) -> Vec<u64>` removes all nodes belonging to a container and returns their IDs.

### Modified destroy

```rust
pub fn destroy(&mut self, id: u64) -> Result<()> {
    // existing dependency check...
    if let Some(tensor) = self.tensors.remove(&id) {
        let size = tensor.meta.size_bytes();
        self.scheduler.free_tensor(id, size);
    }
    Ok(())
}
```

## Error Handling

New `GpuError` variants:

```rust
ContainerNotFound(String),          // formatted ContainerId
ContainerPaused(String),            // formatted ContainerId
ContainerQuotaExceeded(String),     // includes ContainerId + detail
JobNotFound(String),                // formatted JobId
AdmissionRejected(String),          // max_pending_jobs exceeded
```

Using `String` rather than structured IDs to match the existing pattern (`InvalidTensor(String)`, `GraphError(String)`, etc.).

## Python API

### New Module Functions

```python
gpu.register_container(priority="normal", max_memory_mb=512, max_tensors=1000, max_pending=100) -> int
gpu.deregister_container(container_id: int) -> list[int]  # returns owned tensor IDs
gpu.pause_container(container_id: int)
gpu.resume_container(container_id: int)
gpu.submit_job(container_id: int, tensor) -> int  # returns job_id
gpu.run_next() -> Optional[int]  # returns job_id or None
gpu.job_status(job_id: int) -> str  # "queued", "running", "completed", "failed"
gpu.container_usage(container_id: int) -> tuple[int, int]  # (bytes, count)
gpu.global_usage() -> tuple[int, int]
gpu.queue_depth() -> int
```

### Backward Compatibility

All existing functions (`tensor()`, `eval()`, `to_list()`, `destroy()`, etc.) continue to work unchanged. They use the default container (`ContainerId(0)`) internally. The `set_limits()` function updates the default container's limits and the global limits.

## Pre-existing Fix

`MemoryTracker::track_free` in `limits.rs` uses `usize` subtraction which can panic on underflow. Change to `saturating_sub` for both `current_bytes` and `current_count`.

## Testing Strategy (TDD)

### Rust Unit Tests (in scheduler.rs, ~20 tests)

1. `test_new_creates_default_container` -- default container exists at ContainerId(0)
2. `test_register_container` -- returns unique IDs, stores config
3. `test_register_exceeding_global_capacity` -- fails when quota exceeds global limits
4. `test_deregister_container` -- removes container, returns owned tensors
5. `test_deregister_rejects_running_jobs` -- fails if job is Running
6. `test_deregister_drains_queued_jobs` -- queued jobs removed, queue_depth decreases
7. `test_cannot_deregister_default` -- ContainerId(0) cannot be removed
8. `test_pause_resume` -- paused container skipped by next_job, resume re-enables
9. `test_submit_rejects_paused` -- submit to paused container fails
10. `test_submit_admission_control` -- fails when pending_jobs >= max_pending_jobs
11. `test_priority_ordering` -- High jobs scheduled before Normal before Low
12. `test_fairness_within_tier` -- container with less GPU time scheduled first
13. `test_starvation_prevention` -- long-waiting low-priority container gets boosted
14. `test_track_alloc_per_container` -- enforces per-container limits
15. `test_track_alloc_global` -- enforces global limits across containers
16. `test_track_free_tensor` -- decrements correct container and global trackers
17. `test_register_tensor_ownership` -- tensor tracked in container and reverse map
18. `test_job_status_transitions` -- Queued -> Running -> Completed/Failed
19. `test_complete_job_not_running` -- returns JobNotFound
20. `test_container_isolation` -- container A at limit does not block container B
21. `test_free_unknown_tensor` -- free_tensor for unknown tensor_id is a no-op
22. `test_register_quota_exceeds_global` -- registering container whose quota exceeds global limit fails
23. `test_complete_already_completed` -- complete_job on non-Running job returns JobNotFound

### Rust Integration Tests (scheduler_integration.rs, ~5 tests)

1. `test_run_next_dequeue_eval_complete` -- full cycle with real tensors
2. `test_multi_container_round_robin` -- two containers get fair scheduling
3. `test_deregister_cleanup_with_graph` -- graph nodes removed on deregister
4. `test_default_container_backward_compat` -- insert_tensor_default works
5. `test_resource_tracking_through_lifecycle` -- alloc, eval, free, verify counts
6. `test_eval_attributes_intermediates_to_container` -- intermediate tensors from eval tracked under correct container

### Python Tests (~8 tests)

1. `test_register_deregister_container` -- lifecycle from Python
2. `test_submit_and_run_job` -- submit, run_next, check status
3. `test_container_usage_query` -- verify usage reporting
4. `test_queue_depth` -- verify queue depth reporting
5. `test_priority_scheduling` -- high-priority container scheduled first
6. `test_backward_compatibility` -- existing tensor/eval/to_list work unchanged
7. `test_admission_control` -- submit fails when queue full
8. `test_global_usage` -- verify global usage after multi-container alloc

## Extensibility Notes (Phase C Backlog)

The following are explicitly NOT in scope but the design accommodates them:

- **Per-container backend routing**: `ContainerConfig` can gain `backend: Option<Backend>` later. Requires changes to `backend.rs` singleton model.
- **Container location**: `ContainerState` can gain `location: ContainerLocation` (Local/Remote) for multi-node. IPC path already supports per-socket routing.
- **Work stealing**: `cumulative_exec_time_ns` and per-container queues enable load-aware stealing.
- **Multi-GPU dispatch**: `next_job()` would need a `device_id` parameter. Current single-consumer pattern documented as single-GPU assumption.
- **Dynamic container lifecycle**: pause/resume foundation exists. Auto-scaling would add create/destroy triggers based on queue pressure.
