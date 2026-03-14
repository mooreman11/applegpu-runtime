# Multi-Container Scheduler Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-container scheduler with priority-based fair queuing, per-container resource quotas, and backward-compatible default container to the applegpu_runtime GPU runtime.

**Architecture:** The scheduler embeds inside `LazyRuntime` (no second Mutex). A default `ContainerId(0)` container is auto-created with global limits, so all existing APIs work unchanged. Per-container sub-queues (3 tiers: High/Normal/Low) with deficit-based fair queuing drive scheduling. The scheduler is the single authority for resource tracking — `LazyRuntime`'s `limits`/`tracker` fields are removed.

**Tech Stack:** Rust (applegpu-core crate), PyO3 (Python bindings), pytest (Python tests)

**Spec:** `docs/superpowers/specs/2026-03-14-multi-container-scheduler-design.md`

---

## Chunk 1: Preparatory Changes

### Task 1: Fix MemoryTracker underflow bug

**Files:**
- Modify: `crates/core/src/limits.rs:111-113`

- [ ] **Step 1: Write failing test for underflow**

Add to `crates/core/src/limits.rs` in the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn track_free_underflow_does_not_panic() {
    let mut tracker = MemoryTracker::new();
    tracker.track_alloc(100);
    // Free more than allocated — should not panic
    tracker.track_free(200);
    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.tensor_count(), 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core track_free_underflow`
Expected: FAIL — panic on subtraction underflow in debug mode

- [ ] **Step 3: Fix track_free to use saturating_sub**

In `crates/core/src/limits.rs:111-113`, change:

```rust
pub fn track_free(&mut self, size_bytes: usize) {
    self.current_bytes = self.current_bytes.saturating_sub(size_bytes);
    self.current_count = self.current_count.saturating_sub(1);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p applegpu-core track_free_underflow`
Expected: PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `cargo test -p applegpu-core`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/limits.rs
git commit -m "fix: use saturating_sub in MemoryTracker::track_free to prevent underflow panic"
```

### Task 2: Add new GpuError variants

**Files:**
- Modify: `crates/core/src/error.rs:3-31`

- [ ] **Step 1: Write test for new error variants**

Add to `crates/core/src/error.rs` in the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn scheduler_error_display() {
    let e = GpuError::ContainerNotFound("container 5".to_string());
    assert!(e.to_string().contains("container 5"));

    let e = GpuError::ContainerPaused("container 3".to_string());
    assert!(e.to_string().contains("paused"));

    let e = GpuError::ContainerQuotaExceeded("container 1: memory".to_string());
    assert!(e.to_string().contains("quota"));

    let e = GpuError::JobNotFound("job 42".to_string());
    assert!(e.to_string().contains("42"));

    let e = GpuError::AdmissionRejected("container 2: queue full".to_string());
    assert!(e.to_string().contains("queue full"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core scheduler_error_display`
Expected: FAIL — variants do not exist

- [ ] **Step 3: Add the new variants to GpuError enum and Display impl**

In `crates/core/src/error.rs`, add to the `GpuError` enum after `ResourceLimitExceeded(String)`:

```rust
/// Container not found in scheduler
ContainerNotFound(String),
/// Container is paused
ContainerPaused(String),
/// Container resource quota exceeded
ContainerQuotaExceeded(String),
/// Job not found in scheduler
JobNotFound(String),
/// Job submission rejected (queue full)
AdmissionRejected(String),
```

Add to the `Display` match:

```rust
GpuError::ContainerNotFound(msg) => write!(f, "Container not found: {}", msg),
GpuError::ContainerPaused(msg) => write!(f, "Container paused: {}", msg),
GpuError::ContainerQuotaExceeded(msg) => write!(f, "Container quota exceeded: {}", msg),
GpuError::JobNotFound(msg) => write!(f, "Job not found: {}", msg),
GpuError::AdmissionRejected(msg) => write!(f, "Admission rejected: {}", msg),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p applegpu-core scheduler_error_display`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/error.rs
git commit -m "feat: add scheduler error variants to GpuError"
```

### Task 3: Add container_id field to OpNode

**Files:**
- Modify: `crates/core/src/graph.rs:87-98` (OpNode struct)
- Modify: `crates/core/src/graph.rs` (add `remove_nodes_for_container`)
- Modify: `crates/core/src/ops.rs:27-33,41-48` (OpNode construction in `lazy_binary_op` and `lazy_unary_op`)
- Modify: `crates/core/src/ops.rs:110-117,129-136,148-155,162-169` (matmul, softmax, transpose, scalar_mul OpNode construction)
- Modify: `crates/core/src/lazy.rs` tests (OpNode construction in tests)
- Modify: `crates/core/src/serial.rs` (serialize/deserialize OpNode — add container_id)
- Modify: `crates/core/src/fusion.rs` (OpNode construction in fusion pass)
- Modify: `crates/core/tests/` (integration test OpNode construction)

Note: `ContainerId` must be importable from `scheduler` module. We define it there first with minimal types.

**Important:** This step replaces the existing stub `Scheduler::new() -> Self` (no args). Nothing outside `scheduler.rs` depends on the stub, so this is safe. The full `Scheduler` struct with `new(global_limits)` will be implemented in Task 4.

- [ ] **Step 1: Define ContainerId and JobId in scheduler.rs**

Replace the entire contents of `crates/core/src/scheduler.rs` with:

```rust
use std::fmt;

/// Unique identifier for a container in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContainerId(pub u64);

impl ContainerId {
    /// The default container ID (used for backward compatibility).
    pub const DEFAULT: ContainerId = ContainerId(0);
}

impl fmt::Display for ContainerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "container-{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_id_display() {
        assert_eq!(ContainerId(5).to_string(), "container-5");
        assert_eq!(ContainerId::DEFAULT.to_string(), "container-0");
    }

    #[test]
    fn container_id_is_copy_hash() {
        let id = ContainerId(1);
        let id2 = id; // Copy
        assert_eq!(id, id2);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(id);
        assert!(set.contains(&id));
    }
}
```

- [ ] **Step 2: Run scheduler tests**

Run: `cargo test -p applegpu-core scheduler`
Expected: PASS

- [ ] **Step 3: Add container_id to OpNode and add remove_nodes_for_container**

In `crates/core/src/graph.rs`, add import at top:

```rust
use crate::scheduler::ContainerId;
```

Add `container_id` field to `OpNode` struct (after `out_dtype`):

```rust
pub struct OpNode {
    pub id: u64,
    pub op: OpKind,
    pub inputs: Vec<u64>,
    pub out_shape: Shape,
    pub out_dtype: DType,
    /// Container that owns this operation. Defaults to ContainerId::DEFAULT.
    pub container_id: ContainerId,
}
```

Add method to `Graph` impl (after `len()`):

```rust
/// Remove all nodes belonging to a container. Returns their IDs.
pub fn remove_nodes_for_container(&mut self, container_id: ContainerId) -> Vec<u64> {
    let ids: Vec<u64> = self.nodes.iter()
        .filter(|(_, node)| node.container_id == container_id)
        .map(|(&id, _)| id)
        .collect();
    for &id in &ids {
        self.nodes.remove(&id);
    }
    ids
}
```

- [ ] **Step 4: Fix all OpNode construction sites to include container_id: ContainerId::DEFAULT**

Every place that constructs an `OpNode` must add `container_id: ContainerId::DEFAULT`. These locations are:

**`crates/core/src/ops.rs`** — `lazy_binary_op` (line 27-34), `lazy_unary_op` (line 41-48), `matmul` (line 110-117), `softmax` (line 129-136), `transpose` (line 148-155), `scalar_mul` (line 162-169):

Add `use crate::scheduler::ContainerId;` at the top of ops.rs. Then in each `OpNode { ... }` add `container_id: ContainerId::DEFAULT,` after `out_dtype`.

**Design note:** All ops default to `ContainerId::DEFAULT`. Container attribution happens at eval-time via `resolve_container()` in `LazyRuntime::eval()`, which looks up the target tensor's owner. This means `OpNode.container_id` serves primarily for `Graph::remove_nodes_for_container()` cleanup. Future work (Phase C) could add a `record_op_for(node, container_id)` method to set container affinity at record time.

**`crates/core/src/fusion.rs`** — the fused OpNode construction (search for `OpNode {` in fusion.rs):

Add `use crate::scheduler::ContainerId;` and add `container_id: ContainerId::DEFAULT,` to each OpNode construction.

**`crates/core/src/serial.rs`** — the OpNode deserialization (search for `OpNode {` in serial.rs):

Add `use crate::scheduler::ContainerId;` and add `container_id: ContainerId::DEFAULT,` to the deserialized OpNode. Note: `container_id` is NOT serialized over IPC — it's a local-only field. Remote nodes always get `DEFAULT`.

**`crates/core/src/lazy.rs`** tests — OpNode constructions in test functions:

Add `use crate::scheduler::ContainerId;` and add `container_id: ContainerId::DEFAULT,` to each test's OpNode.

**`crates/core/tests/`** — integration test files that construct OpNode:

Add `use applegpu_core::scheduler::ContainerId;` and add `container_id: ContainerId::DEFAULT,` to each OpNode.

**`crates/core/src/graph.rs`** tests — OpNode constructions in graph tests:

Add `container_id: ContainerId::DEFAULT,` to each OpNode in the test module.

- [ ] **Step 5: Write test for remove_nodes_for_container**

Add to `crates/core/src/graph.rs` tests:

```rust
#[test]
fn remove_nodes_for_container() {
    let c1 = ContainerId(1);
    let c2 = ContainerId(2);
    let mut g = Graph::new();
    g.add_node(OpNode {
        id: 10, op: OpKind::Add, inputs: vec![1, 2],
        out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        container_id: c1,
    });
    g.add_node(OpNode {
        id: 11, op: OpKind::Neg, inputs: vec![3],
        out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        container_id: c2,
    });
    g.add_node(OpNode {
        id: 12, op: OpKind::Relu, inputs: vec![10],
        out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        container_id: c1,
    });

    let removed = g.remove_nodes_for_container(c1);
    assert_eq!(removed.len(), 2);
    assert!(removed.contains(&10));
    assert!(removed.contains(&12));
    assert!(g.has_node(11));
    assert!(!g.has_node(10));
    assert!(!g.has_node(12));
}
```

- [ ] **Step 6: Run all tests to verify everything compiles and passes**

Run: `cargo test -p applegpu-core`
Expected: All tests pass (including existing graph, ops, lazy, fusion tests)

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/scheduler.rs crates/core/src/graph.rs crates/core/src/ops.rs crates/core/src/fusion.rs crates/core/src/serial.rs crates/core/src/lazy.rs crates/core/tests/
git commit -m "feat: add ContainerId type and container_id field to OpNode"
```

---

## Chunk 2: Scheduler Core (Types, Construction, Container Lifecycle)

### Task 4: Scheduler struct with construction and default container

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: Write tests for Scheduler::new and default container**

Add to `crates/core/src/scheduler.rs` tests:

```rust
use crate::limits::{MemoryTracker, ResourceLimits};

#[test]
fn test_new_creates_default_container() {
    let limits = ResourceLimits::default_limits();
    let sched = Scheduler::new(limits);
    assert_eq!(sched.container_count(), 1);
    assert!(sched.container_usage(ContainerId::DEFAULT).is_some());
}

#[test]
fn test_global_usage_starts_at_zero() {
    let limits = ResourceLimits::default_limits();
    let sched = Scheduler::new(limits);
    assert_eq!(sched.global_usage(), (0, 0));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p applegpu-core test_new_creates_default`
Expected: FAIL — Scheduler struct has wrong signature

- [ ] **Step 3: Implement Scheduler struct, Priority, ContainerConfig, ContainerState, JobId, Job, JobStatus**

In `crates/core/src/scheduler.rs`, replace the entire file with the full implementation:

```rust
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::Instant;

use crate::error::{GpuError, Result};
use crate::limits::{MemoryTracker, ResourceLimits};

/// Unique identifier for a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContainerId(pub u64);

impl ContainerId {
    pub const DEFAULT: ContainerId = ContainerId(0);
}

impl fmt::Display for ContainerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "container-{}", self.0)
    }
}

/// Unique identifier for a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub u64);

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "job-{}", self.0)
    }
}

/// Scheduling priority tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    High = 0,
    Normal = 1,
    Low = 2,
}

/// Configuration for registering a container.
pub struct ContainerConfig {
    pub priority: Priority,
    pub max_memory_bytes: usize,
    pub max_tensor_count: usize,
    pub max_tensor_size_bytes: usize, // 0 = inherit from global
    pub max_pending_jobs: usize,
}

/// Internal state of a registered container.
pub struct ContainerState {
    pub id: ContainerId,
    pub config: ContainerConfig,
    pub limits: ResourceLimits,
    pub tracker: MemoryTracker,
    pub tensor_ids: HashSet<u64>,
    pub pending_jobs: usize,
    pub total_jobs_completed: u64,
    pub cumulative_exec_time_ns: u64,
    pub last_scheduled_at: Option<Instant>,
    pub created_at: Instant,
    pub paused: bool,
}

/// A scheduled evaluation job.
#[derive(Debug)]
pub struct Job {
    pub id: JobId,
    pub container_id: ContainerId,
    pub target_tensor_id: u64,
    pub submitted_at: Instant,
    pub priority: Priority,
}

/// Status of a job in the scheduler.
pub enum JobStatus {
    Queued,
    Running { started_at: Instant },
    Completed { tensor_id: u64, exec_time_ns: u64 },
    Failed { error: String },
}

/// Multi-container GPU scheduler.
pub struct Scheduler {
    global_limits: ResourceLimits,
    global_tracker: MemoryTracker,
    containers: HashMap<ContainerId, ContainerState>,
    queues: [HashMap<ContainerId, VecDeque<Job>>; 3],
    jobs: HashMap<JobId, (ContainerId, JobStatus)>,
    tensor_owners: HashMap<u64, ContainerId>,
    next_container_id: u64,
    next_job_id: u64,
    starvation_threshold_ns: u64,
}

impl Scheduler {
    /// Create a new scheduler with the given global limits.
    /// Auto-registers the default container (ContainerId(0)) with those limits.
    pub fn new(global_limits: ResourceLimits) -> Self {
        let default_state = ContainerState {
            id: ContainerId::DEFAULT,
            config: ContainerConfig {
                priority: Priority::Normal,
                max_memory_bytes: global_limits.max_total_memory_bytes,
                max_tensor_count: global_limits.max_tensor_count,
                max_tensor_size_bytes: global_limits.max_tensor_size_bytes,
                max_pending_jobs: usize::MAX,
            },
            limits: global_limits.clone(),
            tracker: MemoryTracker::new(),
            tensor_ids: HashSet::new(),
            pending_jobs: 0,
            total_jobs_completed: 0,
            cumulative_exec_time_ns: 0,
            last_scheduled_at: None,
            created_at: Instant::now(),
            paused: false,
        };

        let mut containers = HashMap::new();
        containers.insert(ContainerId::DEFAULT, default_state);

        Scheduler {
            global_limits,
            global_tracker: MemoryTracker::new(),
            containers,
            queues: [HashMap::new(), HashMap::new(), HashMap::new()],
            jobs: HashMap::new(),
            tensor_owners: HashMap::new(),
            next_container_id: 1,
            next_job_id: 0,
            starvation_threshold_ns: 10_000_000_000, // 10 seconds
        }
    }

    /// Create a scheduler with a custom starvation threshold (for testing).
    pub fn with_starvation_threshold(global_limits: ResourceLimits, threshold_ns: u64) -> Self {
        let mut sched = Self::new(global_limits);
        sched.starvation_threshold_ns = threshold_ns;
        sched
    }

    pub fn container_count(&self) -> usize {
        self.containers.len()
    }

    pub fn container_usage(&self, id: ContainerId) -> Option<(usize, usize)> {
        self.containers.get(&id).map(|c| {
            (c.tracker.memory_usage(), c.tracker.tensor_count())
        })
    }

    pub fn global_usage(&self) -> (usize, usize) {
        (self.global_tracker.memory_usage(), self.global_tracker.tensor_count())
    }

    pub fn queue_depth(&self) -> usize {
        self.queues.iter()
            .flat_map(|tier| tier.values())
            .map(|q| q.len())
            .sum()
    }

    pub fn pending_job_count(&self, id: ContainerId) -> usize {
        self.containers.get(&id).map(|c| c.pending_jobs).unwrap_or(0)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p applegpu-core test_new_creates_default test_global_usage_starts_at_zero`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat: implement Scheduler struct with default container and query methods"
```

### Task 5: Container registration and deregistration

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: Write tests for register/deregister**

Add to scheduler tests:

```rust
fn test_limits() -> ResourceLimits {
    ResourceLimits {
        max_tensor_size_bytes: 1024 * 1024,
        max_total_memory_bytes: 10 * 1024 * 1024,
        max_tensor_count: 100,
    }
}

fn test_config() -> ContainerConfig {
    ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 2 * 1024 * 1024,
        max_tensor_count: 20,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    }
}

#[test]
fn test_register_container() {
    let mut sched = Scheduler::new(test_limits());
    let id = sched.register_container(test_config()).unwrap();
    assert_eq!(sched.container_count(), 2); // default + new
    assert!(sched.container_usage(id).is_some());
}

#[test]
fn test_register_exceeding_global_capacity() {
    let limits = ResourceLimits {
        max_tensor_size_bytes: 0,
        max_total_memory_bytes: 1024,
        max_tensor_count: 10,
    };
    let mut sched = Scheduler::new(limits);
    // Default container already claims full 1024 bytes
    let config = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 1,
        max_tensor_count: 1,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let result = sched.register_container(config);
    assert!(result.is_err());
}

#[test]
fn test_deregister_container() {
    let mut sched = Scheduler::new(test_limits());
    let id = sched.register_container(test_config()).unwrap();
    let tensors = sched.deregister_container(id).unwrap();
    assert!(tensors.is_empty());
    assert_eq!(sched.container_count(), 1); // only default remains
}

#[test]
fn test_cannot_deregister_default() {
    let mut sched = Scheduler::new(test_limits());
    let result = sched.deregister_container(ContainerId::DEFAULT);
    assert!(result.is_err());
}

#[test]
fn test_deregister_returns_owned_tensors() {
    let mut sched = Scheduler::new(test_limits());
    let id = sched.register_container(test_config()).unwrap();
    sched.allocate_tensor(id, 100, 256).unwrap();
    sched.allocate_tensor(id, 101, 256).unwrap();
    let tensors = sched.deregister_container(id).unwrap();
    assert_eq!(tensors.len(), 2);
    assert!(tensors.contains(&100));
    assert!(tensors.contains(&101));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p applegpu-core test_register_container test_deregister`
Expected: FAIL — methods do not exist

- [ ] **Step 3: Implement register_container and deregister_container**

**Important:** The `jobs` field in `Scheduler` must be `HashMap<JobId, (ContainerId, JobStatus)>` (not `HashMap<JobId, JobStatus>`). This was defined in Task 4 Step 3. The tuple tracks which container owns each job, needed for running-job detection during deregister.

Add to `Scheduler` impl:

```rust
pub fn register_container(&mut self, config: ContainerConfig) -> Result<ContainerId> {
    let committed_memory: usize = self.containers.values()
        .map(|c| c.config.max_memory_bytes)
        .sum();
    let committed_tensors: usize = self.containers.values()
        .map(|c| c.config.max_tensor_count)
        .sum();

    if self.global_limits.max_total_memory_bytes > 0
        && committed_memory + config.max_memory_bytes > self.global_limits.max_total_memory_bytes
    {
        return Err(GpuError::ContainerQuotaExceeded(format!(
            "Requested {} bytes but only {} remaining of {} total",
            config.max_memory_bytes,
            self.global_limits.max_total_memory_bytes.saturating_sub(committed_memory),
            self.global_limits.max_total_memory_bytes,
        )));
    }

    if self.global_limits.max_tensor_count > 0
        && committed_tensors + config.max_tensor_count > self.global_limits.max_tensor_count
    {
        return Err(GpuError::ContainerQuotaExceeded(format!(
            "Requested {} tensors but only {} remaining of {} total",
            config.max_tensor_count,
            self.global_limits.max_tensor_count.saturating_sub(committed_tensors),
            self.global_limits.max_tensor_count,
        )));
    }

    let id = ContainerId(self.next_container_id);
    self.next_container_id += 1;

    let limits = ResourceLimits {
        max_total_memory_bytes: config.max_memory_bytes,
        max_tensor_count: config.max_tensor_count,
        max_tensor_size_bytes: if config.max_tensor_size_bytes == 0 {
            self.global_limits.max_tensor_size_bytes
        } else {
            config.max_tensor_size_bytes
        },
    };

    let state = ContainerState {
        id,
        config,
        limits,
        tracker: MemoryTracker::new(),
        tensor_ids: HashSet::new(),
        pending_jobs: 0,
        total_jobs_completed: 0,
        cumulative_exec_time_ns: 0,
        last_scheduled_at: None,
        created_at: Instant::now(),
        paused: false,
    };

    self.containers.insert(id, state);
    Ok(id)
}

pub fn deregister_container(&mut self, id: ContainerId) -> Result<Vec<u64>> {
    if id == ContainerId::DEFAULT {
        return Err(GpuError::ContainerNotFound(
            "Cannot deregister the default container".to_string()
        ));
    }

    self.containers.get(&id)
        .ok_or_else(|| GpuError::ContainerNotFound(id.to_string()))?;

    // Check for running jobs belonging to this container
    let has_running = self.jobs.iter().any(|(_, (cid, status))| {
        *cid == id && matches!(status, JobStatus::Running { .. })
    });
    if has_running {
        return Err(GpuError::AdmissionRejected(format!(
            "Cannot deregister {} while it has running jobs", id
        )));
    }

    // Drain queued jobs for this container from all tiers
    for tier in &mut self.queues {
        if let Some(container_queue) = tier.remove(&id) {
            for job in container_queue {
                self.jobs.remove(&job.id);
            }
        }
    }

    // Collect owned tensors before removing container
    let owned_tensors: Vec<u64> = self.containers.get(&id).unwrap()
        .tensor_ids.iter().copied().collect();

    // Decrement global tracker for each owned tensor using free_tensor
    // (free_tensor handles global_tracker + tensor_owners cleanup)
    // But free_tensor looks up the container, which we haven't removed yet — perfect.
    // We need tensor sizes though. Store tensor sizes in a separate map, or
    // just decrement the global tracker by the container's aggregate.
    let container = self.containers.remove(&id).unwrap();
    let usage_bytes = container.tracker.memory_usage();
    let usage_count = container.tracker.tensor_count();

    // Clean up tensor_owners
    for &tid in &owned_tensors {
        self.tensor_owners.remove(&tid);
    }

    // Decrement global tracker by aggregate container usage
    // We use a loop to match the count decrements exactly
    if usage_count > 0 && usage_bytes > 0 {
        let per_tensor_avg = usage_bytes / usage_count;
        let remainder = usage_bytes % usage_count;
        for i in 0..usage_count {
            let size = per_tensor_avg + if i == 0 { remainder } else { 0 };
            self.global_tracker.track_free(size);
        }
    }

    Ok(owned_tensors)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p applegpu-core test_register test_deregister test_cannot_deregister`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat: add container registration and deregistration to scheduler"
```

### Task 6: Pause and resume containers

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn test_pause_resume() {
    let mut sched = Scheduler::new(test_limits());
    let id = sched.register_container(test_config()).unwrap();
    sched.pause_container(id).unwrap();
    // Container is paused
    assert!(sched.containers.get(&id).unwrap().paused);
    sched.resume_container(id).unwrap();
    assert!(!sched.containers.get(&id).unwrap().paused);
}

#[test]
fn test_pause_unknown_container() {
    let mut sched = Scheduler::new(test_limits());
    let result = sched.pause_container(ContainerId(99));
    assert!(result.is_err());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p applegpu-core test_pause`
Expected: FAIL

- [ ] **Step 3: Implement pause_container and resume_container**

```rust
pub fn pause_container(&mut self, id: ContainerId) -> Result<()> {
    let container = self.containers.get_mut(&id)
        .ok_or_else(|| GpuError::ContainerNotFound(id.to_string()))?;
    container.paused = true;
    Ok(())
}

pub fn resume_container(&mut self, id: ContainerId) -> Result<()> {
    let container = self.containers.get_mut(&id)
        .ok_or_else(|| GpuError::ContainerNotFound(id.to_string()))?;
    container.paused = false;
    Ok(())
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p applegpu-core test_pause`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat: add pause/resume container support to scheduler"
```

---

## Chunk 3: Resource Tracking and Job Scheduling

### Task 7: Resource tracking (allocate_tensor, free_tensor)

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn test_allocate_tensor_per_container() {
    let mut sched = Scheduler::new(test_limits());
    let id = sched.register_container(test_config()).unwrap();
    sched.allocate_tensor(id, 1, 1024).unwrap();
    assert_eq!(sched.container_usage(id), Some((1024, 1)));
    assert_eq!(sched.global_usage(), (1024, 1));
}

#[test]
fn test_allocate_tensor_exceeds_container_quota() {
    let limits = ResourceLimits {
        max_tensor_size_bytes: 0,
        max_total_memory_bytes: 10_000,
        max_tensor_count: 100,
    };
    let mut sched = Scheduler::new(limits);
    let config = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 500,
        max_tensor_count: 10,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let id = sched.register_container(config).unwrap();
    sched.allocate_tensor(id, 1, 400).unwrap();
    let result = sched.allocate_tensor(id, 2, 200);
    assert!(result.is_err());
}

#[test]
fn test_allocate_tensor_global_limit() {
    let limits = ResourceLimits {
        max_tensor_size_bytes: 0,
        max_total_memory_bytes: 1000,
        max_tensor_count: 100,
    };
    let mut sched = Scheduler::new(limits);
    // Default container claims 1000 quota. Allocate near limit.
    sched.allocate_tensor(ContainerId::DEFAULT, 1, 800).unwrap();
    let result = sched.allocate_tensor(ContainerId::DEFAULT, 2, 300);
    assert!(result.is_err());
}

#[test]
fn test_free_tensor() {
    let mut sched = Scheduler::new(test_limits());
    sched.allocate_tensor(ContainerId::DEFAULT, 1, 1024).unwrap();
    assert_eq!(sched.global_usage(), (1024, 1));
    sched.free_tensor(1, 1024);
    assert_eq!(sched.global_usage(), (0, 0));
    assert_eq!(sched.container_usage(ContainerId::DEFAULT), Some((0, 0)));
}

#[test]
fn test_free_unknown_tensor() {
    let mut sched = Scheduler::new(test_limits());
    // Should be a no-op, not panic
    sched.free_tensor(999, 1024);
    assert_eq!(sched.global_usage(), (0, 0));
}

#[test]
fn test_container_isolation() {
    let limits = ResourceLimits {
        max_tensor_size_bytes: 0,
        max_total_memory_bytes: 10_000,
        max_tensor_count: 100,
    };
    let mut sched = Scheduler::new(limits.clone());
    // Reduce default container quota so there's room for others
    // (default gets full global limits, so we need to work within that)
    let config_a = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 500,
        max_tensor_count: 10,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let config_b = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 500,
        max_tensor_count: 10,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let a = sched.register_container(config_a).unwrap();
    let b = sched.register_container(config_b).unwrap();

    // Fill container A to its limit
    sched.allocate_tensor(a, 1, 500).unwrap();
    // Container A is full
    assert!(sched.allocate_tensor(a, 2, 100).is_err());
    // Container B still has headroom
    assert!(sched.allocate_tensor(b, 3, 400).is_ok());
}

#[test]
fn test_register_tensor_ownership() {
    let mut sched = Scheduler::new(test_limits());
    let id = sched.register_container(test_config()).unwrap();
    sched.allocate_tensor(id, 42, 256).unwrap();
    assert!(sched.containers.get(&id).unwrap().tensor_ids.contains(&42));
    assert_eq!(sched.tensor_owners.get(&42), Some(&id));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p applegpu-core test_allocate_tensor test_free_tensor test_container_isolation test_register_tensor`
Expected: FAIL

- [ ] **Step 3: Implement allocate_tensor and free_tensor**

```rust
/// Atomically check limits and register a tensor for a container.
pub fn allocate_tensor(&mut self, container_id: ContainerId, tensor_id: u64, size_bytes: usize) -> Result<()> {
    let container = self.containers.get(&container_id)
        .ok_or_else(|| GpuError::ContainerNotFound(container_id.to_string()))?;

    // Check per-container limits
    container.tracker.check_allocation(size_bytes, &container.limits)
        .map_err(|_| GpuError::ContainerQuotaExceeded(format!(
            "{}: would exceed container memory/tensor limits", container_id
        )))?;

    // Check global limits
    self.global_tracker.check_allocation(size_bytes, &self.global_limits)?;

    // Both checks passed — update both trackers
    let container = self.containers.get_mut(&container_id).unwrap();
    container.tracker.track_alloc(size_bytes);
    container.tensor_ids.insert(tensor_id);
    self.global_tracker.track_alloc(size_bytes);
    self.tensor_owners.insert(tensor_id, container_id);

    Ok(())
}

/// Free a tensor. Looks up the owning container. No-op if tensor is unknown.
pub fn free_tensor(&mut self, tensor_id: u64, size_bytes: usize) {
    if let Some(container_id) = self.tensor_owners.remove(&tensor_id) {
        if let Some(container) = self.containers.get_mut(&container_id) {
            container.tracker.track_free(size_bytes);
            container.tensor_ids.remove(&tensor_id);
        }
        self.global_tracker.track_free(size_bytes);
    }
}

/// Look up which container owns a tensor.
pub fn tensor_owner(&self, tensor_id: u64) -> Option<ContainerId> {
    self.tensor_owners.get(&tensor_id).copied()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p applegpu-core test_allocate test_free test_container_isolation test_register_tensor`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat: add resource tracking (allocate_tensor, free_tensor) to scheduler"
```

### Task 8: Job submission

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn test_submit_job() {
    let mut sched = Scheduler::new(test_limits());
    let job_id = sched.submit(ContainerId::DEFAULT, 42).unwrap();
    assert_eq!(sched.queue_depth(), 1);
    assert!(matches!(sched.job_status(job_id), Some(&(_, JobStatus::Queued))));
}

#[test]
fn test_submit_rejects_paused() {
    let mut sched = Scheduler::new(test_limits());
    let id = sched.register_container(test_config()).unwrap();
    sched.pause_container(id).unwrap();
    let result = sched.submit(id, 42);
    assert!(result.is_err());
}

#[test]
fn test_submit_admission_control() {
    let limits = test_limits();
    let mut sched = Scheduler::new(limits);
    let config = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 2 * 1024 * 1024,
        max_tensor_count: 20,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 2, // only allow 2 pending
    };
    let id = sched.register_container(config).unwrap();
    sched.submit(id, 1).unwrap();
    sched.submit(id, 2).unwrap();
    let result = sched.submit(id, 3);
    assert!(result.is_err());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p applegpu-core test_submit`
Expected: FAIL

- [ ] **Step 3: Implement submit**

```rust
pub fn submit(&mut self, container_id: ContainerId, target_tensor_id: u64) -> Result<JobId> {
    let container = self.containers.get(&container_id)
        .ok_or_else(|| GpuError::ContainerNotFound(container_id.to_string()))?;

    if container.paused {
        return Err(GpuError::ContainerPaused(container_id.to_string()));
    }

    if container.pending_jobs >= container.config.max_pending_jobs {
        return Err(GpuError::AdmissionRejected(format!(
            "{}: pending jobs {} >= max {}",
            container_id, container.pending_jobs, container.config.max_pending_jobs
        )));
    }

    let job_id = JobId(self.next_job_id);
    self.next_job_id += 1;

    let priority = container.config.priority;
    let job = Job {
        id: job_id,
        container_id,
        target_tensor_id,
        submitted_at: Instant::now(),
        priority,
    };

    let tier = priority as usize;
    self.queues[tier]
        .entry(container_id)
        .or_insert_with(VecDeque::new)
        .push_back(job);

    self.jobs.insert(job_id, (container_id, JobStatus::Queued));
    self.containers.get_mut(&container_id).unwrap().pending_jobs += 1;

    Ok(job_id)
}

pub fn job_status(&self, job_id: JobId) -> Option<&(ContainerId, JobStatus)> {
    self.jobs.get(&job_id)
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core test_submit`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat: add job submission with admission control to scheduler"
```

### Task 9: Job scheduling (next_job) with fairness and starvation prevention

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn test_priority_ordering() {
    let limits = test_limits();
    let mut sched = Scheduler::new(limits);

    let high_config = ContainerConfig {
        priority: Priority::High,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 10,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let low_config = ContainerConfig {
        priority: Priority::Low,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 10,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };

    let low_id = sched.register_container(low_config).unwrap();
    let high_id = sched.register_container(high_config).unwrap();

    sched.submit(low_id, 1).unwrap();
    sched.submit(high_id, 2).unwrap();

    let job = sched.next_job().unwrap();
    assert_eq!(job.container_id, high_id);
}

#[test]
fn test_fairness_within_tier() {
    let limits = test_limits();
    let mut sched = Scheduler::new(limits);

    let config_a = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 10,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let config_b = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 10,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let a = sched.register_container(config_a).unwrap();
    let b = sched.register_container(config_b).unwrap();

    // Give A more exec time so B should be scheduled first
    sched.containers.get_mut(&a).unwrap().cumulative_exec_time_ns = 1000;
    sched.containers.get_mut(&b).unwrap().cumulative_exec_time_ns = 100;

    sched.submit(a, 1).unwrap();
    sched.submit(b, 2).unwrap();

    let job = sched.next_job().unwrap();
    assert_eq!(job.container_id, b); // B has less exec time
}

#[test]
fn test_pause_skips_container() {
    let mut sched = Scheduler::new(test_limits());
    let id = sched.register_container(test_config()).unwrap();
    sched.submit(id, 1).unwrap();
    sched.pause_container(id).unwrap();
    let job = sched.next_job();
    assert!(job.is_none()); // paused container's jobs are skipped
}

#[test]
fn test_starvation_prevention() {
    let limits = test_limits();
    // Use 0ns threshold so starvation triggers immediately
    let mut sched = Scheduler::with_starvation_threshold(limits, 0);

    let high_config = ContainerConfig {
        priority: Priority::High,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 10,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let low_config = ContainerConfig {
        priority: Priority::Low,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 10,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };

    let high_id = sched.register_container(high_config).unwrap();
    let low_id = sched.register_container(low_config).unwrap();

    // Submit low-priority job first
    sched.submit(low_id, 1).unwrap();
    // Submit high-priority job
    sched.submit(high_id, 2).unwrap();

    // High should go first
    let job1 = sched.next_job().unwrap();
    assert_eq!(job1.container_id, high_id);
    sched.complete_job(job1.id, 1000).unwrap();

    // Submit another high-priority job
    sched.submit(high_id, 3).unwrap();

    // Low-priority job should get promoted due to starvation (threshold=0ns)
    let job2 = sched.next_job().unwrap();
    assert_eq!(job2.container_id, low_id); // promoted!
}

#[test]
fn test_next_job_empty_queue() {
    let mut sched = Scheduler::new(test_limits());
    assert!(sched.next_job().is_none());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p applegpu-core test_priority_ordering test_fairness test_pause_skips test_next_job_empty`
Expected: FAIL

- [ ] **Step 3: Implement next_job with fairness and starvation prevention**

Change `next_job` to take `&mut self`:

```rust
pub fn next_job(&mut self) -> Option<Job> {
    // Starvation prevention: promote starved lower-tier jobs
    self.promote_starved_jobs();

    // Scan tiers High -> Normal -> Low
    for tier_idx in 0..3 {
        let tier = &self.queues[tier_idx];
        if tier.is_empty() {
            continue;
        }

        // Find the non-paused container with the lowest cumulative_exec_time_ns
        let mut best_container: Option<ContainerId> = None;
        let mut best_time = u64::MAX;

        for (&cid, queue) in tier.iter() {
            if queue.is_empty() {
                continue;
            }
            if let Some(container) = self.containers.get(&cid) {
                if container.paused {
                    continue;
                }
                if container.cumulative_exec_time_ns < best_time {
                    best_time = container.cumulative_exec_time_ns;
                    best_container = Some(cid);
                }
            }
        }

        if let Some(cid) = best_container {
            let queue = self.queues[tier_idx].get_mut(&cid).unwrap();
            if let Some(job) = queue.pop_front() {
                // Clean up empty queues
                if queue.is_empty() {
                    self.queues[tier_idx].remove(&cid);
                }

                // Transition to Running
                let job_id = job.id;
                self.jobs.insert(job_id, (cid, JobStatus::Running {
                    started_at: Instant::now(),
                }));

                // Update last_scheduled_at
                if let Some(container) = self.containers.get_mut(&cid) {
                    container.last_scheduled_at = Some(Instant::now());
                }

                return Some(job);
            }
        }
    }

    None
}

fn promote_starved_jobs(&mut self) {
    let now = Instant::now();
    let threshold = self.starvation_threshold_ns;

    // Check tiers Low (2) and Normal (1) for starvation
    for tier_idx in (1..=2).rev() {
        let mut promotions: Vec<(ContainerId, Job)> = Vec::new();

        for (&cid, queue) in &self.queues[tier_idx] {
            if queue.is_empty() {
                continue;
            }
            if let Some(container) = self.containers.get(&cid) {
                if container.paused {
                    continue;
                }
                let starved = match container.last_scheduled_at {
                    None => true,
                    Some(last) => now.duration_since(last).as_nanos() as u64 > threshold,
                };
                if starved {
                    // Will promote the oldest job
                    promotions.push((cid, Job {
                        id: JobId(0), // placeholder — we'll pop from queue
                        container_id: cid,
                        target_tensor_id: 0,
                        submitted_at: Instant::now(),
                        priority: Priority::High, // placeholder
                    }));
                }
            }
        }

        // Actually do the promotions (pop from source, push to target)
        for (cid, _) in promotions {
            let source = self.queues[tier_idx].get_mut(&cid);
            if let Some(queue) = source {
                if let Some(mut job) = queue.pop_front() {
                    if queue.is_empty() {
                        self.queues[tier_idx].remove(&cid);
                    }
                    // Promote one tier up
                    let target_tier = tier_idx - 1;
                    job.priority = match target_tier {
                        0 => Priority::High,
                        1 => Priority::Normal,
                        _ => Priority::Low,
                    };
                    self.queues[target_tier]
                        .entry(cid)
                        .or_insert_with(VecDeque::new)
                        .push_back(job);
                }
            }
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core test_priority_ordering test_fairness test_pause_skips test_next_job_empty`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat: add job scheduling with deficit-based fairness and starvation prevention"
```

### Task 10: Job completion and failure

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn test_job_status_transitions() {
    let mut sched = Scheduler::new(test_limits());
    let job_id = sched.submit(ContainerId::DEFAULT, 42).unwrap();
    assert!(matches!(sched.job_status(job_id), Some(&(_, JobStatus::Queued))));

    let _job = sched.next_job().unwrap();
    assert!(matches!(sched.job_status(job_id), Some(&(_, JobStatus::Running { .. }))));

    sched.complete_job(job_id, 5000).unwrap();
    assert!(matches!(sched.job_status(job_id), Some(&(_, JobStatus::Completed { .. }))));
}

#[test]
fn test_complete_job_not_running() {
    let mut sched = Scheduler::new(test_limits());
    let job_id = sched.submit(ContainerId::DEFAULT, 42).unwrap();
    // Job is Queued, not Running
    let result = sched.complete_job(job_id, 5000);
    assert!(result.is_err());
}

#[test]
fn test_complete_already_completed() {
    let mut sched = Scheduler::new(test_limits());
    let job_id = sched.submit(ContainerId::DEFAULT, 42).unwrap();
    let _job = sched.next_job().unwrap();
    sched.complete_job(job_id, 5000).unwrap();
    let result = sched.complete_job(job_id, 5000);
    assert!(result.is_err());
}

#[test]
fn test_fail_job() {
    let mut sched = Scheduler::new(test_limits());
    let job_id = sched.submit(ContainerId::DEFAULT, 42).unwrap();
    let _job = sched.next_job().unwrap();
    sched.fail_job(job_id, "eval error".to_string()).unwrap();
    assert!(matches!(sched.job_status(job_id), Some(&(_, JobStatus::Failed { .. }))));
}

#[test]
fn test_deregister_rejects_running_jobs() {
    let mut sched = Scheduler::new(test_limits());
    let id = sched.register_container(test_config()).unwrap();
    sched.submit(id, 42).unwrap();
    let _job = sched.next_job().unwrap();
    let result = sched.deregister_container(id);
    assert!(result.is_err());
}

#[test]
fn test_deregister_drains_queued_jobs() {
    let mut sched = Scheduler::new(test_limits());
    let id = sched.register_container(test_config()).unwrap();
    sched.submit(id, 1).unwrap();
    sched.submit(id, 2).unwrap();
    assert_eq!(sched.queue_depth(), 2);
    let tensors = sched.deregister_container(id).unwrap();
    assert_eq!(sched.queue_depth(), 0);
    assert!(tensors.is_empty()); // no tensors allocated
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p applegpu-core test_job_status test_complete test_fail_job test_deregister_rejects test_deregister_drains`
Expected: FAIL

- [ ] **Step 3: Implement complete_job and fail_job**

```rust
pub fn complete_job(&mut self, job_id: JobId, exec_time_ns: u64) -> Result<()> {
    let (container_id, status) = self.jobs.get(&job_id)
        .ok_or_else(|| GpuError::JobNotFound(job_id.to_string()))?;

    if !matches!(status, JobStatus::Running { .. }) {
        return Err(GpuError::JobNotFound(format!(
            "{} is not in Running state", job_id
        )));
    }

    let container_id = *container_id;

    // Update job status
    self.jobs.insert(job_id, (container_id, JobStatus::Completed {
        tensor_id: 0, // filled by caller if needed
        exec_time_ns,
    }));

    // Update container stats
    if let Some(container) = self.containers.get_mut(&container_id) {
        container.cumulative_exec_time_ns += exec_time_ns;
        container.total_jobs_completed += 1;
        container.pending_jobs = container.pending_jobs.saturating_sub(1);
    }

    Ok(())
}

pub fn fail_job(&mut self, job_id: JobId, error: String) -> Result<()> {
    let (container_id, status) = self.jobs.get(&job_id)
        .ok_or_else(|| GpuError::JobNotFound(job_id.to_string()))?;

    if !matches!(status, JobStatus::Running { .. }) {
        return Err(GpuError::JobNotFound(format!(
            "{} is not in Running state", job_id
        )));
    }

    let container_id = *container_id;

    self.jobs.insert(job_id, (container_id, JobStatus::Failed { error }));

    if let Some(container) = self.containers.get_mut(&container_id) {
        container.pending_jobs = container.pending_jobs.saturating_sub(1);
    }

    Ok(())
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core test_job_status test_complete test_fail_job test_deregister_rejects test_deregister_drains`
Expected: PASS

- [ ] **Step 5: Run full Rust test suite**

Run: `cargo test -p applegpu-core`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat: add job completion, failure, and status tracking to scheduler"
```

---

## Chunk 4: LazyRuntime Integration

### Task 11: Integrate Scheduler into LazyRuntime

**Files:**
- Modify: `crates/core/src/lazy.rs:15-34` (struct fields and constructor)
- Modify: `crates/core/src/lazy.rs:38-43` (insert_tensor)
- Modify: `crates/core/src/lazy.rs:86-116` (eval — container resolution + resource tracking)
- Modify: `crates/core/src/lazy.rs:222-288` (eval_remote — container resolution)
- Modify: `crates/core/src/lazy.rs:290-306` (destroy — free_tensor)
- Modify: `crates/core/src/lazy.rs:308-318` (memory_usage, set_limits — delegate to scheduler)

- [ ] **Step 1: Modify LazyRuntime struct**

Remove `limits` and `tracker` fields. Add `scheduler` field. Update constructor:

```rust
use crate::scheduler::{ContainerId, Scheduler};

pub struct LazyRuntime {
    tensors: HashMap<u64, Tensor>,
    graph: Graph,
    pub scheduler: Scheduler,
}

impl LazyRuntime {
    pub fn new() -> Self {
        LazyRuntime {
            tensors: HashMap::new(),
            graph: Graph::new(),
            scheduler: Scheduler::new(ResourceLimits::from_env()),
        }
    }
```

- [ ] **Step 2: Modify insert_tensor to use scheduler**

```rust
pub fn insert_tensor(&mut self, tensor: Tensor) -> Result<()> {
    let size = tensor.buffer.len();
    let id = tensor.meta.id;
    self.scheduler.allocate_tensor(ContainerId::DEFAULT, id, size)?;
    self.tensors.insert(id, tensor);
    Ok(())
}

/// Insert a tensor attributed to a specific container.
pub fn insert_tensor_for(&mut self, tensor: Tensor, container_id: ContainerId) -> Result<()> {
    let size = tensor.buffer.len();
    let id = tensor.meta.id;
    self.scheduler.allocate_tensor(container_id, id, size)?;
    self.tensors.insert(id, tensor);
    Ok(())
}
```

- [ ] **Step 3: Modify eval to resolve container and track intermediates**

```rust
pub fn eval(&mut self, device: &Device, id: u64) -> Result<()> {
    if self.is_materialized(id) {
        return Ok(());
    }

    // Resolve container for this eval
    let container_id = self.resolve_container(id);

    let mut order = self.graph.topo_sort(id)?;
    if order.is_empty() {
        return Err(GpuError::GraphError(format!("Tensor {} not found", id)));
    }

    order = crate::fusion::optimize(&mut self.graph, &order);

    for node_id in order {
        if self.is_materialized(node_id) {
            continue;
        }

        let node = self.graph.remove_node(node_id).ok_or_else(|| {
            GpuError::GraphError(format!("Node {} not found in graph", node_id))
        })?;

        let result = self.execute_node(device, &node)?;
        let size = result.buffer.len();
        self.scheduler.allocate_tensor(container_id, node_id, size)?;
        self.tensors.insert(node_id, result);
    }

    Ok(())
}

/// Resolve which container a tensor belongs to.
fn resolve_container(&self, id: u64) -> ContainerId {
    // Check tensor_owners first (materialized tensors)
    if let Some(cid) = self.scheduler.tensor_owner(id) {
        return cid;
    }
    // Check graph node (pending ops)
    if let Some(node) = self.graph.get_node(id) {
        return node.container_id;
    }
    // Fallback to default
    ContainerId::DEFAULT
}
```

- [ ] **Step 4: Modify eval_remote to use container tracking**

In `eval_remote`, replace:
```rust
self.tracker.track_alloc(size);
```
with:
```rust
let container_id = self.resolve_container(id);
self.scheduler.allocate_tensor(container_id, tensor_id, size)?;
```

(Keep the full method but update the tracking lines.)

- [ ] **Step 5: Modify destroy to use free_tensor**

```rust
pub fn destroy(&mut self, id: u64) -> Result<()> {
    for node in self.graph.iter_nodes() {
        if node.inputs.contains(&id) {
            return Err(GpuError::GraphError(format!(
                "Cannot destroy tensor {} while pending op {} depends on it",
                id, node.id
            )));
        }
    }
    if let Some(tensor) = self.tensors.remove(&id) {
        self.scheduler.free_tensor(id, tensor.buffer.len());
    }
    self.graph.remove_node(id);
    Ok(())
}
```

- [ ] **Step 6: Update query methods to delegate to scheduler**

```rust
pub fn memory_usage(&self) -> usize {
    self.scheduler.global_usage().0
}

pub fn live_tensor_count(&self) -> usize {
    self.scheduler.global_usage().1
}

pub fn set_limits(&mut self, limits: ResourceLimits) {
    self.scheduler.update_global_limits(limits);
}
```

This requires adding `update_global_limits` to `Scheduler`:

```rust
/// Update global limits and the default container's limits in-place.
/// Preserves all registered containers and state.
pub fn update_global_limits(&mut self, limits: ResourceLimits) {
    self.global_limits = limits.clone();
    if let Some(default) = self.containers.get_mut(&ContainerId::DEFAULT) {
        default.limits = limits.clone();
        default.config.max_memory_bytes = limits.max_total_memory_bytes;
        default.config.max_tensor_count = limits.max_tensor_count;
        default.config.max_tensor_size_bytes = limits.max_tensor_size_bytes;
    }
}
```

Add this method as part of Task 4 Step 3's Scheduler impl.

- [ ] **Step 7: Fix LazyRuntime tests**

Update tests in `lazy.rs` that reference `tracker` or `limits` to use `scheduler` instead. The key changes:
- Tests that call `rt.insert_tensor(t).unwrap()` still work (signature unchanged)
- Tests that check `rt.tracker` or `rt.limits` should use `rt.scheduler.global_usage()` or `rt.scheduler.container_usage(ContainerId::DEFAULT)`

- [ ] **Step 8: Run all Rust tests**

Run: `cargo test -p applegpu-core`
Expected: All tests pass

- [ ] **Step 9: Commit**

```bash
git add crates/core/src/lazy.rs
git commit -m "feat: integrate scheduler into LazyRuntime as single resource authority"
```

### Task 12: Add run_next execution loop

**Files:**
- Modify: `crates/core/src/lazy.rs`

- [ ] **Step 1: Write test**

This test requires a GPU device, so it goes in the lazy.rs test module:

```rust
#[test]
fn test_run_next_dequeue_eval_complete() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = crate::ops::add(&mut rt, a_id, b_id).unwrap();

    // Submit a job for the lazy tensor
    let job_id = rt.scheduler.submit(ContainerId::DEFAULT, c_id).unwrap();
    assert_eq!(rt.scheduler.queue_depth(), 1);

    // Run the next job
    let result = rt.run_next(&device).unwrap();
    assert_eq!(result, Some(job_id));
    assert!(rt.is_materialized(c_id));
    assert_eq!(rt.read_f32(c_id).unwrap(), &[11.0, 22.0, 33.0, 44.0]);
    assert_eq!(rt.scheduler.queue_depth(), 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core test_run_next`
Expected: FAIL

- [ ] **Step 3: Implement run_next**

```rust
/// Dequeue the next scheduled job, evaluate it, and complete it.
/// Returns the job ID if a job was executed, or None if the queue is empty.
pub fn run_next(&mut self, device: &Device) -> Result<Option<JobId>> {
    let job = match self.scheduler.next_job() {
        Some(j) => j,
        None => return Ok(None),
    };

    let job_id = job.id;
    let target_id = job.target_tensor_id;
    let start = std::time::Instant::now();

    match self.eval(device, target_id) {
        Ok(()) => {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.scheduler.complete_job(job_id, elapsed)?;
            Ok(Some(job_id))
        }
        Err(e) => {
            self.scheduler.fail_job(job_id, e.to_string())?;
            Err(e)
        }
    }
}
```

- [ ] **Step 4: Run test**

Run: `cargo test -p applegpu-core test_run_next`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/lazy.rs
git commit -m "feat: add run_next execution loop to LazyRuntime"
```

---

## Chunk 5: Python Bindings and Integration Tests

### Task 13: Add scheduler Python bindings

**Files:**
- Modify: `crates/python/src/lib.rs`

- [ ] **Step 1: Add new Python functions**

Add these functions before the `#[pymodule]` block in `crates/python/src/lib.rs`:

```rust
use applegpu_core::scheduler::{ContainerId, ContainerConfig, Priority};

#[pyfunction]
fn register_container(
    priority: &str,
    max_memory_mb: usize,
    max_tensors: usize,
    max_pending: usize,
) -> PyResult<u64> {
    let priority = match priority {
        "high" => Priority::High,
        "normal" => Priority::Normal,
        "low" => Priority::Low,
        _ => return Err(PyValueError::new_err(format!("Invalid priority: {}", priority))),
    };
    let config = ContainerConfig {
        priority,
        max_memory_bytes: max_memory_mb * 1024 * 1024,
        max_tensor_count: max_tensors,
        max_tensor_size_bytes: 0,
        max_pending_jobs: max_pending,
    };
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = rt.scheduler.register_container(config)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(id.0)
}

#[pyfunction]
fn deregister_container(container_id: u64) -> PyResult<Vec<u64>> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let tensors = rt.scheduler.deregister_container(ContainerId(container_id))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Free the tensors from the runtime
    for &tid in &tensors {
        rt.remove_tensor_raw(tid);
    }
    Ok(tensors)
}

#[pyfunction]
fn pause_container(container_id: u64) -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.scheduler.pause_container(ContainerId(container_id))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn resume_container(container_id: u64) -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.scheduler.resume_container(ContainerId(container_id))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn submit_job(container_id: u64, t: &GpuTensor) -> PyResult<u64> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let job_id = rt.scheduler.submit(ContainerId(container_id), t.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(job_id.0)
}

#[pyfunction]
fn run_next() -> PyResult<Option<u64>> {
    let runtime = get_device_runtime()?;
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let result = rt.run_next(&runtime.device)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(result.map(|j| j.0))
}

#[pyfunction]
fn job_status(job_id: u64) -> PyResult<String> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    use applegpu_core::scheduler::JobId;
    match rt.scheduler.job_status(JobId(job_id)) {
        Some((_, applegpu_core::scheduler::JobStatus::Queued)) => Ok("queued".to_string()),
        Some((_, applegpu_core::scheduler::JobStatus::Running { .. })) => Ok("running".to_string()),
        Some((_, applegpu_core::scheduler::JobStatus::Completed { .. })) => Ok("completed".to_string()),
        Some((_, applegpu_core::scheduler::JobStatus::Failed { .. })) => Ok("failed".to_string()),
        None => Err(PyValueError::new_err(format!("Job {} not found", job_id))),
    }
}

#[pyfunction]
fn container_usage(container_id: u64) -> PyResult<(usize, usize)> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    rt.scheduler.container_usage(ContainerId(container_id))
        .ok_or_else(|| PyValueError::new_err(format!("Container {} not found", container_id)))
}

#[pyfunction]
fn global_usage() -> PyResult<(usize, usize)> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    Ok(rt.scheduler.global_usage())
}

#[pyfunction]
fn queue_depth() -> PyResult<usize> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    Ok(rt.scheduler.queue_depth())
}
```

Note: `deregister_container` calls `rt.remove_tensor_raw(tid)` — we need to add a `tensors_remove` helper to `LazyRuntime` (a simple `self.tensors.remove(&id)` wrapper), or handle cleanup differently. Add to `lazy.rs`:

```rust
/// Remove a tensor from the runtime without any tracking (used by scheduler deregister cleanup).
pub fn remove_tensor_raw(&mut self, id: u64) {
    self.tensors.remove(&id);
    self.graph.remove_node(id);
}
```

Update `deregister_container` in Python to use `rt.remove_tensor_raw(tid)`.

- [ ] **Step 2: Register all new functions in the pymodule block**

Add to the `#[pymodule]` function:

```rust
m.add_function(wrap_pyfunction!(register_container, m)?)?;
m.add_function(wrap_pyfunction!(deregister_container, m)?)?;
m.add_function(wrap_pyfunction!(pause_container, m)?)?;
m.add_function(wrap_pyfunction!(resume_container, m)?)?;
m.add_function(wrap_pyfunction!(submit_job, m)?)?;
m.add_function(wrap_pyfunction!(run_next, m)?)?;
m.add_function(wrap_pyfunction!(job_status, m)?)?;
m.add_function(wrap_pyfunction!(container_usage, m)?)?;
m.add_function(wrap_pyfunction!(global_usage, m)?)?;
m.add_function(wrap_pyfunction!(queue_depth, m)?)?;
```

- [ ] **Step 3: Update set_limits and memory_usage/tensor_count to delegate through scheduler**

`set_limits`, `memory_usage`, and `tensor_count` already work because `LazyRuntime` methods now delegate to the scheduler.

- [ ] **Step 4: Build the Python extension**

Run: `cd /Users/noahmoore/applegpu_runtime && uv run maturin develop`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add crates/python/src/lib.rs crates/core/src/lazy.rs
git commit -m "feat: add scheduler Python bindings (register, submit, run_next, etc.)"
```

### Task 14: Python tests

**Files:**
- Create: `python/tests/test_scheduler.py`

- [ ] **Step 1: Write Python tests**

```python
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


def test_backward_compatibility():
    """Existing API works unchanged with scheduler underneath."""
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], [4])
    c = a + b
    result = c.to_list()
    assert result == [11.0, 22.0, 33.0, 44.0]


def test_register_deregister_container():
    cid = gpu.register_container(priority="normal", max_memory_mb=1, max_tensors=10, max_pending=5)
    assert cid > 0
    tensors = gpu.deregister_container(cid)
    assert tensors == []


def test_container_usage_query():
    cid = gpu.register_container(priority="normal", max_memory_mb=10, max_tensors=100, max_pending=50)
    bytes_used, count = gpu.container_usage(cid)
    assert bytes_used == 0
    assert count == 0
    gpu.deregister_container(cid)


def test_global_usage():
    before_bytes, before_count = gpu.global_usage()
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    after_bytes, after_count = gpu.global_usage()
    assert after_bytes > before_bytes
    assert after_count == before_count + 1


def test_queue_depth():
    assert gpu.queue_depth() == 0
    a = gpu.tensor([1.0, 2.0], [2])
    b = gpu.tensor([3.0, 4.0], [2])
    c = a + b  # lazy
    job_id = gpu.submit_job(0, c)  # submit to default container
    assert gpu.queue_depth() == 1
    result_id = gpu.run_next()
    assert result_id == job_id
    assert gpu.queue_depth() == 0
    assert gpu.job_status(job_id) == "completed"


def test_submit_and_run_job():
    a = gpu.tensor([1.0, 2.0, 3.0], [3])
    b = gpu.tensor([4.0, 5.0, 6.0], [3])
    c = a + b
    job_id = gpu.submit_job(0, c)
    assert gpu.job_status(job_id) == "queued"
    gpu.run_next()
    assert gpu.job_status(job_id) == "completed"
    assert c.to_list() == [5.0, 7.0, 9.0]


def test_priority_scheduling():
    high_cid = gpu.register_container(priority="high", max_memory_mb=1, max_tensors=10, max_pending=10)
    low_cid = gpu.register_container(priority="low", max_memory_mb=1, max_tensors=10, max_pending=10)
    # Create lazy tensors for each container
    # (Both use default container for data, but submit jobs to their own containers)
    a = gpu.tensor([1.0, 2.0], [2])
    b = gpu.tensor([3.0, 4.0], [2])
    c_low = a + b
    c_high = a + b
    gpu.submit_job(low_cid, c_low)
    high_job = gpu.submit_job(high_cid, c_high)
    # High-priority job should run first
    result_id = gpu.run_next()
    assert result_id == high_job
    gpu.deregister_container(high_cid)
    gpu.deregister_container(low_cid)


def test_admission_control():
    cid = gpu.register_container(priority="normal", max_memory_mb=1, max_tensors=10, max_pending=1)
    a = gpu.tensor([1.0], [1])
    b = gpu.tensor([2.0], [1])
    c1 = a + b
    c2 = a + b
    gpu.submit_job(cid, c1)
    with pytest.raises(ValueError, match="Admission rejected"):
        gpu.submit_job(cid, c2)
    gpu.deregister_container(cid)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest python/tests/test_scheduler.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_scheduler.py
git commit -m "test: add Python scheduler tests"
```

### Task 15: Rust integration tests

**Files:**
- Create: `crates/core/tests/scheduler_integration.rs`

- [ ] **Step 1: Write integration tests**

```rust
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::scheduler::ContainerId;
use applegpu_core::tensor::Tensor;

fn get_device() -> Option<Device> {
    Device::new().ok()
}

#[test]
fn test_default_container_backward_compat() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();
    let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let id = t.meta.id;
    rt.insert_tensor(t).unwrap();
    assert!(rt.is_materialized(id));
    assert_eq!(rt.scheduler.container_usage(ContainerId::DEFAULT), Some((16, 1))); // 4 f32 = 16 bytes
}

#[test]
fn test_resource_tracking_through_lifecycle() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let id = t.meta.id;
    rt.insert_tensor(t).unwrap();

    let (bytes, count) = rt.scheduler.global_usage();
    assert!(bytes > 0);
    assert_eq!(count, 1);

    rt.destroy(id).unwrap();
    assert_eq!(rt.scheduler.global_usage(), (0, 0));
}

#[test]
fn test_multi_container_round_robin() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    use applegpu_core::scheduler::{ContainerConfig, Priority};

    let config_a = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 100,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let config_b = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 100,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };

    let a = rt.scheduler.register_container(config_a).unwrap();
    let b = rt.scheduler.register_container(config_b).unwrap();

    // Create tensors
    let t1 = Tensor::from_f32(&device, vec![2], &[1.0, 2.0]).unwrap();
    let t2 = Tensor::from_f32(&device, vec![2], &[3.0, 4.0]).unwrap();
    let t1_id = t1.meta.id;
    let t2_id = t2.meta.id;
    rt.insert_tensor(t1).unwrap();
    rt.insert_tensor(t2).unwrap();

    // Create lazy ops
    let c1 = applegpu_core::ops::add(&mut rt, t1_id, t2_id).unwrap();
    let c2 = applegpu_core::ops::mul(&mut rt, t1_id, t2_id).unwrap();

    // Submit to different containers
    rt.scheduler.submit(a, c1).unwrap();
    rt.scheduler.submit(b, c2).unwrap();

    // Both should complete (fair round-robin since both at 0 exec time)
    let j1 = rt.run_next(&device).unwrap();
    assert!(j1.is_some());
    let j2 = rt.run_next(&device).unwrap();
    assert!(j2.is_some());

    assert!(rt.is_materialized(c1));
    assert!(rt.is_materialized(c2));
}

#[test]
fn test_eval_attributes_intermediates_to_container() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    use applegpu_core::scheduler::{ContainerConfig, Priority};

    let config = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 100,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let cid = rt.scheduler.register_container(config).unwrap();

    // Create input tensors in default container
    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let a_id = a.meta.id;
    rt.insert_tensor(a).unwrap();

    // Record ops - they get DEFAULT container_id since ops.rs uses DEFAULT
    let neg_id = applegpu_core::ops::neg(&mut rt, a_id).unwrap();
    let relu_id = applegpu_core::ops::relu(&mut rt, neg_id).unwrap();

    // Eval - intermediate (neg_id) should be attributed to the same container as the target
    rt.eval(&device, relu_id).unwrap();
    assert!(rt.is_materialized(relu_id));

    // Verify container attribution: both intermediate and result belong to DEFAULT
    // (because ops.rs uses DEFAULT, and resolve_container falls back to DEFAULT)
    assert_eq!(rt.scheduler.tensor_owner(relu_id), Some(ContainerId::DEFAULT));
}
```

- [ ] **Step 2: Run integration tests**

Run: `cargo test -p applegpu-core --test scheduler_integration`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add crates/core/tests/scheduler_integration.rs
git commit -m "test: add scheduler integration tests"
```

### Task 16: Run full test suite and update README

- [ ] **Step 1: Run all Rust tests**

Run: `cargo test -p applegpu-core`
Expected: All pass

- [ ] **Step 2: Build and run Python tests**

Run: `cd /Users/noahmoore/applegpu_runtime && uv run maturin develop && uv run pytest -v`
Expected: All pass

- [ ] **Step 3: Update README with scheduler capabilities**

Add a section about multi-container scheduling to the README.

- [ ] **Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: update README for Phase 6 multi-container scheduler"
```

- [ ] **Step 5: Update project status memory**

Update the project status memory file with:
- Phase 6 completed
- New test count
- Updated remaining phases
