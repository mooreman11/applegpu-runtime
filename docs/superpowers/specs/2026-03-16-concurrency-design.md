# Concurrency: Concurrent Queues + Fine-Grained Locking + Async Eval

**Date:** 2026-03-16
**Status:** Draft
**Phases:** 2c (concurrent queues) + 2d (async eval + fine-grained locking), merged

## Problem

The runtime serializes all work behind a single `Mutex<LazyRuntime>`. GPU ops within an eval are encoded into one `MTLCommandBuffer` on one `MTLCommandQueue`, executing sequentially even when the computation graph has independent branches. This leaves GPU ALUs idle when individual kernels don't saturate them.

Three improvements are needed:
1. **GPU-side parallelism** — dispatch independent subgraphs to separate Metal command queues
2. **CPU-side parallelism** — split the single Mutex so metadata queries, op recording, and eval don't block each other
3. **Non-blocking Python** — `eval_async()` returns a future so Python can do useful work while the GPU executes

## Current Architecture

```
MetalBackend {
    runtime: Mutex<LazyRuntime>    // ONE lock for everything
}

LazyRuntime {
    tensors: HashMap<u64, Tensor>  // materialized GPU buffers
    graph: Graph                    // pending computation DAG
    scheduler: Scheduler            // multi-container resource tracking
    pool: BufferPool                // GPU buffer recycling
}
```

**Eval flow** (`lazy.rs:102-163`):
1. Lock mutex (held for entire eval)
2. `graph.topo_sort(id)` → flat Vec<u64>
3. `fusion::optimize()` → fuse elementwise chains
4. `begin_batch(queue)` → create single MTLCommandBuffer
5. Loop: for each node, acquire output buffer, encode op into CB
6. `end_batch()` → commit CB
7. `wait_command_buffer()` → block until GPU finishes
8. Unlock mutex

**Swift batch model** (`compute.swift`):
- One global `sharedCommandQueue: MTLCommandQueue?`
- One global `activeBatchCommandBuffer: MTLCommandBuffer?`
- 30 `_nb` dispatch functions check `activeBatchCommandBuffer` to decide encoding target
- `batchLock: NSLock` guards the global CB

## Phase I: Concurrent Metal Queues

**Goal:** GPU-side parallelism within a single eval call. Single Mutex retained — all changes are internal to eval().

### 1.1 Parallel Level Computation

Add `Graph::parallel_levels(target_id) -> Vec<Vec<u64>>` that partitions the topo-sorted subgraph into depth levels. Nodes at the same level have no data dependencies on each other and can execute concurrently.

**Algorithm** (BFS from leaves):
```
fn parallel_levels(&self, target_id: u64) -> Result<Vec<Vec<u64>>> {
    let order = self.topo_sort(target_id)?;
    let mut depth: HashMap<u64, usize> = HashMap::new();

    for &id in &order {
        let node = self.get_node(id).unwrap();
        let d = node.inputs.iter()
            .filter_map(|&inp| depth.get(&inp))
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        depth.insert(id, d);
    }

    let max_depth = depth.values().copied().max().unwrap_or(0);
    let mut levels = vec![Vec::new(); max_depth + 1];
    for &id in &order {
        levels[depth[&id]].push(id);
    }
    Ok(levels)
}
```

This is target-scoped (only nodes reachable from `target_id`), runs in O(V+E), and produces levels where:
- Level 0: nodes whose inputs are all materialized tensors (leaves)
- Level N: nodes whose deepest input dependency is at level N-1

**Linearity detection:** If `levels.iter().all(|l| l.len() == 1)`, the graph is linear — fall back to existing single-CB path with zero overhead.

### 1.2 Metal Queue Pool

Replace the single `sharedCommandQueue` with a reusable pool of 2-4 queues, created once at initialization.

```swift
private var queuePool: [MTLCommandQueue] = []
private let queuePoolLock = NSLock()
private let maxQueues = 4

@_cdecl("gpu_bridge_get_queue")
public func gpuBridgeGetQueue(_ devicePtr: UnsafeRawPointer?, _ index: UInt32) -> UnsafeMutableRawPointer?
```

Queues are heavyweight Metal objects — create at init, reuse forever. The pool size (4) matches typical independent branch counts in transformer graphs (Q, K, V projections + residual).

### 1.3 Batch Context (Replacing Global State)

Instead of changing all 30 `_nb` function signatures, replace the global `activeBatchCommandBuffer` with a **batch context** system:

```swift
/// Thread-local or explicit batch context.
/// Each concurrent dispatch lane has its own context.
private var batchContexts: [UInt32: MTLCommandBuffer] = [:]
private let contextLock = NSLock()

@_cdecl("gpu_bridge_set_batch_context")
public func gpuBridgeSetBatchContext(_ contextId: UInt32, _ queuePtr: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer?

@_cdecl("gpu_bridge_commit_batch_context")
public func gpuBridgeCommitBatchContext(_ contextId: UInt32) -> UnsafeMutableRawPointer?
```

The existing `_nb` functions are updated to check `batchContexts[currentContextId]` instead of the single global. Context ID 0 maps to the legacy single-batch behavior for backward compatibility.

**Thread safety across phases:** In Phase I, all encoding is single-threaded (the outer Mutex serializes eval calls), so `set_active_context` with a module-level `currentContextId` is safe. In Phase II, concurrent eval calls from different threads must use **thread-local storage** for `currentContextId` (Swift's `Thread.current.threadDictionary` or a pthread-specific key). The batch context map itself is already guarded by `contextLock`.

**Rust FFI additions** (`ffi.rs`, `compute.rs`):
```rust
extern "C" {
    fn gpu_bridge_get_queue(device: *const GPUDeviceHandle, index: u32) -> *mut c_void;
    fn gpu_bridge_set_batch_context(context_id: u32, queue: *mut c_void) -> *mut c_void;
    fn gpu_bridge_commit_batch_context(context_id: u32) -> *mut c_void;
}
```

### 1.4 MTLEvent Synchronization

Use `MTLEvent` for zero-bubble inter-level synchronization. All levels are encoded upfront on the CPU; the GPU pipelines level transitions automatically.

```swift
private var sharedEvent: MTLEvent?

@_cdecl("gpu_bridge_create_event")
public func gpuBridgeCreateEvent(_ devicePtr: UnsafeRawPointer?) -> UnsafeMutableRawPointer?

@_cdecl("gpu_bridge_encode_signal_event")
public func gpuBridgeEncodeSignalEvent(_ cbPtr: UnsafeMutableRawPointer?, _ eventPtr: UnsafeMutableRawPointer?, _ value: UInt64)

@_cdecl("gpu_bridge_encode_wait_event")
public func gpuBridgeEncodeWaitEvent(_ cbPtr: UnsafeMutableRawPointer?, _ eventPtr: UnsafeMutableRawPointer?, _ value: UInt64)
```

**Flow:**
1. Create one `MTLEvent` per eval
2. For level N: encode ops into per-queue CBs, then `encodeSignalEvent(event, value: N)`
3. For level N+1: `encodeWaitEvent(event, value: N)` at the start of each CB
4. Commit all CBs across all queues at once
5. Wait only on the final level's CBs

This eliminates GPU bubbles — level N+1 starts the instant level N finishes, because the work was already encoded and queued.

### 1.5 Restructured eval()

**Important:** The fusion pass must run before parallel levels are computed, since fusion rewrites the graph topology (replacing elementwise chains with `FusedElementwise` nodes).

```rust
pub fn eval(&mut self, device: &Device, id: u64) -> Result<()> {
    if self.is_materialized(id) {
        return Ok(());
    }

    let container_id = self.resolve_container(id);

    // Run fusion BEFORE computing parallel levels (fusion mutates graph topology)
    let order = self.graph.topo_sort(id)?;
    let _fused_order = crate::fusion::optimize(&mut self.graph, &order);

    let levels = self.graph.parallel_levels(id)?;

    // Fast path: linear graph → existing single-CB path
    if levels.iter().all(|l| l.len() == 1) {
        return self.eval_single_cb(device, id, container_id, levels);
    }

    let event = compute::create_event(device);
    if event.is_null() {
        // MTLEvent creation failed — fall back to single-CB path
        return self.eval_single_cb(device, id, container_id, levels);
    }

    let num_queues = std::cmp::min(
        levels.iter().map(|l| l.len()).max().unwrap_or(1),
        4,
    );
    let queues: Vec<_> = (0..num_queues)
        .map(|i| compute::get_queue(device, i as u32))
        .collect();

    // Track all command buffers for final wait
    let mut all_cbs: Vec<*mut c_void> = Vec::new();

    let loop_result: Result<()> = (|| {
        for (level_idx, level) in levels.iter().enumerate() {
            // Create ONE command buffer per QUEUE (not per node).
            // Multiple nodes sharing a queue share a CB.
            let mut queue_cbs: HashMap<usize, *mut c_void> = HashMap::new();
            for (i, _) in level.iter().enumerate() {
                let queue_idx = i % num_queues;
                queue_cbs.entry(queue_idx).or_insert_with(|| {
                    let ctx_id = (queue_idx + 1) as u32;
                    let cb = compute::set_batch_context(ctx_id, queues[queue_idx]);
                    // Wait for previous level's completion
                    if level_idx > 0 {
                        compute::encode_wait_event(cb, event, level_idx as u64);
                    }
                    cb
                });
            }

            // Encode ops for this level — each node uses its queue's CB
            for (node_idx, &node_id) in level.iter().enumerate() {
                if self.is_materialized(node_id) { continue; }
                let node = self.graph.remove_node(node_id)
                    .ok_or_else(|| GpuError::GraphError(format!("Node {} not found", node_id)))?;
                let out_buf = self.pool.acquire(device, node.out_shape.numel() * node.out_dtype.size_bytes())?;
                let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);

                // Set active context to this node's queue
                let queue_idx = node_idx % num_queues;
                compute::set_active_context((queue_idx + 1) as u32);
                self.execute_node_nb(device, queues[queue_idx], &node, &out)?;

                self.scheduler.allocate_tensor(container_id, node_id, node.out_shape.numel() * node.out_dtype.size_bytes())?;
                self.tensors.insert(node_id, out);
            }

            // Signal completion + commit each queue's CB for this level
            for (&queue_idx, &cb) in &queue_cbs {
                let ctx_id = (queue_idx + 1) as u32;
                compute::encode_signal_event(cb, event, (level_idx + 1) as u64);
                let committed = compute::commit_batch_context(ctx_id);
                all_cbs.push(committed);
            }
        }
        Ok(())
    })();

    // Cleanup: always destroy the event, even on error
    compute::destroy_event(event);

    // Wait for all committed command buffers
    if loop_result.is_ok() {
        for cb in &all_cbs {
            if !cb.is_null() {
                compute::wait_command_buffer(*cb);
            }
        }
    } else {
        // On error, still wait for any committed CBs to avoid dangling GPU work
        for cb in &all_cbs {
            if !cb.is_null() {
                compute::wait_command_buffer(*cb);
            }
        }
    }

    loop_result
}
```

### 1.6 Buffer Initialization Safety

Ops like `embedding_backward` call `out.buffer.zero()` (CPU-side memset) before kernel dispatch. With concurrent queues, this zeroing must complete before the kernel reads/writes the buffer. Since zeroing happens on the CPU before the compute encoder is created, and Metal command buffer encoding is CPU-side, the ordering is naturally correct — the `zero()` call completes before `encode_compute_command` adds the dispatch. No additional synchronization needed.

### 1.7 Testing

- **Correctness**: existing test suite must pass unchanged (single-CB fast path)
- **Parallelism**: construct a graph with 4 independent branches, verify all execute (compare output to single-CB)
- **Benchmark**: GPT-2 small (Q/K/V parallel branches), measure wall-clock time vs single-CB baseline
- **Edge cases**: single-node graph, linear chain, diamond DAG (A→B, A→C, B→D, C→D)

## Phase II: Fine-Grained Locking

**Goal:** CPU-side parallelism. Multiple Python threads can record ops, query metadata, and eval concurrently.

### 2.1 Lock Decomposition

Replace `Mutex<LazyRuntime>` with internally-synchronized `ConcurrentRuntime`:

```rust
pub struct ConcurrentRuntime {
    tensors: RwLock<HashMap<u64, Tensor>>,
    graph: Mutex<Graph>,
    scheduler: Mutex<Scheduler>,
    pool: Mutex<BufferPool>,
}
```

### 2.2 Lock Ordering Protocol

**Strict acquisition order:** `graph → tensors → scheduler → pool`

Any operation that needs multiple locks must acquire them in this order. Operations may skip locks they don't need, but must never acquire a later-priority lock before an earlier one.

**Verification by operation:**

| Operation | Locks needed | Order | Notes |
|-----------|-------------|-------|-------|
| `record_op()` | graph(write) | graph only | OK |
| `shape()` / `dtype()` | graph(read), tensors(read) | **graph → tensors** | Restructure to try graph first ✓ |
| `is_pending()` | graph(read) | graph only | OK |
| `is_materialized()` | tensors(read) | tensors only | OK |
| `insert_tensor()` | tensors(write), scheduler(write) | **tensors → scheduler** | OK (skips graph) ✓ |
| `destroy()` | See below | Phased | Restructure into phases ✓ |
| `eval()` | See 2.3 | Phased | ✓ |
| `pool_stats()` | pool(read) | pool only | OK |
| `scheduler ops` | scheduler(read/write) | scheduler only | OK |

**Note on `shape()`:** Currently tries tensors first, then graph. Must be restructured to try graph first (graph → tensors) to respect lock ordering. This is semantically equivalent — if the node is in the graph, its shape is in `node.out_shape`; if materialized, it's in `tensor.meta`.

**Note on `destroy()`:** Currently needs graph(read) + tensors(write) + scheduler(write) + pool(write) in interleaved fashion. Must be restructured into phases:
1. Lock graph(read): check if any node depends on this tensor. Release graph lock.
2. Lock tensors(write): remove tensor, capture buffer. Release tensors lock.
3. Lock scheduler(write): free_tensor. Release scheduler lock.
4. Lock pool(write): release buffer. Release pool lock.

Each phase acquires and releases one lock. The lock ordering is respected because phases proceed in order: graph → tensors → scheduler → pool.

**TOCTOU safety for `destroy()`:** Between checking graph dependents (phase 1) and removing from tensors (phase 2), another thread could add a new dependent. Mitigation: `destroy()` re-checks under the tensors write lock by calling `graph.ref_count(id)` (requires briefly re-acquiring graph read lock inside tensors write lock — this is graph → tensors order, which is correct). If ref count changed, abort the destroy.

**Note on `eval()` Phase II window:** Between graph extraction (Phase 1 of eval) and tensor insertion (Phase 5), `shape()` queries for in-flight nodes will find them in neither location and return an error. This is expected and documented behavior. Callers should retry or use `exists()` which will return false for in-flight nodes.

**Note on `eval_remote()`:** The remote eval path (`eval_remote` in `metal_backend.rs`) is out of scope for this spec. It serializes ops over a socket and executes on a remote gpu-service. It will continue to use a simple mutex-style lock for the entire remote call. Fine-grained locking and concurrent queues are Metal-local optimizations.

### 2.3 Phased Eval

The eval loop currently interleaves access to all four components per node. Restructure into phases that hold each lock briefly:

```rust
pub fn eval(&self, device: &Device, id: u64) -> Result<()> {
    // Phase 1: Extract execution plan (lock graph briefly)
    let (levels, nodes) = {
        let mut graph = self.graph.lock().unwrap();
        let levels = graph.parallel_levels(id)?;
        let nodes: HashMap<u64, OpNode> = levels.iter()
            .flat_map(|l| l.iter())
            .filter_map(|&id| graph.remove_node(id).map(|n| (id, n)))
            .collect();
        (levels, nodes)
    }; // graph lock released

    // Phase 2: Acquire all output buffers (lock pool briefly)
    let output_buffers = {
        let mut pool = self.pool.lock().unwrap();
        let mut bufs = HashMap::new();
        for (&id, node) in &nodes {
            let size = node.out_shape.numel() * node.out_dtype.size_bytes();
            bufs.insert(id, pool.acquire(device, size)?);
        }
        bufs
    }; // pool lock released

    // Phase 3: Read input tensors + encode ops (lock tensors for read)
    let output_tensors = {
        let tensors = self.tensors.read().unwrap();
        // ... encode ops using tensors for input buffer lookups ...
        // ... returns Vec<(u64, Tensor)> of outputs to insert ...
    }; // tensors read lock released

    // Phase 4: GPU execution (no locks held)
    // commit all command buffers, wait for completion

    // Phase 5: Insert results (lock tensors for write, then scheduler)
    {
        let mut tensors = self.tensors.write().unwrap();
        for (id, tensor) in output_tensors {
            tensors.insert(id, tensor);
        }
    }
    {
        let mut scheduler = self.scheduler.lock().unwrap();
        for (&id, node) in &nodes {
            let size = node.out_shape.numel() * node.out_dtype.size_bytes();
            scheduler.allocate_tensor(container_id, id, size)?;
        }
    }

    Ok(())
}
```

**Key property:** No locks are held during GPU execution (Phase 4). This is the critical enabler for async eval.

### 2.4 RwLock Considerations

The `tensors` RwLock allows concurrent reads (shape queries, input lookups during eval) while exclusive writes only happen during Phase 5 (insert results). This avoids the "collapses to Mutex" problem because reads and writes are in separate phases.

### 2.5 MetalBackend Changes

`MetalBackend` no longer wraps `Mutex<LazyRuntime>`. Instead it holds `ConcurrentRuntime` directly, and each trait method acquires only the locks it needs:

```rust
pub struct MetalBackend {
    runtime: ConcurrentRuntime,  // internally synchronized
}

impl Backend for MetalBackend {
    fn shape(&self, id: u64) -> BackendResult<Vec<usize>> {
        // Only acquires graph read + tensors read
        self.runtime.shape(id).map_err(|e| e.to_string())
    }

    fn eval(&self, id: u64) -> BackendResult<()> {
        // Acquires locks in phases per 2.3
        let rt = get_device_runtime()?;
        self.runtime.eval(&rt.device, id).map_err(|e| e.to_string())
    }
}
```

### 2.6 Testing

- **Deadlock detection**: run tests under `RUST_LOG=trace` with lock acquisition logging
- **Concurrent ops**: spawn 4 threads: 2 recording ops, 1 evaluating, 1 querying shapes
- **Stress test**: 1000 concurrent `shape()` calls during eval
- **TOCTOU safety**: verify `destroy()` correctly rejects tensors with pending dependents under concurrent access

## Phase III: Async Eval

**Goal:** Non-blocking Python. `gpu.eval_async()` returns a future; Python continues while GPU works.

### 3.1 Python API

```python
future = gpu.eval_async(tensor)     # returns GpuFuture, non-blocking
future.is_ready()                    # poll without blocking
future.wait()                        # block until complete
tensor.to_list()                     # auto-waits if future pending
```

### 3.2 Rust Implementation

```rust
#[pyclass]
pub struct GpuFuture {
    handle: Arc<FutureHandle>,
}

struct FutureHandle {
    ready: AtomicBool,
    result: Mutex<Option<BackendResult<()>>>,
    condvar: Condvar,
}

#[pymethods]
impl GpuFuture {
    fn is_ready(&self) -> bool {
        self.handle.ready.load(Ordering::Acquire)
    }

    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        // Release GIL while waiting
        py.allow_threads(|| {
            let mut result = self.handle.result.lock().unwrap();
            while result.is_none() {
                result = self.handle.condvar.wait(result).unwrap();
            }
        });
        // ...
    }
}
```

### 3.3 Eval Thread

```rust
fn eval_async(&self, id: u64) -> BackendResult<GpuFuture> {
    let handle = Arc::new(FutureHandle::new());
    let runtime = self.runtime.clone(); // ConcurrentRuntime is Arc-wrapped
    let device = get_device_runtime()?.device.clone();

    std::thread::spawn(move || {
        let result = runtime.eval(&device, id);
        let mut guard = handle.result.lock().unwrap();
        *guard = Some(result);
        handle.ready.store(true, Ordering::Release);
        handle.condvar.notify_all();
    });

    Ok(GpuFuture { handle })
}
```

### 3.4 Overlapping Eval Deduplication

When two `eval_async` calls target overlapping subgraphs, both will try to extract and execute shared nodes. This is handled by **node claiming** in the graph extraction phase:

```rust
impl Graph {
    /// Extract nodes for eval, marking them as claimed.
    /// Returns only unclaimed nodes. Claimed nodes are skipped
    /// (another eval is responsible for materializing them).
    pub fn claim_and_extract(&mut self, target_id: u64) -> Result<ClaimedPlan> {
        let levels = self.parallel_levels(target_id)?;
        let mut claimed_nodes = HashMap::new();
        let mut wait_on: Vec<u64> = Vec::new();

        for level in &levels {
            for &id in level {
                if self.is_claimed(id) {
                    // Another eval owns this node — we'll wait for it
                    wait_on.push(id);
                } else {
                    self.claim_node(id);
                    if let Some(node) = self.remove_node(id) {
                        claimed_nodes.insert(id, node);
                    }
                }
            }
        }

        Ok(ClaimedPlan { levels, claimed_nodes, wait_on })
    }
}
```

Nodes in `wait_on` are being materialized by another in-flight eval. The waiting mechanism uses a **condvar-based notification**, not polling:

```rust
/// Per-node materialization signal. Created when a node is claimed.
struct NodeSignal {
    ready: AtomicBool,
    result: Mutex<Option<Result<()>>>,
    condvar: Condvar,
}

/// Map of claimed node IDs to their signals.
claimed_nodes: HashMap<u64, Arc<NodeSignal>>,
```

When the claiming eval materializes a node, it sets `ready = true` and notifies. When it fails, it sets `result = Err(...)` and notifies. Waiters check the result and either proceed (success) or propagate the error (failure). This prevents infinite spinning if the claiming eval fails.

**Timeout:** Waiters have a 30-second timeout. If exceeded, the node is considered lost — the waiter re-inserts a fresh graph node and retries materialization itself.

### 3.5 Auto-Wait on Read

`to_list()`, `to_numpy()`, `read_bytes()`, and similar read operations check for pending futures and auto-wait:

```rust
fn read_bytes(&self, id: u64) -> BackendResult<Vec<u8>> {
    self.wait_if_pending(id)?;  // blocks if async eval in-flight
    // ... existing read logic ...
}
```

### 3.6 Testing

- **Basic**: `eval_async` → `wait()` → `to_list()` matches sync eval
- **Concurrent**: two `eval_async` calls on independent tensors, both complete correctly
- **Overlapping**: two `eval_async` calls sharing a subgraph, both get correct results
- **Auto-wait**: `to_list()` on a tensor with pending async eval blocks and returns correct data
- **GIL release**: verify Python remains responsive during `eval_async` (run Python code between async and wait)

## Migration & Backward Compatibility

- Phase I: internal to eval(), no API changes. Existing tests pass unchanged.
- Phase II: `MetalBackend` struct changes but `Backend` trait is unchanged. Python API unchanged.
- Phase III: new `eval_async()` function and `GpuFuture` class. All existing sync APIs unchanged.

Each phase can be shipped independently. Phase I provides measurable speedup. Phase II enables Phase III.

## Files Changed

| Phase | File | Change |
|-------|------|--------|
| I | `crates/core/src/graph.rs` | Add `parallel_levels()` |
| I | `crates/core/src/lazy.rs` | Restructure `eval()` for multi-queue |
| I | `crates/core/src/compute.rs` | Queue pool, batch context, event FFI wrappers |
| I | `crates/core/src/ffi.rs` | New extern declarations for queue/context/event |
| I | `swift/.../compute.swift` | Queue pool, batch contexts, MTLEvent support |
| I | `crates/core/src/fusion.rs` | Ensure fusion runs before parallel_levels |
| I | `swift/.../include/bridge.h` | New C ABI declarations |
| II | `crates/core/src/lazy.rs` | Split into `ConcurrentRuntime` with per-component locks |
| II | `crates/python/src/metal_backend.rs` | Remove outer Mutex, use ConcurrentRuntime |
| III | `crates/python/src/lib.rs` | Add `eval_async()`, `GpuFuture` class |
| III | `crates/python/src/metal_backend.rs` | Add `eval_async()` to Backend trait |
