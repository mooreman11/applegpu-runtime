# Concurrency Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GPU-side parallelism via concurrent Metal command queues, CPU-side parallelism via fine-grained locking, and non-blocking Python eval via async futures.

**Architecture:** Three phases, each independently shippable. Phase I adds concurrent Metal queues within the existing single Mutex. Phase II splits the Mutex into per-component locks. Phase III adds `eval_async()` with GpuFuture.

**Tech Stack:** Rust (graph analysis, locking), Swift/Metal (queue pool, MTLEvent, batch contexts), PyO3 (async eval API)

**Spec:** `docs/superpowers/specs/2026-03-16-concurrency-design.md`

---

## Chunk 1: Phase I — Concurrent Metal Queues

### Task 1: parallel_levels() graph algorithm

**Files:**
- Modify: `crates/core/src/graph.rs`
- Test: `crates/core/src/graph.rs` (inline `#[cfg(test)]`)

- [ ] **Step 1: Write failing test for parallel_levels on a diamond DAG**

```rust
#[test]
fn test_parallel_levels_diamond() {
    // A -> B, A -> C, B -> D, C -> D
    let mut g = Graph::new();
    let a = OpNode { id: 1, inputs: vec![], out_shape: Shape::from(&[2]), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT };
    let b = OpNode { id: 2, inputs: vec![1], out_shape: Shape::from(&[2]), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT };
    let c = OpNode { id: 3, inputs: vec![1], out_shape: Shape::from(&[2]), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT };
    let d = OpNode { id: 4, inputs: vec![2, 3], out_shape: Shape::from(&[2]), out_dtype: DType::Float32, op: OpKind::Add, container_id: ContainerId::DEFAULT };
    g.add_node(a); g.add_node(b); g.add_node(c); g.add_node(d);

    let levels = g.parallel_levels(4).unwrap();
    assert_eq!(levels.len(), 3); // level 0: [A], level 1: [B, C], level 2: [D]
    assert_eq!(levels[0].len(), 1);
    assert_eq!(levels[1].len(), 2);
    assert_eq!(levels[2].len(), 1);
    assert!(levels[1].contains(&2) && levels[1].contains(&3));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core test_parallel_levels_diamond`
Expected: FAIL — method `parallel_levels` not found

- [ ] **Step 3: Implement parallel_levels**

Add to `Graph` in `crates/core/src/graph.rs`:

```rust
/// Partition a topo-sorted subgraph into parallel depth levels.
/// Nodes at the same level have no data dependencies on each other.
pub fn parallel_levels(&self, target_id: u64) -> crate::error::Result<Vec<Vec<u64>>> {
    let order = self.topo_sort(target_id)?;
    if order.is_empty() {
        return Ok(vec![]);
    }
    let mut depth: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();

    for &id in &order {
        let node = self.get_node(id).ok_or_else(|| {
            crate::error::GpuError::GraphError(format!("Node {} not in graph", id))
        })?;
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

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p applegpu-core test_parallel_levels_diamond`
Expected: PASS

- [ ] **Step 5: Write additional tests — linear chain and single node**

```rust
#[test]
fn test_parallel_levels_linear() {
    // A -> B -> C (all linear, each level has 1 node)
    let mut g = Graph::new();
    g.add_node(OpNode { id: 1, inputs: vec![], out_shape: Shape::from(&[2]), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT });
    g.add_node(OpNode { id: 2, inputs: vec![1], out_shape: Shape::from(&[2]), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT });
    g.add_node(OpNode { id: 3, inputs: vec![2], out_shape: Shape::from(&[2]), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT });

    let levels = g.parallel_levels(3).unwrap();
    assert_eq!(levels.len(), 3);
    assert!(levels.iter().all(|l| l.len() == 1)); // all linear
}

#[test]
fn test_parallel_levels_wide() {
    // 4 independent nodes (no deps) -> all at level 0
    let mut g = Graph::new();
    for i in 1..=4 {
        g.add_node(OpNode { id: i, inputs: vec![], out_shape: Shape::from(&[2]), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT });
    }
    // Fan-in: node 5 depends on all four
    g.add_node(OpNode { id: 5, inputs: vec![1, 2, 3, 4], out_shape: Shape::from(&[2]), out_dtype: DType::Float32, op: OpKind::Add, container_id: ContainerId::DEFAULT });

    let levels = g.parallel_levels(5).unwrap();
    assert_eq!(levels.len(), 2);
    assert_eq!(levels[0].len(), 4); // all independent at level 0
    assert_eq!(levels[1].len(), 1); // fan-in at level 1
}
```

- [ ] **Step 6: Run all graph tests**

Run: `cargo test -p applegpu-core graph`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/graph.rs
git commit -m "feat: add Graph::parallel_levels() for concurrent queue dispatch"
```

### Task 2: Swift queue pool + batch context + MTLEvent infrastructure

**Files:**
- Modify: `swift/Sources/AppleGPUBridge/compute.swift`
- Modify: `swift/Sources/AppleGPUBridge/include/bridge.h`
- Test: `swift/Tests/AppleGPUBridgeTests/` (new test file)

- [ ] **Step 1: Add C header declarations**

Add to `bridge.h`:

```c
// Concurrent queue pool
void* gpu_bridge_get_queue(const void* device, uint32_t index);

// Batch context system
void* gpu_bridge_set_batch_context(uint32_t context_id, void* queue);
void* gpu_bridge_commit_batch_context(uint32_t context_id);
void gpu_bridge_set_active_context(uint32_t context_id);

// MTLEvent synchronization
void* gpu_bridge_create_event(const void* device);
void gpu_bridge_encode_signal_event(void* command_buffer, void* event, uint64_t value);
void gpu_bridge_encode_wait_event(void* command_buffer, void* event, uint64_t value);
void gpu_bridge_destroy_event(void* event);
```

- [ ] **Step 2: Implement queue pool in compute.swift**

Add at the top of `compute.swift`, alongside the existing `sharedCommandQueue`:

```swift
// --- Queue pool for concurrent dispatch ---
private var queuePool: [MTLCommandQueue] = []
private let queuePoolLock = NSLock()
private let maxQueueCount: Int = 4

@_cdecl("gpu_bridge_get_queue")
public func gpuBridgeGetQueue(_ devicePtr: UnsafeRawPointer?, _ index: UInt32) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr else { return nil }
    let gpuDevice = getGPUDevice(from: devicePtr)

    queuePoolLock.lock()
    defer { queuePoolLock.unlock() }

    let idx = Int(index) % maxQueueCount
    // Lazily create queues as needed
    while queuePool.count <= idx {
        guard let q = gpuDevice.device.makeCommandQueue() else { return nil }
        queuePool.append(q)
    }
    return Unmanaged.passUnretained(queuePool[idx]).toOpaque()
}
```

- [ ] **Step 3: Implement batch context system in compute.swift**

```swift
// --- Batch context system ---
private var batchContexts: [UInt32: MTLCommandBuffer] = [:]
private let contextLock = NSLock()
private var currentContextId: UInt32 = 0

@_cdecl("gpu_bridge_set_active_context")
public func gpuBridgeSetActiveContext(_ contextId: UInt32) {
    currentContextId = contextId
}

@_cdecl("gpu_bridge_set_batch_context")
public func gpuBridgeSetBatchContext(_ contextId: UInt32, _ queuePtr: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let queuePtr = queuePtr else { return nil }
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue() as! MTLCommandQueue
    guard let cb = queue.makeCommandBuffer() else { return nil }

    contextLock.lock()
    batchContexts[contextId] = cb
    contextLock.unlock()

    return Unmanaged.passUnretained(cb).toOpaque()
}

@_cdecl("gpu_bridge_commit_batch_context")
public func gpuBridgeCommitBatchContext(_ contextId: UInt32) -> UnsafeMutableRawPointer? {
    contextLock.lock()
    guard let cb = batchContexts.removeValue(forKey: contextId) else {
        contextLock.unlock()
        return nil
    }
    contextLock.unlock()

    cb.commit()
    return Unmanaged.passRetained(cb as AnyObject).toOpaque()
}
```

- [ ] **Step 4: Update _nb functions to use batch context lookup**

In every `_nb` function, replace the existing pattern:

```swift
// OLD:
batchLock.lock()
if let batchCB = activeBatchCommandBuffer {
    commandBuffer = batchCB
    isBatch = true
    batchLock.unlock()
} else {
    batchLock.unlock()
    // ...
}
```

With:

```swift
// NEW:
let ctxId = currentContextId
contextLock.lock()
if ctxId > 0, let batchCB = batchContexts[ctxId] {
    commandBuffer = batchCB
    isBatch = true
    contextLock.unlock()
} else if let batchCB = activeBatchCommandBuffer {
    // Legacy single-batch path (context 0)
    commandBuffer = batchCB
    isBatch = true
    contextLock.unlock()
} else {
    contextLock.unlock()
    // Create standalone CB as before...
}
```

This maintains backward compatibility — context 0 falls through to the existing `activeBatchCommandBuffer`.

- [ ] **Step 5: Implement MTLEvent FFI**

```swift
@_cdecl("gpu_bridge_create_event")
public func gpuBridgeCreateEvent(_ devicePtr: UnsafeRawPointer?) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr else { return nil }
    let gpuDevice = getGPUDevice(from: devicePtr)
    guard let event = gpuDevice.device.makeEvent() else { return nil }
    return Unmanaged.passRetained(event as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_encode_signal_event")
public func gpuBridgeEncodeSignalEvent(_ cbPtr: UnsafeMutableRawPointer?, _ eventPtr: UnsafeMutableRawPointer?, _ value: UInt64) {
    guard let cbPtr = cbPtr, let eventPtr = eventPtr else { return }
    let cb = Unmanaged<AnyObject>.fromOpaque(cbPtr).takeUnretainedValue() as! MTLCommandBuffer
    let event = Unmanaged<AnyObject>.fromOpaque(eventPtr).takeUnretainedValue() as! MTLEvent
    cb.encodeSignalEvent(event, value: value)
}

@_cdecl("gpu_bridge_encode_wait_event")
public func gpuBridgeEncodeWaitEvent(_ cbPtr: UnsafeMutableRawPointer?, _ eventPtr: UnsafeMutableRawPointer?, _ value: UInt64) {
    guard let cbPtr = cbPtr, let eventPtr = eventPtr else { return }
    let cb = Unmanaged<AnyObject>.fromOpaque(cbPtr).takeUnretainedValue() as! MTLCommandBuffer
    let event = Unmanaged<AnyObject>.fromOpaque(eventPtr).takeUnretainedValue() as! MTLEvent
    cb.encodeWaitForEvent(event, value: value)
}

@_cdecl("gpu_bridge_destroy_event")
public func gpuBridgeDestroyEvent(_ eventPtr: UnsafeMutableRawPointer?) {
    guard let eventPtr = eventPtr else { return }
    Unmanaged<AnyObject>.fromOpaque(eventPtr).release()
}
```

- [ ] **Step 6: Build Swift to verify compilation**

Run: `cd swift && swift build`
Expected: BUILD SUCCEEDED

- [ ] **Step 7: Commit**

```bash
git add swift/Sources/AppleGPUBridge/compute.swift swift/Sources/AppleGPUBridge/include/bridge.h
git commit -m "feat: Swift queue pool, batch contexts, and MTLEvent infrastructure"
```

### Task 3: Rust FFI bindings for new Swift functions

**Files:**
- Modify: `crates/core/src/ffi.rs`
- Modify: `crates/core/src/compute.rs`

- [ ] **Step 1: Add extern "C" declarations in ffi.rs**

```rust
extern "C" {
    pub fn gpu_bridge_get_queue(device: *const GPUDeviceHandle, index: u32) -> *mut std::ffi::c_void;
    pub fn gpu_bridge_set_batch_context(context_id: u32, queue: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    pub fn gpu_bridge_commit_batch_context(context_id: u32) -> *mut std::ffi::c_void;
    pub fn gpu_bridge_set_active_context(context_id: u32);
    pub fn gpu_bridge_create_event(device: *const GPUDeviceHandle) -> *mut std::ffi::c_void;
    pub fn gpu_bridge_encode_signal_event(cb: *mut std::ffi::c_void, event: *mut std::ffi::c_void, value: u64);
    pub fn gpu_bridge_encode_wait_event(cb: *mut std::ffi::c_void, event: *mut std::ffi::c_void, value: u64);
    pub fn gpu_bridge_destroy_event(event: *mut std::ffi::c_void);
}
```

- [ ] **Step 2: Add safe wrapper functions in compute.rs**

```rust
pub fn get_queue(device: &Device, index: u32) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_get_queue(device.raw_handle(), index) }
}

pub fn set_batch_context(context_id: u32, queue: *mut std::ffi::c_void) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_set_batch_context(context_id, queue) }
}

pub fn commit_batch_context(context_id: u32) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_commit_batch_context(context_id) }
}

pub fn set_active_context(context_id: u32) {
    unsafe { ffi::gpu_bridge_set_active_context(context_id) }
}

pub fn create_event(device: &Device) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_create_event(device.raw_handle()) }
}

pub fn encode_signal_event(cb: *mut std::ffi::c_void, event: *mut std::ffi::c_void, value: u64) {
    unsafe { ffi::gpu_bridge_encode_signal_event(cb, event, value) }
}

pub fn encode_wait_event(cb: *mut std::ffi::c_void, event: *mut std::ffi::c_void, value: u64) {
    unsafe { ffi::gpu_bridge_encode_wait_event(cb, event, value) }
}

pub fn destroy_event(event: *mut std::ffi::c_void) {
    unsafe { ffi::gpu_bridge_destroy_event(event) }
}
```

- [ ] **Step 3: Build Rust core to verify linkage**

Run: `cargo build -p applegpu-core`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/ffi.rs crates/core/src/compute.rs
git commit -m "feat: Rust FFI bindings for queue pool, batch contexts, and MTLEvent"
```

### Task 4: Restructure eval() for multi-queue dispatch

**Files:**
- Modify: `crates/core/src/lazy.rs`
- Test: `python/tests/test_concurrent_queues.py` (new)

- [ ] **Step 1: Add eval_single_cb fallback method**

In `lazy.rs`, extract the current eval loop into `eval_single_cb`:

```rust
fn eval_single_cb(&mut self, device: &Device, _id: u64, container_id: ContainerId, levels: Vec<Vec<u64>>) -> Result<()> {
    let order: Vec<u64> = levels.into_iter().flat_map(|l| l.into_iter()).collect();
    let queue = crate::compute::get_shared_queue(device);
    let batch_cb = crate::compute::begin_batch(queue);
    let use_batch = !batch_cb.is_null();
    let mut last_cb: Option<*mut std::ffi::c_void> = None;

    let loop_result: Result<()> = (|| {
        for node_id in order {
            if self.is_materialized(node_id) { continue; }
            let node = self.graph.remove_node(node_id).ok_or_else(|| {
                GpuError::GraphError(format!("Node {} not found in graph", node_id))
            })?;
            let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let cb = self.execute_node_nb(device, queue, &node, &out)?;
            if !use_batch { last_cb = Some(cb); }
            let size = node.out_shape.numel() * node.out_dtype.size_bytes();
            self.scheduler.allocate_tensor(container_id, node_id, size)?;
            self.tensors.insert(node_id, out);
        }
        Ok(())
    })();

    if use_batch {
        if loop_result.is_ok() {
            let cb = crate::compute::end_batch();
            if !cb.is_null() { crate::compute::wait_command_buffer(cb); }
        } else {
            crate::compute::abort_batch();
        }
    } else if let Some(cb) = last_cb {
        crate::compute::wait_command_buffer(cb);
    }
    loop_result
}
```

- [ ] **Step 2: Rewrite eval() to use parallel_levels + multi-queue**

Replace the existing `eval()` body with the code from spec Section 1.5 (see spec for full implementation). Key structure:
1. topo_sort → fusion → parallel_levels
2. Linear fast path → eval_single_cb
3. Create event + queues
4. For each level: create one CB per queue, encode ops, signal event
5. Wait all CBs, destroy event

- [ ] **Step 3: Run existing test suite to verify backward compat**

Run: `make test-rust && make test-python`
Expected: All tests PASS (linear graphs hit the fast path)

- [ ] **Step 4: Write Python test for parallel graph execution**

Create `python/tests/test_concurrent_queues.py`:

```python
import applegpu_runtime as gpu

def test_diamond_graph():
    """A -> B, A -> C, B+C -> D. B and C should execute on separate queues."""
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = a + a          # depends on a
    c = a * a          # depends on a, independent of b
    d = b + c          # depends on b and c
    d.eval()
    result = d.to_list()
    expected = [1+1+1*1, 2+2+2*2, 3+3+3*3, 4+4+4*4]
    assert result == expected, f"Got {result}, expected {expected}"

def test_wide_independent_ops():
    """4 independent operations that can all execute in parallel."""
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = a + a
    c = a * a
    d = a - a
    e = gpu.relu(a)
    # Force a join point
    result = b + c + d + e
    result.eval()
    vals = result.to_list()
    # b=2,4,6,8  c=1,4,9,16  d=0,0,0,0  e=1,2,3,4
    expected = [2+1+0+1, 4+4+0+2, 6+9+0+3, 8+16+0+4]
    assert vals == expected
```

- [ ] **Step 5: Run the new test**

Run: `uv run pytest python/tests/test_concurrent_queues.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/lazy.rs python/tests/test_concurrent_queues.py
git commit -m "feat: multi-queue eval with parallel levels and MTLEvent sync"
```

### Task 5: Full test suite + benchmark validation

- [ ] **Step 1: Run complete test suite**

Run: `make test`
Expected: All Rust + Swift + Python tests PASS

- [ ] **Step 2: Run GPT-2 benchmark (if examples exist)**

Run: `uv run python examples/gpt2_generate.py --prompt "Hello world" --max-tokens 5`
Expected: Generates tokens without error. Note timing.

- [ ] **Step 3: Commit and tag Phase I complete**

```bash
git commit --allow-empty -m "milestone: Phase I concurrent queues complete"
```

---

## Chunk 2: Phase II — Fine-Grained Locking

### Task 6: Create ConcurrentRuntime with per-component locks

**Files:**
- Modify: `crates/core/src/lazy.rs`
- Modify: `crates/python/src/metal_backend.rs`

- [ ] **Step 1: Add ConcurrentRuntime struct alongside LazyRuntime**

In `lazy.rs`, add:

```rust
use std::sync::{RwLock, Mutex as StdMutex};

pub struct ConcurrentRuntime {
    pub tensors: RwLock<HashMap<u64, Tensor>>,
    pub graph: StdMutex<Graph>,
    pub scheduler: StdMutex<Scheduler>,
    pub pool: StdMutex<BufferPool>,
}
```

- [ ] **Step 2: Implement core methods on ConcurrentRuntime**

Implement `shape()`, `dtype()`, `is_materialized()`, `is_pending()`, `record_op()`, `insert_tensor()`, `destroy()` — each acquiring only the locks they need per the lock ordering protocol in spec Section 2.2.

- [ ] **Step 3: Implement phased eval on ConcurrentRuntime**

Per spec Section 2.3: Phase 1 (extract plan, lock graph briefly), Phase 2 (acquire buffers, lock pool briefly), Phase 3 (encode ops, read-lock tensors), Phase 4 (GPU execution, no locks), Phase 5 (insert results, write-lock tensors + scheduler).

- [ ] **Step 4: Update MetalBackend to use ConcurrentRuntime**

In `metal_backend.rs`, replace:
```rust
pub struct MetalBackend {
    runtime: Mutex<LazyRuntime>,
}
```
With:
```rust
pub struct MetalBackend {
    runtime: ConcurrentRuntime,
}
```

Update all `Backend` trait methods to call `self.runtime.<method>()` directly instead of `self.runtime.lock().unwrap().<method>()`.

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All PASS

- [ ] **Step 6: Add multi-threaded Python test**

```python
import threading
import applegpu_runtime as gpu

def test_concurrent_shape_during_eval():
    gpu.init_backend()
    a = gpu.tensor([1.0] * 1000, [1000])
    b = a + a  # pending

    results = []
    def query_shape():
        try:
            s = a.shape
            results.append(("ok", s))
        except Exception as e:
            results.append(("err", str(e)))

    threads = [threading.Thread(target=query_shape) for _ in range(10)]
    for t in threads: t.start()
    b.eval()
    for t in threads: t.join()

    # All shape queries should succeed (a is materialized)
    assert all(r[0] == "ok" for r in results)
```

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/lazy.rs crates/python/src/metal_backend.rs python/tests/test_concurrent_locking.py
git commit -m "feat: fine-grained locking with ConcurrentRuntime"
```

---

## Chunk 3: Phase III — Async Eval

### Task 7: GpuFuture and eval_async

**Files:**
- Modify: `crates/python/src/lib.rs`
- Modify: `crates/python/src/backend.rs`
- Modify: `crates/python/src/metal_backend.rs`
- Test: `python/tests/test_async_eval.py` (new)

- [ ] **Step 1: Add GpuFuture PyO3 class**

In `crates/python/src/lib.rs`:

```rust
use std::sync::{Arc, Condvar, atomic::{AtomicBool, Ordering}};

struct FutureHandle {
    ready: AtomicBool,
    result: std::sync::Mutex<Option<Result<(), String>>>,
    condvar: Condvar,
}

#[pyclass]
pub struct GpuFuture {
    handle: Arc<FutureHandle>,
}

#[pymethods]
impl GpuFuture {
    fn is_ready(&self) -> bool {
        self.handle.ready.load(Ordering::Acquire)
    }

    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut guard = self.handle.result.lock().unwrap();
            while guard.is_none() {
                guard = self.handle.condvar.wait(guard).unwrap();
            }
        });
        let guard = self.handle.result.lock().unwrap();
        match guard.as_ref().unwrap() {
            Ok(()) => Ok(()),
            Err(e) => Err(PyRuntimeError::new_err(e.clone())),
        }
    }
}
```

- [ ] **Step 2: Add eval_async to Backend trait**

In `crates/python/src/backend.rs`:

```rust
fn eval_async(&self, id: u64) -> BackendResult<Arc<FutureHandle>>;
```

- [ ] **Step 3: Implement eval_async in MetalBackend**

Spawn a thread that runs `eval()`, then signals the FutureHandle.

- [ ] **Step 4: Add eval_async Python function and auto-wait on read**

- [ ] **Step 5: Write tests**

```python
def test_eval_async_basic():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0], [2])
    b = a + a
    future = gpu.eval_async(b)
    assert isinstance(future, gpu.GpuFuture)
    future.wait()
    assert b.to_list() == [2.0, 4.0]

def test_eval_async_auto_wait():
    gpu.init_backend()
    a = gpu.tensor([3.0, 4.0], [2])
    b = a * a
    future = gpu.eval_async(b)
    # to_list should auto-wait
    assert b.to_list() == [9.0, 16.0]
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest python/tests/test_async_eval.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/python/src/lib.rs crates/python/src/backend.rs crates/python/src/metal_backend.rs python/tests/test_async_eval.py
git commit -m "feat: eval_async with GpuFuture for non-blocking GPU evaluation"
```

### Task 8: Final validation

- [ ] **Step 1: Run complete test suite**

Run: `make test`
Expected: All PASS

- [ ] **Step 2: Commit milestone**

```bash
git commit --allow-empty -m "milestone: Phase 2c+2d concurrency complete"
```
