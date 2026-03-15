# Single Command Buffer (Phase 2b) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode all ops within an eval() into a single MTLCommandBuffer, reducing per-op CB creation overhead from N to 1.

**Architecture:** Add begin_batch/end_batch/abort_batch FFI that stores an active command buffer in Swift. All 30 existing `_nb` dispatch functions become batch-aware: when a batch is active, they encode into the shared CB and return an unretained pointer to it (so Rust null-checks pass). Rust eval() wraps the dispatch loop with begin/end batch calls, with abort_batch for error recovery.

**Tech Stack:** Swift (Metal), C header, Rust (FFI + compute + lazy runtime)

**Spec:** `docs/superpowers/specs/2026-03-15-single-command-buffer-design.md`

**Key design decisions (from reviewer feedback):**
- `_nb` functions return **unretained batch CB pointer** in batch mode (not nil) — all 28 Rust `dispatch_*_nb` wrappers treat null as error, so nil would abort eval
- `begin_batch` returns **unretained** pointer (Rust only null-checks it), `end_batch` returns **retained** pointer (consumed by `wait_command_buffer`)
- `abort_batch` clears batch state without committing on mid-batch errors
- Zero-element ops return batch CB pointer in batch mode (no-op, not error)

---

## Chunk 1: Swift Batch Infrastructure + FFI

### Task 1: Add batch state and begin/end/abort FFI to Swift

**Files:**
- Modify: `swift/Sources/AppleGPUBridge/compute.swift:1-9` (add batch state)
- Modify: `swift/Sources/AppleGPUBridge/compute.swift:1160-1178` (add begin/end/abort after shared queue section)
- Modify: `swift/Sources/AppleGPUBridge/include/bridge.h:181-184` (add batch declarations)

- [ ] **Step 1: Write Swift test for begin_batch / end_batch**

In `swift/Tests/AppleGPUBridgeTests/ComputeTests.swift`, add:

```swift
@Test func testBeginEndBatchBasic() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let queuePtr = Unmanaged.passUnretained(queue).toOpaque()

    // Begin batch
    let batchCB = gpu_bridge_begin_batch(queuePtr)
    #expect(batchCB != nil)

    // Double begin should fail
    let batchCB2 = gpu_bridge_begin_batch(queuePtr)
    #expect(batchCB2 == nil)

    // End batch (commits)
    let cb = gpu_bridge_end_batch()
    #expect(cb != nil)

    // Wait for completion
    gpu_bridge_wait_command_buffer(cb!)

    // End batch with no active batch should return nil
    let cb2 = gpu_bridge_end_batch()
    #expect(cb2 == nil)
}

@Test func testAbortBatch() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let queuePtr = Unmanaged.passUnretained(queue).toOpaque()

    let batchCB = gpu_bridge_begin_batch(queuePtr)
    #expect(batchCB != nil)

    // Abort without committing
    gpu_bridge_abort_batch()

    // Should be able to begin a new batch after abort
    let batchCB2 = gpu_bridge_begin_batch(queuePtr)
    #expect(batchCB2 != nil)

    let cb = gpu_bridge_end_batch()
    #expect(cb != nil)
    gpu_bridge_wait_command_buffer(cb!)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd swift && swift test --filter testBeginEndBatchBasic`
Expected: FAIL — `gpu_bridge_begin_batch` not found

- [ ] **Step 3: Add batch state variables to compute.swift**

At the top of `compute.swift` (after line 8, the `sharedQueueLock` declaration), add:

```swift
/// Active batch command buffer for single-CB encoding mode.
/// When non-nil, _nb dispatch functions encode into this CB instead of creating their own.
private var activeBatchCommandBuffer: MTLCommandBuffer?
private let batchLock = NSLock()
```

- [ ] **Step 4: Add begin_batch/end_batch/abort_batch C ABI exports**

After the `gpuBridgeWaitCommandBuffer` function (~line 1178), add:

```swift
@_cdecl("gpu_bridge_begin_batch")
public func gpuBridgeBeginBatch(_ queuePtr: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let queuePtr = queuePtr else { return nil }
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    batchLock.lock()
    defer { batchLock.unlock() }
    guard activeBatchCommandBuffer == nil else { return nil }
    guard let cb = queue.makeCommandBuffer() else { return nil }
    activeBatchCommandBuffer = cb
    // passUnretained: Rust only checks non-null. CB is owned by activeBatchCommandBuffer.
    return Unmanaged.passUnretained(cb as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_end_batch")
public func gpuBridgeEndBatch() -> UnsafeMutableRawPointer? {
    batchLock.lock()
    defer { batchLock.unlock() }
    guard let cb = activeBatchCommandBuffer else { return nil }
    cb.commit()
    activeBatchCommandBuffer = nil
    // passRetained: ownership transfers to Rust. wait_command_buffer calls takeRetainedValue.
    return Unmanaged.passRetained(cb as AnyObject).toOpaque()
}

@_cdecl("gpu_bridge_abort_batch")
public func gpuBridgeAbortBatch() {
    batchLock.lock()
    defer { batchLock.unlock() }
    activeBatchCommandBuffer = nil
}
```

- [ ] **Step 5: Add C header declarations**

In `swift/Sources/AppleGPUBridge/include/bridge.h`, after the `gpu_bridge_wait_command_buffer` declaration, add:

```c
// Batch encoding: encode all ops into a single command buffer per eval.
// begin_batch creates a command buffer; all subsequent _nb calls encode into it.
// end_batch commits and returns the CB handle for waiting.
// abort_batch discards the batch on mid-encode error.
void* gpu_bridge_begin_batch(void* queue);
void* gpu_bridge_end_batch(void);
void gpu_bridge_abort_batch(void);
```

- [ ] **Step 6: Run Swift tests to verify begin/end/abort batch works**

Run: `cd swift && swift test --filter testBeginEndBatchBasic && cd swift && swift test --filter testAbortBatch`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add swift/Sources/AppleGPUBridge/compute.swift swift/Sources/AppleGPUBridge/include/bridge.h swift/Tests/
git commit -m "feat(swift): add begin_batch/end_batch/abort_batch for single command buffer encoding"
```

### Task 2: Add Rust FFI declarations and safe wrappers

**Files:**
- Modify: `crates/core/src/ffi.rs:175-181` (add extern declarations in non-blocking section)
- Modify: `crates/core/src/compute.rs:4048-4055` (add safe wrappers after wait_command_buffer)

- [ ] **Step 1: Add extern declarations to ffi.rs**

In the `extern "C"` block, after `gpu_bridge_wait_command_buffer` declaration (~line 181), add:

```rust
    pub fn gpu_bridge_begin_batch(queue: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    pub fn gpu_bridge_end_batch() -> *mut std::ffi::c_void;
    pub fn gpu_bridge_abort_batch();
```

- [ ] **Step 2: Add safe wrappers to compute.rs**

After the `wait_command_buffer` function (~line 4055), add:

```rust
/// Begin batch encoding: creates a single command buffer for all subsequent _nb dispatches.
/// Returns a non-null handle on success (unretained — only for null-checking).
/// All _nb calls will encode into this CB until end_batch() or abort_batch() is called.
pub fn begin_batch(queue: *mut std::ffi::c_void) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_begin_batch(queue) }
}

/// End batch encoding: commits the batch command buffer and returns its handle for waiting.
/// The returned handle is retained — pass to wait_command_buffer() which consumes it.
pub fn end_batch() -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_end_batch() }
}

/// Abort batch encoding: discards the batch command buffer without committing.
/// Call this when an error occurs mid-batch to clean up batch state.
pub fn abort_batch() {
    unsafe { ffi::gpu_bridge_abort_batch() }
}
```

- [ ] **Step 3: Verify Rust compiles**

Run: `cargo build -p applegpu-core 2>&1 | tail -5`
Expected: successful build (linker finds the new symbols in Swift static lib)

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/ffi.rs crates/core/src/compute.rs
git commit -m "feat(ffi): add begin_batch/end_batch/abort_batch Rust FFI declarations and wrappers"
```

## Chunk 2: Make _nb Functions Batch-Aware

### Task 3: Modify _nb dispatch functions in Swift (functions 1-15)

**Files:**
- Modify: `swift/Sources/AppleGPUBridge/compute.swift` (first 15 `_nb` C ABI exports)

The same pattern applies to all `_nb` functions. The key change: in batch mode, return an **unretained** pointer to the batch CB (not nil), so Rust dispatch wrappers' null-checks pass.

**Pattern — Before (in each _nb function):**
```swift
let count = Int(elementCount)
if count == 0 { return nil }

guard let commandBuffer = queue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
// ... encode ...
encoder.endEncoding()
commandBuffer.commit()
return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
```

**Pattern — After:**
```swift
let count = Int(elementCount)
if count == 0 {
    // In batch mode return non-null (batch CB pointer); in non-batch return nil
    batchLock.lock()
    let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
    batchLock.unlock()
    return ptr
}

let commandBuffer: MTLCommandBuffer
let isBatch: Bool
batchLock.lock()
if let batchCB = activeBatchCommandBuffer {
    commandBuffer = batchCB
    isBatch = true
    batchLock.unlock()
} else {
    batchLock.unlock()
    guard let cb = queue.makeCommandBuffer() else { return nil }
    commandBuffer = cb
    isBatch = false
}
guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
// ... encode (unchanged) ...
encoder.endEncoding()
if isBatch {
    return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
} else {
    commandBuffer.commit()
    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}
```

- [ ] **Step 1: Write Swift test for batch-aware multi-op dispatch**

In `swift/Tests/AppleGPUBridgeTests/ComputeTests.swift`, add:

```swift
@Test func testBatchMultiOpCorrectness() throws {
    let gpuDevice = gpu_bridge_create_device()!

    let addSource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void add_f32(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        uint idx [[thread_position_in_grid]]) {
        if (idx < count) { out[idx] = a[idx] + b[idx]; }
    }
    """
    let mulSource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void mul_f32(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        uint idx [[thread_position_in_grid]]) {
        if (idx < count) { out[idx] = a[idx] * b[idx]; }
    }
    """

    let addCompute = gpu_bridge_create_compute(gpuDevice, addSource, "add_f32")!
    let mulCompute = gpu_bridge_create_compute(gpuDevice, mulSource, "mul_f32")!

    var aData: [Float] = [1, 2, 3, 4]
    var bData: [Float] = [10, 20, 30, 40]
    let bufA = gpu_bridge_create_buffer_with_data(gpuDevice, &aData, UInt64(aData.count * 4))!
    let bufB = gpu_bridge_create_buffer_with_data(gpuDevice, &bData, UInt64(bData.count * 4))!
    let bufC = gpu_bridge_create_buffer(gpuDevice, UInt64(4 * 4))!
    let bufD = gpu_bridge_create_buffer(gpuDevice, UInt64(4 * 4))!

    let queue = gpu_bridge_get_shared_queue(gpuDevice)!
    let batchCB = gpu_bridge_begin_batch(queue)
    #expect(batchCB != nil)

    // In batch mode, _nb returns non-null (unretained batch CB pointer)
    let cb1 = gpu_bridge_compute_elementwise_nb(addCompute, queue, bufA, bufB, bufC, 4)
    #expect(cb1 != nil)  // batch mode returns unretained batch CB pointer

    let cb2 = gpu_bridge_compute_elementwise_nb(mulCompute, queue, bufC, bufB, bufD, 4)
    #expect(cb2 != nil)

    let finalCB = gpu_bridge_end_batch()
    #expect(finalCB != nil)
    gpu_bridge_wait_command_buffer(finalCB!)

    let ptr = gpu_bridge_buffer_contents(bufD)!.assumingMemoryBound(to: Float.self)
    #expect(ptr[0] == 110.0)  // (1+10)*10
    #expect(ptr[1] == 440.0)  // (2+20)*20
    #expect(ptr[2] == 990.0)  // (3+30)*30
    #expect(ptr[3] == 1760.0) // (4+40)*40

    gpu_bridge_destroy_buffer(bufA)
    gpu_bridge_destroy_buffer(bufB)
    gpu_bridge_destroy_buffer(bufC)
    gpu_bridge_destroy_buffer(bufD)
    gpu_bridge_destroy_compute(addCompute)
    gpu_bridge_destroy_compute(mulCompute)
    gpu_bridge_destroy_device(gpuDevice)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd swift && swift test --filter testBatchMultiOpCorrectness`
Expected: FAIL — _nb functions still create their own CB and commit (batch CB is empty)

- [ ] **Step 3: Modify first 15 _nb C ABI functions to be batch-aware**

Apply the pattern change to each of these functions in `compute.swift`:

1. `gpuBridgeComputeGatherNB` (line ~1037)
2. `gpuBridgeComputeElementwiseNB` (line ~1180)
3. `gpuBridgeComputeUnaryNB` (line ~1228)
4. `gpuBridgeComputeMatmulNB` (line ~1272) — note: delegates to batched variant
5. `gpuBridgeComputeMatmulBatchedNB` (line ~1286)
6. `gpuBridgeComputeSoftmaxCausalNB` (line ~1342)
7. `gpuBridgeComputeSoftmaxNB` (line ~1388)
8. `gpuBridgeComputeTransposeNB` (line ~1434)
9. `gpuBridgeComputeTransposeBatchedNB` (line ~1478)
10. `gpuBridgeComputeScalarMulNB` (line ~1524)
11. `gpuBridgeComputeFusedNB` (line ~1571)
12. `gpuBridgeComputeLayerNormNB` (line ~1624)
13. `gpuBridgeComputeSoftmaxBackwardNB` (line ~1680)
14. `gpuBridgeComputeLayerNormBackwardNB` (line ~1730)
15. `gpuBridgeComputeBlitCopyNB` (line ~1786) — **uses `makeBlitCommandEncoder()`**, not compute encoder

**Special case: blit_copy_nb (#15)** uses `MTLBlitCommandEncoder` instead of `MTLComputeCommandEncoder`. Same batch-aware pattern but call `commandBuffer.makeBlitCommandEncoder()` instead.

- [ ] **Step 4: Run Swift test to verify batch multi-op works**

Run: `cd swift && swift test --filter testBatchMultiOpCorrectness`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add swift/Sources/AppleGPUBridge/compute.swift swift/Tests/
git commit -m "feat(swift): make first 15 _nb functions batch-aware (elementwise through blit)"
```

### Task 4: Modify remaining 15 _nb dispatch functions in Swift

**Files:**
- Modify: `swift/Sources/AppleGPUBridge/compute.swift` (remaining 15 `_nb` C ABI exports)

- [ ] **Step 1: Modify remaining 15 _nb C ABI functions to be batch-aware**

Apply the same pattern to:

16. `gpuBridgeComputeEmbeddingNB` (line ~1808)
17. `gpuBridgeComputeSliceDim0NB` (line ~1858)
18. `gpuBridgeComputeSliceDim1NB` (line ~1900)
19. `gpuBridgeComputeConcatDim0NB` (line ~1945)
20. `gpuBridgeComputeConcatDim1NB` (line ~1990)
21. `gpuBridgeComputeAddBiasNB` (line ~2037)
22. `gpuBridgeComputeBinaryNdNB` (line ~2142)
23. `gpuBridgeComputeUnaryNdNB` (line ~2255)
24. `gpuBridgeComputePowNdNB` (line ~2366)
25. `gpuBridgeComputeClampNdNB` (line ~2483)
26. `gpuBridgeComputeWhereNdNB` (line ~2611)
27. `gpuBridgeComputeMaskedFillNdNB` (line ~2743)
28. `gpuBridgeComputeTriangularNB` (line ~2855)
29. `gpuBridgeComputeFusedNdNB` (line ~2972)
30. `gpuBridgeCompute3dNB` (line ~3116)

- [ ] **Step 2: Run full Swift test suite**

Run: `cd swift && swift test`
Expected: All tests PASS (existing non-batch tests work unchanged since no batch is active)

- [ ] **Step 3: Commit**

```bash
git add swift/Sources/AppleGPUBridge/compute.swift
git commit -m "feat(swift): make remaining 15 _nb functions batch-aware (embedding through 3d)"
```

## Chunk 3: Rust Integration + eval() Update

### Task 5: Update Rust eval() to use begin/end batch

**Files:**
- Modify: `crates/core/src/lazy.rs:103-149` (eval method)

- [ ] **Step 1: Write Rust integration test for single-CB eval**

Add to `crates/core/tests/batched_ops_integration.rs`:

```rust
#[test]
fn test_single_cb_eval_chain() {
    // Chain: a + b -> c, c * b -> d (5 elements)
    // Verifies the single-CB path produces correct results.
    let device = applegpu_core::device::Device::new().expect("Need Metal GPU");
    let runtime = applegpu_core::backend::Runtime::new();
    let mut rt = runtime.lock().unwrap();

    let a_id = rt.create_from_f32(&device, &[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let b_id = rt.create_from_f32(&device, &[10.0, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();

    let c_id = rt.record_binary_op(a_id, b_id, "add");
    let d_id = rt.record_binary_op(c_id, b_id, "mul");

    rt.eval(&device, d_id).unwrap();

    let result = rt.read_f32(&device, d_id).unwrap();
    // c = [11, 22, 33, 44, 55], d = c * b = [110, 440, 990, 1760, 2750]
    assert_eq!(result, vec![110.0, 440.0, 990.0, 1760.0, 2750.0]);
}
```

Note: Adapt the test to match the actual Runtime API in `crates/core/tests/batched_ops_integration.rs` — check existing tests in that file for the exact method names and patterns.

- [ ] **Step 2: Run test to verify it passes with current code (baseline)**

Run: `cargo test -p applegpu-core test_single_cb_eval_chain`
Expected: PASS (works with Phase 2a too — same correctness check)

- [ ] **Step 3: Update eval() to use begin_batch / end_batch / abort_batch**

In `crates/core/src/lazy.rs`, modify the `eval` method. Replace lines 118-146:

**Before:**
```rust
        let queue = crate::compute::get_shared_queue(device);
        let mut last_cb: Option<*mut std::ffi::c_void> = None;

        for node_id in order {
            if self.is_materialized(node_id) {
                continue; // already evaluated (shared subexpression)
            }

            let node = self.graph.remove_node(node_id).ok_or_else(|| {
                GpuError::GraphError(format!("Node {} not found in graph", node_id))
            })?;

            let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);

            let cb = self.execute_node_nb(device, queue, &node, &out)?;
            last_cb = Some(cb);

            let size = node.out_shape.numel() * node.out_dtype.size_bytes();
            self.scheduler.allocate_tensor(container_id, node_id, size)?;
            self.tensors.insert(node_id, out);
        }

        // Wait only on the last command buffer — Metal in-order queue guarantees
        // all prior submissions are complete when this one finishes.
        if let Some(cb) = last_cb {
            crate::compute::wait_command_buffer(cb);
        }
```

**After:**
```rust
        let queue = crate::compute::get_shared_queue(device);
        let batch_cb = crate::compute::begin_batch(queue);
        let use_batch = !batch_cb.is_null();
        let mut last_cb: Option<*mut std::ffi::c_void> = None;

        let loop_result: Result<()> = (|| {
            for node_id in order {
                if self.is_materialized(node_id) {
                    continue;
                }

                let node = self.graph.remove_node(node_id).ok_or_else(|| {
                    GpuError::GraphError(format!("Node {} not found in graph", node_id))
                })?;

                let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();
                let out_buf = self.pool.acquire(device, out_size)?;
                let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);

                let cb = self.execute_node_nb(device, queue, &node, &out)?;
                if !use_batch {
                    last_cb = Some(cb);
                }

                let size = node.out_shape.numel() * node.out_dtype.size_bytes();
                self.scheduler.allocate_tensor(container_id, node_id, size)?;
                self.tensors.insert(node_id, out);
            }
            Ok(())
        })();

        // Finalize: commit or abort the batch, then wait
        if use_batch {
            if loop_result.is_ok() {
                let cb = crate::compute::end_batch();
                if !cb.is_null() {
                    crate::compute::wait_command_buffer(cb);
                }
            } else {
                crate::compute::abort_batch();
            }
        } else if let Some(cb) = last_cb {
            crate::compute::wait_command_buffer(cb);
        }

        loop_result
```

- [ ] **Step 4: Run the integration test**

Run: `cargo test -p applegpu-core test_single_cb_eval_chain`
Expected: PASS

- [ ] **Step 5: Run full Rust test suite**

Run: `cargo test -p applegpu-core`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/lazy.rs crates/core/tests/batched_ops_integration.rs
git commit -m "feat(eval): use single command buffer encoding via begin/end batch"
```

### Task 6: Handle blit_copy_nb in lazy.rs reshape path

**Files:**
- Modify: `crates/core/src/lazy.rs` (reshape eval path that calls `ffi::gpu_bridge_blit_copy_nb` directly)

The reshape path (~line 597) calls `ffi::gpu_bridge_blit_copy_nb` directly (not through a dispatch wrapper) and has its own inline null-check. This path also needs to work with batch mode.

- [ ] **Step 1: Find and update the blit_copy_nb direct call**

Search for `gpu_bridge_blit_copy_nb` in `lazy.rs`. The current pattern:

```rust
let cb = unsafe { ffi::gpu_bridge_blit_copy_nb(...) };
if cb.is_null() { return Err(...); }
```

In batch mode, the blit function now returns the unretained batch CB pointer (non-null), so this null-check passes correctly. No change needed unless the function has additional logic around the returned CB. Verify and document.

- [ ] **Step 2: Run tests to verify reshape still works**

Run: `cargo test -p applegpu-core reshape`
Expected: PASS

- [ ] **Step 3: Commit if any changes were needed**

### Task 7: Run full test suite across all layers

**Files:** None (verification only)

- [ ] **Step 1: Run Rust tests**

Run: `cargo test -p applegpu-core`
Expected: All PASS

- [ ] **Step 2: Run Swift tests**

Run: `cd swift && swift test`
Expected: All PASS

- [ ] **Step 3: Run Python tests**

Run: `uv run pytest -v`
Expected: All PASS

- [ ] **Step 4: Commit any test fixes if needed**

### Task 8: Update backlog and spec status

**Files:**
- Modify: `docs/BACKLOG.md:58-60` (mark Phase 2b items as done)
- Modify: `docs/superpowers/specs/2026-03-15-single-command-buffer-design.md:2` (mark as Shipped)

- [ ] **Step 1: Mark Phase 2b items as complete in BACKLOG.md**

Change:
```markdown
**Phase 2b: Single command buffer** _(future)_
- [ ] **Encode all ops into one MTLCommandBuffer** — reduce CB creation overhead
- [ ] **begin_batch/end_batch FFI** — new Swift/Rust API for batch encoding
```

To:
```markdown
**Phase 2b: Single command buffer** _(DONE)_
- [x] **Encode all ops into one MTLCommandBuffer** — reduce CB creation overhead
- [x] **begin_batch/end_batch/abort_batch FFI** — new Swift/Rust API for batch encoding
- [x] **Spec:** `docs/superpowers/specs/2026-03-15-single-command-buffer-design.md`
```

- [ ] **Step 2: Update spec status**

Change `**Status:** Approved (v2 — incorporates reviewer feedback)` to `**Status:** Shipped`

- [ ] **Step 3: Commit**

```bash
git add docs/BACKLOG.md docs/superpowers/specs/2026-03-15-single-command-buffer-design.md
git commit -m "docs: mark Phase 2b (single command buffer) as done"
```
