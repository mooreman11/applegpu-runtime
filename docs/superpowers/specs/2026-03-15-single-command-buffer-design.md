# Single Command Buffer (Phase 2b)

**Date:** 2026-03-15
**Status:** Shipped
**Scope:** Encode all ops within an eval into a single MTLCommandBuffer, reducing per-op CB creation overhead.
**Depends on:** Phase 2a (command buffer batching) — already shipped.

## Problem

Phase 2a eliminated per-op `waitUntilCompleted()` blocking, but each `_nb` dispatch still creates its own `MTLCommandBuffer` from the shared queue. For a 50-op eval graph, that's 50 CB allocations (each involves Metal driver allocation + scheduling overhead). Profiling shows CB creation is ~2-5us per call — significant at scale.

## Goal

Encode all ops in a single `eval()` call into ONE `MTLCommandBuffer` with one `MTLComputeCommandEncoder` per op. Reduce CB creation from N to 1 per eval.

## Approach: Batch Mode via Shared State

Add `begin_batch` / `end_batch` FFI calls. Swift stores the active command buffer in a lock-protected module variable. Existing `_nb` functions detect an active batch and encode into it (without creating a new CB or committing). When no batch is active, `_nb` functions behave exactly as before (full backward compatibility).

### Why this approach over alternatives

- **vs. new `_be` function set:** Would require ~25 new FFI functions (one per dispatch type) across Swift, C header, and Rust. Massive duplication.
- **vs. refactoring `_nb` signatures:** Breaking change to every call site. High risk for a perf optimization.
- **Batch mode:** 2 new FFI functions total. Zero changes to existing dispatch signatures. Fully backward compatible.

### Implicit state tradeoff

The active batch CB is module-level shared state behind `NSLock`. This is safe because:
1. The Rust `eval()` loop holds `Mutex<LazyRuntime>` — only one eval runs at a time
2. `begin_batch` / `end_batch` bracket the loop — well-scoped lifecycle
3. Metal command buffer encoding is not thread-safe anyway — single-threaded encoding is the norm

## Swift Changes

### New module-level state (compute.swift)

```swift
private var activeBatchCommandBuffer: MTLCommandBuffer?
private let batchLock = NSLock()
```

### `gpu_bridge_begin_batch(queue) -> void*`

Creates a command buffer from the queue and stores it as the active batch CB. Returns the CB handle as an **unretained** pointer (Rust uses it only for null-check; does not own it). If a batch is already active, returns NULL (error).

```swift
@_cdecl("gpu_bridge_begin_batch")
public func gpuBridgeBeginBatch(_ queuePtr: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let queuePtr = queuePtr else { return nil }
    let queue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    batchLock.lock()
    defer { batchLock.unlock() }
    guard activeBatchCommandBuffer == nil else { return nil } // already in batch
    guard let cb = queue.makeCommandBuffer() else { return nil }
    activeBatchCommandBuffer = cb
    // passUnretained: Rust only checks non-null. The CB is owned by activeBatchCommandBuffer.
    return Unmanaged.passUnretained(cb as AnyObject).toOpaque()
}
```

### `gpu_bridge_end_batch() -> void*`

Commits the active batch CB, clears the state, returns the CB handle with a **retained** reference for `wait_command_buffer` to consume.

```swift
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
```

### `gpu_bridge_abort_batch()`

Clears the active batch CB without committing. Called by Rust when an encode fails mid-batch.

```swift
@_cdecl("gpu_bridge_abort_batch")
public func gpuBridgeAbortBatch() {
    batchLock.lock()
    defer { batchLock.unlock() }
    activeBatchCommandBuffer = nil
}
```

### Modified `_nb` dispatch functions

Each `_nb` function gains a batch-aware path. The pattern change in every `_nb` function:

**Before:**
```swift
guard let commandBuffer = queue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
// ... encode ...
encoder.endEncoding()
commandBuffer.commit()
return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
```

**After:**
```swift
let commandBuffer: MTLCommandBuffer
let isBatch: Bool
batchLock.lock()
if let batchCB = activeBatchCommandBuffer {
    commandBuffer = batchCB
    isBatch = true
} else {
    guard let cb = queue.makeCommandBuffer() else {
        batchLock.unlock()
        return nil
    }
    commandBuffer = cb
    isBatch = false
}
batchLock.unlock()

guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
// ... encode (unchanged) ...
encoder.endEncoding()

if isBatch {
    // In batch mode: don't commit. Return unretained batch CB pointer so Rust
    // null-checks pass (all dispatch_*_nb wrappers treat null as error).
    return Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
} else {
    commandBuffer.commit()
    return Unmanaged.passRetained(commandBuffer as AnyObject).toOpaque()
}
```

**Key design decision (reviewer fix):** In batch mode, `_nb` functions return an **unretained** pointer to the batch CB (not nil). This is critical because all 28 Rust `dispatch_*_nb` wrappers in `compute.rs` treat a null return as `Err(GpuError::ComputeFailed(...))`. Returning the batch CB pointer (unretained, so no refcount overhead) satisfies the null check without requiring changes to any Rust dispatch wrapper.

**Zero-element ops:** Functions with `if count == 0 { return nil }` should instead return the batch CB pointer in batch mode:
```swift
if count == 0 {
    batchLock.lock()
    let ptr = activeBatchCommandBuffer.map { Unmanaged.passUnretained($0 as AnyObject).toOpaque() }
    batchLock.unlock()
    return ptr  // non-null in batch mode, nil in non-batch mode
}
```

### blit_copy_nb

The `gpu_bridge_blit_copy_nb` function uses a blit encoder, not a compute encoder. The same pattern applies — check for active batch CB, use blit encoder on it, skip commit in batch mode.

## C Header Changes

Add to `bridge.h`:

```c
// Batch encoding: encode all ops into a single command buffer per eval.
// begin_batch creates a command buffer; all subsequent _nb calls encode into it.
// end_batch commits and returns the CB handle for waiting.
// abort_batch discards the batch on mid-encode error.
void* gpu_bridge_begin_batch(void* queue);
void* gpu_bridge_end_batch(void);
void gpu_bridge_abort_batch(void);
```

## Rust FFI Changes

Add to `ffi.rs`:

```rust
pub fn gpu_bridge_begin_batch(queue: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
pub fn gpu_bridge_end_batch() -> *mut std::ffi::c_void;
pub fn gpu_bridge_abort_batch();
```

Add safe wrappers in `compute.rs`:

```rust
pub fn begin_batch(queue: *mut std::ffi::c_void) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_begin_batch(queue) }
}

pub fn end_batch() -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_end_batch() }
}

pub fn abort_batch() {
    unsafe { ffi::gpu_bridge_abort_batch() }
}
```

## Rust eval() Changes (lazy.rs)

```rust
pub fn eval(&mut self, device: &Device, id: u64) -> Result<()> {
    // ... topo sort, fusion (unchanged) ...

    let queue = crate::compute::get_shared_queue(device);
    let batch_cb = crate::compute::begin_batch(queue);
    let use_batch = !batch_cb.is_null();

    let eval_result = (|| -> Result<()> {
        for node_id in order {
            if self.is_materialized(node_id) { continue; }
            // ... remove node, acquire output buffer (unchanged) ...
            let cb = self.execute_node_nb(device, queue, &node, &out)?;
            // In batch mode: cb is the unretained batch CB pointer (non-null).
            // In non-batch mode: cb is the per-op CB (Phase 2a behavior).
            if !use_batch {
                last_cb = Some(cb);
            }
            // ... insert tensor (unchanged) ...
        }
        Ok(())
    })();

    // Finalize: commit or abort the batch
    if use_batch {
        if eval_result.is_ok() {
            let cb = crate::compute::end_batch();
            if !cb.is_null() {
                crate::compute::wait_command_buffer(cb);
            }
        } else {
            crate::compute::abort_batch(); // discard on error
        }
    } else if let Some(cb) = last_cb {
        crate::compute::wait_command_buffer(cb);
    }

    eval_result
}
```

**Key changes from Phase 2a:**
- `begin_batch` before the loop creates the shared CB
- `execute_node_nb` returns non-null in both modes (batch CB pointer or per-op CB)
- `end_batch` after the loop commits and returns the CB for waiting
- On error mid-loop, `abort_batch` clears the batch state without committing
- Graceful fallback: if `begin_batch` returns null, falls back to Phase 2a behavior

## Memory Ordering

Unchanged from Phase 2a. Within a single command buffer, Metal executes compute encoders in submission order. Buffer writes from encoder N are visible to encoder N+1 within the same command buffer. This is strictly stronger than the Phase 2a guarantee (which relied on queue ordering across command buffers).

## Error Handling

- `begin_batch` returns NULL if queue is invalid or a batch is already active → **graceful fallback** to Phase 2a (per-op CB mode), not a hard error
- Individual `_nb` encodes: if `makeComputeCommandEncoder()` fails, returns NULL → Rust treats as dispatch error → Rust calls `abort_batch()` to clean up batch state
- `abort_batch` clears the batch CB without committing → no GPU work submitted for the failed batch
- `end_batch` returns NULL only if no batch was active (programming error) → should not happen with correct begin/end bracketing
- GPU errors: detected at `waitUntilCompleted()` after `end_batch` — same as Phase 2a
- Zero-element ops: in batch mode, return the batch CB pointer (no encoding needed, not an error); in non-batch mode, return nil as before

## Testing Strategy

### Swift tests (~2 tests)
1. `test_begin_end_batch_basic` — begin batch, encode one op, end batch, wait, verify result
2. `test_batch_multi_op` — begin batch, encode 3 ops (add → mul → relu), end batch, verify final result

### Rust integration tests (~3 tests)
1. `test_single_cb_eval_correctness` — 5-op chain, verify result matches Phase 2a behavior
2. `test_single_cb_eval_with_fusion` — fused chain works in single-CB mode
3. `test_single_cb_backward_compat` — single op eval still works (begin_batch with 1 op)

### Python tests (~1 test)
1. `test_single_cb_transparent` — complex expression, verify result unchanged from before

### Existing tests
All existing tests pass unchanged. The `_nb` functions behave identically when no batch is active.

## Rollout

1. Implement Swift batch state + begin/end FFI
2. Modify all `_nb` functions to be batch-aware
3. Add Rust FFI + safe wrappers
4. Update `eval()` to use begin/end batch
5. Run full test suite — all existing tests must pass
6. Add new tests specific to single-CB behavior
