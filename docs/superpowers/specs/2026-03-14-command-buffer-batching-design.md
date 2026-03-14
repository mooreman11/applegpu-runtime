# Command Buffer Batching

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Remove per-op GPU blocking, batch command buffers per eval, wait once at end.

## Overview

Currently every GPU dispatch in `eval()` creates a command buffer, commits it, and blocks with `waitUntilCompleted()`. A 10-op chain = 10 GPU round-trips. This change removes the per-op blocking and waits only once at the end of eval, eliminating N-1 GPU stalls.

## Approach: Shared Queue + Deferred Wait

### Phase 1 (this spec): No-wait batching
- Create a single shared `MTLCommandQueue` per device (not per pipeline)
- Each dispatch creates its own command buffer from the shared queue, commits it, but does NOT call `waitUntilCompleted()`
- Metal guarantees that command buffers submitted to the same queue execute in order
- At the end of eval, wait on the last command buffer only
- This captures the main win (GPU stays busy between ops) with minimal code change

### Phase 2 (future): Single command buffer
- Encode all ops into one command buffer with one encoder per op
- Reduces command buffer creation overhead
- Requires new FFI surface (begin_batch/end_batch)
- Only worth pursuing if profiling shows command buffer overhead matters

## Swift Changes

### Shared command queue

Add a device-level shared queue to `bridge.swift`:

```swift
private var sharedCommandQueue: MTLCommandQueue?

@_cdecl("gpu_bridge_get_shared_queue")
public func gpuBridgeGetSharedQueue(_ deviceHandle: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<GPUDevice>.fromOpaque(deviceHandle).takeUnretainedValue()
    if sharedCommandQueue == nil {
        sharedCommandQueue = device.device.makeCommandQueue()
    }
    return sharedCommandQueue.map { Unmanaged.passUnretained($0).toOpaque() }
}
```

### Non-blocking dispatch variants

Add new dispatch functions that take a queue handle, create a command buffer, encode, commit, and return the command buffer handle (for later waiting):

```swift
@_cdecl("gpu_bridge_compute_elementwise_nb")
public func gpuBridgeComputeElementwiseNonBlocking(
    _ pipelineHandle: UnsafeMutableRawPointer,
    _ queueHandle: UnsafeMutableRawPointer,
    _ bufA: UnsafeMutableRawPointer,
    _ bufB: UnsafeMutableRawPointer,
    _ bufOut: UnsafeMutableRawPointer,
    _ count: UInt32
) -> UnsafeMutableRawPointer? {
    // Create command buffer from shared queue (not pipeline's queue)
    // Encode compute command
    // Commit (but do NOT waitUntilCompleted)
    // Return command buffer handle for later waiting
}
```

### Wait function

```swift
@_cdecl("gpu_bridge_wait_command_buffer")
public func gpuBridgeWaitCommandBuffer(_ cbHandle: UnsafeMutableRawPointer) {
    let cb = Unmanaged<MTLCommandBuffer>.fromOpaque(cbHandle).takeUnretainedValue()
    cb.waitUntilCompleted()
}
```

## Rust Changes

### FFI declarations

Add to `ffi.rs`:
```rust
pub fn gpu_bridge_get_shared_queue(device: *mut GPUDeviceHandle) -> *mut c_void;
pub fn gpu_bridge_compute_elementwise_nb(pipeline, queue, a, b, out, count) -> *mut c_void;
// ... similar for unary, matmul, softmax, etc.
pub fn gpu_bridge_wait_command_buffer(cb: *mut c_void);
```

### KernelRegistry

Add non-blocking dispatch methods that return a command buffer handle:
```rust
pub fn dispatch_binary_nb(&self, device, queue, name, a, b, out, count, dtype) -> Result<*mut c_void>
```

Add wait method:
```rust
pub fn wait_command_buffer(cb: *mut c_void)
```

### LazyRuntime::eval

Change eval loop to collect command buffer handles and wait only on the last one:

```rust
pub fn eval(&mut self, device: &Device, id: u64) -> Result<()> {
    // ... topo sort, fusion ...
    let queue = get_shared_queue(device);
    let mut last_cb: Option<*mut c_void> = None;

    for node_id in order {
        // ... execute_node_nb returns command buffer handle ...
        last_cb = Some(self.execute_node_nb(device, queue, &node)?);
        // ... insert tensor ...
    }

    // Wait only once at the end
    if let Some(cb) = last_cb {
        wait_command_buffer(cb);
    }
    Ok(())
}
```

### Existing per-op dispatch unchanged

The blocking dispatch methods stay for single-op use cases and backward compat. The non-blocking variants are used only within eval's batched loop.

## Memory Ordering

Metal guarantees that command buffers submitted to the same queue execute in submission order. Buffer writes from command buffer N are visible to command buffer N+1 on the same queue. This means intermediate tensor data flows correctly through the chain without explicit barriers.

## Python Changes

None. `eval()`, `to_list()`, `to_numpy()`, `to_torch()` all work unchanged — they call `eval()` which now batches internally.

## Error Handling

- If a non-blocking dispatch fails to create a command buffer, return error immediately (before waiting)
- If the GPU reports an error on commit, it's detected at `waitUntilCompleted()` — check `commandBuffer.error` and propagate
- Mid-chain failures: all preceding command buffers have already committed. The wait will still complete for the last one. Error propagation is the same as today.

## Testing Strategy

### Rust integration tests (~4 tests)
1. `test_batched_eval_correctness` — chain of 5 ops, verify result matches unbatched
2. `test_batched_eval_matmul_chain` — matmul + add + relu chain, verify correctness
3. `test_single_op_still_works` — single op eval (backward compat)
4. `test_batched_eval_with_fusion` — fused chain still works in batched mode

### Python tests (~2 tests)
1. `test_batched_eval_transparent` — complex expression, verify result unchanged
2. `test_batched_eval_performance` — time 100 iterations of same graph, verify no regression (smoke test, not strict benchmark)

## Backlog: Future Improvements
- Single command buffer batching (encode all ops into one CB)
- Concurrent queues for independent subgraphs
- Async eval (Python-level non-blocking API)
- Fine-grained locking (split Mutex)
