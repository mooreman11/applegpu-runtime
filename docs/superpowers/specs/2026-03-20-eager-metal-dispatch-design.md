# Eager Metal Dispatch — Design Spec

**Date**: 2026-03-20
**Status**: Draft
**Solves**: P1 (dispatch overhead), P2 (view tensor identity)

## Problem

The PrivateUse1 C++ backend routes every op through the Rust graph engine:
```
C++ op → Rust FFI → graph record → ... → eval → Swift compute → Metal encode → commit → wait
```

This adds ~6µs per op recording + ~275µs per eval (Metal command buffer round-trip). MLP training at h=1024 is 10x slower than CPU.

## Solution: Eager Metal Dispatch

Bypass the graph engine. Encode Metal compute commands directly into a streaming command buffer as ops arrive. GPU executes in parallel with CPU encoding. Only commit+wait at explicit sync points.

```
C++ op → Rust FFI → Metal encode (into open CB) → return immediately
                                                    ↓ GPU already executing
```

## Architecture

### Streaming Command Buffer (already exists)

`compute.rs` has `begin_streaming_batch()` / `flush_streaming_batch()`. Currently used by eval. For eager dispatch, we keep this CB open permanently and encode directly into it.

**Lifecycle:**
1. `applegpu_ffi_init()` → open streaming CB
2. Each op → encode compute command into streaming CB
3. `.cpu()` / `.item()` / `synchronize()` → commit CB, wait, open new CB
4. Between syncs, GPU executes pipelined with CPU encoding

### Tensor Metadata (new: stride-aware)

Current: `tensor_id → (buffer_ptr, shape, dtype)` in Rust runtime.

New: `tensor_id → (buffer_ptr, shape, strides, offset, dtype)`.

```rust
struct EagerTensor {
    buffer: *mut u8,           // Metal buffer data pointer
    buffer_id: u64,            // For buffer pool reference counting
    shape: Vec<usize>,
    strides: Vec<usize>,       // NEW: element strides per dimension
    offset: usize,             // NEW: element offset into buffer
    dtype: DType,
    nbytes: usize,             // Total buffer size (for the base allocation)
}
```

Views create a new `EagerTensor` sharing the same `buffer`/`buffer_id` with different shape/strides/offset. No copies.

### Buffer Pool (keep, simplify)

Pre-allocate Metal buffers of common sizes. `pool.acquire(nbytes)` returns a buffer. `pool.release(buffer_id)` returns it. Reference counted — a buffer is only released when all tensors (including views) referencing it are freed.

### FFI Layer (new: direct dispatch)

Replace the `_out` pattern (graph record + alloc) with direct encode:

```rust
// Current (graph-based):
fn applegpu_ffi_add_out(a_id, b_id, out_id) -> *mut u8 {
    let result_id = ops::add(&mut rt, a_id, b_id)?;  // records graph node
    alloc_output(&mut rt, result_id, out_id)           // pre-allocates buffer
}

// New (eager):
fn applegpu_ffi_add_eager(a_id, b_id, out_id) -> *mut u8 {
    let a = get_tensor(a_id);
    let b = get_tensor(b_id);
    let out_shape = broadcast_shape(&a.shape, &b.shape);
    let out_buf = pool.acquire(out_shape.nbytes(a.dtype));
    encode_binary_kernel("add_f32", &a, &b, &out_buf);  // direct Metal encode
    let out = register_tensor(out_buf, out_shape, a.strides, 0, a.dtype);
    *out_id = out.id;
    out.buffer
}
```

### Metal Kernel Changes (stride-aware)

Current kernels assume contiguous memory: `data[thread_position_in_grid]`.

New kernels accept stride parameters:

```metal
kernel void add_strided(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    constant uint* a_strides [[buffer(3)]],
    constant uint* b_strides [[buffer(4)]],
    constant uint* out_shape [[buffer(5)]],
    constant uint& ndim     [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    // Convert linear index to multi-dim index, apply strides
    uint a_idx = 0, b_idx = 0;
    uint remaining = tid;
    for (uint d = ndim - 1; d < ndim; d--) {
        uint dim_idx = remaining % out_shape[d];
        remaining /= out_shape[d];
        a_idx += dim_idx * a_strides[d];  // broadcasts: stride=0 for broadcast dims
        b_idx += dim_idx * b_strides[d];
    }
    out[tid] = a[a_idx] + b[b_idx];
}
```

For contiguous tensors (common case), we detect this and dispatch the fast non-strided kernel. Strided kernels are only used when views are involved.

### View Ops (zero-cost metadata)

```cpp
// C++ side: view ops just create metadata
at::Tensor applegpu_t(const at::Tensor& self) {
    // Create new tensor_id with swapped strides, same buffer
    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    std::swap(sizes[0], sizes[1]);
    std::swap(strides[0], strides[1]);
    uint64_t view_id = applegpu_ffi_create_view(
        get_tensor_id(self), sizes, strides, self.storage_offset());
    return wrap_tensor(view_id, sizes, self.scalar_type());
}
```

No copies. No `ensure_op_ready`. No shape mismatch issues.

### In-Place Ops (direct buffer write)

```cpp
at::Tensor& applegpu_add_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    // Encode kernel that reads self + other, writes INTO self's buffer
    if (alpha.toDouble() != 1.0) {
        applegpu_ffi_add_scaled_inplace(
            get_tensor_id(self), get_tensor_id(other), alpha.toFloat());
    } else {
        applegpu_ffi_add_inplace(get_tensor_id(self), get_tensor_id(other));
    }
    return self;  // No storage swap, no eval, no copy
}
```

The Metal kernel writes directly into self's buffer. Since we haven't committed the CB yet, this is just another encoded command — the GPU handles ordering.

### Sync Points

Only these trigger commit + wait:
- `tensor.cpu()` → commit CB, wait, read from shared memory
- `tensor.item()` → same
- `applegpu_ffi_synchronize()` → commit CB, wait
- `loss.backward()` → PyTorch autograd needs gradient values, triggers sync internally (via `.cpu()` on scalar loss)

Between sync points, the GPU processes the command buffer in parallel with CPU encoding. This is the MPS model.

### What Gets Removed

From the C++ path's hot loop:
- `lazy.rs` graph recording (no more `record_op`, `OpNode`, topo sort)
- `eval` / `eval_single_cb` (no separate eval step)
- `ensure_op_ready()` (strides handle views natively)
- `eval_applegpu_tensor_if_needed()` (no lazy tensors)
- Pre-allocated buffer dance (`insert_preallocated`, `materialize_preallocated`)
- Deferred-free tracking (`try_deferred_free`, `process_deferred_frees`)

What stays:
- Buffer pool (acquire/release)
- Tensor metadata registry (now stride-aware)
- Streaming command buffer (now the primary dispatch path)
- All Metal kernels (+ strided variants)
- CPU fallback (for unregistered ops)

### What Stays for PyO3 Path

The graph engine (`lazy.rs`, `fusion.rs`) remains for the PyO3 `__torch_dispatch__` path. It's still useful there because Python dispatch overhead makes batched eval worthwhile.

### What Stays for Future `torch.compile`

The graph engine becomes the `torch.compile` backend (Phase 2). When users `torch.compile(model)`, PyTorch captures a graph which we optimize with fusion, then replay via Metal. This is the correct use of a graph engine — optimizing a known-static graph, not recording eager ops.

## Safety Invariants

### 1. CPU reads require flush
All `storageModeShared` buffers are visible to both CPU and GPU. Any CPU-side read (`Buffer::contents()`, `read_bytes()`, `as_slice()`, `from_blob()`) MUST be preceded by `flush_and_wait()`. The eager runtime tracks a `dirty` flag per buffer — set when a GPU write is encoded, cleared after flush. Assert on CPU read of a dirty buffer.

### 2. In-place ops with non-identity strides must copy
For `add_(self, other)` where each thread reads `self[tid]` and writes `self[tid]`, there's no race (identity mapping). But if `self` has non-trivial strides (e.g., a transposed view), the read index and write index for the same `tid` may differ, causing a WAR hazard across threadgroups. **Rule**: in-place ops on non-contiguous `self` must first copy `self` to a temp buffer, then compute `temp + other → self`.

### 3. Single-thread assumption
The streaming CB is thread-local in Swift (`activeBatchCommandBuffer`). The eager path assumes all FFI calls come from the same thread. This is guaranteed by `FfiState.runtime` being behind a `Mutex`. Document this constraint.

### 4. View buffer lifetime
Views share a `buffer_id` with their base tensor. Use `Arc<Buffer>` in `EagerTensor` so the Metal buffer is only released when all tensors (base + views) are dropped.

### 5. Auto-flush threshold
The current `streaming_tick()` auto-flushes at 512 ops. For eager mode, raise this to 65536 (or disable) to avoid mid-pass commits. Only explicit sync points should flush.

### 6. CPU fallback = sync point
Any op hitting `cpu_fallback()` must `flush_and_wait()` first, since it reads tensor data via shared memory. This is already the behavior (current `applegpu_cpu_fallback` calls `applegpu_ffi_synchronize()`), but document it as a hard requirement.

### 7. Use only `_nb` dispatch functions
The eager path must NEVER call synchronous Swift dispatch methods (`dispatchElementwise`, `dispatchMatmul`) which create standalone command buffers with `commit()+waitUntilCompleted()`. Only `_nb` (non-blocking) variants that encode into the streaming batch CB.

## Expected Performance

| Metric | Current | After |
|--------|---------|-------|
| Per-op encode | ~6µs (record + alloc) | ~1.5µs (direct encode) |
| Sync cost | ~275µs (eval per sync) | ~275µs (commit per sync, but fewer syncs) |
| Forward (h=1024) | 0.96ms (lazy + contiguous copies) | ~0.02ms (direct encode, GPU pipelined) |
| Training step (h=1024) | 8.66ms | ~1-2ms (fewer syncs, no mid-step eval) |
| CPU comparison (h=1024) | 3.35ms | GPU wins at ~1-2ms |

## Implementation Phases

### D1: Eager Runtime in Rust (~1 session)
- New `eager.rs` module: `EagerRuntime` with stride-aware tensor registry
- Buffer pool integration (reuse existing `pool.rs`)
- Direct encode API: `encode_binary`, `encode_unary`, `encode_matmul`
- Streaming CB management: `ensure_cb_open`, `flush_and_wait`
- FFI: `applegpu_ffi_*_eager()` functions

### D2: Stride-Aware Metal Kernels (~1 session)
- Strided variants of: add, sub, mul, div, relu, neg, threshold_backward
- Contiguous fast-path detection (skip stride math when all strides are default)
- Matmul: must be contiguous (encode a copy kernel first if strided)
- Scalar mul, mean reduction kernels

### D3: C++ Shim Rewrite (~1 session)
- Replace all `_out` calls with `_eager` calls
- Remove `ensure_op_ready()` entirely
- View ops → `applegpu_ffi_create_view()`
- In-place ops → direct buffer write kernels
- Sync points → `applegpu_ffi_flush_and_wait()`

### D4: Tests + Benchmarks (~1 session)
- Port all 15 existing tests to eager path
- Add view-correctness tests (t → mm, reshape → add, etc.)
- Add stride-correctness tests (non-contiguous inputs)
- Benchmark: GPU vs CPU at h=128,256,512,1024,2048
- Target: GPU faster than CPU at h≥512

### D5 (Future): `torch.compile` Backend
- Register as a PyTorch compile backend
- Graph capture → fusion optimization → batched Metal replay
- Targets small tensor sizes where per-op dispatch still dominates

## Success Criteria

1. Zero CPU fallback for MLP training (forward + backward + optimizer)
2. GPU faster than CPU at h≥512, b≥128
3. GPT-2 inference runs without crashes (views work natively)
4. All 15 existing tests pass
5. No `ensure_op_ready()` in the codebase
