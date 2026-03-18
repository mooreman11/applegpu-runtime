# PrivateUse1 C++ Backend for applegpu_runtime

**Date**: 2026-03-18
**Status**: Approved
**Success metric**: GPU faster than CPU for training with batch_size >= 16, hidden_size >= 256 (transformer/CNN). LSTM h=128 as secondary regression target.

## Problem

The current `__torch_dispatch__` Python integration adds ~20µs per op. LSTM training generates 583K dispatch calls = 11.7s of pure Python overhead. CPU runs the same model in 15ms. Streaming command buffers reduced Metal overhead to ~5ms, but Python dispatch dominates.

## Solution: PrivateUse1 C++ Backend

Replace the Python dispatch hot path with C++ dispatch at the PyTorch dispatcher level. Each op goes: C++ dispatcher (~2ns) → C++ shim → extern "C" Rust → graph record (~100-300ns). Total ~200-400ns per op vs 20,000ns today.

### Execution Model: Record-then-Fuse-then-Encode

```
PyTorch C++ Dispatcher (~2ns)
    ↓
C++ Shim (extract tensor_id + sizes + strides from DataPtr context)
    ↓
extern "C" Rust FFI (~100-300ns, record op to LazyRuntime graph)
    ↓ [at sync point: .item(), .cpu(), synchronize()]
fusion::optimize() → eval_single_cb() → Streaming Metal CB → flush
    ↓
Swift Metal bridge → GPU
```

Ops are NOT encoded into Metal per-call. They record graph nodes in Rust. At sync points, the graph is fused and executed as a single batched Metal dispatch. This preserves CLAUDE.md's "Lazy execution with kernel fusion" principle.

### Tensor Identity

Tensor IDs are stored in `c10::DataPtr`'s opaque `void* context`. The custom allocator creates a small context struct `{ uint64_t tensor_id; }` and attaches it to each DataPtr. Op wrappers extract the tensor_id from each input tensor's DataPtr context.

### Memory

Custom `c10::Allocator` backed by Metal `storageModeShared` buffers via Rust's `BufferPool`. PyTorch's `at::Tensor` directly holds Metal buffer pointers (zero-copy). Deallocation callback calls `applegpu_ffi_free(tensor_id)` which returns the buffer to the pool.

Pre-allocated output buffers: `empty_strided` allocates the Metal buffer immediately and inserts it into `LazyRuntime::tensors` as a pre-allocated tensor. When an op records with this tensor as output, `eval_single_cb()` writes into the existing buffer instead of calling `pool.acquire()`.

### Autograd

Works automatically at the PrivateUse1 dispatch key. No `AutogradPrivateUse1` registration needed. PyTorch's built-in differentiation rules apply.

### CPU Fallback

A BackendFallback kernel (~60 lines) handles unregistered ops. It MUST flush the streaming batch first, then copy tensors to CPU, dispatch the CPU op, copy results back.

## Components

### Component 1: Rust FFI Bridge (`crates/core/src/backend_ffi.rs`, ~500 lines)

New module in `applegpu-core`. Global `Mutex<LazyRuntime>`. Exposes:

- `applegpu_ffi_init() → bool`
- `applegpu_ffi_alloc(size, dtype_i8, *out_tensor_id) → *mut u8`
- `applegpu_ffi_free(tensor_id)`
- `applegpu_ffi_add(a_id, b_id, a_offset, a_sizes_ptr, a_ndim, ...) → u64`
- ... (one per op)
- `applegpu_ffi_eval(tensor_id)`
- `applegpu_ffi_synchronize()` — flush streaming batch, wait
- `applegpu_ffi_shape(tensor_id, *out_dims, *out_ndim)`
- `applegpu_ffi_dtype(tensor_id) → i8`

### Component 2: C++ Shim (`backend_cpp/applegpu_backend.cpp`, ~400 lines)

Links against libtorch + `libapplegpu_core.a` + `libAppleGPUBridge.a` + `-lswiftCore -framework Metal -framework MetalPerformanceShaders -framework Foundation`.

- **Custom Allocator** (~60 lines): `allocate()` calls `applegpu_ffi_alloc()`. Returns `c10::DataPtr` with tensor_id in context. Deleter calls `applegpu_ffi_free()`.
- **Op Registrations** (~200 lines): `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`. 13 minimum ops + top-20 hot ops incrementally. Each wrapper: extract tensor_id from DataPtr context → call extern "C" Rust → wrap result.
- **CPU Fallback** (~60 lines): Flush streaming batch → copy to CPU → dispatch → copy back.
- **DeviceGuardImpl** (~50 lines): Single device, single stream. `synchronize()` calls `applegpu_ffi_synchronize()`.

### Component 3: Modified LazyRuntime (`crates/core/src/lazy.rs`)

- `insert_preallocated(id, buffer)`: stores a Tensor in `self.tensors` with pre-allocated buffer, no graph node. Compatible with topo_sort (treated as materialized leaf).
- `eval_single_cb()`: before `pool.acquire()`, check if tensor_id already in `self.tensors`. If yes, use existing buffer.
- Tensor quota: remove or make configurable for PrivateUse1 (PyTorch manages lifetime via refcounting).

### Component 4: Op Recording Optimizations (`crates/core/src/ops.rs`)

- `SmallVec<[u64; 4]>` for op inputs (eliminates heap alloc for ≤4 inputs)
- Cache shape/dtype inline on tensor lookup
- Target: ~50-100ns per op recording (from current ~200-300ns)

### Component 5: Python Entry Point (`python/applegpu_runtime/cpp_backend.py`)

```python
def load_cpp_backend():
    torch.ops.load_library(_find_backend_dylib())
    torch.utils.rename_privateuse1_backend("applegpu")
    torch.utils.generate_methods_for_privateuse1_backend("applegpu")
```

### Component 6: Build Integration

- `make build-cpp-backend`: `cargo build -p applegpu-core --release` → `torch.utils.cpp_extension` builds C++ shim
- `make test-cpp-backend`: runs C++ backend tests
- `make setup-torch`: full setup including C++ backend
- C++ shim and PyO3 module are mutually exclusive at runtime

## Implementation Order

### Phase 0: Fix tensor lifetime + pre-allocated buffers (Rust-only)
**Tests first**: pre-allocated buffer insertion, eval writes to pre-allocated buffer, tensor persists after eval
- Add `insert_preallocated()` to LazyRuntime
- Modify `eval_single_cb()` to use pre-allocated buffers
- Remove/increase tensor quota for external-lifetime tensors
- `cargo test -p applegpu-core`

### Phase 1: Rust FFI Bridge (foundation)
**Tests first**: FFI alloc/free round-trip, FFI add+eval produces correct result, FFI eval flushes streaming batch
- Create `backend_ffi.rs` with alloc/free/init + 5 core ops (empty_strided, copy, add, matmul, relu)
- Global `Mutex<LazyRuntime>` with device init
- `cargo test -p applegpu-core`

### Phase 2: C++ Shim (minimum viable)
**Tests first**: `torch.empty(3,3, device='applegpu')` works, tensor copy to/from CPU works
- Custom allocator + DataPtr context
- 13 minimum ops + CPU fallback (with streaming flush)
- DeviceGuardImpl
- Build with `torch.utils.cpp_extension`
- `make test-cpp-backend`

### Phase 3: Op Migration (incremental)
**Test each op**: register → run → verify against CPU reference
- Top 20: matmul, add, mul, sub, div, relu, softmax, layer_norm, embedding, transpose, reshape, cat, slice, copy, neg, exp, log, gelu, sigmoid, tanh
- Run full Python test suite after each batch

### Phase 4: Training Validation + Op Recording Optimization
- SmallVec, cached shapes for ~50-100ns per op recording
- Run benchmarks: LSTM, CNN, Transformer
- Compare GPU vs CPU, profile remaining bottlenecks

### Phase 5: Polish
- Python entry point, make targets, documentation
- README update with benchmark results
- Deprecate `__torch_dispatch__` path (keep as debug fallback)

## Key Files

| File | Changes |
|------|---------|
| `crates/core/src/backend_ffi.rs` | NEW: extern "C" FFI bridge (~500 lines) |
| `crates/core/src/lazy.rs` | MODIFY: insert_preallocated, eval uses pre-allocated buffers, quota |
| `crates/core/src/ops.rs` | MODIFY: SmallVec inputs, cached shape/dtype |
| `crates/core/src/lib.rs` | MODIFY: add `pub mod backend_ffi;` |
| `backend_cpp/applegpu_backend.cpp` | NEW: C++ shim (~400 lines) |
| `backend_cpp/setup.py` | NEW: build config for torch.utils.cpp_extension |
| `python/applegpu_runtime/cpp_backend.py` | NEW: load_cpp_backend() entry point |
| `Makefile` | MODIFY: add build-cpp-backend, test-cpp-backend targets |

## Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Per-op dispatch | 20,000ns | 200-400ns (record), reducible to 50-100ns |
| LSTM 583K ops recording | 11,700ms | 117-234ms (→ 29-58ms optimized) |
| CNN 22K ops recording | 440ms | 4.4-8.8ms (→ 1.1-2.2ms optimized) |
| Metal readback | 5ms (streaming) | 5ms (unchanged) |
| GPU eval (fused) | depends on model | single batched dispatch per sync point |
