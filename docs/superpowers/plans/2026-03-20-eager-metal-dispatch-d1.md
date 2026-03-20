# Eager Metal Dispatch D1: Eager Runtime in Rust

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the graph-based `_out` FFI functions with eager Metal dispatch functions that encode directly into a streaming command buffer, bypassing `lazy.rs` entirely.

**Architecture:** New `eager.rs` module provides `EagerRuntime` — a stride-aware tensor registry with `Arc<Buffer>` sharing for views, a `KernelRegistry` for pipeline caching, and direct `_nb` dispatch into a streaming CB. New `eager_ffi.rs` exposes `applegpu_eager_*` extern "C" functions. The C++ backend switches from `_out` to `_eager` calls.

**Tech Stack:** Rust (crates/core), existing Swift `_nb` dispatch functions, existing `BufferPool`, existing `KernelRegistry`, existing `ComputePipeline`

**Spec:** `docs/superpowers/specs/2026-03-20-eager-metal-dispatch-design.md`

---

## Critical API Reference (from codebase exploration)

These are the exact signatures the plan depends on:

```rust
// Kernel resolution (compute.rs:1423) — STATIC method
KernelRegistry::resolve_kernel(base_name: &str, dtype: DType) -> (String, String)
// base_name examples: "elementwise_add", "elementwise_sub", "elementwise_mul",
//   "elementwise_div", "elementwise_neg", "elementwise_relu", "matmul", "softmax", "scalar_mul"
// Returns: (msl_source, suffixed_function_name) e.g. ("kernel void...", "elementwise_add_f32")

// Pipeline cache (compute.rs:1391)
KernelRegistry::new() -> Self
KernelRegistry::get_or_create(&self, device: &Device, kernel_source: &str, function_name: &str) -> Result<Arc<ComputePipeline>>

// Non-blocking binary dispatch (compute.rs:1003) — on ComputePipeline
ComputePipeline::dispatch_binary_nd_nb(&self, queue: *mut c_void,
    buf_a: &Buffer, a_strides: &[u32; MAX_DIMS],
    buf_b: &Buffer, b_strides: &[u32; MAX_DIMS],
    buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
    ndim: u32, numel: u32) -> Result<*mut c_void>

// Non-blocking unary dispatch (compute.rs)
ComputePipeline::dispatch_unary_nd_nb(&self, queue: *mut c_void,
    buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
    buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
    ndim: u32, numel: u32) -> Result<*mut c_void>

// Non-blocking matmul (compute.rs)
ComputePipeline::dispatch_matmul_batched_nb(&self, queue: *mut c_void,
    buf_a: &Buffer, buf_b: &Buffer, buf_out: &Buffer,
    m: u32, n: u32, k: u32, batch_size: u32,
    a_batch_stride: u32, b_batch_stride: u32) -> Result<*mut c_void>

// Buffer handle (buffer.rs:148) — pub(crate)
Buffer::raw_handle(&self) -> *mut ffi::GPUBufferHandle

// Buffer contents (buffer.rs) — shared memory pointer
Buffer::contents(&self) -> *mut u8

// Streaming CB (compute.rs:2542-2642)
compute::begin_streaming_batch(queue: *mut c_void)
compute::flush_streaming_batch()
compute::end_streaming_batch()
compute::streaming_is_active() -> bool
compute::streaming_tick()
compute::get_shared_queue(device: &Device) -> *mut c_void

// TensorLayout (tensor.rs)
TensorLayout::contiguous(shape: Shape) -> Self
TensorLayout::is_contiguous(&self) -> bool
TensorLayout::broadcast_strides_for(source: &Shape, target: &Shape) -> [usize; MAX_DIMS]
```

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `crates/core/src/eager.rs` | Create | EagerRuntime: tensor registry, buffer ref-counting, kernel dispatch, view creation, flush/sync |
| `crates/core/src/eager_ffi.rs` | Create | Eager FFI functions (extern "C") for C++ backend |
| `crates/core/src/lib.rs` | Modify | Add `pub mod eager; pub mod eager_ffi;` |
| `crates/core/src/tensor.rs` | Modify | Add `TensorLayout::set_stride()` |
| `crates/core/src/compute.rs` | Modify | Raise streaming auto-flush default to 65536 |
| `backend_cpp/applegpu_ffi.h` | Modify | Add eager FFI declarations |
| `backend_cpp/applegpu_backend.cpp` | Modify | Switch ops from `_out` to `_eager` calls |
| `crates/core/tests/eager_integration.rs` | Create | Integration tests for eager runtime |

---

### Task 1: EagerRuntime — Tensor Registry with Arc\<Buffer\>

**Files:**
- Create: `crates/core/src/eager.rs`
- Modify: `crates/core/src/lib.rs`
- Modify: `crates/core/src/tensor.rs` (add `set_stride`)
- Test: `crates/core/tests/eager_integration.rs`

- [ ] **Step 1: Add `TensorLayout::set_stride()` to tensor.rs**

In `crates/core/src/tensor.rs`, add to `impl TensorLayout`:
```rust
/// Set stride for a specific dimension (used by view creation).
pub fn set_stride(&mut self, dim: usize, stride: usize) {
    assert!(dim < MAX_DIMS, "dim {} exceeds MAX_DIMS {}", dim, MAX_DIMS);
    self.strides[dim] = stride;
}
```

- [ ] **Step 2: Write failing test for tensor registration**

In `crates/core/tests/eager_integration.rs`:
```rust
use applegpu_core::eager::EagerRuntime;
use applegpu_core::tensor::DType;
use applegpu_core::device::Device;

#[test]
fn test_eager_register_and_query() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    let (id, ptr) = rt.alloc(&device, &[2, 3], DType::Float32).unwrap();
    assert!(!ptr.is_null());
    assert_eq!(rt.shape(id).unwrap(), vec![2, 3]);
    assert_eq!(rt.dtype(id).unwrap(), DType::Float32);
    assert!(rt.is_contiguous(id).unwrap());
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p applegpu-core test_eager_register_and_query`
Expected: FAIL — module `eager` not found

- [ ] **Step 4: Write EagerRuntime**

Create `crates/core/src/eager.rs` with:
- `EagerTensor` struct: `buffer: Arc<Buffer>`, `layout: TensorLayout`, `dtype: DType`, `offset: usize`
- Helper methods: `data_ptr()`, `is_contiguous()`, `nbytes()`, `numel()`, `shape_vec()`, `strides_u32()`, `shape_u32()`
- `EagerRuntime` struct: `tensors: HashMap<u64, EagerTensor>`, `pool: BufferPool`, `registry: KernelRegistry`
- Methods: `new()`, `alloc()`, `free()`, `get()`, `shape()`, `dtype()`, `is_contiguous()`, `create_view()`, `tensor_count()`, `pool_stats()`

Key design notes:
- `alloc()` takes `&[usize]` shape directly (NOT inferred from bytes)
- `free()` uses `Arc::into_inner()` — if last reference, returns buffer to pool
- `create_view()` shares `Arc<Buffer>` with base tensor, applies custom strides/offset
- `KernelRegistry` stored as field (NOT global/static)

Add `pub mod eager;` to `crates/core/src/lib.rs`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p applegpu-core test_eager_register_and_query`
Expected: PASS

- [ ] **Step 6: Write test for view creation and buffer sharing**

```rust
#[test]
fn test_eager_view_shares_buffer() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    let (base_id, base_ptr) = rt.alloc(&device, &[4, 8], DType::Float32).unwrap();

    // Transposed view: shape [8, 4], strides [1, 8]
    let view_id = rt.create_view(base_id, &[8, 4], &[1, 8], 0).unwrap();

    let base_t = rt.get(base_id).unwrap();
    let view_t = rt.get(view_id).unwrap();
    assert_eq!(base_t.data_ptr(), view_t.data_ptr()); // same buffer
    assert_eq!(rt.shape(view_id).unwrap(), vec![8, 4]);
    assert!(!rt.is_contiguous(view_id).unwrap());

    // Free base — view still valid (Arc holds buffer)
    rt.free(base_id);
    assert!(rt.get(base_id).is_err());
    assert!(rt.get(view_id).is_ok());
    rt.free(view_id);
}
```

- [ ] **Step 7: Run test**

Run: `cargo test -p applegpu-core test_eager_view`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add crates/core/src/eager.rs crates/core/src/lib.rs crates/core/src/tensor.rs crates/core/tests/eager_integration.rs
git commit -m "feat(D1): EagerRuntime with stride-aware tensor registry and Arc<Buffer> views"
```

---

### Task 2: Streaming CB Management + Dispatch Helpers

**Files:**
- Modify: `crates/core/src/eager.rs`
- Modify: `crates/core/src/compute.rs`
- Test: `crates/core/tests/eager_integration.rs`

- [ ] **Step 1: Write failing test for streaming lifecycle**

```rust
#[test]
fn test_eager_streaming_lifecycle() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);
    assert!(rt.is_streaming());
    rt.flush_and_wait();
    assert!(rt.is_streaming()); // still active
    rt.end_streaming();
    assert!(!rt.is_streaming());
}
```

- [ ] **Step 2: Run test — expected fail**

Run: `cargo test -p applegpu-core test_eager_streaming_lifecycle`

- [ ] **Step 3: Add streaming + dispatch methods to EagerRuntime**

Add to `eager.rs`:
```rust
use crate::compute::{self, KernelRegistry, ComputePipeline};
use std::sync::Arc;

impl EagerRuntime {
    pub fn begin_streaming(&mut self, device: &Device) {
        if !compute::streaming_is_active() {
            let queue = compute::get_shared_queue(device);
            compute::begin_streaming_batch(queue);
        }
    }

    pub fn flush_and_wait(&self) {
        if compute::streaming_is_active() {
            compute::flush_streaming_batch();
        }
    }

    pub fn end_streaming(&self) {
        if compute::streaming_is_active() {
            compute::end_streaming_batch();
        }
    }

    pub fn is_streaming(&self) -> bool {
        compute::streaming_is_active()
    }

    /// Resolve kernel name + compile pipeline. Caches by function name.
    fn get_pipeline(&self, device: &Device, base_name: &str, dtype: DType) -> Result<Arc<ComputePipeline>> {
        let (source, func_name) = KernelRegistry::resolve_kernel(base_name, dtype);
        self.registry.get_or_create(device, &source, &func_name)
    }
}
```

- [ ] **Step 4: Raise auto-flush threshold in compute.rs**

In `crates/core/src/compute.rs`, change default:
```rust
fn streaming_flush_interval() -> u32 {
    std::env::var("APPLEGPU_STREAMING_FLUSH_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536) // Was 512; raised for eager dispatch
}
```

- [ ] **Step 5: Run test**

Run: `cargo test -p applegpu-core test_eager_streaming_lifecycle`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/eager.rs crates/core/src/compute.rs
git commit -m "feat(D1): streaming CB management + kernel pipeline resolution"
```

---

### Task 3: Eager Binary Op Dispatch (add, sub, mul, div)

**Files:**
- Modify: `crates/core/src/eager.rs`
- Test: `crates/core/tests/eager_integration.rs`

- [ ] **Step 1: Write failing test for eager add**

```rust
#[test]
fn test_eager_add_contiguous() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    let (b_id, b_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 4).copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        std::slice::from_raw_parts_mut(b_ptr as *mut f32, 4).copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
    }

    let (out_id, out_ptr) = rt.binary_op(&device, "elementwise_add", a_id, b_id).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 4) };
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    rt.end_streaming();
}
```

- [ ] **Step 2: Run test — expected fail**

- [ ] **Step 3: Implement binary_op**

Key implementation pattern (handles borrow checker):
```rust
pub fn binary_op(
    &mut self, device: &Device, base_kernel: &str, a_id: u64, b_id: u64,
) -> Result<(u64, *mut u8)> {
    // 1. Extract all data from immutable borrows FIRST
    let a = self.get(a_id)?;
    let b = self.get(b_id)?;
    let a_shape = a.layout.shape;
    let b_shape = b.layout.shape;
    let dtype = a.dtype;
    let a_buf = Arc::clone(&a.buffer);
    let b_buf = Arc::clone(&b.buffer);
    let a_strides_raw = TensorLayout::broadcast_strides_for(&a.layout.shape, &out_shape);
    let b_strides_raw = TensorLayout::broadcast_strides_for(&b.layout.shape, &out_shape);
    // ... convert to [u32; MAX_DIMS] ...

    // 2. Compute output shape
    let out_shape = a_shape.broadcast_with(&b_shape)?;

    // 3. Now do mutable operations (pool.acquire)
    let out_buffer = Arc::new(self.pool.acquire(device, nbytes)?);
    let out_ptr = out_buffer.contents();

    // 4. Get pipeline and dispatch
    let pipeline = self.get_pipeline(device, base_kernel, dtype)?;
    let queue = compute::get_shared_queue(device);
    let _cb = pipeline.dispatch_binary_nd_nb(
        queue, &*a_buf, &a_strides, &*b_buf, &b_strides,
        &*out_buffer, &shape_u32, ndim as u32, numel as u32,
    )?;
    compute::streaming_tick();

    // 5. Register output
    let out_id = next_tensor_id();
    self.tensors.insert(out_id, EagerTensor { buffer: out_buffer, layout: out_layout, dtype, offset: 0 });
    Ok((out_id, out_ptr))
}
```

Note: Pass `&*a_buf` to deref `Arc<Buffer>` → `&Buffer`. The dispatch method takes `&Buffer` internally.

- [ ] **Step 4: Run test**

Run: `cargo test -p applegpu-core test_eager_add_contiguous`
Expected: PASS

- [ ] **Step 5: Write broadcast test**

```rust
#[test]
fn test_eager_add_broadcast() {
    // [2, 3] + [3] → [2, 3]
    // ... (init, write data, dispatch, flush, verify) ...
}
```

- [ ] **Step 6: Run test, commit**

```bash
git commit -m "feat(D1): eager binary_op dispatch with broadcast support"
```

---

### Task 4: Eager Unary Ops + Matmul

**Files:**
- Modify: `crates/core/src/eager.rs`
- Test: `crates/core/tests/eager_integration.rs`

- [ ] **Step 1: Write failing test for relu**

- [ ] **Step 2: Implement `unary_op`** — same pattern as binary_op but with `dispatch_unary_nd_nb`

- [ ] **Step 3: Run test, verify pass**

- [ ] **Step 4: Write failing test for matmul**

```rust
#[test]
fn test_eager_matmul() {
    // [2, 3] @ [3, 4] → [2, 4] with identity-ish matrix
}
```

- [ ] **Step 5: Implement `matmul`**

Uses `dispatch_matmul_batched_nb`. Matmul requires contiguous inputs. If non-contiguous, call `make_contiguous` first. `make_contiguous` dispatches a strided unary copy (any elementwise kernel with strided input → contiguous output acts as a copy).

- [ ] **Step 6: Run test, commit**

```bash
git commit -m "feat(D1): eager unary_op and matmul dispatch"
```

---

### Task 5: make_contiguous + In-Place Ops

**Files:**
- Modify: `crates/core/src/eager.rs`
- Test: `crates/core/tests/eager_integration.rs`

- [ ] **Step 1: Write test for make_contiguous on transposed view**

- [ ] **Step 2: Implement `make_contiguous`**

If already contiguous → return same id. Otherwise dispatch `unary_op(device, "elementwise_relu", id)` — NO, use an identity kernel. Check if `"elementwise_abs"` of positive data works, or add a `"copy"` entry to `resolve_kernel`. Simplest: use `elementwise_neg` twice (neg of neg = identity). Or better: just do a CPU memcpy via `flush_and_wait` + stride-aware copy on shared memory. For D1, CPU-side strided copy is fine — D2 adds the GPU copy kernel.

```rust
pub fn make_contiguous(&mut self, device: &Device, id: u64) -> Result<(u64, *mut u8)> {
    if self.is_contiguous(id)? {
        let ptr = self.get(id)?.data_ptr();
        return Ok((id, ptr));
    }
    // For D1: CPU-side strided copy (flush first, then memcpy with strides)
    self.flush_and_wait();
    let src = self.get(id)?;
    let shape = src.layout.shape;
    let dtype = src.dtype;
    let strides = src.layout.strides().to_vec();
    let src_ptr = src.data_ptr();
    let numel = shape.numel();
    let elem_size = dtype.size_bytes();

    let (out_id, out_ptr) = self.alloc(device, shape.dims(), dtype)?;

    // Strided copy: iterate over multi-dim indices
    for linear in 0..numel {
        let mut remaining = linear;
        let mut src_offset = 0usize;
        for d in (0..shape.ndim()).rev() {
            let idx = remaining % shape.dims()[d];
            remaining /= shape.dims()[d];
            src_offset += idx * strides[d] * elem_size;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_ptr.add(src_offset), out_ptr.add(linear * elem_size), elem_size);
        }
    }
    Ok((out_id, out_ptr))
}
```

- [ ] **Step 3: Run test**

- [ ] **Step 4: Write test for in-place add**

- [ ] **Step 5: Implement `inplace_binary_op`**

If self is non-contiguous → call `make_contiguous` first, then dispatch in-place (spec Safety Invariant #2). The output buffer IS self's buffer — binary dispatch writes `out = op(self, other)` where `out` pointer = self's buffer.

- [ ] **Step 6: Run test, commit**

```bash
git commit -m "feat(D1): make_contiguous (CPU stride copy) + inplace_binary_op"
```

---

### Task 6: Eager FFI Bridge

**Files:**
- Create: `crates/core/src/eager_ffi.rs`
- Modify: `crates/core/src/lib.rs`
- Modify: `backend_cpp/applegpu_ffi.h`

- [ ] **Step 1: Create eager_ffi.rs**

Key design decisions:
- **Shared Device**: Use same `FfiState` device from `backend_ffi.rs` via `get_state().device`. Do NOT create a second Device. Add `EagerRuntime` as a second field in `FfiState`, or use a separate `OnceLock<Mutex<EagerRuntime>>` that borrows the device from `FfiState`.
- **Init**: `applegpu_eager_init()` initializes `EagerRuntime` and calls `begin_streaming()`.
- **Alloc**: `applegpu_eager_alloc(dims, ndim, dtype, out_id)` — takes N-D shape directly (NOT bytes).
- **Error handling**: Thread-local `CString` (same pattern as `backend_ffi.rs`).

FFI functions to expose:
```c
// Lifecycle
bool applegpu_eager_init(void);
const char* applegpu_eager_last_error(void);

// Memory
uint8_t* applegpu_eager_alloc(const uint64_t* dims, uint32_t ndim, int8_t dtype, uint64_t* out_id);
void applegpu_eager_free(uint64_t id);
int32_t applegpu_eager_register_shape(uint64_t id, const uint64_t* dims, uint32_t ndim);
int32_t applegpu_eager_shape(uint64_t id, uint64_t* out_dims, uint32_t* out_ndim);
int8_t applegpu_eager_dtype(uint64_t id);

// Binary ops (all return data_ptr, write tensor_id to *out_id)
uint8_t* applegpu_eager_add(uint64_t a, uint64_t b, uint64_t* out_id);
uint8_t* applegpu_eager_sub(uint64_t a, uint64_t b, uint64_t* out_id);
uint8_t* applegpu_eager_mul(uint64_t a, uint64_t b, uint64_t* out_id);
uint8_t* applegpu_eager_div(uint64_t a, uint64_t b, uint64_t* out_id);

// Unary ops
uint8_t* applegpu_eager_relu(uint64_t input, uint64_t* out_id);
uint8_t* applegpu_eager_neg(uint64_t input, uint64_t* out_id);

// Matmul
uint8_t* applegpu_eager_matmul(uint64_t a, uint64_t b, uint64_t* out_id);

// Compound ops (for MLP training)
uint8_t* applegpu_eager_threshold_backward(uint64_t grad, uint64_t input, float threshold, uint64_t* out_id);
uint8_t* applegpu_eager_scalar_mul(uint64_t input, float scale, uint64_t* out_id);
uint8_t* applegpu_eager_mean_all(uint64_t input, uint64_t* out_id);

// Views
uint8_t* applegpu_eager_create_view(uint64_t base, const uint64_t* shape, const uint64_t* strides, uint32_t ndim, uint64_t offset, uint64_t* out_id);

// In-place
int32_t applegpu_eager_add_inplace(uint64_t self_id, uint64_t other_id);
int32_t applegpu_eager_add_scaled_inplace(uint64_t self_id, uint64_t other_id, float alpha);

// Sync
void applegpu_eager_flush_and_wait(void);
void applegpu_eager_synchronize(void);
```

- [ ] **Step 2: Add `pub mod eager_ffi;` to lib.rs**

- [ ] **Step 3: Update `backend_cpp/applegpu_ffi.h`**

Add all declarations above inside the `extern "C"` block.

- [ ] **Step 4: Build Rust**

Run: `cargo build -p applegpu-core --release`
Expected: Compiles

- [ ] **Step 5: Run all Rust tests**

Run: `cargo test -p applegpu-core`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/eager_ffi.rs crates/core/src/lib.rs backend_cpp/applegpu_ffi.h
git commit -m "feat(D1): eager FFI bridge with all ops for MLP training"
```

---

### Task 7: C++ Backend Integration

**Files:**
- Modify: `backend_cpp/applegpu_backend.cpp`
- Test: `python/tests/test_cpp_backend.py`

This is the integration task. Switch C++ ops from graph-based `_out` calls to eager `_eager` calls. Do incrementally:

- [ ] **Step 1: Add eager init + helper functions**

In `applegpu_backend.cpp`:
- Call `applegpu_eager_init()` during module init
- Add `query_eager_shape(id)` helper that calls `applegpu_eager_shape`
- Add `wrap_eager_output(ptr, id, shape, dtype)` helper

- [ ] **Step 2: Switch `applegpu_add` to eager**

Replace `applegpu_ffi_add_out` → `applegpu_eager_add`. Remove `ensure_op_ready()`.

- [ ] **Step 3: Run `test_native_add`**

Run: `uv run pytest python/tests/test_cpp_backend.py::test_native_add -v`
Expected: PASS

- [ ] **Step 4: Switch remaining ops one at a time**

For each op: switch implementation, run its test, move on.
Order: mul → sub → mm → relu → neg → addmm → threshold_backward → mse_loss → mse_loss_backward → add_ → mul_ → fill_ → zero_ → t → view → as_strided

For ops without eager FFI equivalents (sum, copy_, resize_), keep existing implementations.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest python/tests/test_cpp_backend.py -v`
Expected: 15/15 pass

- [ ] **Step 6: Benchmark**

Run: `uv run python benchmarks/bench_mlp_cpp.py --hidden 1024 --batch 256 --iters 20`
Expected: Significant improvement (target: within 2x of CPU at h=1024)

- [ ] **Step 7: Commit**

```bash
git add backend_cpp/applegpu_backend.cpp
git commit -m "feat(D1): switch C++ backend to eager Metal dispatch

All ops encode directly into streaming Metal command buffer.
No graph recording, no eval, no ensure_op_ready.
Views are zero-cost metadata (Arc<Buffer> sharing)."
```

---

## Verification Checklist

- [ ] `cargo test -p applegpu-core` — all Rust tests pass (existing + ~8 new eager tests)
- [ ] `uv run pytest python/tests/test_cpp_backend.py -v` — 15/15 pass
- [ ] `APPLEGPU_LOG_FALLBACK=1` training step shows minimal fallback
- [ ] Views (t, reshape, as_strided) work without copies or `ensure_op_ready`
- [ ] `uv run python benchmarks/bench_mlp_cpp.py --hidden 1024 --batch 256` shows improvement
- [ ] No `ensure_op_ready()` calls for eager-path ops
