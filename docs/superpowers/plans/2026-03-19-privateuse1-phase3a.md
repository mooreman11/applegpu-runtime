# PrivateUse1 Phase 3a: CPU Fallback Fix + Native MLP Ops

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the CPU fallback crash so ALL ops work, then register native GPU dispatch for the MLP-critical ops (add, mul, matmul, relu) to eliminate Python dispatch overhead for the forward pass.

**Architecture:** Fix `_copy_from_and_resize` in the C++ shim so cpu_fallback can copy results back. Then add C++ op wrappers that extract tensor_ids from DataPtr context, call Rust FFI `applegpu_ffi_*` to record graph ops, create output tensors with pre-allocated Metal buffers, and return them. PyTorch autograd works automatically for aten ops.

**Tech Stack:** C++ (libtorch/ATen), Rust (applegpu-core FFI), Python (pytest)

**Spec:** `docs/superpowers/specs/2026-03-18-privateuse1-backend-design.md`

---

## File Structure

**Task 1 (CPU fallback fix):**
- Modify: `backend_cpp/applegpu_backend.cpp` — add `_copy_from_and_resize`, fix `_copy_from`
- Modify: `python/tests/test_cpp_backend.py` — unskip CPU fallback test

**Task 2 (Rust FFI: mul, sub, neg):**
- Modify: `crates/core/src/backend_ffi.rs` — add `applegpu_ffi_mul`, `applegpu_ffi_sub`, `applegpu_ffi_neg`
- Modify: `backend_cpp/applegpu_ffi.h` — declare new FFI functions
- Modify: `crates/core/tests/backend_ffi_integration.rs` — FFI round-trip tests

**Task 3 (C++ native ops: add, sub, mul, matmul, relu, neg):**
- Modify: `backend_cpp/applegpu_backend.cpp` — add C++ wrappers + TORCH_LIBRARY_IMPL registrations
- Modify: `python/tests/test_cpp_backend.py` — add native op tests

**Task 4 (MLP benchmark):**
- Create: `benchmarks/bench_mlp_cpp.py` — MLP training benchmark comparing CPU vs C++ backend
- Modify: `Makefile` — add `bench-mlp-cpp` target

---

## Task 1: Fix CPU Fallback (`_copy_from_and_resize` + `_copy_from`)

The CPU fallback crashes because `at::native::cpu_fallback` calls `_copy_from_and_resize` to copy results back to PrivateUse1 tensors. This op is not registered on the CPU dispatch key — it needs to be registered on PrivateUse1. The crash is a segfault because the fallback internally creates a CPU tensor, runs the op on CPU, then tries to copy back.

The root cause: `cpu_fallback` calls `_copy_from_and_resize(cpu_result, original_privateuse1_tensor)` — this dispatches on the `original_privateuse1_tensor` which is PrivateUse1. Our fallback catches it, but it recursively falls back again since the op itself is `_copy_from_and_resize`. We need to register it explicitly.

**Files:**
- Modify: `backend_cpp/applegpu_backend.cpp:170-180` (registration section)
- Modify: `python/tests/test_cpp_backend.py` (unskip fallback test)

- [ ] **Step 1: Write failing test — CPU fallback for torch.sin**

In `python/tests/test_cpp_backend.py`, change the `@pytest.mark.skip` on `test_cpu_fallback_ops` to remove the skip decorator.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend_cpp && rm -rf build *.so && ARCHFLAGS="-arch arm64" uv run python setup.py build_ext --inplace && cd .. && uv run pytest python/tests/test_cpp_backend.py::test_cpu_fallback_ops -v`
Expected: FAIL — segfault or `_copy_from_and_resize` error

- [ ] **Step 3: Implement `_copy_from_and_resize` and `_copy_from`**

In `backend_cpp/applegpu_backend.cpp`, add before the registration block:

```cpp
// _copy_from: copy src (any device) into dst (PrivateUse1).
// Required by CPU fallback to copy results back from CPU to PrivateUse1.
at::Tensor applegpu_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
    applegpu_ffi_synchronize();
    dst.copy_(self);
    return dst;
}

// _copy_from_and_resize: resize dst to match self, then copy.
// Called by cpu_fallback after running op on CPU to write result back.
at::Tensor applegpu_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    applegpu_ffi_synchronize();
    dst.resize_(self.sizes());
    dst.copy_(self);
    return dst;
}
```

Add to `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`:

```cpp
    m.impl("_copy_from", applegpu_copy_from);
    m.impl("_copy_from_and_resize", applegpu_copy_from_and_resize);
```

Note: `dst.resize_()` will allocate a new Metal buffer via our allocator if the size changes, then `dst.copy_()` dispatches to our `copy_` impl which does memcpy. This is safe because our Metal buffers are storageModeShared.

- [ ] **Step 4: Rebuild and run test**

Run: `cd backend_cpp && rm -rf build *.so && ARCHFLAGS="-arch arm64" uv run python setup.py build_ext --inplace && cd .. && uv run pytest python/tests/test_cpp_backend.py -v`
Expected: All 6 tests PASS (including the previously skipped fallback test)

If the segfault persists, debug with:
```python
import faulthandler; faulthandler.enable()
```
The likely cause is `resize_` dispatching to fallback which recurse. If so, register `resize_` explicitly:
```cpp
const at::Tensor& applegpu_resize_(const at::Tensor& self, c10::IntArrayRef size,
                                    std::optional<at::MemoryFormat> fmt) {
    // Reallocate if size changed
    auto* impl = self.unsafeGetTensorImpl();
    auto strides = c10::contiguous_strides(size);
    auto nbytes = at::detail::computeStorageNbytes(size, strides, self.dtype().itemsize());
    if (nbytes > (int64_t)self.storage().nbytes()) {
        auto new_storage = c10::Storage(
            c10::Storage::use_byte_size_t(), nbytes,
            global_allocator.allocate(nbytes), &global_allocator);
        impl->set_storage_and_dtype(std::move(new_storage), self.dtype());
    }
    impl->set_sizes_and_strides(size, strides);
    // Register in Rust if tensor has context
    if (nbytes > 0 && self.storage().data_ptr().get_context()) {
        uint64_t tid = get_tensor_id(self);
        std::vector<uint64_t> dims(size.begin(), size.end());
        applegpu_ffi_register_tensor(tid, dims.data(), dims.size(), scalar_type_to_wire(self.scalar_type()));
    }
    return self;
}
```
Register: `m.impl("resize_", applegpu_resize_);`

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest python/tests/test_cpp_backend.py -v`
Expected: 6 passed, 0 skipped

- [ ] **Step 6: Commit**

```bash
git add backend_cpp/applegpu_backend.cpp python/tests/test_cpp_backend.py
git commit -m "fix: CPU fallback crash — register _copy_from_and_resize

The CPU fallback needs _copy_from_and_resize to copy results back to
PrivateUse1 tensors. Without it, the fallback segfaults. Also registers
_copy_from and resize_ for proper lifecycle management."
```

---

## Task 2: Add Rust FFI for mul, sub, neg

The Rust ops module has `mul`, `sub`, `neg` but no FFI wrappers. Add them so the C++ shim can call them.

**Files:**
- Modify: `crates/core/src/backend_ffi.rs:224-266` (ops section)
- Modify: `backend_cpp/applegpu_ffi.h`
- Modify: `crates/core/tests/backend_ffi_integration.rs`

- [ ] **Step 1: Write failing test — FFI mul round-trip**

Add to `crates/core/tests/backend_ffi_integration.rs`:

```rust
#[test]
fn ffi_mul_eval_readback() {
    backend_ffi::applegpu_ffi_init();
    let mut a_id: u64 = 0;
    let mut b_id: u64 = 0;
    let a_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut a_id);
    let b_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut b_id);

    unsafe {
        let a = a_ptr as *mut f32;
        let b = b_ptr as *mut f32;
        for i in 0..4 {
            *a.add(i) = (i + 1) as f32;       // [1, 2, 3, 4]
            *b.add(i) = (i + 1) as f32 * 2.0; // [2, 4, 6, 8]
        }
    }

    let dims: [u64; 1] = [4];
    assert_eq!(backend_ffi::applegpu_ffi_register_tensor(a_id, dims.as_ptr(), 1, 0), 0);
    assert_eq!(backend_ffi::applegpu_ffi_register_tensor(b_id, dims.as_ptr(), 1, 0), 0);

    let result_id = backend_ffi::applegpu_ffi_mul(a_id, b_id);
    assert!(result_id > 0);

    assert_eq!(backend_ffi::applegpu_ffi_eval(result_id), 0);

    let mut out = [0.0f32; 4];
    assert_eq!(backend_ffi::applegpu_ffi_read_f32(result_id, out.as_mut_ptr(), 4), 0);
    assert_eq!(out, [2.0, 8.0, 18.0, 32.0]);

    backend_ffi::applegpu_ffi_free(a_id);
    backend_ffi::applegpu_ffi_free(b_id);
    backend_ffi::applegpu_ffi_free(result_id);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core --test backend_ffi_integration -- ffi_mul`
Expected: FAIL — `applegpu_ffi_mul` doesn't exist

- [ ] **Step 3: Implement mul, sub, neg FFI functions**

In `crates/core/src/backend_ffi.rs`, after the `applegpu_ffi_relu` function, add:

```rust
/// Record a mul op. Returns the output tensor_id, or 0 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_mul(a_id: u64, b_id: u64) -> u64 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match crate::ops::mul(&mut rt, a_id, b_id) {
        Ok(id) => id,
        Err(e) => { set_error(format!("mul failed: {}", e)); 0 }
    }
}

/// Record a sub op. Returns the output tensor_id, or 0 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_sub(a_id: u64, b_id: u64) -> u64 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match crate::ops::sub(&mut rt, a_id, b_id) {
        Ok(id) => id,
        Err(e) => { set_error(format!("sub failed: {}", e)); 0 }
    }
}

/// Record a neg op. Returns the output tensor_id, or 0 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_neg(input_id: u64) -> u64 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match crate::ops::neg(&mut rt, input_id) {
        Ok(id) => id,
        Err(e) => { set_error(format!("neg failed: {}", e)); 0 }
    }
}
```

In `backend_cpp/applegpu_ffi.h`, add to the ops section:

```c
uint64_t applegpu_ffi_mul(uint64_t a_id, uint64_t b_id);
uint64_t applegpu_ffi_sub(uint64_t a_id, uint64_t b_id);
uint64_t applegpu_ffi_neg(uint64_t input_id);
```

- [ ] **Step 4: Write sub and neg tests**

Add to `crates/core/tests/backend_ffi_integration.rs`:

```rust
#[test]
fn ffi_sub_eval_readback() {
    backend_ffi::applegpu_ffi_init();
    let mut a_id: u64 = 0;
    let mut b_id: u64 = 0;
    let a_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut a_id);
    let b_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut b_id);
    unsafe {
        let a = a_ptr as *mut f32;
        let b = b_ptr as *mut f32;
        *a.add(0) = 10.0; *a.add(1) = 20.0; *a.add(2) = 30.0; *a.add(3) = 40.0;
        *b.add(0) = 1.0;  *b.add(1) = 2.0;  *b.add(2) = 3.0;  *b.add(3) = 4.0;
    }
    let dims: [u64; 1] = [4];
    assert_eq!(backend_ffi::applegpu_ffi_register_tensor(a_id, dims.as_ptr(), 1, 0), 0);
    assert_eq!(backend_ffi::applegpu_ffi_register_tensor(b_id, dims.as_ptr(), 1, 0), 0);
    let result_id = backend_ffi::applegpu_ffi_sub(a_id, b_id);
    assert!(result_id > 0);
    assert_eq!(backend_ffi::applegpu_ffi_eval(result_id), 0);
    let mut out = [0.0f32; 4];
    assert_eq!(backend_ffi::applegpu_ffi_read_f32(result_id, out.as_mut_ptr(), 4), 0);
    assert_eq!(out, [9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn ffi_neg_eval_readback() {
    backend_ffi::applegpu_ffi_init();
    let mut a_id: u64 = 0;
    let a_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut a_id);
    unsafe {
        let a = a_ptr as *mut f32;
        *a.add(0) = -2.0; *a.add(1) = 0.0; *a.add(2) = 3.0; *a.add(3) = -1.0;
    }
    let dims: [u64; 1] = [4];
    assert_eq!(backend_ffi::applegpu_ffi_register_tensor(a_id, dims.as_ptr(), 1, 0), 0);
    let result_id = backend_ffi::applegpu_ffi_neg(a_id);
    assert!(result_id > 0);
    assert_eq!(backend_ffi::applegpu_ffi_eval(result_id), 0);
    let mut out = [0.0f32; 4];
    assert_eq!(backend_ffi::applegpu_ffi_read_f32(result_id, out.as_mut_ptr(), 4), 0);
    assert_eq!(out, [2.0, 0.0, -3.0, 1.0]);
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p applegpu-core --test backend_ffi_integration`
Expected: All pass (11 tests)

- [ ] **Step 6: Run full Rust suite**

Run: `cargo test -p applegpu-core`
Expected: All pass, no regressions

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/backend_ffi.rs backend_cpp/applegpu_ffi.h crates/core/tests/backend_ffi_integration.rs
git commit -m "feat: add mul, sub, neg FFI functions for PrivateUse1

Extends the extern \"C\" bridge with three more ops. All tested via
alloc → register → op → eval → readback round-trip."
```

---

## Task 3: Register Native C++ Ops (add, sub, mul, matmul, relu, neg)

Wire the Rust FFI ops into PyTorch's C++ dispatcher. Each C++ wrapper:
1. Extracts `tensor_id` from each input's `DataPtr` context
2. Calls `applegpu_ffi_<op>()` to record a graph node (returns output `tensor_id`)
3. Creates an output tensor via `applegpu_empty_strided()` (allocates Metal buffer)
4. Stores the output's `tensor_id` mapping
5. Returns the output tensor

**Critical detail**: The output tensor is created by `empty_strided` which allocates a buffer and registers it in Rust via `register_tensor`. But the FFI op (e.g., `applegpu_ffi_add`) also creates a graph node with its own output tensor_id. We need to connect these: the FFI op should use the pre-allocated buffer from the output tensor. This is already handled by `eval_single_cb` which checks for pre-allocated buffers.

However, there's a sequencing issue: `empty_strided` creates buffer with tensor_id X, `applegpu_ffi_add` records a graph node with output tensor_id Y. We need Y's output to go into X's buffer. The simpler approach: don't call `empty_strided` for the output. Instead, call `applegpu_ffi_add` which returns a tensor_id, then create a PyTorch tensor that points to that tensor_id's buffer AFTER eval. But eval is deferred...

**Revised approach**: The C++ op wrapper should:
1. Extract input tensor_ids
2. Call FFI op → get output tensor_id
3. Create a PyTorch tensor with `empty_strided` using a SEPARATE allocation
4. Store the output tensor_id in the DataPtr context
5. At eval time (triggered by `.cpu()`, `.item()`, etc.), the graph writes into this buffer

Wait — this is exactly the pre-allocated buffer flow from Phase 0. The FFI op records a graph node. `eval_single_cb` checks `self.preallocated` for the output tensor_id and uses that buffer. But the output tensor_id from the FFI op is different from the one allocated by `empty_strided`.

**Correct approach**: Don't use `empty_strided` for op outputs. Instead:
1. Call FFI op → get output tensor_id (graph node recorded)
2. Allocate a Metal buffer via `applegpu_ffi_alloc` with that tensor_id's expected size
3. Store the buffer as pre-allocated for that tensor_id (already done by `alloc`)
4. Create a PyTorch tensor wrapping that buffer pointer + tensor_id in context
5. When PyTorch triggers eval (via copy_, .item(), etc.), the graph writes into the pre-allocated buffer

Actually, `applegpu_ffi_alloc` creates a NEW tensor_id. We need the buffer for the GRAPH NODE's tensor_id. Let me re-examine...

The simplest correct approach: **make the FFI op allocate the output buffer itself** and return both the tensor_id AND the data pointer. The C++ shim wraps the returned pointer in a PyTorch tensor. No separate allocation needed.

**Files:**
- Modify: `crates/core/src/backend_ffi.rs` — add `applegpu_ffi_add_with_output` variant
- Modify: `backend_cpp/applegpu_ffi.h`
- Modify: `backend_cpp/applegpu_backend.cpp` — add C++ op wrappers
- Modify: `python/tests/test_cpp_backend.py`

- [ ] **Step 1: Write failing test — native add produces correct result**

Add to `python/tests/test_cpp_backend.py`:

```python
def test_native_add():
    """Native add op (not CPU fallback) produces correct result."""
    _load()
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device='applegpu')
    b = torch.tensor([10.0, 20.0, 30.0, 40.0], device='applegpu')
    result = a + b
    assert result.device.type == 'applegpu'
    cpu_result = result.cpu()
    expected = torch.tensor([11.0, 22.0, 33.0, 44.0])
    assert torch.allclose(cpu_result, expected)
```

- [ ] **Step 2: Run test — verify it currently uses CPU fallback**

Run: `uv run pytest python/tests/test_cpp_backend.py::test_native_add -v`
Expected: Either passes (via CPU fallback, if Task 1 fixed it) or crashes. Either way, we'll replace the CPU fallback path with native dispatch.

- [ ] **Step 3: Add FFI functions that return output pointer + tensor_id**

In `crates/core/src/backend_ffi.rs`, add new op variants that allocate output:

```rust
/// Record an add op AND allocate the output buffer.
/// Returns output data_ptr (null on failure). Writes tensor_id to *out_id.
#[no_mangle]
pub extern "C" fn applegpu_ffi_add_out(
    a_id: u64,
    b_id: u64,
    out_id: *mut u64,
) -> *mut u8 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();

    let result_id = match crate::ops::add(&mut rt, a_id, b_id) {
        Ok(id) => id,
        Err(e) => { set_error(format!("add failed: {}", e)); return std::ptr::null_mut(); }
    };

    // Get output shape/dtype from the graph node
    let (shape, dtype) = match (rt.shape(result_id), rt.dtype(result_id)) {
        (Ok(s), Ok(d)) => (s, d),
        _ => { set_error("add: cannot determine output shape".into()); return std::ptr::null_mut(); }
    };

    let size = shape.iter().product::<usize>() * dtype.size_bytes();
    let buffer = match rt.pool.acquire(&state.device, size) {
        Ok(b) => b,
        Err(e) => { set_error(format!("add: alloc failed: {}", e)); return std::ptr::null_mut(); }
    };
    let ptr = buffer.contents();
    rt.insert_preallocated_buffer(result_id, buffer);

    unsafe { *out_id = result_id; }
    ptr
}
```

Add similar functions for `applegpu_ffi_mul_out`, `applegpu_ffi_sub_out`, `applegpu_ffi_matmul_out`, `applegpu_ffi_relu_out`, `applegpu_ffi_neg_out`.

To avoid repetition, use a helper macro in Rust:

```rust
macro_rules! ffi_binary_op_out {
    ($fn_name:ident, $op_fn:path, $op_name:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(a_id: u64, b_id: u64, out_id: *mut u64) -> *mut u8 {
            let state = get_state();
            let mut rt = state.runtime.lock().unwrap();
            let result_id = match $op_fn(&mut rt, a_id, b_id) {
                Ok(id) => id,
                Err(e) => { set_error(format!("{} failed: {}", $op_name, e)); return std::ptr::null_mut(); }
            };
            alloc_output(&mut rt, &state.device, result_id, out_id)
        }
    };
}

macro_rules! ffi_unary_op_out {
    ($fn_name:ident, $op_fn:path, $op_name:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(input_id: u64, out_id: *mut u64) -> *mut u8 {
            let state = get_state();
            let mut rt = state.runtime.lock().unwrap();
            let result_id = match $op_fn(&mut rt, input_id) {
                Ok(id) => id,
                Err(e) => { set_error(format!("{} failed: {}", $op_name, e)); return std::ptr::null_mut(); }
            };
            alloc_output(&mut rt, &state.device, result_id, out_id)
        }
    };
}

fn alloc_output(rt: &mut LazyRuntime, device: &Device, result_id: u64, out_id: *mut u64) -> *mut u8 {
    let (shape, dtype) = match (rt.shape(result_id), rt.dtype(result_id)) {
        (Ok(s), Ok(d)) => (s, d),
        _ => { set_error("cannot determine output shape".into()); return std::ptr::null_mut(); }
    };
    let size = shape.iter().product::<usize>() * dtype.size_bytes();
    if size == 0 {
        unsafe { *out_id = result_id; }
        return std::ptr::null_mut();
    }
    let buffer = match rt.pool.acquire(device, size) {
        Ok(b) => b,
        Err(e) => { set_error(format!("alloc output failed: {}", e)); return std::ptr::null_mut(); }
    };
    let ptr = buffer.contents();
    rt.insert_preallocated_buffer(result_id, buffer);
    unsafe { *out_id = result_id; }
    ptr
}

ffi_binary_op_out!(applegpu_ffi_add_out, crate::ops::add, "add");
ffi_binary_op_out!(applegpu_ffi_mul_out, crate::ops::mul, "mul");
ffi_binary_op_out!(applegpu_ffi_sub_out, crate::ops::sub, "sub");
ffi_binary_op_out!(applegpu_ffi_matmul_out, crate::ops::matmul, "matmul");
ffi_unary_op_out!(applegpu_ffi_relu_out, crate::ops::relu, "relu");
ffi_unary_op_out!(applegpu_ffi_neg_out, crate::ops::neg, "neg");
```

Update `backend_cpp/applegpu_ffi.h`:

```c
/* Ops with output allocation (return data_ptr, write tensor_id to *out_id) */
uint8_t* applegpu_ffi_add_out(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_ffi_mul_out(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_ffi_sub_out(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_ffi_matmul_out(uint64_t a_id, uint64_t b_id, uint64_t* out_id);
uint8_t* applegpu_ffi_relu_out(uint64_t input_id, uint64_t* out_id);
uint8_t* applegpu_ffi_neg_out(uint64_t input_id, uint64_t* out_id);
```

- [ ] **Step 4: Run Rust tests**

Run: `cargo test -p applegpu-core`
Expected: All pass

- [ ] **Step 5: Add C++ op wrappers**

In `backend_cpp/applegpu_backend.cpp`, add a helper to create tensors from FFI results, then register native ops:

```cpp
namespace {

// Create a PyTorch tensor from an FFI op result.
// ptr: Metal buffer data pointer, tid: tensor_id, sizes: output shape, dtype: output dtype
at::Tensor wrap_ffi_output(void* ptr, uint64_t tid, c10::IntArrayRef sizes, at::ScalarType dtype) {
    if (ptr == nullptr) {
        const char* err = applegpu_ffi_last_error();
        TORCH_CHECK(false, "applegpu op failed: ", err ? err : "unknown");
    }

    auto strides = c10::contiguous_strides(sizes);
    int64_t nbytes = at::detail::computeStorageNbytes(sizes, strides, at::elementSize(dtype));

    auto* ctx = new TensorContext{tid};
    auto deleter = [](void* ctx_ptr) {
        auto* tc = static_cast<TensorContext*>(ctx_ptr);
        applegpu_ffi_free(tc->tensor_id);
        delete tc;
    };
    c10::DataPtr dptr{ptr, ctx, deleter, c10::Device(c10::DeviceType::PrivateUse1, 0)};

    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(), nbytes,
        std::move(dptr), &global_allocator);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        at::scalarTypeToTypeMeta(dtype));
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    return tensor;
}

} // namespace

// Query output shape from Rust graph node (handles broadcasting correctly).
std::vector<int64_t> query_output_shape(uint64_t tid) {
    uint64_t dims[8];
    uint32_t ndim = 0;
    int32_t rc = applegpu_ffi_shape(tid, dims, &ndim);
    TORCH_CHECK(rc == 0, "applegpu: failed to query output shape");
    std::vector<int64_t> sizes(ndim);
    for (uint32_t i = 0; i < ndim; i++) sizes[i] = static_cast<int64_t>(dims[i]);
    return sizes;
}

at::Tensor applegpu_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    if (alpha.toDouble() != 1.0) {
        // Fall back to CPU for scaled add (SGD optimizer uses alpha=-lr)
        applegpu_ffi_synchronize();
        return at::add(self.cpu(), other.cpu(), alpha).to(self.device());
    }
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_add_out(get_tensor_id(self), get_tensor_id(other), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_mul(const at::Tensor& self, const at::Tensor& other) {
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_mul_out(get_tensor_id(self), get_tensor_id(other), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_sub(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    if (alpha.toDouble() != 1.0) {
        applegpu_ffi_synchronize();
        return at::sub(self.cpu(), other.cpu(), alpha).to(self.device());
    }
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_sub_out(get_tensor_id(self), get_tensor_id(other), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_mm(const at::Tensor& self, const at::Tensor& mat2) {
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_matmul_out(get_tensor_id(self), get_tensor_id(mat2), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_relu(const at::Tensor& self) {
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_relu_out(get_tensor_id(self), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}

at::Tensor applegpu_neg(const at::Tensor& self) {
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_neg_out(get_tensor_id(self), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}
```

Add to `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`:

```cpp
    m.impl("add.Tensor", applegpu_add);
    m.impl("mul.Tensor", applegpu_mul);
    m.impl("sub.Tensor", applegpu_sub);
    m.impl("mm", applegpu_mm);
    m.impl("relu", applegpu_relu);
    m.impl("neg", applegpu_neg);
```

- [ ] **Step 6: Rebuild C++ extension**

Run: `cd backend_cpp && rm -rf build *.so && ARCHFLAGS="-arch arm64" uv run python setup.py build_ext --inplace`
Expected: Builds successfully

- [ ] **Step 7: Run Python tests**

Run: `uv run pytest python/tests/test_cpp_backend.py -v`
Expected: All pass including `test_native_add`

- [ ] **Step 8: Add more native op tests**

Add to `python/tests/test_cpp_backend.py`:

```python
def test_native_mul():
    _load()
    a = torch.tensor([2.0, 3.0, 4.0], device='applegpu')
    b = torch.tensor([10.0, 20.0, 30.0], device='applegpu')
    result = (a * b).cpu()
    assert torch.allclose(result, torch.tensor([20.0, 60.0, 120.0]))

def test_native_matmul():
    _load()
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='applegpu')
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='applegpu')
    result = torch.mm(a, b).cpu()
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(result, expected)

def test_native_relu():
    _load()
    a = torch.tensor([-2.0, 0.0, 3.0, -1.0], device='applegpu')
    result = torch.relu(a).cpu()
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 3.0, 0.0]))
```

- [ ] **Step 9: Run all tests**

Run: `uv run pytest python/tests/test_cpp_backend.py -v`
Expected: All pass (9+ tests)

- [ ] **Step 10: Commit**

```bash
git add crates/core/src/backend_ffi.rs backend_cpp/applegpu_ffi.h backend_cpp/applegpu_backend.cpp python/tests/test_cpp_backend.py crates/core/tests/backend_ffi_integration.rs
git commit -m "feat: native GPU dispatch for add, sub, mul, matmul, relu, neg

C++ op wrappers extract tensor_ids, call Rust FFI to record graph ops,
allocate output Metal buffers, and return PyTorch tensors. Ops are lazy-
recorded and fused at eval points (.cpu(), .item(), synchronize).

This replaces the Python __torch_dispatch__ path (~20µs/op) with C++
dispatch (~200ns/op) for these core MLP ops."
```

---

## Task 4: MLP Benchmark

Create a focused benchmark that runs an MLP (Linear + ReLU stack) on the C++ backend vs CPU.

**Files:**
- Create: `benchmarks/bench_mlp_cpp.py`
- Modify: `Makefile`

- [ ] **Step 1: Create MLP benchmark**

Create `benchmarks/bench_mlp_cpp.py`:

```python
"""MLP benchmark: C++ PrivateUse1 backend vs CPU.

Runs forward + backward pass of a simple MLP and compares wall time.
Uses only ops registered natively: mm, add, relu.

Usage:
    uv run python benchmarks/bench_mlp_cpp.py
    uv run python benchmarks/bench_mlp_cpp.py --hidden 256 --layers 4 --iters 100
"""
import argparse
import time
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden, num_layers, out_features):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def bench(device_name, model, x, y, criterion, n_iters):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    start = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start

    return {
        "device": device_name,
        "total_ms": elapsed * 1000,
        "per_iter_ms": (elapsed / n_iters) * 1000,
        "final_loss": loss.cpu().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--input", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--output", type=int, default=1)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    print(f"MLP Benchmark: batch={args.batch}, input={args.input}, "
          f"hidden={args.hidden}, layers={args.layers}, iters={args.iters}")
    print("=" * 60)

    torch.manual_seed(42)
    x_cpu = torch.randn(args.batch, args.input)
    y_cpu = torch.randn(args.batch, args.output)
    criterion = nn.MSELoss()

    # CPU baseline
    model_cpu = SimpleMLP(args.input, args.hidden, args.layers, args.output)
    result_cpu = bench("cpu", model_cpu, x_cpu, y_cpu, criterion, args.iters)

    # C++ backend
    try:
        from applegpu_runtime.cpp_backend import load_cpp_backend
        load_cpp_backend()

        model_gpu = SimpleMLP(args.input, args.hidden, args.layers, args.output)
        model_gpu = model_gpu.to("applegpu")
        x_gpu = x_cpu.to("applegpu")
        y_gpu = y_cpu.to("applegpu")

        result_gpu = bench("applegpu", model_gpu, x_gpu, y_gpu, criterion, args.iters)
    except Exception as e:
        print(f"C++ backend error: {e}")
        result_gpu = None

    # Results
    print(f"\n{'Device':<15} {'Total (ms)':>12} {'Per-iter (ms)':>14}")
    print("-" * 43)
    print(f"{'CPU':<15} {result_cpu['total_ms']:>12.2f} {result_cpu['per_iter_ms']:>14.2f}")
    if result_gpu:
        print(f"{'applegpu':<15} {result_gpu['total_ms']:>12.2f} {result_gpu['per_iter_ms']:>14.2f}")
        speedup = result_cpu['per_iter_ms'] / result_gpu['per_iter_ms']
        print(f"\nSpeedup: {speedup:.2f}x {'(GPU faster)' if speedup > 1 else '(CPU faster)'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add Makefile target**

Add to `Makefile`:

```makefile
bench-mlp-cpp: build-cpp-backend
	uv run python benchmarks/bench_mlp_cpp.py --hidden 128 --layers 3 --iters 50
```

- [ ] **Step 3: Run benchmark**

Run: `make bench-mlp-cpp`
Expected: Numbers print. Speedup direction is the key data point.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_mlp_cpp.py Makefile
git commit -m "bench: add MLP training benchmark for C++ backend

Compares CPU vs PrivateUse1 C++ backend on a simple MLP (Linear+ReLU).
Uses only natively registered ops (mm, add, relu)."
```

---

## End of Phase 3a

After completing these 4 tasks:

1. **CPU fallback works** — ALL PyTorch ops usable (via CPU, slow but correct)
2. **6 native ops** — add, sub, mul, matmul, relu, neg dispatch directly to Rust graph engine
3. **MLP benchmark** — quantified speedup vs CPU for the simplest training case

**Success criteria from spec**: GPU faster than CPU for MLP training with batch_size >= 16, hidden_size >= 256. If not met, the benchmark data tells us where the remaining bottleneck is (graph eval? Metal dispatch? Python overhead elsewhere?).

## Explicitly Deferred to Phase 3b+

- **Backward ops** (threshold_backward, etc.) — needed for training but handled by autograd + CPU fallback initially
- **softmax, layer_norm, embedding** — needed for Transformer, deferred to Phase 3c
- **conv2d, batch_norm** — needed for CNN, deferred to Phase 3b
- **View/stride handling** — CPU fallback handles it for now
