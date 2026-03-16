# Zero-Copy Tensor Transfers Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add zero-copy tensor transfer from numpy/torch to Metal GPU via `makeBuffer(bytesNoCopy:)`, with explicit opt-in APIs and page-aligned allocation helpers.

**Architecture:** Bottom-up: Swift buffer base class → Rust BufferKind + FFI → Python APIs. Existing copy-based `from_numpy`/`from_torch` unchanged.

**Tech Stack:** Swift/Metal (`makeBuffer(bytesNoCopy:)`), Rust (BufferKind enum, FFI), PyO3 (new Python functions), `pyo3-numpy` crate for array construction, `libc` for `posix_memalign`

**Spec:** `docs/superpowers/specs/2026-03-16-zero-copy-transfers-design.md`

---

## Chunk 1: Swift + Rust Foundation

### Task 1: Swift GPUBufferBase class hierarchy

**Files:**
- Modify: `swift/Sources/AppleGPUBridge/buffer.swift`
- Modify: `swift/Sources/AppleGPUBridge/include/bridge.h`

- [ ] **Step 1: Refactor GPUBuffer to inherit from GPUBufferBase**

In `buffer.swift`, replace the existing `final class GPUBuffer` with:

```swift
/// Base class for all GPU buffers — provides the MTLBuffer handle.
class GPUBufferBase {
    let buffer: MTLBuffer
    init(buffer: MTLBuffer) { self.buffer = buffer }
}

/// Owned buffer — Metal allocated and owns the memory.
class GPUBuffer: GPUBufferBase {
    init?(device: MTLDevice, length: Int) {
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else { return nil }
        super.init(buffer: buffer)
    }

    init?(device: MTLDevice, bytes: UnsafeRawPointer, length: Int) {
        guard let buffer = device.makeBuffer(bytes: bytes, length: length, options: .storageModeShared) else { return nil }
        super.init(buffer: buffer)
    }
}

/// Borrowed buffer — Metal references external memory. Deallocator fires when released.
class GPUBufferNoCopy: GPUBufferBase {
    // No additional state — Metal's deallocator block handles cleanup
}
```

- [ ] **Step 2: Update all Unmanaged casts to use GPUBufferBase**

Update `gpuBridgeDestroyBuffer`, `gpuBridgeBufferContents`, `gpuBridgeBufferLength` to cast via `Unmanaged<GPUBufferBase>` instead of `Unmanaged<GPUBuffer>`. Also update `gpuBridgeCreateBuffer` and `gpuBridgeCreateBufferWithData` to retain as `GPUBufferBase`.

- [ ] **Step 3: Build Swift to verify**

Run: `cd swift && swift build`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Run Swift tests to verify existing behavior**

Run: `cd swift && swift test`
Expected: All PASS

- [ ] **Step 5: Add gpuBridgeCreateBufferNoCopy**

```swift
@_cdecl("gpu_bridge_create_buffer_no_copy")
public func gpuBridgeCreateBufferNoCopy(
    _ devicePtr: UnsafeRawPointer?,
    _ dataPtr: UnsafeMutableRawPointer?,
    _ sizeBytes: UInt64,
    _ deallocator: (@convention(c) (UnsafeMutableRawPointer?, UInt64, UnsafeMutableRawPointer?) -> Void)?,
    _ deallocatorContext: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let devicePtr = devicePtr, let dataPtr = dataPtr else { return nil }
    let gpuDevice = getGPUDevice(from: devicePtr)
    let length = Int(sizeBytes)

    guard let buffer = gpuDevice.device.makeBuffer(
        bytesNoCopy: dataPtr,
        length: length,
        options: .storageModeShared,
        deallocator: { ptr, len in
            deallocator?(ptr, UInt64(len), deallocatorContext)
        }
    ) else { return nil }

    let buf = GPUBufferNoCopy(buffer: buffer)
    return Unmanaged.passRetained(buf).toOpaque()
}
```

- [ ] **Step 6: Add C header declaration**

In `bridge.h`:

```c
typedef void (*GPUDeallocator)(void* ptr, uint64_t len, void* context);
GPUBufferHandle* gpu_bridge_create_buffer_no_copy(
    const GPUDeviceHandle* device,
    void* data,
    uint64_t size_bytes,
    GPUDeallocator deallocator,
    void* deallocator_context
);
```

- [ ] **Step 7: Build and test**

Run: `cd swift && swift build && swift test`
Expected: BUILD SUCCEEDED, All tests PASS

- [ ] **Step 8: Commit**

```bash
git add swift/Sources/AppleGPUBridge/buffer.swift swift/Sources/AppleGPUBridge/include/bridge.h
git commit -m "feat: GPUBufferBase hierarchy and makeBuffer(bytesNoCopy:) FFI"
```

### Task 2: Rust BufferKind + error types + FFI

**Files:**
- Modify: `crates/core/src/buffer.rs`
- Modify: `crates/core/src/error.rs`
- Modify: `crates/core/src/ffi.rs`
- Modify: `crates/core/src/pool.rs`
- Test: `crates/core/src/buffer.rs` (inline tests)

- [ ] **Step 1: Add ImmutableBuffer error variant**

In `error.rs`, add to `GpuError` enum:

```rust
ImmutableBuffer(u64),
```

And in the `Display` impl:

```rust
GpuError::ImmutableBuffer(id) => write!(f, "Tensor {} has a borrowed (immutable) buffer and cannot be used as output", id),
```

- [ ] **Step 2: Add BufferKind enum and update Buffer struct**

In `buffer.rs`:

```rust
use std::ffi::c_void;

#[derive(Debug)]
pub enum BufferKind {
    Owned,
    Borrowed { _pinned_object: *mut c_void },
}

unsafe impl Send for BufferKind {}
unsafe impl Sync for BufferKind {}

impl BufferKind {
    pub fn is_borrowed(&self) -> bool {
        matches!(self, BufferKind::Borrowed { .. })
    }
}
```

Add `kind: BufferKind` field to `Buffer` struct. Set `kind: BufferKind::Owned` in existing `new()` and `from_bytes()` constructors.

- [ ] **Step 3: Add FFI declaration for no-copy buffer**

In `ffi.rs`:

```rust
pub type GPUDeallocator = Option<unsafe extern "C" fn(*mut c_void, u64, *mut c_void)>;

extern "C" {
    pub fn gpu_bridge_create_buffer_no_copy(
        device: *const GPUDeviceHandle,
        data: *mut c_void,
        size_bytes: u64,
        deallocator: GPUDeallocator,
        deallocator_context: *mut c_void,
    ) -> *mut GPUBufferHandle;
}
```

- [ ] **Step 4: Add Buffer::from_ptr_no_copy**

```rust
impl Buffer {
    pub fn from_ptr_no_copy(
        device: &Device,
        ptr: *mut u8,
        len: usize,
        pinned_object: *mut c_void,
    ) -> Result<Self> {
        let handle = unsafe {
            ffi::gpu_bridge_create_buffer_no_copy(
                device.raw_handle(),
                ptr as *mut c_void,
                len as u64,
                Some(buffer_deallocator),
                pinned_object,
            )
        };
        if handle.is_null() {
            Err(GpuError::BufferAllocationFailed(len))
        } else {
            Ok(Buffer { handle, len, kind: BufferKind::Borrowed { _pinned_object: pinned_object } })
        }
    }
}

unsafe extern "C" fn buffer_deallocator(_ptr: *mut c_void, _len: u64, context: *mut c_void) {
    if !context.is_null() {
        pyo3::Python::with_gil(|_py| {
            pyo3::ffi::Py_DecRef(context as *mut pyo3::ffi::PyObject);
        });
    }
}
```

**Note:** The `buffer_deallocator` needs `pyo3` as a dependency of `applegpu-core`. If that's not desirable, move the deallocator to `crates/python/` and pass it as a function pointer. Check `crates/core/Cargo.toml` — if pyo3 is not a dep, use a callback registration pattern instead.

- [ ] **Step 5: Update BufferPool to reject Borrowed buffers**

In `pool.rs`, at the top of `release()`:

```rust
pub fn release(&mut self, buffer: Buffer) {
    if buffer.kind.is_borrowed() {
        return; // Borrowed buffers cannot be pooled — just drop them
    }
    // ... existing power-of-two logic ...
}
```

- [ ] **Step 6: Build Rust core**

Run: `cargo build -p applegpu-core`
Expected: BUILD SUCCEEDED (or note if pyo3 dep issue needs resolution)

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/buffer.rs crates/core/src/error.rs crates/core/src/ffi.rs crates/core/src/pool.rs
git commit -m "feat: BufferKind enum, no-copy buffer FFI, pool safety"
```

### Task 3: Immutability enforcement in op dispatch

**Files:**
- Modify: `crates/core/src/lazy.rs`

- [ ] **Step 1: Add immutability check at top of execute_node_nb**

In `lazy.rs`, at the start of `execute_node_nb()`:

```rust
// Reject borrowed (immutable) buffers as output targets
if out.buffer.kind.is_borrowed() {
    return Err(GpuError::ImmutableBuffer(out.meta.id));
}
```

- [ ] **Step 2: Also add check in execute_node (blocking variant)**

Same check in the synchronous `execute_node()` method.

- [ ] **Step 3: Build and run Rust tests**

Run: `cargo test -p applegpu-core`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/lazy.rs
git commit -m "feat: immutability enforcement for borrowed buffers in op dispatch"
```

---

## Chunk 2: Python APIs

### Task 4: Backend trait extension + MetalBackend implementation

**Files:**
- Modify: `crates/python/src/backend.rs`
- Modify: `crates/python/src/metal_backend.rs`

- [ ] **Step 1: Add tensor_from_ptr_no_copy to Backend trait**

```rust
fn tensor_from_ptr_no_copy(
    &self, ptr: *mut u8, len: usize, shape: Vec<usize>,
    dtype: BackendDType, release_context: *mut std::ffi::c_void,
) -> BackendResult<u64> {
    // Default: fall back to copy
    let data = unsafe { std::slice::from_raw_parts(ptr, len) };
    let result = self.tensor_from_data(data, shape, dtype);
    if !release_context.is_null() {
        unsafe {
            pyo3::Python::with_gil(|_py| {
                pyo3::ffi::Py_DecRef(release_context as *mut pyo3::ffi::PyObject);
            });
        }
    }
    result
}
```

- [ ] **Step 2: Override in MetalBackend**

```rust
fn tensor_from_ptr_no_copy(
    &self, ptr: *mut u8, len: usize, shape: Vec<usize>,
    dtype: BackendDType, release_context: *mut std::ffi::c_void,
) -> BackendResult<u64> {
    let dt = map_dtype(dtype);
    let runtime = get_device_runtime()?;
    let buffer = applegpu_core::buffer::Buffer::from_ptr_no_copy(
        &runtime.device, ptr, len, release_context,
    ).map_err(|e| e.to_string())?;
    let tensor = applegpu_core::tensor::Tensor::from_raw(
        applegpu_core::tensor::next_tensor_id(),
        shape, dt, buffer,
    );
    let mut rt = self.runtime.lock().unwrap();
    // size=0 for memory accounting (memory belongs to Python)
    rt.scheduler.allocate_tensor(
        applegpu_core::scheduler::ContainerId::DEFAULT,
        tensor.meta.id, 0,
    ).map_err(|e| e.to_string())?;
    let id = tensor.meta.id;
    rt.tensors.insert(id, tensor);
    Ok(id)
}
```

- [ ] **Step 3: Build**

Run: `cargo build -p applegpu-core && uv run maturin develop`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add crates/python/src/backend.rs crates/python/src/metal_backend.rs
git commit -m "feat: tensor_from_ptr_no_copy in Backend trait + MetalBackend"
```

### Task 5: Python from_numpy_shared, from_torch_shared, aligned_numpy

**Files:**
- Modify: `crates/python/src/lib.rs`
- Modify: `python/applegpu_runtime/__init__.py`
- Test: `python/tests/test_zero_copy.py` (new)

- [ ] **Step 1: Add system_page_size helper**

In `lib.rs`:

```rust
use once_cell::sync::Lazy;
static PAGE_SIZE: Lazy<usize> = Lazy::new(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize });
fn system_page_size() -> usize { *PAGE_SIZE }
```

Add `libc` to `crates/python/Cargo.toml` dependencies if not already present.

- [ ] **Step 2: Implement from_numpy_shared**

Add the function from spec Section 7 to `lib.rs`. Register it in the PyO3 module.

- [ ] **Step 3: Implement from_torch_shared**

Add the function from spec Section 7 (torch variant) to `lib.rs`. Register it in the PyO3 module.

- [ ] **Step 4: Implement aligned_numpy**

Add the function from spec Section 3. Use `posix_memalign` + PyCapsule pattern. Register in module.

- [ ] **Step 5: Export from __init__.py**

Add `from_numpy_shared`, `from_torch_shared`, `aligned_numpy` to `__init__.py` exports.

- [ ] **Step 6: Build**

Run: `uv run maturin develop`
Expected: BUILD SUCCEEDED

- [ ] **Step 7: Write tests**

Create `python/tests/test_zero_copy.py`:

```python
import numpy as np
import applegpu_runtime as gpu
import pytest
import sys

gpu.init_backend()

PAGE_SIZE = 16384  # Apple Silicon

def test_aligned_numpy_creates_page_aligned_array():
    arr = gpu.aligned_numpy(shape=(1024, 1024), dtype="float32")
    assert arr.shape == (1024, 1024)
    assert arr.dtype == np.float32
    assert arr.ctypes.data % PAGE_SIZE == 0
    nbytes = arr.size * arr.itemsize
    assert nbytes % PAGE_SIZE == 0

def test_from_numpy_shared_basic():
    arr = gpu.aligned_numpy(shape=(1024,), dtype="float32")
    arr[:] = np.arange(1024, dtype=np.float32)
    t = gpu.from_numpy_shared(arr)
    result = t.to_list()
    assert result[:5] == [0.0, 1.0, 2.0, 3.0, 4.0]

def test_from_numpy_shared_rejects_misaligned():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="page-aligned"):
        gpu.from_numpy_shared(arr)

def test_from_numpy_shared_rejects_non_page_length():
    # Even if pointer is aligned, length must be page multiple
    arr = gpu.aligned_numpy(shape=(100,), dtype="float32")
    # 100 * 4 = 400 bytes, not a page multiple
    with pytest.raises(ValueError, match="page size"):
        gpu.from_numpy_shared(arr)

def test_shared_tensor_as_input():
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    arr[:] = 2.0
    t = gpu.from_numpy_shared(arr)
    result = t + t  # shared tensor used as input
    result.eval()
    vals = result.to_list()
    assert vals[0] == 4.0

def test_shared_tensor_sees_mutations():
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    arr[:] = 1.0
    t = gpu.from_numpy_shared(arr)
    arr[:] = 99.0  # mutate source
    vals = t.to_list()
    assert vals[0] == 99.0  # shared memory — sees mutation

def test_shared_tensor_refcount():
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    initial_refcount = sys.getrefcount(arr)
    t = gpu.from_numpy_shared(arr)
    # Refcount should have increased by 1 (Py_IncRef)
    assert sys.getrefcount(arr) == initial_refcount + 1
    gpu.destroy(t)
    # After destroy, refcount should return to original
    # (Metal deallocator fires Py_DecRef)
    assert sys.getrefcount(arr) == initial_refcount

def test_pool_not_affected_by_shared():
    stats_before = gpu.pool_stats()
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    t = gpu.from_numpy_shared(arr)
    gpu.destroy(t)
    stats_after = gpu.pool_stats()
    # Pool should not have gained a buffer from the shared tensor
    assert stats_after["pooled_bytes"] == stats_before["pooled_bytes"]
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest python/tests/test_zero_copy.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add crates/python/src/lib.rs crates/python/Cargo.toml python/applegpu_runtime/__init__.py python/tests/test_zero_copy.py
git commit -m "feat: from_numpy_shared, from_torch_shared, aligned_numpy for zero-copy transfers"
```

### Task 6: Final validation

- [ ] **Step 1: Run complete test suite**

Run: `make test`
Expected: All Rust + Swift + Python tests PASS

- [ ] **Step 2: Run existing from_numpy/from_torch tests to verify no regressions**

Run: `uv run pytest python/tests/ -k "numpy or torch" -v`
Expected: All PASS

- [ ] **Step 3: Commit milestone**

```bash
git commit --allow-empty -m "milestone: zero-copy tensor transfers complete"
```
