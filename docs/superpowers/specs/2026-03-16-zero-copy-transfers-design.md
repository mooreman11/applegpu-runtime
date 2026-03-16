# Framework Improvements: Zero-Copy Tensor Transfers

**Date:** 2026-03-16
**Status:** Draft
**Scope:** Metal backend only (socket backend always copies)

## Problem

Tensor creation from numpy/torch currently copies data into a new Metal buffer:

```
Python numpy.ndarray / torch.tensor
    → data_ptr extraction (fast, ~0ns)
    → Rust unsafe slice (fast, ~0ns)
    → Swift makeBuffer(bytes:length:options:.storageModeShared) ← COPIES data
    → MTLBuffer with separate memory
```

For large tensors (model weights, large batches), this copy is the dominant cost. v0.7.0 eliminated the Python-side copy (385x faster) but the Metal-side copy remains.

## Current Architecture

**Python → Rust** (`crates/python/src/lib.rs:404-483`):
- `from_numpy`: extracts `ctypes.data` pointer, creates `&[u8]` slice, calls `BACKEND.tensor_from_data()`
- `from_torch`: calls `.detach().cpu().contiguous()`, extracts `data_ptr()`, same path
- Both already use direct pointer access (v0.7.0 optimization)

**Rust → Swift** (`crates/core/src/buffer.rs:31-48`):
- `Buffer::from_bytes()` calls `gpu_bridge_create_buffer_with_data(device, data_ptr, len)`
- Returns opaque `GPUBufferHandle`

**Swift** (`swift/Sources/AppleGPUBridge/buffer.swift:13-16`):
- `device.makeBuffer(bytes: ptr, length: len, options: .storageModeShared)` — **copies bytes**

**Buffer abstraction** (`buffer.rs`):
- `Buffer { handle, len }` — no distinction between owned and borrowed memory
- `Drop` calls `gpu_bridge_destroy_buffer` which releases the Swift `Unmanaged<GPUBuffer>`
- `BufferPool` recycles buffers by power-of-two size class

## Design

### 1. New Explicit APIs (Not Transparent)

Zero-copy has different semantics than copy — the GPU tensor and the source array share memory. This must be opt-in, not a silent behavior change.

**Python API:**

```python
# Existing (unchanged, always copies)
t = gpu.from_numpy(arr)       # copy semantics: arr mutations don't affect t
t = gpu.from_torch(tensor)    # copy semantics

# New: explicit shared memory
t = gpu.from_numpy_shared(arr)   # zero-copy: t and arr share memory
t = gpu.from_torch_shared(tensor) # zero-copy: t and tensor share memory

# New: allocate page-aligned numpy array suitable for zero-copy
arr = gpu.aligned_numpy(shape=(1024, 1024), dtype="float32")
# arr.ctypes.data is guaranteed page-aligned, arr.nbytes is page-multiple
```

**Semantics of shared tensors:**
- GPU reads the same memory as the numpy/torch array — no copy
- CPU mutations to the source array ARE visible to the GPU tensor
- The user must not mutate the source while GPU ops are in flight (documented contract)
- Shared tensors are **immutable from the GPU side** — they cannot be used as output buffers for ops
- The source array's refcount is incremented; it will not be GC'd while the tensor exists

**Error conditions:**
- `ValueError` (Python `PyValueError`) if source memory is not page-aligned (16KB on Apple Silicon)
- `ValueError` if byte length is not a page-size multiple
- `ValueError` if array is not C-contiguous, or torch tensor is not CPU/contiguous/zero-offset
- Users can catch this and fall back to the copy path, or use `gpu.aligned_numpy()`
- Rust-level: `GpuError::ImmutableBuffer(u64)` when a borrowed buffer is used as op output

**Memory accounting:**
- Shared tensors count toward `tensor_count` limits (they occupy a Metal buffer handle)
- Shared tensors do NOT count toward `memory_usage` limits (the memory belongs to numpy/torch)
- `Scheduler::allocate_tensor` is called with `size = 0` for shared tensors

**Read behavior on shared tensors:**
- `to_list()`, `to_numpy()`, `read_bytes()` read from the shared memory (the numpy/torch array's data)
- If the user mutated the source array between creation and read, the mutated data is returned
- This is documented and expected — "shared" means shared

### 2. Page Alignment Requirements

`MTLDevice.makeBuffer(bytesNoCopy:length:options:deallocatorBlock:)` requires:
- Pointer must be aligned to the system page size
- Length must be a multiple of the system page size
- Apple Silicon page size: **16384 bytes (16KB)**, not 4KB

**Runtime detection:**
```rust
fn system_page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}
```

**Practical implications:**
- A 1024x1024 float32 matrix = 4MB → page-aligned length ✓ (4MB / 16KB = 256 pages)
- A 100-element float32 vector = 400 bytes → NOT a page multiple ✗
- Zero-copy is a **large tensor optimization**. Minimum useful size is ~16KB (one page).
- PyTorch's default CPU allocator uses 64-byte alignment (not page-aligned). Most torch tensors will fail the alignment check. Users should use `gpu.aligned_numpy()` for predictable zero-copy.

### 3. `gpu.aligned_numpy()` Helper

Allocates a numpy array with page-aligned memory and page-rounded size:

```python
def aligned_numpy(shape, dtype="float32"):
    """Allocate a page-aligned numpy array suitable for gpu.from_numpy_shared()."""
    # Implementation in Rust for correct alignment
```

**Rust implementation:**
```rust
use once_cell::sync::Lazy;

static PAGE_SIZE: Lazy<usize> = Lazy::new(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize });

fn system_page_size() -> usize { *PAGE_SIZE }

/// PyCapsule destructor — called when numpy array is GC'd, frees the aligned allocation.
unsafe extern "C" fn aligned_buffer_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, std::ptr::null());
    if !ptr.is_null() {
        libc::free(ptr);
    }
}

#[pyfunction]
fn aligned_numpy(py: Python<'_>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyObject> {
    let page_size = system_page_size();
    let dt = BackendDType::from_name(dtype.unwrap_or("float32"))
        .ok_or_else(|| PyValueError::new_err("Unsupported dtype"))?;
    let numel: usize = shape.iter().product();
    let nbytes = numel * dt.size_bytes();

    if nbytes == 0 {
        // Empty array — just return np.empty(shape, dtype)
        let np = py.import_bound("numpy")?;
        return Ok(np.call_method1("empty", (shape, dtype.unwrap_or("float32")))?.into());
    }

    // Round up to page boundary
    let aligned_size = (nbytes + page_size - 1) & !(page_size - 1);

    // Allocate page-aligned memory
    let mut ptr: *mut libc::c_void = std::ptr::null_mut();
    let ret = unsafe { libc::posix_memalign(&mut ptr, page_size, aligned_size) };
    if ret != 0 {
        return Err(PyValueError::new_err("Failed to allocate aligned memory"));
    }

    // Zero-initialize
    unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, aligned_size) };

    // Wrap the pointer in a PyCapsule for correct lifetime management.
    // The capsule destructor calls libc::free when the capsule is GC'd.
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            ptr,
            std::ptr::null(),       // no name
            Some(aligned_buffer_destructor),
        );
        if cap.is_null() {
            libc::free(ptr);
            return Err(PyRuntimeError::new_err("Failed to create PyCapsule"));
        }
        PyObject::from_owned_ptr(py, cap)
    };

    // Create numpy array viewing the capsule's memory.
    // numpy holds a reference to the capsule (via the base attribute),
    // so the capsule (and its memory) lives as long as the array.
    let np = py.import_bound("numpy")?;
    let np_dtype = np.call_method1("dtype", (dtype.unwrap_or("float32"),))?;
    let ctypes = py.import_bound("ctypes")?;
    let c_buf = ctypes.call_method1(
        "cast",
        (ptr as usize, ctypes.getattr("POINTER")?.call1((ctypes.getattr("c_ubyte")?,))?),
    )?;
    let arr = np.call_method1(
        "frombuffer",
        pyo3::types::PyDict::from_sequence_bound(py, &[
            ("buffer", c_buf.as_any()),
            ("dtype", np_dtype.as_any()),
            ("count", numel.into_py(py).bind(py)),
        ])?
    )?;
    let arr = arr.call_method1("reshape", (shape,))?;

    // Attach capsule as the array's base so the memory is freed when array is GC'd
    unsafe {
        pyo3::ffi::PyArray_SetBaseObject(
            arr.as_ptr() as *mut _,
            capsule.into_ptr(),
        );
    }

    Ok(arr.into())
}
```

**Key design: PyCapsule-based ownership.** The aligned memory is owned by a PyCapsule with a destructor that calls `libc::free()`. The numpy array's `base` attribute references the capsule, so the memory lives exactly as long as the array. No leak, no double-free.

### 4. BufferKind Enum

Add explicit ownership tracking to `Buffer`:

```rust
pub enum BufferKind {
    /// Buffer owns its Metal memory. Can be used as op output. Poolable.
    Owned,
    /// Buffer borrows external memory (numpy/torch). Read-only for GPU ops. Not poolable.
    /// The release_context is a PyObject* that was Py_IncRef'd at creation time.
    /// It will be Py_DecRef'd (with GIL) when the Metal buffer deallocator fires.
    Borrowed {
        /// Raw PyObject* pointer to the pinned Python object (numpy array / torch tensor).
        /// This is NOT released in Buffer::Drop — Metal's deallocator handles it.
        _pinned_object: *mut c_void,
    },
}

// Safety: _pinned_object is a reference-counted PyObject*. The pointer itself is never
// dereferenced from Rust — it's only passed to the Swift deallocator which calls Py_DecRef
// with the GIL. Buffer::Drop destroys the Metal handle, which asynchronously triggers the
// deallocator. The worst case is Py_DecRef firing from a non-Python thread, which is handled
// by Python::with_gil() in the deallocator callback.
//
// GIL safety during interpreter shutdown: if Python is finalizing, Python::with_gil() will
// panic. This is acceptable because all GPU resources should be cleaned up before shutdown.
// Users should call gpu.destroy() or let GpuTensor.__del__ clean up during normal GC.
unsafe impl Send for BufferKind {}
unsafe impl Sync for BufferKind {}

pub struct Buffer {
    handle: *mut ffi::GPUBufferHandle,
    len: usize,
    kind: BufferKind,
}
```

**Invariants:**
- `BufferPool::release()` checks `buffer.kind` — rejects `Borrowed` buffers explicitly (not just by power-of-two heuristic)
- Op dispatch functions check `out.buffer.kind` — refuse to use `Borrowed` buffers as output
- `Buffer::drop()` calls `gpu_bridge_destroy_buffer` for both kinds. For `Borrowed`, this triggers Metal's deallocator which calls `Py_DecRef` with the GIL. The Rust-side `_pinned_object` field is informational only (used for debugging/logging).

### 5. Swift FFI: No-Copy Buffer Creation

**Swift type hierarchy:** Both `GPUBuffer` (owned) and `GPUBufferNoCopy` (borrowed) inherit from a common base class, so the existing `gpu_bridge_destroy_buffer`, `gpu_bridge_buffer_contents`, and `gpu_bridge_buffer_length` functions work for both types via the shared base.

```swift
/// Base class for all GPU buffers — provides the MTLBuffer handle.
class GPUBufferBase {
    let buffer: MTLBuffer
    init(buffer: MTLBuffer) { self.buffer = buffer }
}

/// Owned buffer — Metal allocated and owns the memory.
final class GPUBuffer: GPUBufferBase {
    init?(device: MTLDevice, length: Int) {
        guard let buf = device.makeBuffer(length: length, options: .storageModeShared) else { return nil }
        super.init(buffer: buf)
    }
    init?(device: MTLDevice, bytes: UnsafeRawPointer, length: Int) {
        guard let buf = device.makeBuffer(bytes: bytes, length: length, options: .storageModeShared) else { return nil }
        super.init(buffer: buf)
    }
}

/// Borrowed buffer — Metal references external memory. Deallocator fires when released.
final class GPUBufferNoCopy: GPUBufferBase {
    // No additional state needed — Metal's deallocator block handles cleanup
}

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

    // Capture the C deallocator + context. Metal calls this when the buffer is released.
    guard let buffer = gpuDevice.device.makeBuffer(
        bytesNoCopy: dataPtr,
        length: length,
        options: .storageModeShared,
        deallocator: { ptr, len in
            // Forward Metal's 2-arg deallocator to our 3-arg C callback
            deallocator?(ptr, UInt64(len), deallocatorContext)
        }
    ) else { return nil }

    let buf = GPUBufferNoCopy(buffer: buffer)
    return Unmanaged.passRetained(buf).toOpaque()
}
```

The existing `gpu_bridge_destroy_buffer` works unchanged because both `GPUBuffer` and `GPUBufferNoCopy` are subclasses of `GPUBufferBase`:
```swift
// Existing function — works for both types via Unmanaged<GPUBufferBase>
@_cdecl("gpu_bridge_destroy_buffer")
public func gpuBridgeDestroyBuffer(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    Unmanaged<GPUBufferBase>.fromOpaque(ptr).release()
}
```

**C header addition** (`bridge.h`):
```c
/// Deallocator callback signature: (data_ptr, byte_length, user_context)
typedef void (*GPUDeallocator)(void* ptr, uint64_t len, void* context);

GPUBufferHandle* gpu_bridge_create_buffer_no_copy(
    const GPUDeviceHandle* device,
    void* data,
    uint64_t size_bytes,
    GPUDeallocator deallocator,
    void* deallocator_context
);
```

### 6. Rust No-Copy Buffer

```rust
impl Buffer {
    pub fn from_ptr_no_copy(
        device: &Device,
        ptr: *mut u8,
        len: usize,
        pinned_object: *mut c_void,  // PyObject* to pin
    ) -> Result<Self> {
        let handle = unsafe {
            ffi::gpu_bridge_create_buffer_no_copy(
                device.raw_handle(),
                ptr as *mut _,
                len as u64,
                Some(buffer_deallocator),  // 3-arg C callback
                pinned_object,             // passed as deallocator_context
            )
        };
        if handle.is_null() {
            Err(GpuError::BufferAllocationFailed(len))
        } else {
            Ok(Buffer {
                handle,
                len,
                kind: BufferKind::Borrowed { _pinned_object: pinned_object },
            })
        }
    }
}

/// C-compatible deallocator matching GPUDeallocator typedef:
///   void (*)(void* ptr, uint64_t len, void* context)
///
/// Called by Metal (via Swift) when the MTLBuffer is released.
/// `context` is a PyObject* that was Py_IncRef'd during from_numpy_shared.
/// Must acquire the GIL before calling Py_DecRef.
unsafe extern "C" fn buffer_deallocator(
    _ptr: *mut c_void,    // data pointer (unused — Metal already released it)
    _len: u64,            // byte length (unused)
    context: *mut c_void, // PyObject* to release
) {
    if !context.is_null() {
        pyo3::Python::with_gil(|_py| {
            pyo3::ffi::Py_DecRef(context as *mut pyo3::ffi::PyObject);
        });
    }
}
```

### 7. Python from_numpy_shared / from_torch_shared

```rust
#[pyfunction]
fn from_numpy_shared(_py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<GpuTensor> {
    let np_dtype_name: String = arr.getattr("dtype")?.getattr("name")?.extract()?;
    let dtype = BackendDType::from_name(&np_dtype_name)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported dtype: {}", np_dtype_name)))?;

    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let numel: usize = shape.iter().product();
    let nbytes = numel * dtype.size_bytes();
    let page_size = system_page_size();

    // Validate C-contiguous
    let is_c_contiguous: bool = arr.getattr("flags")?
        .get_item("C_CONTIGUOUS")?.extract()?;
    if !is_c_contiguous {
        return Err(PyValueError::new_err("Array must be C-contiguous for shared transfer"));
    }

    let data_ptr: usize = arr.getattr("ctypes")?.getattr("data")?.extract()?;

    // Validate page alignment
    if data_ptr % page_size != 0 {
        return Err(PyValueError::new_err(format!(
            "Array data pointer {:#x} is not page-aligned (page size: {} bytes). \
             Use gpu.aligned_numpy() or gpu.from_numpy() (copy) instead.",
            data_ptr, page_size
        )));
    }
    if nbytes % page_size != 0 {
        return Err(PyValueError::new_err(format!(
            "Array byte size {} is not a multiple of page size {} bytes. \
             Use gpu.aligned_numpy() or gpu.from_numpy() (copy) instead.",
            nbytes, page_size
        )));
    }

    // Pin the numpy array: increment refcount so GC can't collect it
    // Safety: we hold the GIL, arr is a valid Python object
    let py_obj_ptr = arr.as_ptr();
    unsafe { pyo3::ffi::Py_IncRef(py_obj_ptr) };

    // Create no-copy Metal buffer
    let id = BACKEND.tensor_from_ptr_no_copy(
        data_ptr as *mut u8,
        nbytes,
        shape,
        dtype,
        py_obj_ptr as *mut std::ffi::c_void,
    ).map_err(|e| PyRuntimeError::new_err(e))?;

    Ok(GpuTensor { id })
}
```

`from_torch_shared` follows the same pattern with explicit validation (no preparation copies):

```rust
#[pyfunction]
fn from_torch_shared(py: Python<'_>, tensor: &Bound<'_, PyAny>) -> PyResult<GpuTensor> {
    // Validate already on CPU (do NOT call .cpu() — that copies)
    let device_type: String = tensor.getattr("device")?.getattr("type")?.extract()?;
    if device_type != "cpu" {
        return Err(PyValueError::new_err(
            "Tensor must be on CPU for shared transfer. Use tensor.cpu() first, or gpu.from_torch() (copy)."
        ));
    }

    // Validate already contiguous (do NOT call .contiguous() — that copies)
    let is_contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
    if !is_contiguous {
        return Err(PyValueError::new_err(
            "Tensor must be contiguous for shared transfer. Use tensor.contiguous() first, or gpu.from_torch() (copy)."
        ));
    }

    // Validate storage_offset == 0 (views into larger storage may have misaligned data_ptr)
    let storage_offset: usize = tensor.call_method0("storage_offset")?.extract()?;
    if storage_offset != 0 {
        return Err(PyValueError::new_err(
            "Tensor must not be a view with storage_offset != 0. Use tensor.clone() first, or gpu.from_torch() (copy)."
        ));
    }

    // Extract dtype, shape, data pointer
    let torch = py.import_bound("torch")?;
    let tensor_dtype = tensor.getattr("dtype")?;
    // ... same dtype mapping as from_torch ...
    let dtype = BackendDType::from_name(dtype_str)?;
    let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
    let numel: usize = tensor.call_method0("numel")?.extract()?;
    let element_size: usize = tensor.call_method0("element_size")?.extract()?;
    let nbytes = numel * element_size;
    let page_size = system_page_size();

    let data_ptr: usize = tensor.call_method0("data_ptr")?.extract()?;

    // Validate page alignment
    if data_ptr % page_size != 0 {
        return Err(PyValueError::new_err(format!(
            "Tensor data_ptr {:#x} is not page-aligned (page size: {} bytes). \
             PyTorch uses 64-byte alignment by default. Use gpu.from_torch() (copy) instead.",
            data_ptr, page_size
        )));
    }
    if nbytes % page_size != 0 {
        return Err(PyValueError::new_err(format!(
            "Tensor byte size {} is not a multiple of page size {}. \
             Use gpu.from_torch() (copy) instead.",
            nbytes, page_size
        )));
    }

    // Pin the torch tensor
    let py_obj_ptr = tensor.as_ptr();
    unsafe { pyo3::ffi::Py_IncRef(py_obj_ptr) };

    let id = BACKEND.tensor_from_ptr_no_copy(
        data_ptr as *mut u8, nbytes, shape, dtype,
        py_obj_ptr as *mut std::ffi::c_void,
    ).map_err(|e| PyRuntimeError::new_err(e))?;

    Ok(GpuTensor { id })
}
```

### 8. Backend Trait Extension

```rust
pub trait Backend: Send + Sync {
    // Existing
    fn tensor_from_data(&self, data: &[u8], shape: Vec<usize>, dtype: BackendDType) -> BackendResult<u64>;

    // New: no-copy tensor creation (Metal only)
    fn tensor_from_ptr_no_copy(
        &self, ptr: *mut u8, len: usize, shape: Vec<usize>,
        dtype: BackendDType, release_context: *mut std::ffi::c_void,
    ) -> BackendResult<u64> {
        // Default: fall back to copy
        let data = unsafe { std::slice::from_raw_parts(ptr, len) };
        self.tensor_from_data(data, shape, dtype)
    }
}
```

`MetalBackend` overrides with the no-copy path. `SocketBackend` uses the default (copy), since tensor data must be serialized over the wire regardless.

### 9. Immutability Enforcement

Every op dispatch path in `lazy.rs` that selects an output buffer must check:

```rust
// In execute_node_nb, before using out buffer:
if matches!(out.buffer.kind(), BufferKind::Borrowed { .. }) {
    return Err(GpuError::ImmutableBuffer(out.meta.id));
}
```

Add `GpuError::ImmutableBuffer(u64)` variant to `error.rs`.

This check runs once per op dispatch — negligible overhead.

### 10. Testing

**Correctness:**
- `from_numpy_shared` with page-aligned array → tensor values match
- `from_torch_shared` with page-aligned CPU tensor → tensor values match
- Modify source array after shared transfer → GPU tensor sees mutation (documented behavior)
- GPU ops on shared tensor produce correct results (tensor is valid input)

**Safety:**
- Shared tensor cannot be used as op output → `ImmutableBuffer` error
- Source array is not GC'd while tensor exists (check refcount)
- After `gpu.destroy(tensor)`, source array refcount returns to original

**Alignment errors:**
- Non-page-aligned pointer → `AlignmentError`
- Non-page-multiple length → `AlignmentError`
- `gpu.aligned_numpy()` always produces valid input for `from_numpy_shared()`

**Pool safety:**
- Shared tensor buffer is NOT returned to pool on destroy
- Pool stats unchanged after creating/destroying shared tensors

**Benchmark:**
- 1M float32 (4MB): compare `from_numpy` vs `from_numpy_shared` latency
- 100M float32 (400MB): expect ~0ns for shared vs ~50ms for copy
- Model weight loading (GPT-2 large, 774M params): measure total load time improvement

## Files Changed

| File | Change |
|------|--------|
| `crates/core/src/buffer.rs` | Add `BufferKind`, `from_ptr_no_copy()`, kind checks |
| `crates/core/src/error.rs` | Add `ImmutableBuffer` variant |
| `crates/core/src/tensor.rs` | Propagate `BufferKind` checks |
| `crates/core/src/lazy.rs` | Output buffer immutability check in `execute_node_nb` |
| `crates/core/src/pool.rs` | Reject `Borrowed` buffers in `release()` |
| `crates/core/src/ffi.rs` | New `gpu_bridge_create_buffer_no_copy` extern |
| `swift/.../buffer.swift` | `GPUBufferNoCopy` class, `gpuBridgeCreateBufferNoCopy` |
| `swift/.../include/bridge.h` | New C ABI declaration |
| `crates/python/src/backend.rs` | Add `tensor_from_ptr_no_copy` to `Backend` trait |
| `crates/python/src/metal_backend.rs` | Implement `tensor_from_ptr_no_copy` |
| `crates/python/src/lib.rs` | Add `from_numpy_shared`, `from_torch_shared`, `aligned_numpy` |
| `python/applegpu_runtime/__init__.py` | Export new functions |

## Limitations

- **Metal backend only.** SocketBackend always copies (data goes over the wire).
- **Large tensors only.** Below 16KB (one page), the copy is already sub-microsecond.
- **Page alignment required.** Most numpy/torch allocations are NOT page-aligned by default. Users must use `gpu.aligned_numpy()` or allocate with explicit alignment for zero-copy.
- **Shared memory semantics.** CPU mutations are visible to GPU. User is responsible for not mutating during GPU execution.
- **No GPU writes.** Shared tensors are input-only. Cannot be used as output buffers for GPU ops.
