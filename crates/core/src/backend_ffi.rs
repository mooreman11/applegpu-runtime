//! C-ABI FFI bridge for the PrivateUse1 C++ backend.
//!
//! All public functions are `extern "C"` and use only C-compatible types.
//! The C++ shim calls these functions; all real logic lives in Rust.

use std::sync::Mutex;
use once_cell::sync::OnceCell;

use crate::device::Device;
use crate::lazy::LazyRuntime;
use crate::tensor::{DType, next_tensor_id};

/// Global runtime state for the FFI bridge.
struct FfiState {
    runtime: Mutex<LazyRuntime>,
    device: Device,
}

static FFI_STATE: OnceCell<FfiState> = OnceCell::new();

fn get_state() -> &'static FfiState {
    FFI_STATE.get().expect("applegpu FFI not initialized — call applegpu_ffi_init() first")
}

// Thread-local error message for the last failed FFI call.
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<std::ffi::CString>> = std::cell::RefCell::new(None);
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = std::ffi::CString::new(msg).ok();
    });
}

// ── Init ──────────────────────────────────────────────────────────

/// Initialize the FFI backend. Must be called once before any other FFI function.
/// Returns true on success, false on failure (check applegpu_ffi_last_error).
#[no_mangle]
pub extern "C" fn applegpu_ffi_init() -> bool {
    if FFI_STATE.get().is_some() {
        return true; // already initialized
    }

    let device = match Device::new() {
        Ok(d) => d,
        Err(e) => {
            set_error(format!("Failed to create Metal device: {}", e));
            return false;
        }
    };

    let state = FfiState {
        runtime: Mutex::new(LazyRuntime::new()),
        device,
    };

    FFI_STATE.set(state).unwrap_or_else(|_| {
        // Race condition — another thread initialized. That's fine.
    });
    true
}

// ── Error ─────────────────────────────────────────────────────────

/// Get the last error message as a null-terminated C string.
/// Valid until the next set_error call on the same thread.
/// Returns null if no error.
#[no_mangle]
pub extern "C" fn applegpu_ffi_last_error() -> *const std::ffi::c_char {
    LAST_ERROR.with(|e| {
        match e.borrow().as_ref() {
            Some(cstr) => cstr.as_ptr(),
            None => std::ptr::null(),
        }
    })
}

// ── Alloc/Free ────────────────────────────────────────────────────

/// Allocate a Metal buffer and register it as a pre-allocated tensor.
/// Returns the buffer's data_ptr (storageModeShared, CPU+GPU accessible).
/// Writes the assigned tensor_id to *out_tensor_id.
/// Returns null on failure (check applegpu_ffi_last_error).
#[no_mangle]
pub extern "C" fn applegpu_ffi_alloc(
    size_bytes: u64,
    _dtype_i8: i8,
    out_tensor_id: *mut u64,
) -> *mut u8 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();

    let size = size_bytes as usize;
    if size == 0 {
        let id = next_tensor_id();
        unsafe { *out_tensor_id = id; }
        return std::ptr::null_mut(); // zero-size allocation
    }

    match rt.pool.acquire(&state.device, size) {
        Ok(buffer) => {
            let id = next_tensor_id();
            let ptr = buffer.contents();
            rt.insert_preallocated_buffer(id, buffer);
            unsafe { *out_tensor_id = id; }
            ptr
        }
        Err(e) => {
            set_error(format!("alloc failed: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Free a tensor. Defers if the tensor is referenced by pending graph nodes.
#[no_mangle]
pub extern "C" fn applegpu_ffi_free(tensor_id: u64) {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    rt.try_deferred_free(tensor_id);
}

// ── Sync ──────────────────────────────────────────────────────────

/// Evaluate a tensor (flush the graph up to this tensor).
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_eval(tensor_id: u64) -> i32 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match rt.eval(&state.device, tensor_id) {
        Ok(()) => 0,
        Err(e) => {
            set_error(format!("eval failed: {}", e));
            -1
        }
    }
}

/// Flush the streaming batch and wait for GPU completion.
#[no_mangle]
pub extern "C" fn applegpu_ffi_synchronize() {
    crate::compute::flush_streaming_batch();
}

// ── Metadata ──────────────────────────────────────────────────────

/// Get tensor shape. Writes dims to out_dims, ndim to out_ndim.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_shape(
    tensor_id: u64,
    out_dims: *mut u64,
    out_ndim: *mut u32,
) -> i32 {
    let state = get_state();
    let rt = state.runtime.lock().unwrap();
    match rt.shape(tensor_id) {
        Ok(dims) => {
            unsafe {
                *out_ndim = dims.len() as u32;
                for (i, &d) in dims.iter().enumerate() {
                    *out_dims.add(i) = d as u64;
                }
            }
            0
        }
        Err(e) => {
            set_error(format!("shape failed: {}", e));
            -1
        }
    }
}

/// Get tensor dtype as wire protocol discriminant (see DType::to_wire).
/// Returns -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_dtype(tensor_id: u64) -> i8 {
    let state = get_state();
    let rt = state.runtime.lock().unwrap();
    match rt.dtype(tensor_id) {
        Ok(dt) => dt.to_wire() as i8,
        Err(_) => -1,
    }
}

// ── Tensor Registration ───────────────────────────────────────────

/// Register shape metadata for a pre-allocated tensor.
/// Must be called after alloc, before the tensor is used as an op input.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_register_tensor(
    tensor_id: u64,
    dims_ptr: *const u64,
    ndim: u32,
    dtype_i8: i8,
) -> i32 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();

    let dims: Vec<usize> = unsafe {
        (0..ndim as usize).map(|i| *dims_ptr.add(i) as usize).collect()
    };
    let dtype = match DType::from_wire(dtype_i8 as u32) {
        Some(d) => d,
        None => {
            set_error(format!("Invalid dtype wire value: {}", dtype_i8));
            return -1;
        }
    };

    match rt.materialize_preallocated(tensor_id, dims, dtype) {
        Ok(()) => 0,
        Err(e) => {
            set_error(format!("register_tensor failed: {}", e));
            -1
        }
    }
}

// ── Ops ───────────────────────────────────────────────────────────

/// Record an add op. Returns the output tensor_id, or 0 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_add(a_id: u64, b_id: u64) -> u64 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match crate::ops::add(&mut rt, a_id, b_id) {
        Ok(id) => id,
        Err(e) => {
            set_error(format!("add failed: {}", e));
            0
        }
    }
}

/// Record a matmul op. Returns the output tensor_id, or 0 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_matmul(a_id: u64, b_id: u64) -> u64 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match crate::ops::matmul(&mut rt, a_id, b_id) {
        Ok(id) => id,
        Err(e) => {
            set_error(format!("matmul failed: {}", e));
            0
        }
    }
}

/// Record a relu op. Returns the output tensor_id, or 0 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_relu(input_id: u64) -> u64 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();
    match crate::ops::relu(&mut rt, input_id) {
        Ok(id) => id,
        Err(e) => {
            set_error(format!("relu failed: {}", e));
            0
        }
    }
}

/// Copy tensor data from src to dst buffer.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_copy(
    src_id: u64,
    dst_id: u64,
) -> i32 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();

    // Auto-eval source if pending
    if rt.is_pending(src_id) {
        if let Err(e) = rt.eval(&state.device, src_id) {
            set_error(format!("copy: eval src failed: {}", e));
            return -1;
        }
    }

    match rt.read_bytes(src_id) {
        Ok(bytes) => {
            if let Ok(dst_ptr) = rt.get_tensor_ptr(dst_id) {
                unsafe {
                    let dst = dst_ptr as *mut u8;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
                }
                0
            } else {
                set_error(format!("copy: dst tensor {} not found", dst_id));
                -1
            }
        }
        Err(e) => {
            set_error(format!("copy failed: {}", e));
            -1
        }
    }
}

// ── Readback ──────────────────────────────────────────────────────

/// Read tensor data as f32 into the provided buffer.
/// Flushes streaming batch and evaluates if needed.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_ffi_read_f32(
    tensor_id: u64,
    out_ptr: *mut f32,
    max_elements: u64,
) -> i32 {
    let state = get_state();
    let mut rt = state.runtime.lock().unwrap();

    // Auto-eval if pending
    if rt.is_pending(tensor_id) {
        if let Err(e) = rt.eval(&state.device, tensor_id) {
            set_error(format!("eval failed during readback: {}", e));
            return -1;
        }
    }

    match rt.read_f32(tensor_id) {
        Ok(data) => {
            let n = std::cmp::min(data.len(), max_elements as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), out_ptr, n);
            }
            0
        }
        Err(e) => {
            set_error(format!("read_f32 failed: {}", e));
            -1
        }
    }
}
