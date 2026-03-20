//! C-ABI FFI bridge for the eager Metal dispatch path.
//!
//! All public functions are `extern "C"` and use only C-compatible types.
//! The C++ shim calls these functions; all real logic lives in `EagerRuntime`.
//! Unlike `backend_ffi.rs` (graph-based), these ops encode directly into a
//! streaming command buffer with no graph intermediary.

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::sync::{Mutex, OnceLock};

use crate::device::Device;
use crate::eager::EagerRuntime;
use crate::tensor::DType;

struct EagerFfiState {
    runtime: Mutex<EagerRuntime>,
    device: Device,
}

static EAGER_STATE: OnceLock<EagerFfiState> = OnceLock::new();

thread_local! {
    static EAGER_LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_error(msg: String) {
    EAGER_LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

fn get_eager_state() -> &'static EagerFfiState {
    EAGER_STATE
        .get()
        .expect("applegpu eager FFI not initialized — call applegpu_eager_init() first")
}

// ── Init ──────────────────────────────────────────────────────────

/// Initialize the eager dispatch backend. Creates a Metal device and starts
/// the streaming command buffer session. Safe to call multiple times.
#[no_mangle]
pub extern "C" fn applegpu_eager_init() -> bool {
    EAGER_STATE.get_or_init(|| {
        let device = Device::new().expect("Failed to create Metal device");
        let runtime = EagerRuntime::new();
        // Don't start streaming here — it conflicts with the graph-based path.
        // Streaming is started lazily on the first eager dispatch call.
        EagerFfiState {
            runtime: Mutex::new(runtime),
            device,
        }
    });
    true
}

// ── Error ─────────────────────────────────────────────────────────

/// Get the last error message as a null-terminated C string.
/// Valid until the next set_error call on the same thread.
/// Returns null if no error.
#[no_mangle]
pub extern "C" fn applegpu_eager_last_error() -> *const c_char {
    EAGER_LAST_ERROR.with(|e| match e.borrow().as_ref() {
        Some(s) => s.as_ptr(),
        None => std::ptr::null(),
    })
}

// ── Memory ────────────────────────────────────────────────────────

/// Allocate a contiguous Metal buffer for a tensor with given shape and dtype.
/// Returns the buffer's data_ptr (storageModeShared, CPU+GPU accessible).
/// Writes the assigned tensor_id to *out_id.
/// Returns null on failure (check applegpu_eager_last_error).
#[no_mangle]
pub extern "C" fn applegpu_eager_alloc(
    dims: *const u64,
    ndim: u32,
    dtype_i8: i8,
    out_id: *mut u64,
) -> *mut u8 {
    let state = get_eager_state();
    let mut rt = state.runtime.lock().unwrap();
    let dtype = match DType::from_wire(dtype_i8 as u32) {
        Some(d) => d,
        None => {
            set_error(format!("invalid dtype wire value: {}", dtype_i8));
            return std::ptr::null_mut();
        }
    };
    let shape: Vec<usize> = (0..ndim as usize)
        .map(|i| unsafe { *dims.add(i) } as usize)
        .collect();
    match rt.alloc(&state.device, &shape, dtype) {
        Ok((id, ptr)) => {
            unsafe { *out_id = id; }
            ptr
        }
        Err(e) => {
            set_error(format!("alloc failed: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Free a tensor. If this was the last reference to the underlying buffer,
/// the buffer is returned to the pool for reuse.
#[no_mangle]
pub extern "C" fn applegpu_eager_free(id: u64) {
    let state = get_eager_state();
    let mut rt = state.runtime.lock().unwrap();
    rt.free(id);
}

/// Register/update shape metadata for an existing tensor.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_eager_register_shape(
    id: u64,
    dims: *const u64,
    ndim: u32,
) -> i32 {
    let state = get_eager_state();
    let mut rt = state.runtime.lock().unwrap();
    let shape: Vec<usize> = (0..ndim as usize)
        .map(|i| unsafe { *dims.add(i) } as usize)
        .collect();
    if let Some(tensor) = rt.tensors_mut().get_mut(&id) {
        match crate::tensor::Shape::new(shape) {
            Ok(s) => {
                tensor.layout = crate::tensor::TensorLayout::contiguous(s);
                0
            }
            Err(e) => {
                set_error(format!("{}", e));
                -1
            }
        }
    } else {
        set_error(format!("tensor {} not found", id));
        -1
    }
}

/// Get tensor shape. Writes dims to out_dims, ndim to out_ndim.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_eager_shape(
    id: u64,
    out_dims: *mut u64,
    out_ndim: *mut u32,
) -> i32 {
    let state = get_eager_state();
    let rt = state.runtime.lock().unwrap();
    match rt.shape(id) {
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
            set_error(format!("{}", e));
            -1
        }
    }
}

/// Get tensor dtype as wire protocol discriminant (see DType::to_wire).
/// Returns -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_eager_dtype(id: u64) -> i8 {
    let state = get_eager_state();
    let rt = state.runtime.lock().unwrap();
    match rt.dtype(id) {
        Ok(d) => d.to_wire() as i8,
        Err(_) => -1,
    }
}

// ── Binary ops ────────────────────────────────────────────────────

/// Ensure the eager streaming CB is active. Called before any dispatch.
fn ensure_eager_streaming() {
    if !crate::compute::streaming_is_active() {
        let state = get_eager_state();
        let mut rt = state.runtime.lock().unwrap();
        rt.begin_streaming(&state.device);
    }
}

macro_rules! eager_binary {
    ($fn_name:ident, $kernel:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(a_id: u64, b_id: u64, out_id: *mut u64) -> *mut u8 {
            ensure_eager_streaming();
            let state = get_eager_state();
            let mut rt = state.runtime.lock().unwrap();
            match rt.binary_op(&state.device, $kernel, a_id, b_id) {
                Ok((id, ptr)) => {
                    unsafe { *out_id = id; }
                    ptr
                }
                Err(e) => {
                    set_error(format!("{}", e));
                    std::ptr::null_mut()
                }
            }
        }
    };
}

eager_binary!(applegpu_eager_add, "elementwise_add");
eager_binary!(applegpu_eager_sub, "elementwise_sub");
eager_binary!(applegpu_eager_mul, "elementwise_mul");
eager_binary!(applegpu_eager_div, "elementwise_div");

// ── Unary ops ─────────────────────────────────────────────────────

macro_rules! eager_unary {
    ($fn_name:ident, $kernel:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(input_id: u64, out_id: *mut u64) -> *mut u8 {
            let state = get_eager_state();
            let mut rt = state.runtime.lock().unwrap();
            match rt.unary_op(&state.device, $kernel, input_id) {
                Ok((id, ptr)) => {
                    unsafe { *out_id = id; }
                    ptr
                }
                Err(e) => {
                    set_error(format!("{}", e));
                    std::ptr::null_mut()
                }
            }
        }
    };
}

eager_unary!(applegpu_eager_relu, "elementwise_relu");
eager_unary!(applegpu_eager_neg, "elementwise_neg");

// ── Matmul ────────────────────────────────────────────────────────

/// Matmul [M,K] @ [K,N] → [M,N] (or batched). Encodes into the streaming
/// command buffer. Returns output data ptr and writes tensor_id to *out_id.
#[no_mangle]
pub extern "C" fn applegpu_eager_matmul(
    a_id: u64,
    b_id: u64,
    out_id: *mut u64,
) -> *mut u8 {
    let state = get_eager_state();
    let mut rt = state.runtime.lock().unwrap();
    match rt.matmul(&state.device, a_id, b_id) {
        Ok((id, ptr)) => {
            unsafe { *out_id = id; }
            ptr
        }
        Err(e) => {
            set_error(format!("{}", e));
            std::ptr::null_mut()
        }
    }
}

// ── Compound ops (stubs — will use graph-based fallback initially) ──

/// threshold_backward: grad * (input > threshold). ReLU backward.
/// Flushes GPU, computes on CPU via shared memory.
#[no_mangle]
pub extern "C" fn applegpu_eager_threshold_backward(
    grad_id: u64,
    input_id: u64,
    threshold: f32,
    out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut rt = state.runtime.lock().unwrap();
    match rt.threshold_backward(&state.device, grad_id, input_id, threshold) {
        Ok((id, ptr)) => {
            unsafe { *out_id = id; }
            ptr
        }
        Err(e) => {
            set_error(format!("{}", e));
            std::ptr::null_mut()
        }
    }
}

/// Multiply tensor by scalar via broadcast binary mul.
#[no_mangle]
pub extern "C" fn applegpu_eager_scalar_mul(
    input_id: u64,
    scale: f32,
    out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut rt = state.runtime.lock().unwrap();
    match rt.scalar_mul(&state.device, input_id, scale) {
        Ok((id, ptr)) => {
            unsafe { *out_id = id; }
            ptr
        }
        Err(e) => {
            set_error(format!("{}", e));
            std::ptr::null_mut()
        }
    }
}

/// Full mean reduction to scalar [1]. Flushes GPU, computes on CPU via shared memory.
#[no_mangle]
pub extern "C" fn applegpu_eager_mean_all(
    input_id: u64,
    out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut rt = state.runtime.lock().unwrap();
    match rt.mean_all(&state.device, input_id) {
        Ok((id, ptr)) => {
            unsafe { *out_id = id; }
            ptr
        }
        Err(e) => {
            set_error(format!("{}", e));
            std::ptr::null_mut()
        }
    }
}

// ── Views ─────────────────────────────────────────────────────────

/// Create a view of an existing tensor with different shape/strides/offset.
/// The view shares the same underlying Metal buffer (Arc reference).
/// Returns the view's data ptr and writes its tensor_id to *out_id.
#[no_mangle]
pub extern "C" fn applegpu_eager_create_view(
    base_id: u64,
    shape: *const u64,
    strides: *const u64,
    ndim: u32,
    offset_elements: u64,
    out_id: *mut u64,
) -> *mut u8 {
    let state = get_eager_state();
    let mut rt = state.runtime.lock().unwrap();
    let shape_vec: Vec<usize> = (0..ndim as usize)
        .map(|i| unsafe { *shape.add(i) } as usize)
        .collect();
    let strides_vec: Vec<usize> = (0..ndim as usize)
        .map(|i| unsafe { *strides.add(i) } as usize)
        .collect();
    match rt.create_view(base_id, &shape_vec, &strides_vec, offset_elements as usize) {
        Ok(id) => {
            unsafe { *out_id = id; }
            rt.get(id)
                .map(|t| t.data_ptr())
                .unwrap_or(std::ptr::null_mut())
        }
        Err(e) => {
            set_error(format!("{}", e));
            std::ptr::null_mut()
        }
    }
}

// ── In-place ops ──────────────────────────────────────────────────

/// In-place add: self_id += other_id. Self must be contiguous.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_eager_add_inplace(self_id: u64, other_id: u64) -> i32 {
    let state = get_eager_state();
    let mut rt = state.runtime.lock().unwrap();
    match rt.inplace_binary_op(&state.device, "elementwise_add", self_id, other_id) {
        Ok(()) => 0,
        Err(e) => {
            set_error(format!("{}", e));
            -1
        }
    }
}

/// In-place scaled add: self_id += alpha * other_id. SGD optimizer update.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn applegpu_eager_add_scaled_inplace(
    self_id: u64,
    other_id: u64,
    alpha: f32,
) -> i32 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut rt = state.runtime.lock().unwrap();
    match rt.add_scaled_inplace(&state.device, self_id, other_id, alpha) {
        Ok(()) => 0,
        Err(e) => {
            set_error(format!("{}", e));
            -1
        }
    }
}

// ── Sync ──────────────────────────────────────────────────────────

/// Flush the streaming command buffer (commit + wait for GPU completion),
/// then reopen a new command buffer for subsequent ops.
#[no_mangle]
pub extern "C" fn applegpu_eager_flush_and_wait() {
    if let Some(state) = EAGER_STATE.get() {
        let rt = state.runtime.lock().unwrap();
        rt.flush_and_wait();
    }
}

/// Synchronize the GPU. Equivalent to flush_and_wait.
#[no_mangle]
pub extern "C" fn applegpu_eager_synchronize() {
    applegpu_eager_flush_and_wait();
}
