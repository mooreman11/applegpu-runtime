//! C-ABI FFI bridge for the eager dispatch path.
//!
//! All public functions are `extern "C"` and use only C-compatible types.
//! The C++ shim calls these functions; all real logic lives in either:
//! - `EagerRuntime` (local Metal GPU) — default on macOS
//! - `RemoteEagerRuntime` (socket to gpu-service) — when APPLEGPU_SOCKET is set
//!
//! The backend is selected at init time and transparent to the C++ side.

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::sync::{Mutex, OnceLock};

use crate::device::Device;
use crate::eager::EagerRuntime;
use crate::remote_eager::RemoteEagerRuntime;
use crate::tensor::DType;
use applegpu_wire as wire;

// ── Backend enum ─────────────────────────────────────────────────

enum EagerBackend {
    /// Direct Metal GPU dispatch (macOS host).
    Local {
        runtime: EagerRuntime,
        device: Device,
    },
    /// Remote dispatch via socket to gpu-service (containers).
    Remote(RemoteEagerRuntime),
}

struct EagerFfiState {
    backend: Mutex<EagerBackend>,
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

/// Check if we're in remote mode (no Metal device available).
fn is_remote() -> bool {
    if let Some(state) = EAGER_STATE.get() {
        matches!(*state.backend.lock().unwrap(), EagerBackend::Remote(_))
    } else {
        false
    }
}

// ── Init ──────────────────────────────────────────────────────────

/// Initialize the eager dispatch backend.
/// If APPLEGPU_SOCKET is set, connects to gpu-service via socket (remote mode).
/// Otherwise, creates a Metal device for direct GPU dispatch (local mode).
/// Safe to call multiple times.
#[no_mangle]
pub extern "C" fn applegpu_eager_init() -> bool {
    EAGER_STATE.get_or_init(|| {
        // Check for remote mode
        if std::env::var("APPLEGPU_SOCKET").is_ok()
            || std::env::var("APPLEGPU_HOST").is_ok()
        {
            eprintln!("[applegpu] Remote mode: dispatching ops via socket to gpu-service");
            let remote = RemoteEagerRuntime::new();
            return EagerFfiState {
                backend: Mutex::new(EagerBackend::Remote(remote)),
            };
        }

        // Local Metal mode
        let device = Device::new().expect("Failed to create Metal device");
        let runtime = EagerRuntime::new();
        EagerFfiState {
            backend: Mutex::new(EagerBackend::Local { runtime, device }),
        }
    });
    true
}

// ── Error ─────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_last_error() -> *const c_char {
    EAGER_LAST_ERROR.with(|e| match e.borrow().as_ref() {
        Some(s) => s.as_ptr(),
        None => std::ptr::null(),
    })
}

// ── Memory ────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_alloc(
    dims: *const u64,
    ndim: u32,
    dtype_i8: i8,
    out_id: *mut u64,
) -> *mut u8 {
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
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
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.alloc(device, &shape, dtype) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("alloc failed: {}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.alloc(&shape, dtype) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("alloc failed: {}", e)); std::ptr::null_mut() }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_free(id: u64) {
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, .. } => runtime.free(id),
        EagerBackend::Remote(remote) => remote.free(id),
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_register_shape(
    id: u64, dims: *const u64, ndim: u32,
) -> i32 {
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    let shape: Vec<usize> = (0..ndim as usize)
        .map(|i| unsafe { *dims.add(i) } as usize)
        .collect();
    match &mut *backend {
        EagerBackend::Local { runtime, .. } => {
            if let Some(tensor) = runtime.tensors_mut().get_mut(&id) {
                match crate::tensor::Shape::new(shape) {
                    Ok(s) => { tensor.layout = crate::tensor::TensorLayout::contiguous(s); 0 }
                    Err(e) => { set_error(format!("{}", e)); -1 }
                }
            } else {
                set_error(format!("tensor {} not found", id)); -1
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.register_shape(id, &shape) {
                Ok(()) => 0,
                Err(e) => { set_error(e); -1 }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_shape(
    id: u64, out_dims: *mut u64, out_ndim: *mut u32,
) -> i32 {
    let state = get_eager_state();
    let backend = state.backend.lock().unwrap();
    let dims_result: Result<Vec<usize>, String> = match &*backend {
        EagerBackend::Local { runtime, .. } => {
            runtime.shape(id).map(|d| d.to_vec()).map_err(|e| format!("{}", e))
        }
        EagerBackend::Remote(remote) => {
            remote.shape(id).map(|d| d.to_vec())
        }
    };
    match dims_result {
        Ok(dims) => {
            unsafe {
                *out_ndim = dims.len() as u32;
                for (i, &d) in dims.iter().enumerate() {
                    *out_dims.add(i) = d as u64;
                }
            }
            0
        }
        Err(e) => { set_error(e); -1 }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_dtype(id: u64) -> i8 {
    let state = get_eager_state();
    let backend = state.backend.lock().unwrap();
    match &*backend {
        EagerBackend::Local { runtime, .. } => {
            match runtime.dtype(id) { Ok(d) => d.to_wire() as i8, Err(_) => -1 }
        }
        EagerBackend::Remote(remote) => {
            match remote.dtype(id) { Ok(d) => d.to_wire() as i8, Err(_) => -1 }
        }
    }
}

// ── Binary ops ────────────────────────────────────────────────────

fn ensure_eager_streaming() {
    if is_remote() { return; }
    if !crate::compute::streaming_is_active() {
        let state = get_eager_state();
        let mut backend = state.backend.lock().unwrap();
        if let EagerBackend::Local { runtime, device } = &mut *backend {
            runtime.begin_streaming(device);
        }
    }
}

macro_rules! eager_binary {
    ($fn_name:ident, $kernel:expr, $wire_op:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(a_id: u64, b_id: u64, out_id: *mut u64) -> *mut u8 {
            ensure_eager_streaming();
            let state = get_eager_state();
            let mut backend = state.backend.lock().unwrap();
            match &mut *backend {
                EagerBackend::Local { runtime, device } => {
                    match runtime.binary_op(device, $kernel, a_id, b_id) {
                        Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                        Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
                    }
                }
                EagerBackend::Remote(remote) => {
                    match remote.binary_op($wire_op, a_id, b_id) {
                        Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                        Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
                    }
                }
            }
        }
    };
}

eager_binary!(applegpu_eager_add, "elementwise_add", wire::WireOpKind::Add);
eager_binary!(applegpu_eager_sub, "elementwise_sub", wire::WireOpKind::Sub);
eager_binary!(applegpu_eager_mul, "elementwise_mul", wire::WireOpKind::Mul);
eager_binary!(applegpu_eager_div, "elementwise_div", wire::WireOpKind::Div);

// ── Unary ops ─────────────────────────────────────────────────────

macro_rules! eager_unary {
    ($fn_name:ident, $kernel:expr, $wire_op:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(input_id: u64, out_id: *mut u64) -> *mut u8 {
            ensure_eager_streaming();
            let state = get_eager_state();
            let mut backend = state.backend.lock().unwrap();
            match &mut *backend {
                EagerBackend::Local { runtime, device } => {
                    match runtime.unary_op(device, $kernel, input_id) {
                        Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                        Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
                    }
                }
                EagerBackend::Remote(remote) => {
                    match remote.unary_op($wire_op, input_id) {
                        Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                        Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
                    }
                }
            }
        }
    };
}

eager_unary!(applegpu_eager_relu, "elementwise_relu", wire::WireOpKind::Relu);
eager_unary!(applegpu_eager_neg, "elementwise_neg", wire::WireOpKind::Neg);

// ── Matmul ────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_matmul(
    a_id: u64, b_id: u64, out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.matmul(device, a_id, b_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.matmul(a_id, b_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_matmul_ex(
    a_id: u64, b_id: u64, transpose_a: bool, transpose_b: bool, out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.matmul_ex(device, a_id, b_id, transpose_a, transpose_b) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(remote) => {
            // Remote: matmul_ex not yet supported, fall back to regular matmul
            // TODO: add transpose support to wire protocol
            match remote.matmul(a_id, b_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
    }
}

// ── Compound ops ─────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_threshold_backward(
    grad_id: u64, input_id: u64, threshold: f32, out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.threshold_backward(device, grad_id, input_id, threshold) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.binary_op(wire::WireOpKind::ThresholdBackward { threshold }, grad_id, input_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_scalar_mul(
    input_id: u64, scale: f32, out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.scalar_mul(device, input_id, scale) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.scalar_mul(input_id, scale) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_mean_all(
    input_id: u64, out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.mean_all(device, input_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(remote) => {
            // mean_all maps to ScalarMul with shape [1] — use a special op
            // For now, record as unary with Mean wire op
            match remote.unary_op(wire::WireOpKind::Mean, input_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
    }
}

// ── Sum dim ──────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_sum_dim(
    input_id: u64, dim: i64, keepdim: bool, out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.sum_dim(device, input_id, dim, keepdim) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(_remote) => {
            // TODO: implement sum_dim in remote mode
            set_error("sum_dim not yet supported in remote mode".to_string());
            std::ptr::null_mut()
        }
    }
}

// ── Views ─────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_create_view(
    base_id: u64, shape: *const u64, strides: *const u64,
    ndim: u32, offset_elements: u64, out_id: *mut u64,
) -> *mut u8 {
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    let shape_vec: Vec<usize> = (0..ndim as usize)
        .map(|i| unsafe { *shape.add(i) } as usize)
        .collect();
    let strides_vec: Vec<usize> = (0..ndim as usize)
        .map(|i| unsafe { *strides.add(i) } as usize)
        .collect();
    match &mut *backend {
        EagerBackend::Local { runtime, .. } => {
            match runtime.create_view(base_id, &shape_vec, &strides_vec, offset_elements as usize) {
                Ok(id) => {
                    unsafe { *out_id = id; }
                    runtime.get(id)
                        .map(|t| t.data_ptr())
                        .unwrap_or(std::ptr::null_mut())
                }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.create_view(base_id, &shape_vec, &strides_vec, offset_elements as usize) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
    }
}

// ── In-place ops ──────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_add_inplace(self_id: u64, other_id: u64) -> i32 {
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.inplace_binary_op(device, "elementwise_add", self_id, other_id) {
                Ok(()) => 0,
                Err(e) => { set_error(format!("{}", e)); -1 }
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.add_scaled_inplace(self_id, other_id, 1.0) {
                Ok(()) => 0,
                Err(e) => { set_error(e); -1 }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_add_scaled_inplace(
    self_id: u64, other_id: u64, alpha: f32,
) -> i32 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.add_scaled_inplace(device, self_id, other_id, alpha) {
                Ok(()) => 0,
                Err(e) => { set_error(format!("{}", e)); -1 }
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.add_scaled_inplace(self_id, other_id, alpha) {
                Ok(()) => 0,
                Err(e) => { set_error(e); -1 }
            }
        }
    }
}

// ── GPT-2 ops ────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_embedding(
    weight_id: u64, indices_id: u64, out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.embedding(device, weight_id, indices_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.binary_op(wire::WireOpKind::Embedding, weight_id, indices_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_layer_norm(
    input_id: u64, gamma_id: u64, beta_id: u64, eps: f32, out_id: *mut u64,
) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.layer_norm(device, input_id, gamma_id, beta_id, eps) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(_remote) => {
            // TODO: layer_norm needs 3-input wire op
            set_error("layer_norm not yet supported in remote mode".to_string());
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_gelu(input_id: u64, out_id: *mut u64) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.gelu(device, input_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.unary_op(wire::WireOpKind::Gelu, input_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_softmax(input_id: u64, out_id: *mut u64) -> *mut u8 {
    ensure_eager_streaming();
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            match runtime.softmax(device, input_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
        EagerBackend::Remote(remote) => {
            match remote.unary_op(wire::WireOpKind::Softmax, input_id) {
                Ok((id, ptr)) => { unsafe { *out_id = id; } ptr }
                Err(e) => { set_error(format!("{}", e)); std::ptr::null_mut() }
            }
        }
    }
}

// ── Lookup ────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_find_by_data_ptr(ptr: *const u8) -> u64 {
    let state = get_eager_state();
    let backend = state.backend.lock().unwrap();
    match &*backend {
        EagerBackend::Local { runtime, .. } => {
            // First pass: exact match on buffer.contents() with offset=0 (base tensors)
            for (&id, tensor) in runtime.tensors_iter() {
                if tensor.offset == 0 && tensor.buffer.contents() as *const u8 == ptr {
                    return id;
                }
            }
            // Second pass: any tensor whose buffer.contents() matches (views)
            for (&id, tensor) in runtime.tensors_iter() {
                if tensor.buffer.contents() as *const u8 == ptr {
                    return id;
                }
            }
            0
        }
        EagerBackend::Remote(remote) => remote.find_by_data_ptr(ptr),
    }
}

// ── Sync ──────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_flush_and_wait() {
    if let Some(state) = EAGER_STATE.get() {
        let mut backend = state.backend.lock().unwrap();
        match &mut *backend {
            EagerBackend::Local { runtime, .. } => {
                runtime.flush_and_release_pending();
            }
            EagerBackend::Remote(remote) => {
                if let Err(e) = remote.flush_and_wait() {
                    set_error(format!("remote flush failed: {}", e));
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn applegpu_eager_synchronize() {
    applegpu_eager_flush_and_wait();
}

// ── Compiled graph execution ─────────────────────────────────────

#[no_mangle]
pub extern "C" fn applegpu_eager_execute_graph(
    ops_data: *const u8, ops_len: u32,
    input_tids: *const u64, n_inputs: u32,
    output_indices: *const u16, n_outputs: u32,
    out_tids: *mut u64, out_ptrs: *mut *mut u8,
) -> i32 {
    let state = get_eager_state();
    let mut backend = state.backend.lock().unwrap();
    match &mut *backend {
        EagerBackend::Local { runtime, device } => {
            // Flush any pending C++ dispatcher work before our graph
            runtime.flush_and_wait();

            let ops = unsafe { std::slice::from_raw_parts(ops_data, ops_len as usize) };
            let inputs = unsafe { std::slice::from_raw_parts(input_tids, n_inputs as usize) };
            let outputs = unsafe { std::slice::from_raw_parts(output_indices, n_outputs as usize) };
            let tids_out = unsafe { std::slice::from_raw_parts_mut(out_tids, n_outputs as usize) };
            let ptrs_out = unsafe { std::slice::from_raw_parts_mut(out_ptrs, n_outputs as usize) };

            // Try MPSGraph fused execution (opt-in via APPLEGPU_MPSGRAPH=1).
            if std::env::var("APPLEGPU_MPSGRAPH").is_ok() {
                match crate::compiled_graph::execute_mpsgraph(
                    runtime, device, ops, inputs, outputs, tids_out, ptrs_out,
                ) {
                    Ok(n) => return n as i32,
                    Err(e) => {
                        if std::env::var("APPLEGPU_LOG_MPSGRAPH").is_ok() {
                            eprintln!("[mpsgraph] fallback: {}", e);
                        }
                    }
                }
            }

            match crate::compiled_graph::execute(
                runtime, device, ops, inputs, outputs, tids_out, ptrs_out,
            ) {
                Ok(n) => n as i32,
                Err(e) => { set_error(format!("compiled graph failed: {}", e)); -1 }
            }
        }
        EagerBackend::Remote(_remote) => {
            // TODO: compiled graph execution in remote mode
            set_error("compiled graph not supported in remote mode".to_string());
            -1
        }
    }
}
