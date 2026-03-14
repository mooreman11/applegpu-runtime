use std::ffi::CStr;

/// Opaque handle to a GPU device from the Swift side.
#[repr(C)]
pub struct GPUDeviceHandle {
    _opaque: [u8; 0],
}

extern "C" {
    pub fn gpu_bridge_create_device() -> *mut GPUDeviceHandle;
    pub fn gpu_bridge_destroy_device(device: *mut GPUDeviceHandle);
    pub fn gpu_bridge_device_name(device: *const GPUDeviceHandle) -> *const std::ffi::c_char;
}

/// Safe wrapper: create a Metal device. Returns None if no GPU available.
pub fn create_device() -> Option<*mut GPUDeviceHandle> {
    let ptr = unsafe { gpu_bridge_create_device() };
    if ptr.is_null() {
        None
    } else {
        Some(ptr)
    }
}

/// Safe wrapper: destroy a Metal device.
///
/// # Safety
/// The pointer must have been returned by `create_device()` and not yet destroyed.
pub fn destroy_device(device: *mut GPUDeviceHandle) {
    unsafe { gpu_bridge_destroy_device(device) }
}

/// Safe wrapper: get the device name as a Rust string.
///
/// # Safety
/// The pointer must be a valid device handle.
pub fn device_name(device: *const GPUDeviceHandle) -> Option<String> {
    let name_ptr = unsafe { gpu_bridge_device_name(device) };
    if name_ptr.is_null() {
        None
    } else {
        let c_str = unsafe { CStr::from_ptr(name_ptr) };
        Some(c_str.to_string_lossy().into_owned())
    }
}

/// Opaque handle to a GPU buffer from the Swift side.
#[repr(C)]
pub struct GPUBufferHandle {
    _opaque: [u8; 0],
}

extern "C" {
    pub fn gpu_bridge_create_buffer(
        device: *const GPUDeviceHandle,
        size_bytes: u64,
    ) -> *mut GPUBufferHandle;

    pub fn gpu_bridge_create_buffer_with_data(
        device: *const GPUDeviceHandle,
        data: *const std::ffi::c_void,
        size_bytes: u64,
    ) -> *mut GPUBufferHandle;

    pub fn gpu_bridge_destroy_buffer(buffer: *mut GPUBufferHandle);

    pub fn gpu_bridge_buffer_contents(buffer: *const GPUBufferHandle) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_buffer_length(buffer: *const GPUBufferHandle) -> u64;
}

/// Opaque handle to a compute context from the Swift side.
#[repr(C)]
pub struct GPUComputeHandle {
    _opaque: [u8; 0],
}

extern "C" {
    pub fn gpu_bridge_create_compute(
        device: *const GPUDeviceHandle,
        kernel_source: *const std::ffi::c_char,
        function_name: *const std::ffi::c_char,
    ) -> *mut GPUComputeHandle;

    pub fn gpu_bridge_destroy_compute(compute: *mut GPUComputeHandle);

    pub fn gpu_bridge_compute_elementwise(
        compute: *mut GPUComputeHandle,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        element_count: u64,
    ) -> i32;

    pub fn gpu_bridge_compute_unary(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        element_count: u64,
    ) -> i32;

    pub fn gpu_bridge_compute_matmul(
        compute: *mut GPUComputeHandle,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_c: *mut GPUBufferHandle,
        m: u32,
        n: u32,
        k: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_fused(
        compute: *mut GPUComputeHandle,
        input_buffers: *const *const GPUBufferHandle,
        buffer_count: u32,
        output: *mut GPUBufferHandle,
        element_count: u64,
    ) -> i32;

    pub fn gpu_bridge_compute_softmax(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_transpose(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_scalar_mul(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        scale: f32,
        element_count: u64,
    ) -> i32;

    // ── Non-blocking (batched) dispatch ──────────────────────────────────

    pub fn gpu_bridge_get_shared_queue(
        device: *mut GPUDeviceHandle,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_wait_command_buffer(cb: *mut std::ffi::c_void);

    pub fn gpu_bridge_compute_elementwise_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        element_count: u64,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_unary_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        element_count: u64,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_matmul_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_c: *mut GPUBufferHandle,
        m: u32,
        n: u32,
        k: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_softmax_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_transpose_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_scalar_mul_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        scale: f32,
        element_count: u64,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_layer_norm(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_gamma: *const GPUBufferHandle,
        buf_beta: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> i32;

    pub fn gpu_bridge_compute_embedding(
        compute: *mut GPUComputeHandle,
        buf_weights: *const GPUBufferHandle,
        buf_indices: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        seq_len: u32,
        embed_dim: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_layer_norm_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_gamma: *const GPUBufferHandle,
        buf_beta: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_embedding_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_weights: *const GPUBufferHandle,
        buf_indices: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        seq_len: u32,
        embed_dim: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_blit_copy_nb(
        device: *mut GPUDeviceHandle,
        queue: *mut std::ffi::c_void,
        src: *mut GPUBufferHandle,
        dst: *mut GPUBufferHandle,
        size_bytes: u64,
    ) -> *mut std::ffi::c_void;

    // ── Slice dispatch ──

    pub fn gpu_bridge_compute_slice_dim0(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        cols: u32,
        start_row: u32,
        out_rows: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_slice_dim0_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        cols: u32,
        start_row: u32,
        out_rows: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_slice_dim1(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        in_cols: u32,
        out_cols: u32,
        start_col: u32,
        rows: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_slice_dim1_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        in_cols: u32,
        out_cols: u32,
        start_col: u32,
        rows: u32,
    ) -> *mut std::ffi::c_void;

    // ── Concat dispatch ──

    pub fn gpu_bridge_compute_concat_dim0(
        compute: *mut GPUComputeHandle,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows_a: u32,
        cols: u32,
        total_rows: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_concat_dim0_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows_a: u32,
        cols: u32,
        total_rows: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_concat_dim1(
        compute: *mut GPUComputeHandle,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols_a: u32,
        cols_b: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_concat_dim1_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols_a: u32,
        cols_b: u32,
    ) -> *mut std::ffi::c_void;

    // ── AddBias dispatch ──

    pub fn gpu_bridge_compute_add_bias(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_bias: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_add_bias_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_bias: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_fused_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        input_buffers: *const *const GPUBufferHandle,
        buffer_count: u32,
        output: *mut GPUBufferHandle,
        element_count: u64,
    ) -> *mut std::ffi::c_void;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_query_device() {
        if let Some(device) = create_device() {
            let name = device_name(device);
            assert!(name.is_some());
            assert!(!name.unwrap().is_empty());
            destroy_device(device);
        }
        // If no Metal GPU (CI), test passes by skipping
    }
}
