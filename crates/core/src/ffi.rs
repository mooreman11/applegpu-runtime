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
    pub fn gpu_bridge_supports_apple9(device: *const GPUDeviceHandle) -> bool;
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

    pub fn gpu_bridge_create_buffer_no_copy(
        device: *const GPUDeviceHandle,
        data: *mut std::ffi::c_void,
        size_bytes: u64,
        deallocator: Option<unsafe extern "C" fn(*mut std::ffi::c_void, u64, *mut std::ffi::c_void)>,
        deallocator_context: *mut std::ffi::c_void,
    ) -> *mut GPUBufferHandle;

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

    pub fn gpu_bridge_compute_matmul_batched(
        compute: *mut GPUComputeHandle,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_c: *mut GPUBufferHandle,
        m: u32,
        n: u32,
        k: u32,
        batch_size: u32,
        a_batch_stride: u32,
        b_batch_stride: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_softmax_causal(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        batch_size: u32,
        rows: u32,
        cols: u32,
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

    pub fn gpu_bridge_compute_transpose_batched(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        batch_size: u32,
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

    pub fn gpu_bridge_begin_batch(queue: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    pub fn gpu_bridge_end_batch() -> *mut std::ffi::c_void;
    pub fn gpu_bridge_abort_batch();

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

    pub fn gpu_bridge_compute_matmul_batched_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_c: *mut GPUBufferHandle,
        m: u32,
        n: u32,
        k: u32,
        batch_size: u32,
        a_batch_stride: u32,
        b_batch_stride: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_binary_nd_offset_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_a: *const GPUBufferHandle,
        a_byte_offset: u32,
        buf_b: *const GPUBufferHandle,
        b_byte_offset: u32,
        buf_out: *mut GPUBufferHandle,
        a_strides: *const u32,
        b_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_matmul_ex_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_c: *mut GPUBufferHandle,
        m: u32,
        n: u32,
        k: u32,
        batch_size: u32,
        a_batch_stride: u32,
        b_batch_stride: u32,
        transpose_a: bool,
        transpose_b: bool,
    ) -> *mut std::ffi::c_void;

    // MPSGraph fused execution
    pub fn gpu_bridge_mpsgraph_build(
        device: *mut std::ffi::c_void,
        ops_data: *const u8, ops_len: u32,
        n_inputs: u32,
        input_shapes_flat: *const i64,
        input_ndims: *const u32,
        input_dtypes: *const u32,
        n_outputs: u32,
        output_indices: *const u16,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_mpsgraph_run(
        graph_handle: *mut std::ffi::c_void,
        queue_handle: *mut std::ffi::c_void,
        input_buffers: *const *const GPUBufferHandle,
        n_inputs: u32,
        output_buffers: *const *mut GPUBufferHandle,
        n_outputs: u32,
    ) -> i32;

    pub fn gpu_bridge_mpsgraph_destroy(graph_handle: *mut std::ffi::c_void);

    pub fn gpu_bridge_compute_softmax_causal_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        batch_size: u32,
        rows: u32,
        cols: u32,
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

    pub fn gpu_bridge_compute_transpose_batched_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        batch_size: u32,
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

    pub fn gpu_bridge_compute_softmax_backward(
        compute: *mut GPUComputeHandle,
        buf_grad_output: *const GPUBufferHandle,
        buf_output: *const GPUBufferHandle,
        buf_grad_input: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_softmax_backward_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_grad_output: *const GPUBufferHandle,
        buf_output: *const GPUBufferHandle,
        buf_grad_input: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_layer_norm_backward(
        compute: *mut GPUComputeHandle,
        buf_grad_output: *const GPUBufferHandle,
        buf_input: *const GPUBufferHandle,
        buf_gamma: *const GPUBufferHandle,
        buf_grad_input: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> i32;

    pub fn gpu_bridge_compute_layer_norm_backward_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_grad_output: *const GPUBufferHandle,
        buf_input: *const GPUBufferHandle,
        buf_gamma: *const GPUBufferHandle,
        buf_grad_input: *mut GPUBufferHandle,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> *mut std::ffi::c_void;

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

    // Gather: 3 buffers (input, indices, output) + 3 uint params (rows, in_cols, out_cols)
    pub fn gpu_bridge_compute_gather(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_indices: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        in_cols: u32,
        out_cols: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_gather_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_indices: *const GPUBufferHandle,
        buf_output: *mut GPUBufferHandle,
        rows: u32,
        in_cols: u32,
        out_cols: u32,
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

    // ── N-D fused element-wise dispatch ─────────────────────────────────

    pub fn gpu_bridge_compute_fused_nd(
        compute: *mut GPUComputeHandle,
        input_buffers: *const *const GPUBufferHandle,
        buffer_count: u32,
        output: *mut GPUBufferHandle,
        input_strides: *const *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_fused_nd_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        input_buffers: *const *const GPUBufferHandle,
        buffer_count: u32,
        output: *mut GPUBufferHandle,
        input_strides: *const *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
    ) -> *mut std::ffi::c_void;

    // ── N-D stride-based element-wise dispatch ──────────────────────────

    pub fn gpu_bridge_compute_binary_nd(
        compute: *mut GPUComputeHandle,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        a_strides: *const u32,
        b_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_binary_nd_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        a_strides: *const u32,
        b_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_unary_nd(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        in_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_unary_nd_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        in_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_pow_nd(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        in_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
        exponent: f32,
    ) -> i32;

    pub fn gpu_bridge_compute_pow_nd_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        in_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
        exponent: f32,
    ) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_compute_clamp_nd(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        in_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
        min_val: f32,
        max_val: f32,
    ) -> i32;

    pub fn gpu_bridge_compute_clamp_nd_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        in_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
        min_val: f32,
        max_val: f32,
    ) -> *mut std::ffi::c_void;

    // ── Where (ternary) ─────────────────────────────────────────────

    pub fn gpu_bridge_compute_where_nd(
        compute: *mut GPUComputeHandle,
        buf_cond: *const GPUBufferHandle,
        buf_x: *const GPUBufferHandle,
        buf_y: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        cond_strides: *const u32,
        x_strides: *const u32,
        y_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_where_nd_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_cond: *const GPUBufferHandle,
        buf_x: *const GPUBufferHandle,
        buf_y: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        cond_strides: *const u32,
        x_strides: *const u32,
        y_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
    ) -> *mut std::ffi::c_void;

    // ── MaskedFill ──────────────────────────────────────────────────

    pub fn gpu_bridge_compute_masked_fill_nd(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_mask: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        in_strides: *const u32,
        mask_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
        fill_value: f32,
    ) -> i32;

    pub fn gpu_bridge_compute_masked_fill_nd_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_mask: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        in_strides: *const u32,
        mask_strides: *const u32,
        out_shape: *const u32,
        ndim: u32,
        numel: u32,
        fill_value: f32,
    ) -> *mut std::ffi::c_void;

    // ── Triangular (triu/tril) ──────────────────────────────────────

    pub fn gpu_bridge_compute_triangular(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        batch_size: u32,
        rows: u32,
        cols: u32,
        diagonal: i32,
    ) -> i32;

    pub fn gpu_bridge_compute_triangular_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        batch_size: u32,
        rows: u32,
        cols: u32,
        diagonal: i32,
    ) -> *mut std::ffi::c_void;

    // ── Concurrent queue pool ──────────────────────────────────────────
    pub fn gpu_bridge_get_queue(device: *const GPUDeviceHandle, index: u32) -> *mut std::ffi::c_void;

    // ── Batch context system ─────────────────────────────────────────
    pub fn gpu_bridge_set_batch_context(context_id: u32, queue: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    pub fn gpu_bridge_commit_batch_context(context_id: u32) -> *mut std::ffi::c_void;
    pub fn gpu_bridge_set_active_context(context_id: u32);

    // ── MTLEvent synchronization ─────────────────────────────────────
    pub fn gpu_bridge_create_event(device: *const GPUDeviceHandle) -> *mut std::ffi::c_void;
    pub fn gpu_bridge_encode_signal_event(cb: *mut std::ffi::c_void, event: *mut std::ffi::c_void, value: u64);
    pub fn gpu_bridge_encode_wait_event(cb: *mut std::ffi::c_void, event: *mut std::ffi::c_void, value: u64);
    pub fn gpu_bridge_destroy_event(event: *mut std::ffi::c_void);

    // Generic 3D dispatch for CNN ops
    pub fn gpu_bridge_compute_3d(
        compute: *mut GPUComputeHandle,
        input_buffers: *const *const GPUBufferHandle,
        buffer_count: u32,
        output: *mut GPUBufferHandle,
        uint_params: *const u32,
        uint_param_count: u32,
        float_params: *const f32,
        float_param_count: u32,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_3d_nb(
        compute: *mut GPUComputeHandle,
        queue: *mut std::ffi::c_void,
        input_buffers: *const *const GPUBufferHandle,
        buffer_count: u32,
        output: *mut GPUBufferHandle,
        uint_params: *const u32,
        uint_param_count: u32,
        float_params: *const f32,
        float_param_count: u32,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
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
