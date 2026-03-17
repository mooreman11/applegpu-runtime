use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::ffi;
use crate::tensor::{DType, MAX_DIMS};


/// A Metal compute pipeline. Wraps command queue + pipeline state.
pub struct ComputePipeline {
    handle: *mut ffi::GPUComputeHandle,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    /// Create a compute pipeline from MSL source and function name.
    pub fn new(device: &Device, kernel_source: &str, function_name: &str) -> Result<Self> {
        let source = CString::new(kernel_source).map_err(|_| {
            GpuError::ComputeFailed("Invalid kernel source (null byte)".to_string())
        })?;
        let name = CString::new(function_name).map_err(|_| {
            GpuError::ComputeFailed("Invalid function name (null byte)".to_string())
        })?;

        let handle = unsafe {
            ffi::gpu_bridge_create_compute(device.raw_handle(), source.as_ptr(), name.as_ptr())
        };

        if handle.is_null() {
            Err(GpuError::ComputeFailed(format!(
                "Failed to create compute pipeline for '{}'",
                function_name
            )))
        } else {
            Ok(ComputePipeline { handle })
        }
    }

    /// Dispatch binary element-wise operation with N-D strides: out = op(a, b).
    /// For contiguous same-shape inputs, constructs 1-D contiguous strides.
    pub fn dispatch_elementwise(
        &self,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        // Construct 1-D contiguous strides for backward compat
        let mut strides = [0u32; MAX_DIMS];
        strides[0] = 1;
        let mut shape = [0u32; MAX_DIMS];
        shape[0] = element_count as u32;
        self.dispatch_binary_nd(
            buf_a, &strides,
            buf_b, &strides,
            buf_out, &shape,
            1, element_count as u32,
        )
    }

    /// Dispatch unary element-wise operation with N-D strides: out = op(input).
    /// For contiguous inputs, constructs 1-D contiguous strides.
    pub fn dispatch_unary(
        &self,
        buf_input: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let mut strides = [0u32; MAX_DIMS];
        strides[0] = 1;
        let mut shape = [0u32; MAX_DIMS];
        shape[0] = element_count as u32;
        self.dispatch_unary_nd(
            buf_input, &strides,
            buf_out, &shape,
            1, element_count as u32,
        )
    }

    /// Dispatch batched matrix multiplication:
    /// C[batch, M, N] = A[batch, M, K] * B[batch, K, N]
    /// batch_size=1 for standard 2D matmul.
    pub fn dispatch_matmul(
        &self,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.dispatch_matmul_batched(buf_a, buf_b, buf_c, m, n, k, 1, m * k, k * n)
    }

    /// Dispatch batched matmul with explicit batch params.
    pub fn dispatch_matmul_batched(
        &self,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
        batch_size: usize,
        a_batch_stride: usize,
        b_batch_stride: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_matmul_batched(
                self.handle,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_c.raw_handle(),
                m as u32,
                n as u32,
                k as u32,
                batch_size as u32,
                a_batch_stride as u32,
                b_batch_stride as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Batched matmul dispatch failed".to_string())) }
    }

    /// Dispatch a fused element-wise kernel with variable input buffer count.
    pub fn dispatch_fused(
        &self,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let ptrs: Vec<*const ffi::GPUBufferHandle> = input_buffers
            .iter()
            .map(|b| b.raw_handle() as *const _)
            .collect();

        let result = unsafe {
            ffi::gpu_bridge_compute_fused(
                self.handle,
                ptrs.as_ptr(),
                ptrs.len() as u32,
                buf_out.raw_handle(),
                element_count as u64,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Fused kernel dispatch failed".to_string())) }
    }

    /// Dispatch a fused N-D element-wise kernel with stride arrays per input.
    pub fn dispatch_fused_nd(
        &self,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        input_strides: &[&[u32; MAX_DIMS]],
        out_shape: &[u32; MAX_DIMS],
        ndim: u32,
        numel: u32,
    ) -> Result<()> {
        let ptrs: Vec<*const ffi::GPUBufferHandle> = input_buffers
            .iter()
            .map(|b| b.raw_handle() as *const _)
            .collect();
        let stride_ptrs: Vec<*const u32> = input_strides
            .iter()
            .map(|s| s.as_ptr())
            .collect();

        let result = unsafe {
            ffi::gpu_bridge_compute_fused_nd(
                self.handle,
                ptrs.as_ptr(),
                ptrs.len() as u32,
                buf_out.raw_handle(),
                stride_ptrs.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Fused N-D kernel dispatch failed".to_string())) }
    }

    /// Non-blocking fused N-D dispatch. Returns command buffer handle.
    pub fn dispatch_fused_nd_nb(
        &self,
        queue: *mut std::ffi::c_void,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        input_strides: &[&[u32; MAX_DIMS]],
        out_shape: &[u32; MAX_DIMS],
        ndim: u32,
        numel: u32,
    ) -> Result<*mut std::ffi::c_void> {
        let ptrs: Vec<*const ffi::GPUBufferHandle> = input_buffers
            .iter()
            .map(|b| b.raw_handle() as *const _)
            .collect();
        let stride_ptrs: Vec<*const u32> = input_strides
            .iter()
            .map(|s| s.as_ptr())
            .collect();

        let cb = unsafe {
            ffi::gpu_bridge_compute_fused_nd_nb(
                self.handle,
                queue,
                ptrs.as_ptr(),
                ptrs.len() as u32,
                buf_out.raw_handle(),
                stride_ptrs.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking fused N-D dispatch failed".to_string())) } else { Ok(cb) }
    }

    pub fn dispatch_softmax(
        &self, buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_softmax(
                self.handle, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), rows as u32, cols as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Softmax dispatch failed".to_string())) }
    }

    /// Dispatch batched softmax_causal with 2D grid: (row, batch).
    pub fn dispatch_softmax_causal(
        &self, buf_input: &Buffer, buf_output: &Buffer, batch_size: usize, rows: usize, cols: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_softmax_causal(
                self.handle, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), batch_size as u32, rows as u32, cols as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("SoftmaxCausal dispatch failed".to_string())) }
    }

    pub fn dispatch_transpose(
        &self, buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_transpose(
                self.handle, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), rows as u32, cols as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Transpose dispatch failed".to_string())) }
    }

    /// Dispatch batched transpose with 3D grid: (col, row, batch).
    pub fn dispatch_transpose_batched(
        &self, buf_input: &Buffer, buf_output: &Buffer, batch_size: usize, rows: usize, cols: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_transpose_batched(
                self.handle, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), batch_size as u32, rows as u32, cols as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Batched transpose dispatch failed".to_string())) }
    }

    pub fn dispatch_scalar_mul(
        &self, buf_input: &Buffer, buf_output: &Buffer, scale: f32, element_count: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_scalar_mul(
                self.handle, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), scale, element_count as u64,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("ScalarMul dispatch failed".to_string())) }
    }

    /// Dispatch layer normalization: output = gamma * (input - mean) / sqrt(var + eps) + beta.
    pub fn dispatch_layer_norm(
        &self,
        buf_input: &Buffer,
        buf_gamma: &Buffer,
        buf_beta: &Buffer,
        buf_out: &Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_layer_norm(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_gamma.raw_handle() as *const _,
                buf_beta.raw_handle() as *const _,
                buf_out.raw_handle(),
                rows as u32,
                cols as u32,
                eps,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("LayerNorm dispatch failed".to_string())) }
    }

    /// Dispatch softmax backward: grad_input = output * (grad_output - sum(grad_output * output)).
    pub fn dispatch_softmax_backward(
        &self,
        buf_grad_output: &Buffer,
        buf_output: &Buffer,
        buf_grad_input: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_softmax_backward(
                self.handle,
                buf_grad_output.raw_handle() as *const _,
                buf_output.raw_handle() as *const _,
                buf_grad_input.raw_handle(),
                rows as u32,
                cols as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("SoftmaxBackward dispatch failed".to_string())) }
    }

    /// Non-blocking softmax backward. Returns command buffer handle.
    pub fn dispatch_softmax_backward_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_grad_output: &Buffer,
        buf_output: &Buffer,
        buf_grad_input: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_softmax_backward_nb(
                self.handle,
                queue,
                buf_grad_output.raw_handle() as *const _,
                buf_output.raw_handle() as *const _,
                buf_grad_input.raw_handle(),
                rows as u32,
                cols as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking softmax_backward dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Dispatch layer norm backward: computes grad_input from grad_output, input, gamma.
    pub fn dispatch_layer_norm_backward(
        &self,
        buf_grad_output: &Buffer,
        buf_input: &Buffer,
        buf_gamma: &Buffer,
        buf_grad_input: &Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_layer_norm_backward(
                self.handle,
                buf_grad_output.raw_handle() as *const _,
                buf_input.raw_handle() as *const _,
                buf_gamma.raw_handle() as *const _,
                buf_grad_input.raw_handle(),
                rows as u32,
                cols as u32,
                eps,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("LayerNormBackward dispatch failed".to_string())) }
    }

    /// Non-blocking layer norm backward. Returns command buffer handle.
    pub fn dispatch_layer_norm_backward_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_grad_output: &Buffer,
        buf_input: &Buffer,
        buf_gamma: &Buffer,
        buf_grad_input: &Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_layer_norm_backward_nb(
                self.handle,
                queue,
                buf_grad_output.raw_handle() as *const _,
                buf_input.raw_handle() as *const _,
                buf_gamma.raw_handle() as *const _,
                buf_grad_input.raw_handle(),
                rows as u32,
                cols as u32,
                eps,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking layer_norm_backward dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Dispatch embedding lookup: output[i,j] = weights[indices[i],j].
    pub fn dispatch_embedding(
        &self,
        buf_weights: &Buffer,
        buf_indices: &Buffer,
        buf_out: &Buffer,
        seq_len: usize,
        embed_dim: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_embedding(
                self.handle,
                buf_weights.raw_handle() as *const _,
                buf_indices.raw_handle() as *const _,
                buf_out.raw_handle(),
                seq_len as u32,
                embed_dim as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Embedding dispatch failed".to_string())) }
    }

    /// Dispatch gather: output[i][j] uses indices to select from input along a dimension.
    /// Uses the same 3-buffer + 3-uint pattern.
    pub fn dispatch_gather(
        &self,
        buf_input: &Buffer,
        buf_indices: &Buffer,
        buf_out: &Buffer,
        rows: usize,
        in_cols: usize,
        out_cols: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_gather(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_indices.raw_handle() as *const _,
                buf_out.raw_handle(),
                rows as u32,
                in_cols as u32,
                out_cols as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Gather dispatch failed".to_string())) }
    }

    /// Non-blocking gather. Returns command buffer handle.
    pub fn dispatch_gather_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_indices: &Buffer,
        buf_out: &Buffer,
        rows: usize,
        in_cols: usize,
        out_cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_gather_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_indices.raw_handle() as *const _,
                buf_out.raw_handle(),
                rows as u32,
                in_cols as u32,
                out_cols as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking gather dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Dispatch index_select_dim0: output[i,j] = input[indices[i], j].
    /// Same buffer layout as embedding.
    pub fn dispatch_index_select_dim0(
        &self,
        buf_input: &Buffer,
        buf_indices: &Buffer,
        buf_out: &Buffer,
        num_indices: usize,
        cols: usize,
    ) -> Result<()> {
        // Use generic 3D dispatch with the template pipeline (not embedding FFI)
        let in_rows = 0u32; // not used by kernel but passed as buffer(3)
        self.dispatch_3d(
            &[buf_input, buf_indices], buf_out,
            &[in_rows, cols as u32, num_indices as u32],
            &[],
            (cols as u32, num_indices as u32, 1),
        )
    }

    /// Non-blocking index_select_dim0. Returns command buffer handle.
    pub fn dispatch_index_select_dim0_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_indices: &Buffer,
        buf_out: &Buffer,
        num_indices: usize,
        cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let in_rows = 0u32;
        self.dispatch_3d_nb(
            queue,
            &[buf_input, buf_indices], buf_out,
            &[in_rows, cols as u32, num_indices as u32],
            &[],
            (cols as u32, num_indices as u32, 1),
        )
    }

    /// Dispatch index_select_dim1: output[row, i] = input[row, indices[i]].
    pub fn dispatch_index_select_dim1(
        &self,
        buf_input: &Buffer,
        buf_indices: &Buffer,
        buf_out: &Buffer,
        rows: usize,
        in_cols: usize,
        num_indices: usize,
    ) -> Result<()> {
        self.dispatch_3d(
            &[buf_input, buf_indices], buf_out,
            &[rows as u32, in_cols as u32, num_indices as u32],
            &[],
            (num_indices as u32, rows as u32, 1),
        )
    }

    /// Non-blocking index_select_dim1. Returns command buffer handle.
    pub fn dispatch_index_select_dim1_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_indices: &Buffer,
        buf_out: &Buffer,
        rows: usize,
        in_cols: usize,
        num_indices: usize,
    ) -> Result<*mut std::ffi::c_void> {
        self.dispatch_3d_nb(
            queue,
            &[buf_input, buf_indices], buf_out,
            &[rows as u32, in_cols as u32, num_indices as u32],
            &[],
            (num_indices as u32, rows as u32, 1),
        )
    }

    /// Dispatch slice_dim0: output[row,col] = input[(start_row+row), col].
    pub fn dispatch_slice_dim0(
        &self, buf_input: &Buffer, buf_output: &Buffer, cols: usize, start_row: usize, out_rows: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_slice_dim0(
                self.handle, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), cols as u32, start_row as u32, out_rows as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("slice_dim0 dispatch failed".to_string())) }
    }

    /// Dispatch slice_dim1: output[row,col] = input[row, start_col+col].
    pub fn dispatch_slice_dim1(
        &self, buf_input: &Buffer, buf_output: &Buffer, in_cols: usize, out_cols: usize, start_col: usize, rows: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_slice_dim1(
                self.handle, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), in_cols as u32, out_cols as u32, start_col as u32, rows as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("slice_dim1 dispatch failed".to_string())) }
    }

    /// Dispatch concat_dim0: output = [a; b] stacked along rows.
    pub fn dispatch_concat_dim0(
        &self, buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer, rows_a: usize, cols: usize, total_rows: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_concat_dim0(
                self.handle, buf_a.raw_handle() as *const _, buf_b.raw_handle() as *const _,
                buf_output.raw_handle(), rows_a as u32, cols as u32, total_rows as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("concat_dim0 dispatch failed".to_string())) }
    }

    /// Dispatch concat_dim1: output = [a | b] stacked along columns.
    pub fn dispatch_concat_dim1(
        &self, buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer, rows: usize, cols_a: usize, cols_b: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_concat_dim1(
                self.handle, buf_a.raw_handle() as *const _, buf_b.raw_handle() as *const _,
                buf_output.raw_handle(), rows as u32, cols_a as u32, cols_b as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("concat_dim1 dispatch failed".to_string())) }
    }

    /// Dispatch add_bias: output[row,col] = input[row,col] + bias[col].
    pub fn dispatch_add_bias(
        &self, buf_input: &Buffer, buf_bias: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_add_bias(
                self.handle, buf_input.raw_handle() as *const _, buf_bias.raw_handle() as *const _,
                buf_output.raw_handle(), rows as u32, cols as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("add_bias dispatch failed".to_string())) }
    }

    // ── Non-blocking dispatch methods ─────────────────────────────────────

    /// Non-blocking binary elementwise with N-D strides. Returns command buffer handle.
    pub fn dispatch_elementwise_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let mut strides = [0u32; MAX_DIMS];
        strides[0] = 1;
        let mut shape = [0u32; MAX_DIMS];
        shape[0] = element_count as u32;
        self.dispatch_binary_nd_nb(
            queue,
            buf_a, &strides,
            buf_b, &strides,
            buf_out, &shape,
            1, element_count as u32,
        )
    }

    /// Non-blocking unary with N-D strides. Returns command buffer handle.
    pub fn dispatch_unary_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let mut strides = [0u32; MAX_DIMS];
        strides[0] = 1;
        let mut shape = [0u32; MAX_DIMS];
        shape[0] = element_count as u32;
        self.dispatch_unary_nd_nb(
            queue,
            buf_input, &strides,
            buf_out, &shape,
            1, element_count as u32,
        )
    }

    /// Non-blocking matmul. Returns command buffer handle.
    pub fn dispatch_matmul_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut std::ffi::c_void> {
        self.dispatch_matmul_batched_nb(queue, buf_a, buf_b, buf_c, m, n, k, 1, m * k, k * n)
    }

    /// Non-blocking batched matmul. Returns command buffer handle.
    pub fn dispatch_matmul_batched_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
        batch_size: usize,
        a_batch_stride: usize,
        b_batch_stride: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_matmul_batched_nb(
                self.handle,
                queue,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_c.raw_handle(),
                m as u32,
                n as u32,
                k as u32,
                batch_size as u32,
                a_batch_stride as u32,
                b_batch_stride as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking batched matmul dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Non-blocking softmax. Returns command buffer handle.
    pub fn dispatch_softmax_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_output: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_softmax_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_output.raw_handle(),
                rows as u32,
                cols as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking softmax dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Non-blocking batched softmax_causal. Returns command buffer handle.
    pub fn dispatch_softmax_causal_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_output: &Buffer,
        batch_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_softmax_causal_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_output.raw_handle(),
                batch_size as u32,
                rows as u32,
                cols as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking softmax_causal dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Non-blocking transpose. Returns command buffer handle.
    pub fn dispatch_transpose_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_output: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_transpose_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_output.raw_handle(),
                rows as u32,
                cols as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking transpose dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Non-blocking batched transpose. Returns command buffer handle.
    pub fn dispatch_transpose_batched_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_output: &Buffer,
        batch_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_transpose_batched_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_output.raw_handle(),
                batch_size as u32,
                rows as u32,
                cols as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking batched transpose dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Non-blocking scalar multiply. Returns command buffer handle.
    pub fn dispatch_scalar_mul_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_output: &Buffer,
        scale: f32,
        element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_scalar_mul_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_output.raw_handle(),
                scale,
                element_count as u64,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking scalar_mul dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Non-blocking fused dispatch. Returns command buffer handle.
    pub fn dispatch_fused_nb(
        &self,
        queue: *mut std::ffi::c_void,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let ptrs: Vec<*const ffi::GPUBufferHandle> = input_buffers
            .iter()
            .map(|b| b.raw_handle() as *const _)
            .collect();

        let cb = unsafe {
            ffi::gpu_bridge_compute_fused_nb(
                self.handle,
                queue,
                ptrs.as_ptr(),
                ptrs.len() as u32,
                buf_out.raw_handle(),
                element_count as u64,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking fused dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Non-blocking layer norm. Returns command buffer handle.
    pub fn dispatch_layer_norm_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_gamma: &Buffer,
        buf_beta: &Buffer,
        buf_out: &Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_layer_norm_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_gamma.raw_handle() as *const _,
                buf_beta.raw_handle() as *const _,
                buf_out.raw_handle(),
                rows as u32,
                cols as u32,
                eps,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking layer_norm dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Non-blocking embedding. Returns command buffer handle.
    pub fn dispatch_embedding_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_weights: &Buffer,
        buf_indices: &Buffer,
        buf_out: &Buffer,
        seq_len: usize,
        embed_dim: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_embedding_nb(
                self.handle,
                queue,
                buf_weights.raw_handle() as *const _,
                buf_indices.raw_handle() as *const _,
                buf_out.raw_handle(),
                seq_len as u32,
                embed_dim as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking embedding dispatch failed".to_string())) } else { Ok(cb) }
    }

    pub fn dispatch_slice_dim0_nb(
        &self, queue: *mut std::ffi::c_void, buf_input: &Buffer, buf_output: &Buffer,
        cols: usize, start_row: usize, out_rows: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_slice_dim0_nb(
                self.handle, queue, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), cols as u32, start_row as u32, out_rows as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking slice_dim0 dispatch failed".to_string())) } else { Ok(cb) }
    }

    pub fn dispatch_slice_dim1_nb(
        &self, queue: *mut std::ffi::c_void, buf_input: &Buffer, buf_output: &Buffer,
        in_cols: usize, out_cols: usize, start_col: usize, rows: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_slice_dim1_nb(
                self.handle, queue, buf_input.raw_handle() as *const _,
                buf_output.raw_handle(), in_cols as u32, out_cols as u32, start_col as u32, rows as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking slice_dim1 dispatch failed".to_string())) } else { Ok(cb) }
    }

    pub fn dispatch_concat_dim0_nb(
        &self, queue: *mut std::ffi::c_void, buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer,
        rows_a: usize, cols: usize, total_rows: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_concat_dim0_nb(
                self.handle, queue, buf_a.raw_handle() as *const _, buf_b.raw_handle() as *const _,
                buf_output.raw_handle(), rows_a as u32, cols as u32, total_rows as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking concat_dim0 dispatch failed".to_string())) } else { Ok(cb) }
    }

    pub fn dispatch_concat_dim1_nb(
        &self, queue: *mut std::ffi::c_void, buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer,
        rows: usize, cols_a: usize, cols_b: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_concat_dim1_nb(
                self.handle, queue, buf_a.raw_handle() as *const _, buf_b.raw_handle() as *const _,
                buf_output.raw_handle(), rows as u32, cols_a as u32, cols_b as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking concat_dim1 dispatch failed".to_string())) } else { Ok(cb) }
    }

    pub fn dispatch_add_bias_nb(
        &self, queue: *mut std::ffi::c_void, buf_input: &Buffer, buf_bias: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_add_bias_nb(
                self.handle, queue, buf_input.raw_handle() as *const _, buf_bias.raw_handle() as *const _,
                buf_output.raw_handle(), rows as u32, cols as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking add_bias dispatch failed".to_string())) } else { Ok(cb) }
    }

    // ── N-D stride-based dispatch methods ─────────────────────────────────

    /// Dispatch N-D binary element-wise op with stride arrays.
    pub fn dispatch_binary_nd(
        &self,
        buf_a: &Buffer, a_strides: &[u32; MAX_DIMS],
        buf_b: &Buffer, b_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_binary_nd(
                self.handle,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_out.raw_handle(),
                a_strides.as_ptr(),
                b_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Binary N-D dispatch failed".to_string())) }
    }

    /// Non-blocking N-D binary element-wise op.
    pub fn dispatch_binary_nd_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_a: &Buffer, a_strides: &[u32; MAX_DIMS],
        buf_b: &Buffer, b_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_binary_nd_nb(
                self.handle,
                queue,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_out.raw_handle(),
                a_strides.as_ptr(),
                b_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking binary N-D dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Dispatch N-D unary element-wise op with stride arrays.
    pub fn dispatch_unary_nd(
        &self,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_unary_nd(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                in_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Unary N-D dispatch failed".to_string())) }
    }

    /// Non-blocking N-D unary element-wise op.
    pub fn dispatch_unary_nd_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_unary_nd_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                in_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking unary N-D dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Dispatch N-D pow op with stride arrays and exponent constant.
    pub fn dispatch_pow_nd(
        &self,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, exponent: f32,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_pow_nd(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                in_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
                exponent,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Pow N-D dispatch failed".to_string())) }
    }

    /// Non-blocking N-D pow op.
    pub fn dispatch_pow_nd_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, exponent: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_pow_nd_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                in_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
                exponent,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking pow N-D dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Dispatch N-D clamp op with stride arrays, min and max constants.
    pub fn dispatch_clamp_nd(
        &self,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, min_val: f32, max_val: f32,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_clamp_nd(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                in_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
                min_val,
                max_val,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Clamp N-D dispatch failed".to_string())) }
    }

    /// Non-blocking N-D clamp op.
    pub fn dispatch_clamp_nd_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, min_val: f32, max_val: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_clamp_nd_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                in_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
                min_val,
                max_val,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking clamp N-D dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Dispatch ternary where op: 3 inputs with stride arrays.
    pub fn dispatch_where_nd(
        &self,
        buf_cond: &Buffer, cond_strides: &[u32; MAX_DIMS],
        buf_x: &Buffer, x_strides: &[u32; MAX_DIMS],
        buf_y: &Buffer, y_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_where_nd(
                self.handle,
                buf_cond.raw_handle() as *const _,
                buf_x.raw_handle() as *const _,
                buf_y.raw_handle() as *const _,
                buf_out.raw_handle(),
                cond_strides.as_ptr(),
                x_strides.as_ptr(),
                y_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Where N-D dispatch failed".to_string())) }
    }

    /// Non-blocking ternary where op.
    pub fn dispatch_where_nd_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_cond: &Buffer, cond_strides: &[u32; MAX_DIMS],
        buf_x: &Buffer, x_strides: &[u32; MAX_DIMS],
        buf_y: &Buffer, y_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_where_nd_nb(
                self.handle,
                queue,
                buf_cond.raw_handle() as *const _,
                buf_x.raw_handle() as *const _,
                buf_y.raw_handle() as *const _,
                buf_out.raw_handle(),
                cond_strides.as_ptr(),
                x_strides.as_ptr(),
                y_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking where N-D dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Dispatch masked_fill op: binary + fill_value scalar.
    pub fn dispatch_masked_fill_nd(
        &self,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_mask: &Buffer, mask_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, fill_value: f32,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_masked_fill_nd(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_mask.raw_handle() as *const _,
                buf_out.raw_handle(),
                in_strides.as_ptr(),
                mask_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
                fill_value,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("MaskedFill N-D dispatch failed".to_string())) }
    }

    /// Non-blocking masked_fill op.
    pub fn dispatch_masked_fill_nd_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_mask: &Buffer, mask_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, fill_value: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_masked_fill_nd_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_mask.raw_handle() as *const _,
                buf_out.raw_handle(),
                in_strides.as_ptr(),
                mask_strides.as_ptr(),
                out_shape.as_ptr(),
                ndim,
                numel,
                fill_value,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking masked_fill N-D dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Dispatch triu/tril op: batched 3D grid + diagonal constant.
    pub fn dispatch_triangular(
        &self,
        buf_input: &Buffer, buf_out: &Buffer,
        batch_size: usize, rows: usize, cols: usize, diagonal: i32,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_triangular(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                batch_size as u32,
                rows as u32,
                cols as u32,
                diagonal,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Triangular dispatch failed".to_string())) }
    }

    /// Non-blocking triu/tril op.
    pub fn dispatch_triangular_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_out: &Buffer,
        batch_size: usize, rows: usize, cols: usize, diagonal: i32,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_triangular_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                batch_size as u32,
                rows as u32,
                cols as u32,
                diagonal,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking triangular dispatch failed".to_string())) } else { Ok(cb) }
    }

    // ── Generic 3D dispatch for CNN ops ──────────────────────────────────

    /// Generic 3D dispatch: variable input buffers + output + uint/float params + 3D grid.
    pub fn dispatch_3d(
        &self,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        uint_params: &[u32],
        float_params: &[f32],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let ptrs: Vec<*const ffi::GPUBufferHandle> = input_buffers
            .iter()
            .map(|b| b.raw_handle() as *const _)
            .collect();
        let result = unsafe {
            ffi::gpu_bridge_compute_3d(
                self.handle,
                ptrs.as_ptr(),
                ptrs.len() as u32,
                buf_out.raw_handle(),
                uint_params.as_ptr(),
                uint_params.len() as u32,
                float_params.as_ptr(),
                float_params.len() as u32,
                grid.0,
                grid.1,
                grid.2,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("3D dispatch failed".to_string())) }
    }

    /// Non-blocking generic 3D dispatch.
    pub fn dispatch_3d_nb(
        &self,
        queue: *mut std::ffi::c_void,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        uint_params: &[u32],
        float_params: &[f32],
        grid: (u32, u32, u32),
    ) -> Result<*mut std::ffi::c_void> {
        let ptrs: Vec<*const ffi::GPUBufferHandle> = input_buffers
            .iter()
            .map(|b| b.raw_handle() as *const _)
            .collect();
        let cb = unsafe {
            ffi::gpu_bridge_compute_3d_nb(
                self.handle,
                queue,
                ptrs.as_ptr(),
                ptrs.len() as u32,
                buf_out.raw_handle(),
                uint_params.as_ptr(),
                uint_params.len() as u32,
                float_params.as_ptr(),
                float_params.len() as u32,
                grid.0,
                grid.1,
                grid.2,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking 3D dispatch failed".to_string())) } else { Ok(cb) }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { ffi::gpu_bridge_destroy_compute(self.handle) };
    }
}

/// Caches compiled compute pipelines by kernel name.
/// Uses Arc<ComputePipeline> so the lock is released before GPU dispatch.
pub struct KernelRegistry {
    pipelines: Mutex<HashMap<String, Arc<ComputePipeline>>>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        KernelRegistry {
            pipelines: Mutex::new(HashMap::new()),
        }
    }

    /// Get or compile a pipeline for the given op. Returns Arc so caller
    /// can dispatch without holding the registry lock.
    pub fn get_or_create(
        &self,
        device: &Device,
        kernel_source: &str,
        function_name: &str,
    ) -> Result<Arc<ComputePipeline>> {
        let mut map = self.pipelines.lock().unwrap();
        if let Some(pipeline) = map.get(function_name) {
            return Ok(Arc::clone(pipeline));
        }
        let pipeline = Arc::new(ComputePipeline::new(device, kernel_source, function_name)?);
        map.insert(function_name.to_string(), Arc::clone(&pipeline));
        Ok(pipeline)
    }

    /// Resolve the kernel source and function name for a given base name and dtype.
    /// Resolve kernel source and function name for a given base_name + dtype.
    /// Base names are unsuffixed (e.g. "matmul", "softmax", "elementwise_add").
    /// Returns (MSL source, suffixed function name like "matmul_f32").
    pub fn resolve_kernel(base_name: &str, dtype: DType) -> (String, String) {
        use crate::kernel_templates as kt;

        let suffix = kt::dtype_suffix(dtype);
        let func_name = format!("{}{}", base_name, suffix);

        let source = match base_name {
            // Binary element-wise
            n if n.starts_with("elementwise_add") || n.starts_with("elementwise_sub")
                || n.starts_with("elementwise_mul") || n.starts_with("elementwise_div") =>
                kt::binary_kernel_source(dtype),
            // Unary: neg, abs, sign (all numeric types)
            n if n.starts_with("elementwise_neg") || n.starts_with("elementwise_abs")
                || n.starts_with("elementwise_sign") =>
                kt::unary_kernel_source(dtype),
            // Unary: float ops (exp, log, sqrt, relu, tanh)
            n if n.starts_with("elementwise_") => {
                if dtype.is_float() {
                    kt::float_unary_kernel_source(dtype)
                } else {
                    kt::unary_kernel_source(dtype)
                }
            }
            "lt" | "gt" | "le" | "ge" | "eq" | "ne" => kt::comparison_kernel_source(dtype),
            "bitwise_and" | "bitwise_or" | "bitwise_xor" => kt::bitwise_binary_kernel_source(dtype),
            "bitwise_not" => kt::bitwise_not_kernel_source(dtype),
            "shl" | "shr" => kt::shift_kernel_source(dtype),
            "mod" => kt::mod_kernel_source(dtype),
            "elem_min" | "elem_max" => kt::elem_minmax_kernel_source(dtype),
            "logical_not" => return (kt::logical_not_kernel_source(), "logical_not_bool".to_string()),
            "scalar_mul" => kt::scalar_mul_kernel_source(dtype),
            "pow" => kt::pow_kernel_source(dtype),
            "clamp" => kt::clamp_kernel_source(dtype),
            "gelu" => kt::gelu_kernel_source(dtype),
            "gelu_exact" => kt::gelu_exact_kernel_source(dtype),
            "sigmoid" => kt::sigmoid_kernel_source(dtype),
            "softmax" => kt::softmax_kernel_source(dtype),
            "log_softmax" => kt::log_softmax_kernel_source(dtype),
            "softmax_causal" => kt::softmax_causal_kernel_source(dtype),
            "argmax" => kt::argmax_kernel_source(dtype),
            "sum" => kt::sum_kernel_source(dtype),
            "mean" => kt::mean_kernel_source(dtype),
            "var" => kt::var_kernel_source(dtype),
            "add_bias" => kt::add_bias_kernel_source(dtype),
            "where" => kt::where_kernel_source(dtype),
            "masked_fill" => kt::masked_fill_kernel_source(dtype),
            "triu" => kt::triu_kernel_source(dtype),
            "tril" => kt::tril_kernel_source(dtype),
            "gather_dim0" => kt::gather_kernel_source(dtype, 0),
            "gather_dim1" => kt::gather_kernel_source(dtype, 1),
            "index_select_dim0" => kt::index_select_kernel_source(dtype, 0),
            "index_select_dim1" => kt::index_select_kernel_source(dtype, 1),
            "matmul" => kt::matmul_kernel_source(dtype),
            "layer_norm" => kt::layer_norm_kernel_source(dtype),
            "embedding" => kt::embedding_kernel_source(dtype),
            "conv1d" => kt::conv1d_kernel_source(dtype),
            "conv2d" => kt::conv2d_kernel_source(dtype),
            "batch_norm" => kt::batch_norm_kernel_source(dtype),
            "max_pool2d" => kt::max_pool2d_kernel_source(dtype),
            "max_pool2d_idx" => kt::max_pool2d_with_indices_kernel_source(dtype),
            "avg_pool2d" => kt::avg_pool2d_kernel_source(dtype),
            "softmax_backward" => kt::softmax_backward_kernel_source(dtype),
            "layer_norm_backward" => kt::layer_norm_backward_kernel_source(dtype),
            "conv1d_backward_input" => kt::conv1d_backward_input_kernel_source(dtype),
            "conv2d_backward_input" => kt::conv2d_backward_input_kernel_source(dtype),
            "embedding_backward" => kt::embedding_backward_kernel_source(dtype),
            "batch_norm_backward" => kt::batch_norm_backward_kernel_source(dtype),
            "threshold_backward" => kt::threshold_backward_kernel_source(dtype),
            "tanh_backward" => kt::tanh_backward_kernel_source(dtype),
            "sigmoid_backward" => kt::sigmoid_backward_kernel_source(dtype),
            "gelu_backward" => kt::gelu_backward_kernel_source(dtype),
            "gelu_tanh_backward" => kt::gelu_tanh_backward_kernel_source(dtype),
            "gelu_exact_backward" => kt::gelu_exact_backward_kernel_source(dtype),
            "max_pool2d_backward" => kt::max_pool2d_backward_kernel_source(dtype),
            "transpose" => return (kt::byte_copy_transpose_source(dtype.size_bytes()), format!("transpose_bytes{}", dtype.size_bytes())),
            "transpose_batched" => kt::transpose_batched_kernel_source(dtype),
            "copy_strided" => kt::copy_strided_kernel_source(dtype),
            "slice_dim0" => return (kt::byte_copy_slice_dim0_source(dtype.size_bytes()), format!("slice_dim0_bytes{}", dtype.size_bytes())),
            "slice_dim1" => return (kt::byte_copy_slice_dim1_source(dtype.size_bytes()), format!("slice_dim1_bytes{}", dtype.size_bytes())),
            "concat_dim0" => return (kt::byte_copy_concat_dim0_source(dtype.size_bytes()), format!("concat_dim0_bytes{}", dtype.size_bytes())),
            "concat_dim1" => return (kt::byte_copy_concat_dim1_source(dtype.size_bytes()), format!("concat_dim1_bytes{}", dtype.size_bytes())),
            _ => kt::binary_kernel_source(dtype),
        };

        (source, func_name)
    }

    /// Resolve kernel source and function name for a cast operation.
    /// Cast is special: kernel name encodes both source and target dtypes.
    pub fn resolve_cast_kernel(src: DType, dst: DType) -> (String, String) {
        use crate::kernel_templates as kt;
        let source = kt::cast_kernel_source(src, dst);
        let func_name = format!("cast{}_to{}", kt::dtype_suffix(src), kt::dtype_suffix(dst));
        (source, func_name)
    }

    /// Resolve kernel source and function name for a quantize operation.
    pub fn resolve_quantize_kernel(src: DType, dst: DType, scale: f32, zero_point: i32) -> (String, String) {
        use crate::kernel_templates as kt;
        let source = kt::quantize_kernel_source(src, dst, scale, zero_point);
        let func_name = format!("quantize{}_to{}", kt::dtype_suffix(src), kt::dtype_suffix(dst));
        (source, func_name)
    }

    /// Resolve kernel source and function name for a dequantize operation.
    pub fn resolve_dequantize_kernel(src: DType, dst: DType, scale: f32, zero_point: i32) -> (String, String) {
        use crate::kernel_templates as kt;
        let source = kt::dequantize_kernel_source(src, dst, scale, zero_point);
        let func_name = format!("dequantize{}_to{}", kt::dtype_suffix(src), kt::dtype_suffix(dst));
        (source, func_name)
    }

    /// Dispatch a binary op through the registry.
    pub fn dispatch_binary(
        &self,
        device: &Device,
        function_name: &str,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        self.dispatch_binary_typed(device, function_name, DType::Float32, buf_a, buf_b, buf_out, element_count)
    }

    /// Dispatch a binary op with dtype-aware kernel selection.
    pub fn dispatch_binary_typed(
        &self,
        device: &Device,
        function_name: &str,
        dtype: DType,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel(function_name, dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_elementwise(buf_a, buf_b, buf_out, element_count)
    }

    /// Dispatch a unary op through the registry.
    pub fn dispatch_unary(
        &self,
        device: &Device,
        function_name: &str,
        buf_input: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        self.dispatch_unary_typed(device, function_name, DType::Float32, buf_input, buf_out, element_count)
    }

    /// Dispatch a unary op with dtype-aware kernel selection.
    pub fn dispatch_unary_typed(
        &self,
        device: &Device,
        function_name: &str,
        dtype: DType,
        buf_input: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel(function_name, dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_unary(buf_input, buf_out, element_count)
    }

    /// Dispatch matmul through the registry.
    pub fn dispatch_matmul(
        &self,
        device: &Device,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.dispatch_matmul_typed(device, DType::Float32, buf_a, buf_b, buf_c, m, n, k)
    }

    /// Dispatch matmul with dtype-aware kernel selection.
    pub fn dispatch_matmul_typed(
        &self,
        device: &Device,
        dtype: DType,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("matmul", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_matmul(buf_a, buf_b, buf_c, m, n, k)
    }

    /// Dispatch batched matmul with dtype-aware kernel selection.
    pub fn dispatch_matmul_batched_typed(
        &self,
        device: &Device,
        dtype: DType,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
        batch_size: usize,
        a_batch_stride: usize,
        b_batch_stride: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("matmul", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_matmul_batched(buf_a, buf_b, buf_c, m, n, k, batch_size, a_batch_stride, b_batch_stride)
    }

    /// Dispatch a fused kernel with variable input buffers.
    pub fn dispatch_fused(
        &self,
        device: &Device,
        kernel_source: &str,
        function_name: &str,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, kernel_source, function_name)?;
        pipeline.dispatch_fused(input_buffers, buf_out, element_count)
    }

    /// Dispatch a fused N-D kernel with stride arrays per input.
    pub fn dispatch_fused_nd(
        &self,
        device: &Device,
        kernel_source: &str,
        function_name: &str,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        input_strides: &[&[u32; MAX_DIMS]],
        out_shape: &[u32; MAX_DIMS],
        ndim: u32,
        numel: u32,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, kernel_source, function_name)?;
        pipeline.dispatch_fused_nd(input_buffers, buf_out, input_strides, out_shape, ndim, numel)
    }

    pub fn dispatch_softmax(
        &self, device: &Device, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        self.dispatch_softmax_typed(device, DType::Float32, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_softmax_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("softmax", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_log_softmax_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("log_softmax", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_transpose(
        &self, device: &Device, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        self.dispatch_transpose_typed(device, DType::Float32, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_transpose_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("transpose", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_transpose(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_transpose_batched_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        batch_size: usize, rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("transpose_batched", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_transpose_batched(buf_input, buf_output, batch_size, rows, cols)
    }

    pub fn dispatch_scalar_mul(
        &self, device: &Device, buf_input: &Buffer, buf_output: &Buffer,
        scale: f32, element_count: usize,
    ) -> Result<()> {
        self.dispatch_scalar_mul_typed(device, DType::Float32, buf_input, buf_output, scale, element_count)
    }

    pub fn dispatch_scalar_mul_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        scale: f32, element_count: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("scalar_mul", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_scalar_mul(buf_input, buf_output, scale, element_count)
    }

    pub fn dispatch_pow_nd_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, exponent: f32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("pow", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_pow_nd(buf_input, in_strides, buf_out, out_shape, ndim, numel, exponent)
    }

    pub fn dispatch_pow_nd_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, exponent: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("pow", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_pow_nd_nb(queue, buf_input, in_strides, buf_out, out_shape, ndim, numel, exponent)
    }

    pub fn dispatch_clamp_nd_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, min_val: f32, max_val: f32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("clamp", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_clamp_nd(buf_input, in_strides, buf_out, out_shape, ndim, numel, min_val, max_val)
    }

    pub fn dispatch_clamp_nd_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, min_val: f32, max_val: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("clamp", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_clamp_nd_nb(queue, buf_input, in_strides, buf_out, out_shape, ndim, numel, min_val, max_val)
    }

    // ── Where (ternary) ──────────────────────────────────────────────

    pub fn dispatch_where_nd_typed(
        &self, device: &Device, dtype: DType,
        buf_cond: &Buffer, cond_strides: &[u32; MAX_DIMS],
        buf_x: &Buffer, x_strides: &[u32; MAX_DIMS],
        buf_y: &Buffer, y_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("where", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_where_nd(buf_cond, cond_strides, buf_x, x_strides, buf_y, y_strides, buf_out, out_shape, ndim, numel)
    }

    pub fn dispatch_where_nd_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_cond: &Buffer, cond_strides: &[u32; MAX_DIMS],
        buf_x: &Buffer, x_strides: &[u32; MAX_DIMS],
        buf_y: &Buffer, y_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("where", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_where_nd_nb(queue, buf_cond, cond_strides, buf_x, x_strides, buf_y, y_strides, buf_out, out_shape, ndim, numel)
    }

    // ── MaskedFill ──────────────────────────────────────────────────

    pub fn dispatch_masked_fill_nd_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_mask: &Buffer, mask_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, fill_value: f32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("masked_fill", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_masked_fill_nd(buf_input, in_strides, buf_mask, mask_strides, buf_out, out_shape, ndim, numel, fill_value)
    }

    pub fn dispatch_masked_fill_nd_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_mask: &Buffer, mask_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, fill_value: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("masked_fill", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_masked_fill_nd_nb(queue, buf_input, in_strides, buf_mask, mask_strides, buf_out, out_shape, ndim, numel, fill_value)
    }

    // ── Triu / Tril ─────────────────────────────────────────────────

    pub fn dispatch_triu_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, buf_out: &Buffer,
        batch_size: usize, rows: usize, cols: usize, diagonal: i32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("triu", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_triangular(buf_input, buf_out, batch_size, rows, cols, diagonal)
    }

    pub fn dispatch_triu_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_out: &Buffer,
        batch_size: usize, rows: usize, cols: usize, diagonal: i32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("triu", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_triangular_nb(queue, buf_input, buf_out, batch_size, rows, cols, diagonal)
    }

    pub fn dispatch_tril_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, buf_out: &Buffer,
        batch_size: usize, rows: usize, cols: usize, diagonal: i32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("tril", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_triangular(buf_input, buf_out, batch_size, rows, cols, diagonal)
    }

    pub fn dispatch_tril_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_out: &Buffer,
        batch_size: usize, rows: usize, cols: usize, diagonal: i32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("tril", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_triangular_nb(queue, buf_input, buf_out, batch_size, rows, cols, diagonal)
    }

    /// Dispatch GELU with dtype-aware kernel selection (uses unary dispatch pattern).
    pub fn dispatch_gelu_typed(
        &self,
        device: &Device,
        dtype: DType,
        buf_input: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("gelu", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_unary(buf_input, buf_out, element_count)
    }

    /// Dispatch sigmoid with dtype-aware kernel selection.
    pub fn dispatch_sigmoid_typed(
        &self,
        device: &Device,
        dtype: DType,
        buf_input: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("sigmoid", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_unary(buf_input, buf_out, element_count)
    }

    /// Dispatch layer normalization with dtype-aware kernel selection.
    pub fn dispatch_layer_norm_typed(
        &self,
        device: &Device,
        dtype: DType,
        buf_input: &Buffer,
        buf_gamma: &Buffer,
        buf_beta: &Buffer,
        buf_out: &Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("layer_norm", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_layer_norm(buf_input, buf_gamma, buf_beta, buf_out, rows, cols, eps)
    }

    pub fn dispatch_softmax_backward_typed(
        &self, device: &Device, dtype: DType,
        buf_grad_output: &Buffer, buf_output: &Buffer, buf_grad_input: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("softmax_backward", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_backward(buf_grad_output, buf_output, buf_grad_input, rows, cols)
    }

    pub fn dispatch_layer_norm_backward_typed(
        &self, device: &Device, dtype: DType,
        buf_grad_output: &Buffer, buf_input: &Buffer, buf_gamma: &Buffer, buf_grad_input: &Buffer,
        rows: usize, cols: usize, eps: f32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("layer_norm_backward", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_layer_norm_backward(buf_grad_output, buf_input, buf_gamma, buf_grad_input, rows, cols, eps)
    }

    /// Dispatch embedding lookup with dtype-aware kernel selection.
    pub fn dispatch_embedding_typed(
        &self,
        device: &Device,
        dtype: DType,
        buf_weights: &Buffer,
        buf_indices: &Buffer,
        buf_out: &Buffer,
        seq_len: usize,
        embed_dim: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("embedding", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_embedding(buf_weights, buf_indices, buf_out, seq_len, embed_dim)
    }

    // ── New op dispatch methods (typed) ──────────────────────────────────

    pub fn dispatch_slice_dim0_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        cols: usize, start_row: usize, out_rows: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("slice_dim0", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_slice_dim0(buf_input, buf_output, cols, start_row, out_rows)
    }

    pub fn dispatch_slice_dim1_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        in_cols: usize, out_cols: usize, start_col: usize, rows: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("slice_dim1", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_slice_dim1(buf_input, buf_output, in_cols, out_cols, start_col, rows)
    }

    pub fn dispatch_concat_dim0_typed(
        &self, device: &Device, dtype: DType, buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer,
        rows_a: usize, cols: usize, total_rows: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("concat_dim0", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_concat_dim0(buf_a, buf_b, buf_output, rows_a, cols, total_rows)
    }

    pub fn dispatch_concat_dim1_typed(
        &self, device: &Device, dtype: DType, buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer,
        rows: usize, cols_a: usize, cols_b: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("concat_dim1", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_concat_dim1(buf_a, buf_b, buf_output, rows, cols_a, cols_b)
    }

    pub fn dispatch_add_bias_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_bias: &Buffer, buf_output: &Buffer,
        numel: usize, num_channels: usize, channel_stride: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("add_bias", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_3d(
            &[buf_input, buf_bias],
            buf_output,
            &[numel as u32, num_channels as u32, channel_stride as u32],
            &[],
            (numel as u32, 1, 1),
        )
    }

    pub fn dispatch_softmax_causal_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        batch_size: usize, rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("softmax_causal", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_causal(buf_input, buf_output, batch_size, rows, cols)
    }

    pub fn dispatch_sum_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("sum", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_mean_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("mean", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_var_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize, correction: u32,
    ) -> Result<()> {
        use crate::kernel_templates as kt;
        let source = kt::var_kernel_source_with_correction(dtype, correction);
        let s = kt::dtype_suffix(dtype);
        let func = format!("var{s}_c{correction}");
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_var_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize, correction: u32,
    ) -> Result<*mut std::ffi::c_void> {
        use crate::kernel_templates as kt;
        let source = kt::var_kernel_source_with_correction(dtype, correction);
        let s = kt::dtype_suffix(dtype);
        let func = format!("var{s}_c{correction}");
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_argmax_typed(
        &self, device: &Device, input_dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("argmax", input_dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        // Reuse softmax dispatch (same buffer layout: input, output, rows, cols; 1D dispatch per row)
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    // ── Non-blocking new op dispatch methods (typed) ──────────────────────

    pub fn dispatch_slice_dim0_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, cols: usize, start_row: usize, out_rows: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("slice_dim0", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_slice_dim0_nb(queue, buf_input, buf_output, cols, start_row, out_rows)
    }

    pub fn dispatch_slice_dim1_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, in_cols: usize, out_cols: usize, start_col: usize, rows: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("slice_dim1", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_slice_dim1_nb(queue, buf_input, buf_output, in_cols, out_cols, start_col, rows)
    }

    pub fn dispatch_concat_dim0_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer, rows_a: usize, cols: usize, total_rows: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("concat_dim0", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_concat_dim0_nb(queue, buf_a, buf_b, buf_output, rows_a, cols, total_rows)
    }

    pub fn dispatch_concat_dim1_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer, rows: usize, cols_a: usize, cols_b: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("concat_dim1", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_concat_dim1_nb(queue, buf_a, buf_b, buf_output, rows, cols_a, cols_b)
    }

    pub fn dispatch_add_bias_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_bias: &Buffer, buf_output: &Buffer,
        numel: usize, num_channels: usize, channel_stride: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("add_bias", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_3d_nb(
            queue,
            &[buf_input, buf_bias],
            buf_output,
            &[numel as u32, num_channels as u32, channel_stride as u32],
            &[],
            (numel as u32, 1, 1),
        )
    }

    pub fn dispatch_softmax_causal_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, batch_size: usize, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("softmax_causal", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_causal_nb(queue, buf_input, buf_output, batch_size, rows, cols)
    }

    pub fn dispatch_sum_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("sum", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_mean_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("mean", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_argmax_typed_nb(
        &self, device: &Device, input_dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("argmax", input_dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }

    // ── Non-blocking dispatch methods ─────────────────────────────────────

    pub fn dispatch_gelu_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_out: &Buffer, element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("gelu", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_unary_nb(queue, buf_input, buf_out, element_count)
    }

    pub fn dispatch_sigmoid_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_out: &Buffer, element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("sigmoid", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_unary_nb(queue, buf_input, buf_out, element_count)
    }

    pub fn dispatch_layer_norm_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_gamma: &Buffer, buf_beta: &Buffer, buf_out: &Buffer,
        rows: usize, cols: usize, eps: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("layer_norm", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_layer_norm_nb(queue, buf_input, buf_gamma, buf_beta, buf_out, rows, cols, eps)
    }

    pub fn dispatch_softmax_backward_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_grad_output: &Buffer, buf_output: &Buffer, buf_grad_input: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("softmax_backward", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_backward_nb(queue, buf_grad_output, buf_output, buf_grad_input, rows, cols)
    }

    pub fn dispatch_layer_norm_backward_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_grad_output: &Buffer, buf_input: &Buffer, buf_gamma: &Buffer, buf_grad_input: &Buffer,
        rows: usize, cols: usize, eps: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("layer_norm_backward", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_layer_norm_backward_nb(queue, buf_grad_output, buf_input, buf_gamma, buf_grad_input, rows, cols, eps)
    }

    pub fn dispatch_embedding_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_weights: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        seq_len: usize, embed_dim: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("embedding", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_embedding_nb(queue, buf_weights, buf_indices, buf_out, seq_len, embed_dim)
    }

    pub fn dispatch_binary_typed_nb(
        &self, device: &Device, function_name: &str, dtype: DType, queue: *mut std::ffi::c_void,
        buf_a: &Buffer, buf_b: &Buffer, buf_out: &Buffer, element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel(function_name, dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_elementwise_nb(queue, buf_a, buf_b, buf_out, element_count)
    }

    pub fn dispatch_unary_typed_nb(
        &self, device: &Device, function_name: &str, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_out: &Buffer, element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel(function_name, dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_unary_nb(queue, buf_input, buf_out, element_count)
    }

    pub fn dispatch_matmul_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_a: &Buffer, buf_b: &Buffer, buf_c: &Buffer, m: usize, n: usize, k: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("matmul", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_matmul_nb(queue, buf_a, buf_b, buf_c, m, n, k)
    }

    pub fn dispatch_matmul_batched_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_a: &Buffer, buf_b: &Buffer, buf_c: &Buffer,
        m: usize, n: usize, k: usize,
        batch_size: usize, a_batch_stride: usize, b_batch_stride: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("matmul", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_matmul_batched_nb(queue, buf_a, buf_b, buf_c, m, n, k, batch_size, a_batch_stride, b_batch_stride)
    }

    pub fn dispatch_softmax_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("softmax", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_log_softmax_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("log_softmax", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_transpose_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("transpose", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_transpose_nb(queue, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_transpose_batched_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, batch_size: usize, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("transpose_batched", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_transpose_batched_nb(queue, buf_input, buf_output, batch_size, rows, cols)
    }

    pub fn dispatch_scalar_mul_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, scale: f32, element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("scalar_mul", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_scalar_mul_nb(queue, buf_input, buf_output, scale, element_count)
    }

    pub fn dispatch_fused_nb(
        &self, device: &Device, kernel_source: &str, function_name: &str,
        queue: *mut std::ffi::c_void, input_buffers: &[&Buffer], buf_out: &Buffer,
        element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let pipeline = self.get_or_create(device, kernel_source, function_name)?;
        pipeline.dispatch_fused_nb(queue, input_buffers, buf_out, element_count)
    }

    /// Non-blocking N-D fused kernel dispatch with stride arrays per input.
    pub fn dispatch_fused_nd_nb(
        &self, device: &Device, kernel_source: &str, function_name: &str,
        queue: *mut std::ffi::c_void, input_buffers: &[&Buffer], buf_out: &Buffer,
        input_strides: &[&[u32; MAX_DIMS]], out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<*mut std::ffi::c_void> {
        let pipeline = self.get_or_create(device, kernel_source, function_name)?;
        pipeline.dispatch_fused_nd_nb(queue, input_buffers, buf_out, input_strides, out_shape, ndim, numel)
    }

    // ── N-D typed dispatch methods ────────────────────────────────────────

    /// Dispatch N-D binary element-wise op with dtype-aware kernel selection.
    pub fn dispatch_binary_nd_typed(
        &self, device: &Device, function_name: &str, dtype: DType,
        buf_a: &Buffer, a_strides: &[u32; MAX_DIMS],
        buf_b: &Buffer, b_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel(function_name, dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_binary_nd(buf_a, a_strides, buf_b, b_strides, buf_out, out_shape, ndim, numel)
    }

    /// Non-blocking N-D binary element-wise op with dtype-aware kernel selection.
    pub fn dispatch_binary_nd_typed_nb(
        &self, device: &Device, function_name: &str, dtype: DType,
        queue: *mut std::ffi::c_void,
        buf_a: &Buffer, a_strides: &[u32; MAX_DIMS],
        buf_b: &Buffer, b_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel(function_name, dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_binary_nd_nb(queue, buf_a, a_strides, buf_b, b_strides, buf_out, out_shape, ndim, numel)
    }

    /// Dispatch N-D unary element-wise op with dtype-aware kernel selection.
    pub fn dispatch_unary_nd_typed(
        &self, device: &Device, function_name: &str, dtype: DType,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel(function_name, dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_unary_nd(buf_input, in_strides, buf_out, out_shape, ndim, numel)
    }

    /// Non-blocking N-D unary element-wise op with dtype-aware kernel selection.
    pub fn dispatch_unary_nd_typed_nb(
        &self, device: &Device, function_name: &str, dtype: DType,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel(function_name, dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_unary_nd_nb(queue, buf_input, in_strides, buf_out, out_shape, ndim, numel)
    }

    // ── Gather dispatch (typed) ──────────────────────────────────────────

    pub fn dispatch_gather_typed(
        &self, device: &Device, dtype: DType, kernel_base: &str,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        rows: usize, in_cols: usize, out_cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel(kernel_base, dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_gather(buf_input, buf_indices, buf_out, rows, in_cols, out_cols)
    }

    pub fn dispatch_gather_typed_nb(
        &self, device: &Device, dtype: DType, kernel_base: &str,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        rows: usize, in_cols: usize, out_cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel(kernel_base, dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_gather_nb(queue, buf_input, buf_indices, buf_out, rows, in_cols, out_cols)
    }

    // ── IndexSelect dispatch (typed) ─────────────────────────────────────

    pub fn dispatch_index_select_dim0_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        num_indices: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("index_select_dim0", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_index_select_dim0(buf_input, buf_indices, buf_out, num_indices, cols)
    }

    pub fn dispatch_index_select_dim0_typed_nb(
        &self, device: &Device, dtype: DType,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        num_indices: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("index_select_dim0", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_index_select_dim0_nb(queue, buf_input, buf_indices, buf_out, num_indices, cols)
    }

    pub fn dispatch_index_select_dim1_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        rows: usize, in_cols: usize, num_indices: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("index_select_dim1", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_index_select_dim1(buf_input, buf_indices, buf_out, rows, in_cols, num_indices)
    }

    pub fn dispatch_index_select_dim1_typed_nb(
        &self, device: &Device, dtype: DType,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        rows: usize, in_cols: usize, num_indices: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("index_select_dim1", dtype);
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_index_select_dim1_nb(queue, buf_input, buf_indices, buf_out, rows, in_cols, num_indices)
    }

    // ── CNN ops dispatch (generic 3D) ────────────────────────────────────

    pub fn dispatch_cnn_3d(
        &self,
        device: &Device,
        kernel_source: &str,
        function_name: &str,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        uint_params: &[u32],
        float_params: &[f32],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, kernel_source, function_name)?;
        pipeline.dispatch_3d(input_buffers, buf_out, uint_params, float_params, grid)
    }

    pub fn dispatch_cnn_3d_nb(
        &self,
        device: &Device,
        kernel_source: &str,
        function_name: &str,
        queue: *mut std::ffi::c_void,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        uint_params: &[u32],
        float_params: &[f32],
        grid: (u32, u32, u32),
    ) -> Result<*mut std::ffi::c_void> {
        let pipeline = self.get_or_create(device, kernel_source, function_name)?;
        pipeline.dispatch_3d_nb(queue, input_buffers, buf_out, uint_params, float_params, grid)
    }
}

/// Get or create the device-level shared command queue.
pub fn get_shared_queue(device: &Device) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_get_shared_queue(device.raw_handle() as *mut _) }
}

/// Wait for a command buffer to complete (consumes the retained reference).
pub fn wait_command_buffer(cb: *mut std::ffi::c_void) {
    unsafe { ffi::gpu_bridge_wait_command_buffer(cb) }
}

/// Begin batch encoding: creates a single command buffer for all subsequent _nb dispatches.
/// Returns a non-null handle on success (unretained — only for null-checking).
/// All _nb calls will encode into this CB until end_batch() or abort_batch() is called.
pub fn begin_batch(queue: *mut std::ffi::c_void) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_begin_batch(queue) }
}

/// End batch encoding: commits the batch command buffer and returns its handle for waiting.
/// The returned handle is retained — pass to wait_command_buffer() which consumes it.
pub fn end_batch() -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_end_batch() }
}

/// Abort batch encoding: discards the batch command buffer without committing.
/// Call this when an error occurs mid-batch to clean up batch state.
pub fn abort_batch() {
    unsafe { ffi::gpu_bridge_abort_batch() }
}

// ── Concurrent queue pool ──────────────────────────────────────────

/// Get a command queue from the pool by index. Queues are lazily created and reused.
pub fn get_queue(device: &Device, index: u32) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_get_queue(device.raw_handle(), index) }
}

// ── Batch context system ──────────────────────────────────────────

/// Create a batch context: a new command buffer on the given queue, keyed by context_id.
pub fn set_batch_context(context_id: u32, queue: *mut std::ffi::c_void) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_set_batch_context(context_id, queue) }
}

/// Commit a batch context: commits the command buffer and returns its handle for waiting.
pub fn commit_batch_context(context_id: u32) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_commit_batch_context(context_id) }
}

/// Set the active context ID. Subsequent _nb dispatch calls will use this context's CB.
pub fn set_active_context(context_id: u32) {
    unsafe { ffi::gpu_bridge_set_active_context(context_id) }
}

// ── MTLEvent synchronization ──────────────────────────────────────

/// Create an MTLEvent for inter-queue synchronization.
pub fn create_event(device: &Device) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_create_event(device.raw_handle()) }
}

/// Encode a signal on a command buffer: sets event value after all prior work completes.
pub fn encode_signal_event(cb: *mut std::ffi::c_void, event: *mut std::ffi::c_void, value: u64) {
    unsafe { ffi::gpu_bridge_encode_signal_event(cb, event, value) }
}

/// Encode a wait on a command buffer: blocks until event reaches the given value.
pub fn encode_wait_event(cb: *mut std::ffi::c_void, event: *mut std::ffi::c_void, value: u64) {
    unsafe { ffi::gpu_bridge_encode_wait_event(cb, event, value) }
}

/// Destroy an MTLEvent (release the retained reference).
pub fn destroy_event(event: *mut std::ffi::c_void) {
    unsafe { ffi::gpu_bridge_destroy_event(event) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> Option<Device> {
        Device::new().ok()
    }

    fn f32_as_bytes(data: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
    }

    #[test]
    fn elementwise_add() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, f32_as_bytes(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        let b = Buffer::from_bytes(&device, f32_as_bytes(&[10.0, 20.0, 30.0, 40.0])).unwrap();
        let out = Buffer::new(&device, 16).unwrap();
        registry.dispatch_binary(&device, "elementwise_add", &a, &b, &out, 4).unwrap();
        assert_eq!(unsafe { out.as_slice::<f32>() }, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn elementwise_sub() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, f32_as_bytes(&[10.0, 20.0, 30.0, 40.0])).unwrap();
        let b = Buffer::from_bytes(&device, f32_as_bytes(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        let out = Buffer::new(&device, 16).unwrap();
        registry.dispatch_binary(&device, "elementwise_sub", &a, &b, &out, 4).unwrap();
        assert_eq!(unsafe { out.as_slice::<f32>() }, &[9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn elementwise_mul() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, f32_as_bytes(&[2.0, 3.0, 4.0, 5.0])).unwrap();
        let b = Buffer::from_bytes(&device, f32_as_bytes(&[10.0, 10.0, 10.0, 10.0])).unwrap();
        let out = Buffer::new(&device, 16).unwrap();
        registry.dispatch_binary(&device, "elementwise_mul", &a, &b, &out, 4).unwrap();
        assert_eq!(unsafe { out.as_slice::<f32>() }, &[20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn unary_neg() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let input = Buffer::from_bytes(&device, f32_as_bytes(&[1.0, -2.0, 3.0, -4.0])).unwrap();
        let out = Buffer::new(&device, 16).unwrap();
        registry.dispatch_unary(&device, "elementwise_neg", &input, &out, 4).unwrap();
        assert_eq!(unsafe { out.as_slice::<f32>() }, &[-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn unary_relu() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let input = Buffer::from_bytes(&device, f32_as_bytes(&[-1.0, 0.0, 3.0, -4.0])).unwrap();
        let out = Buffer::new(&device, 16).unwrap();
        registry.dispatch_unary(&device, "elementwise_relu", &input, &out, 4).unwrap();
        assert_eq!(unsafe { out.as_slice::<f32>() }, &[0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn matmul_2x2() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, f32_as_bytes(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        let b = Buffer::from_bytes(&device, f32_as_bytes(&[5.0, 6.0, 7.0, 8.0])).unwrap();
        let c = Buffer::new(&device, 16).unwrap();
        registry.dispatch_matmul(&device, &a, &b, &c, 2, 2, 2).unwrap();
        assert_eq!(unsafe { c.as_slice::<f32>() }, &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_non_square() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, f32_as_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        let b = Buffer::from_bytes(&device, f32_as_bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])).unwrap();
        let c = Buffer::new(&device, 4 * 4).unwrap();
        registry.dispatch_matmul(&device, &a, &b, &c, 2, 2, 3).unwrap();
        assert_eq!(unsafe { c.as_slice::<f32>() }, &[58.0, 64.0, 139.0, 154.0]);
    }

    // ── F16 dispatch tests ──────────────────────────────────────────────────

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        use half::f16;
        let bits: Vec<u16> = values.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        unsafe { std::slice::from_raw_parts(bits.as_ptr() as *const u8, bits.len() * 2) }.to_vec()
    }

    fn read_f16(buf: &Buffer, count: usize) -> Vec<f32> {
        use half::f16;
        let slice = unsafe { std::slice::from_raw_parts(buf.contents() as *const u16, count) };
        slice.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    }

    #[test]
    fn dispatch_binary_f16_add() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, &f16_bytes(&[1.0, 2.0])).unwrap();
        let b = Buffer::from_bytes(&device, &f16_bytes(&[3.0, 4.0])).unwrap();
        let out = Buffer::new(&device, 4).unwrap();
        registry.dispatch_binary_typed(&device, "elementwise_add", DType::Float16, &a, &b, &out, 2).unwrap();
        let result = read_f16(&out, 2);
        assert_eq!(result[0], 4.0);
        assert_eq!(result[1], 6.0);
    }

    #[test]
    fn dispatch_binary_f16_mul() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, &f16_bytes(&[2.0, 3.0])).unwrap();
        let b = Buffer::from_bytes(&device, &f16_bytes(&[4.0, 5.0])).unwrap();
        let out = Buffer::new(&device, 4).unwrap();
        registry.dispatch_binary_typed(&device, "elementwise_mul", DType::Float16, &a, &b, &out, 2).unwrap();
        let result = read_f16(&out, 2);
        assert_eq!(result[0], 8.0);
        assert_eq!(result[1], 15.0);
    }

    #[test]
    fn dispatch_unary_f16_relu() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let input = Buffer::from_bytes(&device, &f16_bytes(&[-1.0, 0.0, 3.0, -4.0])).unwrap();
        let out = Buffer::new(&device, 8).unwrap();
        registry.dispatch_unary_typed(&device, "elementwise_relu", DType::Float16, &input, &out, 4).unwrap();
        let result = read_f16(&out, 4);
        assert_eq!(result, &[0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn dispatch_unary_f16_neg() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let input = Buffer::from_bytes(&device, &f16_bytes(&[1.0, -2.0, 3.0])).unwrap();
        let out = Buffer::new(&device, 6).unwrap();
        registry.dispatch_unary_typed(&device, "elementwise_neg", DType::Float16, &input, &out, 3).unwrap();
        let result = read_f16(&out, 3);
        assert_eq!(result, &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn dispatch_matmul_f16() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let a = Buffer::from_bytes(&device, &f16_bytes(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        let b = Buffer::from_bytes(&device, &f16_bytes(&[5.0, 6.0, 7.0, 8.0])).unwrap();
        let c = Buffer::new(&device, 8).unwrap();
        registry.dispatch_matmul_typed(&device, DType::Float16, &a, &b, &c, 2, 2, 2).unwrap();
        let result = read_f16(&c, 4);
        assert!((result[0] - 19.0).abs() < 0.5);
        assert!((result[1] - 22.0).abs() < 0.5);
        assert!((result[2] - 43.0).abs() < 0.5);
        assert!((result[3] - 50.0).abs() < 0.5);
    }

    #[test]
    fn dispatch_softmax_f16() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let input = Buffer::from_bytes(&device, &f16_bytes(&[1.0, 2.0, 3.0])).unwrap();
        let out = Buffer::new(&device, 6).unwrap();
        registry.dispatch_softmax_typed(&device, DType::Float16, &input, &out, 1, 3).unwrap();
        let result = read_f16(&out, 3);
        assert!((result[0] - 0.0900).abs() < 0.01);
        assert!((result[1] - 0.2447).abs() < 0.01);
        assert!((result[2] - 0.6652).abs() < 0.01);
    }

    #[test]
    fn dispatch_transpose_f16() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let input = Buffer::from_bytes(&device, &f16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        let out = Buffer::new(&device, 12).unwrap();
        registry.dispatch_transpose_typed(&device, DType::Float16, &input, &out, 2, 3).unwrap();
        let result = read_f16(&out, 6);
        assert_eq!(result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn dispatch_scalar_mul_f16() {
        let device = match get_device() { Some(d) => d, None => return };
        let registry = KernelRegistry::new();
        let input = Buffer::from_bytes(&device, &f16_bytes(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        let out = Buffer::new(&device, 8).unwrap();
        registry.dispatch_scalar_mul_typed(&device, DType::Float16, &input, &out, 2.0, 4).unwrap();
        let result = read_f16(&out, 4);
        assert_eq!(result, &[2.0, 4.0, 6.0, 8.0]);
    }
}
