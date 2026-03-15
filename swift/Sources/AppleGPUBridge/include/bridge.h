#ifndef APPLE_GPU_BRIDGE_H
#define APPLE_GPU_BRIDGE_H

#include <stdint.h>

// Opaque handle to a GPU device
typedef struct GPUDeviceHandle GPUDeviceHandle;

// Lifecycle
GPUDeviceHandle* gpu_bridge_create_device(void);
void gpu_bridge_destroy_device(GPUDeviceHandle* device);

// Query
const char* gpu_bridge_device_name(const GPUDeviceHandle* device);

// Opaque handle to a GPU buffer
typedef struct GPUBufferHandle GPUBufferHandle;

// Buffer lifecycle
GPUBufferHandle* gpu_bridge_create_buffer(const GPUDeviceHandle* device, uint64_t size_bytes);
GPUBufferHandle* gpu_bridge_create_buffer_with_data(const GPUDeviceHandle* device, const void* data, uint64_t size_bytes);
void gpu_bridge_destroy_buffer(GPUBufferHandle* buffer);

// Buffer data access
void* gpu_bridge_buffer_contents(const GPUBufferHandle* buffer);
uint64_t gpu_bridge_buffer_length(const GPUBufferHandle* buffer);

// Opaque handle to a compute context (command queue + pipeline)
typedef struct GPUComputeHandle GPUComputeHandle;

// Compute lifecycle
GPUComputeHandle* gpu_bridge_create_compute(const GPUDeviceHandle* device, const char* kernel_source, const char* function_name);
void gpu_bridge_destroy_compute(GPUComputeHandle* compute);

// Execute element-wise operation: out = op(a, b)
// Returns 0 on success, -1 on failure.
int32_t gpu_bridge_compute_elementwise(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_out,
    uint64_t element_count
);

// Execute unary operation: out = op(input)
int32_t gpu_bridge_compute_unary(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    uint64_t element_count
);

// Execute matrix multiply: C[M,N] = A[M,K] * B[K,N]
int32_t gpu_bridge_compute_matmul(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_c,
    uint32_t M,
    uint32_t N,
    uint32_t K
);

// Batched matrix multiply: C[batch,M,N] = A[batch,M,K] * B[batch,K,N]
int32_t gpu_bridge_compute_matmul_batched(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_c,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t batch_size,
    uint32_t a_batch_stride,
    uint32_t b_batch_stride
);

// Batched softmax with causal mask: 2D dispatch (row, batch)
int32_t gpu_bridge_compute_softmax_causal(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t batch_size,
    uint32_t rows,
    uint32_t cols
);

// Execute a fused kernel with variable number of input buffers.
int32_t gpu_bridge_compute_fused(
    GPUComputeHandle* compute,
    const GPUBufferHandle* const* input_buffers,
    uint32_t buffer_count,
    GPUBufferHandle* output,
    uint64_t element_count
);

// Softmax along last dimension of 2D tensor [rows, cols]
int32_t gpu_bridge_compute_softmax(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols
);

// 2D transpose: output[cols, rows] = input[rows, cols]
int32_t gpu_bridge_compute_transpose(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols
);

// Batched transpose: output[batch, cols, rows] = input[batch, rows, cols]
int32_t gpu_bridge_compute_transpose_batched(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t batch_size,
    uint32_t rows,
    uint32_t cols
);

// Scalar multiply: output[i] = input[i] * scale
int32_t gpu_bridge_compute_scalar_mul(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    float scale,
    uint64_t element_count
);

// Layer normalization: output = gamma * (input - mean) / sqrt(var + eps) + beta
int32_t gpu_bridge_compute_layer_norm(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    const GPUBufferHandle* buf_gamma,
    const GPUBufferHandle* buf_beta,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols,
    float eps
);

// Softmax backward: grad_input = output * (grad_output - sum(grad_output * output))
int32_t gpu_bridge_compute_softmax_backward(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_grad_output,
    const GPUBufferHandle* buf_output,
    GPUBufferHandle* buf_grad_input,
    uint32_t rows,
    uint32_t cols
);

// Layer norm backward: computes grad_input from grad_output, input, gamma
int32_t gpu_bridge_compute_layer_norm_backward(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_grad_output,
    const GPUBufferHandle* buf_input,
    const GPUBufferHandle* buf_gamma,
    GPUBufferHandle* buf_grad_input,
    uint32_t rows,
    uint32_t cols,
    float eps
);

// Embedding lookup: output[i,j] = weights[indices[i],j]
int32_t gpu_bridge_compute_embedding(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_weights,
    const GPUBufferHandle* buf_indices,
    GPUBufferHandle* buf_output,
    uint32_t seq_len,
    uint32_t embed_dim
);

// ── Non-blocking (batched) dispatch ──────────────────────────────────────

// Get or create a device-level shared command queue for batched dispatch.
void* gpu_bridge_get_shared_queue(GPUDeviceHandle* device);

// Wait for a command buffer to complete (consumes the retained reference).
void gpu_bridge_wait_command_buffer(void* command_buffer);

// Batch encoding: encode all ops into a single command buffer per eval.
// begin_batch creates a command buffer; all subsequent _nb calls encode into it.
// end_batch commits and returns the CB handle for waiting.
// abort_batch discards the batch on mid-encode error.
void* gpu_bridge_begin_batch(void* queue);
void* gpu_bridge_end_batch(void);
void gpu_bridge_abort_batch(void);

// Non-blocking elementwise: returns command buffer handle (or NULL on failure).
void* gpu_bridge_compute_elementwise_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_out,
    uint64_t element_count
);

// Non-blocking unary: returns command buffer handle.
void* gpu_bridge_compute_unary_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    uint64_t element_count
);

// Non-blocking matmul: returns command buffer handle.
void* gpu_bridge_compute_matmul_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_c,
    uint32_t M,
    uint32_t N,
    uint32_t K
);

// Non-blocking batched matmul: returns command buffer handle.
void* gpu_bridge_compute_matmul_batched_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_c,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t batch_size,
    uint32_t a_batch_stride,
    uint32_t b_batch_stride
);

// Non-blocking batched softmax causal: returns command buffer handle.
void* gpu_bridge_compute_softmax_causal_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t batch_size,
    uint32_t rows,
    uint32_t cols
);

// Non-blocking softmax: returns command buffer handle.
void* gpu_bridge_compute_softmax_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols
);

// Non-blocking transpose: returns command buffer handle.
void* gpu_bridge_compute_transpose_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols
);

// Non-blocking batched transpose: returns command buffer handle.
void* gpu_bridge_compute_transpose_batched_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t batch_size,
    uint32_t rows,
    uint32_t cols
);

// Non-blocking scalar multiply: returns command buffer handle.
void* gpu_bridge_compute_scalar_mul_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    float scale,
    uint64_t element_count
);

// Non-blocking fused kernel: returns command buffer handle.
void* gpu_bridge_compute_fused_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* const* input_buffers,
    uint32_t buffer_count,
    GPUBufferHandle* output,
    uint64_t element_count
);

// Non-blocking layer norm: returns command buffer handle.
void* gpu_bridge_compute_layer_norm_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    const GPUBufferHandle* buf_gamma,
    const GPUBufferHandle* buf_beta,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols,
    float eps
);

// Non-blocking softmax backward: returns command buffer handle.
void* gpu_bridge_compute_softmax_backward_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_grad_output,
    const GPUBufferHandle* buf_output,
    GPUBufferHandle* buf_grad_input,
    uint32_t rows,
    uint32_t cols
);

// Non-blocking layer norm backward: returns command buffer handle.
void* gpu_bridge_compute_layer_norm_backward_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_grad_output,
    const GPUBufferHandle* buf_input,
    const GPUBufferHandle* buf_gamma,
    GPUBufferHandle* buf_grad_input,
    uint32_t rows,
    uint32_t cols,
    float eps
);

// Non-blocking embedding: returns command buffer handle.
void* gpu_bridge_compute_embedding_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_weights,
    const GPUBufferHandle* buf_indices,
    GPUBufferHandle* buf_output,
    uint32_t seq_len,
    uint32_t embed_dim
);

// Non-blocking blit copy: returns command buffer handle.
void* gpu_bridge_blit_copy_nb(
    GPUDeviceHandle* device,
    void* queue,
    GPUBufferHandle* src_buf,
    GPUBufferHandle* dst_buf,
    uint64_t size_bytes
);

// ── Slice dispatch ──────────────────────────────────────────────────────────

int32_t gpu_bridge_compute_slice_dim0(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t cols,
    uint32_t start_row,
    uint32_t out_rows
);

void* gpu_bridge_compute_slice_dim0_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t cols,
    uint32_t start_row,
    uint32_t out_rows
);

int32_t gpu_bridge_compute_slice_dim1(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t in_cols,
    uint32_t out_cols,
    uint32_t start_col,
    uint32_t rows
);

void* gpu_bridge_compute_slice_dim1_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_output,
    uint32_t in_cols,
    uint32_t out_cols,
    uint32_t start_col,
    uint32_t rows
);

// ── Concat dispatch ─────────────────────────────────────────────────────────

int32_t gpu_bridge_compute_concat_dim0(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_output,
    uint32_t rows_a,
    uint32_t cols,
    uint32_t total_rows
);

void* gpu_bridge_compute_concat_dim0_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_output,
    uint32_t rows_a,
    uint32_t cols,
    uint32_t total_rows
);

int32_t gpu_bridge_compute_concat_dim1(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols_a,
    uint32_t cols_b
);

void* gpu_bridge_compute_concat_dim1_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols_a,
    uint32_t cols_b
);

// ── AddBias dispatch ────────────────────────────────────────────────────────

int32_t gpu_bridge_compute_add_bias(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    const GPUBufferHandle* buf_bias,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols
);

void* gpu_bridge_compute_add_bias_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    const GPUBufferHandle* buf_bias,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t cols
);

// ── N-D fused element-wise dispatch ──────────────────────────────────────

// Execute a fused N-D kernel with variable input buffers and stride arrays.
// input_strides is an array of (buffer_count) pointers, each pointing to a uint32_t[8] stride array.
int32_t gpu_bridge_compute_fused_nd(
    GPUComputeHandle* compute,
    const GPUBufferHandle* const* input_buffers,
    uint32_t buffer_count,
    GPUBufferHandle* output,
    const uint32_t* const* input_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel
);

// Non-blocking N-D fused kernel: returns command buffer handle.
void* gpu_bridge_compute_fused_nd_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* const* input_buffers,
    uint32_t buffer_count,
    GPUBufferHandle* output,
    const uint32_t* const* input_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel
);

// ── N-D stride-based element-wise dispatch ───────────────────────────────

int32_t gpu_bridge_compute_binary_nd(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_out,
    const uint32_t* a_strides,
    const uint32_t* b_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel
);

void* gpu_bridge_compute_binary_nd_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_a,
    const GPUBufferHandle* buf_b,
    GPUBufferHandle* buf_out,
    const uint32_t* a_strides,
    const uint32_t* b_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel
);

int32_t gpu_bridge_compute_unary_nd(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    const uint32_t* in_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel
);

void* gpu_bridge_compute_unary_nd_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    const uint32_t* in_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel
);

int32_t gpu_bridge_compute_pow_nd(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    const uint32_t* in_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel,
    float exponent
);

void* gpu_bridge_compute_pow_nd_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    const uint32_t* in_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel,
    float exponent
);

int32_t gpu_bridge_compute_clamp_nd(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    const uint32_t* in_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel,
    float min_val,
    float max_val
);

void* gpu_bridge_compute_clamp_nd_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    const uint32_t* in_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel,
    float min_val,
    float max_val
);

// Where (ternary) N-D dispatch
int32_t gpu_bridge_compute_where_nd(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_cond,
    const GPUBufferHandle* buf_x,
    const GPUBufferHandle* buf_y,
    GPUBufferHandle* buf_out,
    const uint32_t* cond_strides,
    const uint32_t* x_strides,
    const uint32_t* y_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel
);

void* gpu_bridge_compute_where_nd_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_cond,
    const GPUBufferHandle* buf_x,
    const GPUBufferHandle* buf_y,
    GPUBufferHandle* buf_out,
    const uint32_t* cond_strides,
    const uint32_t* x_strides,
    const uint32_t* y_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel
);

// MaskedFill N-D dispatch
int32_t gpu_bridge_compute_masked_fill_nd(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    const GPUBufferHandle* buf_mask,
    GPUBufferHandle* buf_out,
    const uint32_t* in_strides,
    const uint32_t* mask_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel,
    float fill_value
);

void* gpu_bridge_compute_masked_fill_nd_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    const GPUBufferHandle* buf_mask,
    GPUBufferHandle* buf_out,
    const uint32_t* in_strides,
    const uint32_t* mask_strides,
    const uint32_t* out_shape,
    uint32_t ndim,
    uint32_t numel,
    float fill_value
);

// Gather dispatch: 3 buffers (input, indices, output) + 3 uint params
int32_t gpu_bridge_compute_gather(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    const GPUBufferHandle* buf_indices,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t in_cols,
    uint32_t out_cols
);

void* gpu_bridge_compute_gather_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    const GPUBufferHandle* buf_indices,
    GPUBufferHandle* buf_output,
    uint32_t rows,
    uint32_t in_cols,
    uint32_t out_cols
);

// Triangular (triu/tril) dispatch
int32_t gpu_bridge_compute_triangular(
    GPUComputeHandle* compute,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    uint32_t batch_size,
    uint32_t rows,
    uint32_t cols,
    int32_t diagonal
);

void* gpu_bridge_compute_triangular_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* buf_input,
    GPUBufferHandle* buf_out,
    uint32_t batch_size,
    uint32_t rows,
    uint32_t cols,
    int32_t diagonal
);

// Generic 3D dispatch for CNN ops
int32_t gpu_bridge_compute_3d(
    GPUComputeHandle* compute,
    const GPUBufferHandle* const* input_buffers,
    uint32_t buffer_count,
    GPUBufferHandle* output,
    const uint32_t* uint_params,
    uint32_t uint_param_count,
    const float* float_params,
    uint32_t float_param_count,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z
);

void* gpu_bridge_compute_3d_nb(
    GPUComputeHandle* compute,
    void* queue,
    const GPUBufferHandle* const* input_buffers,
    uint32_t buffer_count,
    GPUBufferHandle* output,
    const uint32_t* uint_params,
    uint32_t uint_param_count,
    const float* float_params,
    uint32_t float_param_count,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z
);

#endif
