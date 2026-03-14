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

#endif
