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

#endif
