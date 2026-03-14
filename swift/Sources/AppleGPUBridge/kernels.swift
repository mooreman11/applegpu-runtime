import Foundation

/// All Metal Shading Language kernel sources.
enum MetalKernels {

    static let elementwiseBinary = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void elementwise_add(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = a[id] + b[id]; }
    }

    kernel void elementwise_sub(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = a[id] - b[id]; }
    }

    kernel void elementwise_mul(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = a[id] * b[id]; }
    }

    kernel void elementwise_div(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = a[id] / b[id]; }
    }
    """

    static let elementwiseUnary = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void elementwise_neg(
        device const float* input [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = -input[id]; }
    }

    kernel void elementwise_relu(
        device const float* input [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = max(input[id], 0.0f); }
    }

    kernel void elementwise_exp(
        device const float* input [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = exp(input[id]); }
    }

    kernel void elementwise_log(
        device const float* input [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = log(input[id]); }
    }

    kernel void elementwise_sqrt(
        device const float* input [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { out[id] = sqrt(input[id]); }
    }
    """

    static let matmul = """
    #include <metal_stdlib>
    using namespace metal;

    // Simple matmul: C[M,N] = A[M,K] * B[K,N]
    // Each thread computes one element of C.
    kernel void matmul_f32(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint& M [[buffer(3)]],
        constant uint& N [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint row = gid.y;
        uint col = gid.x;
        if (row >= M || col >= N) return;

        float sum = 0.0f;
        for (uint i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
    """

    static let softmax = """
    #include <metal_stdlib>
    using namespace metal;

    // Numerically stable softmax along last dimension.
    // Input/output are 2D: [rows, cols]. Each thread processes one row.
    kernel void softmax_f32(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& rows [[buffer(2)]],
        constant uint& cols [[buffer(3)]],
        uint row [[thread_position_in_grid]]
    ) {
        if (row >= rows) return;

        uint offset = row * cols;

        // Find max for numerical stability
        float max_val = input[offset];
        for (uint j = 1; j < cols; j++) {
            max_val = max(max_val, input[offset + j]);
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (uint j = 0; j < cols; j++) {
            float e = exp(input[offset + j] - max_val);
            output[offset + j] = e;
            sum += e;
        }

        // Normalize
        for (uint j = 0; j < cols; j++) {
            output[offset + j] /= sum;
        }
    }
    """

    static let transpose = """
    #include <metal_stdlib>
    using namespace metal;

    // 2D transpose: output[col, row] = input[row, col]
    kernel void transpose_f32(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& rows [[buffer(2)]],
        constant uint& cols [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint row = gid.y;
        uint col = gid.x;
        if (row >= rows || col >= cols) return;
        output[col * rows + row] = input[row * cols + col];
    }
    """

    static let scalarMul = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void scalar_mul_f32(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant float& scale [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < count) { output[id] = input[id] * scale; }
    }
    """
}
