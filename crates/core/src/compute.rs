use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::ffi;
use crate::tensor::DType;

/// MSL source for binary element-wise ops (add, sub, mul, div).
const BINARY_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] + b[id]; } }
kernel void elementwise_sub(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] - b[id]; } }
kernel void elementwise_mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] * b[id]; } }
kernel void elementwise_div(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] / b[id]; } }
"#;

/// MSL source for unary element-wise ops (neg, relu, exp, log, sqrt).
const UNARY_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_neg(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = -input[id]; } }
kernel void elementwise_relu(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = max(input[id], 0.0f); } }
kernel void elementwise_exp(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = exp(input[id]); } }
kernel void elementwise_log(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = log(input[id]); } }
kernel void elementwise_sqrt(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = sqrt(input[id]); } }
"#;

/// MSL source for matrix multiplication.
const MATMUL_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

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
"#;

const SOFTMAX_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = input[offset];
    for (uint j = 1; j < cols; j++) { max_val = max(max_val, input[offset + j]); }
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) { float e = exp(input[offset + j] - max_val); output[offset + j] = e; sum += e; }
    for (uint j = 0; j < cols; j++) { output[offset + j] /= sum; }
}
"#;

const TRANSPOSE_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

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
"#;

const SCALAR_MUL_KERNEL_SOURCE: &str = r#"
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
"#;

const GELU_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        float x = input[id];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        inner = clamp(inner, -10.0f, 10.0f);
        output[id] = x * 0.5f * (1.0f + tanh(inner));
    }
}
"#;

const LAYER_NORM_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    uint offset = row * cols;
    float mean = 0.0f;
    for (uint j = 0; j < cols; j++) mean += input[offset + j];
    mean /= float(cols);
    float var = 0.0f;
    for (uint j = 0; j < cols; j++) {
        float diff = input[offset + j] - mean;
        var += diff * diff;
    }
    var /= float(cols);
    float inv_std = 1.0f / sqrt(var + eps);
    for (uint j = 0; j < cols; j++) {
        output[offset + j] = gamma[j] * (input[offset + j] - mean) * inv_std + beta[j];
    }
}
"#;

const EMBEDDING_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void embedding_f32(
    device const float* weights [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= seq_len || j >= embed_dim) return;
    int idx = indices[i];
    output[i * embed_dim + j] = weights[idx * embed_dim + j];
}
"#;

// ── Float16 kernel sources ──────────────────────────────────────────────────

/// MSL source for f16 binary element-wise ops (add, sub, mul, div).
const BINARY_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] + b[id]; } }
kernel void elementwise_sub_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] - b[id]; } }
kernel void elementwise_mul_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] * b[id]; } }
kernel void elementwise_div_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = a[id] / b[id]; } }
"#;

/// MSL source for f16 unary element-wise ops (neg, relu, exp, log, sqrt).
const UNARY_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_neg_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = -input[id]; } }
kernel void elementwise_relu_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = max(input[id], (half)0); } }
kernel void elementwise_exp_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = exp(input[id]); } }
kernel void elementwise_log_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = log(input[id]); } }
kernel void elementwise_sqrt_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) { out[id] = sqrt(input[id]); } }
"#;

/// MSL source for f16 matmul with f32-intermediate accumulation.
const MATMUL_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
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
        sum += float(A[row * K + i]) * float(B[i * N + col]);
    }
    C[row * N + col] = half(sum);
}
"#;

/// MSL source for f16 softmax with f32-intermediate computation.
const SOFTMAX_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = float(input[offset]);
    for (uint j = 1; j < cols; j++) { max_val = max(max_val, float(input[offset + j])); }
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) { float e = exp(float(input[offset + j]) - max_val); output[offset + j] = half(e); sum += e; }
    for (uint j = 0; j < cols; j++) { output[offset + j] = half(float(output[offset + j]) / sum); }
}
"#;

/// MSL source for f16 transpose.
const TRANSPOSE_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void transpose_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= cols) return;
    output[col * rows + row] = input[row * cols + col];
}
"#;

const GELU_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gelu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        float x = float(input[id]);
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        inner = clamp(inner, -10.0f, 10.0f);
        output[id] = half(x * 0.5f * (1.0f + tanh(inner)));
    }
}
"#;

const LAYER_NORM_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* gamma [[buffer(1)]],
    device const half* beta [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    uint offset = row * cols;
    float mean = 0.0f;
    for (uint j = 0; j < cols; j++) mean += float(input[offset + j]);
    mean /= float(cols);
    float var = 0.0f;
    for (uint j = 0; j < cols; j++) {
        float diff = float(input[offset + j]) - mean;
        var += diff * diff;
    }
    var /= float(cols);
    float inv_std = 1.0f / sqrt(var + eps);
    for (uint j = 0; j < cols; j++) {
        output[offset + j] = half(float(gamma[j]) * (float(input[offset + j]) - mean) * inv_std + float(beta[j]));
    }
}
"#;

const EMBEDDING_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void embedding_f16(
    device const half* weights [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= seq_len || j >= embed_dim) return;
    int idx = indices[i];
    output[i * embed_dim + j] = weights[idx * embed_dim + j];
}
"#;

// ── Slice kernel sources ────────────────────────────────────────────────────

const SLICE_DIM0_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void slice_dim0_f32(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant uint& cols [[buffer(2)]], constant uint& start_row [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; uint col = gid.x;
    if (col >= cols) return;
    output[row * cols + col] = input[(start_row + row) * cols + col];
}
"#;

const SLICE_DIM0_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void slice_dim0_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], constant uint& cols [[buffer(2)]], constant uint& start_row [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; uint col = gid.x;
    if (col >= cols) return;
    output[row * cols + col] = input[(start_row + row) * cols + col];
}
"#;

const SLICE_DIM1_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void slice_dim1_f32(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant uint& in_cols [[buffer(2)]], constant uint& out_cols [[buffer(3)]], constant uint& start_col [[buffer(4)]], constant uint& rows [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; uint col = gid.x;
    if (row >= rows || col >= out_cols) return;
    output[row * out_cols + col] = input[row * in_cols + (start_col + col)];
}
"#;

const SLICE_DIM1_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void slice_dim1_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], constant uint& in_cols [[buffer(2)]], constant uint& out_cols [[buffer(3)]], constant uint& start_col [[buffer(4)]], constant uint& rows [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; uint col = gid.x;
    if (row >= rows || col >= out_cols) return;
    output[row * out_cols + col] = input[row * in_cols + (start_col + col)];
}
"#;

// ── Concat kernel sources ───────────────────────────────────────────────────

const CONCAT_DIM0_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void concat_dim0_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* output [[buffer(2)]], constant uint& rows_a [[buffer(3)]], constant uint& cols [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; uint col = gid.x;
    if (col >= cols) return;
    if (row < rows_a) output[row * cols + col] = a[row * cols + col];
    else output[row * cols + col] = b[(row - rows_a) * cols + col];
}
"#;

const CONCAT_DIM0_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void concat_dim0_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* output [[buffer(2)]], constant uint& rows_a [[buffer(3)]], constant uint& cols [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; uint col = gid.x;
    if (col >= cols) return;
    if (row < rows_a) output[row * cols + col] = a[row * cols + col];
    else output[row * cols + col] = b[(row - rows_a) * cols + col];
}
"#;

const CONCAT_DIM1_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void concat_dim1_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* output [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols_a [[buffer(4)]], constant uint& cols_b [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; uint col = gid.x;
    uint total_cols = cols_a + cols_b;
    if (row >= rows || col >= total_cols) return;
    if (col < cols_a) output[row * total_cols + col] = a[row * cols_a + col];
    else output[row * total_cols + col] = b[row * cols_b + (col - cols_a)];
}
"#;

const CONCAT_DIM1_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void concat_dim1_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* output [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols_a [[buffer(4)]], constant uint& cols_b [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; uint col = gid.x;
    uint total_cols = cols_a + cols_b;
    if (row >= rows || col >= total_cols) return;
    if (col < cols_a) output[row * total_cols + col] = a[row * cols_a + col];
    else output[row * total_cols + col] = b[row * cols_b + (col - cols_a)];
}
"#;

// ── AddBias kernel sources ──────────────────────────────────────────────────

const ADD_BIAS_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_bias_f32(device const float* input [[buffer(0)]], device const float* bias [[buffer(1)]], device float* output [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; uint col = gid.x;
    if (row >= rows || col >= cols) return;
    output[row * cols + col] = input[row * cols + col] + bias[col];
}
"#;

const ADD_BIAS_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_bias_f16(device const half* input [[buffer(0)]], device const half* bias [[buffer(1)]], device half* output [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; uint col = gid.x;
    if (row >= rows || col >= cols) return;
    output[row * cols + col] = input[row * cols + col] + bias[col];
}
"#;

// ── SoftmaxCausal kernel sources ────────────────────────────────────────────

const SOFTMAX_CAUSAL_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax_causal_f32(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = -1e9f;
    for (uint j = 0; j <= row && j < cols; j++) max_val = max(max_val, input[offset + j]);
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        if (j <= row) { float e = exp(input[offset + j] - max_val); output[offset + j] = e; sum += e; }
        else { output[offset + j] = 0.0f; }
    }
    for (uint j = 0; j <= row && j < cols; j++) output[offset + j] /= sum;
}
"#;

const SOFTMAX_CAUSAL_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax_causal_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = -1e9f;
    for (uint j = 0; j <= row && j < cols; j++) max_val = max(max_val, float(input[offset + j]));
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        if (j <= row) { float e = exp(float(input[offset + j]) - max_val); output[offset + j] = half(e); sum += e; }
        else { output[offset + j] = half(0.0f); }
    }
    for (uint j = 0; j <= row && j < cols; j++) output[offset + j] = half(float(output[offset + j]) / sum);
}
"#;

// ── Argmax kernel sources ───────────────────────────────────────────────────

const ARGMAX_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void argmax_f32(device const float* input [[buffer(0)]], device int* output [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = input[offset]; int max_idx = 0;
    for (uint j = 1; j < cols; j++) { if (input[offset + j] > max_val) { max_val = input[offset + j]; max_idx = int(j); } }
    output[row] = max_idx;
}
"#;

const ARGMAX_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void argmax_f16(device const half* input [[buffer(0)]], device int* output [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = float(input[offset]); int max_idx = 0;
    for (uint j = 1; j < cols; j++) { if (float(input[offset + j]) > max_val) { max_val = float(input[offset + j]); max_idx = int(j); } }
    output[row] = max_idx;
}
"#;

/// MSL source for f16 scalar multiply (scalar stays float, read/write half).
const SCALAR_MUL_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void scalar_mul_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) { output[id] = half(float(input[id]) * scale); }
}
"#;

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

    /// Dispatch binary element-wise operation: out = op(a, b).
    pub fn dispatch_elementwise(
        &self,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_elementwise(
                self.handle,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_out.raw_handle(),
                element_count as u64,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Binary kernel dispatch failed".to_string())) }
    }

    /// Dispatch unary element-wise operation: out = op(input).
    pub fn dispatch_unary(
        &self,
        buf_input: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_unary(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                element_count as u64,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Unary kernel dispatch failed".to_string())) }
    }

    /// Dispatch matrix multiplication: C[M,N] = A[M,K] * B[K,N].
    pub fn dispatch_matmul(
        &self,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_matmul(
                self.handle,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_c.raw_handle(),
                m as u32,
                n as u32,
                k as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Matmul dispatch failed".to_string())) }
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

    /// Non-blocking binary elementwise. Returns command buffer handle.
    pub fn dispatch_elementwise_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_elementwise_nb(
                self.handle,
                queue,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_out.raw_handle(),
                element_count as u64,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking binary dispatch failed".to_string())) } else { Ok(cb) }
    }

    /// Non-blocking unary. Returns command buffer handle.
    pub fn dispatch_unary_nb(
        &self,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cb = unsafe {
            ffi::gpu_bridge_compute_unary_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_out.raw_handle(),
                element_count as u64,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking unary dispatch failed".to_string())) } else { Ok(cb) }
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
        let cb = unsafe {
            ffi::gpu_bridge_compute_matmul_nb(
                self.handle,
                queue,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_c.raw_handle(),
                m as u32,
                n as u32,
                k as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking matmul dispatch failed".to_string())) } else { Ok(cb) }
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
    fn get_or_create(
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
    fn resolve_kernel(base_name: &str, dtype: DType) -> (&'static str, String) {
        match dtype {
            DType::Float16 => {
                let f16_name = format!("{}_f16", base_name);
                let source = match base_name {
                    n if n.starts_with("elementwise_add") || n.starts_with("elementwise_sub")
                        || n.starts_with("elementwise_mul") || n.starts_with("elementwise_div") =>
                        BINARY_KERNEL_SOURCE_F16,
                    n if n.starts_with("elementwise_") => UNARY_KERNEL_SOURCE_F16,
                    "matmul_f32" => MATMUL_KERNEL_SOURCE_F16,
                    "softmax_f32" => SOFTMAX_KERNEL_SOURCE_F16,
                    "transpose_f32" => TRANSPOSE_KERNEL_SOURCE_F16,
                    "scalar_mul_f32" => SCALAR_MUL_KERNEL_SOURCE_F16,
                    "gelu_f32" => GELU_KERNEL_SOURCE_F16,
                    "layer_norm_f32" => LAYER_NORM_KERNEL_SOURCE_F16,
                    "embedding_f32" => EMBEDDING_KERNEL_SOURCE_F16,
                    "slice_dim0_f32" => SLICE_DIM0_KERNEL_SOURCE_F16,
                    "slice_dim1_f32" => SLICE_DIM1_KERNEL_SOURCE_F16,
                    "concat_dim0_f32" => CONCAT_DIM0_KERNEL_SOURCE_F16,
                    "concat_dim1_f32" => CONCAT_DIM1_KERNEL_SOURCE_F16,
                    "add_bias_f32" => ADD_BIAS_KERNEL_SOURCE_F16,
                    "softmax_causal_f32" => SOFTMAX_CAUSAL_KERNEL_SOURCE_F16,
                    "argmax_f32" => ARGMAX_KERNEL_SOURCE_F16,
                    _ => BINARY_KERNEL_SOURCE_F16, // fallback
                };
                // For named kernels like matmul_f32 -> matmul_f16
                let func_name = if base_name.ends_with("_f32") {
                    format!("{}_f16", &base_name[..base_name.len() - 4])
                } else {
                    f16_name
                };
                (source, func_name)
            }
            _ => {
                // Float32 (default)
                let source = match base_name {
                    n if n.starts_with("elementwise_add") || n.starts_with("elementwise_sub")
                        || n.starts_with("elementwise_mul") || n.starts_with("elementwise_div") =>
                        BINARY_KERNEL_SOURCE,
                    n if n.starts_with("elementwise_") => UNARY_KERNEL_SOURCE,
                    "matmul_f32" => MATMUL_KERNEL_SOURCE,
                    "softmax_f32" => SOFTMAX_KERNEL_SOURCE,
                    "transpose_f32" => TRANSPOSE_KERNEL_SOURCE,
                    "scalar_mul_f32" => SCALAR_MUL_KERNEL_SOURCE,
                    "gelu_f32" => GELU_KERNEL_SOURCE,
                    "layer_norm_f32" => LAYER_NORM_KERNEL_SOURCE,
                    "embedding_f32" => EMBEDDING_KERNEL_SOURCE,
                    "slice_dim0_f32" => SLICE_DIM0_KERNEL_SOURCE,
                    "slice_dim1_f32" => SLICE_DIM1_KERNEL_SOURCE,
                    "concat_dim0_f32" => CONCAT_DIM0_KERNEL_SOURCE,
                    "concat_dim1_f32" => CONCAT_DIM1_KERNEL_SOURCE,
                    "add_bias_f32" => ADD_BIAS_KERNEL_SOURCE,
                    "softmax_causal_f32" => SOFTMAX_CAUSAL_KERNEL_SOURCE,
                    "argmax_f32" => ARGMAX_KERNEL_SOURCE,
                    _ => BINARY_KERNEL_SOURCE,
                };
                (source, base_name.to_string())
            }
        }
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
        let pipeline = self.get_or_create(device, BINARY_KERNEL_SOURCE, function_name)?;
        pipeline.dispatch_elementwise(buf_a, buf_b, buf_out, element_count)
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
        let pipeline = self.get_or_create(device, source, &func)?;
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
        let pipeline = self.get_or_create(device, UNARY_KERNEL_SOURCE, function_name)?;
        pipeline.dispatch_unary(buf_input, buf_out, element_count)
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
        let pipeline = self.get_or_create(device, source, &func)?;
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
        let pipeline = self.get_or_create(device, MATMUL_KERNEL_SOURCE, "matmul_f32")?;
        pipeline.dispatch_matmul(buf_a, buf_b, buf_c, m, n, k)
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
        let (source, func) = Self::resolve_kernel("matmul_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_matmul(buf_a, buf_b, buf_c, m, n, k)
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

    pub fn dispatch_softmax(
        &self, device: &Device, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, SOFTMAX_KERNEL_SOURCE, "softmax_f32")?;
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_softmax_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("softmax_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_transpose(
        &self, device: &Device, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, TRANSPOSE_KERNEL_SOURCE, "transpose_f32")?;
        pipeline.dispatch_transpose(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_transpose_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("transpose_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_transpose(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_scalar_mul(
        &self, device: &Device, buf_input: &Buffer, buf_output: &Buffer,
        scale: f32, element_count: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, SCALAR_MUL_KERNEL_SOURCE, "scalar_mul_f32")?;
        pipeline.dispatch_scalar_mul(buf_input, buf_output, scale, element_count)
    }

    pub fn dispatch_scalar_mul_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        scale: f32, element_count: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("scalar_mul_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_scalar_mul(buf_input, buf_output, scale, element_count)
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
        let (source, func) = Self::resolve_kernel("gelu_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
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
        let (source, func) = Self::resolve_kernel("layer_norm_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_layer_norm(buf_input, buf_gamma, buf_beta, buf_out, rows, cols, eps)
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
        let (source, func) = Self::resolve_kernel("embedding_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_embedding(buf_weights, buf_indices, buf_out, seq_len, embed_dim)
    }

    // ── New op dispatch methods (typed) ──────────────────────────────────

    pub fn dispatch_slice_dim0_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        cols: usize, start_row: usize, out_rows: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("slice_dim0_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_slice_dim0(buf_input, buf_output, cols, start_row, out_rows)
    }

    pub fn dispatch_slice_dim1_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        in_cols: usize, out_cols: usize, start_col: usize, rows: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("slice_dim1_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_slice_dim1(buf_input, buf_output, in_cols, out_cols, start_col, rows)
    }

    pub fn dispatch_concat_dim0_typed(
        &self, device: &Device, dtype: DType, buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer,
        rows_a: usize, cols: usize, total_rows: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("concat_dim0_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_concat_dim0(buf_a, buf_b, buf_output, rows_a, cols, total_rows)
    }

    pub fn dispatch_concat_dim1_typed(
        &self, device: &Device, dtype: DType, buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer,
        rows: usize, cols_a: usize, cols_b: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("concat_dim1_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_concat_dim1(buf_a, buf_b, buf_output, rows, cols_a, cols_b)
    }

    pub fn dispatch_add_bias_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_bias: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("add_bias_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_add_bias(buf_input, buf_bias, buf_output, rows, cols)
    }

    pub fn dispatch_softmax_causal_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("softmax_causal_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        // Reuse softmax dispatch (same buffer layout: input, output, rows, cols; 1D dispatch per row)
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_argmax_typed(
        &self, device: &Device, input_dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("argmax_f32", input_dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        // Reuse softmax dispatch (same buffer layout: input, output, rows, cols; 1D dispatch per row)
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    // ── Non-blocking new op dispatch methods (typed) ──────────────────────

    pub fn dispatch_slice_dim0_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, cols: usize, start_row: usize, out_rows: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("slice_dim0_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_slice_dim0_nb(queue, buf_input, buf_output, cols, start_row, out_rows)
    }

    pub fn dispatch_slice_dim1_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, in_cols: usize, out_cols: usize, start_col: usize, rows: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("slice_dim1_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_slice_dim1_nb(queue, buf_input, buf_output, in_cols, out_cols, start_col, rows)
    }

    pub fn dispatch_concat_dim0_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer, rows_a: usize, cols: usize, total_rows: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("concat_dim0_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_concat_dim0_nb(queue, buf_a, buf_b, buf_output, rows_a, cols, total_rows)
    }

    pub fn dispatch_concat_dim1_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_a: &Buffer, buf_b: &Buffer, buf_output: &Buffer, rows: usize, cols_a: usize, cols_b: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("concat_dim1_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_concat_dim1_nb(queue, buf_a, buf_b, buf_output, rows, cols_a, cols_b)
    }

    pub fn dispatch_add_bias_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_bias: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("add_bias_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_add_bias_nb(queue, buf_input, buf_bias, buf_output, rows, cols)
    }

    pub fn dispatch_softmax_causal_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("softmax_causal_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_argmax_typed_nb(
        &self, device: &Device, input_dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("argmax_f32", input_dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }

    // ── Non-blocking dispatch methods ─────────────────────────────────────

    pub fn dispatch_gelu_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_out: &Buffer, element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("gelu_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_unary_nb(queue, buf_input, buf_out, element_count)
    }

    pub fn dispatch_layer_norm_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_gamma: &Buffer, buf_beta: &Buffer, buf_out: &Buffer,
        rows: usize, cols: usize, eps: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("layer_norm_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_layer_norm_nb(queue, buf_input, buf_gamma, buf_beta, buf_out, rows, cols, eps)
    }

    pub fn dispatch_embedding_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_weights: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        seq_len: usize, embed_dim: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("embedding_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_embedding_nb(queue, buf_weights, buf_indices, buf_out, seq_len, embed_dim)
    }

    pub fn dispatch_binary_typed_nb(
        &self, device: &Device, function_name: &str, dtype: DType, queue: *mut std::ffi::c_void,
        buf_a: &Buffer, buf_b: &Buffer, buf_out: &Buffer, element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel(function_name, dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_elementwise_nb(queue, buf_a, buf_b, buf_out, element_count)
    }

    pub fn dispatch_unary_typed_nb(
        &self, device: &Device, function_name: &str, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_out: &Buffer, element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel(function_name, dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_unary_nb(queue, buf_input, buf_out, element_count)
    }

    pub fn dispatch_matmul_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_a: &Buffer, buf_b: &Buffer, buf_c: &Buffer, m: usize, n: usize, k: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("matmul_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_matmul_nb(queue, buf_a, buf_b, buf_c, m, n, k)
    }

    pub fn dispatch_softmax_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("softmax_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_transpose_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("transpose_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_transpose_nb(queue, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_scalar_mul_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, scale: f32, element_count: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("scalar_mul_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
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
}

/// Get or create the device-level shared command queue.
pub fn get_shared_queue(device: &Device) -> *mut std::ffi::c_void {
    unsafe { ffi::gpu_bridge_get_shared_queue(device.raw_handle() as *mut _) }
}

/// Wait for a command buffer to complete (consumes the retained reference).
pub fn wait_command_buffer(cb: *mut std::ffi::c_void) {
    unsafe { ffi::gpu_bridge_wait_command_buffer(cb) }
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
