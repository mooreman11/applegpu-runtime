use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::ffi;
use crate::tensor::{DType, MAX_DIMS};

/// MSL helper for N-D stride-based indexing, prepended to all element-wise kernels.
const ND_INDEX_HELPER: &str = r#"
uint nd_index_to_offset(uint flat_id, constant uint* shape, constant uint* strides, constant uint& ndim) {
    uint offset = 0;
    for (uint d = ndim; d > 0; d--) {
        uint i = d - 1;
        offset += (flat_id % shape[i]) * strides[i];
        flat_id /= shape[i];
    }
    return offset;
}
"#;

/// MSL source for binary element-wise ops (add, sub, mul, div) with N-D stride-based indexing.
const BINARY_KERNEL_SOURCE: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void elementwise_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] + b[b_off];
}
kernel void elementwise_sub(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] - b[b_off];
}
kernel void elementwise_mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] * b[b_off];
}
kernel void elementwise_div(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] / b[b_off];
}
"#
);

/// MSL source for unary element-wise ops (neg, relu, exp, log, sqrt) with N-D stride-based indexing.
const UNARY_KERNEL_SOURCE: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void elementwise_neg(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = -input[in_off];
}
kernel void elementwise_relu(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = max(input[in_off], 0.0f);
}
kernel void elementwise_exp(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = exp(input[in_off]);
}
kernel void elementwise_log(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = log(input[in_off]);
}
kernel void elementwise_sqrt(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = sqrt(input[in_off]);
}
kernel void elementwise_abs(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = abs(input[in_off]);
}
kernel void elementwise_sign(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = sign(input[in_off]);
}
kernel void elementwise_tanh(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = tanh(input[in_off]);
}
"#
);

/// MSL source for batched matrix multiplication.
/// 3D dispatch: (col, row, batch). Supports batch broadcasting via strides.
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
    constant uint& batch_size [[buffer(6)]],
    constant uint& a_batch_stride [[buffer(7)]],
    constant uint& b_batch_stride [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= M || col >= N || batch >= batch_size) return;

    uint a_offset = batch * a_batch_stride;
    uint b_offset = batch * b_batch_stride;

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += A[a_offset + row * K + i] * B[b_offset + i * N + col];
    }
    C[batch * M * N + row * N + col] = sum;
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

const TRANSPOSE_BATCHED_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void transpose_batched_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= rows || col >= cols || batch >= batch_size) return;
    output[batch * cols * rows + col * rows + row] = input[batch * rows * cols + row * cols + col];
}
"#;

const TRANSPOSE_BATCHED_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void transpose_batched_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= rows || col >= cols || batch >= batch_size) return;
    output[batch * cols * rows + col * rows + row] = input[batch * rows * cols + row * cols + col];
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

const POW_KERNEL_SOURCE: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void pow_f32(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], constant float& exponent [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = pow(input[in_off], exponent);
}
"#
);

const POW_KERNEL_SOURCE_F16: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void pow_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], constant float& exponent [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = half(pow(float(input[in_off]), exponent));
}
"#
);

const CLAMP_KERNEL_SOURCE: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void clamp_f32(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], constant float& min_val [[buffer(6)]], constant float& max_val [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = clamp(input[in_off], min_val, max_val);
}
"#
);

const CLAMP_KERNEL_SOURCE_F16: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void clamp_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], constant float& min_val [[buffer(6)]], constant float& max_val [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = half(clamp(float(input[in_off]), min_val, max_val));
}
"#
);

/// MSL source for ternary where op (3 inputs + 3 stride arrays + shape + ndim + numel).
const WHERE_KERNEL_SOURCE: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void where_f32(device const float* condition [[buffer(0)]], device const float* x [[buffer(1)]], device const float* y [[buffer(2)]], device float* out [[buffer(3)]], constant uint* cond_strides [[buffer(4)]], constant uint* x_strides [[buffer(5)]], constant uint* y_strides [[buffer(6)]], constant uint* out_shape [[buffer(7)]], constant uint& ndim [[buffer(8)]], constant uint& numel [[buffer(9)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint c_off = nd_index_to_offset(id, out_shape, cond_strides, ndim);
    uint x_off = nd_index_to_offset(id, out_shape, x_strides, ndim);
    uint y_off = nd_index_to_offset(id, out_shape, y_strides, ndim);
    out[id] = (condition[c_off] != 0.0f) ? x[x_off] : y[y_off];
}
"#
);

const WHERE_KERNEL_SOURCE_F16: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void where_f16(device const half* condition [[buffer(0)]], device const half* x [[buffer(1)]], device const half* y [[buffer(2)]], device half* out [[buffer(3)]], constant uint* cond_strides [[buffer(4)]], constant uint* x_strides [[buffer(5)]], constant uint* y_strides [[buffer(6)]], constant uint* out_shape [[buffer(7)]], constant uint& ndim [[buffer(8)]], constant uint& numel [[buffer(9)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint c_off = nd_index_to_offset(id, out_shape, cond_strides, ndim);
    uint x_off = nd_index_to_offset(id, out_shape, x_strides, ndim);
    uint y_off = nd_index_to_offset(id, out_shape, y_strides, ndim);
    out[id] = (condition[c_off] != half(0)) ? x[x_off] : y[y_off];
}
"#
);

/// MSL source for masked_fill op (2 inputs + 2 stride arrays + shape + ndim + numel + fill_value).
const MASKED_FILL_KERNEL_SOURCE: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void masked_fill_f32(device const float* input [[buffer(0)]], device const float* mask [[buffer(1)]], device float* out [[buffer(2)]], constant uint* in_strides [[buffer(3)]], constant uint* mask_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], constant float& fill_value [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    uint m_off = nd_index_to_offset(id, out_shape, mask_strides, ndim);
    out[id] = (mask[m_off] != 0.0f) ? fill_value : input[in_off];
}
"#
);

const MASKED_FILL_KERNEL_SOURCE_F16: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void masked_fill_f16(device const half* input [[buffer(0)]], device const half* mask [[buffer(1)]], device half* out [[buffer(2)]], constant uint* in_strides [[buffer(3)]], constant uint* mask_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], constant float& fill_value [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    uint m_off = nd_index_to_offset(id, out_shape, mask_strides, ndim);
    out[id] = (mask[m_off] != half(0)) ? half(fill_value) : input[in_off];
}
"#
);

/// MSL source for triu (upper triangular) op.
const TRIU_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void triu_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant int& diagonal [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= rows || col >= cols || batch >= batch_size) return;
    uint idx = batch * rows * cols + row * cols + col;
    out[idx] = (int(col) >= int(row) + diagonal) ? input[idx] : 0.0f;
}
"#;

const TRIU_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void triu_f16(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant int& diagonal [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= rows || col >= cols || batch >= batch_size) return;
    uint idx = batch * rows * cols + row * cols + col;
    out[idx] = (int(col) >= int(row) + diagonal) ? input[idx] : half(0);
}
"#;

/// MSL source for tril (lower triangular) op.
const TRIL_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void tril_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant int& diagonal [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= rows || col >= cols || batch >= batch_size) return;
    uint idx = batch * rows * cols + row * cols + col;
    out[idx] = (int(col) <= int(row) + diagonal) ? input[idx] : 0.0f;
}
"#;

const TRIL_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void tril_f16(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant int& diagonal [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= rows || col >= cols || batch >= batch_size) return;
    uint idx = batch * rows * cols + row * cols + col;
    out[idx] = (int(col) <= int(row) + diagonal) ? input[idx] : half(0);
}
"#;

// ── Gather kernel sources ───────────────────────────────────────────────────

const GATHER_DIM0_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gather_dim0_f32(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_cols [[buffer(4)]],
    constant uint& out_cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= out_cols) return;
    int idx = indices[row * out_cols + col];
    output[row * out_cols + col] = input[idx * in_cols + col];
}
"#;

const GATHER_DIM0_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gather_dim0_f16(
    device const half* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_cols [[buffer(4)]],
    constant uint& out_cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= out_cols) return;
    int idx = indices[row * out_cols + col];
    output[row * out_cols + col] = input[idx * in_cols + col];
}
"#;

const GATHER_DIM1_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gather_dim1_f32(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_cols [[buffer(4)]],
    constant uint& out_cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= out_cols) return;
    int idx = indices[row * out_cols + col];
    output[row * out_cols + col] = input[row * in_cols + idx];
}
"#;

const GATHER_DIM1_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gather_dim1_f16(
    device const half* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_cols [[buffer(4)]],
    constant uint& out_cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= out_cols) return;
    int idx = indices[row * out_cols + col];
    output[row * out_cols + col] = input[row * in_cols + idx];
}
"#;

// ── IndexSelect kernel sources ──────────────────────────────────────────────

const INDEX_SELECT_DIM0_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void index_select_dim0_f32(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= num_indices || j >= cols) return;
    int idx = indices[i];
    output[i * cols + j] = input[idx * cols + j];
}
"#;

const INDEX_SELECT_DIM0_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void index_select_dim0_f16(
    device const half* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= num_indices || j >= cols) return;
    int idx = indices[i];
    output[i * cols + j] = input[idx * cols + j];
}
"#;

const INDEX_SELECT_DIM1_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void index_select_dim1_f32(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_cols [[buffer(4)]],
    constant uint& num_indices [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint i = gid.x;
    if (row >= rows || i >= num_indices) return;
    int idx = indices[i];
    output[row * num_indices + i] = input[row * in_cols + idx];
}
"#;

const INDEX_SELECT_DIM1_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void index_select_dim1_f16(
    device const half* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_cols [[buffer(4)]],
    constant uint& num_indices [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint i = gid.x;
    if (row >= rows || i >= num_indices) return;
    int idx = indices[i];
    output[row * num_indices + i] = input[row * in_cols + idx];
}
"#;

const GELU_KERNEL_SOURCE: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* in_strides [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    float x = input[in_off];
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    inner = clamp(inner, -10.0f, 10.0f);
    output[id] = x * 0.5f * (1.0f + tanh(inner));
}
"#
);

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

/// MSL source for f16 binary element-wise ops (add, sub, mul, div) with N-D stride-based indexing.
const BINARY_KERNEL_SOURCE_F16: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void elementwise_add_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] + b[b_off];
}
kernel void elementwise_sub_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] - b[b_off];
}
kernel void elementwise_mul_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] * b[b_off];
}
kernel void elementwise_div_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] / b[b_off];
}
"#
);

/// MSL source for f16 unary element-wise ops (neg, relu, exp, log, sqrt) with N-D stride-based indexing.
const UNARY_KERNEL_SOURCE_F16: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void elementwise_neg_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = -input[in_off];
}
kernel void elementwise_relu_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = max(input[in_off], (half)0);
}
kernel void elementwise_exp_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = exp(input[in_off]);
}
kernel void elementwise_log_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = log(input[in_off]);
}
kernel void elementwise_sqrt_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = sqrt(input[in_off]);
}
kernel void elementwise_abs_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = abs(input[in_off]);
}
kernel void elementwise_sign_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = sign(input[in_off]);
}
kernel void elementwise_tanh_f16(device const half* input [[buffer(0)]], device half* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = (half)tanh((float)input[in_off]);
}
"#
);

/// MSL source for f16 batched matmul with f32-intermediate accumulation.
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
    constant uint& batch_size [[buffer(6)]],
    constant uint& a_batch_stride [[buffer(7)]],
    constant uint& b_batch_stride [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= M || col >= N || batch >= batch_size) return;
    uint a_offset = batch * a_batch_stride;
    uint b_offset = batch * b_batch_stride;
    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += float(A[a_offset + row * K + i]) * float(B[b_offset + i * N + col]);
    }
    C[batch * M * N + row * N + col] = half(sum);
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

/// MSL source for strided copy (general transpose via arbitrary stride permutation).
const COPY_STRIDED_KERNEL_SOURCE: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void copy_strided_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* in_strides [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    output[id] = input[in_off];
}
"#
);

/// MSL source for f16 strided copy (general transpose).
const COPY_STRIDED_KERNEL_SOURCE_F16: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void copy_strided_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint* in_strides [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    output[id] = input[in_off];
}
"#
);

const GELU_KERNEL_SOURCE_F16: &str = const_format::concatcp!(
    r#"
#include <metal_stdlib>
using namespace metal;
"#,
    ND_INDEX_HELPER,
    r#"
kernel void gelu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint* in_strides [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    float x = float(input[in_off]);
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    inner = clamp(inner, -10.0f, 10.0f);
    output[id] = half(x * 0.5f * (1.0f + tanh(inner)));
}
"#
);

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

kernel void softmax_causal_f32(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant uint& batch_size [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint batch = gid.y;
    if (row >= rows || batch >= batch_size) return;
    uint offset = batch * rows * cols + row * cols;
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

kernel void softmax_causal_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], constant uint& batch_size [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint batch = gid.y;
    if (row >= rows || batch >= batch_size) return;
    uint offset = batch * rows * cols + row * cols;
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

const SUM_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sum_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& total_rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= total_rows) return;
    uint offset = row * cols;
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        sum += input[offset + j];
    }
    output[row] = sum;
}
"#;

const SUM_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sum_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& total_rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= total_rows) return;
    uint offset = row * cols;
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        sum += float(input[offset + j]);
    }
    output[row] = half(sum);
}
"#;

const MEAN_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void mean_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& total_rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= total_rows) return;
    uint offset = row * cols;
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        sum += input[offset + j];
    }
    output[row] = sum / float(cols);
}
"#;

const MEAN_KERNEL_SOURCE_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void mean_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& total_rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= total_rows) return;
    uint offset = row * cols;
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        sum += float(input[offset + j]);
    }
    output[row] = half(sum / float(cols));
}
"#;

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

// ── CNN kernel sources ─────────────────────────────────────────────────────

/// Conv1d MSL kernel. Buffer layout: input(0), weight(1), output(2),
/// then uint params: batch(3), in_channels(4), out_channels(5), in_length(6),
/// out_length(7), kernel_size(8), stride(9), padding(10).
pub(crate) const CONV1D_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void conv1d_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& in_channels [[buffer(4)]],
    constant uint& out_channels [[buffer(5)]],
    constant uint& in_length [[buffer(6)]],
    constant uint& out_length [[buffer(7)]],
    constant uint& kernel_size [[buffer(8)]],
    constant uint& stride [[buffer(9)]],
    constant uint& padding [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint o = gid.x;
    uint oc = gid.y;
    uint b = gid.z;
    if (o >= out_length || oc >= out_channels || b >= batch) return;

    float sum = 0.0f;
    for (uint ic = 0; ic < in_channels; ic++) {
        for (uint k = 0; k < kernel_size; k++) {
            int in_pos = int(o * stride + k) - int(padding);
            if (in_pos >= 0 && uint(in_pos) < in_length) {
                sum += input[b * in_channels * in_length + ic * in_length + in_pos]
                     * weight[oc * in_channels * kernel_size + ic * kernel_size + k];
            }
        }
    }
    output[b * out_channels * out_length + oc * out_length + o] = sum;
}
"#;

/// Conv2d MSL kernel. Buffer layout: input(0), weight(1), output(2),
/// then uint params: batch(3), in_channels(4), out_channels(5), in_h(6), in_w(7),
/// out_h(8), out_w(9), kh(10), kw(11), stride_h(12), stride_w(13), pad_h(14), pad_w(15).
pub(crate) const CONV2D_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void conv2d_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& in_channels [[buffer(4)]],
    constant uint& out_channels [[buffer(5)]],
    constant uint& in_h [[buffer(6)]],
    constant uint& in_w [[buffer(7)]],
    constant uint& out_h [[buffer(8)]],
    constant uint& out_w [[buffer(9)]],
    constant uint& kh [[buffer(10)]],
    constant uint& kw [[buffer(11)]],
    constant uint& stride_h [[buffer(12)]],
    constant uint& stride_w [[buffer(13)]],
    constant uint& pad_h [[buffer(14)]],
    constant uint& pad_w [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint combined = gid.y;
    uint b = gid.z;
    uint oc = combined % out_channels;
    uint oh = combined / out_channels;
    if (ow >= out_w || oh >= out_h || b >= batch) return;

    float sum = 0.0f;
    for (uint ic = 0; ic < in_channels; ic++) {
        for (uint i = 0; i < kh; i++) {
            for (uint j = 0; j < kw; j++) {
                int ih = int(oh * stride_h + i) - int(pad_h);
                int iw = int(ow * stride_w + j) - int(pad_w);
                if (ih >= 0 && uint(ih) < in_h && iw >= 0 && uint(iw) < in_w) {
                    sum += input[b * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw]
                         * weight[oc * in_channels * kh * kw + ic * kh * kw + i * kw + j];
                }
            }
        }
    }
    output[b * out_channels * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = sum;
}
"#;

/// BatchNorm MSL kernel. Buffer layout: input(0), mean(1), var(2), weight(3), bias(4), output(5),
/// then uint params: batch(6), channels(7), spatial(8), then float param: eps(9).
pub(crate) const BATCH_NORM_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batch_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* var_ [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& channels [[buffer(7)]],
    constant uint& spatial [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint s = gid.x;
    uint c = gid.y;
    uint b = gid.z;
    if (s >= spatial || c >= channels || b >= batch) return;

    uint idx = b * channels * spatial + c * spatial + s;
    float inv_std = 1.0f / sqrt(var_[c] + eps);
    output[idx] = (input[idx] - mean[c]) * inv_std * weight[c] + bias[c];
}
"#;

/// MaxPool2d MSL kernel. Buffer layout: input(0), output(1),
/// then uint params: batch(2), channels(3), in_h(4), in_w(5), out_h(6), out_w(7),
/// kh(8), kw(9), stride_h(10), stride_w(11), pad_h(12), pad_w(13).
pub(crate) const MAX_POOL2D_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void max_pool2d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& in_h [[buffer(4)]],
    constant uint& in_w [[buffer(5)]],
    constant uint& out_h [[buffer(6)]],
    constant uint& out_w [[buffer(7)]],
    constant uint& kh [[buffer(8)]],
    constant uint& kw [[buffer(9)]],
    constant uint& stride_h [[buffer(10)]],
    constant uint& stride_w [[buffer(11)]],
    constant uint& pad_h [[buffer(12)]],
    constant uint& pad_w [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint combined = gid.y;
    uint b = gid.z;
    uint c = combined % channels;
    uint oh = combined / channels;
    if (ow >= out_w || oh >= out_h || b >= batch) return;

    float max_val = -1e30f;
    for (uint i = 0; i < kh; i++) {
        for (uint j = 0; j < kw; j++) {
            int ih = int(oh * stride_h + i) - int(pad_h);
            int iw = int(ow * stride_w + j) - int(pad_w);
            if (ih >= 0 && uint(ih) < in_h && iw >= 0 && uint(iw) < in_w) {
                float val = input[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
                max_val = max(max_val, val);
            }
        }
    }
    output[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = max_val;
}
"#;

/// AvgPool2d MSL kernel. Same buffer layout as max_pool2d.
pub(crate) const AVG_POOL2D_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void avg_pool2d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& in_h [[buffer(4)]],
    constant uint& in_w [[buffer(5)]],
    constant uint& out_h [[buffer(6)]],
    constant uint& out_w [[buffer(7)]],
    constant uint& kh [[buffer(8)]],
    constant uint& kw [[buffer(9)]],
    constant uint& stride_h [[buffer(10)]],
    constant uint& stride_w [[buffer(11)]],
    constant uint& pad_h [[buffer(12)]],
    constant uint& pad_w [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint combined = gid.y;
    uint b = gid.z;
    uint c = combined % channels;
    uint oh = combined / channels;
    if (ow >= out_w || oh >= out_h || b >= batch) return;

    float sum = 0.0f;
    uint count = 0;
    for (uint i = 0; i < kh; i++) {
        for (uint j = 0; j < kw; j++) {
            int ih = int(oh * stride_h + i) - int(pad_h);
            int iw = int(ow * stride_w + j) - int(pad_w);
            if (ih >= 0 && uint(ih) < in_h && iw >= 0 && uint(iw) < in_w) {
                sum += input[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
                count++;
            }
        }
    }
    output[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = count > 0 ? sum / float(count) : 0.0f;
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
        // Reuse embedding FFI (same signature: input, indices, output, count, dim)
        let result = unsafe {
            ffi::gpu_bridge_compute_embedding(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_indices.raw_handle() as *const _,
                buf_out.raw_handle(),
                num_indices as u32,
                cols as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("IndexSelect dim0 dispatch failed".to_string())) }
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
        let cb = unsafe {
            ffi::gpu_bridge_compute_embedding_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_indices.raw_handle() as *const _,
                buf_out.raw_handle(),
                num_indices as u32,
                cols as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking index_select dim0 dispatch failed".to_string())) } else { Ok(cb) }
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
        let result = unsafe {
            ffi::gpu_bridge_compute_gather(
                self.handle,
                buf_input.raw_handle() as *const _,
                buf_indices.raw_handle() as *const _,
                buf_out.raw_handle(),
                rows as u32,
                in_cols as u32,
                num_indices as u32,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("IndexSelect dim1 dispatch failed".to_string())) }
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
        let cb = unsafe {
            ffi::gpu_bridge_compute_gather_nb(
                self.handle,
                queue,
                buf_input.raw_handle() as *const _,
                buf_indices.raw_handle() as *const _,
                buf_out.raw_handle(),
                rows as u32,
                in_cols as u32,
                num_indices as u32,
            )
        };
        if cb.is_null() { Err(GpuError::ComputeFailed("Non-blocking index_select dim1 dispatch failed".to_string())) } else { Ok(cb) }
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
                    "transpose_batched_f32" => TRANSPOSE_BATCHED_KERNEL_SOURCE_F16,
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
                    "sum_f32" => SUM_KERNEL_SOURCE_F16,
                    "mean_f32" => MEAN_KERNEL_SOURCE_F16,
                    "copy_strided_f32" => COPY_STRIDED_KERNEL_SOURCE_F16,
                    "pow_f32" => POW_KERNEL_SOURCE_F16,
                    "clamp_f32" => CLAMP_KERNEL_SOURCE_F16,
                    "where_f32" => WHERE_KERNEL_SOURCE_F16,
                    "masked_fill_f32" => MASKED_FILL_KERNEL_SOURCE_F16,
                    "triu_f32" => TRIU_KERNEL_SOURCE_F16,
                    "tril_f32" => TRIL_KERNEL_SOURCE_F16,
                    "gather_dim0_f32" => GATHER_DIM0_KERNEL_SOURCE_F16,
                    "gather_dim1_f32" => GATHER_DIM1_KERNEL_SOURCE_F16,
                    "index_select_dim0_f32" => INDEX_SELECT_DIM0_KERNEL_SOURCE_F16,
                    "index_select_dim1_f32" => INDEX_SELECT_DIM1_KERNEL_SOURCE_F16,
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
                    "transpose_batched_f32" => TRANSPOSE_BATCHED_KERNEL_SOURCE,
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
                    "sum_f32" => SUM_KERNEL_SOURCE,
                    "mean_f32" => MEAN_KERNEL_SOURCE,
                    "copy_strided_f32" => COPY_STRIDED_KERNEL_SOURCE,
                    "pow_f32" => POW_KERNEL_SOURCE,
                    "clamp_f32" => CLAMP_KERNEL_SOURCE,
                    "where_f32" => WHERE_KERNEL_SOURCE,
                    "masked_fill_f32" => MASKED_FILL_KERNEL_SOURCE,
                    "triu_f32" => TRIU_KERNEL_SOURCE,
                    "tril_f32" => TRIL_KERNEL_SOURCE,
                    "gather_dim0_f32" => GATHER_DIM0_KERNEL_SOURCE,
                    "gather_dim1_f32" => GATHER_DIM1_KERNEL_SOURCE,
                    "index_select_dim0_f32" => INDEX_SELECT_DIM0_KERNEL_SOURCE,
                    "index_select_dim1_f32" => INDEX_SELECT_DIM1_KERNEL_SOURCE,
                    "conv1d_f32" => CONV1D_KERNEL_SOURCE,
                    "conv2d_f32" => CONV2D_KERNEL_SOURCE,
                    "batch_norm_f32" => BATCH_NORM_KERNEL_SOURCE,
                    "max_pool2d_f32" => MAX_POOL2D_KERNEL_SOURCE,
                    "avg_pool2d_f32" => AVG_POOL2D_KERNEL_SOURCE,
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
        let (source, func) = Self::resolve_kernel("matmul_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
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

    pub fn dispatch_transpose_batched_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        batch_size: usize, rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("transpose_batched_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_transpose_batched(buf_input, buf_output, batch_size, rows, cols)
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

    pub fn dispatch_pow_nd_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, exponent: f32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("pow_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_pow_nd(buf_input, in_strides, buf_out, out_shape, ndim, numel, exponent)
    }

    pub fn dispatch_pow_nd_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, exponent: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("pow_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_pow_nd_nb(queue, buf_input, in_strides, buf_out, out_shape, ndim, numel, exponent)
    }

    pub fn dispatch_clamp_nd_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, min_val: f32, max_val: f32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("clamp_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_clamp_nd(buf_input, in_strides, buf_out, out_shape, ndim, numel, min_val, max_val)
    }

    pub fn dispatch_clamp_nd_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, min_val: f32, max_val: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("clamp_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
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
        let (source, func) = Self::resolve_kernel("where_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
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
        let (source, func) = Self::resolve_kernel("where_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
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
        let (source, func) = Self::resolve_kernel("masked_fill_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_masked_fill_nd(buf_input, in_strides, buf_mask, mask_strides, buf_out, out_shape, ndim, numel, fill_value)
    }

    pub fn dispatch_masked_fill_nd_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, in_strides: &[u32; MAX_DIMS],
        buf_mask: &Buffer, mask_strides: &[u32; MAX_DIMS],
        buf_out: &Buffer, out_shape: &[u32; MAX_DIMS],
        ndim: u32, numel: u32, fill_value: f32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("masked_fill_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_masked_fill_nd_nb(queue, buf_input, in_strides, buf_mask, mask_strides, buf_out, out_shape, ndim, numel, fill_value)
    }

    // ── Triu / Tril ─────────────────────────────────────────────────

    pub fn dispatch_triu_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, buf_out: &Buffer,
        batch_size: usize, rows: usize, cols: usize, diagonal: i32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("triu_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_triangular(buf_input, buf_out, batch_size, rows, cols, diagonal)
    }

    pub fn dispatch_triu_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_out: &Buffer,
        batch_size: usize, rows: usize, cols: usize, diagonal: i32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("triu_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_triangular_nb(queue, buf_input, buf_out, batch_size, rows, cols, diagonal)
    }

    pub fn dispatch_tril_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, buf_out: &Buffer,
        batch_size: usize, rows: usize, cols: usize, diagonal: i32,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("tril_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_triangular(buf_input, buf_out, batch_size, rows, cols, diagonal)
    }

    pub fn dispatch_tril_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_out: &Buffer,
        batch_size: usize, rows: usize, cols: usize, diagonal: i32,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("tril_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
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
        batch_size: usize, rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("softmax_causal_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_softmax_causal(buf_input, buf_output, batch_size, rows, cols)
    }

    pub fn dispatch_sum_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("sum_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_mean_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("mean_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
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
        buf_input: &Buffer, buf_output: &Buffer, batch_size: usize, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("softmax_causal_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_softmax_causal_nb(queue, buf_input, buf_output, batch_size, rows, cols)
    }

    pub fn dispatch_sum_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("sum_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_mean_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("mean_f32", dtype);
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

    pub fn dispatch_matmul_batched_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_a: &Buffer, buf_b: &Buffer, buf_c: &Buffer,
        m: usize, n: usize, k: usize,
        batch_size: usize, a_batch_stride: usize, b_batch_stride: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("matmul_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_matmul_batched_nb(queue, buf_a, buf_b, buf_c, m, n, k, batch_size, a_batch_stride, b_batch_stride)
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

    pub fn dispatch_transpose_batched_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer, batch_size: usize, rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("transpose_batched_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_transpose_batched_nb(queue, buf_input, buf_output, batch_size, rows, cols)
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
        let pipeline = self.get_or_create(device, source, &func)?;
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
        let pipeline = self.get_or_create(device, source, &func)?;
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
        let pipeline = self.get_or_create(device, source, &func)?;
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
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_unary_nd_nb(queue, buf_input, in_strides, buf_out, out_shape, ndim, numel)
    }

    // ── Gather dispatch (typed) ──────────────────────────────────────────

    pub fn dispatch_gather_typed(
        &self, device: &Device, dtype: DType, kernel_base: &str,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        rows: usize, in_cols: usize, out_cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel(kernel_base, dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_gather(buf_input, buf_indices, buf_out, rows, in_cols, out_cols)
    }

    pub fn dispatch_gather_typed_nb(
        &self, device: &Device, dtype: DType, kernel_base: &str,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        rows: usize, in_cols: usize, out_cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel(kernel_base, dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_gather_nb(queue, buf_input, buf_indices, buf_out, rows, in_cols, out_cols)
    }

    // ── IndexSelect dispatch (typed) ─────────────────────────────────────

    pub fn dispatch_index_select_dim0_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        num_indices: usize, cols: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("index_select_dim0_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_index_select_dim0(buf_input, buf_indices, buf_out, num_indices, cols)
    }

    pub fn dispatch_index_select_dim0_typed_nb(
        &self, device: &Device, dtype: DType,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        num_indices: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("index_select_dim0_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_index_select_dim0_nb(queue, buf_input, buf_indices, buf_out, num_indices, cols)
    }

    pub fn dispatch_index_select_dim1_typed(
        &self, device: &Device, dtype: DType,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        rows: usize, in_cols: usize, num_indices: usize,
    ) -> Result<()> {
        let (source, func) = Self::resolve_kernel("index_select_dim1_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
        pipeline.dispatch_index_select_dim1(buf_input, buf_indices, buf_out, rows, in_cols, num_indices)
    }

    pub fn dispatch_index_select_dim1_typed_nb(
        &self, device: &Device, dtype: DType,
        queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_indices: &Buffer, buf_out: &Buffer,
        rows: usize, in_cols: usize, num_indices: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let (source, func) = Self::resolve_kernel("index_select_dim1_f32", dtype);
        let pipeline = self.get_or_create(device, source, &func)?;
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
