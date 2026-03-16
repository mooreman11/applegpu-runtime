//! Template-based MSL kernel source generation.
//! Replaces duplicated per-dtype kernel string constants with parameterized generators.

use crate::tensor::DType;

/// MSL type name for a given DType.
pub fn metal_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 => "float",
        DType::Float16 => "half",
        DType::BFloat16 => "bfloat",
        DType::Int32 => "int",
        DType::Int64 => "long",
        DType::Int8 => "char",
        DType::Int16 => "short",
        DType::UInt8 => "uchar",
        DType::UInt32 => "uint",
        DType::Bool => "bool",
        DType::Float64 => panic!("Float64 is not supported in MSL"),
    }
}

/// Suffix for kernel function names.
pub fn dtype_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 => "_f32",
        DType::Float16 => "_f16",
        DType::BFloat16 => "_bf16",
        DType::Int32 => "_i32",
        DType::Int64 => "_i64",
        DType::Int8 => "_i8",
        DType::Int16 => "_i16",
        DType::UInt8 => "_u8",
        DType::UInt32 => "_u32",
        DType::Bool => "_bool",
        DType::Float64 => panic!("Float64 is not supported in MSL"),
    }
}

/// N-D index helper shared by all element-wise kernels.
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

/// Generate binary element-wise kernel source (add, sub, mul, div) for a given dtype.
pub fn binary_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void elementwise_add{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] + b[b_off];
}}
kernel void elementwise_sub{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] - b[b_off];
}}
kernel void elementwise_mul{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] * b[b_off];
}}
kernel void elementwise_div{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] / b[b_off];
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
    )
}

/// Generate unary kernel source for ops that work on ALL numeric types (neg, abs, sign).
pub fn unary_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void elementwise_neg{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = -input[in_off];
}}
kernel void elementwise_abs{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = abs(input[in_off]);
}}
kernel void elementwise_sign{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = sign(input[in_off]);
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
    )
}

/// Generate unary kernel source for FLOAT-ONLY ops (exp, log, sqrt, relu, tanh).
pub fn float_unary_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let zero = match dtype {
        DType::Float16 | DType::BFloat16 => format!("({})0", t),
        _ => "0.0f".to_string(),
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void elementwise_exp{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = exp(input[in_off]);
}}
kernel void elementwise_log{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = log(input[in_off]);
}}
kernel void elementwise_sqrt{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = sqrt(input[in_off]);
}}
kernel void elementwise_relu{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = max(input[in_off], {zero});
}}
kernel void elementwise_tanh{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = tanh(input[in_off]);
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s, zero = zero,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_type_mapping() {
        assert_eq!(metal_type(DType::Float32), "float");
        assert_eq!(metal_type(DType::Float16), "half");
        assert_eq!(metal_type(DType::BFloat16), "bfloat");
        assert_eq!(metal_type(DType::Int32), "int");
        assert_eq!(metal_type(DType::Int64), "long");
        assert_eq!(metal_type(DType::UInt8), "uchar");
        assert_eq!(metal_type(DType::Bool), "bool");
    }

    #[test]
    fn suffix_mapping() {
        assert_eq!(dtype_suffix(DType::Float32), "_f32");
        assert_eq!(dtype_suffix(DType::Int64), "_i64");
        assert_eq!(dtype_suffix(DType::BFloat16), "_bf16");
    }

    #[test]
    #[should_panic(expected = "Float64")]
    fn float64_panics() {
        metal_type(DType::Float64);
    }

    #[test]
    fn binary_kernel_contains_correct_type() {
        let src = binary_kernel_source(DType::Int32);
        assert!(src.contains("device const int*"));
        assert!(src.contains("device int* out"));
        assert!(src.contains("elementwise_add_i32"));
        assert!(src.contains("elementwise_sub_i32"));
        assert!(src.contains("elementwise_mul_i32"));
        assert!(src.contains("elementwise_div_i32"));
        assert!(src.contains("nd_index_to_offset"));
    }

    #[test]
    fn binary_kernel_f32_matches_pattern() {
        let src = binary_kernel_source(DType::Float32);
        assert!(src.contains("device const float*"));
        assert!(src.contains("elementwise_add_f32"));
    }

    #[test]
    fn unary_kernel_contains_correct_type() {
        let src = unary_kernel_source(DType::BFloat16);
        assert!(src.contains("device const bfloat*"));
        assert!(src.contains("elementwise_neg_bf16"));
        assert!(src.contains("elementwise_abs_bf16"));
        assert!(src.contains("elementwise_sign_bf16"));
    }

    #[test]
    fn float_unary_kernel_has_transcendentals() {
        let src = float_unary_kernel_source(DType::Float32);
        assert!(src.contains("elementwise_exp_f32"));
        assert!(src.contains("elementwise_relu_f32"));
        assert!(src.contains("elementwise_tanh_f32"));
    }
}
