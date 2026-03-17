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
kernel void elementwise_sin{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = sin(input[in_off]);
}}
kernel void elementwise_cos{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = cos(input[in_off]);
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s, zero = zero,
    )
}

/// Whether a dtype needs float intermediate accumulation for precision.
fn needs_float_acc(dtype: DType) -> bool {
    matches!(dtype, DType::Float16 | DType::BFloat16)
}

/// Cast expression: wraps `expr` in a cast to `t` if accumulation is in float.
fn to_type(expr: &str, t: &str, dtype: DType) -> String {
    if needs_float_acc(dtype) {
        format!("{}({})", t, expr)
    } else {
        expr.to_string()
    }
}

/// Cast expression to float for intermediate computation.
fn to_float(expr: &str, dtype: DType) -> String {
    if needs_float_acc(dtype) {
        format!("float({})", expr)
    } else {
        expr.to_string()
    }
}

// ── Task 8a: Element-wise adjacent ops ──────────────────────────────────────

/// Generate scalar_mul kernel source for a given dtype.
/// Scale parameter stays float; read/write in native dtype.
pub fn scalar_mul_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let body = if needs_float_acc(dtype) {
        format!("if (id < count) {{ output[id] = {t}(float(input[id]) * scale); }}", t = t)
    } else {
        format!("if (id < count) {{ output[id] = {t}(input[id] * {t}(scale)); }}", t = t)
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void scalar_mul{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {{
    {body}
}}
"#,
        s = s, t = t, body = body,
    )
}

/// Generate pow kernel source for a given dtype.
/// Exponent stays float; uses float intermediate for half types.
pub fn pow_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let compute = if needs_float_acc(dtype) {
        format!("out[id] = {t}(pow(float(input[in_off]), exponent));", t = t)
    } else {
        "out[id] = pow(input[in_off], exponent);".to_string()
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void pow{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], constant float& exponent [[buffer(6)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    {compute}
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s, compute = compute,
    )
}

/// Generate clamp kernel source for a given dtype.
/// Min/max stay float; uses float intermediate for half types.
pub fn clamp_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let compute = if needs_float_acc(dtype) {
        format!("out[id] = {t}(clamp(float(input[in_off]), min_val, max_val));", t = t)
    } else {
        "out[id] = clamp(input[in_off], min_val, max_val);".to_string()
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void clamp{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], constant float& min_val [[buffer(6)]], constant float& max_val [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    {compute}
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s, compute = compute,
    )
}

/// Generate GELU kernel source. Always uses float intermediate.
pub fn gelu_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let (load, store) = if needs_float_acc(dtype) || dtype == DType::Float32 {
        (to_float("input[in_off]", dtype),
         if dtype == DType::Float32 { "output[id] = x * 0.5f * (1.0f + tanh(inner));".to_string() }
         else { format!("output[id] = {t}(x * 0.5f * (1.0f + tanh(inner)));", t = t) })
    } else {
        // Int types shouldn't use GELU, but handle gracefully
        (format!("float(input[in_off])"),
         format!("output[id] = {t}(x * 0.5f * (1.0f + tanh(inner)));", t = t))
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void gelu{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint* in_strides [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    float x = {load};
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    inner = clamp(inner, -10.0f, 10.0f);
    {store}
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s, load = load, store = store,
    )
}

pub fn sigmoid_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let store = if dtype == DType::Float32 {
        "output[id] = 1.0f / (1.0f + exp(-x));".to_string()
    } else {
        format!("output[id] = {t}(1.0f / (1.0f + exp(-x)));", t = t)
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void sigmoid{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint* in_strides [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    float x = {load};
    {store}
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
        load = to_float("input[in_off]", dtype),
        store = store,
    )
}

// ── Task 8b: Reduction ops ──────────────────────────────────────────────────

/// Generate softmax kernel source. Uses float accumulation for half types.
pub fn softmax_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| to_float(e, dtype);
    let store = |e: &str| to_type(e, t, dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void softmax{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {{
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = {load_first};
    for (uint j = 1; j < cols; j++) {{ max_val = max(max_val, {load_j}); }}
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {{ float e = exp({load_j} - max_val); output[offset + j] = {store_e}; sum += e; }}
    for (uint j = 0; j < cols; j++) {{ output[offset + j] = {store_div}; }}
}}
"#,
        t = t, s = s,
        load_first = load("input[offset]"),
        load_j = load("input[offset + j]"),
        store_e = store("e"),
        store_div = if acc { format!("{}(float(output[offset + j]) / sum)", t) } else { "output[offset + j] / sum".to_string() },
    )
}

/// Generate log_softmax kernel source. Uses float accumulation for half types.
/// Numerically stable: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
pub fn log_softmax_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let load = |e: &str| to_float(e, dtype);
    let store = |e: &str| to_type(e, t, dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void log_softmax{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {{
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = {load_first};
    for (uint j = 1; j < cols; j++) {{ max_val = max(max_val, {load_j}); }}
    float log_sum_exp = 0.0f;
    for (uint j = 0; j < cols; j++) {{ log_sum_exp += exp({load_j} - max_val); }}
    log_sum_exp = log(log_sum_exp);
    for (uint j = 0; j < cols; j++) {{ output[offset + j] = {store_out}; }}
}}
"#,
        t = t, s = s,
        load_first = load("input[offset]"),
        load_j = load("input[offset + j]"),
        store_out = store(&format!("{} - max_val - log_sum_exp", load("input[offset + j]"))),
    )
}

/// Generate softmax_causal kernel source.
pub fn softmax_causal_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let zero = if acc { format!("{}(0.0f)", t) } else { "0.0f".to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void softmax_causal{s}(device const {t}* input [[buffer(0)]], device {t}* output [[buffer(1)]], constant uint& batch_size [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {{
    uint row = gid.x;
    uint batch = gid.y;
    if (row >= rows || batch >= batch_size) return;
    uint offset = batch * rows * cols + row * cols;
    float max_val = -1e9f;
    for (uint j = 0; j <= row && j < cols; j++) max_val = max(max_val, {load});
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {{
        if (j <= row) {{ float e = exp({load} - max_val); output[offset + j] = {store_e}; sum += e; }}
        else {{ output[offset + j] = {zero}; }}
    }}
    for (uint j = 0; j <= row && j < cols; j++) output[offset + j] = {store_div};
}}
"#,
        t = t, s = s, zero = zero,
        load = if acc { format!("float(input[offset + j])") } else { "input[offset + j]".to_string() },
        store_e = if acc { format!("{}(e)", t) } else { "e".to_string() },
        store_div = if acc { format!("{}(float(output[offset + j]) / sum)", t) } else { "output[offset + j] / sum".to_string() },
    )
}

/// Generate argmax kernel source. Output is always Int32 regardless of input dtype.
pub fn argmax_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void argmax{s}(device const {t}* input [[buffer(0)]], device int* output [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint row [[thread_position_in_grid]]) {{
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = {load_first}; int max_idx = 0;
    for (uint j = 1; j < cols; j++) {{ if ({load_j} > max_val) {{ max_val = {load_j}; max_idx = int(j); }} }}
    output[row] = max_idx;
}}
"#,
        t = t, s = s,
        load_first = if acc { "float(input[offset])".to_string() } else { "input[offset]".to_string() },
        load_j = if acc { "float(input[offset + j])".to_string() } else { "input[offset + j]".to_string() },
    )
}

/// Generate sum reduction kernel source. Uses float accumulation.
pub fn sum_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void sum{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint& total_rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {{
    if (row >= total_rows) return;
    uint offset = row * cols;
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {{
        sum += {load};
    }}
    output[row] = {store};
}}
"#,
        t = t, s = s,
        load = if acc { "float(input[offset + j])".to_string() } else { "input[offset + j]".to_string() },
        store = if acc { format!("{}(sum)", t) } else { "sum".to_string() },
    )
}

/// Generate mean reduction kernel source. Uses float accumulation.
pub fn mean_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void mean{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint& total_rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {{
    if (row >= total_rows) return;
    uint offset = row * cols;
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {{
        sum += {load};
    }}
    output[row] = {store};
}}
"#,
        t = t, s = s,
        load = if acc { "float(input[offset + j])".to_string() } else { "input[offset + j]".to_string() },
        store = if acc { format!("{}(sum / float(cols))", t) } else { "sum / float(cols)".to_string() },
    )
}

/// Generate variance kernel source. Reduces along last dimension.
/// correction is baked into the kernel (0 = population, 1 = sample/Bessel's).
/// Uses the same dispatch signature as mean/sum (rows, cols).
pub fn var_kernel_source_with_correction(dtype: DType, correction: u32) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    // Bake correction into function name for caching: var_c0 or var_c1
    let fn_suffix = format!("{s}_c{correction}");
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void var{fn_suffix}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint& total_rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {{
    if (row >= total_rows) return;
    uint offset = row * cols;
    float mean_val = 0.0f;
    for (uint j = 0; j < cols; j++) {{
        mean_val += {load};
    }}
    mean_val /= float(cols);
    float var_val = 0.0f;
    for (uint j = 0; j < cols; j++) {{
        float diff = {load} - mean_val;
        var_val += diff * diff;
    }}
    float denom = float(cols) - {correction}.0f;
    if (denom <= 0.0f) denom = 1.0f;
    output[row] = {store};
}}
"#,
        t = t, fn_suffix = fn_suffix, correction = correction,
        load = if acc { "float(input[offset + j])".to_string() } else { "input[offset + j]".to_string() },
        store = if acc { format!("{}(var_val / denom)", t) } else { "var_val / denom".to_string() },
    )
}

/// Convenience: generate var kernel with default correction=1 (Bessel's).
pub fn var_kernel_source(dtype: DType) -> String {
    var_kernel_source_with_correction(dtype, 1)
}

/// Generate N-D add_bias kernel source. Adds 1D bias along dim 1 (channels).
/// Works for any N-D input (N >= 2):
///   - 2D [rows, cols]: channel_stride=1, num_channels=cols -> bias[id % cols]
///   - 3D [B, C, L]: channel_stride=L, num_channels=C -> bias[(id/L) % C]
///   - 4D [B, C, H, W]: channel_stride=H*W, num_channels=C -> bias[(id/(H*W)) % C]
pub fn add_bias_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void add_bias{s}(device const {t}* input [[buffer(0)]], device const {t}* bias [[buffer(1)]], device {t}* output [[buffer(2)]], constant uint& numel [[buffer(3)]], constant uint& num_channels [[buffer(4)]], constant uint& channel_stride [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint channel = (id / channel_stride) % num_channels;
    output[id] = input[id] + bias[channel];
}}
"#,
        t = t, s = s,
    )
}

// ── Task 8c: Conditional/index ops ──────────────────────────────────────────

/// Generate where (ternary conditional) kernel source.
pub fn where_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let zero_cmp = if needs_float_acc(dtype) { format!("{}(0)", t) } else { "0.0f".to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void where{s}(device const {t}* condition [[buffer(0)]], device const {t}* x [[buffer(1)]], device const {t}* y [[buffer(2)]], device {t}* out [[buffer(3)]], constant uint* cond_strides [[buffer(4)]], constant uint* x_strides [[buffer(5)]], constant uint* y_strides [[buffer(6)]], constant uint* out_shape [[buffer(7)]], constant uint& ndim [[buffer(8)]], constant uint& numel [[buffer(9)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint c_off = nd_index_to_offset(id, out_shape, cond_strides, ndim);
    uint x_off = nd_index_to_offset(id, out_shape, x_strides, ndim);
    uint y_off = nd_index_to_offset(id, out_shape, y_strides, ndim);
    out[id] = (condition[c_off] != {zero_cmp}) ? x[x_off] : y[y_off];
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s, zero_cmp = zero_cmp,
    )
}

/// Generate masked_fill kernel source.
pub fn masked_fill_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let zero_cmp = if acc { format!("{}(0)", t) } else { "0.0f".to_string() };
    let fill = if acc { format!("{}(fill_value)", t) } else { "fill_value".to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void masked_fill{s}(device const {t}* input [[buffer(0)]], device const {t}* mask [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* in_strides [[buffer(3)]], constant uint* mask_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], constant float& fill_value [[buffer(8)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    uint m_off = nd_index_to_offset(id, out_shape, mask_strides, ndim);
    out[id] = (mask[m_off] != {zero_cmp}) ? {fill} : input[in_off];
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s, zero_cmp = zero_cmp, fill = fill,
    )
}

/// Generate triu (upper triangular) kernel source.
pub fn triu_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let zero = if needs_float_acc(dtype) { format!("{}(0)", t) } else if dtype.is_float() { "0.0f".to_string() } else { "0".to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void triu{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* out [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant int& diagonal [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= rows || col >= cols || batch >= batch_size) return;
    uint idx = batch * rows * cols + row * cols + col;
    out[idx] = (int(col) >= int(row) + diagonal) ? input[idx] : {zero};
}}
"#,
        t = t, s = s, zero = zero,
    )
}

/// Generate tril (lower triangular) kernel source.
pub fn tril_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let zero = if needs_float_acc(dtype) { format!("{}(0)", t) } else if dtype.is_float() { "0.0f".to_string() } else { "0".to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void tril{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* out [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant int& diagonal [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= rows || col >= cols || batch >= batch_size) return;
    uint idx = batch * rows * cols + row * cols + col;
    out[idx] = (int(col) <= int(row) + diagonal) ? input[idx] : {zero};
}}
"#,
        t = t, s = s, zero = zero,
    )
}

/// Generate gather kernel source for a given dim.
pub fn gather_kernel_source(dtype: DType, dim: usize) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let dim_name = if dim == 0 { "dim0" } else { "dim1" };
    if dim == 0 {
        format!(
            r#"#include <metal_stdlib>
using namespace metal;

kernel void gather_{dn}{s}(
    device const {t}* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device {t}* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& idx_rows [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint col = gid.x;
    uint row = gid.y;
    if (row >= idx_rows || col >= cols) return;
    int src_row = indices[row * cols + col];
    output[row * cols + col] = input[src_row * cols + col];
}}
"#,
            t = t, s = s, dn = dim_name,
        )
    } else {
        format!(
            r#"#include <metal_stdlib>
using namespace metal;

kernel void gather_{dn}{s}(
    device const {t}* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device {t}* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_cols [[buffer(4)]],
    constant uint& idx_cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint col = gid.x;
    uint row = gid.y;
    if (row >= rows || col >= idx_cols) return;
    int src_col = indices[row * idx_cols + col];
    output[row * idx_cols + col] = input[row * in_cols + src_col];
}}
"#,
            t = t, s = s, dn = dim_name,
        )
    }
}

/// Generate index_select kernel source for a given dim.
pub fn index_select_kernel_source(dtype: DType, dim: usize) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let dim_name = if dim == 0 { "dim0" } else { "dim1" };
    if dim == 0 {
        format!(
            r#"#include <metal_stdlib>
using namespace metal;

kernel void index_select_{dn}{s}(
    device const {t}* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device {t}* output [[buffer(2)]],
    constant uint& in_rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& num_indices [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint col = gid.x;
    uint i = gid.y;
    if (i >= num_indices || col >= cols) return;
    int src_row = indices[i];
    output[i * cols + col] = input[src_row * cols + col];
}}
"#,
            t = t, s = s, dn = dim_name,
        )
    } else {
        format!(
            r#"#include <metal_stdlib>
using namespace metal;

kernel void index_select_{dn}{s}(
    device const {t}* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device {t}* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_cols [[buffer(4)]],
    constant uint& num_indices [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint col = gid.x;
    uint row = gid.y;
    if (row >= rows || col >= num_indices) return;
    int src_col = indices[col];
    output[row * num_indices + col] = input[row * in_cols + src_col];
}}
"#,
            t = t, s = s, dn = dim_name,
        )
    }
}

// ── Task 8d: Complex ops ────────────────────────────────────────────────────

/// Generate matmul kernel source. Uses float accumulation for half types.
pub fn matmul_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let store = if acc { format!("C[batch * M * N + row * N + col] = {}(sum);", t) }
               else { "C[batch * M * N + row * N + col] = sum;".to_string() };
    let load_a = if acc { "float(A[a_offset + row * K + i])".to_string() } else { "A[a_offset + row * K + i]".to_string() };
    let load_b = if acc { "float(B[b_offset + i * N + col])".to_string() } else { "B[b_offset + i * N + col]".to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void matmul{s}(
    device const {t}* A [[buffer(0)]],
    device const {t}* B [[buffer(1)]],
    device {t}* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& a_batch_stride [[buffer(7)]],
    constant uint& b_batch_stride [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= M || col >= N || batch >= batch_size) return;
    uint a_offset = batch * a_batch_stride;
    uint b_offset = batch * b_batch_stride;
    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {{
        sum += {load_a} * {load_b};
    }}
    {store}
}}
"#,
        t = t, s = s, load_a = load_a, load_b = load_b, store = store,
    )
}

/// Generate layer_norm kernel source. Uses float accumulation for half types.
pub fn layer_norm_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void layer_norm{s}(
    device const {t}* input [[buffer(0)]],
    device const {t}* gamma [[buffer(1)]],
    device const {t}* beta [[buffer(2)]],
    device {t}* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {{
    if (row >= rows) return;
    uint offset = row * cols;
    float mean = 0.0f;
    for (uint j = 0; j < cols; j++) mean += {load_in};
    mean /= float(cols);
    float var = 0.0f;
    for (uint j = 0; j < cols; j++) {{
        float diff = {load_in} - mean;
        var += diff * diff;
    }}
    var /= float(cols);
    float inv_std = 1.0f / sqrt(var + eps);
    for (uint j = 0; j < cols; j++) {{
        output[offset + j] = {store_out};
    }}
}}
"#,
        t = t, s = s,
        load_in = load("input[offset + j]"),
        store_out = store(&format!("{} * ({} - mean) * inv_std + {}", load("gamma[j]"), load("input[offset + j]"), load("beta[j]"))),
    )
}

/// Generate embedding kernel source. Indices always Int32.
pub fn embedding_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void embedding{s}(
    device const {t}* weights [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device {t}* output [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint i = gid.y;
    uint j = gid.x;
    if (i >= seq_len || j >= embed_dim) return;
    int idx = indices[i];
    output[i * embed_dim + j] = weights[idx * embed_dim + j];
}}
"#,
        t = t, s = s,
    )
}

/// Generate copy_strided kernel source (general transpose via stride permutation).
pub fn copy_strided_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void copy_strided{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint* in_strides [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    output[id] = input[in_off];
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
    )
}

/// Generate transpose kernel source (simple 2D transpose).
pub fn transpose_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void transpose{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= cols) return;
    output[col * rows + row] = input[row * cols + col];
}}
"#,
        t = t, s = s,
    )
}

/// Generate transpose_batched kernel source (3D batched transpose).
pub fn transpose_batched_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void transpose_batched{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (row >= rows || col >= cols || batch >= batch_size) return;
    output[batch * cols * rows + col * rows + row] = input[batch * rows * cols + row * cols + col];
}}
"#,
        t = t, s = s,
    )
}

/// Generate conv1d kernel source. Float-only.
pub fn conv1d_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void conv1d{s}(
    device const {t}* input [[buffer(0)]],
    device const {t}* weight [[buffer(1)]],
    device {t}* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& in_channels [[buffer(4)]],
    constant uint& out_channels [[buffer(5)]],
    constant uint& in_length [[buffer(6)]],
    constant uint& out_length [[buffer(7)]],
    constant uint& kernel_size [[buffer(8)]],
    constant uint& stride [[buffer(9)]],
    constant uint& padding [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint o = gid.x;
    uint oc = gid.y;
    uint b = gid.z;
    if (o >= out_length || oc >= out_channels || b >= batch) return;

    float sum = 0.0f;
    for (uint ic = 0; ic < in_channels; ic++) {{
        for (uint k = 0; k < kernel_size; k++) {{
            int in_pos = int(o * stride + k) - int(padding);
            if (in_pos >= 0 && uint(in_pos) < in_length) {{
                sum += {load_i} * {load_w};
            }}
        }}
    }}
    output[b * out_channels * out_length + oc * out_length + o] = {store_sum};
}}
"#,
        t = t, s = s,
        load_i = load("input[b * in_channels * in_length + ic * in_length + in_pos]"),
        load_w = load("weight[oc * in_channels * kernel_size + ic * kernel_size + k]"),
        store_sum = store("sum"),
    )
}

/// Generate conv2d kernel source. Float-only.
pub fn conv2d_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void conv2d{s}(
    device const {t}* input [[buffer(0)]],
    device const {t}* weight [[buffer(1)]],
    device {t}* output [[buffer(2)]],
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
) {{
    uint ow = gid.x;
    uint combined = gid.y;
    uint b = gid.z;
    uint oc = combined % out_channels;
    uint oh = combined / out_channels;
    if (ow >= out_w || oh >= out_h || b >= batch) return;

    float sum = 0.0f;
    for (uint ic = 0; ic < in_channels; ic++) {{
        for (uint i = 0; i < kh; i++) {{
            for (uint j = 0; j < kw; j++) {{
                int ih = int(oh * stride_h + i) - int(pad_h);
                int iw = int(ow * stride_w + j) - int(pad_w);
                if (ih >= 0 && uint(ih) < in_h && iw >= 0 && uint(iw) < in_w) {{
                    sum += {load_i} * {load_w};
                }}
            }}
        }}
    }}
    output[b * out_channels * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = {store_sum};
}}
"#,
        t = t, s = s,
        load_i = load("input[b * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw]"),
        load_w = load("weight[oc * in_channels * kh * kw + ic * kh * kw + i * kw + j]"),
        store_sum = store("sum"),
    )
}

/// Generate batch_norm kernel source. Float-only.
pub fn batch_norm_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void batch_norm{s}(
    device const {t}* input [[buffer(0)]],
    device const {t}* mean [[buffer(1)]],
    device const {t}* var_ [[buffer(2)]],
    device const {t}* weight [[buffer(3)]],
    device const {t}* bias [[buffer(4)]],
    device {t}* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& channels [[buffer(7)]],
    constant uint& spatial [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint s = gid.x;
    uint c = gid.y;
    uint b = gid.z;
    if (s >= spatial || c >= channels || b >= batch) return;

    uint idx = b * channels * spatial + c * spatial + s;
    float inv_std = 1.0f / sqrt({load_var} + eps);
    output[idx] = {store_out};
}}
"#,
        t = t, s = s,
        load_var = load("var_[c]"),
        store_out = store(&format!("({} - {}) * inv_std * {} + {}", load("input[idx]"), load("mean[c]"), load("weight[c]"), load("bias[c]"))),
    )
}

/// Generate max_pool2d kernel source. Float-only.
pub fn max_pool2d_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void max_pool2d{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
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
) {{
    uint ow = gid.x;
    uint combined = gid.y;
    uint b = gid.z;
    uint c = combined % channels;
    uint oh = combined / channels;
    if (ow >= out_w || oh >= out_h || b >= batch) return;

    float max_val = -1e30f;
    for (uint i = 0; i < kh; i++) {{
        for (uint j = 0; j < kw; j++) {{
            int ih = int(oh * stride_h + i) - int(pad_h);
            int iw = int(ow * stride_w + j) - int(pad_w);
            if (ih >= 0 && uint(ih) < in_h && iw >= 0 && uint(iw) < in_w) {{
                float val = {load_v};
                max_val = max(max_val, val);
            }}
        }}
    }}
    output[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = {store_max};
}}
"#,
        t = t, s = s,
        load_v = load("input[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw]"),
        store_max = store("max_val"),
    )
}

/// Generate avg_pool2d kernel source. Float-only.
pub fn avg_pool2d_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void avg_pool2d{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
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
) {{
    uint ow = gid.x;
    uint combined = gid.y;
    uint b = gid.z;
    uint c = combined % channels;
    uint oh = combined / channels;
    if (ow >= out_w || oh >= out_h || b >= batch) return;

    float sum = 0.0f;
    uint count = 0;
    for (uint i = 0; i < kh; i++) {{
        for (uint j = 0; j < kw; j++) {{
            int ih = int(oh * stride_h + i) - int(pad_h);
            int iw = int(ow * stride_w + j) - int(pad_w);
            if (ih >= 0 && uint(ih) < in_h && iw >= 0 && uint(iw) < in_w) {{
                sum += {load_v};
                count++;
            }}
        }}
    }}
    output[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = {store_avg};
}}
"#,
        t = t, s = s,
        load_v = load("input[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw]"),
        store_avg = store("count > 0 ? sum / float(count) : 0.0f"),
    )
}

/// Generate softmax_backward kernel source. Uses float accumulation.
pub fn softmax_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void softmax_backward{s}(
    device const {t}* grad_output [[buffer(0)]],
    device const {t}* output [[buffer(1)]],
    device {t}* grad_input [[buffer(2)]],
    constant uint& total_rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {{
    if (row >= total_rows) return;
    uint offset = row * cols;

    float dot = 0.0f;
    for (uint j = 0; j < cols; j++) {{
        dot += {load_go} * {load_o};
    }}

    for (uint j = 0; j < cols; j++) {{
        grad_input[offset + j] = {store_gi};
    }}
}}
"#,
        t = t, s = s,
        load_go = load("grad_output[offset + j]"),
        load_o = load("output[offset + j]"),
        store_gi = store(&format!("{} * ({} - dot)", load("output[offset + j]"), load("grad_output[offset + j]"))),
    )
}

/// Generate layer_norm_backward kernel source. Uses float accumulation.
pub fn layer_norm_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_backward{s}(
    device const {t}* grad_output [[buffer(0)]],
    device const {t}* input [[buffer(1)]],
    device const {t}* gamma [[buffer(2)]],
    device {t}* grad_input [[buffer(3)]],
    constant uint& total_rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {{
    if (row >= total_rows) return;
    uint offset = row * cols;

    float mean = 0.0f;
    for (uint j = 0; j < cols; j++) mean += {load_in};
    mean /= float(cols);

    float var = 0.0f;
    for (uint j = 0; j < cols; j++) {{
        float diff = {load_in} - mean;
        var += diff * diff;
    }}
    var /= float(cols);
    float inv_std = 1.0f / sqrt(var + eps);

    float sum_dy_gamma = 0.0f;
    float sum_dy_gamma_xhat = 0.0f;
    for (uint j = 0; j < cols; j++) {{
        float xhat = ({load_in} - mean) * inv_std;
        float dy_gamma = {load_go} * {load_g};
        sum_dy_gamma += dy_gamma;
        sum_dy_gamma_xhat += dy_gamma * xhat;
    }}

    float n = float(cols);
    for (uint j = 0; j < cols; j++) {{
        float xhat = ({load_in} - mean) * inv_std;
        float dy_gamma = {load_go} * {load_g};
        grad_input[offset + j] = {store_out};
    }}
}}
"#,
        t = t, s = s,
        load_in = load("input[offset + j]"),
        load_go = load("grad_output[offset + j]"),
        load_g = load("gamma[j]"),
        store_out = store("inv_std * (dy_gamma - sum_dy_gamma / n - xhat * sum_dy_gamma_xhat / n)"),
    )
}

/// Generate conv2d_backward_input kernel source. Float-only.
pub fn conv2d_backward_input_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void conv2d_backward_input{s}(
    device const {t}* grad_output [[buffer(0)]],
    device const {t}* weight [[buffer(1)]],
    device {t}* grad_input [[buffer(2)]],
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
) {{
    uint iw = gid.x;
    uint combined = gid.y;
    uint b = gid.z;
    uint ic = combined % in_channels;
    uint ih = combined / in_channels;
    if (iw >= in_w || ih >= in_h || b >= batch) return;

    float sum = 0.0f;
    for (uint oc = 0; oc < out_channels; oc++) {{
        for (uint i = 0; i < kh; i++) {{
            for (uint j = 0; j < kw; j++) {{
                int oh_candidate = int(ih + pad_h) - int(i);
                int ow_candidate = int(iw + pad_w) - int(j);
                if (oh_candidate >= 0 && oh_candidate % int(stride_h) == 0 &&
                    ow_candidate >= 0 && ow_candidate % int(stride_w) == 0) {{
                    uint oh = uint(oh_candidate) / stride_h;
                    uint ow = uint(ow_candidate) / stride_w;
                    if (oh < out_h && ow < out_w) {{
                        sum += {load_go} * {load_w};
                    }}
                }}
            }}
        }}
    }}
    grad_input[b * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw] = {store_sum};
}}
"#,
        t = t, s = s,
        load_go = load("grad_output[b * out_channels * out_h * out_w + oc * out_h * out_w + oh * out_w + ow]"),
        load_w = load("weight[oc * in_channels * kh * kw + ic * kh * kw + i * kw + j]"),
        store_sum = store("sum"),
    )
}

/// Generate embedding_backward kernel source. Uses atomic float add.
pub fn embedding_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    // Embedding backward always uses float atomic add, cast as needed
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

inline void atomic_add_float(device float* addr, float val) {{
    float expected = *addr;
    while (true) {{
        float desired = expected + val;
        device atomic_uint* addr_uint = (device atomic_uint*)addr;
        uint expected_uint = as_type<uint>(expected);
        uint desired_uint = as_type<uint>(desired);
        if (atomic_compare_exchange_weak_explicit(addr_uint, &expected_uint, desired_uint,
                memory_order_relaxed, memory_order_relaxed)) {{
            break;
        }}
        expected = as_type<float>(expected_uint);
    }}
}}

kernel void embedding_backward{s}(
    device const {t}* grad_output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* grad_weight [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint j = gid.x;
    uint i = gid.y;
    if (i >= seq_len || j >= embed_dim) return;

    int idx = indices[i];
    atomic_add_float(&grad_weight[idx * embed_dim + j], {load});
}}
"#,
        t = t, s = s,
        load = load("grad_output[i * embed_dim + j]"),
    )
}

/// Generate batch_norm_backward kernel source. Float-only.
pub fn batch_norm_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void batch_norm_backward{s}(
    device const {t}* grad_output [[buffer(0)]],
    device const {t}* weight [[buffer(1)]],
    device const {t}* running_var [[buffer(2)]],
    device {t}* grad_input [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& spatial [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint s = gid.x;
    uint c = gid.y;
    uint b = gid.z;
    if (s >= spatial || c >= channels || b >= batch) return;
    uint idx = b * channels * spatial + c * spatial + s;
    float inv_std = 1.0f / sqrt({load_var} + eps);
    grad_input[idx] = {store_out};
}}
"#,
        t = t, s = s,
        load_var = load("running_var[c]"),
        store_out = store(&format!("{} * {} * inv_std", load("grad_output[idx]"), load("weight[c]"))),
    )
}

pub fn threshold_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load_input = if acc { "float(input[id])".to_string() } else { "input[id]".to_string() };
    let load_grad = if acc { "float(grad_output[id])".to_string() } else { "grad_output[id]".to_string() };
    let store = if acc { format!("{}(result)", t) } else { "result".to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void threshold_backward{s}(device const {t}* grad_output [[buffer(0)]], device const {t}* input [[buffer(1)]], device {t}* grad_input [[buffer(2)]], constant uint& numel [[buffer(3)]], constant float& threshold [[buffer(4)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    float inp = {load_input};
    float go = {load_grad};
    float result = (inp > threshold) ? go : 0.0f;
    grad_input[id] = {store};
}}
"#,
        t = t, s = s,
        load_input = load_input,
        load_grad = load_grad,
        store = store,
    )
}

pub fn tanh_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load_out = if acc { "float(output[id])".to_string() } else { "output[id]".to_string() };
    let load_grad = if acc { "float(grad_output[id])".to_string() } else { "grad_output[id]".to_string() };
    let store = if acc { format!("{}(result)", t) } else { "result".to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void tanh_backward{s}(device const {t}* grad_output [[buffer(0)]], device const {t}* output [[buffer(1)]], device {t}* grad_input [[buffer(2)]], constant uint& numel [[buffer(3)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    float out_val = {load_out};
    float go = {load_grad};
    float result = go * (1.0f - out_val * out_val);
    grad_input[id] = {store};
}}
"#,
        t = t, s = s,
        load_out = load_out,
        load_grad = load_grad,
        store = store,
    )
}

pub fn sigmoid_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load_out = if acc { "float(output[id])".to_string() } else { "output[id]".to_string() };
    let load_grad = if acc { "float(grad_output[id])".to_string() } else { "grad_output[id]".to_string() };
    let store = if acc { format!("{}(result)", t) } else { "result".to_string() };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_backward{s}(device const {t}* grad_output [[buffer(0)]], device const {t}* output [[buffer(1)]], device {t}* grad_input [[buffer(2)]], constant uint& numel [[buffer(3)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    float out_val = {load_out};
    float go = {load_grad};
    float result = go * out_val * (1.0f - out_val);
    grad_input[id] = {store};
}}
"#,
        t = t, s = s,
        load_out = load_out,
        load_grad = load_grad,
        store = store,
    )
}

// ── Task 10: Cast op kernel ─────────────────────────────────────────────────

/// Generate cast kernel source (convert from src dtype to dst dtype).
pub fn cast_kernel_source(src: DType, dst: DType) -> String {
    let st = metal_type(src);
    let dt = metal_type(dst);
    let ss = dtype_suffix(src);
    let ds = dtype_suffix(dst);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void cast{ss}_to{ds}(device const {st}* input [[buffer(0)]], device {dt}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = ({dt})input[in_off];
}}
"#,
        nd = ND_INDEX_HELPER, st = st, dt = dt, ss = ss, ds = ds,
    )
}

// ── Quantize/Dequantize ops ─────────────────────────────────────────────────

/// Generate quantize kernel source: float → int8/uint8.
/// Scale and zero_point are baked into the kernel as constants.
pub fn quantize_kernel_source(src: DType, dst: DType, scale: f32, zero_point: i32) -> String {
    let st = metal_type(src);
    let dt = metal_type(dst);
    let ss = dtype_suffix(src);
    let ds = dtype_suffix(dst);
    let (clamp_min, clamp_max) = match dst {
        DType::Int8 => ("-128.0", "127.0"),
        DType::UInt8 => ("0.0", "255.0"),
        _ => panic!("Quantize target must be Int8 or UInt8"),
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void quantize{ss}_to{ds}(device const {st}* input [[buffer(0)]], device {dt}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    float val = round(float(input[in_off]) / {scale}) + float({zero_point});
    out[id] = ({dt})clamp(val, {clamp_min}, {clamp_max});
}}
"#,
        nd = ND_INDEX_HELPER, st = st, dt = dt, ss = ss, ds = ds,
        scale = scale, zero_point = zero_point,
        clamp_min = clamp_min, clamp_max = clamp_max,
    )
}

/// Generate dequantize kernel source: int8/uint8 → float.
/// Scale and zero_point are baked into the kernel as constants.
pub fn dequantize_kernel_source(src: DType, dst: DType, scale: f32, zero_point: i32) -> String {
    let st = metal_type(src);
    let dt = metal_type(dst);
    let ss = dtype_suffix(src);
    let ds = dtype_suffix(dst);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void dequantize{ss}_to{ds}(device const {st}* input [[buffer(0)]], device {dt}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = ({dt})(float(input[in_off] - {zero_point}) * {scale});
}}
"#,
        nd = ND_INDEX_HELPER, st = st, dt = dt, ss = ss, ds = ds,
        scale = scale, zero_point = zero_point,
    )
}

// ── Task 11: Byte-copy shape ops ────────────────────────────────────────────

/// Generate byte-copy transpose kernel parameterized by element size.
pub fn byte_copy_transpose_source(elem_size: usize) -> String {
    let t = match elem_size {
        1 => "char", 2 => "short", 4 => "int", 8 => "long",
        _ => "char",
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void transpose_bytes{es}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= cols) return;
    output[col * rows + row] = input[row * cols + col];
}}
"#,
        t = t, es = elem_size,
    )
}

/// Generate byte-copy slice kernel (dim0) parameterized by element size.
pub fn byte_copy_slice_dim0_source(elem_size: usize) -> String {
    let t = match elem_size {
        1 => "char", 2 => "short", 4 => "int", 8 => "long",
        _ => "char",
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void slice_dim0_bytes{es}(device const {t}* input [[buffer(0)]], device {t}* output [[buffer(1)]], constant uint& cols [[buffer(2)]], constant uint& start_row [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {{
    uint row = gid.y; uint col = gid.x;
    if (col >= cols) return;
    output[row * cols + col] = input[(start_row + row) * cols + col];
}}
"#,
        t = t, es = elem_size,
    )
}

/// Generate byte-copy slice kernel (dim1) parameterized by element size.
pub fn byte_copy_slice_dim1_source(elem_size: usize) -> String {
    let t = match elem_size {
        1 => "char", 2 => "short", 4 => "int", 8 => "long",
        _ => "char",
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void slice_dim1_bytes{es}(device const {t}* input [[buffer(0)]], device {t}* output [[buffer(1)]], constant uint& in_cols [[buffer(2)]], constant uint& out_cols [[buffer(3)]], constant uint& start_col [[buffer(4)]], constant uint& rows [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {{
    uint row = gid.y; uint col = gid.x;
    if (row >= rows || col >= out_cols) return;
    output[row * out_cols + col] = input[row * in_cols + (start_col + col)];
}}
"#,
        t = t, es = elem_size,
    )
}

/// Generate byte-copy concat kernel (dim0) parameterized by element size.
pub fn byte_copy_concat_dim0_source(elem_size: usize) -> String {
    let t = match elem_size {
        1 => "char", 2 => "short", 4 => "int", 8 => "long",
        _ => "char",
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void concat_dim0_bytes{es}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* output [[buffer(2)]], constant uint& rows_a [[buffer(3)]], constant uint& cols [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {{
    uint row = gid.y; uint col = gid.x;
    if (col >= cols) return;
    if (row < rows_a) output[row * cols + col] = a[row * cols + col];
    else output[row * cols + col] = b[(row - rows_a) * cols + col];
}}
"#,
        t = t, es = elem_size,
    )
}

/// Generate byte-copy concat kernel (dim1) parameterized by element size.
pub fn byte_copy_concat_dim1_source(elem_size: usize) -> String {
    let t = match elem_size {
        1 => "char", 2 => "short", 4 => "int", 8 => "long",
        _ => "char",
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void concat_dim1_bytes{es}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* output [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& cols_a [[buffer(4)]], constant uint& cols_b [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {{
    uint row = gid.y; uint col = gid.x;
    uint total_cols = cols_a + cols_b;
    if (row >= rows || col >= total_cols) return;
    if (col < cols_a) output[row * total_cols + col] = a[row * cols_a + col];
    else output[row * total_cols + col] = b[row * cols_b + (col - cols_a)];
}}
"#,
        t = t, es = elem_size,
    )
}

// ── Bitwise ops ─────────────────────────────────────────────────────────────

/// Generate bitwise binary kernel source (and, or, xor) for a given dtype.
/// Works on integer types and Bool.
pub fn bitwise_binary_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void bitwise_and{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] & b[b_off];
}}
kernel void bitwise_or{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] | b[b_off];
}}
kernel void bitwise_xor{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] ^ b[b_off];
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
    )
}

/// Generate bitwise NOT kernel source (unary).
pub fn bitwise_not_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void bitwise_not{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = ~input[in_off];
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
    )
}

/// Generate shift kernel source (shl, shr). Shift amount is passed as float
/// (for FFI compatibility with the pow dispatch path) and cast to uint in the kernel.
pub fn shift_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void shl{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], constant float& shift_f [[buffer(6)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    uint shift = uint(shift_f);
    out[id] = input[in_off] << shift;
}}
kernel void shr{s}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], constant float& shift_f [[buffer(6)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    uint shift = uint(shift_f);
    out[id] = input[in_off] >> shift;
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
    )
}

// ── Modulo op ───────────────────────────────────────────────────────────────

/// Generate modulo kernel source (binary, integer-only).
pub fn mod_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void mod{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] % b[b_off];
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
    )
}

// ── Element-wise min/max ────────────────────────────────────────────────────

/// Generate element-wise min/max kernel source (binary).
pub fn elem_minmax_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void elem_min{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = min(a[a_off], b[b_off]);
}}
kernel void elem_max{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device {t}* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = max(a[a_off], b[b_off]);
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
    )
}

// ── Logical NOT ─────────────────────────────────────────────────────────────

/// Generate logical NOT kernel source (Bool only, uchar -> uchar).
pub fn logical_not_kernel_source() -> String {
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void logical_not_bool(device const uchar* input [[buffer(0)]], device uchar* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = input[in_off] ? 0 : 1;
}}
"#,
        nd = ND_INDEX_HELPER,
    )
}

/// Generate comparison kernel source (lt, gt, le, ge, eq, ne) for a given dtype.
/// Input buffers use the input dtype; output buffer is always `uchar` (Bool).
pub fn comparison_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void lt{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = (a[a_off] < b[b_off]) ? 1 : 0;
}}
kernel void gt{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = (a[a_off] > b[b_off]) ? 1 : 0;
}}
kernel void le{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = (a[a_off] <= b[b_off]) ? 1 : 0;
}}
kernel void ge{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = (a[a_off] >= b[b_off]) ? 1 : 0;
}}
kernel void eq{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = (a[a_off] == b[b_off]) ? 1 : 0;
}}
kernel void ne{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = (a[a_off] != b[b_off]) ? 1 : 0;
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_kernel_has_bool_output() {
        let src = comparison_kernel_source(DType::Float32);
        // Input buffers use the input dtype
        assert!(src.contains("device const float*"));
        // Output buffer is always uchar (Bool)
        assert!(src.contains("device uchar* out"));
        // Contains all 6 comparison functions
        assert!(src.contains("lt_f32"));
        assert!(src.contains("gt_f32"));
        assert!(src.contains("le_f32"));
        assert!(src.contains("ge_f32"));
        assert!(src.contains("eq_f32"));
        assert!(src.contains("ne_f32"));
    }

    #[test]
    fn comparison_kernel_typed() {
        let src = comparison_kernel_source(DType::Int32);
        assert!(src.contains("device const int*"));
        assert!(src.contains("device uchar* out"));
        assert!(src.contains("lt_i32"));
        assert!(src.contains("ne_i32"));
    }

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

    #[test]
    fn float_unary_kernel_has_sincos() {
        let src = float_unary_kernel_source(DType::Float32);
        assert!(src.contains("elementwise_sin_f32"));
        assert!(src.contains("elementwise_cos_f32"));
    }

    #[test]
    fn log_softmax_kernel_typed() {
        let src = log_softmax_kernel_source(DType::Float32);
        assert!(src.contains("log_softmax_f32"));
        assert!(src.contains("log(log_sum_exp)"));
        // f16 version uses float accumulation
        let src_f16 = log_softmax_kernel_source(DType::Float16);
        assert!(src_f16.contains("log_softmax_f16"));
        assert!(src_f16.contains("float(input[offset])"));
    }

    // Task 8a tests
    #[test]
    fn scalar_mul_kernel_typed() {
        let src = scalar_mul_kernel_source(DType::Int32);
        assert!(src.contains("device const int*"));
        assert!(src.contains("scalar_mul_i32"));
    }

    #[test]
    fn scalar_mul_kernel_f16_has_float_acc() {
        let src = scalar_mul_kernel_source(DType::Float16);
        assert!(src.contains("scalar_mul_f16"));
        assert!(src.contains("float(input[id])"));
    }

    #[test]
    fn pow_kernel_typed() {
        let src = pow_kernel_source(DType::BFloat16);
        assert!(src.contains("pow_bf16"));
        assert!(src.contains("device const bfloat*"));
        assert!(src.contains("pow(float("));
    }

    #[test]
    fn clamp_kernel_typed() {
        let src = clamp_kernel_source(DType::Float32);
        assert!(src.contains("clamp_f32"));
        assert!(src.contains("device const float*"));
    }

    #[test]
    fn gelu_kernel_typed() {
        let src = gelu_kernel_source(DType::BFloat16);
        assert!(src.contains("gelu_bf16"));
        assert!(src.contains("device const bfloat*"));
    }

    // Task 8b tests
    #[test]
    fn softmax_kernel_typed() {
        let src = softmax_kernel_source(DType::Float16);
        assert!(src.contains("softmax_f16"));
        assert!(src.contains("device const half*"));
    }

    #[test]
    fn argmax_kernel_always_int_output() {
        let src = argmax_kernel_source(DType::Float32);
        assert!(src.contains("argmax_f32"));
        assert!(src.contains("device int* output"));
    }

    #[test]
    fn sum_kernel_typed() {
        let src = sum_kernel_source(DType::BFloat16);
        assert!(src.contains("sum_bf16"));
        assert!(src.contains("float(input["));
    }

    // Task 8c tests
    #[test]
    fn where_kernel_typed() {
        let src = where_kernel_source(DType::Float32);
        assert!(src.contains("where_f32"));
    }

    #[test]
    fn triu_kernel_typed() {
        let src = triu_kernel_source(DType::Int32);
        assert!(src.contains("triu_i32"));
        assert!(src.contains("device const int*"));
    }

    #[test]
    fn gather_kernel_dim0() {
        let src = gather_kernel_source(DType::Float16, 0);
        assert!(src.contains("gather_dim0_f16"));
    }

    #[test]
    fn index_select_kernel_dim1() {
        let src = index_select_kernel_source(DType::Float32, 1);
        assert!(src.contains("index_select_dim1_f32"));
    }

    // Task 8d tests
    #[test]
    fn matmul_kernel_typed() {
        let src = matmul_kernel_source(DType::Float16);
        assert!(src.contains("matmul_f16"));
        assert!(src.contains("float(A["));
    }

    #[test]
    fn layer_norm_kernel_typed() {
        let src = layer_norm_kernel_source(DType::BFloat16);
        assert!(src.contains("layer_norm_bf16"));
    }

    #[test]
    fn embedding_kernel_typed() {
        let src = embedding_kernel_source(DType::Float16);
        assert!(src.contains("embedding_f16"));
        assert!(src.contains("device const int* indices"));
    }

    #[test]
    fn conv2d_kernel_typed() {
        let src = conv2d_kernel_source(DType::Float32);
        assert!(src.contains("conv2d_f32"));
    }

    #[test]
    fn softmax_backward_kernel_typed() {
        let src = softmax_backward_kernel_source(DType::Float16);
        assert!(src.contains("softmax_backward_f16"));
    }

    // Task 10 tests
    #[test]
    fn cast_kernel_f32_to_i32() {
        let src = cast_kernel_source(DType::Float32, DType::Int32);
        assert!(src.contains("cast_f32_to_i32"));
        assert!(src.contains("device const float*"));
        assert!(src.contains("device int* out"));
    }

    // Task 11 tests
    #[test]
    fn byte_copy_transpose_4byte() {
        let src = byte_copy_transpose_source(4);
        assert!(src.contains("transpose_bytes4"));
        assert!(src.contains("device const int*"));
    }

    #[test]
    fn byte_copy_slice_dim0_2byte() {
        let src = byte_copy_slice_dim0_source(2);
        assert!(src.contains("slice_dim0_bytes2"));
        assert!(src.contains("device const short*"));
    }

    #[test]
    fn byte_copy_concat_dim1_8byte() {
        let src = byte_copy_concat_dim1_source(8);
        assert!(src.contains("concat_dim1_bytes8"));
        assert!(src.contains("device const long*"));
    }

    // ── Bitwise ops tests ───────────────────────────────────────────

    #[test]
    fn bitwise_binary_kernel_typed() {
        let src = bitwise_binary_kernel_source(DType::Int32);
        assert!(src.contains("bitwise_and_i32"));
        assert!(src.contains("bitwise_or_i32"));
        assert!(src.contains("bitwise_xor_i32"));
        assert!(src.contains("device const int*"));
    }

    #[test]
    fn bitwise_binary_kernel_uint8() {
        let src = bitwise_binary_kernel_source(DType::UInt8);
        assert!(src.contains("bitwise_and_u8"));
        assert!(src.contains("device const uchar*"));
    }

    #[test]
    fn bitwise_not_kernel_typed() {
        let src = bitwise_not_kernel_source(DType::UInt32);
        assert!(src.contains("bitwise_not_u32"));
        assert!(src.contains("device const uint*"));
        assert!(src.contains("~input[in_off]"));
    }

    #[test]
    fn shift_kernel_typed() {
        let src = shift_kernel_source(DType::Int32);
        assert!(src.contains("shl_i32"));
        assert!(src.contains("shr_i32"));
        assert!(src.contains("device const int*"));
        assert!(src.contains("<< shift"));
        assert!(src.contains(">> shift"));
    }

    #[test]
    fn mod_kernel_typed() {
        let src = mod_kernel_source(DType::Int32);
        assert!(src.contains("mod_i32"));
        assert!(src.contains("device const int*"));
        assert!(src.contains("a[a_off] % b[b_off]"));
    }

    #[test]
    fn elem_minmax_kernel_typed() {
        let src = elem_minmax_kernel_source(DType::Float32);
        assert!(src.contains("elem_min_f32"));
        assert!(src.contains("elem_max_f32"));
        assert!(src.contains("device const float*"));
        assert!(src.contains("min(a[a_off], b[b_off])"));
        assert!(src.contains("max(a[a_off], b[b_off])"));
    }

    #[test]
    fn elem_minmax_kernel_int() {
        let src = elem_minmax_kernel_source(DType::Int32);
        assert!(src.contains("elem_min_i32"));
        assert!(src.contains("elem_max_i32"));
    }

    #[test]
    fn logical_not_kernel() {
        let src = logical_not_kernel_source();
        assert!(src.contains("logical_not_bool"));
        assert!(src.contains("device const uchar*"));
        assert!(src.contains("device uchar* out"));
        assert!(src.contains("input[in_off] ? 0 : 1"));
    }

    // ── Quantize/Dequantize tests ───────────────────────────────────

    #[test]
    fn quantize_f32_to_i8() {
        let src = quantize_kernel_source(DType::Float32, DType::Int8, 0.1, 0);
        assert!(src.contains("quantize_f32_to_i8"));
        assert!(src.contains("device const float*"));
        assert!(src.contains("device char* out"));
        assert!(src.contains("clamp(val, -128.0, 127.0)"));
        assert!(src.contains("/ 0.1)"));
    }

    #[test]
    fn quantize_f32_to_u8() {
        let src = quantize_kernel_source(DType::Float32, DType::UInt8, 0.05, 128);
        assert!(src.contains("quantize_f32_to_u8"));
        assert!(src.contains("device const float*"));
        assert!(src.contains("device uchar* out"));
        assert!(src.contains("clamp(val, 0.0, 255.0)"));
        assert!(src.contains("+ float(128)"));
    }

    #[test]
    fn quantize_f16_to_i8() {
        let src = quantize_kernel_source(DType::Float16, DType::Int8, 0.02, -5);
        assert!(src.contains("quantize_f16_to_i8"));
        assert!(src.contains("device const half*"));
    }

    #[test]
    fn dequantize_i8_to_f32() {
        let src = dequantize_kernel_source(DType::Int8, DType::Float32, 0.1, 0);
        assert!(src.contains("dequantize_i8_to_f32"));
        assert!(src.contains("device const char*"));
        assert!(src.contains("device float* out"));
        assert!(src.contains("* 0.1)"));
    }

    #[test]
    fn dequantize_u8_to_f32() {
        let src = dequantize_kernel_source(DType::UInt8, DType::Float32, 0.05, 128);
        assert!(src.contains("dequantize_u8_to_f32"));
        assert!(src.contains("device const uchar*"));
        assert!(src.contains("device float* out"));
        assert!(src.contains("- 128)"));
    }

    #[test]
    fn dequantize_u8_to_f16() {
        let src = dequantize_kernel_source(DType::UInt8, DType::Float16, 0.03, 10);
        assert!(src.contains("dequantize_u8_to_f16"));
        assert!(src.contains("device const uchar*"));
        assert!(src.contains("device half* out"));
    }

    #[test]
    fn add_bias_nd_kernel_structure() {
        let src = add_bias_kernel_source(DType::Float32);
        assert!(src.contains("add_bias_f32"));
        assert!(src.contains("numel"));
        assert!(src.contains("num_channels"));
        assert!(src.contains("channel_stride"));
        assert!(src.contains("(id / channel_stride) % num_channels"));
        assert!(src.contains("uint id [[thread_position_in_grid]]"));
    }
}
