# Multi-Dtype Kernel Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the kernel infrastructure from hand-duplicated MSL strings to template-based generation, enabling multi-dtype compute kernels without combinatorial explosion.

**Architecture:** Replace ~70 `const` MSL string constants with runtime-generated kernel sources via `fn kernel_source(metal_type, suffix) -> String`. Add `UnsupportedDtype` error, `ScalarValue` enum, BFloat16 bug fix, and template generators for binary/unary ops. This is Plan 1 of 3 — infrastructure foundation only. Plan 2 will template remaining ops, add Cast, and wire in new dtypes.

**Scope reduced from review feedback:** Tasks 8 (template all remaining ops), 9 (is_compute_supported refactor), 10 (Cast op), 11 (byte-copy shape ops) moved to Plan 2 to keep this plan small and shippable.

**Tech Stack:** Rust (applegpu-core), MSL (Metal Shading Language)

**Spec:** `docs/superpowers/specs/2026-03-16-multi-dtype-compute-kernels-design.md`

---

## Chunk 1: Error Variant + BFloat16 Bug Fix + ScalarValue

### Task 1: Add UnsupportedDtype error variant

**Files:**
- Modify: `crates/core/src/error.rs`

- [ ] **Step 1: Write test**

Add to `crates/core/src/error.rs` test module:
```rust
#[test]
fn unsupported_dtype_display() {
    let e = GpuError::UnsupportedDtype("Float64 not supported in Metal".to_string());
    assert!(e.to_string().contains("Float64"));
    assert!(e.to_string().contains("not supported"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core -- error::tests::unsupported_dtype_display --test-threads=1`
Expected: FAIL — variant does not exist

- [ ] **Step 3: Implement**

Add to the `GpuError` enum in `crates/core/src/error.rs`:
```rust
/// Operation not supported for the given dtype
UnsupportedDtype(String),
```

Add to the Display impl:
```rust
GpuError::UnsupportedDtype(msg) => write!(f, "Unsupported dtype: {}", msg),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p applegpu-core -- error::tests::unsupported_dtype_display --test-threads=1`

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/error.rs
git commit -m "feat: add UnsupportedDtype error variant

Needed for multi-dtype compute kernel gating.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 2: Fix BFloat16 from_name roundtrip bug

**Files:**
- Modify: `crates/core/src/tensor.rs`

- [ ] **Step 1: Write test**

Add to tensor.rs test module (or find existing `test_dtype_name_roundtrip`):
```rust
#[test]
fn bfloat16_name_roundtrip() {
    assert_eq!(DType::from_name("bfloat16"), Some(DType::BFloat16));
    assert_eq!(DType::from_name("bf16"), Some(DType::BFloat16));
    assert_eq!(DType::BFloat16.name(), "bfloat16");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core -- bfloat16_name_roundtrip --test-threads=1`
Expected: FAIL — `from_name("bfloat16")` returns None

- [ ] **Step 3: Fix from_name**

In `crates/core/src/tensor.rs` line 47, before the `_ => None` arm, add:
```rust
"bfloat16" | "bf16" => Some(DType::BFloat16),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p applegpu-core -- bfloat16_name_roundtrip --test-threads=1`

- [ ] **Step 5: Run all tests**

Run: `cargo test -p applegpu-core -- --test-threads=1`

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/tensor.rs
git commit -m "fix: BFloat16 from_name roundtrip bug

Was missing 'bfloat16' | 'bf16' match arm in DType::from_name().
name() returned 'bfloat16' but from_name('bfloat16') returned None.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 3: Add ScalarValue enum to graph.rs

**Files:**
- Modify: `crates/core/src/graph.rs`

- [ ] **Step 1: Write test**

Add to graph.rs (create test module if needed):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_value_to_f64() {
        assert_eq!(ScalarValue::Float(3.14).as_f64(), 3.14);
        assert_eq!(ScalarValue::Int(42).as_f64(), 42.0);
        assert_eq!(ScalarValue::UInt(255).as_f64(), 255.0);
        assert_eq!(ScalarValue::Bool(true).as_f64(), 1.0);
    }

    #[test]
    fn scalar_value_to_msl_literal() {
        assert_eq!(ScalarValue::Float(1.5).to_msl_literal(), "1.5");
        assert_eq!(ScalarValue::Int(-3).to_msl_literal(), "-3");
        assert_eq!(ScalarValue::UInt(0).to_msl_literal(), "0u");
        assert_eq!(ScalarValue::Bool(true).to_msl_literal(), "true");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core -- graph::tests::scalar_value --test-threads=1`

- [ ] **Step 3: Implement ScalarValue**

Add to `crates/core/src/graph.rs` before the OpKind enum:
```rust
/// A scalar value that can represent any dtype's scalar.
/// Used by ops that carry scalar parameters (Pow, ScalarMul, Clamp, MaskedFill).
#[derive(Debug, Clone, Copy)]
pub enum ScalarValue {
    Float(f64),  // covers f32, f16, bf16
    Int(i64),    // covers i32, i64, i16, i8
    UInt(u64),   // covers u32, u8
    Bool(bool),
}

impl ScalarValue {
    pub fn as_f64(&self) -> f64 {
        match self {
            ScalarValue::Float(v) => *v,
            ScalarValue::Int(v) => *v as f64,
            ScalarValue::UInt(v) => *v as f64,
            ScalarValue::Bool(v) => if *v { 1.0 } else { 0.0 },
        }
    }

    /// Render as an MSL literal for code generation.
    pub fn to_msl_literal(&self) -> String {
        match self {
            ScalarValue::Float(v) => format!("{}", v),
            ScalarValue::Int(v) => format!("{}", v),
            ScalarValue::UInt(v) => format!("{}u", v),
            ScalarValue::Bool(v) => if *v { "true".to_string() } else { "false".to_string() },
        }
    }

    /// Create from f32 (backward compat with existing OpKind f32 fields).
    pub fn from_f32(v: f32) -> Self {
        ScalarValue::Float(v as f64)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p applegpu-core -- graph::tests::scalar_value --test-threads=1`

- [ ] **Step 5: Run tests**

Run: `cargo test -p applegpu-core -- graph::tests --test-threads=1`

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/graph.rs
git commit -m "feat: add ScalarValue enum for dtype-aware scalar parameters

Prepares for migrating OpKind's f32 fields to ScalarValue in Task 3b.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 3b: Migrate OpKind scalar fields to ScalarValue

**Files:**
- Modify: `crates/core/src/graph.rs`
- Modify: `crates/core/src/ops.rs`
- Modify: `crates/core/src/compute.rs`

Only ScalarMul, Pow, Clamp, MaskedFill change — eps is always a float precision parameter.

- [ ] **Step 1: Change OpKind variants in graph.rs**

Change these 4 variants only:
```rust
ScalarMul(ScalarValue),            // was ScalarMul(f32)
Pow { exponent: ScalarValue },     // was Pow { exponent: f32 }
Clamp { min_val: ScalarValue, max_val: ScalarValue },  // was f32, f32
MaskedFill { value: ScalarValue }, // was MaskedFill { value: f32 }
```

- [ ] **Step 2: Fix ops.rs callsites**

Run `grep -rn "ScalarMul\|Pow {\|Clamp {\|MaskedFill {" crates/core/src/ops.rs` to find all construction sites. Wrap each `f32` with `ScalarValue::from_f32(value)`.

- [ ] **Step 3: Fix compute.rs dispatch**

Run `grep -rn "ScalarMul\|Pow {\|Clamp {\|MaskedFill {" crates/core/src/compute.rs` to find all destructuring sites. Extract the f32 value with `.as_f64() as f32`.

- [ ] **Step 4: Fix any remaining compile errors**

Run: `cargo build -p applegpu-core 2>&1 | head -30` — fix any remaining callsites in fusion.rs, wire serialization, or Python bindings.

- [ ] **Step 5: Run all tests**

Run: `cargo test -p applegpu-core -- --test-threads=1`
Expected: ALL tests pass (zero behavior change)

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/graph.rs crates/core/src/ops.rs crates/core/src/compute.rs
git commit -m "refactor: migrate ScalarMul/Pow/Clamp/MaskedFill to ScalarValue

All scalar OpKind fields now use ScalarValue instead of raw f32.
Enables dtype-aware scalar parameters for integer types.
Zero behavior change — all existing tests pass.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 2: Template-Based Kernel Generation

### Task 4: Create kernel template module

**Files:**
- Create: `crates/core/src/kernel_templates.rs`
- Modify: `crates/core/src/lib.rs` (add `pub mod kernel_templates;`)

This is the core of the refactor. Instead of duplicated `const` MSL strings per dtype, we generate MSL at init time from templates.

- [ ] **Step 1: Write test**

Create `crates/core/src/kernel_templates.rs`:
```rust
//! Template-based MSL kernel source generation.
//! Instead of duplicating kernel strings per dtype, generate them from templates.

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
        DType::Float64 => panic!("Float64 is not supported in MSL — use UnsupportedDtype error before reaching here"),
    }
}

/// Suffix for kernel function names (e.g., "_f32", "_i32", "_bf16").
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
        DType::Float64 => "_f32", // fallback
    }
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
}
```

- [ ] **Step 2: Add `is_float()` to DType**

In `crates/core/src/tensor.rs`, add to the `impl DType` block:
```rust
pub fn is_float(&self) -> bool {
    matches!(self, DType::Float32 | DType::Float16 | DType::BFloat16 | DType::Float64)
}
```

- [ ] **Step 3: Add module to lib.rs**

In `crates/core/src/lib.rs`, add:
```rust
pub mod kernel_templates;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core -- kernel_templates --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/kernel_templates.rs crates/core/src/lib.rs crates/core/src/tensor.rs
git commit -m "feat: add kernel_templates module with metal_type, dtype_suffix, is_float

Foundation for template-based MSL kernel generation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 5: Add binary kernel template generator

**Files:**
- Modify: `crates/core/src/kernel_templates.rs`

- [ ] **Step 1: Write test**

Add to `kernel_templates.rs` tests:
```rust
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p applegpu-core -- kernel_templates::tests::binary_kernel --test-threads=1`

- [ ] **Step 3: Implement binary_kernel_source**

Add to `kernel_templates.rs`:
```rust
/// N-D index helper, shared by all element-wise kernels.
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
{ND_INDEX_HELPER}
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
        ND_INDEX_HELPER = ND_INDEX_HELPER,
        t = t,
        s = s,
    )
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core -- kernel_templates --test-threads=1`

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/kernel_templates.rs
git commit -m "feat: add binary_kernel_source template generator

Generates MSL for add/sub/mul/div with any Metal type.
Replaces hand-duplicated BINARY_KERNEL_SOURCE / BINARY_KERNEL_SOURCE_F16.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 6: Add unary kernel template generator

**Files:**
- Modify: `crates/core/src/kernel_templates.rs`

- [ ] **Step 1: Write test**

```rust
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
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

Add to `kernel_templates.rs`:

```rust
/// Generate unary kernel source for ops that work on ALL types (neg, abs, sign).
pub fn unary_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{ND_INDEX_HELPER}
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
        ND_INDEX_HELPER = ND_INDEX_HELPER, t = t, s = s,
    )
}

/// Generate unary kernel source for FLOAT-ONLY ops (exp, log, sqrt, relu, tanh, gelu).
/// Only valid for Float32, Float16, BFloat16.
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
{ND_INDEX_HELPER}
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
        ND_INDEX_HELPER = ND_INDEX_HELPER, t = t, s = s, zero = zero,
    )
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core -- kernel_templates --test-threads=1`

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/kernel_templates.rs
git commit -m "feat: add unary and float_unary kernel template generators

unary: neg, abs, sign — works for all types.
float_unary: exp, log, sqrt, relu, tanh — float types only.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 7: Wire templates into resolve_kernel for F32+F16+BF16

**Files:**
- Modify: `crates/core/src/compute.rs`

This is the critical integration step — replace the hardcoded `resolve_kernel` match arms with template-generated sources. Start with binary and unary ops only. Keep all existing `const` strings for ops not yet templated (matmul, softmax, etc.).

- [ ] **Step 1: Modify resolve_kernel to use templates for binary/unary ops**

In `crates/core/src/compute.rs`, change the `resolve_kernel` function to call `kernel_templates::binary_kernel_source(dtype)` and `kernel_templates::unary_kernel_source(dtype)` instead of matching to `BINARY_KERNEL_SOURCE` / `BINARY_KERNEL_SOURCE_F16` / `UNARY_KERNEL_SOURCE` / `UNARY_KERNEL_SOURCE_F16`.

The key change: instead of returning `(&'static str, String)`, `resolve_kernel` should return `(String, String)` — the source is now owned, not a static reference. Alternatively, cache the generated sources in a `HashMap<DType, String>` on the KernelRegistry.

Recommended approach — change the signature:
```rust
fn resolve_kernel(base_name: &str, dtype: DType) -> (String, String) {
    let suffix = kernel_templates::dtype_suffix(dtype);
    let func_name = if base_name.ends_with("_f32") {
        format!("{}{}", &base_name[..base_name.len() - 4], suffix)
    } else {
        format!("{}{}", base_name, suffix)
    };

    let source = match base_name {
        n if n.starts_with("elementwise_add") || n.starts_with("elementwise_sub")
            || n.starts_with("elementwise_mul") || n.starts_with("elementwise_div") =>
            kernel_templates::binary_kernel_source(dtype),
        n if n.starts_with("elementwise_") =>
            if dtype.is_float() {
                kernel_templates::float_unary_kernel_source(dtype)
            } else {
                kernel_templates::unary_kernel_source(dtype)
            },
        // Keep existing const sources for non-templated ops
        "matmul_f32" => match dtype {
            DType::Float16 => MATMUL_KERNEL_SOURCE_F16.to_string(),
            _ => MATMUL_KERNEL_SOURCE.to_string(),
        },
        // ... (keep existing match arms for matmul, softmax, etc. — convert &str to String)
        _ => BINARY_KERNEL_SOURCE.to_string(), // fallback
    };

    (source, func_name)
}
```

`DType::is_float()` was already added in Task 4.

- [ ] **Step 2: Update get_or_create to accept &str from String**

The `get_or_create` signature at compute.rs:3185 takes `kernel_source: &str`. This still works — just pass `&source` where `source: String` is returned from `resolve_kernel`. No signature change needed.

Update `resolve_kernel` return type at compute.rs:3201 from `(&'static str, String)` to `(String, String)`:
```rust
fn resolve_kernel(base_name: &str, dtype: DType) -> (String, String) {
```

Then update every call to `resolve_kernel` in compute.rs to use `&source` instead of `source` when passing to `get_or_create`. The callers currently destructure as `let (source, func_name) = Self::resolve_kernel(...)` — just change `source` to `&source` at the `get_or_create` call.

- [ ] **Step 3: Run all tests**

Run: `cargo test -p applegpu-core -- --test-threads=1`
Expected: ALL existing tests pass — the generated templates produce identical MSL to the old const strings.

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/compute.rs crates/core/src/tensor.rs
git commit -m "refactor: wire template generators into resolve_kernel

Binary and unary ops now use kernel_templates::binary_kernel_source()
and kernel_templates::unary_kernel_source() instead of hardcoded consts.
Other ops (matmul, softmax, etc.) still use const strings for now.
Zero behavior change — all existing tests pass.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Deferred to Plan 2

Tasks 8-11 (template all remaining ops, is_compute_supported refactor, Cast op, byte-copy shape ops) are deferred to Plan 2 to keep this plan small and shippable. Plan 1 establishes the template pattern and proves it works for binary/unary ops. Plan 2 extends it to all ops and adds new dtype kernels.

### ~~Task 8: Template remaining ops (matmul, softmax, transpose, etc.)~~

**Files:**
- Modify: `crates/core/src/kernel_templates.rs`
- Modify: `crates/core/src/compute.rs`

This task adds template generators for all remaining ops that currently have F16 variants, and wires them into resolve_kernel. The pattern is the same as Tasks 5-7 but covers: matmul, softmax, softmax_causal, transpose, scalar_mul, gelu, layer_norm, embedding, slice, concat, add_bias, argmax, sum, mean, pow, clamp, where, masked_fill, triu, tril, gather, index_select, copy_strided.

This is a large but mechanical task — each op follows the same pattern:
1. Read the existing `const` F32 source
2. Replace `float` with `{t}` and `_f32` with `{s}`
3. Add as a `pub fn <op>_kernel_source(dtype: DType) -> String` function
4. Wire into `resolve_kernel`
5. Remove the old `const` F32 and F16 variants

- [ ] **Step 1: Add template generators for all remaining ops**

Follow the pattern from `binary_kernel_source` for each op. Group related ops:
- `matmul_kernel_source`
- `softmax_kernel_source`, `softmax_causal_kernel_source`
- `transpose_kernel_source`, `transpose_batched_kernel_source`
- `scalar_mul_kernel_source`
- `gelu_kernel_source`
- `layer_norm_kernel_source`
- `embedding_kernel_source`
- `slice_kernel_source` (dim0, dim1)
- `concat_kernel_source` (dim0, dim1)
- `reduction_kernel_source` (sum, mean, argmax)
- `conditional_kernel_source` (where, masked_fill, triu, tril)
- `gather_kernel_source`, `index_select_kernel_source`
- `pow_kernel_source`, `clamp_kernel_source`

For ops with dtype-specific logic (e.g., softmax needs float accumulation), the template should only accept float dtypes and panic/error on integer types.

- [ ] **Step 2: Wire all templates into resolve_kernel**

Replace the entire `resolve_kernel` function body with template calls for all ops.

- [ ] **Step 3: Remove old const kernel sources**

Delete all `const BINARY_KERNEL_SOURCE: &str`, `const BINARY_KERNEL_SOURCE_F16: &str`, etc. from compute.rs. This should remove ~2500+ lines of duplicated MSL strings.

- [ ] **Step 4: Run all tests**

Run: `cargo test -p applegpu-core -- --test-threads=1`
Expected: ALL tests pass

Run: `uv run maturin develop && uv run pytest -v`
Expected: ALL Python tests pass (306)

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/kernel_templates.rs crates/core/src/compute.rs
git commit -m "refactor: template all kernel sources, remove ~2500 lines of duplication

All MSL kernels now generated via kernel_templates module.
compute.rs reduced from ~4400 to ~1900 lines.
Zero behavior change — all Rust + Python tests pass.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 9: Refactor is_compute_supported to per-op capability query

**Files:**
- Modify: `crates/core/src/tensor.rs`
- Modify: `crates/core/src/compute.rs`
- Modify: `crates/core/src/ops.rs`

- [ ] **Step 1: Write test**

Add to tensor.rs tests:
```rust
#[test]
fn is_float_dtype() {
    assert!(DType::Float32.is_float());
    assert!(DType::Float16.is_float());
    assert!(DType::BFloat16.is_float());
    assert!(!DType::Int32.is_float());
    assert!(!DType::Bool.is_float());
}
```

Add to compute.rs or a new test file:
```rust
#[test]
fn compute_supported_for_new_dtypes() {
    // BFloat16 should support the same ops as Float16
    assert!(DType::BFloat16.is_compute_supported());
    // Int32 should now be compute-supported
    assert!(DType::Int32.is_compute_supported());
}
```

- [ ] **Step 2: Update is_compute_supported**

In `crates/core/src/tensor.rs`, change `is_compute_supported`:
```rust
pub fn is_compute_supported(&self) -> bool {
    // Float64 is NOT supported in MSL. All other types have at least some kernel support.
    !matches!(self, DType::Float64)
}
```

- [ ] **Step 3: Update validate_compute_dtype in ops.rs**

Find `validate_compute_dtype` in ops.rs and ensure it checks both `is_compute_supported()` and returns `UnsupportedDtype` for Float64:
```rust
fn validate_compute_dtype(dtype: DType) -> Result<()> {
    if !dtype.is_compute_supported() {
        return Err(GpuError::UnsupportedDtype(
            format!("{} is not supported for GPU compute", dtype.name())
        ));
    }
    Ok(())
}
```

- [ ] **Step 4: Run all tests**

Run: `cargo test -p applegpu-core -- --test-threads=1`

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/tensor.rs crates/core/src/compute.rs crates/core/src/ops.rs
git commit -m "refactor: expand is_compute_supported to all types except Float64

BFloat16, Int32, Int64, UInt8, UInt32, Int8, Int16, Bool now marked as
compute-supported. Float64 returns UnsupportedDtype error.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 10: Add cast kernel template

**Files:**
- Modify: `crates/core/src/kernel_templates.rs`
- Modify: `crates/core/src/graph.rs`
- Modify: `crates/core/src/ops.rs`
- Modify: `crates/core/src/compute.rs`

- [ ] **Step 1: Add Cast OpKind variant**

In `crates/core/src/graph.rs`, add:
```rust
/// Cast tensor to a different dtype
Cast { target_dtype: DType },
```

Add to `kernel_name()`:
```rust
OpKind::Cast { .. } => "cast",
```

- [ ] **Step 2: Add cast kernel template**

In `kernel_templates.rs`:
```rust
/// Generate a cast kernel: reads as src_type, writes as dst_type.
pub fn cast_kernel_source(src: DType, dst: DType) -> String {
    let st = metal_type(src);
    let dt = metal_type(dst);
    let ss = dtype_suffix(src);
    let ds = dtype_suffix(dst);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{ND_INDEX_HELPER}
kernel void cast{ss}_to{ds}(device const {st}* input [[buffer(0)]], device {dt}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = ({dt})input[in_off];
}}
"#,
        ND_INDEX_HELPER = ND_INDEX_HELPER, st = st, dt = dt, ss = ss, ds = ds,
    )
}
```

- [ ] **Step 3: Add cast op to ops.rs**

```rust
pub fn cast(rt: &mut LazyRuntime, tensor_id: u64, target_dtype: DType) -> Result<u64> {
    let meta = rt.tensor_meta(tensor_id)?;
    if meta.dtype == target_dtype {
        return Ok(tensor_id); // no-op
    }
    let new_shape = meta.shape.clone();
    let node_id = rt.graph.add_node(
        OpKind::Cast { target_dtype },
        vec![tensor_id],
        new_shape,
        target_dtype,
    );
    Ok(node_id)
}
```

- [ ] **Step 4: Wire into compute.rs dispatch**

Add cast handling in the executor that calls `cast_kernel_source(src_dtype, target_dtype)`.

- [ ] **Step 5: Write test**

```rust
#[test]
fn cast_f32_to_i32() {
    // Create f32 tensor, cast to i32, verify values
}
```

- [ ] **Step 6: Run all tests**

Run: `cargo test -p applegpu-core -- --test-threads=1`

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/
git commit -m "feat: add Cast op with template-based kernel generation

Supports all dtype pairs. cast_kernel_source generates MSL for any
src→dst type conversion.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 11: Add byte-copy transpose/slice/concat kernels

**Files:**
- Modify: `crates/core/src/kernel_templates.rs`
- Modify: `crates/core/src/compute.rs`

- [ ] **Step 1: Add byte-copy kernel template**

```rust
/// Generate a dtype-agnostic copy kernel that works on raw bytes.
/// Element size determines the stride. Works for any dtype.
pub fn byte_copy_transpose_source(elem_size: usize) -> String {
    // Use char* for 1-byte, short* for 2-byte, int* for 4-byte, long* for 8-byte
    let t = match elem_size {
        1 => "char",
        2 => "short",
        4 => "int",
        8 => "long",
        _ => "char",
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void transpose_bytes_{elem}(device const {t}* input [[buffer(0)]], device {t}* out [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint id [[thread_position_in_grid]]) {{
    if (id >= rows * cols) return;
    uint row = id / cols;
    uint col = id % cols;
    out[col * rows + row] = input[row * cols + col];
}}
"#,
        t = t, elem = elem_size,
    )
}
```

- [ ] **Step 2: Wire into compute.rs for transpose/slice/concat**

Update the dispatch to use byte-copy kernels parameterized by `dtype.size_bytes()` instead of dtype-specific kernels.

- [ ] **Step 3: Run all tests**

Run: `cargo test -p applegpu-core -- --test-threads=1`
Run: `uv run pytest -v`

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/kernel_templates.rs crates/core/src/compute.rs
git commit -m "refactor: byte-copy transpose/slice/concat — dtype agnostic

Shape ops now use element-size-parameterized kernels instead of
typed pointers. Works for any dtype without needing per-dtype variants.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
