# Multi-Dtype Kernel Templates (Plan 2 of 3) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Template all remaining MSL kernel sources, expand `is_compute_supported`, add Cast op, and make shape ops dtype-agnostic. Builds on Plan 1's template infrastructure.

**Architecture:** Extend `kernel_templates.rs` with generators for all ~25 remaining op categories. Replace all `const` F16 kernel strings with template calls in `resolve_kernel`. Add Cast op to graph + ops + compute. Replace typed transpose/slice/concat with byte-copy kernels. Delete ~2000 lines of duplicated MSL.

**Tech Stack:** Rust (applegpu-core), MSL (Metal Shading Language)

**Spec:** `docs/superpowers/specs/2026-03-16-multi-dtype-compute-kernels-design.md`

---

## Chunk 1: Template Remaining Ops

### Task 8a: Template element-wise adjacent ops (scalar_mul, pow, clamp, gelu)

**Files:**
- Modify: `crates/core/src/kernel_templates.rs`
- Modify: `crates/core/src/compute.rs` (wire into resolve_kernel)

- [ ] **Step 1: Add template generators**

Add to `kernel_templates.rs`: `scalar_mul_kernel_source(dtype)`, `pow_kernel_source(dtype)`, `clamp_kernel_source(dtype)`, `gelu_kernel_source(dtype)`.

Each follows the same pattern as `binary_kernel_source` — read the existing F32 `const` in `compute.rs`, replace `float` with `{t}` and suffix with `{s}`.

For gelu: use f32 intermediate accumulation for f16/bf16 (existing pattern from GELU_KERNEL_SOURCE_F16).

- [ ] **Step 2: Add tests**

```rust
#[test]
fn scalar_mul_kernel_typed() {
    let src = scalar_mul_kernel_source(DType::Int32);
    assert!(src.contains("device const int*"));
    assert!(src.contains("scalar_mul_i32"));
}

#[test]
fn gelu_kernel_typed() {
    let src = gelu_kernel_source(DType::BFloat16);
    assert!(src.contains("gelu_bf16"));
}
```

- [ ] **Step 3: Wire into resolve_kernel for F16 path**

In `resolve_kernel`, replace the F16 const match arms for these ops with template calls.

- [ ] **Step 4: Run all tests**

Run: `cargo test -p applegpu-core -- --test-threads=1`

- [ ] **Step 5: Commit**

### Task 8b: Template reduction ops (softmax, softmax_causal, argmax, sum, mean, add_bias)

**Files:** Same as 8a.

- [ ] **Step 1: Add template generators**

`softmax_kernel_source(dtype)`, `softmax_causal_kernel_source(dtype)`, `argmax_kernel_source(dtype)`, `sum_kernel_source(dtype)`, `mean_kernel_source(dtype)`, `add_bias_kernel_source(dtype)`.

Important: softmax/sum/mean need float accumulation even for f16/bf16. The template should cast to float for intermediate computation. argmax always outputs Int32.

- [ ] **Step 2: Tests + wire into resolve_kernel**
- [ ] **Step 3: Run all tests, commit**

### Task 8c: Template conditional/index ops (where, masked_fill, triu, tril, gather, index_select)

**Files:** Same.

- [ ] **Step 1: Add template generators**

`where_kernel_source(dtype)`, `masked_fill_kernel_source(dtype)`, `triu_kernel_source(dtype)`, `tril_kernel_source(dtype)`, `gather_kernel_source(dtype, dim)`, `index_select_kernel_source(dtype, dim)`.

Note: `where` condition should use `bool`/`uchar` type for the condition tensor, regardless of data dtype.

- [ ] **Step 2: Tests + wire + commit**

### Task 8d: Template complex ops (matmul, layer_norm, embedding, CNN, backward)

**Files:** Same.

- [ ] **Step 1: Add template generators**

`matmul_kernel_source(dtype)`, `layer_norm_kernel_source(dtype)`, `embedding_kernel_source(dtype)`, `conv1d_kernel_source(dtype)`, `conv2d_kernel_source(dtype)`, `batch_norm_kernel_source(dtype)`, `max_pool2d_kernel_source(dtype)`, `avg_pool2d_kernel_source(dtype)`.

Also backward ops: `softmax_backward_kernel_source(dtype)`, `layer_norm_backward_kernel_source(dtype)`, `conv2d_backward_input_kernel_source(dtype)`, `embedding_backward_kernel_source(dtype)`, `batch_norm_backward_kernel_source(dtype)`.

`copy_strided_kernel_source(dtype)` for the strided copy utility.

These are float-only ops (matmul, softmax, layer_norm, CNN, backward). The template only needs to support F32/F16/BF16.

- [ ] **Step 2: Tests + wire into resolve_kernel**
- [ ] **Step 3: Delete all old const F16 kernel sources**

After all ops are templated, delete every `const *_KERNEL_SOURCE_F16: &str` from compute.rs. This should remove ~1500-2000 lines.

- [ ] **Step 4: Run ALL tests (Rust + Python)**

Run: `cargo test -p applegpu-core -- --test-threads=1`
Run: `uv run maturin develop && uv run pytest -v`

- [ ] **Step 5: Commit**

---

## Chunk 2: is_compute_supported + Cast + Byte-copy Shape Ops

### Task 9: Expand is_compute_supported

**Files:**
- Modify: `crates/core/src/tensor.rs`
- Modify: `crates/core/src/ops.rs`

- [ ] **Step 1: Update is_compute_supported**

```rust
pub fn is_compute_supported(&self) -> bool {
    !matches!(self, DType::Float64)
}
```

- [ ] **Step 2: Update validate_compute_dtype in ops.rs**

Use `UnsupportedDtype` error for Float64:
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

- [ ] **Step 3: Test + commit**

### Task 10: Add Cast op

**Files:**
- Modify: `crates/core/src/kernel_templates.rs`
- Modify: `crates/core/src/graph.rs`
- Modify: `crates/core/src/ops.rs`
- Modify: `crates/core/src/compute.rs` (or `lazy.rs` — wherever ops are dispatched)

- [ ] **Step 1: Add Cast to OpKind**

```rust
Cast { target_dtype: DType },
```

Add to `kernel_name()`: `OpKind::Cast { .. } => "cast"`

- [ ] **Step 2: Add cast_kernel_source to kernel_templates.rs**

```rust
pub fn cast_kernel_source(src: DType, dst: DType) -> String {
    let st = metal_type(src);
    let dt = metal_type(dst);
    let ss = dtype_suffix(src);
    let ds = dtype_suffix(dst);
    format!(r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void cast{ss}_to{ds}(device const {st}* input [[buffer(0)]], device {dt}* out [[buffer(1)]], constant uint* in_strides [[buffer(2)]], constant uint* out_shape [[buffer(3)]], constant uint& ndim [[buffer(4)]], constant uint& numel [[buffer(5)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = ({dt})input[in_off];
}}
"#, nd = ND_INDEX_HELPER, st = st, dt = dt, ss = ss, ds = ds)
}
```

- [ ] **Step 3: Add cast op to ops.rs**

Follow existing op pattern (check `ops::add` for reference). Record a `Cast { target_dtype }` node with the input's shape but the target dtype.

- [ ] **Step 4: Add dispatch in lazy.rs/compute.rs**

In the executor, handle `OpKind::Cast { target_dtype }`: generate kernel source via `cast_kernel_source(input_dtype, target_dtype)`, dispatch.

- [ ] **Step 5: Test**

```rust
#[test]
fn cast_f32_to_i32() {
    // Create f32 tensor [1.5, 2.7, 3.0], cast to i32, verify [1, 2, 3]
}
```

- [ ] **Step 6: Expose to Python**

Add `gpu.cast(tensor, "int32")` to the Python API in `crates/python/src/lib.rs` (or metal_backend.rs).

- [ ] **Step 7: Commit**

### Task 11: Byte-copy transpose/slice/concat

**Files:**
- Modify: `crates/core/src/kernel_templates.rs`
- Modify: `crates/core/src/compute.rs`

- [ ] **Step 1: Add byte-copy kernel templates**

```rust
pub fn byte_copy_transpose_source(elem_size: usize) -> String {
    let t = match elem_size {
        1 => "char", 2 => "short", 4 => "int", 8 => "long", _ => "char",
    };
    // ... transpose kernel using {t} as the copy type
}
```

Same for slice and concat — parameterize by element size instead of dtype.

- [ ] **Step 2: Wire into resolve_kernel**

For transpose/slice/concat, use `dtype.size_bytes()` to select the byte-copy variant instead of a typed kernel.

- [ ] **Step 3: Delete old typed transpose/slice/concat const sources**

Remove the F32 and F16 const strings for these ops.

- [ ] **Step 4: Run ALL tests**

Run: `cargo test -p applegpu-core -- --test-threads=1`
Run: `uv run maturin develop && uv run pytest -v`

- [ ] **Step 5: Commit**
