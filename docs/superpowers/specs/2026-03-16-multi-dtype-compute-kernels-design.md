# Multi-Dtype Compute Kernels Design

**Date:** 2026-03-16
**Updated:** 2026-03-16 (post Plan 2 review)
**Status:** Approved
**Scope:** Validate and test all existing ops for new dtypes. Add new ops (comparison, bitwise, element-wise min/max, quantize/dequantize). Add op-level dtype validation. Gate Int64 on Apple9+ hardware.

## Overview

Plan 2 completed the kernel template infrastructure — all ~25 MSL kernel categories are generated from parameterized templates in `kernel_templates.rs`, with a unified `resolve_kernel()` dispatch path for all dtypes. Cast, byte-copy shape ops, and `is_compute_supported()` expansion are done.

What remains: validating that each template produces correct results for each dtype, adding ~14 new op types, enforcing the coverage matrix (preventing nonsensical combos like `exp(Int32)`), and gating Int64 on Apple9+ hardware.

## Hardware Constraints (verified on M4)

- **Float64 (`double`)**: NOT supported in MSL. Hard compile error. Deferred to backlog — revisit when Apple hardware adds support.
- **Int64 (`long`)**: Compiles on Apple9+ GPUs (M3, M4). Not available on M1/M2. Runtime-gated with `device.supportsFamily(.apple9)`.
- **BFloat16 (`bfloat`)**: Supported on Apple GPU family 7+ (all Apple Silicon M1+).
- **Int32, Int8, Int16, UInt8, UInt32, Bool**: Fully supported in MSL on all Apple Silicon.

## Coverage Matrix

### Existing ops — templates exist for all, but only these combos are semantically valid

| Op | F32/F16 | BF16 | Int32 | Int64 (Apple9+) | UInt8/32 | Int8/16 | Bool |
|---|---|---|---|---|---|---|---|
| add, sub, mul, div | Yes | Yes | Yes | Yes | Yes | No | No |
| neg, abs, sign | Yes | Yes | Yes | Yes | abs/sign | No | No |
| exp, log, sqrt, tanh | Yes | Yes | No | No | No | No | No |
| relu, gelu | Yes | Yes | No | No | No | No | No |
| pow | Yes | Yes | Int32 only | No | No | No | No |
| clamp | Yes | Yes | Yes | Yes | Yes | No | No |
| softmax, softmax_causal | Yes | Yes | No | No | No | No | No |
| matmul | Yes | Yes | No | No | No | No | No |
| layer_norm, batch_norm | Yes | Yes | No | No | No | No | No |
| conv1d, conv2d | Yes | Yes | No | No | No | No | No |
| max_pool2d, avg_pool2d | Yes | Yes | No | No | No | No | No |
| embedding | Yes | Yes | N/A | N/A | No | No | No |
| sum, mean, argmax | Yes | Yes | sum/argmax | sum/argmax | sum→u32, argmax | sum→i32, argmax | sum (count) |
| where (condition=Bool) | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| masked_fill (mask=Bool) | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| triu, tril | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| gather, index_select | Yes | Yes | Yes | Yes | Yes | Yes | No |
| scalar_mul | Yes | Yes | Yes | Yes | Yes | No | No |
| add_bias | Yes | Yes | No | No | No | No | No |
| cast | all↔all (via resolve_cast_kernel) |
| transpose, slice, concat | Byte-copy kernel (parameterized by element size) |
| reshape | Metadata only (no kernel needed) |

**Important:** Template generators produce valid MSL for ANY dtype, but many combinations are semantically invalid (e.g., `exp(Int32)`, `softmax(Bool)`). The op-level dtype validation layer (Section 7) must enforce this matrix.

### New ops (not yet in OpKind enum)

| Op | BF16 | Int32 | Int64 (Apple9+) | UInt8/32 | Int8/16 | Bool |
|---|---|---|---|---|---|---|
| **lt, gt, le, ge, eq, ne** → Bool output | Yes | Yes | Yes | Yes | Yes | eq/ne |
| **min, max** (element-wise binary) | Yes | Yes | Yes | Yes | No | No |
| **bitwise** (and, or, xor, not, shl, shr) | No | Yes | Yes | Yes | Yes | and/or/xor/not |
| **modulo** (%) | No | Yes | Yes | Yes | No | No |
| **logical_not** | No | No | No | No | No | Yes |
| **quantize/dequantize** | target | No | No | uint8↔f16/f32 | int8↔f16/f32 | No |

### Explicitly out of scope

- **Backward ops** (SoftmaxBackward, LayerNormBackward, Conv2dBackwardInput, EmbeddingBackward, BatchNormBackward) — extend to multi-dtype in a future design.
- **Float64** — MSL does not support `double`. Backlog item.
- **Int8/Int16 arithmetic** — these are storage types for quantization. Use dequantize→compute→quantize instead.

## Condition and Scalar Parameter Typing

**`where` and `masked_fill`:** The condition/mask tensor should be Bool (`device const uchar*`). Currently, both accept the data dtype for the condition tensor — the validation layer should enforce Bool condition inputs.

**Scalar parameters:** Already generalized via `ScalarValue` enum in `graph.rs` (completed in Plan 1). `Pow`, `ScalarMul`, `Clamp`, and `MaskedFill` all use `ScalarValue`.

## Comparison Ops — Bool Output Design

Comparison ops (Lt, Gt, Le, Ge, Eq, Ne) take two same-dtype inputs and produce a Bool output. This breaks the assumption in `lazy_binary_op` (ops.rs) that `out_dtype == input_dtype`.

**Solution:** Add a `lazy_comparison_op` helper parallel to `lazy_binary_op` that:
- Validates both inputs have the same dtype
- Sets `out_dtype = DType::Bool`
- Records the op node with Bool output shape matching input shape

The comparison kernel templates take `device const {t}*` inputs and write `device bool*` output.

## Reduction Output Dtypes

| Input dtype | sum output | mean output | argmax output |
|---|---|---|---|
| Float32/16/BF16 | same | same | Int32 |
| Int32 | Int32 (wrapping) | Float32 | Int32 |
| Int64 | Int64 (wrapping) | Float32 | Int32 |
| UInt8/32 | UInt32 | Float32 | Int32 |
| Int8/16 | Int32 | Float32 | Int32 |
| Bool | Int32 (count of true) | Float32 (fraction) | Int32 |

## Op-Level Dtype Validation

**Problem:** Template generators produce MSL for any (op, dtype) pair. Without validation, `gpu.exp(int32_tensor)` silently produces garbage.

**Solution:** Add `fn validate_op_dtype(op: &OpKind, dtype: DType) -> Result<()>` in `ops.rs` that encodes the coverage matrix above. Call it from every op-recording function before `record_op()`.

Classification:
- **Float-only ops:** exp, log, sqrt, tanh, relu, gelu, softmax, softmax_causal, matmul, layer_norm, batch_norm, conv1d, conv2d, max_pool2d, avg_pool2d, add_bias, embedding, backward ops
- **Numeric ops (float + integer, not bool):** add, sub, mul, div, neg, abs, sign, clamp, scalar_mul, pow, sum, argmax, triu, tril, gather, index_select
- **All-dtype ops:** where, masked_fill, cast, transpose, slice, concat, reshape
- **Integer/Bool-only ops:** bitwise and/or/xor/not, shl, shr, modulo
- **Bool-only ops:** logical_not

## FusedElementwise Interaction

The fusion engine generates complete MSL strings with hardcoded types. It already uses `kernel_templates::metal_type()` and `dtype_suffix()` for type mapping. When adding new element-wise ops to the fusion candidate list:
- Comparison ops should NOT be fused (they change output dtype)
- Bitwise ops CAN be fused within integer chains
- Element-wise min/max CAN be fused
- `is_elementwise()` in OpKind should be expanded to include new fusable ops

## Remaining Prerequisites

Six of eight original prerequisites are complete. Two remain:

| # | Prerequisite | Status |
|---|---|---|
| 1 | Template-based kernel generation | **DONE** (Plan 2) |
| 2 | ScalarValue enum | **DONE** (Plan 1) |
| 3 | UnsupportedDtype error | **DONE** (Plan 2) |
| 4 | Per-op dtype validation | **NOT DONE** — `is_compute_supported()` allows all non-Float64 but doesn't prevent invalid combos |
| 5 | BFloat16 from_name fix | **DONE** (already had "bfloat16"/"bf16" arms) |
| 6 | Cast kernels | **DONE** (Plan 2) |
| 7 | Byte-copy shape ops | **DONE** (Plan 2) — note: transpose still uses typed template, not byte-copy |
| 8 | Device GPU family caching | **NOT DONE** — needed for Int64 runtime gating |

## Implementation Order

Remaining work, batched by priority:

1. **Op-level dtype validation** — `validate_op_dtype()` encoding the coverage matrix. Prevents nonsensical operations.
2. **Device GPU family caching + Int64 gating** — cache `supportsFamily(.apple9)` in Device, reject Int64 on older hardware.
3. **Wire transpose to byte-copy** — switch resolve_kernel "transpose" to use `byte_copy_transpose_source`.
4. **BFloat16 validation** — run all existing ops with BFloat16, verify results match Float16 within tolerance.
5. **Int32 arithmetic validation + comparison ops** — verify add/sub/mul/div/neg/abs/sign/clamp/scalar_mul work. Add Lt/Gt/Le/Ge/Eq/Ne with Bool output.
6. **Int64 arithmetic + comparison** (Apple9+ gated) — same as Int32 but gated. Fixes GPT-2 container bug.
7. **Bitwise ops + modulo** — BitwiseAnd/Or/Xor/Not, Shl, Shr, Mod for integer types.
8. **Element-wise min/max** — binary min/max ops.
9. **UInt8/UInt32 arithmetic + comparison** — validate and test.
10. **Bool logical ops** — LogicalNot, eq/ne for Bool.
11. **Int8/UInt8 quantize/dequantize** — enables quantized inference.

## Testing Strategy

- **Per-dtype kernel tests**: each dtype × op combination gets a Rust unit test verifying correctness
- **Python roundtrip tests**: `tensor(data, dtype=X) → op → to_list()` for each dtype
- **Op-dtype rejection tests**: verify `exp(int32_tensor)` raises UnsupportedDtype, not garbage
- **Cross-dtype cast tests**: every supported type pair (already partially done)
- **Int64 capability gating test**: verify error on unsupported hardware (mock or feature flag)
- **GPT-2 container regression**: end-to-end test that Int64 embedding indices work
- **BFloat16 parity test**: verify BFloat16 matches Float16 results within tolerance
- **Comparison op output dtype tests**: verify Lt/Gt etc. produce Bool output

## Backlog

- **Float64 compute kernels** — MSL does not support `double`. Revisit when Apple hardware adds support.
- **Backward ops multi-dtype** — extend backward kernels to BFloat16/Int types
- **`isinf`/`isnan`** for float types → Bool output
- **`fill`/`zeros`/`ones`** compute kernels for all dtypes
- **Quantized matmul** — Int8 weights × Float16 activations with scale factors (dedicated kernel, not generic int matmul)
- **Reduction output dtype overrides** — implement the reduction output dtype table (sum of Int32 → Int32, mean of Int32 → Float32, etc.)
