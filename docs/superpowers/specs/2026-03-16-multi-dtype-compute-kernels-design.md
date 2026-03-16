# Multi-Dtype Compute Kernels Design

**Date:** 2026-03-16
**Status:** Approved
**Scope:** Add compute kernels for all missing dtypes (BFloat16, Int32, Int64, UInt8, UInt32, Int8, Int16, Bool). Refactor kernel generation to templates. Add new ops (cast, comparison, bitwise, element-wise min/max, quantize/dequantize).

## Overview

Currently only Float32 and Float16 have compute kernels (~25 ops each). All other dtypes are storage-only — tensors can be created, serialized, and transferred, but no GPU computation is possible. This blocks GPT-2 in containers (Int64 embedding indices) and limits the library to float-only workloads.

This design adds compute support for 6 additional dtype categories, ~15 new op types, and refactors the kernel infrastructure from hand-duplicated MSL strings to a template-based generator.

## Hardware Constraints (verified on M4)

- **Float64 (`double`)**: NOT supported in MSL. Hard compile error. Deferred to backlog — revisit when Apple hardware adds support.
- **Int64 (`long`)**: Compiles on Apple9+ GPUs (M3, M4). Not available on M1/M2. Runtime-gated with `device.supportsFamily(.apple9)`.
- **BFloat16 (`bfloat`)**: Supported on Apple GPU family 7+ (all Apple Silicon M1+).
- **Int32, Int8, Int16, UInt8, UInt32, Bool**: Fully supported in MSL on all Apple Silicon.

## Coverage Matrix

### Existing ops (currently F32+F16 only, extending to new dtypes)

| Op | BF16 | Int32 | Int64 (Apple9+) | UInt8/32 | Int8/16 | Bool |
|---|---|---|---|---|---|---|
| add, sub, mul, div | Yes | Yes | Yes | Yes | No | No |
| neg, abs, sign | Yes | Yes | Yes | abs/sign | No | No |
| exp, log, sqrt, tanh | Yes | No | No | No | No | No |
| relu, gelu | Yes | No | No | No | No | No |
| pow | Yes | Int32 only | No | No | No | No |
| clamp | Yes | Yes | Yes | Yes | No | No |
| softmax, softmax_causal | Yes | No | No | No | No | No |
| matmul | Yes | No | No | No | No | No |
| layer_norm, batch_norm | Yes | No | No | No | No | No |
| conv1d, conv2d | Yes | No | No | No | No | No |
| max_pool2d, avg_pool2d | Yes | No | No | No | No | No |
| embedding | Yes | N/A (index is always int) | N/A | No | No | No |
| sum, mean, argmax | Yes | sum/argmax | sum/argmax | sum→u32, argmax | sum→i32, argmax | sum (count) |
| max, min (reduction) | Yes | Yes | Yes | No | No | No |
| where (condition=Bool) | Yes | Yes | Yes | Yes | Yes | Yes |
| masked_fill (mask=Bool) | Yes | Yes | Yes | Yes | Yes | Yes |
| triu, tril | Yes | Yes | Yes | Yes | Yes | Yes |
| gather, index_select | Yes | Yes | Yes | Yes | Yes | No |
| scalar_mul | Yes | Yes | Yes | Yes | No | No |
| add_bias | Yes | No | No | No | No | No |
| transpose, slice, concat | Byte-copy kernel (dtype-agnostic, parameterized by element size) |
| reshape | Metadata only (no kernel needed) |

### New ops (not yet in OpKind enum)

| Op | BF16 | Int32 | Int64 (Apple9+) | UInt8/32 | Int8/16 | Bool |
|---|---|---|---|---|---|---|
| **cast** (dtype conversion) | all↔all | all↔all | all↔all | all↔all | all↔all | all↔all |
| **lt, gt, le, ge, eq, ne** → Bool output | Yes | Yes | Yes | Yes | Yes | eq/ne |
| **min, max** (element-wise binary) | Yes | Yes | Yes | Yes | No | No |
| **bitwise** (and, or, xor, not, shl, shr) | No | Yes | Yes | Yes | Yes | and/or/xor/not |
| **modulo** (%) | No | Yes | Yes | Yes | No | No |
| **logical_not** | No | No | No | No | No | Yes |
| **quantize/dequantize** | target | No | No | uint8↔f16/f32 | int8↔f16/f32 | No |

### Explicitly out of scope

- **Backward ops** (SoftmaxBackward, LayerNormBackward, Conv2dBackwardInput, EmbeddingBackward, BatchNormBackward) — extend to multi-dtype in a future design.
- **Float64** — MSL does not support `double`. Backlog item: revisit when Apple hardware adds support.
- **Int8/Int16 arithmetic** — these are storage types for quantization. Use dequantize→compute→quantize instead.

## Condition and Scalar Parameter Typing

**`where` and `masked_fill`:** The condition/mask tensor is always Bool (`device const bool*` or `device const uchar*`). Data tensors can be any supported dtype.

**Scalar parameters need generalization.** Several existing OpKind variants carry `f32` scalar fields:
- `Pow { exponent: f32 }` — needs `ScalarValue` for integer pow
- `ScalarMul(f32)` — needs to carry the scalar in the tensor's dtype
- `MaskedFill { value: f32 }` — needs dtype-aware fill value
- `Clamp { min_val: f32, max_val: f32 }` — needs dtype-aware bounds

Solution: introduce a `ScalarValue` enum in `graph.rs`:
```rust
pub enum ScalarValue {
    Float(f64),   // covers f32, f16, bf16
    Int(i64),     // covers i32, i64, i16, i8
    UInt(u64),    // covers u32, u8
    Bool(bool),
}
```

## Reduction Output Dtypes

| Input dtype | sum output | mean output | argmax output |
|---|---|---|---|
| Float32/16/BF16 | same | same | Int32 |
| Int32 | Int32 (wrapping) | Float32 | Int32 |
| Int64 | Int64 (wrapping) | Float64 (CPU) or Float32 | Int32 |
| UInt8/32 | UInt32 | Float32 | Int32 |
| Int8/16 | Int32 | Float32 | Int32 |
| Bool | Int32 (count of true) | Float32 (fraction) | Int32 |

## FusedElementwise Interaction

The template-based kernel generator and FusedElementwise both produce MSL at runtime. The fusion engine (`OpKind::FusedElementwise { kernel_source, function_name }`) generates complete MSL strings with hardcoded types. When extending fusion to multi-dtype:
- The fusion pass must track the dtype of the fused chain
- Generated MSL must use the correct Metal type (`half`, `bfloat`, `int`, `long`, etc.)
- This is a natural extension of prerequisite 1 — the template generator provides type-mapping functions that the fusion engine can reuse

## Prerequisites (ordered)

1. **Template-based kernel generation** — replace ~70 duplicated MSL string constants with `fn kernel_source(metal_type: &str, suffix: &str) -> String` generator. This is the architectural prerequisite that makes adding 6+ dtypes feasible.

2. **OpKind enum expansion** — add ~15 new variants to `crates/core/src/graph.rs`: Cast, Lt, Gt, Le, Ge, Eq, Ne, BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, Shl, Shr, Mod, ElemMin, ElemMax, LogicalNot, Quantize, Dequantize. Also introduce `ScalarValue` enum for dtype-aware scalar parameters.

3. **`UnsupportedDtype` error variant** — add to `crates/core/src/error.rs`. Needed by all subsequent prerequisites. Int64 on pre-Apple9 returns this error by default (no silent cast to Int32).

4. **`is_compute_supported()` refactor** — replace the boolean at `tensor.rs:31` with a per-op, per-device capability query. Something like `KernelRegistry::supports(op: OpKind, dtype: DType, device: &Device) -> bool`.

5. **BFloat16 `from_name` bug fix** — add `"bfloat16" | "bf16"` match arm to `DType::from_name()` at `tensor.rs:36`. Pre-existing bug that blocks BFloat16 compute.

6. **Cast kernels** — dtype conversion for all supported type pairs. Essential for multi-dtype to be usable.

7. **Byte-copy transpose/slice/concat** — rewrite to use `device const char*` with element-size indexing, making them dtype-agnostic. Eliminates need for per-dtype shape op kernels.

8. **Store device GPU family in Device struct** — cache `device.supportsFamily(.apple9)` etc. at init time for efficient runtime capability checks.

## Implementation Order (after prerequisites)

Batch by dtype category for maximum efficiency:

1. **BFloat16** — one template parameter change if prerequisite 1 is done. Immediate win for transformer training.
2. **Int32 arithmetic + comparison + bitwise** — most commonly needed integer ops. Unblocks index manipulation.
3. **Int64 arithmetic + comparison** (Apple9+ gated) — unblocks GPT-2 container bug via proper Int64 support.
4. **Cast kernels** — enables dtype interop.
5. **UInt8/UInt32 arithmetic + comparison + bitwise** — completes unsigned integer support.
6. **Bool comparison output + logical ops** — completes the type system.
7. **Int8/UInt8 quantize/dequantize** — enables quantized inference.
8. **Element-wise min/max, modulo** — utility ops.

## Testing Strategy

- **Per-dtype kernel tests**: each new dtype × op combination gets a Rust unit test verifying correctness
- **Python roundtrip tests**: `tensor(data, dtype=X) → op → to_list()` for each dtype
- **Cross-dtype cast tests**: every supported type pair
- **Int64 capability gating test**: verify error on unsupported hardware (mock)
- **GPT-2 container regression**: end-to-end test that the Int64 embedding indices work after this
- **BFloat16 parity test**: verify BFloat16 matches Float16 results within tolerance

## Backlog

- **Float64 compute kernels** — MSL does not support `double`. Revisit when Apple hardware adds support.
- **Backward ops multi-dtype** — extend backward kernels to BFloat16/Int types
- **`isinf`/`isnan`** for float types → Bool output
- **`fill`/`zeros`/`ones`** compute kernels for all dtypes
- **Quantized matmul** — Int8 weights × Float16 activations with scale factors (dedicated kernel, not generic int matmul)
