# Foundation Ops for Transformer Inference

**Date:** 2026-03-14
**Status:** Approved
**Scope:** 8 new ops enabling clean GPT-2 inference: scalar_mul exposure, reshape, slice, concat, add_bias, softmax_causal, attention_causal, argmax.

## Overview

Add the missing ops needed for clean end-to-end GPT-2 inference without CPU workarounds. These fill gaps in tensor manipulation (reshape, slice, concat), broadcasting (add_bias), causal masking (softmax_causal, attention_causal), and output selection (argmax).

## Op 1: scalar_mul (Python exposure)

Already implemented in Rust (`ops::scalar_mul`). Not currently exposed to Python.

### Python API
```python
c = a.scalar_mul(0.5)      # method
c = gpu.scalar_mul(a, 0.5)  # function
```

### Changes
- Add `scalar_mul` method on GpuTensor in `lib.rs`
- Add module-level `scalar_mul` pyfunction
- Register in pymodule
- Update `__init__.py`

No new Rust core or Swift changes.

## Op 2: reshape (blit copy)

Changes tensor shape by copying data to a new buffer. Validates `new_shape.numel() == old_shape.numel()`.

### Python API
```python
t2 = t.reshape([2, 3])      # method
t2 = gpu.reshape(t, [2, 3])  # function
```

### Implementation
- `OpKind::Reshape { new_shape: Vec<usize> }` — carries target shape
- On eval: use Metal `MTLBlitCommandEncoder.copy(from:sourceOffset:to:destinationOffset:size:)` for buffer-to-buffer copy. No compute kernel needed.
- Output dtype = input dtype, output shape = new_shape
- Works for ALL dtypes (byte-level copy, dtype-agnostic)
- Does NOT call `validate_compute_dtype` (not a compute op)

### Swift FFI
New function:
```swift
@_cdecl("gpu_bridge_blit_copy")
public func gpuBridgeBlitCopy(
    _ deviceHandle: UnsafeMutableRawPointer,
    _ queueHandle: UnsafeMutableRawPointer,
    _ srcBuf: UnsafeMutableRawPointer,
    _ dstBuf: UnsafeMutableRawPointer,
    _ sizeBytes: UInt64
) -> UnsafeMutableRawPointer?  // returns command buffer handle (non-blocking)
```

Uses shared command queue. Returns CB handle for batching.

### Graph integration
- For materialized tensors: can reshape immediately (record op, eval creates copy with new shape)
- For lazy tensors: records graph node, evaluated during topo-sort like any other op
- `is_reshape()` method on OpKind
- NOT fusible (not element-wise)

## Op 3: slice (extract sub-tensor)

Extract a contiguous slice along a specified dimension.

### Python API
```python
# Slice columns: extract columns [start, end) from 2D tensor
head_0 = gpu.slice(qkv, dim=1, start=0, end=64)    # [seq, 2304] → [seq, 64]
head_1 = gpu.slice(qkv, dim=1, start=64, end=128)   # [seq, 2304] → [seq, 64]

# Slice rows
batch = gpu.slice(x, dim=0, start=0, end=16)  # [32, 768] → [16, 768]
```

### MSL Kernel

**slice_dim1_f32** (column slice, most common for multi-head attention):
```metal
kernel void slice_dim1_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& in_cols [[buffer(2)]],
    constant uint& out_cols [[buffer(3)]],
    constant uint& start_col [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= out_cols) return;
    output[row * out_cols + col] = input[row * in_cols + (start_col + col)];
}
```

**slice_dim0_f32** (row slice):
```metal
kernel void slice_dim0_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& start_row [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (col >= cols) return;
    output[row * cols + col] = input[(start_row + row) * cols + col];
}
```

f16 variants with `half` types.

### Graph integration
- `OpKind::Slice { dim: usize, start: usize, end: usize }` — carries slice params
- Unary op: one input, output shape computed from input shape + slice params
- Output dtype = input dtype
- NOT fusible
- Validates: dim < ndim, start < end, end <= shape[dim]
- Works for all compute-supported dtypes (f32, f16). For non-compute dtypes (Int32 for embedding indices), could support via byte-copy variant, but v1 restricts to f32/f16.

### ops.rs
```rust
pub fn slice(rt: &mut LazyRuntime, input_id: u64, dim: usize, start: usize, end: usize) -> Result<u64>
```

## Op 4: concat (pairwise along dimension)

Concatenate two tensors along a specified dimension.

### Python API
```python
c = gpu.concat(a, b, dim=1)  # [rows, cols_a] + [rows, cols_b] → [rows, cols_a+cols_b]
c = gpu.concat(a, b, dim=0)  # [rows_a, cols] + [rows_b, cols] → [rows_a+rows_b, cols]
```

For N tensors (12 attention heads), Python loops pairwise:
```python
result = heads[0]
for h in heads[1:]:
    result = gpu.concat(result, h, dim=1)
```

### MSL Kernel

**concat_dim1_f32** (column concat):
```metal
kernel void concat_dim1_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols_a [[buffer(4)]],
    constant uint& cols_b [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    uint total_cols = cols_a + cols_b;
    if (row >= rows || col >= total_cols) return;
    if (col < cols_a) {
        output[row * total_cols + col] = a[row * cols_a + col];
    } else {
        output[row * total_cols + col] = b[row * cols_b + (col - cols_a)];
    }
}
```

f16 variants. dim=0 variant similar pattern.

### Graph integration
- `OpKind::Concat { dim: usize }` — binary op
- Output shape: input shapes match except on concat dim, which sums
- Output dtype = input dtype (must match)
- NOT fusible
- Validates: shapes match on all dims except concat dim, dtypes match

## Op 5: add_bias (2D + 1D broadcast)

Element-wise add where a 1D bias `[cols]` is added to each row of a 2D tensor `[rows, cols]`.

### Python API
```python
c = gpu.add_bias(x, bias)  # x: [rows, cols], bias: [cols] → [rows, cols]
```

### MSL Kernel

```metal
kernel void add_bias_f32(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= cols) return;
    output[row * cols + col] = input[row * cols + col] + bias[col];
}
```

f16 variant with `half`.

### Graph integration
- `OpKind::AddBias` — binary op (input 2D + bias 1D)
- Output shape = input shape, output dtype = input dtype
- NOT fusible (different buffer binding pattern than standard element-wise)
- Validates: input is 2D, bias is 1D, bias.len() == input.shape[1]

## Op 6: softmax_causal (composable building block)

Softmax with built-in upper-triangular causal mask. For position `(row, col)` where `col > row`, the value is treated as `-inf` before softmax normalization.

### Python API
```python
weights = gpu.softmax_causal(scores)  # scores: [seq, seq] → weights: [seq, seq]
```

### MSL Kernel

```metal
kernel void softmax_causal_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    uint offset = row * cols;

    // Find max (only causal positions: col <= row)
    float max_val = -1e9f;
    for (uint j = 0; j <= row && j < cols; j++) {
        max_val = max(max_val, input[offset + j]);
    }

    // Compute exp and sum (only causal positions)
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        if (j <= row) {
            float e = exp(input[offset + j] - max_val);
            output[offset + j] = e;
            sum += e;
        } else {
            output[offset + j] = 0.0f;  // masked position
        }
    }

    // Normalize
    for (uint j = 0; j <= row && j < cols; j++) {
        output[offset + j] /= sum;
    }
}
```

f16 variant with f32 intermediate computation (same pattern as existing softmax_f16).

### Graph integration
- `OpKind::SoftmaxCausal` — unary op, requires 2D square-ish input (rows can differ from cols for cross-attention future use, but for causal self-attention rows == cols)
- Output shape = input shape, output dtype = input dtype
- NOT fusible (reduction)
- `is_softmax_causal()` method

## Op 7: attention_causal (fused kernel)

Complete causal attention in one op: `softmax_causal(Q @ K^T / sqrt(d_k)) @ V`.

### Python API
```python
out = gpu.attention_causal(q, k, v)
```

### Implementation
Composite op in `ops.rs` (same pattern as existing `attention`):
```rust
pub fn attention_causal(rt, q_id, k_id, v_id) -> Result<u64> {
    let kt_id = transpose(rt, k_id)?;
    let scores_id = matmul(rt, q_id, kt_id)?;
    let scale = 1.0 / (d_k as f32).sqrt();
    let scaled_id = scalar_mul(rt, scores_id, scale)?;
    let weights_id = softmax_causal(rt, scaled_id)?;  // uses softmax_causal instead of softmax
    let output_id = matmul(rt, weights_id, v_id)?;
    Ok(output_id)
}
```

This reuses existing ops (transpose, matmul, scalar_mul) plus the new `softmax_causal`. No new kernel needed for attention_causal itself — it composes existing ops.

The Python method:
```python
out = gpu.attention_causal(q, k, v)
```

## Op 8: argmax (reduction → Int32)

Returns the index of the maximum value along the last dimension. Output dtype is always Int32 regardless of input dtype.

### Python API
```python
indices = gpu.argmax(t)      # t: [rows, cols] → indices: [rows] (Int32)
indices = gpu.argmax(t)      # t: [cols] → indices: [1] (Int32)
idx = t.argmax()             # method
```

### MSL Kernel

```metal
kernel void argmax_f32(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    uint offset = row * cols;
    float max_val = input[offset];
    int max_idx = 0;
    for (uint j = 1; j < cols; j++) {
        if (input[offset + j] > max_val) {
            max_val = input[offset + j];
            max_idx = int(j);
        }
    }
    output[row] = max_idx;
}
```

f16 variant reads `half`, compares in `float`, writes `int`.

### Graph integration
- `OpKind::Argmax` — unary op
- **Cross-dtype output:** input is f32/f16, output is ALWAYS Int32. This breaks the `out_dtype == input_dtype` pattern.
- Output shape: for 2D `[rows, cols]` → `[rows]`. For 1D `[cols]` → `[1]`.
- NOT fusible (reduction)
- `validate_compute_dtype` applies to input only
- `is_argmax()` method

### Special handling in execute_node
Argmax needs explicit `DType::Int32` in `Tensor::from_raw` call (not `node.out_dtype` from the usual path — though if `out_dtype` is set to Int32 in the OpNode, it works naturally).

## Testing Strategy

### Rust Unit Tests (~12 tests)
1. `test_reshape_preserves_data` — reshape [6] → [2,3], verify data
2. `test_reshape_validates_numel` — mismatched numel → error
3. `test_slice_dim1` — slice columns from 2D tensor
4. `test_slice_dim0` — slice rows from 2D tensor
5. `test_concat_dim1` — concat two tensors along columns
6. `test_concat_dim0` — concat along rows
7. `test_add_bias` — 2D + 1D addition
8. `test_softmax_causal` — verify upper triangle is zero, lower triangle sums to 1
9. `test_attention_causal` — compare to manual decomposition
10. `test_argmax_f32` — known input, verify correct indices
11. `test_argmax_returns_int32` — verify output dtype is Int32
12. `test_argmax_f16` — f16 input, Int32 output

### Python Tests (~10 tests)
1. `test_scalar_mul` — method and function
2. `test_reshape_roundtrip`
3. `test_slice_multihead` — slice [seq, 768] into 12 heads of [seq, 64]
4. `test_concat_multihead` — concat 12 heads back to [seq, 768]
5. `test_add_bias_eval`
6. `test_softmax_causal_masking` — verify future positions are zero
7. `test_attention_causal_eval`
8. `test_argmax_eval` — verify correct index
9. `test_argmax_dtype` — verify output is int32
10. `test_gpt2_attention_block` — full multi-head attention: project → slice → attention_causal × 12 → concat → project

## Backlog (not in scope)
- N-way concat kernel (currently pairwise)
- Zero-copy reshape (shared buffer views)
- General broadcasting (beyond 2D+1D)
- Slice for non-compute dtypes (Int32, etc.)
- `linear(x, W, bias)` fused op
