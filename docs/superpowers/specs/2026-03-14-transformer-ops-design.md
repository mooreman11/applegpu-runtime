# Transformer Ops: GELU, LayerNorm, Embedding

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Three new GPU ops enabling transformer layer inference. GELU (element-wise), LayerNorm with affine (reduction), Embedding (index lookup with Int32 tensors).

## Overview

Add the minimum viable op set for a transformer layer: GELU activation, LayerNorm with learnable gamma/beta, and embedding lookup using Int32 index tensors. These combine with existing matmul, softmax, add, and attention to cover a full transformer block (self-attention + FFN).

## Op 1: GELU

Element-wise activation used in transformer FFN layers.

### Formula
`gelu(x) = x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`

Constant: `sqrt(2/pi) ≈ 0.7978845608`

### MSL Kernels

**gelu_f32:**
```metal
kernel void gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        float x = input[id];
        output[id] = x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}
```

**gelu_f16:** Same structure with `half` types, f32 intermediate computation for tanh accuracy:
```metal
kernel void gelu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        float x = float(input[id]);
        output[id] = half(x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))));
    }
}
```

### Graph Integration
- `OpKind::Gelu` added to enum
- Unary op: one input, same shape/dtype output
- **Fusible**: GELU is element-wise, participates in kernel fusion chains
- Add `is_gelu()` method to OpKind
- Fusion engine: GELU expression in `unary_msl`: `(expr * 0.5f * (1.0f + tanh(0.7978845608f * (expr + 0.044715f * expr * expr * expr))))`

### Python API
```python
c = a.gelu()        # method
c = gpu.gelu(a)     # function
```

## Op 2: LayerNorm (with affine parameters)

Per-row normalization with learnable scale (gamma) and bias (beta).

### Formula
For each row: `output = gamma * (x - mean) / sqrt(var + eps) + beta`

Where:
- `mean = sum(x) / cols`
- `var = sum((x - mean)^2) / cols`
- `gamma`: 1D tensor of shape `[cols]` (scale)
- `beta`: 1D tensor of shape `[cols]` (bias)
- `eps`: small constant (default 1e-5)

### MSL Kernels

**layer_norm_f32:** One thread per row.
```metal
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

    // Compute mean
    float mean = 0.0f;
    for (uint j = 0; j < cols; j++) mean += input[offset + j];
    mean /= float(cols);

    // Compute variance
    float var = 0.0f;
    for (uint j = 0; j < cols; j++) {
        float diff = input[offset + j] - mean;
        var += diff * diff;
    }
    var /= float(cols);

    // Normalize with affine
    float inv_std = 1.0f / sqrt(var + eps);
    for (uint j = 0; j < cols; j++) {
        output[offset + j] = gamma[j] * (input[offset + j] - mean) * inv_std + beta[j];
    }
}
```

**layer_norm_f16:** Read half, compute in float, write half. Gamma/beta are also half.

### Graph Integration
- `OpKind::LayerNorm { eps: f32 }` added to enum
- Ternary op: input `[rows, cols]` + gamma `[cols]` + beta `[cols]` → output `[rows, cols]`
- Output dtype = input dtype
- **NOT fusible** (reduction op, like softmax)
- Validates: input is 2D, gamma/beta are 1D with length == cols
- `is_layer_norm()` method on OpKind

### Python API
```python
c = a.layer_norm(gamma, beta, eps=1e-5)  # method
c = gpu.layer_norm(a, gamma, beta, eps=1e-5)  # function
```

Where `gamma` and `beta` are GpuTensors of shape `[cols]`. The eps parameter has a default.

## Op 3: Embedding

Index-based lookup from a weight matrix using Int32 indices.

### Formula
`output[i, j] = weights[indices[i], j]`

### MSL Kernel

**embedding_f32:** One thread per output element.
```metal
kernel void embedding_f32(
    device const float* weights [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;  // sequence position
    uint j = gid.x;  // embedding dimension
    if (i >= seq_len || j >= embed_dim) return;
    int idx = indices[i];
    output[i * embed_dim + j] = weights[idx * embed_dim + j];
}
```

**embedding_f16:** Same but `half` for weights and output. Indices stay `int`.

### Graph Integration
- `OpKind::Embedding` added to enum
- Binary op: weights `[vocab_size, embed_dim]` (f32/f16) + indices `[seq_len]` (Int32)
- Output: `[seq_len, embed_dim]`, dtype = weights dtype
- **NOT fusible** (index-based, not element-wise)
- **Mixed-dtype exception**: this is the first op where inputs have different dtypes. Special validation in ops.rs — indices must be Int32, weights must be Float32 or Float16.
- `is_embedding()` method on OpKind

### Python API
```python
output = gpu.embedding(weights, indices)  # function only (not a method — takes two tensors)
```

## Kernel Dispatch Integration

### compute.rs changes
- Add `GELU_KERNEL_SOURCE` and `GELU_KERNEL_SOURCE_F16` constants
- Add `LAYER_NORM_KERNEL_SOURCE` and `LAYER_NORM_KERNEL_SOURCE_F16` constants
- Add `EMBEDDING_KERNEL_SOURCE` and `EMBEDDING_KERNEL_SOURCE_F16` constants
- Add `dispatch_gelu` (unary pattern, same as relu)
- Add `dispatch_layer_norm` (custom — 3 input buffers + 3 constants)
- Add `dispatch_embedding` (custom — 2 input buffers + 2 constants, 2D grid)
- Non-blocking `_nb` variants for all three (follows existing batching pattern)

### ops.rs changes
- `gelu(rt, input_id) -> Result<u64>` — uses `lazy_unary_op` with OpKind::Gelu
- `layer_norm(rt, input_id, gamma_id, beta_id, eps) -> Result<u64>` — custom validation (2D input, 1D gamma/beta matching cols)
- `embedding(rt, weights_id, indices_id) -> Result<u64>` — custom validation (2D weights, 1D int32 indices)

### fusion.rs changes
- GELU added to `is_elementwise()` check (or handled separately in `unary_msl`)
- `unary_msl` gets GELU expression for fusion
- LayerNorm and Embedding excluded from fusion (already handled by existing `is_elementwise()` filter)

### lazy.rs execute_node changes
- Add GELU dispatch (same pattern as other unary ops)
- Add LayerNorm dispatch (new pattern — 3 input buffers)
- Add Embedding dispatch (new pattern — 2 inputs, 2D grid, mixed dtype)

## Testing Strategy

### Rust Unit Tests (~8 tests)
1. `test_gelu_f32` — compare to manual calculation for known inputs
2. `test_gelu_f16` — same with f16 tolerance
3. `test_layer_norm_f32` — compare to manual mean/var/normalize calculation
4. `test_layer_norm_f16` — same with f16 tolerance
5. `test_embedding_f32` — lookup known indices, verify output rows
6. `test_embedding_f16` — same with f16 weights
7. `test_gelu_fusion` — gelu in a chain (e.g., add → gelu) gets fused
8. `test_embedding_rejects_non_int32_indices` — float indices → error

### Python Tests (~8 tests)
1. `test_gelu_eval` — from_numpy, gelu, to_numpy, compare to scipy/numpy reference
2. `test_layer_norm_eval` — compare to manual or torch.nn.functional.layer_norm
3. `test_embedding_eval` — int32 indices, verify output shape and values
4. `test_gelu_method_and_function` — both `a.gelu()` and `gpu.gelu(a)` work
5. `test_layer_norm_shape_validation` — non-2D input → error
6. `test_embedding_shape_validation` — 1D weights or non-int32 indices → error
7. `test_gelu_f16` — f16 tensor through gelu
8. `test_layer_norm_with_torch_reference` — compare against PyTorch's layer_norm for numerical accuracy

## Backlog (not in scope)
- LayerNorm without affine (just normalize) — could be added as `layer_norm_simple` later
- Embedding with padding_idx support
- Embedding gradient (backward pass)
- RMSNorm (variant used in LLaMA/modern transformers)
