# N-D Generalization + Missing Ops Design

**Date:** 2026-03-16
**Status:** Approved
**Scope:** Generalize 2D-hardcoded ops to N-D, add sin/cos/log_softmax ops, fix Conv1d bias, verify cross-attention shapes.

## Overview

Several ops in `ops.rs` are hardcoded to 2D input when their underlying Metal kernels operate on flattened `rows × cols`. This blocks Conv1d/Conv2d bias addition (3D/4D output), higher-dimensional models, and Whisper inference. Additionally, sin/cos and log_softmax are missing ops needed for Whisper and general transformer use.

## Problem: 2D-Hardcoded Ops

The `add_bias` kernel (`add_bias_{dtype}` in `kernel_templates.rs`) takes `rows`, `cols`, and a 2D grid — it adds `bias[col]` to `input[row * cols + col]`. This works for ANY N-D input if we flatten all non-channel dims into `rows` and use `shape[1]` as `cols`. But `ops.rs:984` rejects non-2D input:

```rust
if input_shape.len() != 2 {
    return Err(GpuError::InvalidTensor("add_bias requires 2D input"));
}
```

The same pattern may exist in other ops. Each needs auditing.

## Ops to Audit

| Op | Current Restriction | Fix |
|----|-------------------|-----|
| `add_bias` | 2D only (ops.rs:984) | Accept N-D, flatten leading dims × trailing dims around channel dim |
| `softmax` | Flattens to rows × cols from last dim — should work for N-D | Verify with test |
| `softmax_causal` | Takes batch_size × rows × cols — 3D specific | Verify or generalize |
| `sum` / `mean` / `argmax` | Flattens to rows × cols — should work | Verify with test |
| `layer_norm` | Takes rows × cols — should work | Verify with test |
| `embedding` | Takes seq_len × embed_dim — 2D specific | Check if 2D+ indices work |

## New Ops

### sin / cos

Float-only unary ops. Same pattern as exp/log/sqrt — already in MSL as built-in `sin()` and `cos()`.

- Add `Sin`, `Cos` to `OpKind` in `graph.rs`
- Add to `float_unary_kernel_source` in `kernel_templates.rs` (alongside exp, log, sqrt, relu, tanh)
- Add `sin()`, `cos()` to `ops.rs`
- Dispatch via existing unary N-D path in `lazy.rs`
- Wire Python API: `gpu.sin(t)`, `gpu.cos(t)`
- Add to `validate_op_dtype` as float-only

### log_softmax

Numerically stable fused log(softmax(x)). Reduction along last dimension, same dispatch pattern as softmax.

```
log_softmax(x)[j] = x[j] - max(x) - log(sum(exp(x[i] - max(x))))
```

Three passes over the row: (1) find max, (2) compute log-sum-exp, (3) subtract.

- Add `LogSoftmax` to `OpKind`
- New template `log_softmax_kernel_source(dtype)` in `kernel_templates.rs`
- Add to `resolve_kernel`, `ops.rs`, `lazy.rs`
- Wire Python API: `gpu.log_softmax(t)`
- Float accumulation for half types (same as softmax)

## Conv1d Bias Fix

After `add_bias` supports N-D, update `torch_backend.py` `_convolution`:

Replace:
```python
if bias is not None:
    result_cpu = result.to_torch_cpu()
    result_cpu = result_cpu + bias.reshape(1, -1, *([1] * ndim))
    result = ApplegpuTensor.from_torch(result_cpu)
```

With:
```python
if bias is not None:
    # Reshape conv output [B, C, ...] to [B*..., C] for add_bias, then reshape back
    # Or: add a dedicated conv_bias op that handles the channel dim
```

Actually, the cleanest fix: `add_bias` for N-D should treat dim 1 as the channel dimension and broadcast `bias[c]` to all other dimensions. The kernel needs to index as `output[idx] = input[idx] + bias[channel_of(idx)]`. For the Conv1d case `[B, C, L]`: `channel = (flat_idx / L) % C`.

Simpler alternative: in `ops.rs`, reshape to 2D before dispatch:
- `[B, C, L]` → flatten to `[B*L, C]` via transpose+reshape, add bias, reshape back
- But this requires a transpose which copies data

Simplest: generalize the `add_bias` kernel to take arbitrary N-D input with channel at dim 1. The kernel grid is `(total_elements)` and each thread computes `channel = (id / stride_after_channel) % num_channels`.

## Cross-Attention Verification

The existing `attention(q, k, v)` op in `ops.rs:1115` already handles `q_len != kv_len` — it computes `Q @ K^T → [q_len, kv_len]` then `attn @ V → [q_len, d_v]`. Just needs a test to confirm.

## Testing Strategy

- **sin/cos**: verify `sin(0)=0, sin(pi/2)=1, cos(0)=1, cos(pi)=-1`
- **log_softmax**: verify `exp(log_softmax(x))` ≈ `softmax(x)` within tolerance
- **add_bias N-D**: test with `[2, 3, 4]` (3D) and `[2, 3, 4, 5]` (4D) inputs
- **Conv1d bias**: verify `conv1d + bias` stays on GPU (no CPU fallback warning)
- **cross-attention**: `attention(q=[2,4,64], k=[2,8,64], v=[2,8,64])` → output `[2,4,64]`
- **N-D audit**: verify softmax, sum, mean, argmax, layer_norm with 3D+ inputs

## Not Included

- Generalized N-D reduction axis (always reduces last dim)
- Custom channel dimension (always dim 1 for add_bias)
- Backward ops for sin/cos/log_softmax
