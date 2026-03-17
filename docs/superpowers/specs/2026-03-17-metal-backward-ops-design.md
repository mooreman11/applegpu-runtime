# Metal Backward Ops — Design Spec

**Date:** 2026-03-17
**Goal:** Eliminate all GPU→CPU→GPU roundtrips during training by implementing 6 backward ops as fused Metal kernels.

---

## Problem

Five backward ops run entirely on CPU, forcing full-size activation gradient tensors through GPU→CPU→GPU transfers on every training step. One additional op (conv1d backward) has no GPU path at all. These transfers dominate training time for models using ReLU, GELU, tanh, sigmoid activations, CNN pooling, or conv1d layers.

## Scope

### In scope — 6 fused Metal backward kernels

| Kernel | Inputs | Formula | Pattern |
|---|---|---|---|
| threshold_backward | grad, **input**, threshold (scalar) | `grad * (input > threshold)` | Element-wise binary + scalar |
| tanh_backward | grad, **output** | `grad * (1 - output²)` | Element-wise binary |
| sigmoid_backward | grad, **output** | `grad * output * (1 - output)` | Element-wise binary |
| gelu_backward | grad, **input** | See below | Element-wise binary (complex) |
| max_pool2d_backward | grad_output, **indices**, input_shape | Scatter: `output[indices[i]] += grad[i]` | Atomic scatter |
| conv1d_backward_input | grad_output, weight, stride, padding | Transposed 1D convolution | Spatial convolution |

**GELU backward formula (tanh approximation):**
```
a = sqrt(2/π) * (x + 0.044715 * x³)    [clamped to -10, 10]
tanh_a = tanh(a)
da = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
grad_input = grad * (0.5 * (1 + tanh_a) + 0.5 * x * (1 - tanh_a²) * da)
```

### Out of scope (tracked separately)

- grad_weight/grad_bias on GPU for conv1d, conv2d, layer_norm, batch_norm (stays CPU — small parameter tensors)
- Grouped convolution backward (groups > 1) — CPU fallback
- Exact GELU mode (`approximate="none"`) — forward kernel only supports tanh approximation
- Forward max_pool2d with GPU-side indices output — currently computes indices on CPU

## Design Decisions

### max_pool2d_backward: Atomic adds required

Pooling windows can overlap when `stride < kernel_size`, meaning multiple output positions can map to the same input position. The kernel must use `atomic_add_float` (CAS-loop pattern), reused from the existing `embedding_backward` kernel.

### gelu_backward: Tanh approximation only

The forward GELU kernel hardcodes tanh approximation. The backward must match. Supporting exact mode would require also updating the forward kernel — out of scope.

### conv1d_backward_input: groups=1 only

Matches the forward conv1d kernel and conv2d_backward_input. Whisper uses groups=1. The Python dispatch falls back to CPU for groups != 1.

### conv1d grad_weight: Stays on CPU

Consistent with conv2d pattern. Weight tensors are small (e.g., Whisper conv1: `[384, 80, 3]`). Reduction over batch dimension is awkward on GPU without shared-memory reduction.

### Dtype handling: Float accumulation for half types

All kernels use `needs_float_acc(dtype)` — promote Float16/BFloat16 to float for computation, cast back on store. For max_pool2d_backward with atomics, the output buffer must be float (atomic CAS only works on uint/float in Metal).

## Architecture

### Files touched per kernel (7 files)

1. **`crates/core/src/kernel_templates.rs`** — MSL kernel source generator, templated by dtype
2. **`crates/core/src/graph.rs`** — `OpKind` enum variant
3. **`crates/core/src/ops.rs`** — Shape/dtype validation + lazy graph node recording
4. **`crates/core/src/compute.rs`** — Kernel dispatch (pipeline creation, buffer binding, grid launch)
5. **`crates/core/src/lazy.rs`** — Match on OpKind, acquire output buffer, call dispatch
6. **`crates/python/src/lib.rs`** + **`backend.rs`** + **`metal_backend.rs`** — PyO3 bindings: `gpu.threshold_backward()` etc.
7. **`python/applegpu_runtime/torch_backend.py`** — Replace CPU fallback with `gpu.*` call

### Batch updates after all 6 kernels

8. **`crates/core/src/serial.rs`** — 6 new discriminant codes (70-75) for wire protocol v3 container support
9. **Tests** — Rust unit tests in `ops.rs`, Python integration tests

### Kernel patterns

**Element-wise (threshold, tanh, sigmoid, gelu):**
- Flat indexing: one thread per element
- Grid: `(numel, 1, 1)`
- Buffers: `input1 [[buffer(0)]], input2 [[buffer(1)]], output [[buffer(2)]], constant uint& numel [[buffer(3)]]`
- threshold_backward adds `constant float& threshold [[buffer(4)]]`

**max_pool2d_backward (scatter):**
- Grid: `(total_output_elements, 1, 1)` — one thread per grad_output element
- Each thread does `atomic_add_float(&grad_input[index], grad_value)`
- Output buffer must be zero-initialized before dispatch
- Buffers: `grad_output [[buffer(0)]], indices [[buffer(1)]], grad_input [[buffer(2)]], constant uint& numel [[buffer(3)]]`

**conv1d_backward_input (spatial):**
- Grid: `(in_len, in_channels, batch)` — one thread per input element
- Each thread iterates over `(oc, k)` to find which output positions depend on this input
- Inverts stride/padding geometry: `oh = (ih + pad - k) / stride` (only if divisible)
- Mirrors existing conv2d_backward_input but for 1D
- Buffers: `grad_output [[buffer(0)]], weight [[buffer(1)]], grad_input [[buffer(2)]], constant uint& batch [[buffer(3)]], ... stride, padding, etc.`

### Python dispatch (torch_backend.py)

Each handler replaces the CPU roundtrip with a direct GPU call:

```python
# Before (CPU fallback):
@register_op(torch.ops.aten.tanh_backward.default)
def _op_tanh_backward(grad_output, output):
    out_cpu = output.to_torch_cpu()
    grad_cpu = grad_output.to_torch_cpu()
    result = grad_cpu * (1 - out_cpu ** 2)
    return ApplegpuTensor.from_torch(result)

# After (GPU):
@register_op(torch.ops.aten.tanh_backward.default)
def _op_tanh_backward(grad_output, output):
    return _wrap(gpu.tanh_backward(_unwrap(grad_output), _unwrap(output)),
                 torch_dtype=grad_output.dtype, requires_grad=grad_output.requires_grad)
```

## Implementation Order

1. **threshold_backward** — simplest, validates the full pipeline pattern
2. **tanh_backward + sigmoid_backward** — same structure, can be done together
3. **gelu_backward** — most complex formula but still element-wise
4. **conv1d_backward_input** — mirrors existing conv2d_backward pattern
5. **max_pool2d_backward** — most complex (atomics, zero-init, index handling)
6. **serial.rs** — wire protocol for all 6 ops
7. **Tests** — Rust unit + Python integration

## Testing Strategy

**Rust unit tests (in `ops.rs`):**
- Random input tensors, compute backward on GPU and CPU reference
- Check output within float32 tolerance (atol=1e-5 for element-wise, atol=1e-4 for conv/pool)
- Test Float32, Float16, BFloat16 dtypes
- Edge cases: zero inputs, large values, negative values

**Python tests (in `python/tests/`):**
- Verify each `torch_backend.py` handler calls GPU path (no CPU fallback warning emitted)
- Compare GPU backward output against PyTorch CPU reference
- Training integration: run a training step on a small model using each activation, verify loss decreases

**Models exercised:**
- MLP with ReLU + GELU + tanh + sigmoid: covers threshold, gelu, tanh, sigmoid backward
- Small CNN with max_pool2d: covers max_pool2d backward
- Whisper-like conv1d encoder: covers conv1d backward

## Success Criteria

1. Zero CPU fallback warnings during training of MLP, CNN, and conv1d models
2. All backward outputs match PyTorch CPU reference within float32 tolerance
3. Training loss decreases correctly for all model types
4. No regression in existing tests (331+ passing)
