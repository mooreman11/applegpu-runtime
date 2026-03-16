# N-D Generalization + Missing Ops Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize 2D-hardcoded ops to N-D, add sin/cos/log_softmax, fix Conv1d bias, verify cross-attention shapes. Unblocks Whisper and higher-dimensional models.

**Architecture:** Three chunks: (1) sin/cos/log_softmax new ops (Rust kernel + dispatch + Python), (2) add_bias N-D generalization + Conv1d bias fix, (3) N-D audit + cross-attention test.

**Tech Stack:** Rust (applegpu-core), MSL (Metal Shading Language), PyO3, Python

**Spec:** `docs/superpowers/specs/2026-03-16-nd-generalization-design.md`

---

## Chunk 1: New Ops (sin, cos, log_softmax)

### Task 1: Add sin/cos to float_unary_kernel_source

**Files:**
- Modify: `crates/core/src/graph.rs` (OpKind::Sin, OpKind::Cos)
- Modify: `crates/core/src/kernel_templates.rs` (add to float_unary_kernel_source)
- Modify: `crates/core/src/ops.rs` (sin/cos functions + validate_op_dtype)
- Modify: `crates/core/src/serial.rs` (discriminants)
- Modify: `crates/python/src/backend.rs`, `metal_backend.rs`, `socket_backend.rs`, `lib.rs`
- Modify: `python/applegpu_runtime/__init__.py`

- [ ] **Step 1: Add Sin, Cos to OpKind**

In `graph.rs`, add to the enum:
```rust
Sin,
Cos,
```

In `kernel_name()`:
```rust
OpKind::Sin => "elementwise_sin",
OpKind::Cos => "elementwise_cos",
```

Add `Sin` and `Cos` to `is_unary()` and `is_elementwise()`.

- [ ] **Step 2: Add sin/cos to float_unary_kernel_source**

In `kernel_templates.rs`, add to `float_unary_kernel_source` alongside exp, log, sqrt, relu, tanh:
```rust
kernel void elementwise_sin{s}(...) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = sin(input[in_off]);
}}
kernel void elementwise_cos{s}(...) {{
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = cos(input[in_off]);
}}
```

Add test:
```rust
#[test]
fn float_unary_kernel_has_sincos() {
    let src = float_unary_kernel_source(DType::Float32);
    assert!(src.contains("elementwise_sin_f32"));
    assert!(src.contains("elementwise_cos_f32"));
}
```

- [ ] **Step 3: Add ops.rs functions**

```rust
pub fn sin(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::Sin)
}
pub fn cos(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::Cos)
}
```

Add `OpKind::Sin | OpKind::Cos` to the float-only arm of `validate_op_dtype`.

- [ ] **Step 4: Wire dispatch, serial, Python**

Sin/Cos are unary ops — they automatically dispatch through the existing unary N-D path in lazy.rs. Just need:
- serial.rs: discriminants (Sin=65, Cos=66)
- Backend trait + MetalBackend + SocketBackend + PyO3 + __init__.py

- [ ] **Step 5: Test + commit**

Run: `cargo test -p applegpu-core kernel_templates`
```bash
git commit -m "feat: add sin/cos unary ops"
```

### Task 2: Add log_softmax

**Files:**
- Modify: `crates/core/src/graph.rs` (OpKind::LogSoftmax)
- Modify: `crates/core/src/kernel_templates.rs` (log_softmax_kernel_source)
- Modify: `crates/core/src/compute.rs` (resolve_kernel)
- Modify: `crates/core/src/ops.rs` (log_softmax function)
- Modify: `crates/core/src/lazy.rs` (dispatch — same pattern as softmax)
- Modify: serial, Python layer

- [ ] **Step 1: Add LogSoftmax to OpKind**

```rust
LogSoftmax,
```

`kernel_name()`: `"log_softmax"`

- [ ] **Step 2: Add log_softmax_kernel_source**

In `kernel_templates.rs`:
```rust
pub fn log_softmax_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    let load = |e: &str| if acc { format!("float({})", e) } else { e.to_string() };
    let store = |e: &str| if acc { format!("{}({})", t, e) } else { e.to_string() };
    format!(r#"#include <metal_stdlib>
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
```

- [ ] **Step 3: Wire into resolve_kernel**

```rust
"log_softmax" => kt::log_softmax_kernel_source(dtype),
```

- [ ] **Step 4: Add ops.rs function**

Follow softmax pattern — flatten to rows × cols from last dim:
```rust
pub fn log_softmax(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_op_dtype(&OpKind::LogSoftmax, dtype)?;
    let shape = rt.shape(input_id)?;
    let cols = shape[shape.len() - 1];
    let total_rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    // ... record op, same pattern as softmax
}
```

- [ ] **Step 5: Add dispatch in lazy.rs**

Same pattern as softmax dispatch:
```rust
if node.op.is_log_softmax() {
    // ... dispatch_log_softmax_typed(device, dtype, &input.buffer, &out.buffer, total_rows, cols)
}
```

Add `dispatch_log_softmax_typed` to KernelRegistry in compute.rs (same as softmax dispatch but different kernel name).

- [ ] **Step 6: Wire serial + Python + test + commit**

Serial: LogSoftmax=67. Python: `gpu.log_softmax(t)`.
Test: verify `exp(log_softmax(x))` ≈ `softmax(x)`.

```bash
git commit -m "feat: add log_softmax with numerical stability"
```

---

## Chunk 2: add_bias N-D + Conv1d Fix

### Task 3: Generalize add_bias to N-D

**Files:**
- Modify: `crates/core/src/ops.rs` (relax 2D check)
- Modify: `crates/core/src/lazy.rs` (compute rows/cols for N-D)
- Test: Python test with 3D and 4D inputs

- [ ] **Step 1: Relax the 2D check in ops.rs**

Replace:
```rust
if input_shape.len() != 2 {
    return Err(GpuError::InvalidTensor(format!(
        "add_bias requires 2D input, got {:?}", input_shape
    )));
}
```

With:
```rust
if input_shape.len() < 2 {
    return Err(GpuError::InvalidTensor(format!(
        "add_bias requires at least 2D input, got {:?}", input_shape
    )));
}
```

And update the cols check:
```rust
if bias_shape[0] != input_shape[1] {
    return Err(GpuError::InvalidTensor(format!(
        "add_bias bias length {} != input channel dim {}", bias_shape[0], input_shape[1]
    )));
}
```

- [ ] **Step 2: Fix rows/cols computation in lazy.rs for N-D**

In the add_bias dispatch in lazy.rs, change the rows/cols computation:

Instead of `let rows = dims[0]; let cols = dims[1];`, compute:
```rust
let cols = dims[1]; // channel dim
let rows = dims[0] * dims[2..].iter().product::<usize>().max(1);
// For [B, C, L]: rows = B*L, cols = C
// For [B, C, H, W]: rows = B*H*W, cols = C
```

Wait — the kernel does `output[row * cols + col] = input[row * cols + col] + bias[col]`. For `[B, C, L]` where C is the channel:
- We need to iterate over B*L rows, each of length C
- But the data layout is `[B, C, L]` which is `B*C*L` contiguous with C as the second dim
- The memory layout is `[b][c][l]` so element `(b,c,l)` is at index `b*C*L + c*L + l`
- add_bias kernel adds `bias[col]` where col indexes the second dimension

The kernel assumes row-major `[rows, cols]` layout where `bias[col]` is added per column. For `[B, C, L]`, we need `bias[c]` added to every `(b, l)` position. This requires either:
a) Transpose to `[B, L, C]`, add_bias, transpose back — expensive
b) A new kernel that handles channel dim at position 1

Option b is better. The kernel should take `channel_stride` and `num_channels` instead of `rows/cols`:

```metal
kernel void add_bias_nd{s}(
    device const {t}* input, device const {t}* bias, device {t}* output,
    constant uint& numel, constant uint& num_channels, constant uint& channel_stride,
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    uint channel = (id / channel_stride) % num_channels;
    output[id] = input[id] + bias[channel];
}
```

Where `channel_stride = product(shape[2:])` and `num_channels = shape[1]`.

- [ ] **Step 3: Add add_bias_nd kernel template**

```rust
pub fn add_bias_nd_kernel_source(dtype: DType) -> String { ... }
```

- [ ] **Step 4: Wire into resolve_kernel and dispatch**

Update resolve_kernel to route `"add_bias"` to the new nd template. Update dispatch in lazy.rs to compute `numel`, `num_channels`, `channel_stride` and pass to the kernel.

- [ ] **Step 5: Test with 3D and 4D**

```python
# 3D: [B, C, L] — Conv1d output shape
inp = gpu.tensor([...], shape=[2, 3, 4])
bias = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
out = gpu.add_bias(inp, bias)
# bias[0]=1 added to all (b,l) where c=0, etc.
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: generalize add_bias to N-D with channel-aware kernel"
```

### Task 4: Fix Conv1d bias CPU fallback in torch_backend

**Files:**
- Modify: `python/applegpu_runtime/torch_backend.py`

- [ ] **Step 1: Replace CPU fallback with gpu.add_bias**

In `_convolution`, replace:
```python
if bias is not None:
    result_cpu = result.to_torch_cpu()
    result_cpu = result_cpu + bias.reshape(1, -1, *([1] * ndim))
    result = ApplegpuTensor.from_torch(result_cpu)
```

With:
```python
if bias is not None:
    result = _wrap(gpu.add_bias(_unwrap(result), _unwrap(bias)))
```

- [ ] **Step 2: Test — verify no CPU fallback warning for Conv1d + bias**

```python
model = torch.nn.Conv1d(3, 8, kernel_size=3, padding=1)
model = gpu.to_applegpu(model)
x = ApplegpuTensor.from_torch(torch.randn(1, 3, 16))
out = model(x)  # should NOT print CPU fallback warning
```

- [ ] **Step 3: Commit**

```bash
git commit -m "fix: Conv1d/Conv2d bias stays on GPU (was falling back to CPU)"
```

---

## Chunk 3: N-D Audit + Cross-Attention Test

### Task 5: Audit and test N-D ops

- [ ] **Step 1: Test softmax with 3D input**

```python
# softmax should already work — it flattens to rows x cols
t = gpu.tensor([...], shape=[2, 3, 4])
out = gpu.softmax(t)  # softmax along last dim (4)
assert out.shape == [2, 3, 4]
```

- [ ] **Step 2: Test sum/mean/argmax with 3D input**

```python
t = gpu.tensor([...], shape=[2, 3, 4])
s = gpu.sum(t)     # reduce last dim → [2, 3]
m = gpu.mean(t)    # reduce last dim → [2, 3]
a = gpu.argmax(t)  # reduce last dim → [2, 3]
```

- [ ] **Step 3: Test layer_norm with 3D input**

```python
t = gpu.tensor([...], shape=[2, 3, 4])
gamma = gpu.tensor([1,1,1,1], shape=[4])
beta = gpu.tensor([0,0,0,0], shape=[4])
out = gpu.layer_norm(t, gamma, beta, 1e-5)
assert out.shape == [2, 3, 4]
```

- [ ] **Step 4: Fix any ops that fail the 3D test**

If any op has a 2D restriction, relax it following the same pattern as add_bias.

- [ ] **Step 5: Commit**

```bash
git commit -m "test: verify softmax, sum, mean, argmax, layer_norm work with N-D input"
```

### Task 6: Cross-attention shape test

- [ ] **Step 1: Test attention with different Q and KV lengths**

```python
import applegpu_runtime as gpu
import numpy as np

gpu.init_backend()

# Cross-attention: Q from decoder (len=4), KV from encoder (len=8)
q = gpu.from_numpy(np.random.randn(2, 4, 64).astype(np.float32))
k = gpu.from_numpy(np.random.randn(2, 8, 64).astype(np.float32))
v = gpu.from_numpy(np.random.randn(2, 8, 64).astype(np.float32))

out = gpu.attention(q, k, v)
out.eval()
assert out.shape == [2, 4, 64]  # q_len output, d_v width
print("Cross-attention shapes OK:", out.shape)
```

- [ ] **Step 2: Commit**

```bash
git commit -m "test: verify cross-attention works with q_len != kv_len"
```

### Task 7: Final validation

- [ ] **Step 1: Run all tests**

```bash
make test-rust && uv run maturin develop && uv run pytest -v
```

- [ ] **Step 2: Commit milestone**

```bash
git commit --allow-empty -m "milestone: N-D generalization + missing ops complete"
```
