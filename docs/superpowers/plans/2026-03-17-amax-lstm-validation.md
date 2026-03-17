# Amax Reduction Kernel + LSTM/GRU Validation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GPU amax (absolute-max) reduction kernel to enable L-inf vector norm on GPU, and validate that LSTM/GRU training works via decomposition.

**Architecture:** Amax follows the exact same pattern as the existing `var` reduction kernel — MSL template, OpKind variant, wire protocol, Python bindings. The LSTM/GRU validation is a test-only task confirming PyTorch auto-decomposition routes gate ops through existing GPU kernels.

**Tech Stack:** Rust (Metal kernel templates, graph engine), MSL (Metal Shading Language), Python (PyO3 bindings, PyTorch backend)

---

## File Structure

**New files:**
- `python/tests/test_amax.py` — amax reduction tests
- `python/tests/test_lstm_gru.py` — LSTM/GRU decomposition validation

**Modified files (amax kernel, following var pattern):**
- `crates/core/src/kernel_templates.rs` — `amax_kernel_source(dtype)`
- `crates/core/src/graph.rs` — `OpKind::Amax` variant + match arms
- `crates/core/src/ops.rs` — `amax()` function + dtype validation
- `crates/core/src/compute.rs` — `dispatch_amax_typed` / `dispatch_amax_typed_nb`
- `crates/core/src/lazy.rs` — amax dispatch in `execute_node` + `execute_node_nb`
- `crates/core/src/serial.rs` — discriminant 83, serialization, wire↔core conversions
- `crates/wire/src/lib.rs` — `WireOpKind::Amax`, discriminant 83, read/write
- `crates/python/src/backend.rs` — `amax()` in Backend trait
- `crates/python/src/metal_backend.rs` — MetalBackend impl
- `crates/python/src/socket_backend.rs` — SocketBackend impl
- `crates/python/src/lib.rs` — PyO3 binding
- `python/applegpu_runtime/__init__.py` — export `amax`
- `python/applegpu_runtime/torch_backend.py` — L-inf branch in `linalg_vector_norm`

---

## Chunk 1: Amax Kernel

### Task 1: MSL Kernel Template

**Files:**
- Modify: `crates/core/src/kernel_templates.rs`

- [ ] **Step 1: Add `amax_kernel_source` function**

Add after the `var_kernel_source_with_correction` function (~line 630):

```rust
/// Amax reduction: max(|x|) along last dimension.
/// Used for L-infinity vector norm.
pub fn amax_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void amax{s}(
    device const {t}* input [[buffer(0)]],
    device {t}* output [[buffer(1)]],
    constant uint& total_rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {{
    if (row >= total_rows) return;
    uint offset = row * cols;
    float max_val = 0.0f;
    for (uint j = 0; j < cols; j++) {{
        float v = abs({load});
        max_val = max(max_val, v);
    }}
    output[row] = {store};
}}
"#,
        s = s, t = t,
        load = if acc { "float(input[offset + j])".to_string() } else { "input[offset + j]".to_string() },
        store = if acc { format!("{t}(max_val)") } else { "max_val".to_string() },
    )
}
```

- [ ] **Step 2: Add kernel template unit test**

Add a test at the bottom of `kernel_templates.rs`:

```rust
#[test]
fn test_amax_kernel_source_f32() {
    let src = amax_kernel_source(DType::Float32);
    assert!(src.contains("kernel void amax_f32"));
    assert!(src.contains("abs("));
    assert!(src.contains("max(max_val"));
}
```

- [ ] **Step 3: Run test**

Run: `cargo test -p applegpu-core test_amax_kernel_source_f32`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/kernel_templates.rs
git commit -m "feat: add amax (absolute max) MSL kernel template"
```

---

### Task 2: OpKind + Graph Integration

**Files:**
- Modify: `crates/core/src/graph.rs`

- [ ] **Step 1: Add OpKind::Amax variant**

Add after `ScatterAdd` (~line 160):

```rust
    // Absolute max reduction along last dim (for L-inf norm)
    Amax,
```

- [ ] **Step 2: Add kernel_name match arm**

In `kernel_name()`, add after `ScatterAdd`:

```rust
            OpKind::Amax => "amax",
```

- [ ] **Step 3: Add is_amax helper**

Add after `is_var()`:

```rust
    pub fn is_amax(&self) -> bool {
        matches!(self, OpKind::Amax)
    }
```

- [ ] **Step 4: Compile check**

Run: `cargo check -p applegpu-core`
Expected: errors in ops.rs (non-exhaustive match) — that's expected, we fix it next task.

---

### Task 3: Op Function + Dtype Validation

**Files:**
- Modify: `crates/core/src/ops.rs`

- [ ] **Step 1: Add Amax to validate_op_dtype**

In the float-only group (~line 44), add `OpKind::Amax` alongside `Var`:

```rust
        OpKind::Mean | OpKind::Var { .. } | OpKind::Amax |
```

- [ ] **Step 2: Add `amax()` function**

Add after the `var()` function:

```rust
pub fn amax(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let shape = rt.shape(input_id)?;
    let dtype = rt.dtype(input_id)?;
    validate_op_dtype(&OpKind::Amax, dtype)?;
    if shape.is_empty() {
        return Err(GpuError::InvalidTensor("amax requires at least 1D tensor".into()));
    }
    let mut out_shape = shape[..shape.len() - 1].to_vec();
    if out_shape.is_empty() { out_shape = vec![1]; }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Amax,
        inputs: vec![input_id],
        out_shape: Shape::new(out_shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}
```

- [ ] **Step 3: Compile check**

Run: `cargo check -p applegpu-core`
Expected: errors in compute.rs/lazy.rs/serial.rs — expected, we fix those next.

---

### Task 4: Compute Dispatch

**Files:**
- Modify: `crates/core/src/compute.rs`

- [ ] **Step 1: Add dispatch_amax_typed**

Note: no `resolve_kernel` entry needed — the standalone dispatch functions call `kt::amax_kernel_source` directly (same pattern as var, which bypasses resolve_kernel due to its correction parameter).

Add after `dispatch_var_typed_nb`:

```rust
    pub fn dispatch_amax_typed(
        &self, device: &Device, dtype: DType, buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<()> {
        use crate::kernel_templates as kt;
        let source = kt::amax_kernel_source(dtype);
        let s = kt::dtype_suffix(dtype);
        let func = format!("amax{s}");
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax(buf_input, buf_output, rows, cols)
    }

    pub fn dispatch_amax_typed_nb(
        &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
        buf_input: &Buffer, buf_output: &Buffer,
        rows: usize, cols: usize,
    ) -> Result<*mut std::ffi::c_void> {
        use crate::kernel_templates as kt;
        let source = kt::amax_kernel_source(dtype);
        let s = kt::dtype_suffix(dtype);
        let func = format!("amax{s}");
        let pipeline = self.get_or_create(device, &source, &func)?;
        pipeline.dispatch_softmax_nb(queue, buf_input, buf_output, rows, cols)
    }
```

---

### Task 5: Lazy Execution

**Files:**
- Modify: `crates/core/src/lazy.rs`

- [ ] **Step 1: Add amax to blocking execute_node**

Find the block `if node.op.is_sum() || node.op.is_mean() || node.op.is_var()` and extend the condition:

```rust
        if node.op.is_sum() || node.op.is_mean() || node.op.is_var() || node.op.is_amax() {
```

Inside the if-else chain, add before the final `else` (mean):

```rust
            } else if node.op.is_amax() {
                REGISTRY.dispatch_amax_typed(device, dtype, &input.buffer, &out.buffer, total_rows, cols)?;
```

- [ ] **Step 2: Add amax to non-blocking execute_node_nb**

Same pattern — extend the condition and add the dispatch branch:

```rust
        if node.op.is_sum() || node.op.is_mean() || node.op.is_var() || node.op.is_amax() {
```

```rust
            } else if node.op.is_amax() {
                return REGISTRY.dispatch_amax_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, total_rows, cols);
```

---

### Task 6: Wire Protocol + Serialization

**Files:**
- Modify: `crates/wire/src/lib.rs`
- Modify: `crates/core/src/serial.rs`

- [ ] **Step 1: Add WireOpKind::Amax to wire enum**

After `ScatterAdd` (~line 261):

```rust
    // 83: absolute max reduction
    Amax,
```

- [ ] **Step 2: Add discriminant in wire**

In `discriminant()`:

```rust
            WireOpKind::Amax => 83,
```

- [ ] **Step 3: Add write_payload in wire**

Add `WireOpKind::Amax` to the no-payload group (the `Ok(())` arm that includes `ScatterWrite | ScatterAdd`).

- [ ] **Step 4: Add read_from in wire**

```rust
            83 => Ok(WireOpKind::Amax),
```

- [ ] **Step 5: Add serial.rs discriminant**

In `op_to_discriminant()`:

```rust
        OpKind::Amax => 83,
```

- [ ] **Step 6: Add serial.rs discriminant_to_op**

```rust
        83 => Ok(OpKind::Amax),
```

- [ ] **Step 7: Add From<&OpKind> for WireOpKind**

```rust
            OpKind::Amax => WireOpKind::Amax,
```

- [ ] **Step 8: Add wire_op_to_core**

```rust
        WireOpKind::Amax => OpKind::Amax,
```

- [ ] **Step 9: Compile check**

Run: `cargo check --workspace`
Expected: PASS (only pre-existing warnings)

- [ ] **Step 10: Commit**

```bash
git add crates/core/src/graph.rs crates/core/src/ops.rs crates/core/src/compute.rs crates/core/src/lazy.rs crates/core/src/serial.rs crates/wire/src/lib.rs
git commit -m "feat: amax reduction kernel — OpKind, wire protocol, Metal dispatch"
```

---

### Task 7: Python Bindings

**Files:**
- Modify: `crates/python/src/backend.rs`
- Modify: `crates/python/src/metal_backend.rs`
- Modify: `crates/python/src/socket_backend.rs`
- Modify: `crates/python/src/lib.rs`
- Modify: `python/applegpu_runtime/__init__.py`

- [ ] **Step 1: Add to Backend trait**

In `backend.rs`, add after `var`:

```rust
    fn amax(&self, a: u64) -> BackendResult<u64>;
```

- [ ] **Step 2: Add MetalBackend impl**

In `metal_backend.rs`:

```rust
    fn amax(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::amax(&mut rt, a))
    }
```

- [ ] **Step 3: Add SocketBackend impl**

In `socket_backend.rs`:

```rust
    fn amax(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Amax)
    }
```

- [ ] **Step 4: Add PyO3 bindings**

In `lib.rs`, add GpuTensor method:

```rust
    fn amax(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.amax(self.id)) }
```

Add free function:

```rust
#[pyfunction]
fn amax(t: &GpuTensor) -> PyResult<GpuTensor> { t.amax() }
```

Register in module:

```rust
    m.add_function(wrap_pyfunction!(amax, m)?)?;
```

- [ ] **Step 5: Add Python export**

In `__init__.py`, add `amax` to the import list and `__all__`.

- [ ] **Step 6: Compile check**

Run: `cargo check --workspace`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/python/src/backend.rs crates/python/src/metal_backend.rs crates/python/src/socket_backend.rs crates/python/src/lib.rs python/applegpu_runtime/__init__.py
git commit -m "feat: amax Python bindings — Backend trait, Metal, Socket, PyO3"
```

---

### Task 8: Amax Tests

**Files:**
- Create: `python/tests/test_amax.py`

- [ ] **Step 1: Write amax tests**

```python
"""Tests for amax (absolute max) reduction kernel."""
import numpy as np
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def _init():
    gpu.init_backend()


def test_amax_1d():
    a = gpu.from_numpy(np.array([-5.0, 3.0, -1.0, 4.0], dtype=np.float32))
    result = gpu.eval(gpu.amax(a))
    out = gpu.to_list(result)
    assert abs(out[0] - 5.0) < 1e-5


def test_amax_2d():
    a = gpu.from_numpy(np.array([[1.0, -3.0], [2.0, -4.0]], dtype=np.float32))
    result = gpu.eval(gpu.amax(a))
    out = gpu.to_list(result)
    assert abs(out[0] - 3.0) < 1e-5
    assert abs(out[1] - 4.0) < 1e-5


def test_amax_3d():
    data = np.random.randn(2, 3, 4).astype(np.float32)
    a = gpu.from_numpy(data)
    result = gpu.eval(gpu.amax(a))
    out = np.array(gpu.to_list(result)).reshape(2, 3)
    expected = np.max(np.abs(data), axis=-1)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_amax_all_positive():
    a = gpu.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    result = gpu.eval(gpu.amax(a))
    out = gpu.to_list(result)
    assert abs(out[0] - 3.0) < 1e-5


def test_amax_float16():
    data = np.array([-2.0, 1.0, -3.0, 0.5], dtype=np.float16)
    a = gpu.from_numpy(data)
    result = gpu.eval(gpu.amax(a))
    out = gpu.to_list(result)
    assert abs(out[0] - 3.0) < 0.1


def test_amax_single_element():
    a = gpu.from_numpy(np.array([-7.0], dtype=np.float32))
    result = gpu.eval(gpu.amax(a))
    out = gpu.to_list(result)
    assert abs(out[0] - 7.0) < 1e-5
```

- [ ] **Step 2: Build and run tests**

Run: `uv run maturin develop && uv run pytest python/tests/test_amax.py -v`
Expected: 6 PASS

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_amax.py
git commit -m "test: amax reduction kernel — 1D/2D/3D, float16, edge cases"
```

---

### Task 9: L-inf Vector Norm on GPU

**Files:**
- Modify: `python/applegpu_runtime/torch_backend.py`

- [ ] **Step 1: Add L-inf branch to linalg_vector_norm**

Replace the `else` fallback block (~line 1471) with an `elif` for inf, keeping the general fallback.

Note: `gpu.amax` always reduces the last dim and already applies `abs()` internally, so no separate `gpu.abs()` call is needed. For non-last-dim or multi-dim reductions, fall back to CPU since amax only supports last-dim reduction.

```python
    elif ord == float('inf'):
        # L-inf: max(|x|) — amax kernel applies abs() internally
        if dim is None:
            # Global L-inf: flatten, then amax over last (only) dim
            gpu_a = _unwrap(a)
            flat = _wrap(gpu.reshape(gpu_a, [-1]))
            return _wrap(gpu.amax(_unwrap(flat)))
        elif isinstance(dim, int) and (dim == -1 or dim == len(a.shape) - 1):
            # Last-dim reduction — direct GPU path
            result = _wrap(gpu.amax(_unwrap(a)))
            if keepdim:
                result_shape = list(a.shape)
                result_shape[dim] = 1
                result = result.reshape(result_shape)
            return result
        else:
            # Non-last-dim L-inf — CPU fallback
            # TODO: GPU kernel supporting arbitrary reduction dim
            kwargs = {}
            if dtype is not None:
                kwargs['dtype'] = dtype
            return _cpu_fallback(torch.ops.aten.linalg_vector_norm.default,
                                 (a, ord, dim, keepdim), kwargs)
    else:
```

- [ ] **Step 2: Commit**

```bash
git add python/applegpu_runtime/torch_backend.py
git commit -m "feat: L-inf vector norm on GPU via amax kernel"
```

---

## Chunk 2: LSTM/GRU Validation

### Task 10: LSTM/GRU Decomposition Tests

**Files:**
- Create: `python/tests/test_lstm_gru.py`

- [ ] **Step 1: Write LSTM/GRU validation tests**

```python
"""Validate LSTM and GRU work via PyTorch decomposition on applegpu Metal backend.

PyTorch auto-decomposes nn.LSTM/nn.GRU into matmul + sigmoid + tanh + element-wise
ops, all of which have GPU kernels. This test validates the decomposition path works
end-to-end for forward and backward passes.

See: https://github.com/mooreman11/applegpu-runtime/issues/10
"""
import pytest
import torch
import torch.nn as nn
import applegpu_runtime as gpu
from applegpu_runtime.torch_backend import ApplegpuTensor, set_eager_mode


@pytest.fixture(autouse=True)
def _backend():
    gpu.init_backend()
    gpu.enable_torch_backend()
    set_eager_mode(True)
    yield
    set_eager_mode(False)


def test_lstm_forward():
    """LSTM forward pass produces valid output shapes."""
    model = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
    model = gpu.to_applegpu(model)
    x = ApplegpuTensor.from_torch(torch.randn(4, 10, 32))
    output, (h_n, c_n) = model(x)
    # Check shapes
    assert output.shape == (4, 10, 64)
    assert h_n.shape == (2, 4, 64)
    assert c_n.shape == (2, 4, 64)
    # Check values are finite
    out_cpu = output.to_torch_cpu()
    assert torch.isfinite(out_cpu).all()


def test_lstm_backward():
    """LSTM backward pass computes gradients."""
    model = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
    model = gpu.to_applegpu(model)
    x = ApplegpuTensor.from_torch(torch.randn(2, 5, 16, requires_grad=True))
    output, _ = model(x)
    loss = output.sum()
    loss.backward()
    # Check that weight gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        grad_cpu = param.grad.to_torch_cpu() if hasattr(param.grad, 'to_torch_cpu') else param.grad
        assert torch.isfinite(grad_cpu).all(), f"Non-finite gradient for {name}"


def test_gru_forward():
    """GRU forward pass produces valid output shapes."""
    model = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
    model = gpu.to_applegpu(model)
    x = ApplegpuTensor.from_torch(torch.randn(4, 10, 32))
    output, h_n = model(x)
    assert output.shape == (4, 10, 64)
    assert h_n.shape == (2, 4, 64)
    out_cpu = output.to_torch_cpu()
    assert torch.isfinite(out_cpu).all()


def test_gru_backward():
    """GRU backward pass computes gradients."""
    model = nn.GRU(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
    model = gpu.to_applegpu(model)
    x = ApplegpuTensor.from_torch(torch.randn(2, 5, 16, requires_grad=True))
    output, _ = model(x)
    loss = output.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_lstm_training_loss_decreases():
    """LSTM training loop shows loss decrease over 5 steps."""
    torch.manual_seed(42)
    model = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = ApplegpuTensor.from_torch(torch.randn(4, 5, 8))
    target = ApplegpuTensor.from_torch(torch.randn(4, 5, 16))

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        output, _ = model(x)
        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.to_torch_cpu().item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
```

- [ ] **Step 2: Build and run tests**

Run: `uv run maturin develop && uv run pytest python/tests/test_lstm_gru.py -v`
Expected: 5 PASS

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_lstm_gru.py
git commit -m "test: validate LSTM/GRU forward+backward via decomposition (#10)"
```

---

### Task 11: Update Docs + Close Issues

**Files:**
- Modify: `docs/BACKLOG.md`
- Modify: `README.md`

- [ ] **Step 1: Update BACKLOG.md**

Add under GPU Op Gaps:
```
- [x] GPU amax reduction kernel for L-inf vector norm (#23)
```

Update linalg_vector_norm entry:
```
- [x] GPU linalg_vector_norm kernel for gradient clipping (#22) — L1/L2/L-inf all on GPU
```

Update wire protocol count from 83 to 84.

- [ ] **Step 2: Update README.md**

Update op count (83 → 84), add `amax` to Reduction row, update wire protocol count.

- [ ] **Step 3: Close GitHub issues**

```bash
gh issue close 22 --comment "Implemented — L1/L2/L-inf vector norms all run on GPU. amax kernel for L-inf added."
gh issue close 23 --comment "Implemented — amax (absolute max) reduction kernel on Metal GPU."
```

- [ ] **Step 4: Comment on issue #10**

```bash
gh issue comment 10 --body "Validated: LSTM/GRU forward + backward work via PyTorch decomposition on Metal GPU. Tests added in test_lstm_gru.py. Keeping open for future fused kernel optimization."
```

- [ ] **Step 5: Commit**

```bash
git add docs/BACKLOG.md README.md
git commit -m "docs: update op counts, close #22 and #23"
```
