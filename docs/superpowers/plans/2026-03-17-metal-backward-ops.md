# Metal Backward Ops Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 6 fused Metal backward kernels to eliminate all GPU→CPU→GPU roundtrips during training.

**Architecture:** Each kernel follows the same 7-file pipeline: MSL source in kernel_templates.rs → OpKind variant in graph.rs → validation + lazy recording in ops.rs → dispatch in compute.rs → eval in lazy.rs → PyO3 binding in lib.rs → torch_backend.py handler replacement. Element-wise ops use flat `(numel, 1, 1)` grids; spatial ops (conv1d, max_pool2d) use multi-dimensional grids.

**Tech Stack:** Rust (core runtime), MSL (Metal Shading Language kernels), Swift (C ABI bridge), Python (PyO3 bindings + torch backend)

**Spec:** `docs/superpowers/specs/2026-03-17-metal-backward-ops-design.md`

---

## Chunk 1: Element-wise backward kernels (threshold, tanh, sigmoid, gelu)

These four kernels share the same structure: two input buffers + one output buffer, flat indexing, one thread per element. They differ only in the per-element formula.

### Task 1: threshold_backward kernel (Rust core)

**Files:**
- Modify: `crates/core/src/kernel_templates.rs` (add kernel source generator after line ~1512)
- Modify: `crates/core/src/graph.rs` (add OpKind variant at line ~127, kernel_name at line ~201)
- Modify: `crates/core/src/ops.rs` (add validation at line ~47, add function after line ~879)

- [ ] **Step 1: Add OpKind variant in graph.rs**

Add after the existing backward op variants (line ~126):

```rust
ThresholdBackward { threshold: f32 },
```

Add kernel_name match arm (line ~201):

```rust
OpKind::ThresholdBackward { .. } => "threshold_backward",
```

Add `is_threshold_backward()` helper method alongside existing `is_*_backward()` helpers:

```rust
pub fn is_threshold_backward(&self) -> bool {
    matches!(self, OpKind::ThresholdBackward { .. })
}
```

- [ ] **Step 2: Add dtype validation in ops.rs**

Add `OpKind::ThresholdBackward { .. }` to the float-only validation match arm at line ~47:

```rust
OpKind::SoftmaxBackward | OpKind::LayerNormBackward { .. } |
OpKind::Conv2dBackwardInput { .. } | OpKind::EmbeddingBackward |
OpKind::BatchNormBackward { .. } | OpKind::ThresholdBackward { .. } => {
```

- [ ] **Step 3: Add threshold_backward op function in ops.rs**

Add after existing backward ops (after `embedding` function around line ~879):

```rust
/// Threshold backward: grad_output * (input > threshold).
/// Used for ReLU backward (threshold=0).
pub fn threshold_backward(rt: &mut LazyRuntime, grad_output_id: u64, input_id: u64, threshold: f32) -> Result<u64> {
    let dtype = rt.dtype(grad_output_id)?;
    validate_op_dtype(&OpKind::ThresholdBackward { threshold }, dtype)?;
    let shape = rt.shape(grad_output_id)?;
    let in_shape = rt.shape(input_id)?;
    if shape != in_shape {
        return Err(GpuError::InvalidTensor(format!(
            "threshold_backward shape mismatch: grad {:?} vs input {:?}", shape, in_shape
        )));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::ThresholdBackward { threshold },
        inputs: vec![grad_output_id, input_id],
        out_shape: Shape::new(shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}
```

- [ ] **Step 4: Add MSL kernel in kernel_templates.rs**

Add after `batch_norm_backward_kernel_source` (after line ~1512):

```rust
pub fn threshold_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void threshold_backward{s}(device const {t}* grad_output [[buffer(0)]], device const {t}* input [[buffer(1)]], device {t}* grad_input [[buffer(2)]], constant uint& numel [[buffer(3)]], constant float& threshold [[buffer(4)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    float inp = {load_input};
    float go = {load_grad};
    float result = (inp > threshold) ? go : 0.0f;
    grad_input[id] = {store};
}}
"#,
        t = t, s = s,
        load_input = if acc { "float(input[id])".to_string() } else { "input[id]".to_string() },
        load_grad = if acc { "float(grad_output[id])".to_string() } else { "grad_output[id]".to_string() },
        store = if acc { format!("{}(result)", t) } else { "result".to_string() },
    )
}
```

- [ ] **Step 5: Compile check**

Run: `cargo build -p applegpu-core 2>&1 | tail -5`
Expected: Successful build (warnings OK)

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/kernel_templates.rs crates/core/src/graph.rs crates/core/src/ops.rs
git commit -m "feat: threshold_backward kernel source, OpKind, and ops function"
```

### Task 2: threshold_backward dispatch and eval (Rust core)

**Files:**
- Modify: `crates/core/src/compute.rs` (add resolve_kernel match + dispatch methods)
- Modify: `crates/core/src/lazy.rs` (add eval paths)

- [ ] **Step 1: Add resolve_kernel match in compute.rs**

Add after the existing backward kernel mappings (line ~1485):

```rust
"threshold_backward" => kt::threshold_backward_kernel_source(dtype),
```

- [ ] **Step 2: Add ComputePipeline dispatch methods in compute.rs**

Add a generic element-wise-binary dispatch. threshold_backward dispatches as a 1D grid of `numel` threads with an extra scalar constant buffer.

Find where `dispatch_softmax_backward` is defined on the `ComputePipeline` struct (around line 500) and add nearby:

```rust
pub fn dispatch_threshold_backward(
    &self, buf_grad_output: &Buffer, buf_input: &Buffer, buf_grad_input: &Buffer,
    numel: usize, threshold: f32,
) -> Result<()> {
    let result = unsafe {
        ffi::gpu_bridge_compute_generic_3buf_1scalar(
            self.handle,
            buf_grad_output.raw_handle() as *const _,
            buf_input.raw_handle() as *const _,
            buf_grad_input.raw_handle(),
            numel as u32,
            threshold,
        )
    };
    if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("threshold_backward dispatch failed".to_string())) }
}
```

Wait — the existing pattern uses typed dispatch through the KernelRegistry, not direct FFI calls for backward ops. Let me check how the typed dispatch works for element-wise ops. Looking at the pattern more carefully:

Actually, the typed backward dispatches use `pipeline.dispatch_3d()` which is a generic dispatch method. Let me use that pattern:

```rust
pub fn dispatch_threshold_backward_typed(
    &self, device: &Device, dtype: DType,
    buf_grad_output: &Buffer, buf_input: &Buffer, buf_grad_input: &Buffer,
    numel: usize, threshold: f32,
) -> Result<()> {
    let (source, func) = Self::resolve_kernel("threshold_backward", dtype);
    let pipeline = self.get_or_create(device, &source, &func)?;
    pipeline.dispatch_3d(
        &[buf_grad_output, buf_input],
        buf_grad_input,
        &[numel as u32],
        &[threshold],
        (numel as u32, 1, 1),
    )
}
```

Add the non-blocking variant:

```rust
pub fn dispatch_threshold_backward_typed_nb(
    &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
    buf_grad_output: &Buffer, buf_input: &Buffer, buf_grad_input: &Buffer,
    numel: usize, threshold: f32,
) -> Result<*mut std::ffi::c_void> {
    let (source, func) = Self::resolve_kernel("threshold_backward", dtype);
    let pipeline = self.get_or_create(device, &source, &func)?;
    pipeline.dispatch_3d_nb(
        queue,
        &[buf_grad_output, buf_input],
        buf_grad_input,
        &[numel as u32],
        &[threshold],
        (numel as u32, 1, 1),
    )
}
```

- [ ] **Step 3: Add eval path in lazy.rs (blocking)**

Add after existing backward op eval checks (around line ~400):

```rust
if let OpKind::ThresholdBackward { threshold } = node.op {
    let out_buf = self.pool.acquire(device, out_size)?;
    let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
    let grad_output = self.get_tensor(node.inputs[0])?;
    let input = self.get_tensor(node.inputs[1])?;
    let numel: usize = node.out_shape.dims().iter().product();
    REGISTRY.dispatch_threshold_backward_typed(device, dtype, &grad_output.buffer, &input.buffer, &out.buffer, numel, threshold)?;
    return Ok(out);
}
```

- [ ] **Step 4: Add eval path in lazy.rs (non-blocking)**

Add in the non-blocking eval function (around line ~1210):

```rust
if let OpKind::ThresholdBackward { threshold } = node.op {
    let grad_output = self.get_tensor(node.inputs[0])?;
    let input = self.get_tensor(node.inputs[1])?;
    let numel: usize = node.out_shape.dims().iter().product();
    return REGISTRY.dispatch_threshold_backward_typed_nb(device, dtype, queue, &grad_output.buffer, &input.buffer, &out.buffer, numel, threshold);
}
```

- [ ] **Step 5: Compile check**

Run: `cargo build -p applegpu-core 2>&1 | tail -5`
Expected: Successful build

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/compute.rs crates/core/src/lazy.rs
git commit -m "feat: threshold_backward dispatch and lazy eval paths"
```

### Task 3: threshold_backward Python bindings

**Files:**
- Modify: `crates/python/src/backend.rs` (add trait method)
- Modify: `crates/python/src/metal_backend.rs` (add trait impl)
- Modify: `crates/python/src/lib.rs` (add pyfunction + module registration)

- [ ] **Step 1: Add backend trait method in backend.rs**

Add after existing backward trait methods (around line ~160):

```rust
fn threshold_backward(&self, grad_output: u64, input: u64, threshold: f32) -> BackendResult<u64>;
```

- [ ] **Step 2: Add metal_backend implementation**

Add after existing backward implementations (around line ~527):

```rust
fn threshold_backward(&self, grad_output: u64, input: u64, threshold: f32) -> BackendResult<u64> {
    let mut rt = self.rt.lock().map_err(|_| BackendError::LockFailed)?;
    Ok(crate::ops::threshold_backward(&mut rt, grad_output, input, threshold)?)
}
```

Also add the same method to `SocketBackend` impl in `crates/python/src/socket_backend.rs` (after existing backward stubs around line ~1160):

```rust
fn threshold_backward(&self, _grad_output: u64, _input: u64, _threshold: f32) -> BackendResult<u64> {
    Err(BackendError::Unsupported("threshold_backward not supported over socket".to_string()))
}
```

- [ ] **Step 3: Add PyO3 function in lib.rs and update `__init__.py`**

Add after existing backward pyfunction bindings (around line ~1156):

```rust
#[pyfunction]
fn threshold_backward(grad_output: &GpuTensor, input: &GpuTensor, threshold: f32) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.threshold_backward(grad_output.id, input.id, threshold))
}
```

Register in the `#[pymodule]` block (around line ~1269):

```rust
m.add_function(wrap_pyfunction!(threshold_backward, m)?)?;
```

Also update `python/applegpu_runtime/__init__.py`:
- Add `threshold_backward` to the import block (around line ~3-107)
- Add `threshold_backward` to the `__all__` list (around line ~229-337)

- [ ] **Step 4: Build Python extension**

Run: `uv run maturin develop --release 2>&1 | tail -3`
Expected: `Installed applegpu_runtime-0.8.0`

- [ ] **Step 5: Commit**

```bash
git add crates/python/src/backend.rs crates/python/src/metal_backend.rs crates/python/src/lib.rs
git commit -m "feat: threshold_backward Python bindings"
```

### Task 4: threshold_backward tests and torch_backend integration

**Files:**
- Modify: `python/applegpu_runtime/torch_backend.py` (replace CPU fallback around line ~442)
- Create: `python/tests/test_backward_ops.py`

- [ ] **Step 1: Write failing Rust test in ops.rs**

Add to the `#[cfg(test)]` module at the bottom of ops.rs:

```rust
#[test]
fn test_threshold_backward() {
    let mut rt = test_runtime();
    // grad_output = [1.0, 2.0, 3.0, 4.0]
    let grad_id = rt.register_data(&[1.0f32, 2.0, 3.0, 4.0], vec![4], DType::Float32);
    // input = [-1.0, 0.5, -0.5, 2.0]
    let input_id = rt.register_data(&[-1.0f32, 0.5, -0.5, 2.0], vec![4], DType::Float32);
    let out_id = threshold_backward(&mut rt, grad_id, input_id, 0.0).unwrap();
    rt.eval_all().unwrap();
    let data = rt.read_f32(out_id).unwrap();
    // threshold=0: input > 0 passes grad, else 0
    // [-1 > 0 = false, 0.5 > 0 = true, -0.5 > 0 = false, 2 > 0 = true]
    // result = [0.0, 2.0, 0.0, 4.0]
    assert_eq!(data, vec![0.0, 2.0, 0.0, 4.0]);
}
```

- [ ] **Step 2: Run Rust test**

Run: `cargo test -p applegpu-core test_threshold_backward -- --nocapture 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 3: Write Python test**

Create `python/tests/test_backward_ops.py`:

```python
"""Tests for Metal backward ops — verifies GPU path matches PyTorch CPU reference."""
import numpy as np
import pytest
import torch
import applegpu_runtime as gpu


@pytest.fixture(scope="module", autouse=True)
def init():
    gpu.init_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)


def _to_numpy(t):
    t.eval()
    return np.array(t.to_list()).reshape(t.shape)


class TestThresholdBackward:
    def test_basic(self):
        grad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        inp = np.array([-1.0, 0.5, -0.5, 2.0], dtype=np.float32)
        result = _to_numpy(gpu.threshold_backward(gpu.from_numpy(grad), gpu.from_numpy(inp), 0.0))
        expected = grad * (inp > 0).astype(np.float32)
        np.testing.assert_allclose(result, expected)

    def test_3d(self):
        np.random.seed(42)
        grad = np.random.randn(2, 3, 4).astype(np.float32)
        inp = np.random.randn(2, 3, 4).astype(np.float32)
        result = _to_numpy(gpu.threshold_backward(gpu.from_numpy(grad), gpu.from_numpy(inp), 0.0))
        expected = grad * (inp > 0).astype(np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_nonzero_threshold(self):
        grad = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        inp = np.array([-1.0, 0.5, 1.5, 2.0], dtype=np.float32)
        result = _to_numpy(gpu.threshold_backward(gpu.from_numpy(grad), gpu.from_numpy(inp), 1.0))
        expected = grad * (inp > 1.0).astype(np.float32)
        np.testing.assert_allclose(result, expected)
```

- [ ] **Step 4: Run Python test**

Run: `uv run pytest python/tests/test_backward_ops.py::TestThresholdBackward -v`
Expected: 3 passed

- [ ] **Step 5: Replace CPU fallback in torch_backend.py**

Replace the CPU fallback at line ~442:

```python
@register_op(torch.ops.aten.threshold_backward.default)
def _op_threshold_backward(grad_output, self_tensor, threshold):
    """Backward for relu: grad * (self > threshold) — Metal GPU."""
    return _wrap(gpu.threshold_backward(_unwrap(grad_output), _unwrap(self_tensor), float(threshold)),
                 torch_dtype=grad_output.dtype, requires_grad=grad_output.requires_grad)
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest python/tests/ -v --ignore=python/tests/test_torch_backend.py 2>&1 | tail -5`
Expected: All tests pass, no new threshold_backward CPU fallback warnings

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/ops.rs python/tests/test_backward_ops.py python/applegpu_runtime/torch_backend.py
git commit -m "feat: threshold_backward on Metal GPU, tests, torch_backend integration"
```

### Task 5: tanh_backward and sigmoid_backward kernels (Rust core)

**Files:**
- Modify: `crates/core/src/kernel_templates.rs`
- Modify: `crates/core/src/graph.rs`
- Modify: `crates/core/src/ops.rs`
- Modify: `crates/core/src/compute.rs`
- Modify: `crates/core/src/lazy.rs`

- [ ] **Step 1: Add OpKind variants in graph.rs**

```rust
TanhBackward,
SigmoidBackward,
```

Add kernel_name match arms:

```rust
OpKind::TanhBackward => "tanh_backward",
OpKind::SigmoidBackward => "sigmoid_backward",
```

Add `is_*` helpers:

```rust
pub fn is_tanh_backward(&self) -> bool { matches!(self, OpKind::TanhBackward) }
pub fn is_sigmoid_backward(&self) -> bool { matches!(self, OpKind::SigmoidBackward) }
```

- [ ] **Step 2: Add dtype validation in ops.rs**

Add `OpKind::TanhBackward | OpKind::SigmoidBackward` to the float-only match arm.

- [ ] **Step 3: Add ops functions in ops.rs**

```rust
/// Tanh backward: grad_output * (1 - output^2). Receives forward output.
pub fn tanh_backward(rt: &mut LazyRuntime, grad_output_id: u64, output_id: u64) -> Result<u64> {
    let dtype = rt.dtype(grad_output_id)?;
    validate_op_dtype(&OpKind::TanhBackward, dtype)?;
    let shape = rt.shape(grad_output_id)?;
    if shape != rt.shape(output_id)? {
        return Err(GpuError::InvalidTensor("tanh_backward shape mismatch".to_string()));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id, op: OpKind::TanhBackward, inputs: vec![grad_output_id, output_id],
        out_shape: Shape::new(shape)?, out_dtype: dtype, container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Sigmoid backward: grad_output * output * (1 - output). Receives forward output.
pub fn sigmoid_backward(rt: &mut LazyRuntime, grad_output_id: u64, output_id: u64) -> Result<u64> {
    let dtype = rt.dtype(grad_output_id)?;
    validate_op_dtype(&OpKind::SigmoidBackward, dtype)?;
    let shape = rt.shape(grad_output_id)?;
    if shape != rt.shape(output_id)? {
        return Err(GpuError::InvalidTensor("sigmoid_backward shape mismatch".to_string()));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id, op: OpKind::SigmoidBackward, inputs: vec![grad_output_id, output_id],
        out_shape: Shape::new(shape)?, out_dtype: dtype, container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}
```

- [ ] **Step 4: Add MSL kernels in kernel_templates.rs**

```rust
pub fn tanh_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void tanh_backward{s}(device const {t}* grad_output [[buffer(0)]], device const {t}* output [[buffer(1)]], device {t}* grad_input [[buffer(2)]], constant uint& numel [[buffer(3)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    float out_val = {load_out};
    float go = {load_grad};
    float result = go * (1.0f - out_val * out_val);
    grad_input[id] = {store};
}}
"#,
        t = t, s = s,
        load_out = if acc { "float(output[id])" } else { "output[id]" },
        load_grad = if acc { "float(grad_output[id])" } else { "grad_output[id]" },
        store = if acc { format!("{}(result)", t) } else { "result".to_string() },
    )
}

pub fn sigmoid_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_backward{s}(device const {t}* grad_output [[buffer(0)]], device const {t}* output [[buffer(1)]], device {t}* grad_input [[buffer(2)]], constant uint& numel [[buffer(3)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    float out_val = {load_out};
    float go = {load_grad};
    float result = go * out_val * (1.0f - out_val);
    grad_input[id] = {store};
}}
"#,
        t = t, s = s,
        load_out = if acc { "float(output[id])" } else { "output[id]" },
        load_grad = if acc { "float(grad_output[id])" } else { "grad_output[id]" },
        store = if acc { format!("{}(result)", t) } else { "result".to_string() },
    )
}
```

- [ ] **Step 5: Add dispatch in compute.rs**

Add resolve_kernel match arms:

```rust
"tanh_backward" => kt::tanh_backward_kernel_source(dtype),
"sigmoid_backward" => kt::sigmoid_backward_kernel_source(dtype),
```

Add typed dispatch methods (both blocking and non-blocking). These are simpler than threshold — no scalar parameter, just 2 input buffers + 1 output buffer + numel:

```rust
pub fn dispatch_tanh_backward_typed(
    &self, device: &Device, dtype: DType,
    buf_grad_output: &Buffer, buf_output: &Buffer, buf_grad_input: &Buffer, numel: usize,
) -> Result<()> {
    let (source, func) = Self::resolve_kernel("tanh_backward", dtype);
    let pipeline = self.get_or_create(device, &source, &func)?;
    pipeline.dispatch_3d(&[buf_grad_output, buf_output], buf_grad_input, &[numel as u32], &[], (numel as u32, 1, 1))
}

pub fn dispatch_tanh_backward_typed_nb(
    &self, device: &Device, dtype: DType, queue: *mut std::ffi::c_void,
    buf_grad_output: &Buffer, buf_output: &Buffer, buf_grad_input: &Buffer, numel: usize,
) -> Result<*mut std::ffi::c_void> {
    let (source, func) = Self::resolve_kernel("tanh_backward", dtype);
    let pipeline = self.get_or_create(device, &source, &func)?;
    pipeline.dispatch_3d_nb(queue, &[buf_grad_output, buf_output], buf_grad_input, &[numel as u32], &[], (numel as u32, 1, 1))
}
```

Repeat identically for `sigmoid_backward` (same signature pattern).

- [ ] **Step 6: Add eval paths in lazy.rs**

Blocking (around line ~400):

```rust
if node.op.is_tanh_backward() {
    let out_buf = self.pool.acquire(device, out_size)?;
    let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
    let grad_output = self.get_tensor(node.inputs[0])?;
    let output = self.get_tensor(node.inputs[1])?;
    let numel: usize = node.out_shape.dims().iter().product();
    REGISTRY.dispatch_tanh_backward_typed(device, dtype, &grad_output.buffer, &output.buffer, &out.buffer, numel)?;
    return Ok(out);
}

if node.op.is_sigmoid_backward() {
    let out_buf = self.pool.acquire(device, out_size)?;
    let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
    let grad_output = self.get_tensor(node.inputs[0])?;
    let output = self.get_tensor(node.inputs[1])?;
    let numel: usize = node.out_shape.dims().iter().product();
    REGISTRY.dispatch_sigmoid_backward_typed(device, dtype, &grad_output.buffer, &output.buffer, &out.buffer, numel)?;
    return Ok(out);
}
```

Non-blocking (around line ~1210):

```rust
if node.op.is_tanh_backward() {
    let grad_output = self.get_tensor(node.inputs[0])?;
    let output = self.get_tensor(node.inputs[1])?;
    let numel: usize = node.out_shape.dims().iter().product();
    return REGISTRY.dispatch_tanh_backward_typed_nb(device, dtype, queue, &grad_output.buffer, &output.buffer, &out.buffer, numel);
}

if node.op.is_sigmoid_backward() {
    let grad_output = self.get_tensor(node.inputs[0])?;
    let output = self.get_tensor(node.inputs[1])?;
    let numel: usize = node.out_shape.dims().iter().product();
    return REGISTRY.dispatch_sigmoid_backward_typed_nb(device, dtype, queue, &grad_output.buffer, &output.buffer, &out.buffer, numel);
}
```

- [ ] **Step 7: Compile check**

Run: `cargo build -p applegpu-core 2>&1 | tail -5`
Expected: Successful build

- [ ] **Step 8: Commit**

```bash
git add crates/core/src/kernel_templates.rs crates/core/src/graph.rs crates/core/src/ops.rs crates/core/src/compute.rs crates/core/src/lazy.rs
git commit -m "feat: tanh_backward and sigmoid_backward Metal kernels"
```

### Task 6: tanh_backward and sigmoid_backward bindings, tests, torch integration

**Files:**
- Modify: `crates/python/src/backend.rs`
- Modify: `crates/python/src/metal_backend.rs`
- Modify: `crates/python/src/lib.rs`
- Modify: `python/applegpu_runtime/torch_backend.py`
- Modify: `python/tests/test_backward_ops.py`

- [ ] **Step 1: Add backend trait methods**

```rust
fn tanh_backward(&self, grad_output: u64, output: u64) -> BackendResult<u64>;
fn sigmoid_backward(&self, grad_output: u64, output: u64) -> BackendResult<u64>;
```

- [ ] **Step 2: Add metal_backend implementations**

```rust
fn tanh_backward(&self, grad_output: u64, output: u64) -> BackendResult<u64> {
    let mut rt = self.rt.lock().map_err(|_| BackendError::LockFailed)?;
    Ok(crate::ops::tanh_backward(&mut rt, grad_output, output)?)
}

fn sigmoid_backward(&self, grad_output: u64, output: u64) -> BackendResult<u64> {
    let mut rt = self.rt.lock().map_err(|_| BackendError::LockFailed)?;
    Ok(crate::ops::sigmoid_backward(&mut rt, grad_output, output)?)
}
```

Add socket_backend stubs in `crates/python/src/socket_backend.rs`:

```rust
fn tanh_backward(&self, _grad_output: u64, _output: u64) -> BackendResult<u64> {
    Err(BackendError::Unsupported("tanh_backward not supported over socket".to_string()))
}
fn sigmoid_backward(&self, _grad_output: u64, _output: u64) -> BackendResult<u64> {
    Err(BackendError::Unsupported("sigmoid_backward not supported over socket".to_string()))
}
```

- [ ] **Step 3: Add PyO3 functions and register**

```rust
#[pyfunction]
fn tanh_backward(grad_output: &GpuTensor, output: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.tanh_backward(grad_output.id, output.id))
}

#[pyfunction]
fn sigmoid_backward(grad_output: &GpuTensor, output: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.sigmoid_backward(grad_output.id, output.id))
}
```

Register both in `#[pymodule]`.

Also update `python/applegpu_runtime/__init__.py`: add `tanh_backward` and `sigmoid_backward` to both the import block and `__all__`.

- [ ] **Step 4: Build and run Rust tests**

```rust
#[test]
fn test_tanh_backward() {
    let mut rt = test_runtime();
    let grad_id = rt.register_data(&[1.0f32, 2.0, 3.0], vec![3], DType::Float32);
    let out_id = rt.register_data(&[0.5f32, -0.3, 0.9], vec![3], DType::Float32);
    let result_id = tanh_backward(&mut rt, grad_id, out_id).unwrap();
    rt.eval_all().unwrap();
    let data = rt.read_f32(result_id).unwrap();
    // grad * (1 - output^2): [1*(1-0.25), 2*(1-0.09), 3*(1-0.81)] = [0.75, 1.82, 0.57]
    assert!((data[0] - 0.75).abs() < 1e-5);
    assert!((data[1] - 1.82).abs() < 1e-5);
    assert!((data[2] - 0.57).abs() < 1e-5);
}

#[test]
fn test_sigmoid_backward() {
    let mut rt = test_runtime();
    let grad_id = rt.register_data(&[1.0f32, 2.0, 3.0], vec![3], DType::Float32);
    let out_id = rt.register_data(&[0.5f32, 0.3, 0.8], vec![3], DType::Float32);
    let result_id = sigmoid_backward(&mut rt, grad_id, out_id).unwrap();
    rt.eval_all().unwrap();
    let data = rt.read_f32(result_id).unwrap();
    // grad * output * (1 - output): [1*0.5*0.5, 2*0.3*0.7, 3*0.8*0.2] = [0.25, 0.42, 0.48]
    assert!((data[0] - 0.25).abs() < 1e-5);
    assert!((data[1] - 0.42).abs() < 1e-5);
    assert!((data[2] - 0.48).abs() < 1e-5);
}
```

Run: `cargo test -p applegpu-core test_tanh_backward test_sigmoid_backward -- --nocapture`

- [ ] **Step 5: Build Python extension and add Python tests**

Run: `uv run maturin develop --release`

Add to `test_backward_ops.py`:

```python
class TestTanhBackward:
    def test_basic(self):
        grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        output = np.array([0.5, -0.3, 0.9], dtype=np.float32)
        result = _to_numpy(gpu.tanh_backward(gpu.from_numpy(grad), gpu.from_numpy(output)))
        expected = grad * (1 - output ** 2)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_3d(self):
        np.random.seed(42)
        grad = np.random.randn(2, 3, 4).astype(np.float32)
        output = np.tanh(np.random.randn(2, 3, 4)).astype(np.float32)
        result = _to_numpy(gpu.tanh_backward(gpu.from_numpy(grad), gpu.from_numpy(output)))
        expected = grad * (1 - output ** 2)
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestSigmoidBackward:
    def test_basic(self):
        grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        output = np.array([0.5, 0.3, 0.8], dtype=np.float32)
        result = _to_numpy(gpu.sigmoid_backward(gpu.from_numpy(grad), gpu.from_numpy(output)))
        expected = grad * output * (1 - output)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_3d(self):
        np.random.seed(42)
        grad = np.random.randn(2, 3, 4).astype(np.float32)
        output = 1 / (1 + np.exp(-np.random.randn(2, 3, 4))).astype(np.float32)
        result = _to_numpy(gpu.sigmoid_backward(gpu.from_numpy(grad), gpu.from_numpy(output)))
        expected = grad * output * (1 - output)
        np.testing.assert_allclose(result, expected, atol=1e-5)
```

Run: `uv run pytest python/tests/test_backward_ops.py -v`
Expected: All tests pass

- [ ] **Step 6: Replace torch_backend.py CPU fallbacks**

Replace tanh_backward (line ~469):

```python
@register_op(torch.ops.aten.tanh_backward.default)
def _op_tanh_backward(grad_output, output):
    """Backward for tanh: grad * (1 - output^2) — Metal GPU."""
    return _wrap(gpu.tanh_backward(_unwrap(grad_output), _unwrap(output)),
                 torch_dtype=grad_output.dtype, requires_grad=grad_output.requires_grad)
```

Replace sigmoid_backward (line ~478):

```python
@register_op(torch.ops.aten.sigmoid_backward.default)
def _op_sigmoid_backward(grad_output, output):
    """Backward for sigmoid: grad * output * (1 - output) — Metal GPU."""
    return _wrap(gpu.sigmoid_backward(_unwrap(grad_output), _unwrap(output)),
                 torch_dtype=grad_output.dtype, requires_grad=grad_output.requires_grad)
```

- [ ] **Step 7: Run full test suite and commit**

Run: `uv run pytest python/tests/ -v --ignore=python/tests/test_torch_backend.py 2>&1 | tail -5`
Expected: All pass

```bash
git add crates/python/src/ python/tests/test_backward_ops.py python/applegpu_runtime/torch_backend.py
git commit -m "feat: tanh_backward and sigmoid_backward on Metal GPU"
```

### Task 7: gelu_backward kernel (Rust core + bindings + tests)

**Files:** Same 7-file pattern as above.

- [ ] **Step 1: Add OpKind, kernel_name, is_* helper in graph.rs**

```rust
GeluBackward,
```

```rust
OpKind::GeluBackward => "gelu_backward",
```

- [ ] **Step 2: Add dtype validation and ops function in ops.rs**

Add `OpKind::GeluBackward` to float-only match. Add function:

```rust
/// GELU backward (tanh approximation). Receives original input.
pub fn gelu_backward(rt: &mut LazyRuntime, grad_output_id: u64, input_id: u64) -> Result<u64> {
    let dtype = rt.dtype(grad_output_id)?;
    validate_op_dtype(&OpKind::GeluBackward, dtype)?;
    let shape = rt.shape(grad_output_id)?;
    if shape != rt.shape(input_id)? {
        return Err(GpuError::InvalidTensor("gelu_backward shape mismatch".to_string()));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id, op: OpKind::GeluBackward, inputs: vec![grad_output_id, input_id],
        out_shape: Shape::new(shape)?, out_dtype: dtype, container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}
```

- [ ] **Step 3: Add MSL kernel in kernel_templates.rs**

```rust
pub fn gelu_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void gelu_backward{s}(device const {t}* grad_output [[buffer(0)]], device const {t}* input [[buffer(1)]], device {t}* grad_input [[buffer(2)]], constant uint& numel [[buffer(3)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    float x = {load_input};
    float go = {load_grad};
    float sqrt_2_pi = 0.7978845608f;
    float a = sqrt_2_pi * (x + 0.044715f * x * x * x);
    a = clamp(a, -10.0f, 10.0f);
    float tanh_a = tanh(a);
    float da = sqrt_2_pi * (1.0f + 3.0f * 0.044715f * x * x);
    float gelu_grad = 0.5f * (1.0f + tanh_a) + 0.5f * x * (1.0f - tanh_a * tanh_a) * da;
    float result = go * gelu_grad;
    grad_input[id] = {store};
}}
"#,
        t = t, s = s,
        load_input = if acc { "float(input[id])" } else { "input[id]" },
        load_grad = if acc { "float(grad_output[id])" } else { "grad_output[id]" },
        store = if acc { format!("{}(result)", t) } else { "result".to_string() },
    )
}
```

- [ ] **Step 4: Add dispatch in compute.rs and eval in lazy.rs**

Follow the same pattern as tanh_backward (2 input buffers + 1 output + numel, `(numel, 1, 1)` grid). Add resolve_kernel match, blocking/non-blocking dispatch, blocking/non-blocking eval.

- [ ] **Step 5: Add Python bindings**

backend.rs, metal_backend.rs, socket_backend.rs (stub), lib.rs, `__init__.py` — same pattern as tanh_backward.

- [ ] **Step 6: Build, write tests, verify**

Rust test:

```rust
#[test]
fn test_gelu_backward() {
    let mut rt = test_runtime();
    let grad_id = rt.register_data(&[1.0f32, 1.0, 1.0], vec![3], DType::Float32);
    let input_id = rt.register_data(&[0.0f32, 1.0, -1.0], vec![3], DType::Float32);
    let result_id = gelu_backward(&mut rt, grad_id, input_id).unwrap();
    rt.eval_all().unwrap();
    let data = rt.read_f32(result_id).unwrap();
    // gelu'(0) = 0.5, gelu'(1) ≈ 1.083, gelu'(-1) ≈ -0.083
    assert!((data[0] - 0.5).abs() < 0.01);
    assert!((data[1] - 1.083).abs() < 0.02);
    assert!((data[2] - (-0.083)).abs() < 0.02);
}
```

Python test comparing against PyTorch CPU reference:

```python
class TestGeluBackward:
    def test_matches_pytorch(self):
        np.random.seed(42)
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        grad_np = np.random.randn(2, 3, 4).astype(np.float32)
        # PyTorch reference
        x_t = torch.tensor(x_np, requires_grad=True)
        y = torch.nn.functional.gelu(x_t, approximate="tanh")
        y.backward(torch.tensor(grad_np))
        expected = x_t.grad.numpy()
        # Our GPU
        result = _to_numpy(gpu.gelu_backward(gpu.from_numpy(grad_np), gpu.from_numpy(x_np)))
        np.testing.assert_allclose(result, expected, atol=1e-4)
```

- [ ] **Step 7: Replace torch_backend.py CPU fallback and commit**

```python
@register_op(torch.ops.aten.gelu_backward.default)
def _op_gelu_backward(grad_output, self_tensor, approximate="none"):
    """Backward for gelu (tanh approximation) — Metal GPU.
    Falls back to CPU for exact mode (approximate='none') since forward kernel
    only supports tanh approximation. See GitHub issue #18.
    """
    if approximate == "none":
        # Exact GELU backward not implemented — CPU fallback
        return _cpu_fallback(torch.ops.aten.gelu_backward.default,
                             (grad_output, self_tensor, approximate), {})
    return _wrap(gpu.gelu_backward(_unwrap(grad_output), _unwrap(self_tensor)),
                 torch_dtype=grad_output.dtype, requires_grad=grad_output.requires_grad)
```

```bash
git add crates/core/src/ crates/python/src/ python/tests/test_backward_ops.py python/applegpu_runtime/
git commit -m "feat: gelu_backward on Metal GPU"
```

---

## Chunk 2: Spatial backward kernels (conv1d, max_pool2d)

### Task 8: conv1d_backward_input kernel

**Files:** Same 7-file pattern.

- [ ] **Step 1: Add OpKind in graph.rs**

```rust
Conv1dBackwardInput { stride: usize, padding: usize },
```

kernel_name: `"conv1d_backward_input"`

- [ ] **Step 2: Add ops function in ops.rs**

```rust
/// Conv1d backward w.r.t. input. Output shape = original input shape [batch, in_channels, in_len].
pub fn conv1d_backward_input(
    rt: &mut LazyRuntime, grad_output_id: u64, weight_id: u64,
    in_channels: usize, in_len: usize, stride: usize, padding: usize,
) -> Result<u64> {
    let dtype = rt.dtype(grad_output_id)?;
    validate_op_dtype(&OpKind::Conv1dBackwardInput { stride, padding }, dtype)?;
    let go_shape = rt.shape(grad_output_id)?;  // [batch, out_channels, out_len]
    let w_shape = rt.shape(weight_id)?;         // [out_channels, in_channels, kernel_len]
    let batch = go_shape[0];
    let out_shape = vec![batch, in_channels, in_len];
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Conv1dBackwardInput { stride, padding },
        inputs: vec![grad_output_id, weight_id],
        out_shape: Shape::new(out_shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}
```

- [ ] **Step 3: Add MSL kernel in kernel_templates.rs**

Mirrors `conv2d_backward_input` but for 1D:

```rust
pub fn conv1d_backward_input_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    let acc = needs_float_acc(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void conv1d_backward_input{s}(
    device const {t}* grad_output [[buffer(0)]],
    device const {t}* weight [[buffer(1)]],
    device {t}* grad_input [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& in_channels [[buffer(4)]],
    constant uint& out_channels [[buffer(5)]],
    constant uint& in_len [[buffer(6)]],
    constant uint& out_len [[buffer(7)]],
    constant uint& kernel_len [[buffer(8)]],
    constant uint& stride [[buffer(9)]],
    constant uint& pad [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint il = gid.x;
    uint ic = gid.y;
    uint b = gid.z;
    if (il >= in_len || ic >= in_channels || b >= batch) return;

    float sum = 0.0f;
    for (uint oc = 0; oc < out_channels; oc++) {{
        for (uint k = 0; k < kernel_len; k++) {{
            int pos = int(il) + int(pad) - int(k);
            if (pos < 0 || pos % int(stride) != 0) continue;
            uint ol = uint(pos) / stride;
            if (ol >= out_len) continue;
            uint go_idx = b * out_channels * out_len + oc * out_len + ol;
            uint w_idx = oc * in_channels * kernel_len + ic * kernel_len + k;
            sum += {load_go} * {load_w};
        }}
    }}
    uint gi_idx = b * in_channels * in_len + ic * in_len + il;
    grad_input[gi_idx] = {store};
}}
"#,
        t = t, s = s,
        load_go = if acc { "float(grad_output[go_idx])" } else { "grad_output[go_idx]" },
        load_w = if acc { "float(weight[w_idx])" } else { "weight[w_idx]" },
        store = if acc { format!("{}(sum)", t) } else { "sum".to_string() },
    )
}
```

- [ ] **Step 4: Add dispatch in compute.rs**

The dispatch needs many constant parameters. Use `dispatch_3d` with uint constants and the 3D grid `(in_len, in_channels, batch)`:

```rust
pub fn dispatch_conv1d_backward_input_typed(
    &self, device: &Device, dtype: DType,
    buf_grad_output: &Buffer, buf_weight: &Buffer, buf_grad_input: &Buffer,
    batch: usize, in_channels: usize, out_channels: usize,
    in_len: usize, out_len: usize, kernel_len: usize,
    stride: usize, padding: usize,
) -> Result<()> {
    let (source, func) = Self::resolve_kernel("conv1d_backward_input", dtype);
    let pipeline = self.get_or_create(device, &source, &func)?;
    pipeline.dispatch_3d(
        &[buf_grad_output, buf_weight],
        buf_grad_input,
        &[batch as u32, in_channels as u32, out_channels as u32,
          in_len as u32, out_len as u32, kernel_len as u32,
          stride as u32, padding as u32],
        &[],
        (in_len as u32, in_channels as u32, batch as u32),
    )
}
```

Add non-blocking variant.

- [ ] **Step 5: Add eval in lazy.rs (blocking AND non-blocking)**

Both blocking and non-blocking eval paths. Extract shape parameters from input tensors and OpKind config. The blocking path calls `dispatch_conv1d_backward_input_typed()`, the non-blocking path calls `dispatch_conv1d_backward_input_typed_nb()` and returns the command buffer handle.

- [ ] **Step 6: Add Python bindings, tests, torch_backend integration**

Add backend trait method, metal_backend impl, socket_backend stub, PyO3 function, `__init__.py` export — same pattern as previous tasks.

Update the `_op_conv_backward` handler in torch_backend.py. In the `is_conv1d` and `output_mask[0]` branch, use `gpu.conv1d_backward_input()` only when `groups == 1`:

```python
if is_conv1d:
    if groups == 1:
        # Conv1d grad_input on Metal GPU
        go_gpu = _unwrap(grad_output)
        w_gpu = _unwrap(weight)
        in_shape_actual = input.shape if isinstance(input, ApplegpuTensor) else input.shape
        grad_input = _wrap(gpu.conv1d_backward_input(
            go_gpu, w_gpu,
            in_channels=in_shape_actual[1], in_len=in_shape_actual[2],
            stride=stride[0], padding=padding[0]),
            torch_dtype=grad_output.dtype)
    else:
        # Groups > 1: CPU fallback (no GPU kernel)
        go_cpu = grad_output.to_torch_cpu() if isinstance(grad_output, ApplegpuTensor) else grad_output
        in_cpu = input.to_torch_cpu() if isinstance(input, ApplegpuTensor) else input
        w_cpu = weight.to_torch_cpu() if isinstance(weight, ApplegpuTensor) else weight
        gi_cpu = torch.ops.aten.convolution_backward(
            go_cpu, in_cpu, w_cpu, bias_sizes, stride, padding, dilation,
            transposed, output_padding, groups, [True, False, False]
        )[0]
        grad_input = ApplegpuTensor.from_torch(gi_cpu)
```

Python test:

```python
class TestConv1dBackwardInput:
    def test_matches_pytorch(self):
        np.random.seed(42)
        x = torch.randn(1, 3, 16, requires_grad=True)
        w = torch.randn(8, 3, 3)
        y = torch.nn.functional.conv1d(x, w, stride=1, padding=1)
        grad = torch.randn_like(y)
        y.backward(grad)
        expected = x.grad.numpy()
        # GPU
        result = _to_numpy(gpu.conv1d_backward_input(
            gpu.from_numpy(grad.numpy()), gpu.from_numpy(w.numpy()),
            in_channels=3, in_len=16, stride=1, padding=1))
        np.testing.assert_allclose(result, expected, atol=1e-4)
```

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/ crates/python/src/ python/tests/test_backward_ops.py python/applegpu_runtime/
git commit -m "feat: conv1d_backward_input on Metal GPU"
```

### Task 9: max_pool2d_backward kernel

**Files:** Same 7-file pattern.

- [ ] **Step 1: Add OpKind in graph.rs**

```rust
MaxPool2dBackward,
```

- [ ] **Step 2: Add ops function in ops.rs**

```rust
/// MaxPool2d backward: scatter grad_output to max positions using indices.
/// grad_output: [batch, channels, out_h, out_w]
/// indices: [batch, channels, out_h, out_w] (Int32, flat spatial index into input)
/// Output: [batch, channels, in_h, in_w]
pub fn max_pool2d_backward(
    rt: &mut LazyRuntime, grad_output_id: u64, indices_id: u64,
    batch: usize, channels: usize, in_h: usize, in_w: usize,
) -> Result<u64> {
    let dtype = rt.dtype(grad_output_id)?;
    validate_op_dtype(&OpKind::MaxPool2dBackward, dtype)?;
    let out_shape = vec![batch, channels, in_h, in_w];
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id, op: OpKind::MaxPool2dBackward,
        inputs: vec![grad_output_id, indices_id],
        out_shape: Shape::new(out_shape)?, out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}
```

- [ ] **Step 3: Add MSL kernel with atomic adds**

```rust
pub fn max_pool2d_backward_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    // Always use float output buffer for atomics, same as embedding_backward
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

// Atomic float add via compare-and-swap
static void atomic_add_float(device float* addr, float val) {{
    uint expected = __metal_atomic_load_explicit((device atomic_uint*)addr, memory_order_relaxed);
    while (true) {{
        float old_f = as_type<float>(expected);
        float new_f = old_f + val;
        uint new_u = as_type<uint>(new_f);
        bool ok = __metal_atomic_compare_exchange_weak_explicit(
            (device atomic_uint*)addr, &expected, new_u,
            memory_order_relaxed, memory_order_relaxed);
        if (ok) break;
    }}
}}

kernel void max_pool2d_backward{s}(
    device const {t}* grad_output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& numel_out [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& in_hw [[buffer(5)]],
    constant uint& out_hw [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {{
    if (id >= numel_out) return;
    // id indexes into grad_output (flat: batch*channels*out_h*out_w)
    uint bc = id / out_hw;  // batch*channels index
    int idx = indices[id];  // flat spatial index into input (ih*in_w + iw)
    if (idx < 0 || uint(idx) >= in_hw) return;
    uint gi_pos = bc * in_hw + uint(idx);
    float go_val = float(grad_output[id]);
    atomic_add_float(&grad_input[gi_pos], go_val);
}}
"#,
        t = t, s = s,
    )
}
```

- [ ] **Step 4: Add dispatch in compute.rs (blocking AND non-blocking)**

Add both `dispatch_max_pool2d_backward_typed()` and `dispatch_max_pool2d_backward_typed_nb()`. The output buffer must be **zero-initialized** before dispatch. The dispatch launches a 1D grid of `numel_out` threads.

- [ ] **Step 5: Add eval in lazy.rs**

Important: zero-initialize the output buffer before dispatch. Use the same pattern as embedding_backward — check how it handles zero-init. If `pool.acquire` doesn't zero memory, use `ffi::gpu_bridge_blit_fill_zero` or allocate a zeroed buffer via `Buffer::new_with_zeros`. Alternatively, allocate the buffer and use `memset` on the raw pointer before GPU dispatch (the buffer uses shared memory so CPU writes are visible to GPU).

```rust
if node.op.is_max_pool2d_backward() {
    let out_buf = self.pool.acquire(device, out_size)?;
    // Zero-initialize output for atomic accumulation — required for correctness
    unsafe {
        std::ptr::write_bytes(out_buf.raw_handle() as *mut u8, 0, out_size);
    }
    let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
    let grad_output = self.get_tensor(node.inputs[0])?;
    let indices = self.get_tensor(node.inputs[1])?;
    let dims = node.out_shape.dims();
    let channels = dims[1];
    let in_hw = dims[2] * dims[3];
    let go_dims = grad_output.meta.layout.shape.dims();
    let out_hw = go_dims[2] * go_dims[3];
    let numel_out: usize = go_dims.iter().product();
    REGISTRY.dispatch_max_pool2d_backward_typed(device, dtype, &grad_output.buffer, &indices.buffer, &out.buffer, numel_out, channels, in_hw, out_hw)?;
    return Ok(out);
}
```

- [ ] **Step 6: Add Python bindings, tests, torch_backend integration**

Update `_op_max_pool2d_backward` in torch_backend.py to use GPU path.

Python test comparing against PyTorch:

```python
class TestMaxPool2dBackward:
    def test_matches_pytorch(self):
        x = torch.randn(1, 2, 4, 4, requires_grad=True)
        pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        y, indices = pool(x)
        grad = torch.randn_like(y)
        y.backward(grad)
        expected = x.grad.numpy()
        # GPU
        result = _to_numpy(gpu.max_pool2d_backward(
            gpu.from_numpy(grad.detach().numpy()),
            gpu.from_numpy(indices.numpy().astype(np.int32)),
            batch=1, channels=2, in_h=4, in_w=4))
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_overlapping_pools(self):
        """stride < kernel_size — multiple outputs map to same input (needs atomics)."""
        x = torch.randn(1, 1, 6, 6, requires_grad=True)
        pool = torch.nn.MaxPool2d(3, stride=2, padding=0, return_indices=True)
        y, indices = pool(x)
        grad = torch.ones_like(y)
        y.backward(grad)
        expected = x.grad.numpy()
        result = _to_numpy(gpu.max_pool2d_backward(
            gpu.from_numpy(grad.detach().numpy()),
            gpu.from_numpy(indices.numpy().astype(np.int32)),
            batch=1, channels=1, in_h=6, in_w=6))
        np.testing.assert_allclose(result, expected, atol=1e-5)
```

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/ crates/python/src/ python/tests/test_backward_ops.py python/applegpu_runtime/
git commit -m "feat: max_pool2d_backward on Metal GPU with atomic scatter"
```

---

## Chunk 3: Wire protocol + full test suite

### Task 10: Wire protocol serialization (serial.rs)

**Files:**
- Modify: `crates/core/src/serial.rs`

- [ ] **Step 1: Assign discriminant codes**

Add to `op_to_discriminant()` (use next available codes after existing ones):

```rust
OpKind::ThresholdBackward { .. } => 70,
OpKind::TanhBackward => 71,
OpKind::SigmoidBackward => 72,
OpKind::GeluBackward => 73,
OpKind::Conv1dBackwardInput { .. } => 74,
OpKind::MaxPool2dBackward => 75,
```

- [ ] **Step 2: Add deserialization in discriminant_to_op()**

```rust
70 => {
    let threshold = read_f32(r)?;
    Ok(OpKind::ThresholdBackward { threshold })
}
71 => Ok(OpKind::TanhBackward),
72 => Ok(OpKind::SigmoidBackward),
73 => Ok(OpKind::GeluBackward),
74 => {
    let stride = read_usize(r)?;
    let padding = read_usize(r)?;
    Ok(OpKind::Conv1dBackwardInput { stride, padding })
}
75 => Ok(OpKind::MaxPool2dBackward),
```

- [ ] **Step 3: Add parameter serialization in EvalRequest::serialize**

For ops with parameters (ThresholdBackward, Conv1dBackwardInput), serialize the config values after the discriminant:

```rust
OpKind::ThresholdBackward { threshold } => {
    w.write_all(&threshold.to_le_bytes())?;
}
OpKind::Conv1dBackwardInput { stride, padding } => {
    w.write_all(&(stride as u32).to_le_bytes())?;
    w.write_all(&(padding as u32).to_le_bytes())?;
}
```

- [ ] **Step 4: Compile and test**

Run: `cargo test -p applegpu-core 2>&1 | tail -5`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/serial.rs
git commit -m "feat: wire protocol serialization for 6 backward ops (codes 70-75)"
```

### Task 11: Training integration test

**Files:**
- Modify: `python/tests/test_backward_ops.py`

- [ ] **Step 1: Add training integration test**

```python
class TestTrainingIntegration:
    """Verify backward ops work in a real training loop."""

    def test_mlp_training_step(self):
        """MLP with ReLU + GELU + tanh + sigmoid — exercises all 4 element-wise backward ops."""
        from applegpu_runtime.torch_backend import enable, to_applegpu, set_eager_mode
        enable()
        set_eager_mode(True)

        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),        # threshold_backward
            torch.nn.Linear(16, 16),
            torch.nn.GELU(),         # gelu_backward
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),         # tanh_backward
            torch.nn.Linear(16, 4),
            torch.nn.Sigmoid(),      # sigmoid_backward
        )
        to_applegpu(model)

        x = torch.randn(2, 8)
        x_gpu = to_applegpu(x)
        target = torch.zeros(2, 4)
        target_gpu = to_applegpu(target)

        y = model(x_gpu)
        loss = ((y - target_gpu) ** 2).mean()
        loss_val = loss.to_torch_cpu().item()
        loss.backward()

        # Verify gradients exist and are non-zero
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            grad_cpu = param.grad.to_torch_cpu() if hasattr(param.grad, 'to_torch_cpu') else param.grad
            assert grad_cpu.abs().sum() > 0, f"Zero gradient for {name}"

        assert loss_val > 0, "Loss should be positive"

        set_eager_mode(False)

    def test_cnn_training_step(self):
        """CNN with max_pool2d — exercises max_pool2d_backward."""
        from applegpu_runtime.torch_backend import enable, to_applegpu, set_eager_mode
        enable()
        set_eager_mode(True)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),  # max_pool2d_backward
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 4 * 4, 2),
        )
        to_applegpu(model)

        x = torch.randn(2, 1, 8, 8)
        x_gpu = to_applegpu(x)
        target = torch.zeros(2, 2)
        target_gpu = to_applegpu(target)

        y = model(x_gpu)
        loss = ((y - target_gpu) ** 2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

        set_eager_mode(False)

    def test_conv1d_training_step(self):
        """Conv1d encoder — exercises conv1d_backward_input."""
        from applegpu_runtime.torch_backend import enable, to_applegpu, set_eager_mode
        enable()
        set_eager_mode(True)

        model = torch.nn.Sequential(
            torch.nn.Conv1d(8, 16, 3, padding=1),  # conv1d_backward_input
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 4, 3, padding=1),
            torch.nn.ReLU(),
        )
        to_applegpu(model)

        x = torch.randn(2, 8, 32)
        x_gpu = to_applegpu(x)

        y = model(x_gpu)
        loss = y.mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

        set_eager_mode(False)
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest python/tests/test_backward_ops.py -v`
Expected: All tests pass

Run: `uv run pytest python/tests/ -v --ignore=python/tests/test_torch_backend.py 2>&1 | tail -5`
Expected: All tests pass, no backward op CPU fallback warnings for the 6 ops we implemented

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_backward_ops.py
git commit -m "test: training integration test for Metal backward ops"
```

### Task 12: Final verification and cleanup

- [ ] **Step 1: Run Rust tests**

Run: `cargo test -p applegpu-core 2>&1 | tail -10`
Expected: All pass

- [ ] **Step 2: Run full Python test suite**

Run: `uv run pytest python/tests/ -v --ignore=python/tests/test_torch_backend.py 2>&1 | tail -10`
Expected: All pass

- [ ] **Step 3: Remove stale TODO comments from torch_backend.py**

Remove the TODO comments from the replaced backward op handlers (lines ~446, ~460, ~482, ~495, ~1476 conv1d portion, ~1607).

- [ ] **Step 4: Commit cleanup**

```bash
git add python/applegpu_runtime/torch_backend.py
git commit -m "chore: remove stale TODO comments for implemented backward ops"
```
