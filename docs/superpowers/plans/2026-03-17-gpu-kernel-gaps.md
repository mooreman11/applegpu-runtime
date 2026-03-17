# GPU Kernel Gaps Implementation Plan (Issues #16–#22)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate CPU fallbacks for 7 GPU operations across three PRs, ordered by complexity.

**Architecture:** Three PRs targeting the full stack (MSL kernels → Rust core → PyO3 → Python torch_backend). PR 1 is plumbing/composition, PR 2 adds new Metal kernels, PR 3 adds complex kernels with groups/atomics.

**Tech Stack:** Metal Shading Language (MSL), Rust (applegpu-core crate), Swift (AppleGPUBridge), PyO3, Python/pytest

**Spec:** `docs/superpowers/specs/2026-03-17-gpu-kernel-gaps-design.md`

---

## PR 1 — Quick Wins (#20 blit copy, #22 vector norm)

**Branch:** `gpu-kernel-gaps-pr1`

### Task 1: GPU→GPU Blit Copy (#20)

**Files:**
- Modify: `crates/core/src/compute.rs` — add standalone `blit_copy()` function
- Modify: `crates/python/src/backend.rs` — add `blit_copy` to Backend trait
- Modify: `crates/python/src/metal_backend.rs` — implement `blit_copy`
- Modify: `crates/python/src/lib.rs` — expose `blit_copy` to Python
- Modify: `python/applegpu_runtime/__init__.py` — re-export `blit_copy`
- Modify: `python/applegpu_runtime/torch_backend.py:1101-1118` — update `_op_copy_`
- Test: `python/tests/test_blit_copy.py`

- [ ] **Step 1: Write failing Python test**

```python
# python/tests/test_blit_copy.py
import applegpu_runtime as gpu
import torch

def test_blit_copy_gpu_to_gpu():
    """GPU→GPU copy should not roundtrip through CPU."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    src = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    dst = gpu.tensor([0.0, 0.0, 0.0, 0.0], shape=[2, 2])
    gpu.blit_copy(dst, src)
    result = dst.to_list()
    assert abs(result[0] - 1.0) < 1e-6
    assert abs(result[3] - 4.0) < 1e-6

def test_copy_op_gpu_tensor():
    """torch copy_ with GPU tensors should use blit path."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    a_gpu = gpu.to_applegpu(a)
    b_gpu = gpu.to_applegpu(b)
    a_gpu.copy_(b_gpu)
    result = a_gpu.to_torch_cpu()
    assert torch.allclose(result, b, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest python/tests/test_blit_copy.py -v`
Expected: FAIL — `blit_copy` not defined

- [ ] **Step 3: Add `blit_copy` to Rust compute.rs**

In `crates/core/src/compute.rs`, add a standalone function (not on KernelRegistry — blit uses MTLBlitCommandEncoder, not MTLComputeCommandEncoder):

```rust
/// GPU→GPU buffer copy via Metal blit encoder.
/// Uses the non-blocking FFI and waits for completion.
pub fn blit_copy(device: &Device, src: &Buffer, dst: &Buffer, size_bytes: usize) -> Result<()> {
    let queue = get_shared_queue(device);  // returns raw pointer, not Result
    if queue.is_null() {
        return Err(GpuError::ComputeFailed("failed to get shared queue".into()));
    }
    let cb = unsafe {
        ffi::gpu_bridge_blit_copy_nb(
            device.raw_handle(),
            queue,
            src.raw_handle() as *mut _,
            dst.raw_handle() as *mut _,
            size_bytes as u64,
        )
    };
    if cb.is_null() {
        return Err(GpuError::ComputeFailed("blit copy failed".into()));
    }
    unsafe { ffi::gpu_bridge_wait_command_buffer(cb) };
    Ok(())
}
```

- [ ] **Step 4: Add `blit_copy` to Backend trait and MetalBackend**

In `crates/python/src/backend.rs`, add to the Backend trait:
```rust
fn blit_copy(&self, dst: u64, src: u64) -> BackendResult<()>;
```

In `crates/python/src/metal_backend.rs`, implement:
```rust
fn blit_copy(&self, dst: u64, src: u64) -> BackendResult<()> {
    let rt = self.runtime.lock().unwrap();
    let src_tensor = rt.get_tensor(src).map_err(|e| e.to_string())?;
    let dst_tensor = rt.get_tensor(dst).map_err(|e| e.to_string())?;
    let size = src_tensor.buffer.length().min(dst_tensor.buffer.length());
    let device = rt.device();
    applegpu_core::compute::blit_copy(device, &src_tensor.buffer, &dst_tensor.buffer, size)
        .map_err(|e| e.to_string())
}
```

- [ ] **Step 5: Expose `blit_copy` to Python via PyO3**

In `crates/python/src/lib.rs`, add:
```rust
#[pyfunction]
fn blit_copy(dst: &GpuTensor, src: &GpuTensor) -> PyResult<()> {
    BACKEND.blit_copy(dst.id, src.id).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}
```

Register in module: `m.add_function(wrap_pyfunction!(blit_copy, m)?)?;`

In `python/applegpu_runtime/__init__.py`, add `blit_copy` to imports and `__all__`.

- [ ] **Step 6: Update `_op_copy_` in torch_backend.py**

Replace the handler at `python/applegpu_runtime/torch_backend.py:1101-1118`:

```python
@register_op(torch.ops.aten.copy_.default)
def _op_copy_(dst, src, non_blocking=False):
    """In-place copy of src into dst. Uses GPU→GPU blit when both are GPU tensors."""
    if isinstance(src, ApplegpuTensor) and isinstance(dst, ApplegpuTensor):
        # GPU→GPU blit copy (no CPU roundtrip)
        try:
            gpu.blit_copy(dst._gpu_tensor, _unwrap(src))
            return dst
        except Exception:
            pass  # Fall through to CPU path on size mismatch
    if isinstance(src, ApplegpuTensor):
        cpu_data = src.to_torch_cpu()
        new_gpu = gpu.from_torch(cpu_data)
    elif isinstance(src, torch.Tensor):
        new_gpu = gpu.from_torch(src)
    else:
        return dst
    dst._gpu_tensor = new_gpu
    _gpu_tensor_registry[dst.data_ptr()] = new_gpu
    return dst
```

- [ ] **Step 7: Run tests and verify pass**

Run: `uv run pytest python/tests/test_blit_copy.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add crates/core/src/compute.rs crates/python/src/backend.rs crates/python/src/metal_backend.rs crates/python/src/lib.rs python/applegpu_runtime/__init__.py python/applegpu_runtime/torch_backend.py python/tests/test_blit_copy.py
git commit -m "feat: GPU→GPU blit copy eliminates CPU roundtrip in copy_ (#20)"
```

---

### Task 2: GPU linalg_vector_norm via Composition (#22)

**Files:**
- Modify: `python/applegpu_runtime/torch_backend.py:1397-1407` — replace CPU fallback with GPU composition
- Test: `python/tests/test_vector_norm_gpu.py`

- [ ] **Step 1: Write failing Python test**

```python
# python/tests/test_vector_norm_gpu.py
import applegpu_runtime as gpu
import torch
import math

def test_l2_norm_1d():
    """L2 norm of 1D tensor on GPU."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.tensor([3.0, 4.0])
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=2)
    assert abs(result.to_torch_cpu().item() - 5.0) < 1e-5

def test_l1_norm_1d():
    """L1 norm of 1D tensor on GPU."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.tensor([-3.0, 4.0])
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=1)
    assert abs(result.to_torch_cpu().item() - 7.0) < 1e-5

def test_l2_norm_2d_dim():
    """L2 norm along specific dim."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4)
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=2, dim=1)
    expected = torch.linalg.vector_norm(x, ord=2, dim=1)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)

def test_l2_norm_keepdim():
    """L2 norm with keepdim=True."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4)
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=2, dim=1, keepdim=True)
    expected = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
    assert result.to_torch_cpu().shape == expected.shape
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)

def test_linf_norm_falls_back():
    """L∞ norm should still work (CPU fallback)."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.tensor([-5.0, 3.0, 4.0])
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=float('inf'))
    assert abs(result.to_torch_cpu().item() - 5.0) < 1e-5
```

- [ ] **Step 2: Run test to verify current state**

Run: `uv run pytest python/tests/test_vector_norm_gpu.py -v`
Expected: Tests pass (CPU fallback works), but we want GPU path

- [ ] **Step 3: Replace CPU fallback with GPU composition**

Replace `_op_linalg_vector_norm` in `torch_backend.py:1397-1407`:

```python
@register_op(torch.ops.aten.linalg_vector_norm.default)
def _op_linalg_vector_norm(a, ord=2.0, dim=None, keepdim=False, dtype=None):
    """Vector norm reduction. L1/L2 composed on GPU, others CPU fallback.

    TODO: L∞ norm needs an amax reduction kernel (see backlog).
    """
    if not isinstance(a, ApplegpuTensor):
        return torch.ops.aten.linalg_vector_norm.default(a, ord, dim, keepdim, dtype=dtype)

    # L1 and L2 can be composed from existing GPU ops
    if ord == 2.0 or ord == 2:
        # L2: sqrt(sum(x^2))
        squared = _wrap(gpu.mul(_unwrap(a), _unwrap(a)))
        summed = _op_sum(squared, dim=[dim] if isinstance(dim, int) else dim,
                         keepdim=keepdim) if dim is not None else _op_sum_default(squared)
        return _wrap(gpu.sqrt(_unwrap(summed)))
    elif ord == 1.0 or ord == 1:
        # L1: sum(|x|)
        abs_a = _wrap(gpu.abs(_unwrap(a)))
        if dim is not None:
            return _op_sum(abs_a, dim=[dim] if isinstance(dim, int) else dim, keepdim=keepdim)
        else:
            return _op_sum_default(abs_a)
    else:
        # L∞ and exotic norms: CPU fallback
        # TODO: L∞ needs amax reduction kernel — see GitHub backlog issue
        a_cpu = a.to_torch_cpu()
        result = torch.ops.aten.linalg_vector_norm.default(a_cpu, ord, dim, keepdim, dtype=dtype)
        return ApplegpuTensor.from_torch(result)
```

Note: `_op_sum` is the handler for `torch.ops.aten.sum.dim_IntList` and `_op_sum_default` for `torch.ops.aten.sum.default`. Both are already defined in the file and can be called directly.

- [ ] **Step 4: Run tests and verify pass**

Run: `uv run pytest python/tests/test_vector_norm_gpu.py -v`
Expected: PASS

- [ ] **Step 5: File L∞ backlog issue**

```bash
gh issue create --title "GPU amax reduction kernel for L∞ vector norm" --body "## Problem

linalg_vector_norm with ord=inf falls back to CPU because there is no amax (absolute-value max) reduction kernel on GPU.

## Context

L1 and L2 norms are now composed from existing GPU ops (mul+sum+sqrt for L2, abs+sum for L1). L∞ needs a max-reduction kernel which does not exist.

## Proposed Solution

Add an amax reduction kernel template in kernel_templates.rs, similar to the existing sum/mean/var patterns. The kernel computes max(|x|) along the last dimension.

## Impact

Low — L∞ norm is rarely used in gradient clipping. CPU fallback is acceptable for now."
```

- [ ] **Step 6: Commit**

```bash
git add python/applegpu_runtime/torch_backend.py python/tests/test_vector_norm_gpu.py
git commit -m "feat: GPU-composed L1/L2 vector norm, CPU fallback for L∞ (#22)"
```

---

### Task 3: Run full test suite and create PR 1

- [ ] **Step 1: Run full test suite**

```bash
make test
```
Expected: All existing tests pass, new tests pass

- [ ] **Step 2: Create PR**

```bash
gh pr create --title "feat: GPU blit copy and vector norm (#20, #22)" --body "$(cat <<'EOF'
## Summary
- GPU→GPU blit copy eliminates CPU roundtrip in copy_ op (#20)
- L1/L2 vector norm composed from existing GPU ops (#22)
- L∞ norm deferred to backlog (needs amax kernel)

## Test plan
- [ ] `python/tests/test_blit_copy.py` — GPU→GPU copy correctness
- [ ] `python/tests/test_vector_norm_gpu.py` — L1, L2, dim, keepdim
- [ ] Full test suite (`make test`) passes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## PR 2 — New Kernel Variants (#16 pool indices, #18 exact GELU)

**Branch:** `gpu-kernel-gaps-pr2`

### Task 4: max_pool2d_with_indices GPU Kernel (#16)

**Files:**
- Modify: `crates/core/src/kernel_templates.rs` — add `max_pool2d_with_indices_kernel_source`
- Modify: `crates/core/src/graph.rs` — add `MaxPool2dWithIndices` OpKind
- Modify: `crates/core/src/ops.rs` — add `max_pool2d_with_indices()` function
- Modify: `crates/core/src/lazy.rs` — add dispatch for new op (both sync and async paths)
- Modify: `crates/core/src/compute.rs` — add kernel resolution for `max_pool2d_with_indices`
- Modify: `crates/python/src/backend.rs` — add to Backend trait
- Modify: `crates/python/src/metal_backend.rs` — implement
- Modify: `crates/python/src/lib.rs` — PyO3 binding
- Modify: `python/applegpu_runtime/__init__.py` — re-export
- Modify: `python/applegpu_runtime/torch_backend.py:1377-1392` — use GPU indices
- Test: `python/tests/test_pool_indices.py`

- [ ] **Step 1: Write failing Python test**

```python
# python/tests/test_pool_indices.py
import applegpu_runtime as gpu
import torch

def test_max_pool2d_indices_match_cpu():
    """GPU max_pool2d indices should match CPU."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 1, 4, 4)
    x_gpu = gpu.to_applegpu(x)
    values_gpu, indices_gpu = torch.nn.functional.max_pool2d_with_indices(x_gpu, 2, stride=2)
    values_cpu, indices_cpu = torch.nn.functional.max_pool2d_with_indices(x, 2, stride=2)
    assert torch.allclose(values_gpu.to_torch_cpu(), values_cpu, atol=1e-5)
    assert torch.equal(indices_gpu.to_torch_cpu(), indices_cpu)

def test_max_pool2d_indices_with_padding():
    """GPU max_pool2d indices with padding."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(2, 3, 8, 8)
    x_gpu = gpu.to_applegpu(x)
    values_gpu, indices_gpu = torch.nn.functional.max_pool2d_with_indices(
        x_gpu, kernel_size=3, stride=2, padding=1
    )
    values_cpu, indices_cpu = torch.nn.functional.max_pool2d_with_indices(
        x, kernel_size=3, stride=2, padding=1
    )
    assert torch.allclose(values_gpu.to_torch_cpu(), values_cpu, atol=1e-5)
    assert torch.equal(indices_gpu.to_torch_cpu(), indices_cpu)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest python/tests/test_pool_indices.py -v`
Expected: Indices may not match (currently CPU-computed, will match — but we want GPU path)

- [ ] **Step 3: Add MSL kernel template**

In `crates/core/src/kernel_templates.rs`, add after `max_pool2d_kernel_source`:

```rust
pub fn max_pool2d_with_indices_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void max_pool2d_idx{s}(
    device const {t}* input [[buffer(0)]],
    device int* out_indices [[buffer(1)]],
    device {t}* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& in_h [[buffer(5)]],
    constant uint& in_w [[buffer(6)]],
    constant uint& out_h [[buffer(7)]],
    constant uint& out_w [[buffer(8)]],
    constant uint& kh [[buffer(9)]],
    constant uint& kw [[buffer(10)]],
    constant uint& stride_h [[buffer(11)]],
    constant uint& stride_w [[buffer(12)]],
    constant uint& pad_h [[buffer(13)]],
    constant uint& pad_w [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint ow = gid.x;
    uint combined = gid.y;
    uint b = gid.z;
    uint c = combined % channels;
    uint oh = combined / channels;
    if (ow >= out_w || oh >= out_h || b >= batch) return;

    float max_val = -1e30f;
    int max_idx = 0;
    for (uint i = 0; i < kh; i++) {{
        for (uint j = 0; j < kw; j++) {{
            int ih = int(oh * stride_h + i) - int(pad_h);
            int iw = int(ow * stride_w + j) - int(pad_w);
            if (ih >= 0 && uint(ih) < in_h && iw >= 0 && uint(iw) < in_w) {{
                uint in_off = b * channels * in_h * in_w + c * in_h * in_w + uint(ih) * in_w + uint(iw);
                float val = float(input[in_off]);
                if (val > max_val) {{
                    max_val = val;
                    max_idx = ih * int(in_w) + iw;
                }}
            }}
        }}
    }}
    uint out_off = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
    output[out_off] = {t}(max_val);
    out_indices[out_off] = max_idx;
}}
"#,
        t = t, s = s,
    )
}
```

Note: `out_indices` at buffer(1) is declared writable (`device int*`) even though it's passed through `input_buffers[1]` in the dispatch. This is valid Metal — `setBuffer` doesn't enforce read-only.

- [ ] **Step 4: Add OpKind, kernel resolution, ops function**

In `crates/core/src/graph.rs`, add variant:
```rust
MaxPool2dWithIndices { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
```
Add kernel_name mapping: `OpKind::MaxPool2dWithIndices { .. } => "max_pool2d_with_indices"`
Add `is_max_pool2d_with_indices()` predicate.

In `crates/core/src/compute.rs`, add kernel resolution:
```rust
"max_pool2d_with_indices" => kt::max_pool2d_with_indices_kernel_source(dtype),
```

In `crates/core/src/ops.rs`, add:
```rust
pub fn max_pool2d_with_indices(
    rt: &mut LazyRuntime, input_id: u64,
    kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize),
) -> Result<(u64, u64)> {
    // Same shape validation as max_pool2d...
    let dtype = rt.dtype(input_id)?;
    let in_shape = rt.shape(input_id)?;
    // ... validate 4D, compute out_h, out_w ...
    let out_h = (in_shape[2] + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_w = (in_shape[3] + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
    let out_shape = vec![in_shape[0], in_shape[1], out_h, out_w];

    let values_id = next_id();
    rt.record_op(OpNode {
        id: values_id,
        op: OpKind::MaxPool2dWithIndices { kernel_size, stride, padding },
        inputs: vec![input_id],
        out_shape: Shape::new(out_shape.clone())?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });

    // Indices tensor recorded as a dependent (same shape, Int32)
    let indices_id = next_id();
    rt.record_op(OpNode {
        id: indices_id,
        op: OpKind::Noop,  // Placeholder — indices computed alongside values
        inputs: vec![values_id],
        out_shape: Shape::new(out_shape)?,
        out_dtype: DType::Int32,
        container_id: ContainerId::DEFAULT,
    });

    Ok((values_id, indices_id))
}
```

**Multi-output design:** The lazy executor must produce both tensors when it encounters `MaxPool2dWithIndices`. The recommended approach:
1. `ops::max_pool2d_with_indices()` calls `next_id()` twice to get `values_id` and `indices_id`
2. Only the values op is recorded in the graph (with `MaxPool2dWithIndices` kind)
3. The `indices_id` is stored as metadata on the values OpNode (e.g., add an `extra_output_id: Option<u64>` field to OpNode, or store in a side HashMap on LazyRuntime)
4. When the lazy executor dispatches `MaxPool2dWithIndices`, it allocates both buffers, runs the kernel, and registers both tensors in the runtime's tensor store
5. The Python layer receives both IDs from the ops function and wraps them as separate GpuTensor objects

Do NOT use `node.id + 1` — tensor IDs from `next_id()` are atomic counters and `+1` could collide. Always use the IDs allocated by `next_id()` in the ops function.

- [ ] **Step 5: Add lazy execution dispatch**

In `crates/core/src/lazy.rs`, add to the synchronous execution path (near the existing MaxPool2d handler):

```rust
if let crate::graph::OpKind::MaxPool2dWithIndices { kernel_size, stride, padding } = node.op {
    let out_buf = self.pool.acquire(device, out_size)?;
    let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
    let input = self.get_tensor(node.inputs[0])?;
    let in_dims = input.meta.layout.shape.dims();
    let out_dims = node.out_shape.dims();

    // Allocate indices buffer (Int32, same shape as output)
    let idx_size = out_dims.iter().product::<usize>() * 4; // 4 bytes per Int32
    let idx_buf = self.pool.acquire(device, idx_size)?;

    let uint_params: Vec<u32> = vec![
        in_dims[0] as u32, in_dims[1] as u32,
        in_dims[2] as u32, in_dims[3] as u32,
        out_dims[2] as u32, out_dims[3] as u32,
        kernel_size.0 as u32, kernel_size.1 as u32,
        stride.0 as u32, stride.1 as u32,
        padding.0 as u32, padding.1 as u32,
    ];
    let grid_y = out_dims[2] as u32 * in_dims[1] as u32;
    let (k_src, k_fn) = KernelRegistry::resolve_kernel("max_pool2d_with_indices", dtype);
    // NOTE: idx_buf passed as input_buffers[1] but is WRITTEN by the kernel
    REGISTRY.dispatch_cnn_3d(
        device, &k_src, &k_fn,
        &[&input.buffer, &idx_buf], &out.buffer,
        &uint_params, &[], (out_dims[3] as u32, grid_y, in_dims[0] as u32),
    )?;

    // Store indices tensor using the pre-allocated indices_id from ops
    // Retrieve indices_id from node metadata (extra_output_id field or side map)
    let indices_id = node.extra_output_id.expect("MaxPool2dWithIndices must have indices_id");
    let idx_tensor = Tensor::from_raw(
        indices_id,
        node.out_shape.dims().to_vec(),
        DType::Int32,
        idx_buf,
    );
    self.store_tensor(idx_tensor);  // Register in runtime's tensor store
    return Ok(out);
}
```

Also add the async (non-blocking) variant in the NB section.

- [ ] **Step 6: Add PyO3 binding and Python handler**

In `crates/python/src/lib.rs`:
```rust
#[pyfunction]
#[pyo3(signature = (input, kh=2, kw=2, stride_h=0, stride_w=0, pad_h=0, pad_w=0))]
fn max_pool2d_with_indices(input: &GpuTensor, kh: usize, kw: usize,
    stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize
) -> PyResult<(GpuTensor, GpuTensor)> {
    // ... similar to max_pool2d but returns tuple
}
```

Update torch_backend.py `_max_pool2d` handler:
```python
@register_op(torch.ops.aten.max_pool2d_with_indices.default)
def _max_pool2d(input, kernel_size, stride=None, padding=(0, 0), dilation=(1, 1), ceil_mode=False):
    if stride is None or len(stride) == 0:
        stride = kernel_size
    kh, kw = kernel_size[0], kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]
    sh, sw = stride[0], stride[1] if len(stride) > 1 else stride[0]
    ph, pw = padding[0], padding[1] if len(padding) > 1 else padding[0]
    values, indices = gpu.max_pool2d_with_indices(_unwrap(input), kh, kw, sh, sw, ph, pw)
    return _wrap(values), _wrap(indices)
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest python/tests/test_pool_indices.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git commit -m "feat: max_pool2d outputs indices on GPU, eliminates CPU roundtrip (#16)"
```

---

### Task 5: Exact GELU Forward + Backward Kernels (#18)

**Files:**
- Modify: `crates/core/src/kernel_templates.rs` — add 3 new kernel templates
- Modify: `crates/core/src/graph.rs` — add `GeluExact`, `GeluTanhBackward`, `GeluExactBackward`
- Modify: `crates/core/src/ops.rs` — add op functions
- Modify: `crates/core/src/lazy.rs` — add dispatch paths
- Modify: `crates/core/src/compute.rs` — kernel resolution
- Modify: `crates/core/src/fusion.rs:21-23` — add exact GELU fusion expression
- Modify: `crates/python/src/backend.rs` — add to trait
- Modify: `crates/python/src/metal_backend.rs` — implement
- Modify: `crates/python/src/lib.rs` — PyO3 bindings
- Modify: `python/applegpu_runtime/__init__.py` — re-export
- Modify: `python/applegpu_runtime/torch_backend.py:365-367,479-498` — route by `approximate` param
- Test: `python/tests/test_gelu_exact.py`

- [ ] **Step 1: Write failing Python test**

```python
# python/tests/test_gelu_exact.py
import applegpu_runtime as gpu
import torch
import math

def test_gelu_exact_forward():
    """Exact GELU should use erf, not tanh approximation."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.gelu(x_gpu, approximate="none")
    expected = torch.nn.functional.gelu(x, approximate="none")
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)

def test_gelu_tanh_forward_unchanged():
    """Tanh GELU should still work."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.gelu(x_gpu, approximate="tanh")
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)

def test_gelu_exact_backward():
    """Exact GELU backward on GPU matches CPU."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4, requires_grad=True)
    x_gpu = gpu.to_applegpu(x.detach())
    grad = torch.randn(3, 4)

    # CPU reference
    y_cpu = torch.nn.functional.gelu(x, approximate="none")
    y_cpu.backward(grad)
    expected_grad = x.grad.clone()

    # GPU
    grad_gpu = gpu.to_applegpu(grad)
    result = torch.ops.aten.gelu_backward(grad_gpu, x_gpu, approximate="none")
    assert torch.allclose(result.to_torch_cpu(), expected_grad, atol=1e-4)

def test_gelu_tanh_backward_gpu():
    """Tanh GELU backward should now run on GPU (was CPU fallback)."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4, requires_grad=True)
    x_gpu = gpu.to_applegpu(x.detach())
    grad = torch.randn(3, 4)

    y_cpu = torch.nn.functional.gelu(x, approximate="tanh")
    y_cpu.backward(grad)
    expected_grad = x.grad.clone()

    grad_gpu = gpu.to_applegpu(grad)
    result = torch.ops.aten.gelu_backward(grad_gpu, x_gpu, approximate="tanh")
    assert torch.allclose(result.to_torch_cpu(), expected_grad, atol=1e-4)
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest python/tests/test_gelu_exact.py -v`
Expected: `test_gelu_exact_forward` FAIL (currently uses tanh for all), backward tests FAIL

- [ ] **Step 3: Add kernel templates**

In `crates/core/src/kernel_templates.rs`, add three new functions:

**gelu_exact_kernel_source:**
```rust
pub fn gelu_exact_kernel_source(dtype: DType) -> String {
    // kernel void gelu_exact{s}(input, output, strides, shape, ndim, numel, id)
    // float x = input[offset];
    // output[id] = x * 0.5f * (1.0f + erf(x * M_SQRT1_2_F));
}
```

**gelu_tanh_backward_kernel_source:**
```rust
pub fn gelu_tanh_backward_kernel_source(dtype: DType) -> String {
    // kernel void gelu_tanh_bwd{s}(grad_output, self_input, output, strides_go, strides_self, shape, ndim, numel, id)
    // float x = self_input[...];
    // float inner = clamp(0.7978845608f * (x + 0.044715f * x*x*x), -10.0f, 10.0f);
    // float t = tanh(inner);
    // float dt = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
    // output[id] = grad_output[...] * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t*t) * dt);
}
```

**gelu_exact_backward_kernel_source:**
```rust
pub fn gelu_exact_backward_kernel_source(dtype: DType) -> String {
    // kernel void gelu_exact_bwd{s}(grad_output, self_input, output, strides_go, strides_self, shape, ndim, numel, id)
    // float x = self_input[...];
    // float cdf = 0.5f * (1.0f + erf(x * M_SQRT1_2_F));
    // float pdf = 0.3989422804f * exp(-0.5f * x * x);  // 1/sqrt(2π) * exp(-x²/2)
    // output[id] = grad_output[...] * (cdf + x * pdf);
}
```

Note: The backward kernels take two inputs (grad_output and self_input), so they need the binary N-D dispatch pattern (two sets of strides). Follow the pattern from existing binary N-D kernels.

- [ ] **Step 4: Add OpKind variants and dispatch**

In `crates/core/src/graph.rs`:
```rust
GeluExact,
GeluTanhBackward,
GeluExactBackward,
```

Add to `is_elementwise()`, `is_unary()` (for GeluExact), `kernel_name()`.
Note: GeluTanhBackward and GeluExactBackward are binary ops (grad_output, self_input).

In `crates/core/src/fusion.rs`, add after existing Gelu case:
```rust
OpKind::GeluExact => {
    format!("({expr} * 0.5f * (1.0f + erf({expr} * 0.70710678118f)))")
}
```

In `crates/core/src/compute.rs`, add kernel resolution:
```rust
"gelu_exact" => kt::gelu_exact_kernel_source(dtype),
"gelu_tanh_backward" => kt::gelu_tanh_backward_kernel_source(dtype),
"gelu_exact_backward" => kt::gelu_exact_backward_kernel_source(dtype),
```

- [ ] **Step 5: Add ops, backend, PyO3, Python handler**

In `crates/core/src/ops.rs`:
```rust
pub fn gelu_exact(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::GeluExact)
}
pub fn gelu_tanh_backward(rt: &mut LazyRuntime, grad_output_id: u64, self_input_id: u64) -> Result<u64> {
    lazy_binary_op(rt, grad_output_id, self_input_id, OpKind::GeluTanhBackward)
}
pub fn gelu_exact_backward(rt: &mut LazyRuntime, grad_output_id: u64, self_input_id: u64) -> Result<u64> {
    lazy_binary_op(rt, grad_output_id, self_input_id, OpKind::GeluExactBackward)
}
```

Wire through backend trait → metal_backend → PyO3 → __init__.py (same pattern as other ops).

Update torch_backend.py:

```python
@register_op(torch.ops.aten.gelu.default)
def _op_gelu(a, approximate="none"):
    if approximate == "tanh":
        return _wrap(gpu.gelu(_unwrap(a)))
    else:
        return _wrap(gpu.gelu_exact(_unwrap(a)))

@register_op(torch.ops.aten.gelu_backward.default)
def _op_gelu_backward(grad_output, self_tensor, approximate="none"):
    if approximate == "tanh":
        return _wrap(gpu.gelu_tanh_backward(_unwrap(grad_output), _unwrap(self_tensor)))
    else:
        return _wrap(gpu.gelu_exact_backward(_unwrap(grad_output), _unwrap(self_tensor)))
```

- [ ] **Step 6: Add lazy.rs dispatch for new ops**

GeluExact dispatches via `dispatch_unary_nd_typed` (same as existing Gelu).
GeluTanhBackward and GeluExactBackward dispatch via `dispatch_binary_nd_typed`.

- [ ] **Step 7: Run tests**

Run: `uv run pytest python/tests/test_gelu_exact.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git commit -m "feat: exact GELU mode + GPU backward for both modes (#18)"
```

---

### Task 6: Run full test suite and create PR 2

- [ ] **Step 1: Run full test suite**

```bash
make test
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --title "feat: GPU pool indices and exact GELU (#16, #18)" --body "..."
```

---

## PR 3 — Heavy Kernels (#17 conv grad_weight, #19 grouped conv, #21 index/scatter)

**Branch:** `gpu-kernel-gaps-pr3`

### Task 7: Conv1d Backward Input Kernel (#17 partial)

**Files:**
- Modify: `crates/core/src/kernel_templates.rs` — add `conv1d_backward_input_kernel_source`
- Modify: `crates/core/src/graph.rs` — add `Conv1dBackwardInput`
- Modify: `crates/core/src/ops.rs` — add `conv1d_backward_input()`
- Modify: `crates/core/src/lazy.rs` — dispatch
- Modify: `crates/core/src/compute.rs` — kernel resolution
- Modify: `crates/python/src/backend.rs`, `metal_backend.rs`, `lib.rs`
- Modify: `python/applegpu_runtime/torch_backend.py:1579-1588` — use GPU kernel
- Test: `python/tests/test_conv1d_backward.py`

- [ ] **Step 1: Write failing test**

```python
# python/tests/test_conv1d_backward.py
def test_conv1d_backward_input():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(2, 3, 16, requires_grad=True)
    w = torch.randn(4, 3, 3)
    x_gpu = gpu.to_applegpu(x.detach())
    w_gpu = gpu.to_applegpu(w)
    grad_out = torch.randn(2, 4, 14)
    grad_out_gpu = gpu.to_applegpu(grad_out)
    # GPU backward
    gi_gpu = gpu.conv1d_backward_input(grad_out_gpu, w_gpu, 16, 1, 0)
    # CPU reference
    y = torch.nn.functional.conv1d(x, w)
    y.backward(grad_out)
    assert torch.allclose(gi_gpu.to_list_tensor(), x.grad, atol=1e-4)
```

- [ ] **Steps 2-5: Implement kernel, OpKind, dispatch, wire through all layers**

The conv1d_backward_input kernel follows the same transposed-conv pattern as conv2d_backward_input but in 1D:

```metal
kernel void conv1d_backward_input{s}(
    device const {t}* grad_output [[buffer(0)]],
    device const {t}* weight [[buffer(1)]],
    device {t}* grad_input [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& in_channels [[buffer(4)]],
    constant uint& out_channels [[buffer(5)]],
    constant uint& in_length [[buffer(6)]],
    constant uint& out_length [[buffer(7)]],
    constant uint& kernel_size [[buffer(8)]],
    constant uint& stride [[buffer(9)]],
    constant uint& padding [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint il = gid.x; uint ic = gid.y; uint b = gid.z;
    if (il >= in_length || ic >= in_channels || b >= batch) return;
    float sum = 0.0f;
    for (uint oc = 0; oc < out_channels; oc++) {
        for (uint k = 0; k < kernel_size; k++) {
            int ol_candidate = int(il + padding) - int(k);
            if (ol_candidate >= 0 && ol_candidate % int(stride) == 0) {
                uint ol = uint(ol_candidate) / stride;
                if (ol < out_length) {
                    sum += grad_output[b*out_channels*out_length + oc*out_length + ol]
                         * weight[oc*in_channels*kernel_size + ic*kernel_size + k];
                }
            }
        }
    }
    grad_input[b*in_channels*in_length + ic*in_length + il] = sum;
}
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: conv1d backward_input Metal kernel (#17)"
```

---

### Task 8: Conv2d Backward Weight Kernel (#17)

**Files:**
- Modify: `crates/core/src/kernel_templates.rs` — add `conv2d_backward_weight_kernel_source`
- Modify: `crates/core/src/graph.rs` — add `Conv2dBackwardWeight`
- Modify: `crates/core/src/ops.rs`, `lazy.rs`, `compute.rs`
- Modify: backend.rs, metal_backend.rs, lib.rs
- Modify: `python/applegpu_runtime/torch_backend.py:1610` — use GPU kernel
- Test: `python/tests/test_conv2d_grad_weight.py`

- [ ] **Step 1: Write failing test**

```python
# python/tests/test_conv2d_grad_weight.py
def test_conv2d_grad_weight():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    w = torch.randn(4, 3, 3, 3, requires_grad=True)
    grad_out = torch.randn(2, 4, 6, 6)
    # CPU reference
    y = torch.nn.functional.conv2d(x, w)
    y.backward(grad_out)
    expected_gw = w.grad.clone()
    # GPU
    grad_out_gpu = gpu.to_applegpu(grad_out)
    x_gpu = gpu.to_applegpu(x.detach())
    gw_gpu = gpu.conv2d_backward_weight(grad_out_gpu, x_gpu, 3, 3, 4, 3, 1, 1, 0, 0)
    assert torch.allclose(gw_gpu.to_torch_cpu(), expected_gw, atol=1e-4)

def test_conv2d_grad_weight_nonsquare():
    """Non-square kernel and channels to catch indexing bugs."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 2, 6, 8, requires_grad=True)
    w = torch.randn(5, 2, 3, 4, requires_grad=True)
    grad_out = torch.randn(1, 5, 4, 5)
    y = torch.nn.functional.conv2d(x, w)
    y.backward(grad_out)
    expected_gw = w.grad.clone()
    grad_out_gpu = gpu.to_applegpu(grad_out)
    x_gpu = gpu.to_applegpu(x.detach())
    gw_gpu = gpu.conv2d_backward_weight(grad_out_gpu, x_gpu, 3, 4, 5, 2, 1, 1, 0, 0)
    assert torch.allclose(gw_gpu.to_torch_cpu(), expected_gw, atol=1e-4)
```

- [ ] **Steps 2-4: Implement kernel**

Thread per weight element (no atomics needed):

```metal
kernel void conv2d_backward_weight{s}(
    device const {t}* grad_output [[buffer(0)]],
    device const {t}* input [[buffer(1)]],
    device {t}* grad_weight [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& in_channels [[buffer(4)]],
    constant uint& out_channels [[buffer(5)]],
    constant uint& in_h [[buffer(6)]],
    constant uint& in_w [[buffer(7)]],
    constant uint& out_h [[buffer(8)]],
    constant uint& out_w [[buffer(9)]],
    constant uint& kh [[buffer(10)]],
    constant uint& kw [[buffer(11)]],
    constant uint& stride_h [[buffer(12)]],
    constant uint& stride_w [[buffer(13)]],
    constant uint& pad_h [[buffer(14)]],
    constant uint& pad_w [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint kw_idx = gid.x;
    uint combined = gid.y;  // kh_idx * out_channels + oc
    uint ic = gid.z;
    uint oc = combined % out_channels;
    uint kh_idx = combined / out_channels;
    if (kw_idx >= kw || kh_idx >= kh || ic >= in_channels || oc >= out_channels) return;

    float sum = 0.0f;
    for (uint b = 0; b < batch; b++) {
        for (uint oh = 0; oh < out_h; oh++) {
            for (uint ow = 0; ow < out_w; ow++) {
                uint ih = oh * stride_h + kh_idx - pad_h;
                uint iw = ow * stride_w + kw_idx - pad_w;
                if (ih < in_h && iw < in_w) {
                    sum += float(grad_output[b*out_channels*out_h*out_w + oc*out_h*out_w + oh*out_w + ow])
                         * float(input[b*in_channels*in_h*in_w + ic*in_h*in_w + ih*in_w + iw]);
                }
            }
        }
    }
    grad_weight[oc*in_channels*kh*kw + ic*kh*kw + kh_idx*kw + kw_idx] = {t}(sum);
}
```

Grid: `(kw, kh * out_channels, in_channels)`

- [ ] **Step 5: Wire through all layers and update torch_backend.py**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: conv2d backward_weight Metal kernel (#17)"
```

---

### Task 9: Grad Bias via GPU Sum + Conv1d Backward Weight (#17 completion)

- [ ] **Step 1: Update torch_backend.py grad_bias to use GPU sum**

```python
# In _op_conv_backward, replace CPU grad_bias with:
if output_mask[2] and bias_sizes is not None:
    # grad_bias = sum over batch and spatial dims
    go = _unwrap(grad_output) if isinstance(grad_output, ApplegpuTensor) else gpu.from_torch(grad_output)
    # Reshape to [batch*spatial, channels] then sum
    # ... compose from existing gpu.sum() + reshape
```

- [ ] **Step 2: Test and commit**

---

### Task 10: Grouped Convolution (#19)

**Files:**
- Modify: `crates/core/src/graph.rs` — add `groups: usize` to Conv1d, Conv2d, and backward variants
- Modify: `crates/core/src/kernel_templates.rs` — add groups param to all conv kernel templates
- Modify: `crates/core/src/ops.rs` — add `groups` param + validation
- Modify: `crates/core/src/lazy.rs` — pass groups to dispatch
- Modify: `crates/python/src/backend.rs` — update trait signatures
- Modify: `crates/python/src/metal_backend.rs` — pass groups through
- Modify: `crates/python/src/lib.rs` — add `groups=1` default
- Modify: `python/applegpu_runtime/__init__.py` — no change needed (same function names)
- Modify: `python/applegpu_runtime/torch_backend.py:1341` — remove `groups != 1` check
- Test: `python/tests/test_grouped_conv.py`

- [ ] **Step 1: Write failing test**

```python
# python/tests/test_grouped_conv.py
def test_conv2d_groups2():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 4, 8, 8)
    w = torch.randn(4, 2, 3, 3)  # groups=2: 4 out, 4 in / 2 groups = 2 ic per group
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv2d(x_gpu, gpu.to_applegpu(w), groups=2)
    expected = torch.nn.functional.conv2d(x, w, groups=2)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)

def test_depthwise_conv2d():
    """Depthwise: groups == in_channels == out_channels."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 3, 8, 8)
    w = torch.randn(3, 1, 3, 3)  # groups=3 (depthwise)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv2d(x_gpu, gpu.to_applegpu(w), groups=3)
    expected = torch.nn.functional.conv2d(x, w, groups=3)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)

def test_conv1d_groups():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 4, 16)
    w = torch.randn(4, 2, 3)  # groups=2
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv1d(x_gpu, gpu.to_applegpu(w), groups=2)
    expected = torch.nn.functional.conv1d(x, w, groups=2)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)

def test_conv2d_groups1_regression():
    """groups=1 should still work exactly as before."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv2d(x_gpu, gpu.to_applegpu(w))
    expected = torch.nn.functional.conv2d(x, w)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)
```

- [ ] **Step 2: Implement groups support across all layers**

Key kernel modification (conv2d example):
```metal
constant uint& groups [[buffer(16)]],
// ...
uint out_channels_per_group = out_channels / groups;
uint in_channels_per_group = in_channels / groups;
uint group = oc / out_channels_per_group;
uint ic_start = group * in_channels_per_group;
for (uint ic = 0; ic < in_channels_per_group; ic++) {
    uint ic_abs = ic_start + ic;
    // ... use ic_abs for input indexing, ic for weight indexing
    sum += input[... + ic_abs * in_h * in_w + ...]
         * weight[oc * in_channels_per_group * kh * kw + ic * kh * kw + ...];
}
```

Update OpKind:
```rust
Conv1d { stride: usize, padding: usize, groups: usize },
Conv2d { stride: (usize, usize), padding: (usize, usize), groups: usize },
```

Update all match arms referencing these variants. Update ops.rs validation:
```rust
if in_shape[1] % groups != 0 { return Err(...) }
if w_shape[0] % groups != 0 { return Err(...) }
if w_shape[1] != in_shape[1] / groups { return Err(...) }
```

Pass `groups as u32` as additional uint_param in dispatch.

Update torch_backend.py — remove line 1341's `groups != 1` check, pass groups:
```python
if ndim == 1:
    result = _wrap(gpu.conv1d(_unwrap(input), _unwrap(weight), stride[0], padding[0], groups))
elif ndim == 2:
    result = _wrap(gpu.conv2d(_unwrap(input), _unwrap(weight), stride[0], stride[1], padding[0], padding[1], groups))
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest python/tests/test_grouped_conv.py -v`

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: grouped convolution support (groups > 1) on GPU (#19)"
```

---

### Task 11: Index/Scatter GPU Kernels (#21)

**Files:**
- Modify: `crates/core/src/kernel_templates.rs` — add scatter_write and scatter_add kernels
- Modify: `crates/core/src/graph.rs` — add `ScatterWrite`, `ScatterAdd`
- Modify: `crates/core/src/ops.rs` — add scatter ops
- Modify: `crates/core/src/lazy.rs` — dispatch
- Modify: `crates/core/src/compute.rs` — kernel resolution
- Modify: `crates/python/src/backend.rs`, `metal_backend.rs`, `lib.rs`
- Modify: `python/applegpu_runtime/torch_backend.py:1181-1222` — route integer indices to GPU
- Test: `python/tests/test_index_scatter.py`

- [ ] **Step 1: Write failing test**

```python
# python/tests/test_index_scatter.py
def test_index_put_no_accumulate():
    """Simple scatter write."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.zeros(5, 3)
    idx = torch.tensor([0, 2, 4])
    vals = torch.ones(3, 3)
    x_gpu = gpu.to_applegpu(x)
    x_gpu.index_put_([gpu.to_applegpu(idx)], gpu.to_applegpu(vals))
    expected = x.clone()
    expected.index_put_([idx], vals)
    assert torch.allclose(x_gpu.to_torch_cpu(), expected, atol=1e-6)

def test_index_put_accumulate():
    """Scatter add with duplicate indices."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.zeros(3, 2)
    idx = torch.tensor([0, 0, 1])
    vals = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_gpu = gpu.to_applegpu(x)
    torch.ops.aten.index_put_(x_gpu, [gpu.to_applegpu(idx)], gpu.to_applegpu(vals), accumulate=True)
    expected = x.clone()
    expected.index_put_([idx], vals, accumulate=True)
    assert torch.allclose(x_gpu.to_torch_cpu(), expected, atol=1e-5)

def test_index_tensor_integer():
    """Integer indexing on GPU."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(5, 3)
    idx = torch.tensor([0, 2, 4])
    x_gpu = gpu.to_applegpu(x)
    result = x_gpu[gpu.to_applegpu(idx)]
    expected = x[idx]
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-6)
```

- [ ] **Step 2: Implement scatter_write kernel**

```metal
kernel void scatter_write{s}(
    device const {t}* values, device const int* indices, device {t}* output,
    constant uint& num_indices, constant uint& cols,
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x; uint i = gid.y;
    if (i >= num_indices || col >= cols) return;
    int dst_row = indices[i];
    output[dst_row * cols + col] = values[i * cols + col];
}
```

- [ ] **Step 3: Implement scatter_add kernel (with atomics)**

```metal
inline void atomic_add_float(device float* addr, float val) { /* CAS loop */ }

kernel void scatter_add{s}(
    device const {t}* values, device const int* indices, device float* output,
    constant uint& num_indices, constant uint& cols,
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x; uint i = gid.y;
    if (i >= num_indices || col >= cols) return;
    int dst_row = indices[i];
    atomic_add_float(&output[dst_row * cols + col], float(values[i * cols + col]));
}
```

- [ ] **Step 4: Wire through all layers**

- [ ] **Step 5: Update torch_backend.py index handlers**

In `_op_index_tensor`, detect single integer index on dim 0 for 2D tensors and route to existing `gpu.index_select()`. Keep boolean mask and multi-index as CPU fallback.

In `_op_index_put_`, detect single integer index on dim 0 for 2D tensors:
- `accumulate=False` → `gpu.scatter_write()`
- `accumulate=True` → `gpu.scatter_add()` (pre-zero output not needed, writes into existing buffer)

Add comments clarifying Phase 1 scope:
```python
# Phase 1: single integer index on dim 0, 2D tensors only.
# Multi-index advanced indexing and boolean masks remain CPU fallback.
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest python/tests/test_index_scatter.py -v`

- [ ] **Step 7: Commit**

```bash
git commit -m "feat: GPU scatter write/add and integer index routing (#21)"
```

---

### Task 12: Run full test suite and create PR 3

- [ ] **Step 1: Run full test suite**

```bash
make test
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --title "feat: conv grad_weight, grouped conv, index/scatter (#17,#19,#21)" --body "..."
```

---

## Dependency Graph

```
PR 1 (independent):
  Task 1 (blit copy) → Task 2 (vector norm) → Task 3 (PR)

PR 2 (independent of PR 1):
  Task 4 (pool indices) → Task 5 (exact GELU) → Task 6 (PR)

PR 3 (independent of PR 1 & 2):
  Task 7 (conv1d backward) → Task 8 (conv2d grad_weight) → Task 9 (grad_bias)
  → Task 10 (grouped conv) → Task 11 (index/scatter) → Task 12 (PR)
```

All three PRs can be developed in parallel on separate branches. Tasks within a PR are sequential.

---

## Cross-Cutting Concerns

### Dtype and Edge-Case Tests
Each PR must include at least one Float16 test per new kernel path with `atol=1e-2`. Add single-element tensor tests for vector norm and pool indices. Example:

```python
def test_l2_norm_float16():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(4, 4, dtype=torch.float16)
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=2)
    expected = torch.linalg.vector_norm(x.float(), ord=2)
    assert abs(result.to_torch_cpu().float().item() - expected.item()) < 1e-2
```

### scatter_add Output Dtype
The `scatter_add` kernel accumulates into `device float*` (Float32) regardless of input dtype, matching the `atomic_add_float` CAS pattern from embedding_backward. Callers with Float16/BFloat16 inputs get Float32 output — cast afterward if needed.

### Grouped Conv OpKind Blast Radius
Adding `groups: usize` to `Conv1d`/`Conv2d` OpKind variants breaks every existing match arm. Do this in a single atomic commit within Task 10: update the OpKind definition AND all match sites (graph.rs, lazy.rs, compute.rs, fusion.rs) together.

### README Updates
Per CLAUDE.md policy, each PR's final task should include updating `README.md` to reflect new capabilities (GPU blit copy, L1/L2 norm, pool indices, exact GELU, grouped conv, scatter ops).

### Fusion System and erf()
The exact GELU fusion expression uses `erf()`. Verify that fused kernel templates include `#include <metal_stdlib>` — if not, the `erf` builtin won't be available and Metal will produce a runtime compilation error. Check `generate_fused_msl()` in fusion.rs for the header inclusion.
