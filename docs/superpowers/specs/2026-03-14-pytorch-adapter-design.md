# PyTorch Adapter

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Copy-based data interchange (from_torch, to_torch). Custom device backend and autograd deferred to backlog.

## Overview

Add `gpu.from_torch(tensor)` and `tensor.to_torch()` to applegpu_runtime for data interchange with PyTorch. Routes through the existing NumPy adapter internally (torch→numpy is zero-copy for contiguous CPU tensors). Only `torch.float32` supported.

## API

```python
import torch
import applegpu_runtime as gpu

# PyTorch → GPU (copies to Metal buffer)
t_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
t = gpu.from_torch(t_torch)

# GPU → PyTorch (auto-evals if lazy, returns torch.Tensor)
result = t.to_torch()
```

## from_torch

`gpu.from_torch(tensor: PyAny) -> GpuTensor`

1. Lazy-import `torch` via `py.import("torch")`.
2. Validate dtype: compare `tensor.dtype` against `torch.float32` via PyO3 object equality. Error: `"Expected float32 tensor, got {dtype}. Use tensor.float()."`
3. Prepare: `tensor.detach().cpu().contiguous()` — removes autograd, moves to CPU if on MPS/CUDA (blocks on GPU sync), ensures C-contiguous layout.
4. Convert: `.numpy()` — zero-copy view for contiguous CPU float32 tensors.
5. Delegate to existing `from_numpy()` which copies data to Metal buffer.

The `.cpu()` call on an MPS/CUDA tensor triggers a GPU-to-CPU sync. This is expected and unavoidable.

## to_torch

`GpuTensor.to_torch(py) -> torch.Tensor`

1. Call `self.to_numpy(py)` — auto-evals if lazy, returns `PyArrayDyn<f32>`.
2. Lazy-import `torch`.
3. `torch.from_numpy(np_array).clone()` — `from_numpy` is zero-copy view, `.clone()` ensures independent storage so the NumPy array can be freed.

## Implementation

All in `crates/python/src/lib.rs`:
- `#[pyfunction] fn from_torch(py, tensor: &Bound<'_, PyAny>) -> PyResult<GpuTensor>` — module-level
- `#[pymethods] fn to_torch(&self, py) -> PyResult<Py<PyAny>>` — method on GpuTensor

No new Rust dependencies. No Rust core changes. No Swift changes. PyTorch accessed via `PyAny` + `py.import("torch")`.

### Lazy import pattern

```rust
let torch = py.import("torch")?;
```

Called inside each function body, never cached in a static. Python modules can be reloaded.

### dtype validation

```rust
let float32 = torch.getattr("float32")?;
let tensor_dtype = tensor.getattr("dtype")?;
if !tensor_dtype.eq(&float32)? {
    return Err(PyValueError::new_err(format!(
        "Expected float32 tensor, got {}. Use tensor.float().", tensor_dtype
    )));
}
```

Object equality, not string comparison.

## Error Handling

- Wrong dtype: `ValueError("Expected float32 tensor, got {dtype}. Use tensor.float().")`
- PyTorch not installed: `ModuleNotFoundError` from `py.import("torch")` — propagates naturally
- Resource limit exceeded: propagated from scheduler via `from_numpy` path
- Destroyed tensor: propagated from `to_numpy`

## Testing Strategy (TDD)

### Python Tests (test_torch.py, 9 tests)

1. `test_from_torch_roundtrip` — create torch tensor, from_torch, to_torch, compare
2. `test_from_torch_preserves_shape` — 2D tensor (3,4), verify to_torch shape
3. `test_from_torch_rejects_non_float32` — float64 raises ValueError
4. `test_to_torch_auto_evals` — lazy tensor auto-materializes on to_torch
5. `test_from_torch_copies_data` — modify original torch tensor, gpu tensor unaffected
6. `test_from_torch_non_contiguous` — transposed tensor handled (auto-contiguous)
7. `test_from_torch_requires_grad` — tensor with requires_grad=True handled (auto-detach)
8. `test_from_torch_scalar` — 0-dim tensor roundtrips correctly
9. `test_from_torch_empty` — empty tensor works

## Backlog

- Custom PyTorch device backend (`torch.device('applegpu')`) — requires PyTorch's PrivateUse1 device API
- Autograd integration (`torch.autograd.Function` wrappers) — requires backward ops
- Direct `data_ptr()` path for zero-copy (when Metal bytesNoCopy is implemented)
