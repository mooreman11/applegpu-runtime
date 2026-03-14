# NumPy Adapter

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Copy-based NumPy interop (from_numpy, to_numpy). Zero-copy deferred to backlog.

## Overview

Add `gpu.from_numpy(arr)` and `tensor.to_numpy()` to applegpu_runtime for seamless data interchange with the NumPy ecosystem. All transfers copy data (NumPy → Metal buffer on import, Metal buffer → NumPy on export). Only `np.float32` is supported, matching the current f32-only kernel pipeline.

## API

```python
import applegpu_runtime as gpu
import numpy as np

# NumPy → GPU (always copies)
arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
t = gpu.from_numpy(arr)

# GPU → NumPy (auto-evals if lazy, always copies)
result = t.to_numpy()   # np.ndarray, shape preserved, dtype=float32
```

## from_numpy

`gpu.from_numpy(arr: numpy.ndarray) -> GpuTensor`

1. Validate `arr.dtype == np.float32`. Error: `"Expected float32 array, got {dtype}. Use arr.astype(np.float32)."`
2. Validate C-contiguous (`arr.flags['C_CONTIGUOUS']`). Error: `"Array must be C-contiguous. Use np.ascontiguousarray()."`
3. Extract shape as `Vec<usize>` from `arr.shape()`.
4. Extract data as `&[f32]` from `arr.as_slice()`.
5. Call existing `Tensor::from_f32(device, shape, data)` → `insert_tensor()` → return `GpuTensor`.

This reuses the existing tensor creation pipeline, so resource limits (scheduler) are enforced automatically.

## to_numpy

`GpuTensor.to_numpy() -> numpy.ndarray` (method on GpuTensor)

1. If tensor is lazy (pending), auto-eval (same pattern as `to_list()`).
2. Read f32 data via `rt.read_f32(id)` → `Vec<f32>`.
3. Get shape via `rt.shape(id)`.
4. Construct `PyArrayDyn<f32>` with the correct shape. Use `PyArray::from_vec` then reshape, or `PyArray::from_shape_vec`.
5. Return the NumPy array.

GIL is released during the eval step via `py.allow_threads()` for multi-threaded Python programs.

## Implementation

### Dependency

Add `numpy` (pyo3-numpy) to `crates/python/Cargo.toml`, pinned to match the pyo3 version:

```toml
[dependencies]
numpy = { version = "0.22", package = "pyo3-numpy" }
```

### Changes

All changes in `crates/python/src/lib.rs`:

1. `#[pyfunction] fn from_numpy(arr: &Bound<'_, PyAny>) -> PyResult<GpuTensor>` — module-level function
2. `#[pymethods] fn to_numpy(&self, py: Python<'_>) -> PyResult<Py<numpy::PyArrayDyn<f32>>>` — method on GpuTensor

Plus update `__init__.py` with `from_numpy` export and `__all__` entry.

No Rust core changes. No Swift changes.

### Return type for to_numpy

Uses `PyArrayDyn<f32>` (dynamically-dimensioned NumPy array) to preserve N-dimensional shapes. This is the standard pyo3-numpy type for arrays whose dimensionality is not known at compile time.

## Validation

- **dtype**: Only `np.float32`. All other dtypes rejected with actionable error message.
- **contiguity**: Only C-contiguous arrays. Fortran-order or sliced arrays rejected with actionable error.
- **empty arrays**: `np.array([], dtype=np.float32)` with shape `[0]` is accepted (creates zero-element tensor).

## Error Handling

- `from_numpy` with wrong dtype: `ValueError("Expected float32 array, got float64. Use arr.astype(np.float32).")`
- `from_numpy` with non-contiguous: `ValueError("Array must be C-contiguous. Use np.ascontiguousarray().")`
- `from_numpy` resource limit exceeded: propagated from scheduler (existing behavior)
- `to_numpy` on destroyed tensor: existing error propagation from `read_f32`

## Testing Strategy (TDD)

### Python Tests (test_numpy.py, 7 tests)

1. `test_from_numpy_roundtrip` — create float32 array, from_numpy, to_numpy, compare values
2. `test_from_numpy_preserves_shape` — 2D array (3,4), verify to_numpy returns same shape
3. `test_from_numpy_rejects_non_float32` — float64 array raises ValueError with helpful message
4. `test_from_numpy_rejects_non_contiguous` — Fortran-order array raises ValueError
5. `test_to_numpy_auto_evals` — from_numpy two arrays, add them (lazy), to_numpy triggers eval
6. `test_from_numpy_copies_data` — modify original array after from_numpy, tensor unaffected
7. `test_from_numpy_empty_array` — empty float32 array accepted, to_numpy returns empty

## Backlog: Zero-Copy from_numpy

Deferred to a future phase. Requires:
- Swift FFI: new `gpu_bridge_create_buffer_no_copy` using Metal's `makeBuffer(bytesNoCopy:length:options:deallocator:)`
- Rust core: new `Buffer` constructor that skips deallocation on Drop (memory owned by NumPy)
- Page-alignment validation (Metal requires page-aligned pointers for bytesNoCopy)
- Python: `Py<PyAny>` reference on GpuTensor to prevent NumPy GC, plus `arr.flags.writeable = False` to prevent data races
- Fallback to copy when alignment requirements not met

This is three-layer work and does not block shipping.
