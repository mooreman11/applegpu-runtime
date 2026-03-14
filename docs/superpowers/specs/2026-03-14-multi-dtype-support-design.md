# Multi-DType Tensor Support

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Generic tensor constructors, 10 dtype support in storage/interop layer. Compute kernels unchanged (f32/f16 only). Compute kernel expansion is the next phase.

## Overview

Add full multi-dtype tensor support to applegpu_runtime. All 10 supported types get tensor creation, storage, NumPy/PyTorch interop, and Python API support. GPU compute remains gated to Float32/Float16 — multi-dtype compute kernels are the next phase.

This builds the foundation that future work (embedding with Int32 indices, multi-dtype kernels, gather/scatter with Int64) depends on.

## Supported Types

| DType | NumPy | PyTorch | Rust | Size | Compute? |
|-------|-------|---------|------|------|----------|
| Float16 | float16 | float16 | u16 (half) | 2 | Yes |
| Float32 | float32 | float32 | f32 | 4 | Yes |
| Float64 | float64 | float64 | f64 | 8 | No (future) |
| Int8 | int8 | int8 | i8 | 1 | No (future) |
| Int16 | int16 | int16 | i16 | 2 | No (future) |
| Int32 | int32 | int32 | i32 | 4 | No (future) |
| Int64 | int64 | int64 | i64 | 8 | No (future) |
| UInt8 | uint8 | uint8 | u8 | 1 | No (future) |
| UInt32 | uint32 | uint32 | u32 | 4 | No (future) |
| Bool | bool_ | bool | u8 | 1 | No (future) |

BFloat16 excluded (no Apple Silicon hardware). On the backlog.

## Rust Core Changes

### Generic Tensor Constructors

```rust
impl Tensor {
    /// Create a tensor from raw bytes + dtype. Validates byte count.
    pub fn from_data(device: &Device, shape: Vec<usize>, dtype: DType, data: &[u8]) -> Result<Self> {
        let expected = shape.iter().product::<usize>() * dtype.size_bytes();
        if data.len() != expected {
            return Err(GpuError::InvalidTensor(format!(
                "Shape {:?} with {:?} expects {} bytes but got {}",
                shape, dtype, expected, data.len()
            )));
        }
        let buffer = Buffer::from_bytes(device, data)?;
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Ok(Tensor {
            meta: TensorMeta { id, shape: Shape::new(shape), dtype, location: TensorLocation::Shared },
            buffer,
        })
    }

    /// Allocate an uninitialized tensor of any dtype.
    pub fn empty(device: &Device, shape: Vec<usize>, dtype: DType) -> Result<Self> {
        let size_bytes = shape.iter().product::<usize>() * dtype.size_bytes();
        let buffer = Buffer::new(device, size_bytes)?;
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Ok(Tensor {
            meta: TensorMeta { id, shape: Shape::new(shape), dtype, location: TensorLocation::Shared },
            buffer,
        })
    }

    /// Read tensor data as raw bytes. Uses logical size (not physical buffer size).
    pub fn as_bytes(&self) -> &[u8] {
        let byte_count = self.meta.size_bytes();
        unsafe { std::slice::from_raw_parts(self.buffer.contents(), byte_count) }
    }
}
```

Existing `from_f32`, `from_f16`, `empty_f32`, `empty_f16` become thin wrappers around `from_data`/`empty`. They stay as public convenience methods.

### LazyRuntime

```rust
/// Read tensor data as raw bytes. Requires materialized tensor.
pub fn read_bytes(&self, id: u64) -> Result<Vec<u8>> {
    let t = self.get_tensor(id)?;
    Ok(t.as_bytes().to_vec())
}
```

Existing `read_f32`/`read_f16` stay as convenience wrappers.

### DType helpers

Add to `DType`:
```rust
impl DType {
    /// Whether this dtype has GPU compute kernels.
    pub fn is_compute_supported(&self) -> bool {
        matches!(self, DType::Float32 | DType::Float16)
    }

    /// Map from string name to DType.
    pub fn from_name(name: &str) -> Option<DType> {
        match name {
            "float16" | "f16" => Some(DType::Float16),
            "float32" | "f32" => Some(DType::Float32),
            "float64" | "f64" => Some(DType::Float64),
            "int8" | "i8" => Some(DType::Int8),
            "int16" | "i16" => Some(DType::Int16),
            "int32" | "i32" => Some(DType::Int32),
            "int64" | "i64" => Some(DType::Int64),
            "uint8" | "u8" => Some(DType::UInt8),
            "uint32" | "u32" => Some(DType::UInt32),
            "bool" => Some(DType::Bool),
            _ => None,
        }
    }

    /// Map to string name.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt32 => "uint32",
            DType::Bool => "bool",
            DType::BFloat16 => "bfloat16",
        }
    }

    /// Map to numpy dtype name.
    pub fn numpy_name(&self) -> &'static str {
        match self {
            DType::Bool => "bool",
            other => other.name(),
        }
    }
}
```

### Compute dtype validation

In `ops.rs`, add validation at graph-build time:

```rust
fn validate_compute_dtype(dtype: DType) -> Result<()> {
    if !dtype.is_compute_supported() {
        return Err(GpuError::InvalidTensor(format!(
            "No compute kernel for {:?}. Supported: Float32, Float16.", dtype
        )));
    }
    Ok(())
}
```

Called at the top of `lazy_binary_op`, `lazy_unary_op`, `matmul`, `softmax`, `transpose`, `scalar_mul`. This fails fast at graph-build time, not at eval time.

**Note for future multi-dtype compute phase:** When new dtype kernels are added, simply expand `is_compute_supported()` to include them. The validation call sites don't change.

### eval_remote dtype handling

For v1: `eval_remote` validates that tensor dtype is Float32 before serialization. Error: "Remote eval only supports Float32 tensors." Full multi-dtype IPC is a backlog item (requires updating the wire format in serial.rs).

## Python Changes

### tensor() function — generic via PyAny

```rust
#[pyfunction]
#[pyo3(signature = (data, shape, dtype=None))]
fn tensor(py: Python<'_>, data: &Bound<'_, PyAny>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<GpuTensor> {
    let dtype_str = dtype.unwrap_or("float32");
    let dtype = DType::from_name(dtype_str)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported dtype: {}", dtype_str)))?;

    let runtime = get_device_runtime()?;
    let bytes = python_data_to_bytes(py, data, &shape, dtype)?;
    let t = Tensor::from_data(&runtime.device, shape, dtype, &bytes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let id = t.meta.id;
    RUNTIME_LAZY.lock().unwrap().insert_tensor(t)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}
```

`python_data_to_bytes` helper: inspects Python data type and converts to raw bytes:
- Python float list → cast each to target float type (f16/f32/f64) → bytes
- Python int list → cast each to target int type (i8/i16/i32/i64/u8/u32) → bytes
- Python bool list → convert to u8 (0/1) → bytes
- Uses `half` crate for f16, standard casts for everything else
- Int64 values preserved exactly (no f64 round-trip)

### dtype getter — all 10 types

```rust
#[getter]
fn dtype(&self) -> PyResult<String> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    let dtype = rt.dtype(self.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(dtype.name().to_string())
}
```

### to_list — returns proper Python types

```rust
fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
    // auto-eval ...
    let rt = RUNTIME_LAZY.lock().unwrap();
    let dtype = rt.dtype(self.id)?;
    let bytes = rt.read_bytes(self.id)?;
    drop(rt);

    match dtype {
        DType::Float32 => { /* cast bytes to f32, return list of Python float */ }
        DType::Float16 => { /* cast bytes to u16, convert via half to f64, return list of float */ }
        DType::Float64 => { /* cast bytes to f64, return list of float */ }
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 |
        DType::UInt8 | DType::UInt32 => { /* cast bytes to native int type, return list of Python int */ }
        DType::Bool => { /* cast bytes to u8, return list of Python bool */ }
        _ => Err(...)
    }
}
```

Returns `PyObject` (not `Vec<f64>`) so each dtype gets the correct Python type.

### from_numpy — generic via raw bytes

```rust
fn from_numpy(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<GpuTensor> {
    // Get numpy dtype name
    let np_dtype_name = arr.getattr("dtype")?.getattr("name")?.extract::<String>()?;
    let dtype = DType::from_name(&np_dtype_name)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported numpy dtype: {}", np_dtype_name)))?;

    // Optimized path for f32/f16 (zero-copy read via as_slice)
    if dtype == DType::Float32 || dtype == DType::Float16 {
        return from_numpy_float(py, arr, dtype);  // existing optimized path
    }

    // Generic path for all other types
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let bytes: Vec<u8> = arr.call_method0("tobytes")?.extract()?;

    let runtime = get_device_runtime()?;
    let t = Tensor::from_data(&runtime.device, shape, dtype, &bytes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let id = t.meta.id;
    RUNTIME_LAZY.lock().unwrap().insert_tensor(t)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}
```

Keeps optimized f32/f16 paths (zero-copy `as_slice`). Uses `tobytes()` generic path for the other 8 types.

### to_numpy — generic via raw bytes

```rust
fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    // auto-eval ...
    let rt = RUNTIME_LAZY.lock().unwrap();
    let dtype = rt.dtype(self.id)?;
    let bytes = rt.read_bytes(self.id)?;
    let shape = rt.shape(self.id)?;
    drop(rt);

    let np = py.import("numpy")?;
    let np_dtype = np.call_method1("dtype", (dtype.numpy_name(),))?;
    let byte_obj = PyBytes::new(py, &bytes);
    let arr = np.call_method1("frombuffer", (byte_obj, np_dtype))?;
    let reshaped = arr.call_method1("reshape", (shape,))?;
    // Return a copy so the bytes can be freed
    Ok(reshaped.call_method0("copy")?)
}
```

One code path for all 10 types.

### from_torch — widened dtype gate

Remove the float-only restriction. Accept all dtypes that from_numpy supports. The `.detach().cpu().contiguous().numpy()` chain works for all torch dtypes.

### to_torch — unchanged

Routes through `to_numpy()` which now handles all types. `torch.from_numpy().clone()` preserves dtype.

## Backward Compatibility

- Existing `from_f32`, `from_f16`, `read_f32`, `read_f16` stay as convenience methods
- `tensor([1.0, 2.0], shape=[2])` still defaults to float32
- `from_numpy(float32_array)` still uses the optimized path
- All existing tests pass unchanged
- GPU compute still restricted to f32/f16 — new types are storage/interop only

## Testing Strategy (TDD)

### Rust Unit Tests (~8 tests)

1. `test_from_data_all_dtypes` — create tensor for each of 10 dtypes, verify size_bytes and dtype
2. `test_from_data_validates_byte_count` — wrong byte count → error
3. `test_empty_all_dtypes` — empty tensor for each dtype, verify allocation
4. `test_as_bytes_roundtrip` — from_data then as_bytes, compare
5. `test_as_bytes_uses_logical_size` — pooled buffer, as_bytes returns logical count
6. `test_dtype_from_name_all` — all 10 names map correctly
7. `test_dtype_name_roundtrip` — name → from_name → name for all types
8. `test_compute_rejects_non_float` — int32 binary op → error at record time

### Python Tests (~10 tests)

1. `test_tensor_all_dtypes` — create with each dtype string, verify dtype getter
2. `test_from_numpy_all_dtypes` — roundtrip for all 10 numpy dtypes
3. `test_to_numpy_preserves_dtype` — verify numpy array has correct dtype
4. `test_from_torch_all_dtypes` — roundtrip for all 10 torch dtypes
5. `test_to_list_float_types` — float32/float16/float64 → list of Python float
6. `test_to_list_int_types` — int8/int16/int32/int64/uint8/uint32 → list of Python int
7. `test_to_list_bool` — bool → list of Python True/False
8. `test_dtype_getter_all` — verify string for each type
9. `test_unsupported_compute_error` — int32 + int32 → ValueError at record time
10. `test_backward_compat` — existing float32 tensor creation and eval unchanged

## Backlog (not in scope)

- BFloat16 dtype support (no Apple Silicon hardware)
- Multi-dtype compute kernels (next phase — touches all MSL kernels + dispatch)
- Multi-dtype IPC serialization (eval_remote wire format)
- Mixed-dtype compute ops (e.g., int32 indices + float32 weights in embedding)
