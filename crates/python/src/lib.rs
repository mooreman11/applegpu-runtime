mod backend;
#[cfg(target_os = "macos")]
mod metal_backend;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::cell::Cell;
use std::collections::HashMap;

use applegpu_core::tensor::DType;

use backend::Backend;

#[cfg(target_os = "macos")]
type ActiveBackend = metal_backend::MetalBackend;

static BACKEND: once_cell::sync::Lazy<ActiveBackend> = once_cell::sync::Lazy::new(|| {
    ActiveBackend::new()
});

/// Helper: map BackendResult to PyResult.
fn py_err<T>(r: Result<T, String>) -> PyResult<T> {
    r.map_err(|e| PyValueError::new_err(e))
}

/// Helper: wrap a backend tensor ID result into a GpuTensor.
fn wrap_tensor(r: Result<u64, String>) -> PyResult<GpuTensor> {
    let id = py_err(r)?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

/// Convert Python data (list of floats, ints, or bools) to raw bytes for the given dtype.
fn python_data_to_bytes(_py: Python<'_>, data: &Bound<'_, PyAny>, shape: &[usize], dtype: DType) -> PyResult<Vec<u8>> {
    let expected_len: usize = shape.iter().product();

    match dtype {
        DType::Float32 => {
            let vals: Vec<f64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect())
        }
        DType::Float64 => {
            let vals: Vec<f64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        DType::Float16 => {
            use half::f16;
            let vals: Vec<f64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| f16::from_f64(v).to_le_bytes()).collect())
        }
        DType::Int8 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as i8).to_le_bytes()).collect())
        }
        DType::Int16 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as i16).to_le_bytes()).collect())
        }
        DType::Int32 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect())
        }
        DType::Int64 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        DType::UInt8 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as u8).to_le_bytes()).collect())
        }
        DType::UInt32 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as u32).to_le_bytes()).collect())
        }
        DType::Bool => {
            let vals: Vec<bool> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().map(|&v| if v { 1u8 } else { 0u8 }).collect())
        }
        _ => Err(PyValueError::new_err(format!("Unsupported dtype: {:?}", dtype))),
    }
}

/// A GPU tensor. Wraps a lazy tensor ID with automatic memory cleanup.
///
/// Operations on GpuTensor are lazy — they build a computation graph.
/// Data is materialized on the GPU only when you call `.to_list()` or `.eval()`.
///
/// GPU memory is freed automatically when the Python object is garbage-collected,
/// or explicitly via `gpu.destroy(t)`.
#[pyclass(name = "GpuTensor")]
struct GpuTensor {
    id: u64,
    /// Tracks whether this tensor has been destroyed (explicit or via Drop).
    /// Prevents double-destroy.
    destroyed: Cell<bool>,
}

#[pymethods]
impl GpuTensor {
    /// Get the shape as a list of dimensions.
    /// Works on both materialized and lazy tensors.
    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        py_err(BACKEND.shape(self.id))
    }

    /// Get the tensor ID (internal handle).
    #[getter]
    fn id(&self) -> u64 {
        self.id
    }

    /// Get the dtype as a string (all 10 types supported).
    #[getter]
    fn dtype(&self) -> PyResult<String> {
        let dtype = py_err(BACKEND.dtype(self.id))?;
        Ok(dtype.name().to_string())
    }

    /// Read tensor data as a NumPy ndarray with the tensor's shape.
    /// Auto-evaluates if the tensor is lazy.
    /// Supports all 10 dtypes.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dtype = py_err(BACKEND.dtype(self.id))?;
        let bytes = py_err(BACKEND.read_bytes(self.id))?;
        let shape = py_err(BACKEND.shape(self.id))?;

        let np = py.import_bound("numpy")?;
        let np_dtype_name = dtype.name();
        // numpy uses "bool_" not "bool"
        let np_dtype_str = if np_dtype_name == "bool" { "bool_" } else { np_dtype_name };
        let np_dtype = np.call_method1("dtype", (np_dtype_str,))?;

        let pybytes = pyo3::types::PyBytes::new_bound(py, &bytes);
        let arr = np.call_method("frombuffer", (&pybytes, &np_dtype), None)?;
        let reshaped = arr.call_method1("reshape", (shape,))?;
        Ok(reshaped.call_method0("copy")?)
    }

    /// Read tensor data as a flat list of Python values.
    /// Auto-evaluates if the tensor is lazy.
    /// Returns floats for float types, ints for int types, bools for bool type.
    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dtype = py_err(BACKEND.dtype(self.id))?;
        let bytes = py_err(BACKEND.read_bytes(self.id))?;

        match dtype {
            DType::Float32 => {
                let data: Vec<f64> = bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            DType::Float16 => {
                use half::f16;
                let data: Vec<f64> = bytes.chunks_exact(2)
                    .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f64())
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            DType::Float64 => {
                let data: Vec<f64> = bytes.chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            DType::Int8 => {
                let data: Vec<i64> = bytes.iter().map(|&b| b as i8 as i64).collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            DType::Int16 => {
                let data: Vec<i64> = bytes.chunks_exact(2)
                    .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            DType::Int32 => {
                let data: Vec<i64> = bytes.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            DType::Int64 => {
                let data: Vec<i64> = bytes.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            DType::UInt8 => {
                let data: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            DType::UInt32 => {
                let data: Vec<i64> = bytes.chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            DType::Bool => {
                let data: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            _ => Err(PyValueError::new_err("Unsupported dtype")),
        }
    }

    /// Explicitly evaluate this tensor, materializing its result on the GPU.
    fn eval(&self) -> PyResult<()> {
        py_err(BACKEND.eval(self.id))
    }

    // Unary ops

    fn neg(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.neg(self.id)) }
    fn relu(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.relu(self.id)) }
    fn exp(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.exp(self.id)) }
    fn log(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.log(self.id)) }
    fn softmax(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.softmax(self.id)) }
    fn tanh(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.tanh(self.id)) }
    fn gelu(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.gelu(self.id)) }
    fn sqrt(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.sqrt(self.id)) }
    fn abs(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.abs(self.id)) }
    fn sign(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.sign(self.id)) }
    fn transpose(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.transpose(self.id)) }
    fn argmax(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.argmax(self.id)) }
    fn sum(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.sum(self.id)) }
    fn mean(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.mean(self.id)) }
    fn softmax_causal(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.softmax_causal(self.id)) }

    #[pyo3(signature = (gamma, beta, eps=1e-5))]
    fn layer_norm(&self, gamma: &GpuTensor, beta: &GpuTensor, eps: f32) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.layer_norm(self.id, gamma.id, beta.id, eps))
    }

    #[pyo3(signature = (dim0, dim1))]
    fn transpose_dims(&self, dim0: usize, dim1: usize) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.transpose_dims(self.id, dim0, dim1))
    }

    #[pyo3(signature = (exponent))]
    fn pow(&self, exponent: f32) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.pow(self.id, exponent))
    }

    #[pyo3(signature = (min_val, max_val))]
    fn clamp(&self, min_val: f32, max_val: f32) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.clamp(self.id, min_val, max_val))
    }

    #[pyo3(signature = (mask, value))]
    fn masked_fill(&self, mask: &GpuTensor, value: f32) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.masked_fill(self.id, mask.id, value))
    }

    #[pyo3(signature = (diagonal=0))]
    fn triu(&self, diagonal: i32) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.triu(self.id, diagonal))
    }

    #[pyo3(signature = (diagonal=0))]
    fn tril(&self, diagonal: i32) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.tril(self.id, diagonal))
    }

    fn gather(&self, dim: usize, index: &GpuTensor) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.gather(self.id, dim, index.id))
    }

    fn index_select(&self, dim: usize, index: &GpuTensor) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.index_select(self.id, dim, index.id))
    }

    #[pyo3(signature = (scale))]
    fn scalar_mul(&self, scale: f32) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.scalar_mul(self.id, scale))
    }

    #[pyo3(signature = (new_shape))]
    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<GpuTensor> {
        wrap_tensor(BACKEND.reshape(self.id, new_shape))
    }

    // Binary ops

    fn add(&self, other: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.add(self.id, other.id)) }
    fn sub(&self, other: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.sub(self.id, other.id)) }
    fn mul(&self, other: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.mul(self.id, other.id)) }
    fn div(&self, other: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.div(self.id, other.id)) }
    fn matmul(&self, other: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.matmul(self.id, other.id)) }

    // Python operator overloads — all delegate to named methods

    fn __add__(&self, other: &GpuTensor) -> PyResult<GpuTensor> { self.add(other) }
    fn __sub__(&self, other: &GpuTensor) -> PyResult<GpuTensor> { self.sub(other) }
    fn __mul__(&self, other: &GpuTensor) -> PyResult<GpuTensor> { self.mul(other) }
    fn __truediv__(&self, other: &GpuTensor) -> PyResult<GpuTensor> { self.div(other) }
    fn __matmul__(&self, other: &GpuTensor) -> PyResult<GpuTensor> { self.matmul(other) }
    fn __neg__(&self) -> PyResult<GpuTensor> { self.neg() }

    /// Convert to a PyTorch tensor. Auto-evaluates if lazy.
    /// Returns an independent copy (clone) so mutations don't affect GPU data.
    fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Get numpy array via existing to_numpy (handles all dtypes)
        let np_array = self.to_numpy(py)?;

        // Lazy import torch
        let torch = py.import_bound("torch")?;

        // torch.from_numpy(arr).clone() — from_numpy is zero-copy, clone makes independent
        let torch_tensor = torch
            .call_method1("from_numpy", (&np_array,))?
            .call_method0("clone")?;

        Ok(torch_tensor)
    }

    fn __repr__(&self) -> PyResult<String> {
        if self.destroyed.get() {
            return Ok(format!("GpuTensor(id={}, destroyed)", self.id));
        }
        let status = if BACKEND.is_materialized(self.id) {
            "materialized"
        } else if BACKEND.is_pending(self.id) {
            "lazy"
        } else {
            "destroyed"
        };
        match BACKEND.shape(self.id) {
            Ok(shape) => Ok(format!("GpuTensor(id={}, shape={:?}, {})", self.id, shape, status)),
            Err(_) => Ok(format!("GpuTensor(id={}, {})", self.id, status)),
        }
    }
}

/// Drop: best-effort GPU memory cleanup when Python GCs the object.
/// Uses try_lock() to avoid deadlocking if Drop fires while the Mutex is held.
/// Checks `destroyed` flag to prevent double-destroy.
impl Drop for GpuTensor {
    fn drop(&mut self) {
        if self.destroyed.get() {
            return;
        }
        BACKEND.try_destroy(self.id);
    }
}

// ============================================================
// Module-level functions (backward compatibility + utilities)
// ============================================================

#[pyfunction]
fn version() -> &'static str {
    applegpu_core::version()
}

#[pyfunction]
fn init_backend() -> PyResult<HashMap<String, String>> {
    py_err(BACKEND.init())
}

#[pyfunction]
fn device_name() -> PyResult<String> {
    py_err(BACKEND.device_name())
}

#[pyfunction]
fn dtype_size(name: &str) -> PyResult<usize> {
    use applegpu_core::tensor::DType;
    let dt = match name {
        "float16" | "f16" => DType::Float16,
        "float32" | "f32" => DType::Float32,
        "float64" | "f64" => DType::Float64,
        "bfloat16" | "bf16" => DType::BFloat16,
        "int8" | "i8" => DType::Int8,
        "int16" | "i16" => DType::Int16,
        "int32" | "i32" => DType::Int32,
        "int64" | "i64" => DType::Int64,
        "uint8" | "u8" => DType::UInt8,
        "uint32" | "u32" => DType::UInt32,
        "bool" => DType::Bool,
        _ => return Err(PyValueError::new_err(format!("Unknown dtype: {}", name))),
    };
    Ok(dt.size_bytes())
}

/// Create a GpuTensor from a NumPy ndarray (any supported dtype).
/// Data is copied; mutations to the original array do not affect the tensor.
/// Uses direct data_ptr access for C-contiguous arrays (fast path).
#[pyfunction]
#[pyo3(signature = (arr))]
fn from_numpy(_py: Python<'_>, arr: &Bound<'_, pyo3::types::PyAny>) -> PyResult<GpuTensor> {
    let np_dtype_name: String = arr.getattr("dtype")?.getattr("name")?.extract()?;
    let dtype = DType::from_name(&np_dtype_name)
        .ok_or_else(|| PyValueError::new_err(format!(
            "Unsupported numpy dtype: {}. Supported: float16, float32, float64, int8, int16, int32, int64, uint8, uint32, bool",
            np_dtype_name
        )))?;

    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let numel: usize = shape.iter().product();
    let nbytes = numel * dtype.size_bytes();

    // Fast path: read raw bytes via ctypes data pointer for C-contiguous arrays
    let is_c_contiguous: bool = arr.getattr("flags")
        .and_then(|f| f.get_item("C_CONTIGUOUS"))
        .and_then(|v| v.extract())
        .unwrap_or(false);

    if is_c_contiguous && nbytes > 0 {
        let data_ptr: usize = arr.getattr("ctypes")?.getattr("data")?.extract()?;
        // Safety: array is C-contiguous, data_ptr points to nbytes of valid memory,
        // and the `arr` Python reference keeps the data alive during this call.
        let data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, nbytes) };
        wrap_tensor(BACKEND.tensor_from_data(data, shape, dtype))
    } else {
        // Fallback: use tobytes() for non-contiguous or empty arrays
        let bytes: Vec<u8> = arr.call_method0("tobytes")?.extract()?;
        wrap_tensor(BACKEND.tensor_from_data(&bytes, shape, dtype))
    }
}

/// Create a GpuTensor from a PyTorch tensor (any supported dtype).
/// Uses direct data_ptr() access for fast zero-copy reads from CPU tensor memory.
/// Data is copied to GPU; mutations to the original tensor do not affect the GPU tensor.
#[pyfunction]
fn from_torch(py: Python<'_>, tensor: &Bound<'_, pyo3::types::PyAny>) -> PyResult<GpuTensor> {
    // Prepare: detach from autograd, move to CPU, make contiguous
    let prepared = tensor
        .call_method0("detach")?
        .call_method0("cpu")?
        .call_method0("contiguous")?;

    // Map torch dtype to our dtype string
    let torch = py.import_bound("torch")?;
    let tensor_dtype = prepared.getattr("dtype")?;
    let dtype_str = if tensor_dtype.eq(torch.getattr("float16")?)? { "float16" }
        else if tensor_dtype.eq(torch.getattr("float32")?)? { "float32" }
        else if tensor_dtype.eq(torch.getattr("float64")?)? { "float64" }
        else if tensor_dtype.eq(torch.getattr("int8")?)? { "int8" }
        else if tensor_dtype.eq(torch.getattr("int16")?)? { "int16" }
        else if tensor_dtype.eq(torch.getattr("int32")?)? { "int32" }
        else if tensor_dtype.eq(torch.getattr("int64")?)? { "int64" }
        else if tensor_dtype.eq(torch.getattr("uint8")?)? { "uint8" }
        else if tensor_dtype.eq(torch.getattr("bool")?)? { "bool" }
        else {
            return Err(PyValueError::new_err(format!(
                "Unsupported torch dtype: {}", tensor_dtype
            )));
        };

    let dtype = DType::from_name(dtype_str)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown dtype: {}", dtype_str)))?;
    let shape: Vec<usize> = prepared.getattr("shape")?.extract()?;
    let numel: usize = prepared.call_method0("numel")?.extract()?;
    let element_size: usize = prepared.call_method0("element_size")?.extract()?;
    let nbytes = numel * element_size;

    if nbytes > 0 {
        // Fast path: read raw bytes directly from torch tensor's data_ptr().
        // Safety: tensor is CPU + contiguous (guaranteed by .cpu().contiguous()),
        // data_ptr() returns a valid pointer to numel * element_size bytes,
        // and the `prepared` Python reference keeps the data alive.
        let data_ptr: usize = prepared.call_method0("data_ptr")?.extract()?;
        let data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, nbytes) };
        wrap_tensor(BACKEND.tensor_from_data(data, shape, dtype))
    } else {
        // Empty tensor
        wrap_tensor(BACKEND.tensor_from_data(&[], shape, dtype))
    }
}

/// Create a GpuTensor from raw bytes with explicit shape and dtype.
/// This is the fastest path for tensor creation -- no numpy/torch overhead.
#[pyfunction]
fn from_bytes(_py: Python<'_>, data: &[u8], shape: Vec<usize>, dtype: &str) -> PyResult<GpuTensor> {
    let dtype = DType::from_name(dtype)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown dtype: {}", dtype)))?;
    wrap_tensor(BACKEND.tensor_from_data(data, shape, dtype))
}

/// Create a tensor from data. Returns a GpuTensor object.
/// Accepts lists of floats, ints, or bools depending on dtype.
/// Supported dtypes: float16, float32, float64, int8, int16, int32, int64, uint8, uint32, bool.
#[pyfunction]
#[pyo3(signature = (data, shape, dtype=None))]
fn tensor(py: Python<'_>, data: &Bound<'_, PyAny>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<GpuTensor> {
    let dtype_str = dtype.unwrap_or("float32");
    let dtype = DType::from_name(dtype_str)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported dtype: {}", dtype_str)))?;

    let bytes = python_data_to_bytes(py, data, &shape, dtype)?;
    wrap_tensor(BACKEND.tensor_from_data(&bytes, shape, dtype))
}

/// Set resource limits. Pass 0 for any field to make it unlimited.
#[pyfunction]
fn set_limits(max_tensor_size_mb: usize, max_memory_mb: usize, max_tensors: usize) -> PyResult<()> {
    BACKEND.set_limits(max_tensor_size_mb, max_memory_mb, max_tensors);
    Ok(())
}

/// Get current GPU memory usage in bytes.
#[pyfunction]
fn memory_usage() -> PyResult<usize> {
    Ok(BACKEND.memory_usage())
}

/// Get current number of live tensors.
#[pyfunction]
fn tensor_count() -> PyResult<usize> {
    Ok(BACKEND.tensor_count())
}

// Backward-compatible module-level functions that accept GpuTensor

#[pyfunction]
fn to_list(py: Python<'_>, t: &GpuTensor) -> PyResult<PyObject> { t.to_list(py) }

#[pyfunction]
fn shape(t: &GpuTensor) -> PyResult<Vec<usize>> { t.shape() }

#[pyfunction]
fn eval(t: &GpuTensor) -> PyResult<()> { t.eval() }

/// Destroy a tensor, freeing GPU memory.
#[pyfunction]
fn destroy(t: &GpuTensor) -> PyResult<()> {
    if t.destroyed.get() {
        return Ok(());
    }
    py_err(BACKEND.destroy(t.id))?;
    t.destroyed.set(true);
    Ok(())
}

// Backward-compatible module-level op functions

#[pyfunction]
fn add(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { a.add(b) }
#[pyfunction]
fn sub(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { a.sub(b) }
#[pyfunction]
fn mul(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { a.mul(b) }
#[pyfunction]
fn div(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { a.div(b) }
#[pyfunction]
fn matmul(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { a.matmul(b) }
#[pyfunction]
fn neg(t: &GpuTensor) -> PyResult<GpuTensor> { t.neg() }
#[pyfunction]
fn relu(t: &GpuTensor) -> PyResult<GpuTensor> { t.relu() }
#[pyfunction]
fn exp(t: &GpuTensor) -> PyResult<GpuTensor> { t.exp() }
#[pyfunction]
fn log(t: &GpuTensor) -> PyResult<GpuTensor> { t.log() }
#[pyfunction]
fn sqrt(t: &GpuTensor) -> PyResult<GpuTensor> { t.sqrt() }
#[pyfunction]
fn abs(t: &GpuTensor) -> PyResult<GpuTensor> { t.abs() }
#[pyfunction]
fn sign(t: &GpuTensor) -> PyResult<GpuTensor> { t.sign() }
#[pyfunction]
fn pow(t: &GpuTensor, exponent: f32) -> PyResult<GpuTensor> { t.pow(exponent) }
#[pyfunction]
fn clamp(t: &GpuTensor, min_val: f32, max_val: f32) -> PyResult<GpuTensor> { t.clamp(min_val, max_val) }
#[pyfunction]
fn where_cond(cond: &GpuTensor, x: &GpuTensor, y: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.where_cond(cond.id, x.id, y.id))
}
#[pyfunction]
fn masked_fill(t: &GpuTensor, mask: &GpuTensor, value: f32) -> PyResult<GpuTensor> { t.masked_fill(mask, value) }
#[pyfunction]
#[pyo3(signature = (t, diagonal=0))]
fn triu(t: &GpuTensor, diagonal: i32) -> PyResult<GpuTensor> { t.triu(diagonal) }
#[pyfunction]
#[pyo3(signature = (t, diagonal=0))]
fn tril(t: &GpuTensor, diagonal: i32) -> PyResult<GpuTensor> { t.tril(diagonal) }
#[pyfunction]
fn scalar_mul(t: &GpuTensor, scale: f32) -> PyResult<GpuTensor> { t.scalar_mul(scale) }
#[pyfunction]
fn reshape(t: &GpuTensor, new_shape: Vec<usize>) -> PyResult<GpuTensor> { t.reshape(new_shape) }
#[pyfunction]
fn softmax(t: &GpuTensor) -> PyResult<GpuTensor> { t.softmax() }
#[pyfunction]
fn transpose(t: &GpuTensor) -> PyResult<GpuTensor> { t.transpose() }
#[pyfunction]
fn transpose_dims(t: &GpuTensor, dim0: usize, dim1: usize) -> PyResult<GpuTensor> { t.transpose_dims(dim0, dim1) }

#[pyfunction]
fn tanh(t: &GpuTensor) -> PyResult<GpuTensor> { t.tanh() }

#[pyfunction]
fn gelu(t: &GpuTensor) -> PyResult<GpuTensor> { t.gelu() }

#[pyfunction]
#[pyo3(signature = (input, gamma, beta, eps=1e-5))]
fn layer_norm(input: &GpuTensor, gamma: &GpuTensor, beta: &GpuTensor, eps: f32) -> PyResult<GpuTensor> {
    input.layer_norm(gamma, beta, eps)
}

#[pyfunction]
fn embedding(weights: &GpuTensor, indices: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.embedding(weights.id, indices.id))
}

#[pyfunction]
fn gather(input: &GpuTensor, dim: usize, index: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.gather(input.id, dim, index.id))
}

#[pyfunction]
fn index_select(input: &GpuTensor, dim: usize, index: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.index_select(input.id, dim, index.id))
}

#[pyfunction]
fn attention(q: &GpuTensor, k: &GpuTensor, v: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.attention(q.id, k.id, v.id))
}

#[pyfunction]
fn slice(t: &GpuTensor, dim: usize, start: usize, end: usize) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.slice(t.id, dim, start, end))
}

#[pyfunction]
fn concat(a: &GpuTensor, b: &GpuTensor, dim: usize) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.concat(a.id, b.id, dim))
}

#[pyfunction]
fn add_bias(input: &GpuTensor, bias: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.add_bias(input.id, bias.id))
}

#[pyfunction]
fn softmax_causal(t: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.softmax_causal(t.id))
}

#[pyfunction]
fn argmax(t: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.argmax(t.id))
}

#[pyfunction]
fn sum(t: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.sum(t.id))
}

#[pyfunction]
fn mean(t: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.mean(t.id))
}

#[pyfunction]
fn attention_causal(q: &GpuTensor, k: &GpuTensor, v: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.attention_causal(q.id, k.id, v.id))
}

#[pyfunction]
#[pyo3(signature = (input, weight, stride=1, padding=0))]
fn conv1d(input: &GpuTensor, weight: &GpuTensor, stride: usize, padding: usize) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.conv1d(input.id, weight.id, stride, padding))
}

#[pyfunction]
#[pyo3(signature = (input, weight, stride_h=1, stride_w=1, pad_h=0, pad_w=0))]
fn conv2d(input: &GpuTensor, weight: &GpuTensor, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.conv2d(input.id, weight.id, (stride_h, stride_w), (pad_h, pad_w)))
}

#[pyfunction]
#[pyo3(signature = (input, running_mean, running_var, weight, bias, eps=1e-5))]
fn batch_norm(input: &GpuTensor, running_mean: &GpuTensor, running_var: &GpuTensor, weight: &GpuTensor, bias: &GpuTensor, eps: f32) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.batch_norm(input.id, running_mean.id, running_var.id, weight.id, bias.id, eps))
}

#[pyfunction]
#[pyo3(signature = (input, kh=2, kw=2, stride_h=0, stride_w=0, pad_h=0, pad_w=0))]
fn max_pool2d(input: &GpuTensor, kh: usize, kw: usize, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize) -> PyResult<GpuTensor> {
    let sh = if stride_h == 0 { kh } else { stride_h };
    let sw = if stride_w == 0 { kw } else { stride_w };
    wrap_tensor(BACKEND.max_pool2d(input.id, (kh, kw), (sh, sw), (pad_h, pad_w)))
}

#[pyfunction]
#[pyo3(signature = (input, kh=2, kw=2, stride_h=0, stride_w=0, pad_h=0, pad_w=0))]
fn avg_pool2d(input: &GpuTensor, kh: usize, kw: usize, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize) -> PyResult<GpuTensor> {
    let sh = if stride_h == 0 { kh } else { stride_h };
    let sw = if stride_w == 0 { kw } else { stride_w };
    wrap_tensor(BACKEND.avg_pool2d(input.id, (kh, kw), (sh, sw), (pad_h, pad_w)))
}

// ============================================================
// Scheduler bindings
// ============================================================

#[pyfunction]
fn register_container(
    priority: &str,
    max_memory_mb: usize,
    max_tensors: usize,
    max_pending: usize,
) -> PyResult<u64> {
    py_err(BACKEND.register_container(priority, max_memory_mb, max_tensors, max_pending))
}

#[pyfunction]
fn deregister_container(container_id: u64) -> PyResult<Vec<u64>> {
    py_err(BACKEND.deregister_container(container_id))
}

#[pyfunction]
fn pause_container(container_id: u64) -> PyResult<()> {
    py_err(BACKEND.pause_container(container_id))
}

#[pyfunction]
fn resume_container(container_id: u64) -> PyResult<()> {
    py_err(BACKEND.resume_container(container_id))
}

#[pyfunction]
fn submit_job(container_id: u64, t: &GpuTensor) -> PyResult<u64> {
    py_err(BACKEND.submit_job(container_id, t.id))
}

#[pyfunction]
fn run_next() -> PyResult<Option<u64>> {
    py_err(BACKEND.run_next())
}

#[pyfunction]
fn job_status(job_id: u64) -> PyResult<String> {
    py_err(BACKEND.job_status(job_id))
}

#[pyfunction]
fn container_usage(container_id: u64) -> PyResult<(usize, usize)> {
    py_err(BACKEND.container_usage(container_id))
}

#[pyfunction]
fn global_usage() -> PyResult<(usize, usize)> {
    Ok(BACKEND.global_usage())
}

#[pyfunction]
fn queue_depth() -> PyResult<usize> {
    Ok(BACKEND.queue_depth())
}

#[pyfunction]
fn pool_stats() -> PyResult<HashMap<String, usize>> {
    Ok(BACKEND.pool_stats())
}

#[pyfunction]
fn pool_drain() -> PyResult<()> {
    BACKEND.pool_drain();
    Ok(())
}

#[pyfunction]
fn set_pool_watermark(mb: usize) -> PyResult<()> {
    BACKEND.set_pool_watermark(mb);
    Ok(())
}

#[pyfunction]
fn softmax_backward(grad_output: &GpuTensor, output: &GpuTensor) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.softmax_backward(grad_output.id, output.id))
}

#[pyfunction]
#[pyo3(signature = (grad_output, input, gamma, eps=1e-5))]
fn layer_norm_backward(grad_output: &GpuTensor, input: &GpuTensor, gamma: &GpuTensor, eps: f32) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.layer_norm_backward(grad_output.id, input.id, gamma.id, eps))
}

#[pyfunction]
fn conv2d_backward_input(grad_output: &GpuTensor, weight: &GpuTensor, in_h: usize, in_w: usize, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.conv2d_backward_input(grad_output.id, weight.id, in_h, in_w, (stride_h, stride_w), (pad_h, pad_w)))
}

#[pyfunction]
fn embedding_backward(grad_output: &GpuTensor, indices: &GpuTensor, num_weights: usize) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.embedding_backward(grad_output.id, indices.id, num_weights))
}

#[pyfunction]
#[pyo3(signature = (grad_output, weight, running_var, eps=1e-5))]
fn batch_norm_backward(grad_output: &GpuTensor, weight: &GpuTensor, running_var: &GpuTensor, eps: f32) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.batch_norm_backward(grad_output.id, weight.id, running_var.id, eps))
}

#[pymodule]
fn applegpu_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GpuTensor>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(init_backend, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_size, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(from_torch, m)?)?;
    m.add_function(wrap_pyfunction!(from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(eval, m)?)?;
    m.add_function(wrap_pyfunction!(to_list, m)?)?;
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(destroy, m)?)?;
    m.add_function(wrap_pyfunction!(set_limits, m)?)?;
    m.add_function(wrap_pyfunction!(memory_usage, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_count, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(neg, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(pow, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;
    m.add_function(wrap_pyfunction!(where_cond, m)?)?;
    m.add_function(wrap_pyfunction!(masked_fill, m)?)?;
    m.add_function(wrap_pyfunction!(triu, m)?)?;
    m.add_function(wrap_pyfunction!(tril, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_mul, m)?)?;
    m.add_function(wrap_pyfunction!(reshape, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(transpose_dims, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(embedding, m)?)?;
    m.add_function(wrap_pyfunction!(gather, m)?)?;
    m.add_function(wrap_pyfunction!(index_select, m)?)?;
    m.add_function(wrap_pyfunction!(attention, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(slice, m)?)?;
    m.add_function(wrap_pyfunction!(concat, m)?)?;
    m.add_function(wrap_pyfunction!(add_bias, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_causal, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(attention_causal, m)?)?;
    m.add_function(wrap_pyfunction!(register_container, m)?)?;
    m.add_function(wrap_pyfunction!(deregister_container, m)?)?;
    m.add_function(wrap_pyfunction!(pause_container, m)?)?;
    m.add_function(wrap_pyfunction!(resume_container, m)?)?;
    m.add_function(wrap_pyfunction!(submit_job, m)?)?;
    m.add_function(wrap_pyfunction!(run_next, m)?)?;
    m.add_function(wrap_pyfunction!(job_status, m)?)?;
    m.add_function(wrap_pyfunction!(container_usage, m)?)?;
    m.add_function(wrap_pyfunction!(global_usage, m)?)?;
    m.add_function(wrap_pyfunction!(queue_depth, m)?)?;
    m.add_function(wrap_pyfunction!(pool_stats, m)?)?;
    m.add_function(wrap_pyfunction!(pool_drain, m)?)?;
    m.add_function(wrap_pyfunction!(set_pool_watermark, m)?)?;
    m.add_function(wrap_pyfunction!(conv1d, m)?)?;
    m.add_function(wrap_pyfunction!(conv2d, m)?)?;
    m.add_function(wrap_pyfunction!(batch_norm, m)?)?;
    m.add_function(wrap_pyfunction!(max_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(avg_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_backward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm_backward, m)?)?;
    m.add_function(wrap_pyfunction!(conv2d_backward_input, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_backward, m)?)?;
    m.add_function(wrap_pyfunction!(batch_norm_backward, m)?)?;
    Ok(())
}
