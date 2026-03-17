mod backend;
#[cfg(target_os = "macos")]
mod metal_backend;
#[cfg(target_os = "linux")]
mod socket_backend;

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use std::cell::Cell;
use std::collections::HashMap;

use backend::{Backend, BackendDType};

use once_cell::sync::Lazy;

static PAGE_SIZE: Lazy<usize> = Lazy::new(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize });
static ALIGNED_ARRAY_CLASS: once_cell::sync::OnceCell<PyObject> = once_cell::sync::OnceCell::new();
fn system_page_size() -> usize { *PAGE_SIZE }

#[cfg(target_os = "macos")]
type ActiveBackend = metal_backend::MetalBackend;
#[cfg(target_os = "linux")]
type ActiveBackend = socket_backend::SocketBackend;

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
fn python_data_to_bytes(_py: Python<'_>, data: &Bound<'_, PyAny>, shape: &[usize], dtype: BackendDType) -> PyResult<Vec<u8>> {
    let expected_len: usize = shape.iter().product();

    match dtype {
        BackendDType::Float32 => {
            let vals: Vec<f64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect())
        }
        BackendDType::Float64 => {
            let vals: Vec<f64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        BackendDType::Float16 => {
            use half::f16;
            let vals: Vec<f64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| f16::from_f64(v).to_le_bytes()).collect())
        }
        BackendDType::Int8 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as i8).to_le_bytes()).collect())
        }
        BackendDType::Int16 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as i16).to_le_bytes()).collect())
        }
        BackendDType::Int32 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect())
        }
        BackendDType::Int64 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        BackendDType::UInt8 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as u8).to_le_bytes()).collect())
        }
        BackendDType::UInt32 => {
            let vals: Vec<i64> = data.extract()?;
            if vals.len() != expected_len { return Err(PyValueError::new_err("data length mismatch")); }
            Ok(vals.iter().flat_map(|&v| (v as u32).to_le_bytes()).collect())
        }
        BackendDType::Bool => {
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
            BackendDType::Float32 => {
                let data: Vec<f64> = bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            BackendDType::Float16 => {
                use half::f16;
                let data: Vec<f64> = bytes.chunks_exact(2)
                    .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f64())
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            BackendDType::Float64 => {
                let data: Vec<f64> = bytes.chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            BackendDType::Int8 => {
                let data: Vec<i64> = bytes.iter().map(|&b| b as i8 as i64).collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            BackendDType::Int16 => {
                let data: Vec<i64> = bytes.chunks_exact(2)
                    .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            BackendDType::Int32 => {
                let data: Vec<i64> = bytes.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            BackendDType::Int64 => {
                let data: Vec<i64> = bytes.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            BackendDType::UInt8 => {
                let data: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            BackendDType::UInt32 => {
                let data: Vec<i64> = bytes.chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &data).into())
            }
            BackendDType::Bool => {
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
    fn log_softmax(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.log_softmax(self.id)) }
    fn tanh(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.tanh(self.id)) }
    fn sin(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.sin(self.id)) }
    fn cos(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.cos(self.id)) }
    fn gelu(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.gelu(self.id)) }
    fn sigmoid(&self) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.sigmoid(self.id)) }
    #[pyo3(signature = (correction=1))]
    fn var(&self, correction: u32) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.var(self.id, correction)) }
    #[pyo3(signature = (correction=1))]
    fn std(&self, correction: u32) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.std_dev(self.id, correction)) }
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
    #[cfg(target_os = "macos")]
    { applegpu_core::version() }
    #[cfg(not(target_os = "macos"))]
    { env!("CARGO_PKG_VERSION") }
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
    let dt = match name {
        "float16" | "f16" => BackendDType::Float16,
        "float32" | "f32" => BackendDType::Float32,
        "float64" | "f64" => BackendDType::Float64,
        "bfloat16" | "bf16" => BackendDType::BFloat16,
        "int8" | "i8" => BackendDType::Int8,
        "int16" | "i16" => BackendDType::Int16,
        "int32" | "i32" => BackendDType::Int32,
        "int64" | "i64" => BackendDType::Int64,
        "uint8" | "u8" => BackendDType::UInt8,
        "uint32" | "u32" => BackendDType::UInt32,
        "bool" => BackendDType::Bool,
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
    let dtype = BackendDType::from_name(&np_dtype_name)
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

    let dtype = BackendDType::from_name(dtype_str)
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
    let dtype = BackendDType::from_name(dtype)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown dtype: {}", dtype)))?;
    wrap_tensor(BACKEND.tensor_from_data(data, shape, dtype))
}

/// Create a GpuTensor that shares memory with a numpy array (zero-copy).
/// The array must be C-contiguous, page-aligned, and page-size-multiple in byte length.
/// Mutations to the source array ARE visible to the GPU tensor (shared memory).
/// The source array will not be garbage-collected while the tensor exists.
#[pyfunction]
#[pyo3(signature = (arr))]
fn from_numpy_shared(_py: Python<'_>, arr: &Bound<'_, pyo3::types::PyAny>) -> PyResult<GpuTensor> {
    let np_dtype_name: String = arr.getattr("dtype")?.getattr("name")?.extract()?;
    let dtype = BackendDType::from_name(&np_dtype_name)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported dtype: {}", np_dtype_name)))?;

    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let numel: usize = shape.iter().product();
    let nbytes = numel * dtype.size_bytes();
    let page_size = system_page_size();

    // Validate C-contiguous
    let is_c_contiguous: bool = arr.getattr("flags")?
        .get_item("C_CONTIGUOUS")?.extract()?;
    if !is_c_contiguous {
        return Err(PyValueError::new_err("Array must be C-contiguous for shared transfer"));
    }

    let data_ptr: usize = arr.getattr("ctypes")?.getattr("data")?.extract()?;

    // Validate page alignment
    if data_ptr % page_size != 0 {
        return Err(PyValueError::new_err(format!(
            "Array data pointer {:#x} is not page-aligned (page size: {} bytes). \
             Use gpu.aligned_numpy() or gpu.from_numpy() (copy) instead.",
            data_ptr, page_size
        )));
    }
    if nbytes % page_size != 0 {
        return Err(PyValueError::new_err(format!(
            "Array byte size {} is not a multiple of page size {} bytes. \
             Use gpu.aligned_numpy() or gpu.from_numpy() (copy) instead.",
            nbytes, page_size
        )));
    }

    // Pin the numpy array: increment refcount so GC can't collect it
    let py_obj_ptr = arr.as_ptr();
    unsafe { pyo3::ffi::Py_IncRef(py_obj_ptr) };

    // Create no-copy Metal buffer. On failure, release the ref we just added.
    let id = BACKEND.tensor_from_ptr_no_copy(
        data_ptr as *mut u8,
        nbytes,
        shape,
        dtype,
        py_obj_ptr as *mut std::ffi::c_void,
    ).map_err(|e| {
        unsafe { pyo3::ffi::Py_DecRef(py_obj_ptr) };
        PyRuntimeError::new_err(e)
    })?;

    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

/// Create a GpuTensor that shares memory with a PyTorch tensor (zero-copy).
/// The tensor must already be on CPU, contiguous, with storage_offset == 0.
/// The tensor must be page-aligned and page-size-multiple in byte length.
/// Unlike from_torch(), this does NOT call .detach().cpu().contiguous() — you must do that first.
#[pyfunction]
fn from_torch_shared(py: Python<'_>, tensor: &Bound<'_, pyo3::types::PyAny>) -> PyResult<GpuTensor> {
    // Validate already on CPU (do NOT call .cpu() — that copies)
    let device_type: String = tensor.getattr("device")?.getattr("type")?.extract()?;
    if device_type != "cpu" {
        return Err(PyValueError::new_err(
            "Tensor must be on CPU for shared transfer. Use tensor.cpu() first, or gpu.from_torch() (copy)."
        ));
    }

    // Validate already contiguous (do NOT call .contiguous() — that copies)
    let is_contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
    if !is_contiguous {
        return Err(PyValueError::new_err(
            "Tensor must be contiguous for shared transfer. Use tensor.contiguous() first, or gpu.from_torch() (copy)."
        ));
    }

    // Validate storage_offset == 0
    let storage_offset: usize = tensor.call_method0("storage_offset")?.extract()?;
    if storage_offset != 0 {
        return Err(PyValueError::new_err(
            "Tensor must not be a view with storage_offset != 0. Use tensor.clone() first, or gpu.from_torch() (copy)."
        ));
    }

    // Map torch dtype
    let torch = py.import_bound("torch")?;
    let tensor_dtype = tensor.getattr("dtype")?;
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

    let dtype = BackendDType::from_name(dtype_str)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown dtype: {}", dtype_str)))?;
    let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
    let numel: usize = tensor.call_method0("numel")?.extract()?;
    let element_size: usize = tensor.call_method0("element_size")?.extract()?;
    let nbytes = numel * element_size;
    let page_size = system_page_size();

    let data_ptr: usize = tensor.call_method0("data_ptr")?.extract()?;

    // Validate page alignment
    if data_ptr % page_size != 0 {
        return Err(PyValueError::new_err(format!(
            "Tensor data_ptr {:#x} is not page-aligned (page size: {} bytes). \
             PyTorch uses 64-byte alignment by default. Use gpu.from_torch() (copy) instead.",
            data_ptr, page_size
        )));
    }
    if nbytes % page_size != 0 {
        return Err(PyValueError::new_err(format!(
            "Tensor byte size {} is not a multiple of page size {}. \
             Use gpu.from_torch() (copy) instead.",
            nbytes, page_size
        )));
    }

    // Pin the torch tensor
    let py_obj_ptr = tensor.as_ptr();
    unsafe { pyo3::ffi::Py_IncRef(py_obj_ptr) };

    let id = BACKEND.tensor_from_ptr_no_copy(
        data_ptr as *mut u8, nbytes, shape, dtype,
        py_obj_ptr as *mut std::ffi::c_void,
    ).map_err(|e| {
        unsafe { pyo3::ffi::Py_DecRef(py_obj_ptr) };
        PyRuntimeError::new_err(e)
    })?;

    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

/// PyCapsule destructor — called when numpy array is GC'd, frees the aligned allocation.
unsafe extern "C" fn aligned_buffer_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, std::ptr::null());
    if !ptr.is_null() {
        libc::free(ptr);
    }
}

/// Allocate a page-aligned numpy array suitable for gpu.from_numpy_shared().
/// The returned array's data pointer is page-aligned and its byte size is a page-size multiple.
/// The array element count is padded so total byte size is a multiple of the page size.
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn aligned_numpy(py: Python<'_>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyObject> {
    let page_size = system_page_size();
    let dtype_str = dtype.unwrap_or("float32");
    let dt = BackendDType::from_name(dtype_str)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported dtype: {}", dtype_str)))?;
    let numel: usize = shape.iter().product();
    let nbytes = numel * dt.size_bytes();

    if nbytes == 0 {
        let np = py.import_bound("numpy")?;
        return Ok(np.call_method1("empty", (shape, dtype_str))?.into());
    }

    if nbytes % page_size != 0 {
        let elems_per_page = page_size / dt.size_bytes();
        return Err(PyValueError::new_err(format!(
            "Data size {} bytes is not a multiple of page size {} bytes. Choose a shape where numel * dtype_size is a page multiple (e.g., multiples of {} elements for float32).",
            nbytes, page_size, elems_per_page
        )));
    }

    // Round up to page boundary
    let aligned_size = (nbytes + page_size - 1) & !(page_size - 1);

    // Allocate page-aligned memory
    let mut ptr: *mut libc::c_void = std::ptr::null_mut();
    let ret = unsafe { libc::posix_memalign(&mut ptr, page_size, aligned_size) };
    if ret != 0 {
        return Err(PyValueError::new_err("Failed to allocate aligned memory"));
    }

    // Zero-initialize
    unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, aligned_size) };

    // Wrap the pointer in a PyCapsule for correct lifetime management.
    // The capsule destructor calls libc::free when the capsule is GC'd.
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            ptr,
            std::ptr::null(),
            Some(aligned_buffer_destructor),
        );
        if cap.is_null() {
            libc::free(ptr);
            return Err(PyRuntimeError::new_err("Failed to create PyCapsule"));
        }
        PyObject::from_owned_ptr(py, cap)
    };

    // Get or create the _AlignedArray subclass (cached across calls)
    let aligned_cls = ALIGNED_ARRAY_CLASS.get_or_try_init(|| -> PyResult<PyObject> {
        let locals = pyo3::types::PyDict::new_bound(py);
        locals.set_item("np", py.import_bound("numpy")?)?;
        py.run_bound(
            r#"
class _AlignedArray(np.ndarray):
    """ndarray subclass that holds a reference to aligned memory capsule."""
    pass
"#,
            None,
            Some(&locals),
        )?;
        let cls = locals.get_item("_AlignedArray")?
            .ok_or_else(|| PyRuntimeError::new_err("Failed to create _AlignedArray class"))?;
        Ok(cls.into())
    })?;

    // Create the array using Python code. We use a numpy.ndarray subclass
    // to store the capsule reference and prevent GC of the aligned memory.
    let locals = pyo3::types::PyDict::new_bound(py);
    locals.set_item("np", py.import_bound("numpy")?)?;
    locals.set_item("ctypes", py.import_bound("ctypes")?)?;
    locals.set_item("capsule", capsule)?;
    locals.set_item("ptr_val", ptr as usize)?;
    locals.set_item("aligned_size", aligned_size)?;
    locals.set_item("numel", numel)?;
    locals.set_item("target_shape", pyo3::types::PyTuple::new_bound(py, &shape))?;
    locals.set_item("dtype_str", dtype_str)?;
    locals.set_item("_AlignedArray", aligned_cls.bind(py))?;

    py.run_bound(
        r#"
# Create ctypes pointer to aligned memory
_c_ptr = ctypes.cast(ptr_val, ctypes.POINTER(ctypes.c_ubyte))
# Create flat uint8 array viewing the entire aligned allocation
_raw = np.ctypeslib.as_array(_c_ptr, shape=(aligned_size,))
# View as requested dtype, slice to exact element count, reshape
_typed = _raw.view(dtype_str)[:numel].reshape(target_shape)
# Convert to subclass so we can store capsule reference
result = _typed.view(_AlignedArray)
result._mem_capsule = capsule
"#,
        None,
        Some(&locals),
    )?;

    let result = locals.get_item("result")?
        .ok_or_else(|| PyRuntimeError::new_err("Failed to create aligned array"))?;
    Ok(result.into())
}

/// Create a tensor from data. Returns a GpuTensor object.
/// Accepts lists of floats, ints, or bools depending on dtype.
/// Supported dtypes: float16, float32, float64, int8, int16, int32, int64, uint8, uint32, bool.
#[pyfunction]
#[pyo3(signature = (data, shape, dtype=None))]
fn tensor(py: Python<'_>, data: &Bound<'_, PyAny>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<GpuTensor> {
    let dtype_str = dtype.unwrap_or("float32");
    let dtype = BackendDType::from_name(dtype_str)
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
fn log_softmax(t: &GpuTensor) -> PyResult<GpuTensor> { t.log_softmax() }
#[pyfunction]
fn transpose(t: &GpuTensor) -> PyResult<GpuTensor> { t.transpose() }
#[pyfunction]
fn transpose_dims(t: &GpuTensor, dim0: usize, dim1: usize) -> PyResult<GpuTensor> { t.transpose_dims(dim0, dim1) }

#[pyfunction]
fn tanh(t: &GpuTensor) -> PyResult<GpuTensor> { t.tanh() }

#[pyfunction]
fn sin(t: &GpuTensor) -> PyResult<GpuTensor> { t.sin() }

#[pyfunction]
fn cos(t: &GpuTensor) -> PyResult<GpuTensor> { t.cos() }

#[pyfunction]
fn gelu(t: &GpuTensor) -> PyResult<GpuTensor> { t.gelu() }

#[pyfunction]
fn sigmoid(t: &GpuTensor) -> PyResult<GpuTensor> { t.sigmoid() }

#[pyfunction]
#[pyo3(signature = (t, correction=1))]
fn var(t: &GpuTensor, correction: u32) -> PyResult<GpuTensor> { t.var(correction) }

#[pyfunction]
#[pyo3(signature = (t, correction=1))]
fn std_dev(t: &GpuTensor, correction: u32) -> PyResult<GpuTensor> { t.std(correction) }

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
fn lt(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.lt(a.id, b.id)) }
#[pyfunction]
fn gt(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.gt(a.id, b.id)) }
#[pyfunction]
fn le(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.le(a.id, b.id)) }
#[pyfunction]
fn ge(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.ge(a.id, b.id)) }
#[pyfunction]
fn eq_(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.eq_op(a.id, b.id)) }
#[pyfunction]
fn ne_(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.ne_op(a.id, b.id)) }

#[pyfunction]
fn bitwise_and(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.bitwise_and(a.id, b.id)) }
#[pyfunction]
fn bitwise_or(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.bitwise_or(a.id, b.id)) }
#[pyfunction]
fn bitwise_xor(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.bitwise_xor(a.id, b.id)) }
#[pyfunction]
fn bitwise_not(a: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.bitwise_not(a.id)) }
#[pyfunction]
fn shl(a: &GpuTensor, shift: u32) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.shl(a.id, shift)) }
#[pyfunction]
fn shr(a: &GpuTensor, shift: u32) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.shr(a.id, shift)) }
#[pyfunction]
fn mod_(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.mod_op(a.id, b.id)) }
#[pyfunction]
fn elem_min(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.elem_min(a.id, b.id)) }
#[pyfunction]
fn elem_max(a: &GpuTensor, b: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.elem_max(a.id, b.id)) }
#[pyfunction]
fn logical_not(a: &GpuTensor) -> PyResult<GpuTensor> { wrap_tensor(BACKEND.logical_not(a.id)) }

#[pyfunction]
fn cast(t: &GpuTensor, dtype: &str) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.cast(t.id, dtype))
}

#[pyfunction]
#[pyo3(signature = (t, dtype, scale=0.1, zero_point=0))]
fn quantize(t: &GpuTensor, dtype: &str, scale: f32, zero_point: i32) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.quantize(t.id, dtype, scale, zero_point))
}

#[pyfunction]
#[pyo3(signature = (t, dtype, scale=0.1, zero_point=0))]
fn dequantize(t: &GpuTensor, dtype: &str, scale: f32, zero_point: i32) -> PyResult<GpuTensor> {
    wrap_tensor(BACKEND.dequantize(t.id, dtype, scale, zero_point))
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
    m.add_function(wrap_pyfunction!(from_numpy_shared, m)?)?;
    m.add_function(wrap_pyfunction!(from_torch, m)?)?;
    m.add_function(wrap_pyfunction!(from_torch_shared, m)?)?;
    m.add_function(wrap_pyfunction!(from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(aligned_numpy, m)?)?;
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
    m.add_function(wrap_pyfunction!(log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(transpose_dims, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
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
    m.add_function(wrap_pyfunction!(cast, m)?)?;
    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(shl, m)?)?;
    m.add_function(wrap_pyfunction!(shr, m)?)?;
    m.add_function(wrap_pyfunction!(mod_, m)?)?;
    m.add_function(wrap_pyfunction!(elem_min, m)?)?;
    m.add_function(wrap_pyfunction!(elem_max, m)?)?;
    m.add_function(wrap_pyfunction!(logical_not, m)?)?;
    m.add_function(wrap_pyfunction!(lt, m)?)?;
    m.add_function(wrap_pyfunction!(gt, m)?)?;
    m.add_function(wrap_pyfunction!(le, m)?)?;
    m.add_function(wrap_pyfunction!(ge, m)?)?;
    m.add_function(wrap_pyfunction!(eq_, m)?)?;
    m.add_function(wrap_pyfunction!(ne_, m)?)?;
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
