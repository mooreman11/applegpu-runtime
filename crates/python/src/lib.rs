use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;

use applegpu_core::lazy::LazyRuntime;
use applegpu_core::tensor::{DType, Tensor};
use applegpu_core::scheduler::{ContainerId, ContainerConfig, Priority, JobId, JobStatus};

/// Global lazy runtime.
static RUNTIME_LAZY: once_cell::sync::Lazy<Mutex<LazyRuntime>> =
    once_cell::sync::Lazy::new(|| Mutex::new(LazyRuntime::new()));

/// Helper: get the backend device, ensuring init_backend() was called.
fn get_device_runtime() -> PyResult<&'static applegpu_core::backend::Runtime> {
    applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Helper: auto-evaluate a tensor if it is pending (lazy).
fn auto_eval(rt: &mut LazyRuntime, id: u64) -> PyResult<()> {
    if rt.is_pending(id) {
        let runtime = get_device_runtime()?;
        if let Some(ref socket_path) = runtime.socket_path {
            rt.eval_remote(&runtime.device, id, socket_path)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        } else {
            rt.eval(&runtime.device, id)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    Ok(())
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
        let rt = RUNTIME_LAZY.lock().unwrap();
        rt.shape(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get the tensor ID (internal handle).
    #[getter]
    fn id(&self) -> u64 {
        self.id
    }

    /// Get the dtype as a string (all 10 types supported).
    #[getter]
    fn dtype(&self) -> PyResult<String> {
        let rt = RUNTIME_LAZY.lock().unwrap();
        let dtype = rt.dtype(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dtype.name().to_string())
    }

    /// Read tensor data as a NumPy ndarray with the tensor's shape.
    /// Auto-evaluates if the tensor is lazy.
    /// Supports all 10 dtypes.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        auto_eval(&mut rt, self.id)?;

        let dtype = rt.dtype(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let bytes = rt.read_bytes(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let shape = rt.shape(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        drop(rt);

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
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        auto_eval(&mut rt, self.id)?;

        let dtype = rt.dtype(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let bytes = rt.read_bytes(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        drop(rt);

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
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        auto_eval(&mut rt, self.id)
    }

    // Unary ops

    fn neg(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::neg(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn relu(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::relu(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn exp(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::exp(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn log(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::log(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn softmax(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::softmax(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn gelu(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::gelu(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    #[pyo3(signature = (gamma, beta, eps=1e-5))]
    fn layer_norm(&self, gamma: &GpuTensor, beta: &GpuTensor, eps: f32) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::layer_norm(&mut rt, self.id, gamma.id, beta.id, eps)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn transpose(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::transpose(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    #[pyo3(signature = (dim0, dim1))]
    fn transpose_dims(&self, dim0: usize, dim1: usize) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::transpose_dims(&mut rt, self.id, dim0, dim1)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn sqrt(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::sqrt(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn abs(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::abs(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn sign(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::sign(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    #[pyo3(signature = (exponent))]
    fn pow(&self, exponent: f32) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::pow(&mut rt, self.id, exponent)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    #[pyo3(signature = (min_val, max_val))]
    fn clamp(&self, min_val: f32, max_val: f32) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::clamp(&mut rt, self.id, min_val, max_val)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    #[pyo3(signature = (mask, value))]
    fn masked_fill(&self, mask: &GpuTensor, value: f32) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::masked_fill(&mut rt, self.id, mask.id, value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    #[pyo3(signature = (diagonal=0))]
    fn triu(&self, diagonal: i32) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::triu(&mut rt, self.id, diagonal)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    #[pyo3(signature = (diagonal=0))]
    fn tril(&self, diagonal: i32) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::tril(&mut rt, self.id, diagonal)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn gather(&self, dim: usize, index: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::gather(&mut rt, self.id, dim, index.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn index_select(&self, dim: usize, index: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::index_select(&mut rt, self.id, dim, index.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    #[pyo3(signature = (scale))]
    fn scalar_mul(&self, scale: f32) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::scalar_mul(&mut rt, self.id, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    #[pyo3(signature = (new_shape))]
    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::reshape(&mut rt, self.id, new_shape)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn argmax(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::argmax(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn sum(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::sum(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn mean(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::mean(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn softmax_causal(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::softmax_causal(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    // Binary ops

    fn add(&self, other: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::add(&mut rt, self.id, other.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn sub(&self, other: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::sub(&mut rt, self.id, other.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn mul(&self, other: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::mul(&mut rt, self.id, other.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn div(&self, other: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::div(&mut rt, self.id, other.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn matmul(&self, other: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::matmul(&mut rt, self.id, other.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

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
        let rt = RUNTIME_LAZY.lock().unwrap();
        let status = if rt.is_materialized(self.id) {
            "materialized"
        } else if rt.is_pending(self.id) {
            "lazy"
        } else {
            "destroyed"
        };
        match rt.shape(self.id) {
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
        if let Ok(mut rt) = RUNTIME_LAZY.try_lock() {
            let _ = rt.destroy(self.id);
        }
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
    let runtime = applegpu_core::backend::init_backend()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut info = HashMap::new();
    info.insert("backend".to_string(), format!("{:?}", runtime.backend).to_lowercase());
    info.insert("device".to_string(), runtime.device.name());
    Ok(info)
}

#[pyfunction]
fn device_name() -> PyResult<String> {
    let runtime = get_device_runtime()?;
    Ok(runtime.device.name())
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
    let bytes: Vec<u8> = arr.call_method0("tobytes")?.extract()?;

    let runtime = get_device_runtime()?;
    let t = Tensor::from_data(&runtime.device, shape, dtype, &bytes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let id = t.meta.id;
    RUNTIME_LAZY.lock().unwrap().insert_tensor(t)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

/// Create a GpuTensor from a PyTorch tensor (any supported dtype).
/// Internally converts via NumPy: detach -> cpu -> contiguous -> numpy -> from_numpy.
/// Data is copied; mutations to the original tensor do not affect the GPU tensor.
#[pyfunction]
fn from_torch(py: Python<'_>, tensor: &Bound<'_, pyo3::types::PyAny>) -> PyResult<GpuTensor> {
    // Prepare: detach from autograd, move to CPU, make contiguous
    let prepared = tensor
        .call_method0("detach")?
        .call_method0("cpu")?
        .call_method0("contiguous")?;

    // Convert to numpy (zero-copy for contiguous CPU tensors)
    let np_array = prepared.call_method0("numpy")?;

    // Delegate to existing from_numpy (handles all dtypes)
    from_numpy(py, &np_array)
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

    let runtime = get_device_runtime()?;
    let bytes = python_data_to_bytes(py, data, &shape, dtype)?;
    let t = Tensor::from_data(&runtime.device, shape, dtype, &bytes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let id = t.meta.id;
    RUNTIME_LAZY.lock().unwrap().insert_tensor(t)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

/// Set resource limits. Pass 0 for any field to make it unlimited.
#[pyfunction]
fn set_limits(max_tensor_size_mb: usize, max_memory_mb: usize, max_tensors: usize) -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.set_limits(applegpu_core::limits::ResourceLimits {
        max_tensor_size_bytes: if max_tensor_size_mb > 0 { max_tensor_size_mb * 1024 * 1024 } else { 0 },
        max_total_memory_bytes: if max_memory_mb > 0 { max_memory_mb * 1024 * 1024 } else { 0 },
        max_tensor_count: max_tensors,
    });
    Ok(())
}

/// Get current GPU memory usage in bytes.
#[pyfunction]
fn memory_usage() -> PyResult<usize> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    Ok(rt.memory_usage())
}

/// Get current number of live tensors.
#[pyfunction]
fn tensor_count() -> PyResult<usize> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    Ok(rt.live_tensor_count())
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
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.destroy(t.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::where_cond(&mut rt, cond.id, x.id, y.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
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
fn gelu(t: &GpuTensor) -> PyResult<GpuTensor> { t.gelu() }

#[pyfunction]
#[pyo3(signature = (input, gamma, beta, eps=1e-5))]
fn layer_norm(input: &GpuTensor, gamma: &GpuTensor, beta: &GpuTensor, eps: f32) -> PyResult<GpuTensor> {
    input.layer_norm(gamma, beta, eps)
}

#[pyfunction]
fn embedding(weights: &GpuTensor, indices: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::embedding(&mut rt, weights.id, indices.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn gather(input: &GpuTensor, dim: usize, index: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::gather(&mut rt, input.id, dim, index.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn index_select(input: &GpuTensor, dim: usize, index: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::index_select(&mut rt, input.id, dim, index.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn attention(q: &GpuTensor, k: &GpuTensor, v: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::attention(&mut rt, q.id, k.id, v.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn slice(t: &GpuTensor, dim: usize, start: usize, end: usize) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::slice(&mut rt, t.id, dim, start, end)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn concat(a: &GpuTensor, b: &GpuTensor, dim: usize) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::concat(&mut rt, a.id, b.id, dim)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn add_bias(input: &GpuTensor, bias: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::add_bias(&mut rt, input.id, bias.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn softmax_causal(t: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::softmax_causal(&mut rt, t.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn argmax(t: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::argmax(&mut rt, t.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn sum(t: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::sum(&mut rt, t.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn mean(t: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::mean(&mut rt, t.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
fn attention_causal(q: &GpuTensor, k: &GpuTensor, v: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::attention_causal(&mut rt, q.id, k.id, v.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
#[pyo3(signature = (input, weight, stride=1, padding=0))]
fn conv1d(input: &GpuTensor, weight: &GpuTensor, stride: usize, padding: usize) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::conv1d(&mut rt, input.id, weight.id, stride, padding)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
#[pyo3(signature = (input, weight, stride_h=1, stride_w=1, pad_h=0, pad_w=0))]
fn conv2d(input: &GpuTensor, weight: &GpuTensor, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::conv2d(&mut rt, input.id, weight.id, (stride_h, stride_w), (pad_h, pad_w))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
#[pyo3(signature = (input, running_mean, running_var, weight, bias, eps=1e-5))]
fn batch_norm(input: &GpuTensor, running_mean: &GpuTensor, running_var: &GpuTensor, weight: &GpuTensor, bias: &GpuTensor, eps: f32) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::batch_norm(&mut rt, input.id, running_mean.id, running_var.id, weight.id, bias.id, eps)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
#[pyo3(signature = (input, kh=2, kw=2, stride_h=0, stride_w=0, pad_h=0, pad_w=0))]
fn max_pool2d(input: &GpuTensor, kh: usize, kw: usize, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize) -> PyResult<GpuTensor> {
    let sh = if stride_h == 0 { kh } else { stride_h };
    let sw = if stride_w == 0 { kw } else { stride_w };
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::max_pool2d(&mut rt, input.id, (kh, kw), (sh, sw), (pad_h, pad_w))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

#[pyfunction]
#[pyo3(signature = (input, kh=2, kw=2, stride_h=0, stride_w=0, pad_h=0, pad_w=0))]
fn avg_pool2d(input: &GpuTensor, kh: usize, kw: usize, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize) -> PyResult<GpuTensor> {
    let sh = if stride_h == 0 { kh } else { stride_h };
    let sw = if stride_w == 0 { kw } else { stride_w };
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::avg_pool2d(&mut rt, input.id, (kh, kw), (sh, sw), (pad_h, pad_w))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
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
    let priority = match priority {
        "high" => Priority::High,
        "normal" => Priority::Normal,
        "low" => Priority::Low,
        _ => return Err(PyValueError::new_err(format!("Invalid priority: {}. Use 'high', 'normal', or 'low'", priority))),
    };
    let config = ContainerConfig {
        priority,
        max_memory_bytes: max_memory_mb * 1024 * 1024,
        max_tensor_count: max_tensors,
        max_tensor_size_bytes: 0,
        max_pending_jobs: max_pending,
    };
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = rt.scheduler.register_container(config)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(id.0)
}

#[pyfunction]
fn deregister_container(container_id: u64) -> PyResult<Vec<u64>> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let tensors = rt.scheduler.deregister_container(ContainerId(container_id))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    for &tid in &tensors {
        rt.remove_tensor_raw(tid);
    }
    Ok(tensors)
}

#[pyfunction]
fn pause_container(container_id: u64) -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.scheduler.pause_container(ContainerId(container_id))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn resume_container(container_id: u64) -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.scheduler.resume_container(ContainerId(container_id))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn submit_job(container_id: u64, t: &GpuTensor) -> PyResult<u64> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let job_id = rt.scheduler.submit(ContainerId(container_id), t.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(job_id.0)
}

#[pyfunction]
fn run_next() -> PyResult<Option<u64>> {
    let runtime = get_device_runtime()?;
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let result = rt.run_next(&runtime.device)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(result.map(|j| j.0))
}

#[pyfunction]
fn job_status(job_id: u64) -> PyResult<String> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    match rt.scheduler.job_status(JobId(job_id)) {
        Some((_, JobStatus::Queued)) => Ok("queued".to_string()),
        Some((_, JobStatus::Running { .. })) => Ok("running".to_string()),
        Some((_, JobStatus::Completed { .. })) => Ok("completed".to_string()),
        Some((_, JobStatus::Failed { .. })) => Ok("failed".to_string()),
        None => Err(PyValueError::new_err(format!("Job {} not found", job_id))),
    }
}

#[pyfunction]
fn container_usage(container_id: u64) -> PyResult<(usize, usize)> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    rt.scheduler.container_usage(ContainerId(container_id))
        .ok_or_else(|| PyValueError::new_err(format!("Container {} not found", container_id)))
}

#[pyfunction]
fn global_usage() -> PyResult<(usize, usize)> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    Ok(rt.scheduler.global_usage())
}

#[pyfunction]
fn queue_depth() -> PyResult<usize> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    Ok(rt.scheduler.queue_depth())
}

#[pyfunction]
fn pool_stats() -> PyResult<HashMap<String, usize>> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    let stats = rt.pool.stats();
    let mut map = HashMap::new();
    map.insert("hits".to_string(), stats.hits as usize);
    map.insert("misses".to_string(), stats.misses as usize);
    map.insert("pooled_bytes".to_string(), stats.pooled_bytes);
    map.insert("bucket_count".to_string(), stats.bucket_count);
    Ok(map)
}

#[pyfunction]
fn pool_drain() -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.pool.drain();
    Ok(())
}

#[pyfunction]
fn set_pool_watermark(mb: usize) -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.pool.set_max_pooled_bytes(mb * 1024 * 1024);
    Ok(())
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
    Ok(())
}
