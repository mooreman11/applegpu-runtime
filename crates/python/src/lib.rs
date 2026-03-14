use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods, IntoPyArray, IxDyn};

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

    /// Get the dtype as a string ("float32" or "float16").
    #[getter]
    fn dtype(&self) -> PyResult<String> {
        let rt = RUNTIME_LAZY.lock().unwrap();
        let dtype = rt.dtype(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(match dtype {
            DType::Float32 => "float32",
            DType::Float16 => "float16",
            other => return Err(PyValueError::new_err(format!("Unsupported dtype: {:?}", other))),
        }.to_string())
    }

    /// Read tensor data as a NumPy ndarray with the tensor's shape.
    /// Auto-evaluates if the tensor is lazy.
    /// Returns float32 ndarray for f32 tensors, float16 ndarray for f16 tensors.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        auto_eval(&mut rt, self.id)?;

        let dtype = rt.dtype(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let shape = rt.shape(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        match dtype {
            DType::Float32 => {
                let data = rt.read_f32(self.id)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                drop(rt);
                let flat: Bound<'py, PyArray1<f32>> = data.into_pyarray_bound(py);
                let reshaped = flat.reshape(IxDyn(&shape))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(reshaped.into_any())
            }
            DType::Float16 => {
                let data = rt.read_f16(self.id)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                drop(rt);
                // pyo3-numpy has no native f16 type. Create u16 array, then view as np.float16.
                let flat: Bound<'py, PyArray1<u16>> = data.into_pyarray_bound(py);
                let np = py.import_bound("numpy")?;
                let f16_dtype = np.getattr("float16")?;
                let viewed = flat.call_method1("view", (&f16_dtype,))?;
                let reshaped = viewed.call_method1("reshape", (shape,))?;
                Ok(reshaped)
            }
            other => Err(PyValueError::new_err(format!("Unsupported dtype for to_numpy: {:?}", other))),
        }
    }

    /// Read tensor data as a flat list of f64 values.
    /// Auto-evaluates if the tensor is lazy.
    fn to_list(&self) -> PyResult<Vec<f64>> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        auto_eval(&mut rt, self.id)?;

        let dtype = rt.dtype(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        match dtype {
            DType::Float32 => {
                let data = rt.read_f32(self.id)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(data.into_iter().map(|v| v as f64).collect())
            }
            DType::Float16 => {
                let data = rt.read_f16(self.id)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(data.into_iter().map(|v| half::f16::from_bits(v).to_f64()).collect())
            }
            other => Err(PyValueError::new_err(format!("Unsupported dtype for to_list: {:?}", other))),
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

    fn transpose(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::transpose(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    fn sqrt(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::sqrt(&mut rt, self.id)
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
        // Get numpy array via existing to_numpy (handles f16 and f32)
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

/// Create a GpuTensor from a NumPy ndarray (float32 or float16, C-contiguous).
/// Data is copied; mutations to the original array do not affect the tensor.
#[pyfunction]
#[pyo3(signature = (arr))]
fn from_numpy(py: Python<'_>, arr: &Bound<'_, pyo3::types::PyAny>) -> PyResult<GpuTensor> {
    // Check the numpy dtype string to dispatch
    let dtype_str: String = arr.getattr("dtype")?.getattr("name")?.extract()?;

    let runtime = get_device_runtime()?;

    match dtype_str.as_str() {
        "float32" => {
            let arr: PyReadonlyArrayDyn<'_, f32> = arr.extract()
                .map_err(|_| PyTypeError::new_err(
                    "from_numpy: failed to extract float32 ndarray."
                ))?;
            let data: Vec<f32> = arr.as_slice()
                .map_err(|_| PyValueError::new_err(
                    "Array must be C-contiguous. Use np.ascontiguousarray()."
                ))?
                .to_vec();
            let shape: Vec<usize> = arr.shape().to_vec();
            let t = Tensor::from_f32(&runtime.device, shape, &data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let id = t.meta.id;
            RUNTIME_LAZY.lock().unwrap().insert_tensor(t)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(GpuTensor { id, destroyed: Cell::new(false) })
        }
        "float16" => {
            // numpy float16 is stored as u16 in memory. View as u16 to extract raw bits.
            let np = py.import_bound("numpy")?;
            let u16_dtype = np.getattr("uint16")?;
            let u16_arr = arr.call_method1("view", (&u16_dtype,))?;
            let flat = u16_arr.call_method0("flatten")?;
            let data: Vec<u16> = flat.extract()?;
            let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
            let t = Tensor::from_f16(&runtime.device, shape, &data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let id = t.meta.id;
            RUNTIME_LAZY.lock().unwrap().insert_tensor(t)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(GpuTensor { id, destroyed: Cell::new(false) })
        }
        other => Err(PyTypeError::new_err(format!(
            "from_numpy requires a float32 or float16 ndarray, got {}. Use arr.astype(np.float32).",
            other
        ))),
    }
}

/// Create a GpuTensor from a PyTorch tensor (float32 or float16).
/// Internally converts via NumPy: detach -> cpu -> contiguous -> numpy -> from_numpy.
/// Data is copied; mutations to the original tensor do not affect the GPU tensor.
#[pyfunction]
fn from_torch(py: Python<'_>, tensor: &Bound<'_, pyo3::types::PyAny>) -> PyResult<GpuTensor> {
    // Lazy import torch for dtype check
    let torch = py.import_bound("torch")?;
    let tensor_dtype = tensor.getattr("dtype")?;
    let float32 = torch.getattr("float32")?;
    let float16 = torch.getattr("float16")?;

    if !tensor_dtype.eq(&float32)? && !tensor_dtype.eq(&float16)? {
        return Err(PyValueError::new_err(format!(
            "Expected float32 or float16 tensor, got {}. Use tensor.float() or tensor.half().",
            tensor_dtype
        )));
    }

    // Prepare: detach from autograd, move to CPU, make contiguous
    let prepared = tensor
        .call_method0("detach")?
        .call_method0("cpu")?
        .call_method0("contiguous")?;

    // Convert to numpy (zero-copy for contiguous CPU tensors)
    let np_array = prepared.call_method0("numpy")?;

    // Delegate to existing from_numpy (handles both f32 and f16)
    from_numpy(py, &np_array)
}

/// Create a tensor from data. Returns a GpuTensor object.
/// dtype: "float32" (default) or "float16".
#[pyfunction]
#[pyo3(signature = (data, shape, dtype=None))]
fn tensor(data: Vec<f64>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<GpuTensor> {
    let runtime = get_device_runtime()?;
    let dtype_str = dtype.unwrap_or("float32");

    let t = match dtype_str {
        "float32" | "f32" => {
            let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            Tensor::from_f32(&runtime.device, shape, &f32_data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        }
        "float16" | "f16" => {
            let u16_data: Vec<u16> = data.iter()
                .map(|&v| half::f16::from_f64(v).to_bits())
                .collect();
            Tensor::from_f16(&runtime.device, shape, &u16_data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        }
        other => return Err(PyValueError::new_err(format!(
            "Unsupported dtype: '{}'. Use 'float32' or 'float16'.", other
        ))),
    };

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
fn to_list(t: &GpuTensor) -> PyResult<Vec<f64>> { t.to_list() }

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
fn softmax(t: &GpuTensor) -> PyResult<GpuTensor> { t.softmax() }
#[pyfunction]
fn transpose(t: &GpuTensor) -> PyResult<GpuTensor> { t.transpose() }

#[pyfunction]
fn attention(q: &GpuTensor, k: &GpuTensor, v: &GpuTensor) -> PyResult<GpuTensor> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    let id = applegpu_core::ops::attention(&mut rt, q.id, k.id, v.id)
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
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(attention, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
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
    Ok(())
}
