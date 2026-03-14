use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::sync::Mutex;

use applegpu_core::compute::ComputePipeline;
use applegpu_core::tensor::Tensor;

/// Global tensor storage. Tensors are stored by ID and accessed by opaque handle.
static TENSORS: once_cell::sync::Lazy<Mutex<HashMap<u64, Tensor>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// Global add pipeline (lazy-initialized).
static ADD_PIPELINE: once_cell::sync::Lazy<Mutex<Option<ComputePipeline>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(None));

fn get_or_create_add_pipeline() -> PyResult<()> {
    let mut pipeline = ADD_PIPELINE.lock().unwrap();
    if pipeline.is_none() {
        let runtime = applegpu_core::backend::get_runtime()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let p = ComputePipeline::add(&runtime.device)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        *pipeline = Some(p);
    }
    Ok(())
}

/// Returns the library version.
#[pyfunction]
fn version() -> &'static str {
    applegpu_core::version()
}

/// Initialize the GPU backend. Returns dict with backend info.
#[pyfunction]
fn init_backend() -> PyResult<HashMap<String, String>> {
    let runtime = applegpu_core::backend::init_backend()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut info = HashMap::new();
    info.insert(
        "backend".to_string(),
        format!("{:?}", runtime.backend).to_lowercase(),
    );
    info.insert("device".to_string(), runtime.device.name());
    Ok(info)
}

/// Get the Metal GPU device name. Requires init_backend() first.
#[pyfunction]
fn device_name() -> PyResult<String> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(runtime.device.name())
}

/// Get the size in bytes of a dtype by name.
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

/// Create a tensor from a list of f32 values and a shape.
/// Returns an opaque tensor handle (u64 ID).
#[pyfunction]
fn tensor(data: Vec<f32>, shape: Vec<usize>) -> PyResult<u64> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let t = Tensor::from_f32(&runtime.device, shape, &data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let id = t.meta.id;
    TENSORS.lock().unwrap().insert(id, t);
    Ok(id)
}

/// Read tensor data as a flat list of f32 values.
#[pyfunction]
fn to_list(tensor_id: u64) -> PyResult<Vec<f32>> {
    let tensors = TENSORS.lock().unwrap();
    let t = tensors
        .get(&tensor_id)
        .ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", tensor_id)))?;
    Ok(t.as_f32_slice().to_vec())
}

/// Get the shape of a tensor as a list.
#[pyfunction]
fn shape(tensor_id: u64) -> PyResult<Vec<usize>> {
    let tensors = TENSORS.lock().unwrap();
    let t = tensors
        .get(&tensor_id)
        .ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", tensor_id)))?;
    Ok(t.meta.shape.dims().to_vec())
}

/// Element-wise add: c = a + b. Returns tensor handle for result.
#[pyfunction]
fn add(a_id: u64, b_id: u64) -> PyResult<u64> {
    get_or_create_add_pipeline()?;

    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Step 1: Lock TENSORS briefly to validate inputs and extract shape.
    let (out_shape, numel) = {
        let tensors = TENSORS.lock().unwrap();
        let a = tensors
            .get(&a_id)
            .ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", a_id)))?;
        let b = tensors
            .get(&b_id)
            .ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", b_id)))?;

        if a.meta.shape != b.meta.shape {
            return Err(PyValueError::new_err(format!(
                "Shape mismatch: {:?} vs {:?}",
                a.meta.shape.dims(),
                b.meta.shape.dims()
            )));
        }
        (a.meta.shape.dims().to_vec(), a.numel())
    }; // TENSORS lock dropped here

    let out = Tensor::empty_f32(&runtime.device, out_shape)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Step 2: Lock TENSORS to get buffer refs, then ADD_PIPELINE to dispatch.
    // Consistent lock ordering: TENSORS first, then ADD_PIPELINE.
    {
        let tensors = TENSORS.lock().unwrap();
        let a = tensors.get(&a_id).unwrap();
        let b = tensors.get(&b_id).unwrap();

        let pipeline = ADD_PIPELINE.lock().unwrap();
        let pipeline = pipeline.as_ref().unwrap();
        pipeline
            .dispatch_elementwise(&a.buffer, &b.buffer, &out.buffer, numel)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    } // Both locks dropped here

    // Step 3: Insert result.
    let id = out.meta.id;
    TENSORS.lock().unwrap().insert(id, out);
    Ok(id)
}

/// The Python module definition.
#[pymodule]
fn applegpu_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(init_backend, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_size, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(to_list, m)?)?;
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
