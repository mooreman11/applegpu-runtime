use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::sync::Mutex;

use applegpu_core::lazy::LazyRuntime;
use applegpu_core::tensor::Tensor;

/// Global lazy runtime.
static RUNTIME_LAZY: once_cell::sync::Lazy<Mutex<LazyRuntime>> =
    once_cell::sync::Lazy::new(|| Mutex::new(LazyRuntime::new()));

/// Helper: run a binary op (records in graph, does not execute).
fn binary_op_py(a_id: u64, b_id: u64, op: fn(&mut LazyRuntime, u64, u64) -> applegpu_core::error::Result<u64>) -> PyResult<u64> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    op(&mut rt, a_id, b_id).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Helper: run a unary op (records in graph, does not execute).
fn unary_op_py(input_id: u64, op: fn(&mut LazyRuntime, u64) -> applegpu_core::error::Result<u64>) -> PyResult<u64> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    op(&mut rt, input_id).map_err(|e| PyValueError::new_err(e.to_string()))
}

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
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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

/// Create a tensor from data (immediately materialized — this is input data).
#[pyfunction]
fn tensor(data: Vec<f32>, shape: Vec<usize>) -> PyResult<u64> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let t = Tensor::from_f32(&runtime.device, shape, &data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let id = t.meta.id;
    RUNTIME_LAZY.lock().unwrap().insert_tensor(t);
    Ok(id)
}

/// Explicitly evaluate a lazy tensor, materializing its result on the GPU.
#[pyfunction]
fn eval(tensor_id: u64) -> PyResult<()> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.eval(&runtime.device, tensor_id)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Read tensor data. Auto-evaluates if the tensor is lazy.
#[pyfunction]
fn to_list(tensor_id: u64) -> PyResult<Vec<f32>> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut rt = RUNTIME_LAZY.lock().unwrap();

    // Auto-eval if pending
    if rt.is_pending(tensor_id) {
        rt.eval(&runtime.device, tensor_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    rt.read_f32(tensor_id)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Get shape (works on both materialized and lazy tensors).
#[pyfunction]
fn shape(tensor_id: u64) -> PyResult<Vec<usize>> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    rt.shape(tensor_id)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Destroy a tensor, freeing its GPU buffer.
/// Errors if pending graph ops depend on this tensor.
#[pyfunction]
fn destroy(tensor_id: u64) -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.destroy(tensor_id)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// Binary ops (lazy — just record in graph)
#[pyfunction]
fn add(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::add) }
#[pyfunction]
fn sub(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::sub) }
#[pyfunction]
fn mul(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::mul) }
#[pyfunction]
fn div(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::div) }
#[pyfunction]
fn matmul(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::matmul) }

// Unary ops (lazy)
#[pyfunction]
fn neg(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::neg) }
#[pyfunction]
fn relu(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::relu) }
#[pyfunction]
fn exp(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::exp) }
#[pyfunction]
fn log(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::log) }
#[pyfunction]
fn sqrt(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::sqrt) }

#[pymodule]
fn applegpu_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(init_backend, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_size, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(eval, m)?)?;
    m.add_function(wrap_pyfunction!(to_list, m)?)?;
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(destroy, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(neg, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
