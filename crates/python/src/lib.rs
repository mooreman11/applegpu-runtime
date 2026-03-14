use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::sync::Mutex;

use applegpu_core::tensor::Tensor;

/// Global tensor storage. Tensors are stored by ID and accessed by opaque handle.
static TENSORS: once_cell::sync::Lazy<Mutex<HashMap<u64, Tensor>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// Helper: run a binary op on two tensor IDs.
fn binary_op_py(a_id: u64, b_id: u64, op: fn(&applegpu_core::device::Device, &Tensor, &Tensor) -> applegpu_core::error::Result<Tensor>) -> PyResult<u64> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out = {
        let tensors = TENSORS.lock().unwrap();
        let a = tensors.get(&a_id).ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", a_id)))?;
        let b = tensors.get(&b_id).ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", b_id)))?;
        op(&runtime.device, a, b).map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    let id = out.meta.id;
    TENSORS.lock().unwrap().insert(id, out);
    Ok(id)
}

/// Helper: run a unary op on a tensor ID.
fn unary_op_py(input_id: u64, op: fn(&applegpu_core::device::Device, &Tensor) -> applegpu_core::error::Result<Tensor>) -> PyResult<u64> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out = {
        let tensors = TENSORS.lock().unwrap();
        let input = tensors.get(&input_id).ok_or_else(|| PyValueError::new_err(format!("Tensor {} not found", input_id)))?;
        op(&runtime.device, input).map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    let id = out.meta.id;
    TENSORS.lock().unwrap().insert(id, out);
    Ok(id)
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

// Binary ops
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

// Unary ops
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
