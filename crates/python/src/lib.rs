use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

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

/// The Python module definition.
#[pymodule]
fn applegpu_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(init_backend, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_size, m)?)?;
    Ok(())
}
