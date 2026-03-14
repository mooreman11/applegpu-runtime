use pyo3::prelude::*;

/// Returns the library version.
#[pyfunction]
fn version() -> &'static str {
    applegpu_core::version()
}

/// The Python module definition.
#[pymodule]
fn applegpu_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
