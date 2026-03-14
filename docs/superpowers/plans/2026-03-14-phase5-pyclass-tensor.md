# Phase 5: PyO3 Class Wrapper with Automatic Cleanup

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace opaque `u64` tensor IDs with a proper `GpuTensor` Python class that has methods (`.to_list()`, `.shape`, `.eval()`), operator overloads (`+`, `-`, `*`, `/`, `@` for matmul), and automatic GPU memory cleanup via `__del__`.

**Architecture:** A `#[pyclass] GpuTensor` wraps the u64 ID and provides `#[pymethods]` for all tensor operations. Operators return new `GpuTensor` instances. `Drop` (via `__del__`) calls `LazyRuntime::destroy()` to free GPU memory automatically when Python garbage-collects the object. Module-level functions (`gpu.add()`, `gpu.to_list()`, etc.) are preserved for backward compatibility but delegate to the class methods.

**Tech Stack:** Rust (PyO3 `#[pyclass]`, `#[pymethods]`), Python

**Breaking changes:** The return type of `gpu.tensor()`, `gpu.add()`, etc. changes from `int` to `GpuTensor`. Code using `gpu.to_list(t)` still works (backward compatible). New style: `t.to_list()`.

---

## File Structure

### Modified Files
- `crates/python/src/lib.rs` — Add `GpuTensor` pyclass, rewrite ops to return it, keep module-level compat functions
- `python/applegpu_runtime/__init__.py` — Export `GpuTensor` class
- `python/tests/test_compute.py` — Update tests to use new API (class methods + operators)
- `python/tests/test_lazy.py` — Update for class-based API
- `python/tests/test_tensor.py` — Update for class-based API

### New Files
- `python/tests/test_class_api.py` — Tests for GpuTensor class methods, operators, repr, and auto-cleanup

---

## Chunk 1: GpuTensor PyO3 Class

### Task 1: Implement GpuTensor class and rewrite Python bindings

**Files:**
- Modify: `crates/python/src/lib.rs`

- [ ] **Step 1: Rewrite lib.rs with GpuTensor pyclass**

Replace `crates/python/src/lib.rs` entirely:

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;

use applegpu_core::lazy::LazyRuntime;
use applegpu_core::tensor::Tensor;

/// Global lazy runtime.
static RUNTIME_LAZY: once_cell::sync::Lazy<Mutex<LazyRuntime>> =
    once_cell::sync::Lazy::new(|| Mutex::new(LazyRuntime::new()));

/// Helper: get the backend device, ensuring init_backend() was called.
fn get_device_runtime() -> PyResult<&'static applegpu_core::backend::Runtime> {
    applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))
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

    /// Read tensor data as a flat list of f32 values.
    /// Auto-evaluates if the tensor is lazy.
    fn to_list(&self) -> PyResult<Vec<f32>> {
        let runtime = get_device_runtime()?;
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        if rt.is_pending(self.id) {
            rt.eval(&runtime.device, self.id)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        rt.read_f32(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Explicitly evaluate this tensor, materializing its result on the GPU.
    fn eval(&self) -> PyResult<()> {
        let runtime = get_device_runtime()?;
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        rt.eval(&runtime.device, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Apply element-wise negation.
    fn neg(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::neg(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    /// Apply ReLU activation.
    fn relu(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::relu(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    /// Apply element-wise exponential.
    fn exp(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::exp(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    /// Apply element-wise natural log.
    fn log(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::log(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    /// Apply element-wise square root.
    fn sqrt(&self) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::sqrt(&mut rt, self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    /// Element-wise addition.
    fn add(&self, other: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::add(&mut rt, self.id, other.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    /// Element-wise subtraction.
    fn sub(&self, other: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::sub(&mut rt, self.id, other.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    /// Element-wise multiplication.
    fn mul(&self, other: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::mul(&mut rt, self.id, other.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    /// Element-wise division.
    fn div(&self, other: &GpuTensor) -> PyResult<GpuTensor> {
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        let id = applegpu_core::ops::div(&mut rt, self.id, other.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GpuTensor { id, destroyed: Cell::new(false) })
    }

    /// Matrix multiply: self @ other
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
/// Uses try_lock() to avoid deadlocking if Drop fires while the Mutex is held
/// (e.g., GC triggered during a locked operation). Checks `destroyed` flag to
/// prevent double-destroy if explicit `gpu.destroy()` was already called.
impl Drop for GpuTensor {
    fn drop(&mut self) {
        if self.destroyed.get() {
            return; // already cleaned up
        }
        // try_lock: if the mutex is held (e.g., GC fired mid-operation), skip cleanup.
        // The tensor will leak, but that's better than a deadlock. Users can call
        // gpu.destroy() explicitly for guaranteed cleanup.
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

/// Create a tensor from data. Returns a GpuTensor object.
#[pyfunction]
fn tensor(data: Vec<f32>, shape: Vec<usize>) -> PyResult<GpuTensor> {
    let runtime = get_device_runtime()?;
    let t = Tensor::from_f32(&runtime.device, shape, &data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let id = t.meta.id;
    RUNTIME_LAZY.lock().unwrap().insert_tensor(t);
    Ok(GpuTensor { id, destroyed: Cell::new(false) })
}

// Backward-compatible module-level functions that accept GpuTensor

/// Read tensor data as a flat list. Auto-evaluates lazy tensors.
#[pyfunction]
fn to_list(t: &GpuTensor) -> PyResult<Vec<f32>> {
    t.to_list()
}

/// Get shape of a tensor.
#[pyfunction]
fn shape(t: &GpuTensor) -> PyResult<Vec<usize>> {
    t.shape()
}

/// Explicitly evaluate a tensor.
#[pyfunction]
fn eval(t: &GpuTensor) -> PyResult<()> {
    t.eval()
}

/// Destroy a tensor, freeing GPU memory.
/// Sets the destroyed flag so Drop won't double-destroy.
#[pyfunction]
fn destroy(t: &GpuTensor) -> PyResult<()> {
    if t.destroyed.get() {
        return Ok(()); // already destroyed
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

#[pymodule]
fn applegpu_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GpuTensor>()?;
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
```

- [ ] **Step 2: Update __init__.py to export GpuTensor**

```python
"""Apple GPU Runtime - Unified API for GPU operations on Apple Silicon."""

from applegpu_runtime.applegpu_runtime import (
    GpuTensor,
    version,
    init_backend,
    device_name,
    dtype_size,
    tensor,
    eval,
    to_list,
    shape,
    destroy,
    add,
    sub,
    mul,
    div,
    neg,
    relu,
    exp,
    log,
    sqrt,
    matmul,
)

__version__ = version()
__all__ = [
    "GpuTensor",
    "version",
    "init_backend",
    "device_name",
    "dtype_size",
    "tensor",
    "eval",
    "to_list",
    "shape",
    "destroy",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "relu",
    "exp",
    "log",
    "sqrt",
    "matmul",
]
```

- [ ] **Step 3: Build and verify compilation**

Run: `uv run maturin develop 2>&1`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add crates/python/src/lib.rs python/applegpu_runtime/__init__.py
git commit -m "feat: add GpuTensor PyO3 class with operators and auto-cleanup"
```

---

### Task 2: Update existing tests for new API and add class-specific tests

**Files:**
- Modify: `python/tests/test_compute.py`
- Modify: `python/tests/test_lazy.py`
- Modify: `python/tests/test_tensor.py`
- Create: `python/tests/test_class_api.py`

- [ ] **Step 1: Verify existing tests pass without modification**

Run: `uv run pytest python/tests/test_compute.py python/tests/test_lazy.py python/tests/test_tensor.py python/tests/test_smoke.py python/tests/test_backend.py python/tests/test_tensor_types.py -v 2>&1`

The module-level functions (`gpu.add(a, b)`, `gpu.to_list(c)`, etc.) now take `GpuTensor` instead of `u64`, but since `gpu.tensor()` now returns `GpuTensor` and all functions accept it, **existing tests should pass unchanged**.

Expected: All existing tests pass

- [ ] **Step 2: Create test_class_api.py with class-specific tests**

```python
import applegpu_runtime as gpu


def test_tensor_is_gpu_tensor():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    assert isinstance(t, gpu.GpuTensor)


def test_shape_property():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    assert t.shape == [2, 3]


def test_to_list_method():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    assert t.to_list() == [1.0, 2.0, 3.0]


def test_eval_method():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0], shape=[2])
    b = gpu.tensor([3.0, 4.0], shape=[2])
    c = a + b
    c.eval()
    assert c.to_list() == [4.0, 6.0]


def test_add_operator():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    b = gpu.tensor([10.0, 20.0, 30.0], shape=[3])
    c = a + b
    assert isinstance(c, gpu.GpuTensor)
    assert c.to_list() == [11.0, 22.0, 33.0]


def test_sub_operator():
    gpu.init_backend()
    a = gpu.tensor([10.0, 20.0], shape=[2])
    b = gpu.tensor([1.0, 2.0], shape=[2])
    assert (a - b).to_list() == [9.0, 18.0]


def test_mul_operator():
    gpu.init_backend()
    a = gpu.tensor([2.0, 3.0], shape=[2])
    b = gpu.tensor([4.0, 5.0], shape=[2])
    assert (a * b).to_list() == [8.0, 15.0]


def test_div_operator():
    gpu.init_backend()
    a = gpu.tensor([10.0, 20.0], shape=[2])
    b = gpu.tensor([2.0, 5.0], shape=[2])
    assert (a / b).to_list() == [5.0, 4.0]


def test_matmul_operator():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = a @ b
    assert c.to_list() == [19.0, 22.0, 43.0, 50.0]
    assert c.shape == [2, 2]


def test_neg_operator():
    gpu.init_backend()
    a = gpu.tensor([1.0, -2.0, 3.0], shape=[3])
    b = -a
    assert b.to_list() == [-1.0, 2.0, -3.0]


def test_chained_operators():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    # (a + b) * a = [11, 22, 33, 44] * [1, 2, 3, 4] = [11, 44, 99, 176]
    c = (a + b) * a
    assert c.to_list() == [11.0, 44.0, 99.0, 176.0]


def test_repr():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    r = repr(t)
    assert "GpuTensor" in r
    assert "[3]" in r
    assert "materialized" in r


def test_repr_lazy():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0], shape=[2])
    b = gpu.tensor([3.0, 4.0], shape=[2])
    c = a + b
    r = repr(c)
    assert "lazy" in r


def test_unary_methods():
    gpu.init_backend()
    a = gpu.tensor([4.0, 9.0, 16.0], shape=[3])
    assert a.sqrt().to_list() == [2.0, 3.0, 4.0]

    b = gpu.tensor([-1.0, 2.0, -3.0], shape=[3])
    assert b.relu().to_list() == [0.0, 2.0, 0.0]
```

- [ ] **Step 3: Run all tests**

Run: `uv run pytest -v 2>&1`
Expected: All tests pass (existing + new class API tests)

- [ ] **Step 4: Commit**

```bash
git add python/tests/test_class_api.py
git commit -m "test: add GpuTensor class API tests with operators and methods"
```

---

### Task 3: End-to-end verification and push

- [ ] **Step 1: Run full test suite from clean**

Run: `make clean && make test 2>&1`
Expected: All tests pass

- [ ] **Step 2: Update backlog**

Mark Phase 5 automatic cleanup as complete.

- [ ] **Step 3: Update README with new API style**

Show the operator-based API as the primary example.

- [ ] **Step 4: Push**

```bash
git push origin main
```
