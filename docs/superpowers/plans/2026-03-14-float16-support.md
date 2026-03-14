# Float16 Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native float16 GPU operations with 2x throughput on Apple Silicon, including f16 tensor creation, dtype-aware dispatch, and NumPy/PyTorch f16 interop.

**Architecture:** MSL preprocessor templates (`#define DTYPE half/float`) for 10 element-wise ops. Custom f32-intermediate kernels for matmul/softmax/scalar_mul. Dtype inferred from input tensors, propagated through graph. `half` crate for Rust f16↔f32 conversion.

**Tech Stack:** Rust (applegpu-core, `half` crate), MSL (Metal Shading Language), PyO3 + pyo3-numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-14-float16-support-design.md`

---

## Chunk 1: Tensor Foundation

### Task 1: Add `half` crate dependency and f16 tensor constructors

**Files:**
- Modify: `crates/core/Cargo.toml` — add `half = "2"`
- Modify: `crates/core/src/tensor.rs` — add `from_raw` dtype param, `empty_f16`, `from_f16`, `as_f16_slice`, dtype check on `as_f32_slice`

- [ ] **Step 1: Add `half` dependency to `crates/core/Cargo.toml`**

- [ ] **Step 2: Write tests in tensor.rs test module**

```rust
#[test]
fn test_empty_f16_creates_correct_size() {
    let device = match crate::device::Device::new() { Ok(d) => d, Err(_) => return };
    let t = Tensor::empty_f16(&device, vec![4]).unwrap();
    assert_eq!(t.meta.dtype, DType::Float16);
    assert_eq!(t.meta.size_bytes(), 8); // 4 elements * 2 bytes
    assert_eq!(t.buffer.len(), 8);
}

#[test]
fn test_from_f16_roundtrip() {
    let device = match crate::device::Device::new() { Ok(d) => d, Err(_) => return };
    use half::f16;
    let data: Vec<u16> = vec![
        f16::from_f32(1.0).to_bits(),
        f16::from_f32(2.0).to_bits(),
        f16::from_f32(3.0).to_bits(),
        f16::from_f32(4.0).to_bits(),
    ];
    let t = Tensor::from_f16(&device, vec![4], &data).unwrap();
    let result = t.as_f16_slice();
    assert_eq!(result, &data);
}

#[test]
fn test_as_f32_slice_errors_on_f16() {
    let device = match crate::device::Device::new() { Ok(d) => d, Err(_) => return };
    let t = Tensor::empty_f16(&device, vec![4]).unwrap();
    // as_f32_slice should panic or return empty for f16 tensor
    // Implementation: add assert or check
}

#[test]
fn test_from_raw_respects_dtype() {
    let device = match crate::device::Device::new() { Ok(d) => d, Err(_) => return };
    let buf = crate::buffer::Buffer::new(&device, 8).unwrap();
    let t = Tensor::from_raw(1, vec![4], DType::Float16, buf);
    assert_eq!(t.meta.dtype, DType::Float16);
    assert_eq!(t.meta.size_bytes(), 8);
}
```

- [ ] **Step 3: Implement changes**

Change `Tensor::from_raw` signature to accept `dtype: DType`:
```rust
pub fn from_raw(id: u64, shape: Vec<usize>, dtype: DType, buffer: Buffer) -> Self {
    Tensor {
        meta: TensorMeta { id, shape: Shape::new(shape), dtype, location: TensorLocation::Shared },
        buffer,
    }
}
```

Add `empty_f16`, `from_f16`, `as_f16_slice`:
```rust
pub fn empty_f16(device: &Device, shape: Vec<usize>) -> Result<Self> {
    let numel: usize = shape.iter().product();
    let size_bytes = numel * 2; // f16 = 2 bytes
    let buffer = Buffer::new(device, size_bytes)?;
    let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(Tensor {
        meta: TensorMeta { id, shape: Shape::new(shape), dtype: DType::Float16, location: TensorLocation::Shared },
        buffer,
    })
}

pub fn from_f16(device: &Device, shape: Vec<usize>, data: &[u16]) -> Result<Self> {
    let expected = shape.iter().product::<usize>();
    if data.len() != expected {
        return Err(crate::error::GpuError::InvalidTensor(format!(
            "Shape {:?} expects {} elements but got {}", shape, expected, data.len()
        )));
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
    };
    let buffer = Buffer::from_bytes(device, bytes)?;
    let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(Tensor {
        meta: TensorMeta { id, shape: Shape::new(shape), dtype: DType::Float16, location: TensorLocation::Shared },
        buffer,
    })
}

pub fn as_f16_slice(&self) -> &[u16] {
    assert_eq!(self.meta.dtype, DType::Float16, "as_f16_slice called on non-f16 tensor");
    let count = self.meta.shape.numel();
    unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const u16, count) }
}
```

Update `as_f32_slice` to check dtype:
```rust
pub fn as_f32_slice(&self) -> &[f32] {
    assert_eq!(self.meta.dtype, DType::Float32, "as_f32_slice called on non-f32 tensor");
    let count = self.meta.shape.numel();
    unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const f32, count) }
}
```

- [ ] **Step 4: Fix ALL `from_raw` call sites** to pass dtype parameter

Every `Tensor::from_raw(id, shape, buf)` becomes `Tensor::from_raw(id, shape, DType::Float32, buf)` (or the correct dtype). Search all files:
- `crates/core/src/lazy.rs` — `execute_node` uses `from_raw` with pool. Pass `node.out_dtype`.
- `crates/core/src/lazy.rs` — `eval_remote` uses `from_raw`. Pass `DType::Float32` (VM backend is f32-only for now).

- [ ] **Step 5: Add `LazyRuntime::dtype()` method and `read_f16()`**

```rust
pub fn dtype(&self, id: u64) -> Result<DType> {
    if let Some(t) = self.tensors.get(&id) {
        return Ok(t.meta.dtype);
    }
    if let Some(node) = self.graph.get_node(id) {
        return Ok(node.out_dtype);
    }
    Err(GpuError::GraphError(format!("Tensor {} not found", id)))
}

pub fn read_f16(&self, id: u64) -> Result<Vec<u16>> {
    let t = self.get_tensor(id)?;
    Ok(t.as_f16_slice().to_vec())
}
```

- [ ] **Step 6: Run all tests, commit**

```bash
cargo test -p applegpu-core
git add crates/core/Cargo.toml crates/core/src/tensor.rs crates/core/src/lazy.rs Cargo.lock
git commit -m "feat: add f16 tensor constructors, dtype-aware from_raw, read_f16"
```

---

## Chunk 2: F16 Kernels and Dispatch

### Task 2: Add f16 MSL kernel sources and dtype-aware dispatch

**Files:**
- Modify: `crates/core/src/compute.rs` — add f16 kernel sources (preprocessor templates + custom kernels), update dispatch methods with dtype parameter

- [ ] **Step 1: Add preprocessor-based f16 kernel templates**

Create f16 versions by prepending `#define DTYPE half` and renaming functions. For element-wise ops (binary + unary + transpose), the kernel body is identical with `DTYPE` substituted.

Add new constants for f16:
```rust
fn binary_kernel_source_f16() -> String {
    BINARY_KERNEL_SOURCE.replace("float", "half")
        .replace("elementwise_add", "elementwise_add_f16")
        .replace("elementwise_sub", "elementwise_sub_f16")
        .replace("elementwise_mul", "elementwise_mul_f16")
        .replace("elementwise_div", "elementwise_div_f16")
}
```

Actually, the simplest reliable approach: write dedicated f16 source constants (not string replacement, which is fragile). Create `BINARY_KERNEL_SOURCE_F16`, `UNARY_KERNEL_SOURCE_F16`, `TRANSPOSE_KERNEL_SOURCE_F16` with `half` types.

For matmul, softmax, scalar_mul: write custom f16 kernels with f32 intermediates.

- [ ] **Step 2: Add dtype parameter to dispatch methods**

`dispatch_binary`, `dispatch_unary`, `dispatch_matmul`, `dispatch_softmax`, `dispatch_transpose`, `dispatch_scalar_mul` all gain `dtype: DType` parameter. They select the correct kernel source and function name based on dtype.

- [ ] **Step 3: Write f16 dispatch tests**

```rust
#[test]
fn test_dispatch_binary_f16() {
    let device = match Device::new() { Ok(d) => d, Err(_) => return };
    use half::f16;
    let a_data: Vec<u16> = vec![f16::from_f32(1.0).to_bits(), f16::from_f32(2.0).to_bits()];
    let b_data: Vec<u16> = vec![f16::from_f32(3.0).to_bits(), f16::from_f32(4.0).to_bits()];
    let a_bytes = unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const u8, 4) };
    let b_bytes = unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, 4) };
    let a = Buffer::from_bytes(&device, a_bytes).unwrap();
    let b = Buffer::from_bytes(&device, b_bytes).unwrap();
    let out = Buffer::new(&device, 4).unwrap();
    // dispatch_binary with f16
    let registry = KernelRegistry::new();
    registry.dispatch_binary(&device, "elementwise_add_f16", &a, &b, &out, 2, DType::Float16).unwrap();
    let result = unsafe { out.as_slice::<u16>() };
    assert_eq!(f16::from_bits(result[0]).to_f32(), 4.0);
    assert_eq!(f16::from_bits(result[1]).to_f32(), 6.0);
}
```

- [ ] **Step 4: Run tests, commit**

```bash
cargo test -p applegpu-core
git add crates/core/src/compute.rs
git commit -m "feat: add f16 MSL kernels and dtype-aware dispatch"
```

### Task 3: Update ops and fusion for dtype inference

**Files:**
- Modify: `crates/core/src/ops.rs` — dtype inference from inputs
- Modify: `crates/core/src/fusion.rs` — dtype-aware MSL generation
- Modify: `crates/core/src/lazy.rs` — pass dtype to dispatch in execute_node

- [ ] **Step 1: Update ops.rs for dtype inference**

Change `lazy_binary_op` and `lazy_unary_op` to infer dtype:
```rust
fn lazy_binary_op(rt: &mut LazyRuntime, a_id: u64, b_id: u64, op: OpKind) -> Result<u64> {
    let a_shape = rt.shape(a_id)?;
    let b_shape = rt.shape(b_id)?;
    let a_dtype = rt.dtype(a_id)?;
    let b_dtype = rt.dtype(b_id)?;
    if a_shape != b_shape { /* existing error */ }
    if a_dtype != b_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "Dtype mismatch: {:?} vs {:?}", a_dtype, b_dtype
        )));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id, op, inputs: vec![a_id, b_id],
        out_shape: Shape::new(a_shape), out_dtype: a_dtype, // was: DType::Float32
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}
```

Similarly for `lazy_unary_op`, `matmul`, `softmax`, `transpose`, `scalar_mul`.

- [ ] **Step 2: Update fusion.rs for dtype-aware MSL**

`generate_fused_msl` uses the chain's `out_dtype` to emit correct buffer types:
- `DType::Float16` → `device const half*`, `device half*`, use `(half)0` for relu
- `DType::Float32` → existing `device const float*` behavior

- [ ] **Step 3: Update execute_node in lazy.rs**

Pass `node.out_dtype` to dispatch methods. The pool acquire already uses logical size which is dtype-aware via `node.out_shape.numel() * node.out_dtype.size_bytes()`.

- [ ] **Step 4: Write f16 ops integration test**

```rust
#[test]
fn test_f16_add_eval() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();
    use half::f16;
    let a_data: Vec<u16> = vec![f16::from_f32(1.0).to_bits(); 4];
    let b_data: Vec<u16> = vec![f16::from_f32(2.0).to_bits(); 4];
    let a = Tensor::from_f16(&device, vec![4], &a_data).unwrap();
    let b = Tensor::from_f16(&device, vec![4], &b_data).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();
    let c_id = crate::ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();
    let result = rt.read_f16(c_id).unwrap();
    assert_eq!(f16::from_bits(result[0]).to_f32(), 3.0);
}

#[test]
fn test_mixed_dtype_errors() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();
    let a = Tensor::from_f32(&device, vec![4], &[1.0; 4]).unwrap();
    let b = Tensor::from_f16(&device, vec![4], &[0u16; 4]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();
    assert!(crate::ops::add(&mut rt, a_id, b_id).is_err());
}
```

- [ ] **Step 5: Run all tests, commit**

```bash
cargo test -p applegpu-core
git add crates/core/src/ops.rs crates/core/src/fusion.rs crates/core/src/lazy.rs
git commit -m "feat: dtype inference in ops, dtype-aware fusion, f16 eval pipeline"
```

---

## Chunk 3: Python Bindings

### Task 4: Python f16 support (tensor creation, dtype getter, to_list, to_numpy, to_torch)

**Files:**
- Modify: `crates/python/src/lib.rs`
- Modify: `crates/python/Cargo.toml` — add `half = "2"`

- [ ] **Step 1: Add `half` to python crate deps**

- [ ] **Step 2: Add dtype getter to GpuTensor**

```rust
#[getter]
fn dtype(&self) -> PyResult<String> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    let dtype = rt.dtype(self.id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(match dtype {
        applegpu_core::tensor::DType::Float32 => "float32".to_string(),
        applegpu_core::tensor::DType::Float16 => "float16".to_string(),
        other => format!("{:?}", other).to_lowercase(),
    })
}
```

- [ ] **Step 3: Update `tensor()` function to accept optional dtype**

```rust
#[pyfunction]
#[pyo3(signature = (data, shape, dtype=None))]
fn tensor(data: Vec<f64>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<GpuTensor> {
    let runtime = get_device_runtime()?;
    let dtype_str = dtype.unwrap_or("float32");
    match dtype_str {
        "float32" | "f32" => {
            let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            let t = Tensor::from_f32(&runtime.device, shape, &f32_data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let id = t.meta.id;
            RUNTIME_LAZY.lock().unwrap().insert_tensor(t)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(GpuTensor { id, destroyed: Cell::new(false) })
        }
        "float16" | "f16" => {
            use half::f16;
            let f16_data: Vec<u16> = data.iter().map(|&x| f16::from_f64(x).to_bits()).collect();
            let t = Tensor::from_f16(&runtime.device, shape, &f16_data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let id = t.meta.id;
            RUNTIME_LAZY.lock().unwrap().insert_tensor(t)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(GpuTensor { id, destroyed: Cell::new(false) })
        }
        _ => Err(PyValueError::new_err(format!("Unsupported dtype: {}. Use 'float32' or 'float16'.", dtype_str))),
    }
}
```

- [ ] **Step 4: Update to_list for f16**

```rust
fn to_list(&self) -> PyResult<Vec<f64>> {
    // ... auto-eval ...
    let rt = RUNTIME_LAZY.lock().unwrap();
    let dtype = rt.dtype(self.id).map_err(|e| PyValueError::new_err(e.to_string()))?;
    match dtype {
        DType::Float32 => {
            let data = rt.read_f32(self.id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(data.iter().map(|&x| x as f64).collect())
        }
        DType::Float16 => {
            use half::f16;
            let data = rt.read_f16(self.id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(data.iter().map(|&x| f16::from_bits(x).to_f64()).collect())
        }
        _ => Err(PyValueError::new_err("Unsupported dtype for to_list")),
    }
}
```

- [ ] **Step 5: Update from_numpy to auto-detect f16**

Add f16 branch: if `arr.dtype == np.float16`, read as u16 and call `Tensor::from_f16`.

- [ ] **Step 6: Update to_numpy for f16**

f16 path: read u16 data, create PyArray of u16, then call `.view(np.float16).reshape(shape)` via PyO3.

- [ ] **Step 7: Update from_torch for f16**

Add check: if `tensor.dtype == torch.float16`, the `.numpy()` call works on CPU f16 tensors.

- [ ] **Step 8: Build and test**

```bash
uv run maturin develop
```

- [ ] **Step 9: Commit**

```bash
git add crates/python/Cargo.toml crates/python/src/lib.rs Cargo.lock
git commit -m "feat: Python f16 support (tensor creation, dtype getter, to_list, to_numpy, from_torch)"
```

### Task 5: Python f16 tests

**Files:**
- Create: `python/tests/test_float16.py`

- [ ] **Step 1: Write tests**

```python
import numpy as np
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


def test_tensor_f16_creation():
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4], dtype="float16")
    assert t.dtype == "float16"
    assert t.shape == [4]


def test_f16_to_list():
    t = gpu.tensor([1.0, 2.0, 3.0], shape=[3], dtype="float16")
    result = t.to_list()
    assert len(result) == 3
    assert abs(result[0] - 1.0) < 0.01
    assert abs(result[1] - 2.0) < 0.01


def test_f16_from_numpy_roundtrip():
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    t = gpu.from_numpy(arr)
    assert t.dtype == "float16"
    result = t.to_numpy()
    assert result.dtype == np.float16
    np.testing.assert_allclose(result, arr, rtol=1e-3)


def test_f16_ops():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4], dtype="float16")
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[4], dtype="float16")
    c = a + b
    result = c.to_list()
    assert abs(result[0] - 6.0) < 0.1
    assert abs(result[3] - 12.0) < 0.1


def test_f16_matmul():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2], dtype="float16")
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2], dtype="float16")
    c = a @ b
    result = c.to_list()
    assert abs(result[0] - 19.0) < 0.5  # f16 tolerance
    assert abs(result[3] - 50.0) < 0.5


def test_mixed_dtype_error():
    a = gpu.tensor([1.0], shape=[1], dtype="float16")
    b = gpu.tensor([1.0], shape=[1])  # float32
    with pytest.raises(ValueError, match="[Dd]type"):
        c = a + b


def test_dtype_getter():
    t32 = gpu.tensor([1.0], shape=[1])
    t16 = gpu.tensor([1.0], shape=[1], dtype="float16")
    assert t32.dtype == "float32"
    assert t16.dtype == "float16"


def test_f16_from_torch_roundtrip():
    import torch
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    g = gpu.from_torch(t)
    assert g.dtype == "float16"
    result = g.to_torch()
    assert result.dtype == torch.float16
    assert torch.allclose(result, t, atol=0.01)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest python/tests/test_float16.py -v
```

- [ ] **Step 3: Run full suite**

```bash
uv run pytest -v
cargo test -p applegpu-core
```

- [ ] **Step 4: Commit**

```bash
git add python/tests/test_float16.py
git commit -m "test: add float16 Python tests"
```

### Task 6: Update README, backlog, project status

- [ ] **Step 1: Update README with f16 capabilities and test count**
- [ ] **Step 2: Update docs/BACKLOG.md — mark multi-dtype (f16) as complete**
- [ ] **Step 3: Update project status memory**
- [ ] **Step 4: Commit and push**
