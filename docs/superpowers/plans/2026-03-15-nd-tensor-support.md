# N-Dimensional Tensor Support — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 2D-only tensor architecture with N-dimensional support (MAX_DIMS=8), stride-based MSL kernels, and NumPy-style broadcasting for element-wise ops.

**Architecture:** Shape becomes a fixed-size stack-allocated struct (no Vec). TensorLayout adds element strides on TensorMeta. All MSL element-wise kernels rewritten with `nd_index_to_offset` stride-based indexing. Broadcasting via stride=0 on size-1 dims. Non-element-wise ops validate ndim==2 with clear errors.

**Tech Stack:** Rust (applegpu-core), MSL (Metal Shading Language), Swift (AppleGPUBridge), PyO3

**Spec:** `docs/superpowers/specs/2026-03-15-nd-tensor-support-design.md`

**CRITICAL:** This is a big-bang rewrite touching 12+ files. Follow the implementation order exactly — each task builds on the previous. Tests will break between tasks and must be fixed progressively.

---

## Chunk 1: Data Structures (Shape + TensorLayout + TensorMeta)

### Task 1: New Shape struct

**Files:**
- Modify: `crates/core/src/tensor.rs`

Replace the current `Shape(Vec<usize>)` with the fixed-size stack-allocated version. This breaks everything downstream — the rest of the plan fixes it.

- [ ] **Step 1: Write Shape tests first**

Add these tests (they define the contract):

```rust
#[cfg(test)]
mod shape_tests {
    use super::*;

    #[test]
    fn test_shape_2d() {
        let s = Shape::new(vec![2, 3]).unwrap();
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.dims(), &[2, 3]);
        assert_eq!(s.numel(), 6);
    }

    #[test]
    fn test_shape_3d() {
        let s = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.dims(), &[2, 3, 4]);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn test_shape_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.dims(), &[]);
    }

    #[test]
    fn test_shape_1d() {
        let s = Shape::new(vec![5]).unwrap();
        assert_eq!(s.ndim(), 1);
        assert_eq!(s.dims(), &[5]);
        assert_eq!(s.numel(), 5);
    }

    #[test]
    fn test_shape_exceeds_max_dims() {
        let dims = vec![1; 9];
        assert!(Shape::new(dims).is_err());
    }

    #[test]
    fn test_shape_equality_ignores_padding() {
        let a = Shape::new(vec![2, 3]).unwrap();
        let b = Shape::new(vec![2, 3]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_shape_broadcast_bias() {
        let a = Shape::new(vec![4, 3]).unwrap();
        let b = Shape::new(vec![3]).unwrap();
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c.dims(), &[4, 3]);
    }

    #[test]
    fn test_shape_broadcast_3d() {
        let a = Shape::new(vec![2, 1, 4]).unwrap();
        let b = Shape::new(vec![3, 4]).unwrap();
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_shape_broadcast_incompatible() {
        let a = Shape::new(vec![2, 3]).unwrap();
        let b = Shape::new(vec![4, 3]).unwrap();
        assert!(a.broadcast_with(&b).is_err());
    }

    #[test]
    fn test_shape_copy() {
        let a = Shape::new(vec![2, 3]).unwrap();
        let b = a; // Copy
        assert_eq!(a, b);
    }
}
```

- [ ] **Step 2: Implement new Shape**

Replace the entire Shape definition in tensor.rs:

```rust
use crate::error::{GpuError, Result};

pub const MAX_DIMS: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct Shape {
    dims: [usize; MAX_DIMS],
    ndim: usize,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Result<Self> {
        if dims.len() > MAX_DIMS {
            return Err(GpuError::InvalidTensor(format!(
                "Shape has {} dimensions, maximum is {}", dims.len(), MAX_DIMS
            )));
        }
        let mut arr = [1usize; MAX_DIMS];
        for (i, &d) in dims.iter().enumerate() {
            arr[i] = d;
        }
        Ok(Shape { dims: arr, ndim: dims.len() })
    }

    pub fn scalar() -> Self {
        Shape { dims: [1; MAX_DIMS], ndim: 0 }
    }

    pub fn ndim(&self) -> usize { self.ndim }

    pub fn dims(&self) -> &[usize] { &self.dims[..self.ndim] }

    pub fn numel(&self) -> usize {
        self.dims[..self.ndim].iter().product::<usize>().max(1)
    }

    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape> {
        let out_ndim = self.ndim.max(other.ndim);
        let mut out_dims = [1usize; MAX_DIMS];
        for i in 0..out_ndim {
            let a = if i < self.ndim { self.dims[self.ndim - 1 - i] } else { 1 };
            let b = if i < other.ndim { other.dims[other.ndim - 1 - i] } else { 1 };
            if a == b {
                out_dims[out_ndim - 1 - i] = a;
            } else if a == 1 {
                out_dims[out_ndim - 1 - i] = b;
            } else if b == 1 {
                out_dims[out_ndim - 1 - i] = a;
            } else {
                return Err(GpuError::InvalidTensor(format!(
                    "Cannot broadcast shapes {:?} and {:?}: dim {} is {} vs {}",
                    self.dims(), other.dims(), out_ndim - 1 - i, a, b
                )));
            }
        }
        Ok(Shape { dims: out_dims, ndim: out_ndim })
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim && self.dims[..self.ndim] == other.dims[..other.ndim]
    }
}
impl Eq for Shape {}

impl std::hash::Hash for Shape {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ndim.hash(state);
        self.dims[..self.ndim].hash(state);
    }
}
```

- [ ] **Step 3: Fix ALL Shape::new call sites to handle Result**

`Shape::new` now returns `Result`. Every call site needs `.unwrap()` or `?`. Search the entire codebase for `Shape::new(` — there are ~50+ occurrences across ops.rs, graph.rs, fusion.rs, serial.rs, lazy.rs tests, ops.rs tests, and integration tests.

**Strategy:** For ops.rs functions that already return Result, use `?`. For test code, use `.unwrap()`. For graph.rs (OpNode construction in ops.rs), the shape is already validated before creating the OpNode, so `.unwrap()` is safe.

This step will break many tests temporarily. Fix the compilation errors first, then verify all tests pass.

- [ ] **Step 4: Run tests, fix remaining issues**

Run: `cargo test -p applegpu-core`
Fix any remaining compilation errors from the Shape change.

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/tensor.rs crates/core/src/ops.rs crates/core/src/graph.rs crates/core/src/fusion.rs crates/core/src/serial.rs crates/core/src/lazy.rs crates/core/tests/
git commit -m "feat: replace Vec-based Shape with fixed-size [usize; MAX_DIMS=8], add broadcasting"
```

### Task 2: TensorLayout struct

**Files:**
- Modify: `crates/core/src/tensor.rs`

- [ ] **Step 1: Write TensorLayout tests**

```rust
#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn test_contiguous_strides_2d() {
        let shape = Shape::new(vec![2, 3]).unwrap();
        let layout = TensorLayout::contiguous(shape);
        assert_eq!(layout.strides(), &[3, 1]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_contiguous_strides_3d() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        let layout = TensorLayout::contiguous(shape);
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_contiguous_strides_1d() {
        let shape = Shape::new(vec![5]).unwrap();
        let layout = TensorLayout::contiguous(shape);
        assert_eq!(layout.strides(), &[1]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_transpose_swaps_dims_and_strides() {
        let shape = Shape::new(vec![2, 3]).unwrap();
        let layout = TensorLayout::contiguous(shape).transpose(0, 1);
        assert_eq!(layout.shape.dims(), &[3, 2]);
        assert_eq!(layout.strides(), &[1, 3]);
        assert!(!layout.is_contiguous());
    }

    #[test]
    fn test_broadcast_strides() {
        let src = Shape::new(vec![3]).unwrap();
        let target = Shape::new(vec![4, 3]).unwrap();
        let strides = TensorLayout::broadcast_strides_for(&src, &target);
        assert_eq!(strides[..2], [0, 1]); // dim 0 broadcast (stride=0), dim 1 normal
    }

    #[test]
    fn test_scalar_layout() {
        let shape = Shape::scalar();
        let layout = TensorLayout::contiguous(shape);
        assert!(layout.is_contiguous());
        assert_eq!(layout.strides(), &[]);
    }
}
```

- [ ] **Step 2: Implement TensorLayout**

```rust
#[derive(Debug, Clone)]
pub struct TensorLayout {
    pub shape: Shape,
    strides: [usize; MAX_DIMS],
}

impl TensorLayout {
    pub fn contiguous(shape: Shape) -> Self {
        let mut strides = [0usize; MAX_DIMS];
        if shape.ndim() > 0 {
            strides[shape.ndim() - 1] = 1;
            for i in (0..shape.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape.dims[i + 1];
            }
        }
        TensorLayout { shape, strides }
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.ndim() == 0 { return true; }
        let mut expected = 1;
        for i in (0..self.shape.ndim()).rev() {
            if self.strides[i] != expected { return false; }
            expected *= self.shape.dims[i];
        }
        true
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.shape.ndim()]
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let mut new = self.clone();
        new.shape.dims.swap(dim0, dim1);
        new.strides.swap(dim0, dim1);
        new
    }

    pub fn broadcast_strides_for(source: &Shape, target: &Shape) -> [usize; MAX_DIMS] {
        let mut strides = [0usize; MAX_DIMS];
        let src_contiguous = TensorLayout::contiguous(*source);
        let offset = target.ndim() - source.ndim();
        for i in 0..source.ndim() {
            let target_i = i + offset;
            if source.dims[i] == target.dims[target_i] {
                strides[target_i] = src_contiguous.strides[i];
            } else {
                strides[target_i] = 0; // broadcast dim
            }
        }
        // Leading dims (not in source) already 0
        strides
    }
}

impl PartialEq for TensorLayout {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.strides[..self.shape.ndim()] == other.strides[..other.shape.ndim()]
    }
}
impl Eq for TensorLayout {}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p applegpu-core layout_tests`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/tensor.rs
git commit -m "feat: add TensorLayout with contiguous strides, transpose, broadcasting"
```

### Task 3: Migrate TensorMeta and Tensor constructors

**Files:**
- Modify: `crates/core/src/tensor.rs` (TensorMeta, all constructors, read methods)
- Modify: `crates/core/src/lazy.rs` (all meta.shape references → meta.layout.shape)

- [ ] **Step 1: Update TensorMeta**

Change `shape: Shape` to `layout: TensorLayout` in TensorMeta. Update `size_bytes()`.

- [ ] **Step 2: Update all Tensor constructors**

`from_data`, `from_f32`, `from_f16`, `empty`, `empty_f32`, `empty_f16`, `from_raw` — all create `TensorLayout::contiguous(shape)` instead of just `shape`.

- [ ] **Step 3: Update read methods to return Result**

`as_f32_slice`, `as_f16_slice`, `as_bytes` — check `self.meta.layout.is_contiguous()`, return `Result` with clear error if non-contiguous.

- [ ] **Step 4: Update all consumers of meta.shape → meta.layout.shape**

Search for `meta.shape` across the codebase. Key locations:
- `lazy.rs`: `shape()` method, `execute_node`, `execute_node_nb`, `eval_remote`
- `ops.rs`: any direct meta.shape access
- `python/lib.rs`: shape getter, to_numpy shape

Also update `LazyRuntime::shape()` to return from `meta.layout.shape.dims()` for materialized tensors.

- [ ] **Step 5: Fix all `as_f32_slice`/`as_f16_slice`/`as_bytes` callers for Result**

These now return Result. All callers need `?` or `.unwrap()`:
- `lazy.rs`: `read_f32`, `read_f16`, `read_bytes`, `eval_remote` serialization
- `python/lib.rs`: `to_list`, `to_numpy`

- [ ] **Step 6: Run ALL tests, fix compilation errors**

Run: `cargo test -p applegpu-core`
This will likely have many compilation errors. Fix them all.

Then: `uv run maturin develop && uv run pytest -v`
Fix any Python-side issues.

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/ crates/python/src/ Cargo.lock
git commit -m "feat: migrate TensorMeta to TensorLayout, read methods return Result"
```

---

## Chunk 2: MSL Kernel Rewrite

### Task 4: Element-wise kernel rewrite (binary + unary)

**Files:**
- Modify: `crates/core/src/compute.rs` (MSL sources + dispatch methods)
- Modify: `crates/core/src/ffi.rs` (new FFI signatures)
- Modify: `swift/Sources/AppleGPUBridge/compute.swift` (new Swift dispatch)
- Modify: `swift/Sources/AppleGPUBridge/include/bridge.h`

This is the largest task. Rewrite ALL element-wise kernel MSL sources and dispatch.

- [ ] **Step 1: Define the MSL index helper**

Create a constant string for the shared helper:

```rust
const ND_INDEX_HELPER: &str = r#"
uint nd_index_to_offset(uint flat_id, constant uint* shape, constant uint* strides, constant uint& ndim) {
    uint offset = 0;
    for (uint d = ndim; d > 0; d--) {
        uint i = d - 1;
        offset += (flat_id % shape[i]) * strides[i];
        flat_id /= shape[i];
    }
    return offset;
}
"#;
```

- [ ] **Step 2: Rewrite binary element-wise kernels**

Replace `BINARY_KERNEL_SOURCE` with N-D stride-based version. Each kernel takes `a_strides`, `b_strides`, `out_shape`, `ndim`, `numel` parameters.

Do the same for `BINARY_KERNEL_SOURCE_F16`.

- [ ] **Step 3: Rewrite unary element-wise kernels**

Replace `UNARY_KERNEL_SOURCE` and `UNARY_KERNEL_SOURCE_F16`.

- [ ] **Step 4: Rewrite GELU kernels**

Replace `GELU_KERNEL_SOURCE` and `GELU_KERNEL_SOURCE_F16`. Keep the clamp fix.

- [ ] **Step 5: Add Swift N-D dispatch functions**

New Swift dispatch functions that accept stride/shape/ndim arrays. These replace or supplement the existing dispatch functions.

The key difference: instead of `setBytes(&count, ...)` for a single uint, we now `setBytes` for stride arrays (uint[8]) and shape arrays (uint[8]).

- [ ] **Step 6: Add Rust FFI declarations**

New extern "C" declarations matching the Swift functions.

- [ ] **Step 7: Update KernelRegistry dispatch methods**

Update `dispatch_binary_typed`, `dispatch_unary_typed`, `dispatch_gelu_typed` (and their `_nb` variants) to pass strides and shapes.

- [ ] **Step 8: Update execute_node in lazy.rs**

Compute element strides from input tensor layouts, pass to dispatch.

For binary ops:
```rust
let a_strides = input_a.meta.layout.strides_as_u32();
let b_strides = // broadcast strides if shapes differ, else input_b strides
let out_shape = node.out_shape.dims_as_u32();
let ndim = node.out_shape.ndim() as u32;
let numel = node.out_shape.numel() as u32;
```

- [ ] **Step 9: Write N-D element-wise tests**

```rust
#[test]
fn test_add_3d() {
    // Create [2,3,4] tensors, add them, verify result
}

#[test]
fn test_add_broadcast() {
    // [4,3] + [3] → [4,3] via broadcasting
}

#[test]
fn test_relu_3d() {
    // 3D relu
}
```

- [ ] **Step 10: Run all tests**

Run: `cargo test -p applegpu-core`
Expected: All pass (including existing 2D tests)

- [ ] **Step 11: Commit**

```bash
git add crates/core/src/compute.rs crates/core/src/ffi.rs crates/core/src/lazy.rs swift/
git commit -m "feat: N-D stride-based element-wise kernels with broadcasting"
```

### Task 5: Ops.rs broadcasting update

**Files:**
- Modify: `crates/core/src/ops.rs`

- [ ] **Step 1: Update lazy_binary_op for broadcasting**

Replace exact shape match with broadcast:
```rust
fn lazy_binary_op(rt: &mut LazyRuntime, a_id: u64, b_id: u64, op: OpKind) -> Result<u64> {
    let a_shape = rt.shape(a_id)?;
    let b_shape = rt.shape(b_id)?;
    let a_dtype = rt.dtype(a_id)?;
    let b_dtype = rt.dtype(b_id)?;
    if a_dtype != b_dtype { /* error */ }
    validate_compute_dtype(a_dtype)?;

    let a_shape_obj = Shape::new(a_shape)?;
    let b_shape_obj = Shape::new(b_shape)?;
    let out_shape = a_shape_obj.broadcast_with(&b_shape_obj)?;

    let out_id = next_id();
    rt.record_op(OpNode { id: out_id, op, inputs: vec![a_id, b_id],
        out_shape, out_dtype: a_dtype, container_id: ContainerId::DEFAULT });
    Ok(out_id)
}
```

- [ ] **Step 2: Update lazy_unary_op for N-D**

Remove any 2D assumptions. Unary ops work on any ndim.

- [ ] **Step 3: Add ndim==2 validation to non-element-wise ops**

Add at the top of `matmul`, `softmax`, `softmax_causal`, `transpose`, `layer_norm`, `embedding`, `slice`, `concat`, `add_bias`, `argmax`:

```rust
if shape.len() != 2 {
    return Err(GpuError::InvalidTensor(format!(
        "{} requires 2D tensor, got {}D", "matmul", shape.len()
    )));
}
```

- [ ] **Step 4: Make add_bias a thin wrapper over add with broadcasting**

```rust
pub fn add_bias(rt: &mut LazyRuntime, input_id: u64, bias_id: u64) -> Result<u64> {
    // Broadcasting handles [rows, cols] + [cols] naturally
    lazy_binary_op(rt, input_id, bias_id, OpKind::Add)
}
```

- [ ] **Step 5: Write broadcast ops tests**

- [ ] **Step 6: Run all tests, commit**

```bash
git add crates/core/src/ops.rs
git commit -m "feat: broadcasting in binary ops, ndim validation for 2D-only ops"
```

### Task 6: Non-element-wise kernel updates (minimal)

**Files:**
- Modify: `crates/core/src/compute.rs` (matmul, softmax, etc. — signature changes only)

Non-element-wise ops keep their 2D kernel logic but their dispatch needs to accept the new stride/shape format for consistency. Since these ops validate ndim==2, strides will always be contiguous 2D.

- [ ] **Step 1: Update matmul, softmax, layer_norm, etc. dispatch to accept stride params**

Even though these ops are 2D-only, the dispatch path must pass the new-format parameters. Extract rows/cols from the shape array inside the kernel or on the Rust side.

- [ ] **Step 2: Verify all existing 2D tests pass**

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/compute.rs crates/core/src/lazy.rs swift/
git commit -m "refactor: update non-element-wise dispatch for N-D format compatibility"
```

---

## Chunk 3: Fusion + Serialization + Polish

### Task 7: Fusion engine update

**Files:**
- Modify: `crates/core/src/fusion.rs`

- [ ] **Step 1: Update fused kernel MSL generation for stride-based indexing**

Fused kernels must use `nd_index_to_offset` for input access. Since fused chains are always contiguous with matching shapes, strides will be contiguous — but the kernel signature must match the new format.

Update `generate_fused_msl` to emit kernels with stride/shape/ndim parameters.

- [ ] **Step 2: Update fused kernel dispatch in lazy.rs**

Pass strides and shapes to fused kernel dispatch.

- [ ] **Step 3: Verify fusion tests pass**

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/fusion.rs crates/core/src/lazy.rs
git commit -m "feat: fusion engine generates stride-based N-D kernels"
```

### Task 8: Serialization update

**Files:**
- Modify: `crates/core/src/serial.rs`

- [ ] **Step 1: Update Shape serialization**

Shape now has fixed-size arrays. Serialize only `dims[0..ndim]` (same wire format as before). Deserialize by calling `Shape::new(dims_vec).unwrap()`.

- [ ] **Step 2: Verify serialization tests pass**

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/serial.rs
git commit -m "refactor: update serialization for new Shape struct"
```

---

## Chunk 4: Python Layer + Tests

### Task 9: Python bindings update

**Files:**
- Modify: `crates/python/src/lib.rs`

- [ ] **Step 1: Update shape getter**

Read from `meta.layout.shape` instead of `meta.shape`.

- [ ] **Step 2: Update from_numpy/to_numpy for N-D**

`from_numpy` already handles N-D (reads shape from numpy). May need minor updates for the new Shape::new returning Result.

`to_numpy` reshape — already handles arbitrary shapes.

- [ ] **Step 3: Handle as_f32_slice Result in to_list, to_numpy**

These now return Result. Propagate errors.

- [ ] **Step 4: Build and test**

```bash
uv run maturin develop
uv run pytest -v
```

- [ ] **Step 5: Commit**

```bash
git add crates/python/src/lib.rs
git commit -m "refactor: Python bindings for N-D tensor support"
```

### Task 10: N-D Python tests

**Files:**
- Create: `python/tests/test_nd_tensors.py`

- [ ] **Step 1: Write N-D Python tests**

```python
import numpy as np
import pytest
import applegpu_runtime as gpu

@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()

def test_tensor_3d():
    t = gpu.tensor([float(i) for i in range(24)], shape=[2, 3, 4])
    assert t.shape == [2, 3, 4]

def test_3d_from_numpy():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = gpu.from_numpy(arr)
    result = t.to_numpy()
    assert result.shape == (2, 3, 4)
    np.testing.assert_array_equal(result, arr)

def test_broadcast_add():
    x = gpu.from_numpy(np.ones((4, 3), dtype=np.float32))
    bias = gpu.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    result = (x + bias).to_numpy()
    assert result.shape == (4, 3)
    np.testing.assert_allclose(result[0], [11.0, 21.0, 31.0])

def test_broadcast_replaces_add_bias():
    x = gpu.from_numpy(np.ones((4, 3), dtype=np.float32))
    bias = gpu.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    result_broadcast = (x + bias).to_numpy()
    result_add_bias = gpu.add_bias(x, bias).to_numpy()
    np.testing.assert_array_equal(result_broadcast, result_add_bias)

def test_3d_relu():
    arr = np.array([[[-1, 2], [3, -4]], [[-5, 6], [7, -8]]], dtype=np.float32)
    t = gpu.from_numpy(arr)
    result = t.relu().to_numpy()
    expected = np.maximum(arr, 0)
    np.testing.assert_array_equal(result, expected)

def test_softmax_rejects_3d():
    t = gpu.tensor([1.0] * 8, shape=[2, 2, 2])
    with pytest.raises(ValueError, match="2D"):
        gpu.softmax(t)

def test_matmul_rejects_3d():
    t = gpu.tensor([1.0] * 8, shape=[2, 2, 2])
    with pytest.raises(ValueError, match="2D"):
        t @ t

def test_reshape_to_3d():
    t = gpu.tensor([float(i) for i in range(12)], shape=[12])
    t2 = gpu.reshape(t, [2, 3, 2])
    assert t2.shape == [2, 3, 2]

def test_existing_gpt2_unchanged():
    """Verify 2D ops still work correctly."""
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = a @ b
    result = c.to_list()
    assert abs(result[0] - 19.0) < 0.1
```

- [ ] **Step 2: Run full test suite**

```bash
cargo test -p applegpu-core
uv run pytest -v
```

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_nd_tensors.py
git commit -m "test: N-D tensor Python tests (3D ops, broadcasting, ndim validation)"
```

### Task 11: Update GPT-2 tests

Verify GPT-2 inference still works with the N-D tensor changes.

- [ ] **Step 1: Run GPT-2 tests**

```bash
uv run pytest python/tests/test_gpt2.py -v
```

If any fail, fix the issues.

- [ ] **Step 2: Commit any fixes**

### Task 12: Update README, backlog, project status

- [ ] **Step 1: Update README** with N-D capabilities, test count
- [ ] **Step 2: Update backlog** — mark Phase 1 N-D as done
- [ ] **Step 3: Update project status memory**
- [ ] **Step 4: Commit and push**

```bash
git add README.md docs/BACKLOG.md
git commit -m "docs: update README and backlog for N-D tensor support"
git push origin main
```
