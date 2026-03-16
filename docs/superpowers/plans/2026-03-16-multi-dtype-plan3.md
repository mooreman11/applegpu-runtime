# Multi-Dtype Compute Kernels (Plan 3 of 3) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate existing ops for all new dtypes, add op-level dtype validation, gate Int64 on Apple9+, add comparison/bitwise/utility ops, and test everything end-to-end.

**Architecture:** Three chunks: (1) validation layer + device gating + transpose byte-copy fix, (2) validate BFloat16/Int32/Int64 existing ops + add comparison ops, (3) bitwise/utility/quantize ops. Each chunk is independently shippable and testable.

**Tech Stack:** Rust (applegpu-core), Swift (AppleGPUBridge for device family query), MSL (Metal Shading Language), PyO3 (Python API)

**Spec:** `docs/superpowers/specs/2026-03-16-multi-dtype-compute-kernels-design.md`

---

## Chunk 1: Validation Layer + Device Gating + Fixups

### Task 1: Op-level dtype validation function

**Files:**
- Modify: `crates/core/src/ops.rs`
- Test: `crates/core/src/ops.rs` (inline `#[cfg(test)]`)

- [ ] **Step 1: Write failing tests for op-dtype validation**

```rust
#[test]
fn exp_rejects_int32() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();
    let t = Tensor::from_data(&device, vec![3], &[1i32, 2, 3]).unwrap();
    rt.insert_tensor(t);
    let id = 1; // the tensor ID
    let result = crate::ops::exp(&mut rt, id);
    assert!(result.is_err());
}

#[test]
fn add_accepts_int32() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();
    let a = Tensor::from_data(&device, vec![2], &[1i32, 2]).unwrap();
    let b = Tensor::from_data(&device, vec![2], &[3i32, 4]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a);
    rt.insert_tensor(b);
    let result = crate::ops::add(&mut rt, a_id, b_id);
    assert!(result.is_ok());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p applegpu-core exp_rejects_int32`
Expected: FAIL — exp currently accepts any non-Float64 dtype

- [ ] **Step 3: Implement validate_op_dtype**

Add to `ops.rs` below `validate_compute_dtype`:

```rust
/// Validate that a specific op is semantically valid for the given dtype.
/// Encodes the coverage matrix from the multi-dtype spec.
fn validate_op_dtype(op: &OpKind, dtype: DType) -> Result<()> {
    validate_compute_dtype(dtype)?;

    let valid = match op {
        // Float-only ops
        OpKind::Exp | OpKind::Log | OpKind::Sqrt | OpKind::Tanh
        | OpKind::Relu | OpKind::Gelu
        | OpKind::Softmax | OpKind::SoftmaxCausal
        | OpKind::Matmul | OpKind::LayerNorm { .. } | OpKind::BatchNorm { .. }
        | OpKind::Conv1d { .. } | OpKind::Conv2d { .. }
        | OpKind::MaxPool2d { .. } | OpKind::AvgPool2d { .. }
        | OpKind::AddBias | OpKind::Embedding
        | OpKind::SoftmaxBackward | OpKind::LayerNormBackward { .. }
        | OpKind::Conv2dBackwardInput { .. } | OpKind::EmbeddingBackward
        | OpKind::BatchNormBackward { .. } => dtype.is_float(),

        // Numeric ops (float + integer, not bool, not int8/int16)
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div
        | OpKind::Neg | OpKind::Abs | OpKind::Sign
        | OpKind::Clamp { .. } | OpKind::ScalarMul(_)
        | OpKind::Pow { .. } => {
            dtype.is_float() || matches!(dtype, DType::Int32 | DType::Int64 | DType::UInt8 | DType::UInt32)
        }

        // All-dtype ops
        OpKind::Cast { .. } | OpKind::Reshape { .. }
        | OpKind::Transpose { .. } | OpKind::Slice { .. } | OpKind::Concat { .. }
        | OpKind::Where | OpKind::MaskedFill { .. }
        | OpKind::Triu { .. } | OpKind::Tril { .. } => true,

        // Gather/index_select: all except Bool
        OpKind::Gather { .. } | OpKind::IndexSelect { .. } => !matches!(dtype, DType::Bool),

        // Reduction ops: float + integer (sum/argmax for all, mean float-only)
        OpKind::Sum | OpKind::Argmax => !matches!(dtype, DType::Bool),
        OpKind::Mean => dtype.is_float(),

        // Fused: validated at fusion time
        OpKind::FusedElementwise { .. } => true,
    };

    if !valid {
        return Err(GpuError::UnsupportedDtype(format!(
            "{} is not supported for {:?}", dtype.name(), op.kernel_name()
        )));
    }
    Ok(())
}
```

- [ ] **Step 4: Wire validate_op_dtype into op-recording functions**

Replace `validate_compute_dtype(dtype)?;` calls in `lazy_binary_op`, `lazy_unary_op`, and each standalone op function with `validate_op_dtype(&op, dtype)?;` (or the equivalent for the specific OpKind).

For `lazy_binary_op`, add after the dtype mismatch check:
```rust
validate_op_dtype(&op, a_dtype)?;
```

For unary ops (exp, log, sqrt, etc.), change the body of each function to call:
```rust
validate_op_dtype(&OpKind::Exp, dtype)?;  // (for exp)
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p applegpu-core exp_rejects_int32 add_accepts_int32`
Expected: Both PASS

- [ ] **Step 6: Add more validation tests**

```rust
#[test]
fn softmax_rejects_bool() {
    // ...setup Bool tensor...
    let result = crate::ops::softmax(&mut rt, id);
    assert!(result.is_err());
}

#[test]
fn where_accepts_all_dtypes() {
    // where should accept Int32, Bool, etc.
}
```

- [ ] **Step 7: Run all tests and commit**

Run: `cargo test -p applegpu-core`
```bash
git add crates/core/src/ops.rs
git commit -m "feat: add op-level dtype validation (validate_op_dtype)"
```

### Task 2: Device GPU family caching + Int64 gating

**Files:**
- Modify: `swift/Sources/AppleGPUBridge/bridge.swift`
- Modify: `swift/Sources/AppleGPUBridge/include/bridge.h`
- Modify: `crates/core/src/ffi.rs`
- Modify: `crates/core/src/device.rs`
- Modify: `crates/core/src/ops.rs` (validate_op_dtype)

- [ ] **Step 1: Add Swift FFI for GPU family query**

In `bridge.h`:
```c
bool gpu_bridge_supports_apple9(const GPUDeviceHandle* device);
```

In `bridge.swift`:
```swift
@_cdecl("gpu_bridge_supports_apple9")
public func gpuBridgeSupportsApple9(_ devicePtr: UnsafeRawPointer?) -> Bool {
    guard let devicePtr = devicePtr else { return false }
    let gpuDevice = getGPUDevice(from: devicePtr)
    return gpuDevice.device.supportsFamily(.apple9)
}
```

- [ ] **Step 2: Add Rust FFI declaration and Device method**

In `ffi.rs`:
```rust
extern "C" {
    pub fn gpu_bridge_supports_apple9(device: *const GPUDeviceHandle) -> bool;
}
```

In `device.rs`, add to `Device`:
```rust
pub fn supports_int64(&self) -> bool {
    unsafe { ffi::gpu_bridge_supports_apple9(self.handle) }
}
```

- [ ] **Step 3: Gate Int64 in validate_op_dtype**

This requires passing device info to the validation. Add a new function:

```rust
pub fn validate_int64_support(dtype: DType, device: &Device) -> Result<()> {
    if matches!(dtype, DType::Int64) && !device.supports_int64() {
        return Err(GpuError::UnsupportedDtype(
            "Int64 requires Apple9+ GPU (M3/M4). This device does not support it.".to_string()
        ));
    }
    Ok(())
}
```

Call this from `LazyRuntime::eval()` before executing each node, checking `node.out_dtype`.

- [ ] **Step 4: Build Swift + Rust**

Run: `cd swift && swift build && cd .. && cargo build -p applegpu-core`
Expected: BUILD SUCCEEDED

- [ ] **Step 5: Commit**

```bash
git add swift/Sources/AppleGPUBridge/bridge.swift swift/Sources/AppleGPUBridge/include/bridge.h crates/core/src/ffi.rs crates/core/src/device.rs crates/core/src/ops.rs
git commit -m "feat: device GPU family caching + Int64 Apple9+ gating"
```

### Task 3: Wire transpose to byte-copy

**Files:**
- Modify: `crates/core/src/compute.rs`

- [ ] **Step 1: Update resolve_kernel for transpose**

In `resolve_kernel`, change the "transpose" arm to use byte-copy:

```rust
"transpose" => return (kt::byte_copy_transpose_source(dtype.size_bytes()), format!("transpose_bytes{}", dtype.size_bytes())),
```

Keep "transpose_batched" and "copy_strided" as typed templates (they have more complex semantics).

- [ ] **Step 2: Run tests**

Run: `cargo test -p applegpu-core`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/compute.rs
git commit -m "fix: wire transpose to byte-copy kernel for dtype-agnostic transpose"
```

---

## Chunk 2: Dtype Validation + Comparison Ops

### Task 4: BFloat16 validation tests

**Files:**
- Test: `python/tests/test_multi_dtype.py`

- [ ] **Step 1: Add BFloat16 Python tests**

```python
def test_bf16_add():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4], dtype="bfloat16")
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4], dtype="bfloat16")
    c = a + b
    c.eval()
    vals = c.to_list()
    assert vals == [11.0, 22.0, 33.0, 44.0]

def test_bf16_matmul():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2], dtype="bfloat16")
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2], dtype="bfloat16")
    c = gpu.matmul(a, b)
    c.eval()
    assert c.to_list() == [19.0, 22.0, 43.0, 50.0]

def test_bf16_softmax():
    a = gpu.tensor([1.0, 2.0, 3.0], shape=[1, 3], dtype="bfloat16")
    b = gpu.softmax(a)
    b.eval()
    assert abs(sum(b.to_list()) - 1.0) < 0.01

def test_bf16_gelu():
    a = gpu.tensor([0.0, 1.0, -1.0], shape=[3], dtype="bfloat16")
    b = gpu.gelu(a)
    b.eval()
    vals = b.to_list()
    assert abs(vals[0]) < 0.01  # gelu(0) ≈ 0

def test_bf16_layer_norm():
    inp = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], dtype="bfloat16")
    gamma = gpu.tensor([1.0, 1.0, 1.0], shape=[3], dtype="bfloat16")
    beta = gpu.tensor([0.0, 0.0, 0.0], shape=[3], dtype="bfloat16")
    out = gpu.layer_norm(inp, gamma, beta, 1e-5)
    out.eval()
    vals = out.to_list()
    assert len(vals) == 6
```

- [ ] **Step 2: Run tests**

Run: `uv run maturin develop && uv run pytest python/tests/test_multi_dtype.py -k bf16 -v`
Expected: PASS

- [ ] **Step 3: Commit**

### Task 5: Int32 arithmetic validation tests

**Files:**
- Test: `python/tests/test_multi_dtype.py`

- [ ] **Step 1: Add Int32 Python tests**

```python
def test_i32_sub():
    a = gpu.tensor([10, 20, 30], shape=[3], dtype="int32")
    b = gpu.tensor([1, 2, 3], shape=[3], dtype="int32")
    c = a - b
    c.eval()
    assert c.to_list() == [9, 18, 27]

def test_i32_div():
    a = gpu.tensor([10, 21, 30], shape=[3], dtype="int32")
    b = gpu.tensor([3, 7, 5], shape=[3], dtype="int32")
    c = a / b
    c.eval()
    assert c.to_list() == [3, 3, 6]  # integer division

def test_i32_scalar_mul():
    a = gpu.tensor([1, 2, 3], shape=[3], dtype="int32")
    b = gpu.scalar_mul(a, 5.0)
    b.eval()
    assert b.to_list() == [5, 10, 15]

def test_i32_clamp():
    a = gpu.tensor([1, 5, 10, -3], shape=[4], dtype="int32")
    b = gpu.clamp(a, 0.0, 7.0)
    b.eval()
    assert b.to_list() == [1, 5, 7, 0]

def test_i32_triu():
    a = gpu.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=[3, 3], dtype="int32")
    b = gpu.triu(a, 0)
    b.eval()
    assert b.to_list() == [1, 2, 3, 0, 5, 6, 0, 0, 9]

def test_i32_where():
    cond = gpu.tensor([1, 0, 1, 0], shape=[4], dtype="int32")
    x = gpu.tensor([10, 20, 30, 40], shape=[4], dtype="int32")
    y = gpu.tensor([100, 200, 300, 400], shape=[4], dtype="int32")
    result = gpu.where_cond(cond, x, y)
    result.eval()
    assert result.to_list() == [10, 200, 30, 400]

def test_i32_transpose():
    a = gpu.tensor([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype="int32")
    b = gpu.transpose(a)
    b.eval()
    assert b.to_list() == [1, 4, 2, 5, 3, 6]
    assert b.shape == [3, 2]

def test_i32_cast_to_f32():
    a = gpu.tensor([1, 2, 3], shape=[3], dtype="int32")
    b = gpu.cast(a, "float32")
    b.eval()
    assert b.to_list() == [1.0, 2.0, 3.0]
```

- [ ] **Step 2: Add rejection tests**

```python
def test_i32_exp_rejected():
    a = gpu.tensor([1, 2, 3], shape=[3], dtype="int32")
    with pytest.raises(Exception):
        gpu.exp(a)

def test_i32_softmax_rejected():
    a = gpu.tensor([1, 2, 3], shape=[1, 3], dtype="int32")
    with pytest.raises(Exception):
        gpu.softmax(a)

def test_i32_matmul_rejected():
    a = gpu.tensor([1, 2, 3, 4], shape=[2, 2], dtype="int32")
    b = gpu.tensor([5, 6, 7, 8], shape=[2, 2], dtype="int32")
    with pytest.raises(Exception):
        gpu.matmul(a, b)
```

- [ ] **Step 3: Run and commit**

Run: `uv run pytest python/tests/test_multi_dtype.py -k i32 -v`

### Task 6: Comparison ops (Lt, Gt, Le, Ge, Eq, Ne)

**Files:**
- Modify: `crates/core/src/graph.rs` (add OpKind variants)
- Modify: `crates/core/src/kernel_templates.rs` (comparison kernel template)
- Modify: `crates/core/src/compute.rs` (resolve_kernel)
- Modify: `crates/core/src/ops.rs` (lazy_comparison_op + public functions)
- Modify: `crates/core/src/lazy.rs` (dispatch)
- Modify: `crates/core/src/serial.rs` (wire protocol ID)
- Modify: `crates/python/src/backend.rs` (Backend trait)
- Modify: `crates/python/src/metal_backend.rs`
- Modify: `crates/python/src/socket_backend.rs`
- Modify: `crates/python/src/lib.rs` (PyO3 functions)
- Modify: `python/applegpu_runtime/__init__.py`
- Test: `python/tests/test_comparison_ops.py` (new)

- [ ] **Step 1: Add comparison OpKind variants to graph.rs**

```rust
// Comparison ops (output is always Bool)
Lt, Gt, Le, Ge, Eq, Ne,
```

Add to `kernel_name()`:
```rust
OpKind::Lt => "lt",
OpKind::Gt => "gt",
OpKind::Le => "le",
OpKind::Ge => "ge",
OpKind::Eq => "eq",
OpKind::Ne => "ne",
```

- [ ] **Step 2: Add comparison kernel template**

In `kernel_templates.rs`:

```rust
/// Generate comparison kernel source. Output is always Bool (uchar).
pub fn comparison_kernel_source(dtype: DType) -> String {
    let t = metal_type(dtype);
    let s = dtype_suffix(dtype);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{nd}
kernel void lt{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] < b[b_off] ? 1 : 0;
}}
kernel void gt{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] > b[b_off] ? 1 : 0;
}}
kernel void le{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] <= b[b_off] ? 1 : 0;
}}
kernel void ge{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] >= b[b_off] ? 1 : 0;
}}
kernel void eq{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] == b[b_off] ? 1 : 0;
}}
kernel void ne{s}(device const {t}* a [[buffer(0)]], device const {t}* b [[buffer(1)]], device uchar* out [[buffer(2)]], constant uint* a_strides [[buffer(3)]], constant uint* b_strides [[buffer(4)]], constant uint* out_shape [[buffer(5)]], constant uint& ndim [[buffer(6)]], constant uint& numel [[buffer(7)]], uint id [[thread_position_in_grid]]) {{
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] != b[b_off] ? 1 : 0;
}}
"#,
        nd = ND_INDEX_HELPER, t = t, s = s,
    )
}
```

- [ ] **Step 3: Wire into resolve_kernel**

Add to `resolve_kernel` match:
```rust
"lt" | "gt" | "le" | "ge" | "eq" | "ne" => kt::comparison_kernel_source(dtype),
```

- [ ] **Step 4: Add lazy_comparison_op in ops.rs**

```rust
fn lazy_comparison_op(rt: &mut LazyRuntime, a_id: u64, b_id: u64, op: OpKind) -> Result<u64> {
    let a_dtype = rt.dtype(a_id)?;
    validate_compute_dtype(a_dtype)?;
    let b_dtype = rt.dtype(b_id)?;
    if a_dtype != b_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "Dtype mismatch: {:?} vs {:?}", a_dtype, b_dtype
        )));
    }
    let a_shape_obj = Shape::new(rt.shape(a_id)?)?;
    let b_shape_obj = Shape::new(rt.shape(b_id)?)?;
    let out_shape = a_shape_obj.broadcast_with(&b_shape_obj)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op,
        inputs: vec![a_id, b_id],
        out_shape,
        out_dtype: DType::Bool,  // comparison always outputs Bool
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

pub fn lt(rt: &mut LazyRuntime, a: u64, b: u64) -> Result<u64> { lazy_comparison_op(rt, a, b, OpKind::Lt) }
pub fn gt(rt: &mut LazyRuntime, a: u64, b: u64) -> Result<u64> { lazy_comparison_op(rt, a, b, OpKind::Gt) }
pub fn le(rt: &mut LazyRuntime, a: u64, b: u64) -> Result<u64> { lazy_comparison_op(rt, a, b, OpKind::Le) }
pub fn ge(rt: &mut LazyRuntime, a: u64, b: u64) -> Result<u64> { lazy_comparison_op(rt, a, b, OpKind::Ge) }
pub fn eq(rt: &mut LazyRuntime, a: u64, b: u64) -> Result<u64> { lazy_comparison_op(rt, a, b, OpKind::Eq) }
pub fn ne(rt: &mut LazyRuntime, a: u64, b: u64) -> Result<u64> { lazy_comparison_op(rt, a, b, OpKind::Ne) }
```

- [ ] **Step 5: Add dispatch in lazy.rs**

Comparison ops use binary N-D dispatch but with the INPUT dtype for kernel resolution (not output Bool):

```rust
if matches!(node.op, OpKind::Lt | OpKind::Gt | OpKind::Le | OpKind::Ge | OpKind::Eq | OpKind::Ne) {
    let (a_strides, b_strides, out_shape_u32, ndim, numel) = self.binary_nd_params(node)?;
    let out_buf = self.pool.acquire(device, out_size)?;
    let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
    let a = self.get_tensor(node.inputs[0])?;
    let b = self.get_tensor(node.inputs[1])?;
    let input_dtype = a.meta.dtype;  // use input dtype for kernel resolution
    REGISTRY.dispatch_binary_nd_typed(
        device, node.op.kernel_name(), input_dtype,
        &a.buffer, &a_strides, &b.buffer, &b_strides,
        &out.buffer, &out_shape_u32, ndim, numel,
    )?;
    return Ok(out);
}
```

Add same for `execute_node_nb`.

- [ ] **Step 6: Add comparison ops to validate_op_dtype**

```rust
// Comparison ops: all numeric types + Bool (eq/ne only for Bool)
OpKind::Lt | OpKind::Gt | OpKind::Le | OpKind::Ge => {
    !matches!(dtype, DType::Bool | DType::Int8 | DType::Int16)
}
OpKind::Eq | OpKind::Ne => !matches!(dtype, DType::Int8 | DType::Int16),
```

- [ ] **Step 7: Update serial.rs**

Add IDs for Lt(47), Gt(48), Le(49), Ge(50), Eq(51), Ne(52) and `unimplemented!` for wire conversion.

- [ ] **Step 8: Wire Python API**

Add to Backend trait, MetalBackend, SocketBackend, PyO3 lib.rs, __init__.py:
`lt`, `gt`, `le`, `ge`, `eq_`, `ne_` (trailing underscore to avoid Python keyword conflict for `eq`).

- [ ] **Step 9: Write Python tests**

Create `python/tests/test_comparison_ops.py`:
```python
def test_lt_f32():
    a = gpu.tensor([1.0, 5.0, 3.0], shape=[3])
    b = gpu.tensor([2.0, 4.0, 3.0], shape=[3])
    c = gpu.lt(a, b)
    c.eval()
    assert c.to_list() == [True, False, False]
    assert c.dtype == "bool"

def test_eq_i32():
    a = gpu.tensor([1, 2, 3], shape=[3], dtype="int32")
    b = gpu.tensor([1, 0, 3], shape=[3], dtype="int32")
    c = gpu.eq_(a, b)
    c.eval()
    assert c.to_list() == [True, False, True]

def test_gt_with_broadcast():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = gpu.tensor([2.0, 2.0, 2.0], shape=[3])
    c = gpu.gt(a, b)
    c.eval()
    assert c.to_list() == [False, False, True, True, True, True]
```

- [ ] **Step 10: Build, test, commit**

Run: `cargo test -p applegpu-core && uv run maturin develop && uv run pytest python/tests/test_comparison_ops.py -v`
```bash
git commit -m "feat: comparison ops (lt, gt, le, ge, eq, ne) with Bool output"
```

---

## Chunk 3: Bitwise, Utility, and Quantize Ops

### Task 7: Bitwise ops (and, or, xor, not, shl, shr)

**Files:** Same pattern as Task 6 (graph.rs, kernel_templates.rs, compute.rs, ops.rs, lazy.rs, serial.rs, Python layer)

- [ ] **Step 1: Add OpKind variants**

```rust
BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot,
Shl { shift: u32 }, Shr { shift: u32 },
```

- [ ] **Step 2: Add bitwise kernel template**

```rust
pub fn bitwise_kernel_source(dtype: DType) -> String { ... }
```

Generate kernels for `bitwise_and`, `bitwise_or`, `bitwise_xor` (binary), `bitwise_not` (unary), `shl`, `shr` (unary with shift param).

- [ ] **Step 3: Wire into resolve_kernel, ops.rs, lazy.rs, serial.rs, Python**
- [ ] **Step 4: Add validate_op_dtype rules** — integer and Bool only
- [ ] **Step 5: Write tests**
- [ ] **Step 6: Commit**

### Task 8: Modulo op

- [ ] **Step 1: Add OpKind::Mod**
- [ ] **Step 2: Template** — `out[id] = a[a_off] % b[b_off];` for integer types
- [ ] **Step 3: Wire everything + validate (integer only)**
- [ ] **Step 4: Test + commit**

### Task 9: Element-wise min/max

- [ ] **Step 1: Add OpKind::ElemMin, OpKind::ElemMax**
- [ ] **Step 2: Template** — `out[id] = min(a[a_off], b[b_off]);` / `max(...)`
- [ ] **Step 3: Wire + validate (float + integer, not bool)**
- [ ] **Step 4: Test + commit**

### Task 10: LogicalNot (Bool only)

- [ ] **Step 1: Add OpKind::LogicalNot**
- [ ] **Step 2: Template** — `out[id] = input[in_off] ? 0 : 1;` (Bool → Bool)
- [ ] **Step 3: Wire + validate (Bool only)**
- [ ] **Step 4: Test + commit**

### Task 11: Quantize/Dequantize

- [ ] **Step 1: Add OpKind::Quantize { scale, zero_point }, OpKind::Dequantize { scale, zero_point }**
- [ ] **Step 2: Templates**

Quantize (f32/f16 → int8/uint8):
```
out[id] = (int8_t)clamp(round(input[in_off] / scale) + zero_point, -128, 127);
```

Dequantize (int8/uint8 → f32/f16):
```
out[id] = (float)(input[in_off] - zero_point) * scale;
```

- [ ] **Step 3: Wire + validate (specific dtype pairs only)**
- [ ] **Step 4: Test + commit**

### Task 12: Final validation

- [ ] **Step 1: Run complete test suite**

Run: `make test`
Expected: All Rust + Swift + Python tests PASS

- [ ] **Step 2: Run GPT-2 benchmark (if Int64 indices fixed)**

Run: `uv run python examples/gpt2_generate.py --prompt "Hello world" --max-tokens 5`

- [ ] **Step 3: Commit milestone**

```bash
git commit --allow-empty -m "milestone: multi-dtype Plan 3 complete"
```
