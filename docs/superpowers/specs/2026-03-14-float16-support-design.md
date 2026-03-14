# Float16 Support

**Date:** 2026-03-14
**Status:** Approved
**Scope:** f16 variants of all 14 ops, f16 tensor creation/read, NumPy/PyTorch f16 interop. f32 accumulation for matmul/softmax/scalar_mul.

## Overview

Add native float16 support to applegpu_runtime. Apple Silicon has native `half` ALUs at 2x float32 throughput. All 14 existing GPU operations get f16 variants via MSL preprocessor-based kernel templates. Matmul, softmax, and scalar_mul use f32 intermediate computation to preserve numerical stability.

## MSL Kernel Strategy

### Preprocessor-Based Templates

Instead of duplicating kernel source strings, use a single template with `#define DTYPE half` or `#define DTYPE float` prepended:

```metal
// Template (shared for f16 and f32):
kernel void elementwise_add_SUFFIX(
    device const DTYPE* a [[buffer(0)]],
    device const DTYPE* b [[buffer(1)]],
    device DTYPE* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) { out[id] = a[id] + b[id]; }
}
```

The Rust side prepends `#define DTYPE half\n#define SUFFIX f16\n` or `#define DTYPE float\n#define SUFFIX f32\n` before compilation. Kernel names are dtype-suffixed for cache keying.

**Applies to:** add, sub, mul, div, neg, relu, exp, log, sqrt, transpose (10 ops)

**Note:** relu uses `max(input[id], (DTYPE)0)` to handle both float and half literal correctly.

### Custom f32-Intermediate Kernels

Three ops need separate kernel bodies because they require f32 computation internally:

**matmul_f16:**
```metal
kernel void matmul_f16(...) {
    float acc = 0.0f;
    for (uint i = 0; i < K; i++)
        acc += float(A[row * K + i]) * float(B[i * N + col]);
    C[row * N + col] = half(acc);
}
```

**softmax_f16:** Read half, compute max/exp/sum in float, write half.

**scalar_mul_f16:** Scalar param stays f32 (`constant float& scale`), read half, multiply in float, write half.

### Fusion Engine

`generate_fused_msl` in `fusion.rs` gains a `dtype: DType` parameter. Emits `device const half*` for f16 chains. The `FusionChain.out_dtype` already carries the correct type. MSL expressions like relu use `(DTYPE)0` cast or dtype-aware literals.

## Tensor Changes

### from_raw gains dtype parameter

```rust
pub fn from_raw(id: u64, shape: Vec<usize>, dtype: DType, buffer: Buffer) -> Self
```

All call sites in `lazy.rs` (`execute_node`, `eval_remote`) updated to pass the correct dtype.

### New constructors

```rust
pub fn empty_f16(device: &Device, shape: Vec<usize>) -> Result<Self>
pub fn from_f16(device: &Device, shape: Vec<usize>, data: &[u16]) -> Result<Self>
```

`from_f16` takes `&[u16]` (raw f16 bit patterns). Internally creates buffer from bytes.

### Read methods

```rust
pub fn as_f16_slice(&self) -> &[u16]  // errors if dtype != Float16
pub fn as_f32_slice(&self) -> &[f32]  // errors if dtype != Float32 (existing, add check)
```

`LazyRuntime` gets `read_f16(id) -> Result<Vec<u16>>` parallel to `read_f32`.

## Ops Changes

### Dtype inference

`ops.rs` functions infer output dtype from inputs:

```rust
fn lazy_binary_op(rt: &mut LazyRuntime, a_id: u64, b_id: u64, op: OpKind) -> Result<u64> {
    let a_dtype = rt.dtype(a_id)?;   // new method on LazyRuntime
    let b_dtype = rt.dtype(b_id)?;
    if a_dtype != b_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "Dtype mismatch: {:?} vs {:?}", a_dtype, b_dtype
        )));
    }
    // ... out_dtype = a_dtype
}
```

`LazyRuntime::dtype(id)` looks up dtype from materialized tensor or pending graph node.

### Dispatch

`KernelRegistry` dispatch methods gain a `dtype: DType` parameter to select the correct kernel name. `execute_node` passes `node.out_dtype` to dispatch.

## Python API

### tensor() with dtype

```python
t = gpu.tensor([1.0, 2.0], shape=[2], dtype="float16")  # converts Python floats to f16
t = gpu.tensor([1.0, 2.0], shape=[2])                    # default: float32
```

Conversion from Python `float` (f64) to f16 uses the `half` Rust crate. Precision loss is inherent and documented.

### from_numpy auto-detects dtype

```python
arr = np.array([1.0, 2.0], dtype=np.float16)
t = gpu.from_numpy(arr)  # creates f16 tensor
t.dtype                   # "float16"
```

### to_numpy dispatches on dtype

- f32 → `PyArrayDyn<f32>` (existing path)
- f16 → create `PyArray1<u16>` from raw data, then `arr.view(np.float16).reshape(shape)` on Python side

### to_torch dispatches on dtype

Routes through `to_numpy` which already handles f16. `torch.from_numpy().clone()` preserves dtype.

### to_list dispatches on dtype

- f32 → list of Python floats (existing)
- f16 → read u16 bit patterns, convert to f32 via `half` crate, return as list of Python floats

### dtype getter

```python
t.dtype  # "float32" or "float16"
```

### Mixed-dtype error

```python
a = gpu.tensor([1.0], shape=[1], dtype="float16")
b = gpu.tensor([1.0], shape=[1])  # float32
c = a + b  # ValueError: Dtype mismatch: Float16 vs Float32
```

No implicit casting. User must cast explicitly.

## Dependencies

- `half = "2"` Rust crate for f16 ↔ f32 conversion (in `applegpu-core` and/or `applegpu-python`)

## eval_remote Changes

- Line computing logical size: `shape.iter().product::<usize>() * dtype.size_bytes()` (not `* 4`)
- `as_f32_slice()` call in tensor serialization: dispatch on dtype
- `Tensor::from_raw` call: pass dtype from response metadata

Note: The IPC serialization format (`serial.rs`) does not currently send dtype in the response. For v1 of f16, assume all remote eval is f32 (the VM backend is a separate concern). Add dtype to the wire format as a follow-up.

## Testing Strategy (TDD)

### Rust Unit Tests

**tensor.rs (~5 tests):**
- `test_empty_f16_creates_correct_size` — shape [4] → 8 bytes
- `test_from_f16_roundtrip` — write u16 data, read back via as_f16_slice
- `test_as_f32_slice_errors_on_f16` — f16 tensor, as_f32_slice returns error
- `test_as_f16_slice_errors_on_f32` — f32 tensor, as_f16_slice returns error
- `test_from_raw_respects_dtype` — from_raw with Float16 sets correct meta

**compute.rs (~4 tests):**
- `test_dispatch_binary_f16` — add two f16 buffers
- `test_dispatch_unary_f16` — relu on f16 buffer
- `test_dispatch_matmul_f16` — matmul with f32 accumulation
- `test_dispatch_softmax_f16` — softmax with f32 intermediates

**ops.rs (~3 tests):**
- `test_f16_add_eval` — create f16 tensors, add, eval, read
- `test_f16_matmul_eval` — f16 matmul correctness
- `test_mixed_dtype_errors` — f32 + f16 → error

**fusion.rs (~2 tests):**
- `test_fusion_f16_chain` — f16 neg+relu fused correctly
- `test_fusion_preserves_f16_dtype` — fused output has Float16 dtype

### Python Tests (test_float16.py, ~8 tests)

1. `test_tensor_f16_creation` — dtype="float16" creates f16 tensor
2. `test_f16_to_list` — f16 tensor to_list returns Python floats
3. `test_f16_from_numpy_roundtrip` — np.float16 array → from_numpy → to_numpy
4. `test_f16_from_torch_roundtrip` — torch.float16 tensor → from_torch → to_torch
5. `test_f16_ops` — add, mul on f16 tensors
6. `test_f16_matmul` — matmul on f16 tensors (check numerical accuracy)
7. `test_mixed_dtype_error` — f32 + f16 raises ValueError
8. `test_dtype_getter` — tensor.dtype returns correct string

## Backlog (not in scope)

- Float64, integer types, BFloat16
- Mixed-dtype operations / implicit promotion
- f16 ↔ f32 cast ops
- f16 in IPC wire format (VM backend)
