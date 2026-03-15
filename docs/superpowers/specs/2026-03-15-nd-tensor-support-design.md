# N-Dimensional Tensor Support — Phase 1

**Date:** 2026-03-15
**Status:** Approved
**Scope:** N-D Shape/TensorLayout, stride-based MSL kernels, NumPy-style broadcasting for element-wise ops. Non-element-wise ops remain 2D.

## Overview

Replace the 2D-only tensor architecture with N-dimensional support (MAX_DIMS=8). All MSL kernels use unified stride-based indexing. Element-wise ops gain NumPy-style broadcasting. Non-element-wise ops (softmax, matmul, etc.) validate ndim==2 with clear errors. This is the foundation for Phase 2 (batched transformer ops) and Phase 3 (batch inference).

## Architecture

### Shape (dims only, stack-allocated)

```rust
pub const MAX_DIMS: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct Shape {
    dims: [usize; MAX_DIMS],  // unused dims = 1
    ndim: usize,
}
```

- `Shape::new(dims: Vec<usize>) -> Result<Shape>` — returns error if dims.len() > MAX_DIMS
- `Shape::scalar() -> Shape` — ndim=0, numel()=1
- `shape.ndim()`, `shape.dims() -> &[usize]` (returns `dims[0..ndim]`)
- `shape.numel()` — product of dims[0..ndim] (empty product = 1 for scalars)
- `shape.broadcast_with(other) -> Result<Shape>` — NumPy broadcasting rules
- Copy semantics (72 bytes, fits in cache line)
- Manual Hash/PartialEq/Eq — only considers dims[0..ndim]
- Used in: OpNode.out_shape, ops.rs validation, fusion, serialization

### TensorLayout (dims + element strides)

```rust
#[derive(Debug, Clone)]  // Clone, NOT Copy (136 bytes)
pub struct TensorLayout {
    pub shape: Shape,
    strides: [usize; MAX_DIMS],  // element strides (not byte strides)
}
```

- `TensorLayout::contiguous(shape: Shape) -> TensorLayout` — row-major strides
- `layout.is_contiguous() -> bool` — strides match row-major
- `layout.transpose(dim0, dim1) -> TensorLayout` — swap dims and strides (zero-copy view metadata)
- `layout.strides() -> &[usize]` — returns strides[0..ndim]
- `layout.element_stride(dim) -> usize`
- `TensorLayout::broadcast_strides_for(source_shape: &Shape, broadcast_shape: &Shape) -> [usize; MAX_DIMS]` — computes element strides with stride=0 for broadcast dims
- Manual Hash/PartialEq/Eq — considers dims[0..ndim] + strides[0..ndim]
- Lives on TensorMeta only

### TensorMeta (updated)

```rust
pub struct TensorMeta {
    pub id: u64,
    pub layout: TensorLayout,  // was: shape: Shape
    pub dtype: DType,
    pub location: TensorLocation,
}

impl TensorMeta {
    /// Logical size in bytes (assumes contiguous). Used for buffer allocation.
    pub fn size_bytes(&self) -> usize {
        self.layout.shape.numel() * self.dtype.size_bytes()
    }
}
```

## MSL Kernel Rewrite

### Shared index helper

Included in every MSL kernel source string:

```metal
uint nd_index_to_offset(uint flat_id, constant uint* shape, constant uint* strides, constant uint& ndim) {
    uint offset = 0;
    for (uint d = ndim; d > 0; d--) {
        uint i = d - 1;
        offset += (flat_id % shape[i]) * strides[i];
        flat_id /= shape[i];
    }
    return offset;
}
```

Returns element offset (not byte offset). Correct for:
- Contiguous tensors: strides are row-major, gives same result as flat indexing
- Broadcast dims: stride=0, reads same element for all indices
- Transposed views: swapped strides, reads transposed layout
- Scalars (ndim=0): loop doesn't execute, returns 0

### Unified kernel signature (element-wise binary example)

```metal
kernel void elementwise_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* a_strides [[buffer(3)]],
    constant uint* b_strides [[buffer(4)]],
    constant uint* out_shape [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& numel [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    uint a_off = nd_index_to_offset(id, out_shape, a_strides, ndim);
    uint b_off = nd_index_to_offset(id, out_shape, b_strides, ndim);
    out[id] = a[a_off] + b[b_off];
}
```

- Output always contiguous (indexed by flat `id`)
- Inputs can be strided, broadcast, or transposed
- No contiguous fast path — single code path for all layouts
- f16 variants follow same pattern with `half` types

### Unary kernel pattern

Same but with one input:
```metal
kernel void elementwise_neg_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* in_strides [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    uint in_off = nd_index_to_offset(id, out_shape, in_strides, ndim);
    out[id] = -input[in_off];
}
```

### Non-element-wise ops unchanged (2D only)

Matmul, softmax, softmax_causal, layer_norm, transpose, embedding, slice, concat, add_bias, argmax — these keep their current 2D kernel signatures. They validate ndim==2 in ops.rs and return `GpuError::InvalidTensor("op requires 2D tensor, got {ndim}D")`.

These ops get N-D support in Phase 2 (batched transformer ops).

## Broadcasting

### Rules (NumPy-compatible)

Shapes aligned from the right. Each dimension pair must be:
- Equal, or
- One is 1 (broadcast that input along this dim)

Missing leading dims are treated as 1.

### Examples

```
[4, 3] + [3]       → [4, 3]      # bias addition (replaces add_bias)
[2, 1, 4] + [3, 4] → [2, 3, 4]   # 3D broadcast
[5] + [5]           → [5]         # exact match
[2, 3] + [4, 3]    → ERROR        # dim 0: 2 vs 4, neither is 1
```

### Implementation

In `ops.rs`, `lazy_binary_op`:
1. Get shapes of both inputs
2. Call `shape_a.broadcast_with(&shape_b)?` → output shape
3. For each input, compute broadcast strides via `TensorLayout::broadcast_strides_for(input_shape, output_shape)`
4. Record OpNode with output shape
5. At eval, dispatch kernel with each input's broadcast strides

### add_bias becomes a thin wrapper

```rust
pub fn add_bias(rt, input_id, bias_id) -> Result<u64> {
    // Broadcasting handles [rows, cols] + [cols] naturally
    lazy_binary_op(rt, input_id, bias_id, OpKind::Add)
}
```

## Dispatch Changes

### Rust dispatch

Every dispatch method gains stride/shape/ndim parameters:

```rust
pub fn dispatch_binary_nd(
    &self, device: &Device,
    kernel_name: &str,
    a: &Buffer, a_strides: &[u32; MAX_DIMS],
    b: &Buffer, b_strides: &[u32; MAX_DIMS],
    out: &Buffer, out_shape: &[u32; MAX_DIMS],
    ndim: u32, numel: u32,
    dtype: DType,
) -> Result<()>
```

Element strides converted to u32 at dispatch time (MSL uses uint).

### Swift FFI

New dispatch function signatures with stride/shape/ndim parameters. Each op type (binary, unary, matmul, etc.) gets a new `_nd` variant or replaces the existing one.

### Non-blocking variants

Same pattern as existing `_nb` methods — return command buffer handle. Gain stride/shape/ndim parameters.

## Read Methods — Return Result

```rust
impl Tensor {
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        if !self.meta.layout.is_contiguous() {
            return Err(GpuError::InvalidTensor("as_f32_slice requires contiguous tensor".into()));
        }
        // ... existing logic ...
    }

    pub fn as_f16_slice(&self) -> Result<&[u16]> { /* same pattern */ }
    pub fn as_bytes(&self) -> Result<&[u8]> { /* same pattern */ }
}
```

All callers updated to handle Result (read_f32, read_f16, read_bytes, to_list, to_numpy, etc.).

## Tensor Constructors

All unchanged externally. Internally create `TensorLayout::contiguous(shape)`:

```rust
pub fn from_data(device, shape: Vec<usize>, dtype, data: &[u8]) -> Result<Self> {
    let shape = Shape::new(shape)?;
    let layout = TensorLayout::contiguous(shape);
    // ... rest unchanged ...
}
```

`from_raw` creates contiguous layout from the shape parameter.

## Fusion

Fused kernels require contiguous inputs (v1). The fusion engine's `generate_fused_msl` updates to use stride-based indexing for the output shape, but since all fused inputs are contiguous with matching shapes, strides are always row-major. The MSL uses the `nd_index_to_offset` helper.

Fused kernel signature gains the same stride/shape/ndim parameters.

## Serialization

Wire format unchanged — only `dims[0..ndim]` are serialized (same as current). `VERSION` stays at 1. Strides are not serialized (reconstructed as contiguous on deserialization).

## Backward Compatibility

- Python `tensor.shape` returns `list[int]` — unchanged
- All existing 2D code works (ndim=2 is valid N-D)
- `add_bias` stays as convenience (internally uses broadcast add)
- `from_numpy`/`to_numpy` handle N-D naturally
- Existing tests updated for `Shape::new` returning Result and `as_f32_slice` returning Result

## Testing Strategy (TDD)

### Rust Unit Tests (~20 tests)

**Shape:**
1. `test_shape_new_2d` — Shape::new(vec![2,3]) → ndim=2, numel=6
2. `test_shape_new_3d` — Shape::new(vec![2,3,4]) → ndim=3, numel=24
3. `test_shape_scalar` — Shape::scalar() → ndim=0, numel=1
4. `test_shape_exceeds_max_dims` — 9 dims → error
5. `test_shape_broadcast_basic` — [4,3] + [3] → [4,3]
6. `test_shape_broadcast_3d` — [2,1,4] + [3,4] → [2,3,4]
7. `test_shape_broadcast_incompatible` — [2,3] + [4,3] → error
8. `test_shape_eq_ignores_padding` — two shapes with same dims but different padding are equal

**TensorLayout:**
9. `test_layout_contiguous_strides` — [2,3,4] → strides [12,4,1]
10. `test_layout_is_contiguous` — contiguous layout returns true
11. `test_layout_transpose` — transpose(0,1) swaps dims and strides, is_contiguous returns false
12. `test_layout_broadcast_strides` — broadcast [3] to [4,3] → strides [0,1]

**N-D element-wise ops:**
13. `test_add_3d` — [2,3,4] + [2,3,4] → correct result
14. `test_add_broadcast` — [4,3] + [3] → broadcast addition
15. `test_gelu_3d` — gelu on 3D tensor
16. `test_relu_broadcast` — [2,1,4] + [3,4] broadcast

**Validation:**
17. `test_softmax_rejects_3d` — 3D input → clear error
18. `test_matmul_rejects_3d` — 3D input → clear error
19. `test_as_f32_slice_rejects_non_contiguous` — transposed tensor → error

**Backward compat:**
20. `test_existing_2d_ops_unchanged` — 2D add/matmul/attention chain works

### Python Tests (~8 tests)

1. `test_tensor_3d_creation` — gpu.tensor(..., shape=[2,3,4])
2. `test_3d_from_numpy_roundtrip` — 3D numpy array roundtrip
3. `test_broadcast_add` — [4,3] + [3] via standard + operator
4. `test_broadcast_replaces_add_bias` — same result as explicit add_bias
5. `test_3d_element_wise` — relu on 3D tensor
6. `test_softmax_rejects_3d_python` — clear ValueError
7. `test_reshape_to_3d` — reshape [6] → [2,3]
8. `test_existing_gpt2_still_works` — GPT-2 forward pass unchanged

## Implementation Order (keeps tests green throughout)

1. Shape struct (new, with tests)
2. TensorLayout struct (new, with tests)
3. TensorMeta migration (shape → layout)
4. Update all tensor constructors
5. Update read methods to return Result
6. Update ops.rs (broadcasting, ndim validation)
7. MSL kernel rewrites + new dispatch signatures
8. Swift FFI changes
9. Fusion MSL updates
10. Serialization (minimal — wire format unchanged)
11. Python layer updates
12. Update existing tests for new APIs

## Backlog (Phase 2+)

- N-D matmul (batched matmul)
- N-D softmax, layer_norm, attention (batched variants)
- Non-contiguous output tensors
- Contiguous fast path kernels (if profiling shows need)
- N-D slice, concat, transpose (generalized beyond 2D)
