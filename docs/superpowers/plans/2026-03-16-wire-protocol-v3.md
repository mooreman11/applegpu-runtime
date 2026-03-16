# Wire Protocol v3: Multi-Dtype + New Ops Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the wire protocol, SocketBackend, and gpu-service to support all Plan 2/3 ops (Cast, comparison, bitwise, modulo, elem min/max, logical_not, quantize/dequantize) and fix Int32 compute over containers.

**Architecture:** Three layers need updating: (1) `applegpu-wire` crate (WireOpKind enum + serialization), (2) `SocketBackend` in applegpu-python (replace stubs with real implementations), (3) `gpu-service` (dispatch new wire ops to applegpu-core). Also fix the memory quota default.

**Tech Stack:** Rust (applegpu-wire, applegpu-python, gpu-service)

**Spec:** `docs/superpowers/specs/2026-03-16-multi-dtype-compute-kernels-design.md`

---

## Chunk 1: Wire Protocol + SocketBackend

### Task 1: Fix SocketBackend default memory quota

**Files:**
- Modify: `crates/python/src/socket_backend.rs`

- [ ] **Step 1: Change default from 4096 to 1024 MB**

At line ~212, `.unwrap_or(4096)` → `.unwrap_or(1024)`.

- [ ] **Step 2: Commit**

```bash
git commit -m "fix: lower SocketBackend default memory quota to 1024 MB"
```

### Task 2: Add new WireOpKind variants

**Files:**
- Modify: `crates/wire/src/lib.rs`

- [ ] **Step 1: Add variants to WireOpKind enum**

```rust
// Type conversion
Cast { target_dtype: u8 },
// Comparison (output Bool)
Lt, Gt, Le, Ge, Eq, Ne,
// Bitwise
BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot,
Shl { shift: u32 }, Shr { shift: u32 },
// Utility
Mod, ElemMin, ElemMax, LogicalNot,
// Quantization
Quantize { scale: f32, zero_point: i32, target_dtype: u8 },
Dequantize { scale: f32, zero_point: i32, target_dtype: u8 },
```

- [ ] **Step 2: Add serialization (write_op/read_op)**

Follow existing pattern — each variant gets a discriminant byte (continue from 46). Cast=47, Lt=48, Gt=49, Le=50, Ge=51, Eq=52, Ne=53, BitwiseAnd=54, BitwiseOr=55, BitwiseXor=56, BitwiseNot=57, Shl=58, Shr=59, Mod=60, ElemMin=61, ElemMax=62, LogicalNot=63, Quantize=64, Dequantize=65.

For Cast: write target_dtype as u8. For Shl/Shr: write shift as u32. For Quantize/Dequantize: write scale (f32), zero_point (i32), target_dtype (u8).

- [ ] **Step 3: Test serialization roundtrip**
- [ ] **Step 4: Commit**

### Task 3: Update serial.rs conversions (OpKind ↔ WireOpKind)

**Files:**
- Modify: `crates/core/src/serial.rs`

- [ ] **Step 1: Replace `unimplemented!` stubs with real conversions**

In `From<&OpKind> for WireOpKind`:
```rust
OpKind::Cast { target_dtype } => WireOpKind::Cast { target_dtype: *target_dtype as u8 },
OpKind::Lt => WireOpKind::Lt,
OpKind::Gt => WireOpKind::Gt,
// ... etc for all new ops
```

In `From<&WireOpKind> for OpKind` (reverse direction):
```rust
WireOpKind::Cast { target_dtype } => OpKind::Cast { target_dtype: DType::from_wire(*target_dtype) },
WireOpKind::Lt => OpKind::Lt,
// ... etc
```

- [ ] **Step 2: Add DType ↔ u8 conversion helpers if not already present**
- [ ] **Step 3: Test + commit**

### Task 4: Wire SocketBackend ops (replace stubs)

**Files:**
- Modify: `crates/python/src/socket_backend.rs`

- [ ] **Step 1: Implement Cast in SocketBackend**

Replace the `Err("Cast not supported")` stub with a real implementation that records a Cast op in the client-side graph:

```rust
fn cast(&self, a: u64, target_dtype: &str) -> BackendResult<u64> {
    let dt = self.parse_dtype(target_dtype)?;
    let shape = self.shape(a)?;
    let out_id = self.next_id();
    // Record in client graph with new output dtype
    let mut graph = self.graph.lock().unwrap();
    graph.add_node(WireOpNode {
        id: out_id,
        op: WireOpKind::Cast { target_dtype: dt as u8 },
        inputs: vec![a],
        out_shape: shape,
        out_dtype: dt as u8,
    });
    // Track in tensors
    let mut tensors = self.tensors.lock().unwrap();
    tensors.insert(out_id, TensorMeta { shape, dtype: dt });
    Ok(out_id)
}
```

- [ ] **Step 2: Implement comparison ops**

These produce Bool output:
```rust
fn lt(&self, a: u64, b: u64) -> BackendResult<u64> {
    self.record_binary_with_out_dtype(a, b, WireOpKind::Lt, BackendDType::Bool)
}
```

Need a `record_binary_with_out_dtype` helper (or modify `record_binary` to accept optional out_dtype override).

- [ ] **Step 3: Implement remaining ops (bitwise, mod, elem_min/max, logical_not, quantize, dequantize)**

Follow existing patterns. Bitwise binary → `record_binary`. BitwiseNot/LogicalNot → `record_unary`. Shl/Shr → record with shift param. Quantize/Dequantize → custom recording with target_dtype.

- [ ] **Step 4: Build and verify compilation**
- [ ] **Step 5: Commit**

---

## Chunk 2: gpu-service Dispatch + Testing

### Task 5: gpu-service dispatch for new ops

**Files:**
- Modify: `crates/gpu-service/src/main.rs`

- [ ] **Step 1: Add dispatch cases for new WireOpKind variants**

In the op dispatch match, add cases for Cast, Lt, Gt, Le, Ge, Eq, Ne, BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, Shl, Shr, Mod, ElemMin, ElemMax, LogicalNot, Quantize, Dequantize.

Each calls the corresponding `applegpu_core::ops::xxx()` function on the server's LazyRuntime.

- [ ] **Step 2: Build gpu-service**

Run: `cargo build -p applegpu-service --release`

- [ ] **Step 3: Commit**

### Task 6: Debug Int32 container compute

- [ ] **Step 1: Investigate Int32 add returning zeros**

The wire protocol already supports Int32 tensors (dtype byte 5). Check:
- Is the SocketBackend sending the correct dtype in tensor creation?
- Is the gpu-service creating Int32 buffers correctly?
- Is the eval response reading Int32 bytes correctly?

- [ ] **Step 2: Fix the root cause**
- [ ] **Step 3: Commit**

### Task 7: End-to-end Docker container test

- [ ] **Step 1: Rebuild Linux wheel + gpu-service**

```bash
uv run maturin build --target aarch64-unknown-linux-gnu --zig -i python3.11
cargo build -p applegpu-service --release
```

- [ ] **Step 2: Restart gpu-service and run Docker test**

```bash
docker run --rm \
  -v ~/.applegpu/runtime.sock:/var/run/applegpu.sock \
  -e APPLEGPU_SOCKET=/var/run/applegpu.sock \
  -v $(pwd)/target/wheels:/wheels \
  python:3.11-slim \
  bash -c "pip install /wheels/*manylinux*.whl numpy && python3 -c '
import applegpu_runtime as gpu
gpu.init_backend()
print(gpu.device_name())

# F32 ops
a = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
print(\"Add:\", (a + a).to_list())

# Int32 ops
x = gpu.tensor([1, 2, 3], shape=[3], dtype=\"int32\")
print(\"Int32 add:\", (x + x).to_list())

# Cast + embedding (the container bug)
weights = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
idx = gpu.tensor([1, 0], shape=[2], dtype=\"int64\")
idx32 = gpu.cast(idx, \"int32\")
result = gpu.embedding(weights, idx32)
result.eval()
print(\"Embedding:\", result.to_list())

# Comparison
print(\"Lt:\", gpu.lt(a, gpu.tensor([2.0, 2.0, 2.0], shape=[3])).to_list())

print(\"\\nAll tests PASSED!\")
'"
```

- [ ] **Step 3: Commit milestone**

```bash
git commit --allow-empty -m "milestone: wire protocol v3 — all Plan 2/3 ops work in containers"
```
