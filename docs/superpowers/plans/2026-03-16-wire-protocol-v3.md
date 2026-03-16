# Wire Protocol v3: Multi-Dtype + New Ops Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the wire protocol, SocketBackend, and gpu-service to support all Plan 2/3 ops (Cast, comparison, bitwise, modulo, elem min/max, logical_not, quantize/dequantize) and fix Int32 compute over containers.

**Architecture:** Three layers need updating: (1) `applegpu-wire` crate (WireOpKind enum + serialization), (2) `SocketBackend` in applegpu-python (replace stubs with real implementations), (3) `serial.rs` conversions (`wire_op_to_core`, `wire_node_to_core`). The gpu-service dispatch is automatic via `wire_node_to_core` — no changes needed in `main.rs` itself. Also fix hardcoded `DType::Float32` in legacy serializer (root cause of Int32 zeros bug).

**Tech Stack:** Rust (applegpu-wire, applegpu-core, applegpu-python, gpu-service)

**Spec:** `docs/superpowers/specs/2026-03-16-multi-dtype-compute-kernels-design.md`

**Key review findings addressed:**
- Discriminant IDs must match serial.rs (Cast=46, not 47)
- `wire_node_to_core` hardcodes `out_dtype: DType::Float32` — must propagate actual dtype
- Legacy serializer hardcodes dtype to F32 — root cause of Int32 zeros bug
- `wire_op_to_core` needs new match arms for all new ops
- Missing Cast=46 deserialization arm in legacy `discriminant_to_op`
- Comparison ops need Bool output dtype in SocketBackend
- Bump PROTOCOL_VERSION to 2

---

## Chunk 1: Fix Legacy Serializer + Wire Protocol

### Task 1: Fix hardcoded Float32 in legacy serializer (Int32 zeros root cause)

**Files:**
- Modify: `crates/core/src/serial.rs`

The legacy `EvalRequest::serialize` writes dtype as 0 (Float32) for all tensors (lines ~362, ~380). `EvalRequest::deserialize` reads it back as `DType::Float32` (lines ~535, ~563). And `wire_node_to_core` hardcodes `out_dtype: DType::Float32` (line ~786). These cause Int32 data to be misinterpreted as Float32 on the server, producing zeros.

- [ ] **Step 1: Fix tensor dtype serialization in EvalRequest::serialize**

Find every `write_u32(&mut buf, 0)` that represents dtype and replace with the actual tensor's dtype discriminant. Use `DType` → u8 mapping matching the wire crate's dtype encoding.

- [ ] **Step 2: Fix tensor dtype deserialization in EvalRequest::deserialize**

Replace hardcoded `dtype: DType::Float32` with `DType::from_wire(dtype_byte)`. Add a `DType::from_wire(u8) -> DType` helper if not present.

- [ ] **Step 3: Fix wire_node_to_core out_dtype**

Replace `out_dtype: DType::Float32` with actual dtype propagation. The `WireOpNode` has an `out_dtype: u8` field — convert it via `DType::from_wire(node.out_dtype)`.

- [ ] **Step 4: Fix missing Cast=46 deserialization arm**

In `discriminant_to_op`, add the missing arm for discriminant 46 (Cast). Read `target_dtype` as u8 from the payload.

- [ ] **Step 5: Test + commit**

```bash
cargo test -p applegpu-core serial
git commit -m "fix: propagate actual dtype in legacy serializer (was hardcoded Float32)"
```

### Task 2: Add new WireOpKind variants + serialization

**Files:**
- Modify: `crates/wire/src/lib.rs`

- [ ] **Step 1: Bump PROTOCOL_VERSION to 2**

- [ ] **Step 2: Add variants to WireOpKind enum**

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

- [ ] **Step 3: Add serialization (write_op/read_op)**

**Discriminant IDs must match serial.rs:** Cast=46, Lt=47, Gt=48, Le=49, Ge=50, Eq=51, Ne=52, BitwiseAnd=53, BitwiseOr=54, BitwiseXor=55, BitwiseNot=56, Shl=57, Shr=58, Mod=59, ElemMin=60, ElemMax=61, LogicalNot=62, Quantize=63, Dequantize=64.

Payload for each:
- Cast: 1 byte (target_dtype u8)
- Lt/Gt/Le/Ge/Eq/Ne: no payload
- BitwiseAnd/Or/Xor/Not: no payload
- Shl/Shr: 4 bytes (shift u32)
- Mod/ElemMin/ElemMax/LogicalNot: no payload
- Quantize/Dequantize: 9 bytes (scale f32 + zero_point i32 + target_dtype u8)

- [ ] **Step 4: Test serialization roundtrip**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: wire protocol v3 — 19 new WireOpKind variants"
```

### Task 3: Update serial.rs conversions (all three functions)

**Files:**
- Modify: `crates/core/src/serial.rs`

Three functions need updating:

- [ ] **Step 1: Update `From<&OpKind> for WireOpKind`**

Replace all `unimplemented!("... not supported over wire")` stubs with real conversions:
```rust
OpKind::Cast { target_dtype } => WireOpKind::Cast { target_dtype: dtype_to_wire(*target_dtype) },
OpKind::Lt => WireOpKind::Lt,
OpKind::Gt => WireOpKind::Gt,
// ... etc for all 19 ops
```

- [ ] **Step 2: Update `wire_op_to_core`**

Add match arms for all new WireOpKind variants:
```rust
WireOpKind::Cast { target_dtype } => OpKind::Cast { target_dtype: DType::from_wire(*target_dtype) },
WireOpKind::Lt => OpKind::Lt,
// ... etc
```

This is the function the gpu-service actually uses (called by `wire_node_to_core`).

- [ ] **Step 3: Add DType ↔ u8 helpers if not present**

```rust
fn dtype_to_wire(dt: DType) -> u8 { ... }  // DType → wire discriminant
impl DType { fn from_wire(b: u8) -> DType { ... } }  // wire discriminant → DType
```

Check `crates/wire/src/lib.rs` for existing dtype encoding — match it exactly.

- [ ] **Step 4: Test + commit**

```bash
cargo test -p applegpu-core serial
git commit -m "feat: serial.rs conversions for all 19 new wire ops"
```

---

## Chunk 2: SocketBackend + Testing

### Task 4: Wire SocketBackend ops (replace stubs)

**Files:**
- Modify: `crates/python/src/socket_backend.rs`

- [ ] **Step 1: Add `record_binary_with_out_dtype` helper**

Comparison ops produce Bool output. The existing `record_binary` always uses input dtype as output dtype. Add a variant:

```rust
fn record_binary_with_out_dtype(&self, a: u64, b: u64, op: WireOpKind, out_dtype: WireDType) -> BackendResult<u64> {
    let tensors = self.tensors.lock().unwrap();
    let graph = self.graph.lock().unwrap();
    // ... same as record_binary but uses out_dtype parameter instead of input dtype
}
```

- [ ] **Step 2: Implement Cast**

```rust
fn cast(&self, a: u64, target_dtype: &str) -> BackendResult<u64> {
    let dt = self.parse_dtype(target_dtype)?;
    // Record as unary op with different output dtype
    self.record_unary_with_out_dtype(a, WireOpKind::Cast { target_dtype: dt as u8 }, dt)
}
```

- [ ] **Step 3: Implement comparison ops (Bool output)**

```rust
fn lt(&self, a: u64, b: u64) -> BackendResult<u64> {
    self.record_binary_with_out_dtype(a, b, WireOpKind::Lt, WireDType::Bool)
}
// ... gt, le, ge, eq_op, ne_op
```

- [ ] **Step 4: Implement bitwise binary ops**

```rust
fn bitwise_and(&self, a: u64, b: u64) -> BackendResult<u64> { self.record_binary(a, b, WireOpKind::BitwiseAnd) }
fn bitwise_or(&self, a: u64, b: u64) -> BackendResult<u64> { self.record_binary(a, b, WireOpKind::BitwiseOr) }
fn bitwise_xor(&self, a: u64, b: u64) -> BackendResult<u64> { self.record_binary(a, b, WireOpKind::BitwiseXor) }
fn mod_(&self, a: u64, b: u64) -> BackendResult<u64> { self.record_binary(a, b, WireOpKind::Mod) }
fn elem_min(&self, a: u64, b: u64) -> BackendResult<u64> { self.record_binary(a, b, WireOpKind::ElemMin) }
fn elem_max(&self, a: u64, b: u64) -> BackendResult<u64> { self.record_binary(a, b, WireOpKind::ElemMax) }
```

- [ ] **Step 5: Implement unary ops**

```rust
fn bitwise_not(&self, a: u64) -> BackendResult<u64> { self.record_unary(a, WireOpKind::BitwiseNot) }
fn logical_not(&self, a: u64) -> BackendResult<u64> {
    self.record_unary_with_out_dtype(a, WireOpKind::LogicalNot, WireDType::Bool)
}
fn shl(&self, a: u64, shift: u32) -> BackendResult<u64> { self.record_unary(a, WireOpKind::Shl { shift }) }
fn shr(&self, a: u64, shift: u32) -> BackendResult<u64> { self.record_unary(a, WireOpKind::Shr { shift }) }
```

- [ ] **Step 6: Implement quantize/dequantize**

These change output dtype:
```rust
fn quantize(&self, a: u64, target_dtype: &str, scale: f32, zero_point: i32) -> BackendResult<u64> {
    let dt = self.parse_dtype(target_dtype)?;
    self.record_unary_with_out_dtype(a,
        WireOpKind::Quantize { scale, zero_point, target_dtype: dt as u8 }, dt)
}
fn dequantize(&self, a: u64, target_dtype: &str, scale: f32, zero_point: i32) -> BackendResult<u64> {
    let dt = self.parse_dtype(target_dtype)?;
    self.record_unary_with_out_dtype(a,
        WireOpKind::Dequantize { scale, zero_point, target_dtype: dt as u8 }, dt)
}
```

- [ ] **Step 7: Build and verify**

Run: `cargo check -p applegpu-python`

- [ ] **Step 8: Commit**

```bash
git commit -m "feat: SocketBackend real implementations for all Plan 2/3 ops"
```

### Task 5: End-to-end Docker container test

- [ ] **Step 1: Rebuild everything**

```bash
cargo build -p applegpu-service --release
uv run maturin build --target aarch64-unknown-linux-gnu --zig -i python3.11
```

- [ ] **Step 2: Restart gpu-service**

```bash
kill $(cat ~/.applegpu/gpu-service.pid 2>/dev/null) 2>/dev/null
sleep 1
nohup target/release/gpu-service > /tmp/gpu-service.log 2>&1 &
sleep 2
```

- [ ] **Step 3: Run Docker test**

```bash
docker run --rm \
  -v ~/.applegpu/runtime.sock:/var/run/applegpu.sock \
  -e APPLEGPU_SOCKET=/var/run/applegpu.sock \
  -v $(pwd)/target/wheels:/wheels \
  python:3.11-slim \
  bash -c "pip install /wheels/*manylinux*.whl numpy && python3 -c '
import applegpu_runtime as gpu
gpu.init_backend()
print(\"Backend:\", gpu.device_name())

# F32 ops
a = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
print(\"Add:\", (a + a).to_list())

# Int32 ops (was returning zeros)
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
a2 = gpu.tensor([1.0, 5.0, 3.0], shape=[3])
b2 = gpu.tensor([2.0, 4.0, 3.0], shape=[3])
print(\"Lt:\", gpu.lt(a2, b2).to_list())

print()
print(\"All container tests PASSED!\")
'"
```

Expected output:
```
Backend: gpu-service (remote)
Add: [2.0, 4.0, 6.0]
Int32 add: [2, 4, 6]
Embedding: [3.0, 4.0, 1.0, 2.0]
Lt: [True, False, False]

All container tests PASSED!
```

- [ ] **Step 4: Commit milestone**

```bash
git commit --allow-empty -m "milestone: wire protocol v3 — all Plan 2/3 ops work in containers"
```
