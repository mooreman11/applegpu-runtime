# Phase 7b: Container GPU Bridge Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-client GPU service with dual transport (Unix socket + vsock) that allows containers to submit GPU workloads to a shared Metal GPU with fair scheduling and per-container isolation.

**Architecture:** Extract wire protocol into a shared crate (`crates/wire`), upgrade the existing single-client `gpu-service` to thread-per-connection with a shared `LazyRuntime`, add a handshake protocol for container registration, and create a Linux-compilable client crate (`crates/client`).

**Tech Stack:** Rust (std only for wire/client crates, applegpu-core for service), Unix domain sockets, AF_VSOCK (via libc), existing Metal compute infrastructure

**Spec:** `docs/superpowers/specs/2026-03-15-container-gpu-bridge-design.md`

---

## File Structure

### New Files

- `crates/wire/Cargo.toml` — Wire protocol crate (no platform deps, compiles on Linux + macOS)
- `crates/wire/src/lib.rs` — Wire types, serialization, handshake, framing helpers
- `crates/client/Cargo.toml` — Container-side client library (depends on wire only)
- `crates/client/src/lib.rs` — `GpuClient` with connect/handshake/eval/disconnect
- `crates/client/src/transport.rs` — Unix socket + vsock transport implementations

### Modified Files

- `Cargo.toml` — Add `crates/wire` and `crates/client` to workspace members
- `crates/core/Cargo.toml` — Add `applegpu-wire` dependency
- `crates/core/src/serial.rs` — Replace inline serialization with `From`/`TryFrom` conversions to wire types, re-export wire types
- `crates/core/src/lazy.rs` — Add `cleanup_container()` method
- `crates/core/src/lib.rs` — No changes needed (serial module stays)
- `crates/gpu-service/Cargo.toml` — Add `applegpu-wire` dependency
- `crates/gpu-service/src/main.rs` — Rewrite: thread-per-connection, shared state, handshake, cleanup

### Test Files

- `crates/wire/src/lib.rs` — Inline `#[cfg(test)]` unit tests for all wire types
- `crates/client/src/lib.rs` — Inline unit tests with mock transport
- `crates/core/tests/cleanup_integration.rs` — Container cleanup integration test
- `crates/gpu-service/tests/multi_client.rs` — Multi-client concurrency integration test

---

## Chunk 1: Wire Protocol Crate

### Task 1: Create `crates/wire` with framing helpers

**Files:**
- Create: `crates/wire/Cargo.toml`
- Create: `crates/wire/src/lib.rs`
- Modify: `Cargo.toml` (workspace)

- [ ] **Step 1: Create wire crate Cargo.toml**

Create `crates/wire/Cargo.toml`:

```toml
[package]
name = "applegpu-wire"
version.workspace = true
edition.workspace = true

[lib]
name = "applegpu_wire"
```

- [ ] **Step 2: Add wire to workspace members**

In root `Cargo.toml`, update members:

```toml
[workspace]
members = ["crates/core", "crates/python", "crates/gpu-service", "crates/wire"]
```

- [ ] **Step 3: Write failing test for framing helpers**

Create `crates/wire/src/lib.rs` with tests first:

```rust
use std::io::{self, Read, Write, Cursor};

// ── Primitive helpers ────────────────────────────────────────────────

pub fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
pub fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
pub fn write_f32(w: &mut impl Write, v: f32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
pub fn write_i32(w: &mut impl Write, v: i32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }

pub fn read_u32(r: &mut impl Read) -> io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
pub fn read_u64(r: &mut impl Read) -> io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
pub fn read_f32(r: &mut impl Read) -> io::Result<f32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b)) }
pub fn read_i32(r: &mut impl Read) -> io::Result<i32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(i32::from_le_bytes(b)) }

// ── Length-prefixed framing ──────────────────────────────────────────

/// Write a length-prefixed message to a stream: [u32 len][payload].
pub fn write_message(w: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    write_u32(w, payload.len() as u32)?;
    w.write_all(payload)?;
    w.flush()
}

/// Read a length-prefixed message from a stream. Returns the payload bytes.
/// Rejects messages larger than `max_size` bytes.
pub fn read_message(r: &mut impl Read, max_size: usize) -> io::Result<Vec<u8>> {
    let len = read_u32(r)? as usize;
    if len > max_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Message too large: {} bytes (max {})", len, max_size),
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Maximum message size: 256 MB.
pub const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_roundtrip() {
        let payload = b"hello wire";
        let mut buf = Vec::new();
        write_message(&mut buf, payload).unwrap();

        let mut cursor = Cursor::new(&buf);
        let result = read_message(&mut cursor, MAX_MESSAGE_SIZE).unwrap();
        assert_eq!(result, payload);
    }

    #[test]
    fn framing_rejects_oversized() {
        let mut buf = Vec::new();
        write_u32(&mut buf, 1_000_000).unwrap(); // claim 1MB payload
        buf.extend_from_slice(&[0u8; 100]); // but only 100 bytes

        let mut cursor = Cursor::new(&buf);
        let result = read_message(&mut cursor, 100); // max 100 bytes
        assert!(result.is_err());
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-wire 2>&1`
Expected: 2 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/wire/ Cargo.toml
git commit -m "feat: add wire crate with length-prefixed framing helpers"
```

---

### Task 2: Add handshake types to wire crate

**Files:**
- Modify: `crates/wire/src/lib.rs`

- [ ] **Step 1: Write failing tests for handshake roundtrip**

Add to `crates/wire/src/lib.rs`, above the existing tests module:

```rust
// ── Handshake protocol ───────────────────────────────────────────────

const MAGIC_HANDSHAKE_REQ: &[u8; 4] = b"AGHI";
const MAGIC_HANDSHAKE_RESP: &[u8; 4] = b"AGHO";

/// Client handshake request.
#[derive(Debug, Clone, PartialEq)]
pub struct HandshakeRequest {
    pub protocol_version: u32,
    pub requested_memory: u64,
}

/// Server handshake response.
#[derive(Debug, Clone, PartialEq)]
pub struct HandshakeResponse {
    pub status: u32,
    pub container_id: u64,
    pub granted_memory: u64,
}

/// Handshake status codes.
pub const HANDSHAKE_OK: u32 = 0;
pub const HANDSHAKE_REJECTED_QUOTA: u32 = 1;
pub const HANDSHAKE_REJECTED_CAPACITY: u32 = 2;

/// Current protocol version.
pub const PROTOCOL_VERSION: u32 = 1;

impl HandshakeRequest {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.write_all(MAGIC_HANDSHAKE_REQ).unwrap();
        write_u32(&mut buf, self.protocol_version).unwrap();
        write_u64(&mut buf, self.requested_memory).unwrap();
        buf
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_HANDSHAKE_REQ {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad handshake request magic"));
        }
        let protocol_version = read_u32(&mut r)?;
        let requested_memory = read_u64(&mut r)?;
        Ok(HandshakeRequest { protocol_version, requested_memory })
    }
}

impl HandshakeResponse {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);
        buf.write_all(MAGIC_HANDSHAKE_RESP).unwrap();
        write_u32(&mut buf, self.status).unwrap();
        write_u64(&mut buf, self.container_id).unwrap();
        write_u64(&mut buf, self.granted_memory).unwrap();
        buf
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_HANDSHAKE_RESP {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad handshake response magic"));
        }
        let status = read_u32(&mut r)?;
        let container_id = read_u64(&mut r)?;
        let granted_memory = read_u64(&mut r)?;
        Ok(HandshakeResponse { status, container_id, granted_memory })
    }
}
```

Add tests:

```rust
    #[test]
    fn handshake_request_roundtrip() {
        let req = HandshakeRequest {
            protocol_version: PROTOCOL_VERSION,
            requested_memory: 64 * 1024 * 1024 * 1024, // 64 GB
        };
        let bytes = req.serialize();
        let decoded = HandshakeRequest::deserialize(&bytes).unwrap();
        assert_eq!(decoded, req);
    }

    #[test]
    fn handshake_response_roundtrip() {
        let resp = HandshakeResponse {
            status: HANDSHAKE_OK,
            container_id: 42,
            granted_memory: 32 * 1024 * 1024 * 1024, // 32 GB
        };
        let bytes = resp.serialize();
        let decoded = HandshakeResponse::deserialize(&bytes).unwrap();
        assert_eq!(decoded, resp);
    }

    #[test]
    fn handshake_response_rejected() {
        let resp = HandshakeResponse {
            status: HANDSHAKE_REJECTED_QUOTA,
            container_id: 0,
            granted_memory: 0,
        };
        let bytes = resp.serialize();
        let decoded = HandshakeResponse::deserialize(&bytes).unwrap();
        assert_eq!(decoded.status, HANDSHAKE_REJECTED_QUOTA);
    }
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p applegpu-wire 2>&1`
Expected: 5 tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/wire/src/lib.rs
git commit -m "feat: add handshake request/response types to wire crate"
```

---

### Task 3: Add EvalRequest/EvalResponse wire types

**Files:**
- Modify: `crates/wire/src/lib.rs`

This is the largest step. The wire crate gets its own `WireOpKind`, `WireOpNode`, `WireTensorData`, `EvalRequest`, and `EvalResponse` types — mirroring the serialization logic currently in `crates/core/src/serial.rs` but decoupled from core types.

- [ ] **Step 1: Add wire eval types and serialization**

Add to `crates/wire/src/lib.rs`:

```rust
// ── Eval protocol ────────────────────────────────────────────────────

const MAGIC_REQUEST: &[u8; 4] = b"AGPU";
const MAGIC_RESPONSE: &[u8; 4] = b"AGPR";
const EVAL_VERSION: u32 = 2;

/// Wire representation of an operation kind.
/// Each variant carries its own payload data for serialization.
#[derive(Debug, Clone)]
pub enum WireOpKind {
    // discriminant 0-9: basic ops
    Add, Sub, Mul, Div, Neg, Relu, Exp, Log, Sqrt, Matmul,
    // 10: fused
    FusedElementwise { kernel_source: String, function_name: String },
    // 11-45: extended ops
    Softmax,
    Transpose { dim0: usize, dim1: usize },
    ScalarMul(f32),
    Gelu,
    LayerNorm { eps: f32 },
    Embedding,
    Reshape { new_shape: Vec<usize> },
    Slice { dim: usize, start: usize, end: usize },
    Concat { dim: usize },
    AddBias,
    SoftmaxCausal,
    Argmax,
    Sum,
    Mean,
    Abs,
    Sign,
    Pow { exponent: f32 },
    Clamp { min_val: f32, max_val: f32 },
    Where,
    MaskedFill { value: f32 },
    Triu { diagonal: i32 },
    Tril { diagonal: i32 },
    Gather { dim: usize },
    IndexSelect { dim: usize },
    Conv1d { stride: usize, padding: usize },
    Conv2d { stride: (usize, usize), padding: (usize, usize) },
    BatchNorm { eps: f32 },
    MaxPool2d { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
    AvgPool2d { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
    Tanh,
    SoftmaxBackward,
    LayerNormBackward { eps: f32 },
    Conv2dBackwardInput { stride: (usize, usize), padding: (usize, usize) },
    EmbeddingBackward,
    BatchNormBackward { eps: f32 },
}

/// Wire representation of a graph node.
#[derive(Debug, Clone)]
pub struct WireOpNode {
    pub id: u64,
    pub op: WireOpKind,
    pub inputs: Vec<u64>,
    pub out_shape: Vec<usize>,
    pub out_dtype: u32,
}

/// Wire representation of tensor data.
#[derive(Debug, Clone)]
pub struct WireTensorData {
    pub id: u64,
    pub shape: Vec<usize>,
    pub dtype: u32,
    pub data: Vec<u8>,
}

/// Request to evaluate a computation graph remotely.
#[derive(Debug)]
pub struct EvalRequest {
    pub target_id: u64,
    pub tensors: Vec<WireTensorData>,
    pub nodes: Vec<WireOpNode>,
}

/// Response from remote evaluation.
#[derive(Debug)]
pub enum EvalResponse {
    Ok {
        tensor_id: u64,
        shape: Vec<usize>,
        data: Vec<u8>,
    },
    Err(String),
}
```

- [ ] **Step 2: Add WireOpKind discriminant conversion**

Add to `crates/wire/src/lib.rs`:

```rust
impl WireOpKind {
    fn discriminant(&self) -> u32 {
        match self {
            WireOpKind::Add => 0, WireOpKind::Sub => 1, WireOpKind::Mul => 2,
            WireOpKind::Div => 3, WireOpKind::Neg => 4, WireOpKind::Relu => 5,
            WireOpKind::Exp => 6, WireOpKind::Log => 7, WireOpKind::Sqrt => 8,
            WireOpKind::Matmul => 9, WireOpKind::FusedElementwise { .. } => 10,
            WireOpKind::Softmax => 11, WireOpKind::Transpose { .. } => 12,
            WireOpKind::ScalarMul(_) => 13, WireOpKind::Gelu => 14,
            WireOpKind::LayerNorm { .. } => 15, WireOpKind::Embedding => 16,
            WireOpKind::Reshape { .. } => 17, WireOpKind::Slice { .. } => 18,
            WireOpKind::Concat { .. } => 19, WireOpKind::AddBias => 20,
            WireOpKind::SoftmaxCausal => 21, WireOpKind::Argmax => 22,
            WireOpKind::Sum => 23, WireOpKind::Mean => 24,
            WireOpKind::Abs => 25, WireOpKind::Sign => 26,
            WireOpKind::Pow { .. } => 27, WireOpKind::Clamp { .. } => 28,
            WireOpKind::Where => 29, WireOpKind::MaskedFill { .. } => 30,
            WireOpKind::Triu { .. } => 31, WireOpKind::Tril { .. } => 32,
            WireOpKind::Gather { .. } => 33, WireOpKind::IndexSelect { .. } => 34,
            WireOpKind::Conv1d { .. } => 35, WireOpKind::Conv2d { .. } => 36,
            WireOpKind::BatchNorm { .. } => 37, WireOpKind::MaxPool2d { .. } => 38,
            WireOpKind::AvgPool2d { .. } => 39, WireOpKind::Tanh => 40,
            WireOpKind::SoftmaxBackward => 41, WireOpKind::LayerNormBackward { .. } => 42,
            WireOpKind::Conv2dBackwardInput { .. } => 43, WireOpKind::EmbeddingBackward => 44,
            WireOpKind::BatchNormBackward { .. } => 45,
        }
    }

    fn write_payload(&self, buf: &mut Vec<u8>) {
        match self {
            WireOpKind::FusedElementwise { kernel_source, function_name } => {
                write_u32(buf, kernel_source.len() as u32).unwrap();
                buf.write_all(kernel_source.as_bytes()).unwrap();
                write_u32(buf, function_name.len() as u32).unwrap();
                buf.write_all(function_name.as_bytes()).unwrap();
            }
            WireOpKind::Transpose { dim0, dim1 } => {
                write_u32(buf, *dim0 as u32).unwrap();
                write_u32(buf, *dim1 as u32).unwrap();
            }
            WireOpKind::ScalarMul(scale) => { write_f32(buf, *scale).unwrap(); }
            WireOpKind::LayerNorm { eps } => { write_f32(buf, *eps).unwrap(); }
            WireOpKind::Reshape { new_shape } => {
                write_u32(buf, new_shape.len() as u32).unwrap();
                for &d in new_shape { write_u64(buf, d as u64).unwrap(); }
            }
            WireOpKind::Slice { dim, start, end } => {
                write_u32(buf, *dim as u32).unwrap();
                write_u64(buf, *start as u64).unwrap();
                write_u64(buf, *end as u64).unwrap();
            }
            WireOpKind::Concat { dim } => { write_u32(buf, *dim as u32).unwrap(); }
            WireOpKind::Pow { exponent } => { write_f32(buf, *exponent).unwrap(); }
            WireOpKind::Clamp { min_val, max_val } => {
                write_f32(buf, *min_val).unwrap();
                write_f32(buf, *max_val).unwrap();
            }
            WireOpKind::MaskedFill { value } => { write_f32(buf, *value).unwrap(); }
            WireOpKind::Triu { diagonal } => { write_i32(buf, *diagonal).unwrap(); }
            WireOpKind::Tril { diagonal } => { write_i32(buf, *diagonal).unwrap(); }
            WireOpKind::Gather { dim } => { write_u32(buf, *dim as u32).unwrap(); }
            WireOpKind::IndexSelect { dim } => { write_u32(buf, *dim as u32).unwrap(); }
            WireOpKind::Conv1d { stride, padding } => {
                write_u64(buf, *stride as u64).unwrap();
                write_u64(buf, *padding as u64).unwrap();
            }
            WireOpKind::Conv2d { stride, padding } => {
                write_u64(buf, stride.0 as u64).unwrap();
                write_u64(buf, stride.1 as u64).unwrap();
                write_u64(buf, padding.0 as u64).unwrap();
                write_u64(buf, padding.1 as u64).unwrap();
            }
            WireOpKind::BatchNorm { eps } => { write_f32(buf, *eps).unwrap(); }
            WireOpKind::MaxPool2d { kernel_size, stride, padding } |
            WireOpKind::AvgPool2d { kernel_size, stride, padding } => {
                write_u64(buf, kernel_size.0 as u64).unwrap();
                write_u64(buf, kernel_size.1 as u64).unwrap();
                write_u64(buf, stride.0 as u64).unwrap();
                write_u64(buf, stride.1 as u64).unwrap();
                write_u64(buf, padding.0 as u64).unwrap();
                write_u64(buf, padding.1 as u64).unwrap();
            }
            WireOpKind::LayerNormBackward { eps } => { write_f32(buf, *eps).unwrap(); }
            WireOpKind::Conv2dBackwardInput { stride, padding } => {
                write_u64(buf, stride.0 as u64).unwrap();
                write_u64(buf, stride.1 as u64).unwrap();
                write_u64(buf, padding.0 as u64).unwrap();
                write_u64(buf, padding.1 as u64).unwrap();
            }
            WireOpKind::BatchNormBackward { eps } => { write_f32(buf, *eps).unwrap(); }
            // All remaining ops have no payload
            _ => {}
        }
    }

    fn read_from(disc: u32, r: &mut impl Read) -> io::Result<Self> {
        match disc {
            0 => Ok(WireOpKind::Add), 1 => Ok(WireOpKind::Sub),
            2 => Ok(WireOpKind::Mul), 3 => Ok(WireOpKind::Div),
            4 => Ok(WireOpKind::Neg), 5 => Ok(WireOpKind::Relu),
            6 => Ok(WireOpKind::Exp), 7 => Ok(WireOpKind::Log),
            8 => Ok(WireOpKind::Sqrt), 9 => Ok(WireOpKind::Matmul),
            10 => {
                let src_len = read_u32(r)? as usize;
                let mut src = vec![0u8; src_len];
                r.read_exact(&mut src)?;
                let kernel_source = String::from_utf8(src)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let name_len = read_u32(r)? as usize;
                let mut name = vec![0u8; name_len];
                r.read_exact(&mut name)?;
                let function_name = String::from_utf8(name)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                Ok(WireOpKind::FusedElementwise { kernel_source, function_name })
            }
            11 => Ok(WireOpKind::Softmax),
            12 => Ok(WireOpKind::Transpose { dim0: read_u32(r)? as usize, dim1: read_u32(r)? as usize }),
            13 => Ok(WireOpKind::ScalarMul(read_f32(r)?)),
            14 => Ok(WireOpKind::Gelu),
            15 => Ok(WireOpKind::LayerNorm { eps: read_f32(r)? }),
            16 => Ok(WireOpKind::Embedding),
            17 => {
                let ndims = read_u32(r)? as usize;
                let mut new_shape = Vec::with_capacity(ndims);
                for _ in 0..ndims { new_shape.push(read_u64(r)? as usize); }
                Ok(WireOpKind::Reshape { new_shape })
            }
            18 => Ok(WireOpKind::Slice { dim: read_u32(r)? as usize, start: read_u64(r)? as usize, end: read_u64(r)? as usize }),
            19 => Ok(WireOpKind::Concat { dim: read_u32(r)? as usize }),
            20 => Ok(WireOpKind::AddBias),
            21 => Ok(WireOpKind::SoftmaxCausal),
            22 => Ok(WireOpKind::Argmax),
            23 => Ok(WireOpKind::Sum),
            24 => Ok(WireOpKind::Mean),
            25 => Ok(WireOpKind::Abs),
            26 => Ok(WireOpKind::Sign),
            27 => Ok(WireOpKind::Pow { exponent: read_f32(r)? }),
            28 => Ok(WireOpKind::Clamp { min_val: read_f32(r)?, max_val: read_f32(r)? }),
            29 => Ok(WireOpKind::Where),
            30 => Ok(WireOpKind::MaskedFill { value: read_f32(r)? }),
            31 => Ok(WireOpKind::Triu { diagonal: read_i32(r)? }),
            32 => Ok(WireOpKind::Tril { diagonal: read_i32(r)? }),
            33 => Ok(WireOpKind::Gather { dim: read_u32(r)? as usize }),
            34 => Ok(WireOpKind::IndexSelect { dim: read_u32(r)? as usize }),
            35 => Ok(WireOpKind::Conv1d { stride: read_u64(r)? as usize, padding: read_u64(r)? as usize }),
            36 => {
                let s0 = read_u64(r)? as usize; let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize; let p1 = read_u64(r)? as usize;
                Ok(WireOpKind::Conv2d { stride: (s0, s1), padding: (p0, p1) })
            }
            37 => Ok(WireOpKind::BatchNorm { eps: read_f32(r)? }),
            38 => {
                let k0 = read_u64(r)? as usize; let k1 = read_u64(r)? as usize;
                let s0 = read_u64(r)? as usize; let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize; let p1 = read_u64(r)? as usize;
                Ok(WireOpKind::MaxPool2d { kernel_size: (k0, k1), stride: (s0, s1), padding: (p0, p1) })
            }
            39 => {
                let k0 = read_u64(r)? as usize; let k1 = read_u64(r)? as usize;
                let s0 = read_u64(r)? as usize; let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize; let p1 = read_u64(r)? as usize;
                Ok(WireOpKind::AvgPool2d { kernel_size: (k0, k1), stride: (s0, s1), padding: (p0, p1) })
            }
            40 => Ok(WireOpKind::Tanh),
            41 => Ok(WireOpKind::SoftmaxBackward),
            42 => Ok(WireOpKind::LayerNormBackward { eps: read_f32(r)? }),
            43 => {
                let s0 = read_u64(r)? as usize; let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize; let p1 = read_u64(r)? as usize;
                Ok(WireOpKind::Conv2dBackwardInput { stride: (s0, s1), padding: (p0, p1) })
            }
            44 => Ok(WireOpKind::EmbeddingBackward),
            45 => Ok(WireOpKind::BatchNormBackward { eps: read_f32(r)? }),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unknown op type: {}", disc))),
        }
    }
}
```

- [ ] **Step 3: Add EvalRequest/EvalResponse serialize/deserialize**

Add to `crates/wire/src/lib.rs`:

```rust
impl EvalRequest {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_REQUEST).unwrap();
        write_u32(&mut buf, EVAL_VERSION).unwrap();
        write_u64(&mut buf, self.target_id).unwrap();

        write_u32(&mut buf, self.tensors.len() as u32).unwrap();
        for t in &self.tensors {
            write_u64(&mut buf, t.id).unwrap();
            write_u32(&mut buf, t.shape.len() as u32).unwrap();
            for &d in &t.shape { write_u64(&mut buf, d as u64).unwrap(); }
            write_u32(&mut buf, t.dtype).unwrap();
            write_u32(&mut buf, t.data.len() as u32).unwrap();
            buf.write_all(&t.data).unwrap();
        }

        write_u32(&mut buf, self.nodes.len() as u32).unwrap();
        for node in &self.nodes {
            write_u64(&mut buf, node.id).unwrap();
            write_u32(&mut buf, node.op.discriminant()).unwrap();
            write_u32(&mut buf, node.inputs.len() as u32).unwrap();
            for &inp in &node.inputs { write_u64(&mut buf, inp).unwrap(); }
            write_u32(&mut buf, node.out_shape.len() as u32).unwrap();
            for &d in &node.out_shape { write_u64(&mut buf, d as u64).unwrap(); }
            write_u32(&mut buf, node.out_dtype).unwrap();
            node.op.write_payload(&mut buf);
        }
        buf
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_REQUEST {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad request magic"));
        }
        let version = read_u32(&mut r)?;
        if version != EVAL_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Unsupported version: {} (expected {})", version, EVAL_VERSION)));
        }
        let target_id = read_u64(&mut r)?;

        let num_tensors = read_u32(&mut r)? as usize;
        let mut tensors = Vec::with_capacity(num_tensors);
        for _ in 0..num_tensors {
            let id = read_u64(&mut r)?;
            let num_dims = read_u32(&mut r)? as usize;
            let mut shape = Vec::with_capacity(num_dims);
            for _ in 0..num_dims { shape.push(read_u64(&mut r)? as usize); }
            let dtype = read_u32(&mut r)?;
            let data_len = read_u32(&mut r)? as usize;
            let mut data = vec![0u8; data_len];
            r.read_exact(&mut data)?;
            tensors.push(WireTensorData { id, shape, dtype, data });
        }

        let num_nodes = read_u32(&mut r)? as usize;
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let id = read_u64(&mut r)?;
            let op_disc = read_u32(&mut r)?;
            let num_inputs = read_u32(&mut r)? as usize;
            let mut inputs = Vec::with_capacity(num_inputs);
            for _ in 0..num_inputs { inputs.push(read_u64(&mut r)?); }
            let num_out_dims = read_u32(&mut r)? as usize;
            let mut out_shape = Vec::with_capacity(num_out_dims);
            for _ in 0..num_out_dims { out_shape.push(read_u64(&mut r)? as usize); }
            let out_dtype = read_u32(&mut r)?;
            let op = WireOpKind::read_from(op_disc, &mut r)?;
            nodes.push(WireOpNode { id, op, inputs, out_shape, out_dtype });
        }

        Ok(EvalRequest { target_id, tensors, nodes })
    }
}

impl EvalResponse {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_RESPONSE).unwrap();
        match self {
            EvalResponse::Ok { tensor_id, shape, data } => {
                write_u32(&mut buf, 0).unwrap();
                write_u64(&mut buf, *tensor_id).unwrap();
                write_u32(&mut buf, shape.len() as u32).unwrap();
                for &d in shape { write_u64(&mut buf, d as u64).unwrap(); }
                write_u32(&mut buf, data.len() as u32).unwrap();
                buf.write_all(data).unwrap();
            }
            EvalResponse::Err(msg) => {
                write_u32(&mut buf, 1).unwrap();
                let bytes = msg.as_bytes();
                write_u32(&mut buf, bytes.len() as u32).unwrap();
                buf.write_all(bytes).unwrap();
            }
        }
        buf
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_RESPONSE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad response magic"));
        }
        let status = read_u32(&mut r)?;
        if status == 0 {
            let tensor_id = read_u64(&mut r)?;
            let num_dims = read_u32(&mut r)? as usize;
            let mut shape = Vec::with_capacity(num_dims);
            for _ in 0..num_dims { shape.push(read_u64(&mut r)? as usize); }
            let data_len = read_u32(&mut r)? as usize;
            let mut data = vec![0u8; data_len];
            r.read_exact(&mut data)?;
            Ok(EvalResponse::Ok { tensor_id, shape, data })
        } else {
            let msg_len = read_u32(&mut r)? as usize;
            let mut msg = vec![0u8; msg_len];
            r.read_exact(&mut msg)?;
            let msg = String::from_utf8(msg)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(EvalResponse::Err(msg))
        }
    }
}
```

- [ ] **Step 4: Add roundtrip tests**

Add to tests module:

```rust
    #[test]
    fn eval_request_roundtrip_basic() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let req = EvalRequest {
            target_id: 42,
            tensors: vec![WireTensorData { id: 1, shape: vec![4], dtype: 0, data: bytes }],
            nodes: vec![WireOpNode {
                id: 42, op: WireOpKind::Neg, inputs: vec![1],
                out_shape: vec![4], out_dtype: 0,
            }],
        };
        let serialized = req.serialize();
        let decoded = EvalRequest::deserialize(&serialized).unwrap();
        assert_eq!(decoded.target_id, 42);
        assert_eq!(decoded.tensors.len(), 1);
        assert_eq!(decoded.tensors[0].id, 1);
        assert_eq!(decoded.nodes.len(), 1);
        assert_eq!(decoded.nodes[0].id, 42);
    }

    #[test]
    fn eval_request_roundtrip_fused() {
        let req = EvalRequest {
            target_id: 10,
            tensors: vec![],
            nodes: vec![WireOpNode {
                id: 10,
                op: WireOpKind::FusedElementwise {
                    kernel_source: "kernel void test() {}".to_string(),
                    function_name: "test".to_string(),
                },
                inputs: vec![1, 2], out_shape: vec![4], out_dtype: 0,
            }],
        };
        let serialized = req.serialize();
        let decoded = EvalRequest::deserialize(&serialized).unwrap();
        match &decoded.nodes[0].op {
            WireOpKind::FusedElementwise { kernel_source, function_name } => {
                assert_eq!(kernel_source, "kernel void test() {}");
                assert_eq!(function_name, "test");
            }
            _ => panic!("Expected FusedElementwise"),
        }
    }

    #[test]
    fn eval_response_ok_roundtrip() {
        let resp = EvalResponse::Ok { tensor_id: 42, shape: vec![2, 3], data: vec![1, 2, 3, 4] };
        let serialized = resp.serialize();
        let decoded = EvalResponse::deserialize(&serialized).unwrap();
        match decoded {
            EvalResponse::Ok { tensor_id, shape, data } => {
                assert_eq!(tensor_id, 42);
                assert_eq!(shape, vec![2, 3]);
                assert_eq!(data, vec![1, 2, 3, 4]);
            }
            _ => panic!("Expected Ok"),
        }
    }

    #[test]
    fn eval_response_err_roundtrip() {
        let resp = EvalResponse::Err("something failed".to_string());
        let serialized = resp.serialize();
        let decoded = EvalResponse::deserialize(&serialized).unwrap();
        match decoded {
            EvalResponse::Err(msg) => assert_eq!(msg, "something failed"),
            _ => panic!("Expected Err"),
        }
    }
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p applegpu-wire 2>&1`
Expected: 9 tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/wire/src/lib.rs
git commit -m "feat: add EvalRequest/EvalResponse wire types with full op serialization"
```

---

## Chunk 2: Core Changes — Wire Integration and Container Cleanup

### Task 4: Update `crates/core` to depend on wire crate

**Files:**
- Modify: `crates/core/Cargo.toml`
- Modify: `crates/core/src/serial.rs`

- [ ] **Step 1: Add wire dependency to core**

In `crates/core/Cargo.toml`, add:

```toml
[dependencies]
half = "2"
once_cell = "1"
const_format = "0.2"
applegpu-wire = { path = "../wire" }
```

- [ ] **Step 2: Add From/TryFrom conversions in serial.rs**

Add at the bottom of `crates/core/src/serial.rs` (before the `#[cfg(test)]` module):

```rust
// ── Wire type conversions ────────────────────────────────────────────

use applegpu_wire::{WireOpKind, WireOpNode, WireTensorData};

impl From<&OpKind> for WireOpKind {
    fn from(op: &OpKind) -> Self {
        match op {
            OpKind::Add => WireOpKind::Add,
            OpKind::Sub => WireOpKind::Sub,
            OpKind::Mul => WireOpKind::Mul,
            OpKind::Div => WireOpKind::Div,
            OpKind::Neg => WireOpKind::Neg,
            OpKind::Relu => WireOpKind::Relu,
            OpKind::Exp => WireOpKind::Exp,
            OpKind::Log => WireOpKind::Log,
            OpKind::Sqrt => WireOpKind::Sqrt,
            OpKind::Matmul => WireOpKind::Matmul,
            OpKind::FusedElementwise { kernel_source, function_name } =>
                WireOpKind::FusedElementwise { kernel_source: kernel_source.clone(), function_name: function_name.clone() },
            OpKind::Softmax => WireOpKind::Softmax,
            OpKind::Transpose { dim0, dim1 } => WireOpKind::Transpose { dim0: *dim0, dim1: *dim1 },
            OpKind::ScalarMul(s) => WireOpKind::ScalarMul(*s),
            OpKind::Gelu => WireOpKind::Gelu,
            OpKind::LayerNorm { eps } => WireOpKind::LayerNorm { eps: *eps },
            OpKind::Embedding => WireOpKind::Embedding,
            OpKind::Reshape { new_shape } => WireOpKind::Reshape { new_shape: new_shape.clone() },
            OpKind::Slice { dim, start, end } => WireOpKind::Slice { dim: *dim, start: *start, end: *end },
            OpKind::Concat { dim } => WireOpKind::Concat { dim: *dim },
            OpKind::AddBias => WireOpKind::AddBias,
            OpKind::SoftmaxCausal => WireOpKind::SoftmaxCausal,
            OpKind::Argmax => WireOpKind::Argmax,
            OpKind::Sum => WireOpKind::Sum,
            OpKind::Mean => WireOpKind::Mean,
            OpKind::Abs => WireOpKind::Abs,
            OpKind::Sign => WireOpKind::Sign,
            OpKind::Pow { exponent } => WireOpKind::Pow { exponent: *exponent },
            OpKind::Clamp { min_val, max_val } => WireOpKind::Clamp { min_val: *min_val, max_val: *max_val },
            OpKind::Where => WireOpKind::Where,
            OpKind::MaskedFill { value } => WireOpKind::MaskedFill { value: *value },
            OpKind::Triu { diagonal } => WireOpKind::Triu { diagonal: *diagonal },
            OpKind::Tril { diagonal } => WireOpKind::Tril { diagonal: *diagonal },
            OpKind::Gather { dim } => WireOpKind::Gather { dim: *dim },
            OpKind::IndexSelect { dim } => WireOpKind::IndexSelect { dim: *dim },
            OpKind::Conv1d { stride, padding } => WireOpKind::Conv1d { stride: *stride, padding: *padding },
            OpKind::Conv2d { stride, padding } => WireOpKind::Conv2d { stride: *stride, padding: *padding },
            OpKind::BatchNorm { eps } => WireOpKind::BatchNorm { eps: *eps },
            OpKind::MaxPool2d { kernel_size, stride, padding } => WireOpKind::MaxPool2d { kernel_size: *kernel_size, stride: *stride, padding: *padding },
            OpKind::AvgPool2d { kernel_size, stride, padding } => WireOpKind::AvgPool2d { kernel_size: *kernel_size, stride: *stride, padding: *padding },
            OpKind::Tanh => WireOpKind::Tanh,
            OpKind::SoftmaxBackward => WireOpKind::SoftmaxBackward,
            OpKind::LayerNormBackward { eps } => WireOpKind::LayerNormBackward { eps: *eps },
            OpKind::Conv2dBackwardInput { stride, padding } => WireOpKind::Conv2dBackwardInput { stride: *stride, padding: *padding },
            OpKind::EmbeddingBackward => WireOpKind::EmbeddingBackward,
            OpKind::BatchNormBackward { eps } => WireOpKind::BatchNormBackward { eps: *eps },
        }
    }
}

impl From<&OpNode> for WireOpNode {
    fn from(node: &OpNode) -> Self {
        WireOpNode {
            id: node.id,
            op: WireOpKind::from(&node.op),
            inputs: node.inputs.clone(),
            out_shape: node.out_shape.dims().to_vec(),
            out_dtype: 0, // TODO: encode DType properly
        }
    }
}

/// Convert a WireOpKind back to a core OpKind.
/// This is the inverse of From<&OpKind> for WireOpKind.
pub fn wire_op_to_core(wire: &WireOpKind) -> OpKind {
    match wire {
        WireOpKind::Add => OpKind::Add,
        WireOpKind::Sub => OpKind::Sub,
        WireOpKind::Mul => OpKind::Mul,
        WireOpKind::Div => OpKind::Div,
        WireOpKind::Neg => OpKind::Neg,
        WireOpKind::Relu => OpKind::Relu,
        WireOpKind::Exp => OpKind::Exp,
        WireOpKind::Log => OpKind::Log,
        WireOpKind::Sqrt => OpKind::Sqrt,
        WireOpKind::Matmul => OpKind::Matmul,
        WireOpKind::FusedElementwise { kernel_source, function_name } =>
            OpKind::FusedElementwise { kernel_source: kernel_source.clone(), function_name: function_name.clone() },
        WireOpKind::Softmax => OpKind::Softmax,
        WireOpKind::Transpose { dim0, dim1 } => OpKind::Transpose { dim0: *dim0, dim1: *dim1 },
        WireOpKind::ScalarMul(s) => OpKind::ScalarMul(*s),
        WireOpKind::Gelu => OpKind::Gelu,
        WireOpKind::LayerNorm { eps } => OpKind::LayerNorm { eps: *eps },
        WireOpKind::Embedding => OpKind::Embedding,
        WireOpKind::Reshape { new_shape } => OpKind::Reshape { new_shape: new_shape.clone() },
        WireOpKind::Slice { dim, start, end } => OpKind::Slice { dim: *dim, start: *start, end: *end },
        WireOpKind::Concat { dim } => OpKind::Concat { dim: *dim },
        WireOpKind::AddBias => OpKind::AddBias,
        WireOpKind::SoftmaxCausal => OpKind::SoftmaxCausal,
        WireOpKind::Argmax => OpKind::Argmax,
        WireOpKind::Sum => OpKind::Sum,
        WireOpKind::Mean => OpKind::Mean,
        WireOpKind::Abs => OpKind::Abs,
        WireOpKind::Sign => OpKind::Sign,
        WireOpKind::Pow { exponent } => OpKind::Pow { exponent: *exponent },
        WireOpKind::Clamp { min_val, max_val } => OpKind::Clamp { min_val: *min_val, max_val: *max_val },
        WireOpKind::Where => OpKind::Where,
        WireOpKind::MaskedFill { value } => OpKind::MaskedFill { value: *value },
        WireOpKind::Triu { diagonal } => OpKind::Triu { diagonal: *diagonal },
        WireOpKind::Tril { diagonal } => OpKind::Tril { diagonal: *diagonal },
        WireOpKind::Gather { dim } => OpKind::Gather { dim: *dim },
        WireOpKind::IndexSelect { dim } => OpKind::IndexSelect { dim: *dim },
        WireOpKind::Conv1d { stride, padding } => OpKind::Conv1d { stride: *stride, padding: *padding },
        WireOpKind::Conv2d { stride, padding } => OpKind::Conv2d { stride: *stride, padding: *padding },
        WireOpKind::BatchNorm { eps } => OpKind::BatchNorm { eps: *eps },
        WireOpKind::MaxPool2d { kernel_size, stride, padding } => OpKind::MaxPool2d { kernel_size: *kernel_size, stride: *stride, padding: *padding },
        WireOpKind::AvgPool2d { kernel_size, stride, padding } => OpKind::AvgPool2d { kernel_size: *kernel_size, stride: *stride, padding: *padding },
        WireOpKind::Tanh => OpKind::Tanh,
        WireOpKind::SoftmaxBackward => OpKind::SoftmaxBackward,
        WireOpKind::LayerNormBackward { eps } => OpKind::LayerNormBackward { eps: *eps },
        WireOpKind::Conv2dBackwardInput { stride, padding } => OpKind::Conv2dBackwardInput { stride: *stride, padding: *padding },
        WireOpKind::EmbeddingBackward => OpKind::EmbeddingBackward,
        WireOpKind::BatchNormBackward { eps } => OpKind::BatchNormBackward { eps: *eps },
    }
}

/// Convert a WireOpNode to a core OpNode.
pub fn wire_node_to_core(wire: &WireOpNode) -> std::result::Result<OpNode, io::Error> {
    let shape = Shape::new(wire.out_shape.clone())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(OpNode {
        id: wire.id,
        op: wire_op_to_core(&wire.op),
        inputs: wire.inputs.clone(),
        out_shape: shape,
        out_dtype: DType::Float32,
        container_id: ContainerId::DEFAULT,
    })
}
```

- [ ] **Step 3: Run cargo check**

Run: `cargo check -p applegpu-core 2>&1`
Expected: Compiles without errors

- [ ] **Step 4: Run existing serial tests to verify nothing broke**

Run: `cargo test -p applegpu-core serial 2>&1`
Expected: All 5 existing serial tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/core/Cargo.toml crates/core/src/serial.rs
git commit -m "feat: add wire type conversions in serial.rs, link core to wire crate"
```

---

### Task 5: Add cleanup_container to LazyRuntime

**Files:**
- Modify: `crates/core/src/lazy.rs`
- Create: `crates/core/tests/cleanup_integration.rs`

- [ ] **Step 1: Write failing test**

Create `crates/core/tests/cleanup_integration.rs`:

```rust
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::scheduler::{ContainerId, ContainerConfig, Priority};
use applegpu_core::graph::{OpKind, OpNode};
use applegpu_core::tensor::{DType, Shape};

#[test]
fn cleanup_container_removes_tensors_and_nodes() {
    let mut rt = LazyRuntime::new();

    // Register a container
    let config = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 10 * 1024 * 1024,
        max_tensor_count: 100,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let cid = rt.scheduler.register_container(config).unwrap();

    // Record an op for this container
    let node = OpNode {
        id: 100,
        op: OpKind::Add,
        inputs: vec![1, 2],
        out_shape: Shape::new(vec![4]).unwrap(),
        out_dtype: DType::Float32,
        container_id: cid,
    };
    rt.record_op(node);
    assert!(rt.is_pending(100));

    // Cleanup
    rt.cleanup_container(cid).unwrap();

    // Graph node should be gone
    assert!(!rt.is_pending(100));
    // Container should be deregistered
    assert!(rt.scheduler.container_usage(cid).is_none());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-core cleanup_container 2>&1`
Expected: FAIL — `cleanup_container` method not found

- [ ] **Step 3: Implement cleanup_container**

Add to `crates/core/src/lazy.rs`, in the `impl LazyRuntime` block:

```rust
    /// Clean up all resources belonging to a container.
    /// Removes tensors, graph nodes, and deregisters from the scheduler.
    /// Called when a container disconnects (clean or crash).
    pub fn cleanup_container(&mut self, container_id: ContainerId) -> Result<()> {
        // 1. Deregister from scheduler — returns owned tensor IDs
        let owned_tensors = self.scheduler.deregister_container(container_id)?;

        // 2. Remove owned tensors, returning buffers to pool
        for tid in &owned_tensors {
            self.remove_tensor_raw(*tid);
        }

        // 3. Remove pending graph nodes for this container
        self.graph.remove_nodes_for_container(container_id);

        Ok(())
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p applegpu-core cleanup_container 2>&1`
Expected: 1 test passes

- [ ] **Step 5: Run all core tests to verify nothing broke**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All existing tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/lazy.rs crates/core/tests/cleanup_integration.rs
git commit -m "feat: add cleanup_container to LazyRuntime for session disconnect"
```

---

## Chunk 3: Multi-Client GPU Service

### Task 6: Rewrite gpu-service with thread-per-connection and handshake

**Files:**
- Modify: `crates/gpu-service/Cargo.toml`
- Modify: `crates/gpu-service/src/main.rs`

- [ ] **Step 1: Add wire dependency to gpu-service**

Update `crates/gpu-service/Cargo.toml`:

```toml
[package]
name = "applegpu-service"
version.workspace = true
edition.workspace = true

[[bin]]
name = "gpu-service"
path = "src/main.rs"

[dependencies]
applegpu-core = { path = "../core" }
applegpu-wire = { path = "../wire" }
```

- [ ] **Step 2: Rewrite main.rs**

Replace `crates/gpu-service/src/main.rs` with:

```rust
use std::io::{Read, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};
use std::thread;

use applegpu_core::buffer::Buffer;
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::scheduler::{ContainerId, ContainerConfig, Priority};
use applegpu_core::serial::wire_node_to_core;
use applegpu_core::tensor::Tensor;

use applegpu_wire::{
    self as wire,
    EvalRequest, EvalResponse, HandshakeRequest, HandshakeResponse,
    HANDSHAKE_OK, HANDSHAKE_REJECTED_QUOTA, HANDSHAKE_REJECTED_CAPACITY,
    PROTOCOL_VERSION, MAX_MESSAGE_SIZE,
};

struct SharedState {
    runtime: Mutex<LazyRuntime>,
    device: Device,
}

fn handle_handshake(
    shared: &SharedState,
    stream: &mut UnixStream,
) -> Option<ContainerId> {
    // Read handshake request
    let msg = match wire::read_message(stream, 1024) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Handshake read failed: {}", e);
            return None;
        }
    };
    let req = match HandshakeRequest::deserialize(&msg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Handshake parse failed: {}", e);
            return None;
        }
    };

    if req.protocol_version != PROTOCOL_VERSION {
        eprintln!("Unsupported protocol version: {}", req.protocol_version);
        let resp = HandshakeResponse {
            status: HANDSHAKE_REJECTED_CAPACITY,
            container_id: 0,
            granted_memory: 0,
        };
        let _ = wire::write_message(stream, &resp.serialize());
        return None;
    }

    // Register container with scheduler
    let mut rt = shared.runtime.lock().unwrap();
    let default_memory = rt.scheduler.global_limits.max_total_memory_bytes;
    let requested = if req.requested_memory == 0 {
        default_memory / 4 // Default: 25% of global
    } else {
        req.requested_memory as usize
    };

    let config = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: requested,
        max_tensor_count: rt.scheduler.global_limits.max_tensor_count / 4,
        max_tensor_size_bytes: 0, // inherit from global
        max_pending_jobs: 64,
    };

    match rt.scheduler.register_container(config) {
        Ok(cid) => {
            let resp = HandshakeResponse {
                status: HANDSHAKE_OK,
                container_id: cid.0,
                granted_memory: requested as u64,
            };
            drop(rt); // release lock before I/O
            if wire::write_message(stream, &resp.serialize()).is_err() {
                return None;
            }
            println!("Container {} registered (memory: {} bytes)", cid, requested);
            Some(cid)
        }
        Err(e) => {
            let resp = HandshakeResponse {
                status: HANDSHAKE_REJECTED_QUOTA,
                container_id: 0,
                granted_memory: 0,
            };
            drop(rt);
            let _ = wire::write_message(stream, &resp.serialize());
            eprintln!("Container registration failed: {}", e);
            None
        }
    }
}

fn handle_eval(
    shared: &SharedState,
    container_id: ContainerId,
    request: &EvalRequest,
) -> EvalResponse {
    let mut rt = shared.runtime.lock().unwrap();

    // Load input tensors for this container
    for td in &request.tensors {
        match Buffer::from_bytes(&shared.device, &td.data) {
            Ok(buffer) => {
                let tensor = Tensor::from_raw(
                    td.id,
                    td.shape.clone(),
                    applegpu_core::tensor::DType::Float32,
                    buffer,
                );
                if let Err(e) = rt.insert_tensor_for(tensor, container_id) {
                    return EvalResponse::Err(format!("Insert tensor failed: {}", e));
                }
            }
            Err(e) => return EvalResponse::Err(format!("Buffer allocation failed: {}", e)),
        }
    }

    // Load graph nodes — stamp container_id
    for wire_node in &request.nodes {
        match wire_node_to_core(wire_node) {
            Ok(mut node) => {
                node.container_id = container_id;
                rt.record_op(node);
            }
            Err(e) => return EvalResponse::Err(format!("Node conversion failed: {}", e)),
        }
    }

    // Evaluate
    if let Err(e) = rt.eval(&shared.device, request.target_id) {
        return EvalResponse::Err(format!("Eval failed: {}", e));
    }

    // Read result
    match rt.read_f32(request.target_id) {
        Ok(data) => {
            let shape = rt.shape(request.target_id).unwrap_or_default();
            let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            EvalResponse::Ok {
                tensor_id: request.target_id,
                shape,
                data: data_bytes,
            }
        }
        Err(e) => EvalResponse::Err(format!("Read failed: {}", e)),
    }
}

fn handle_connection(shared: Arc<SharedState>, mut stream: UnixStream) {
    // 1. Handshake
    let container_id = match handle_handshake(&shared, &mut stream) {
        Some(cid) => cid,
        None => return,
    };

    // 2. Request-response loop
    loop {
        // Read eval request
        let msg = match wire::read_message(&mut stream, MAX_MESSAGE_SIZE) {
            Ok(m) => m,
            Err(_) => break, // client disconnected or error
        };

        let response = match EvalRequest::deserialize(&msg) {
            Ok(request) => handle_eval(&shared, container_id, &request),
            Err(e) => EvalResponse::Err(format!("Deserialization failed: {}", e)),
        };

        // Send response
        if wire::write_message(&mut stream, &response.serialize()).is_err() {
            break;
        }
    }

    // 3. Cleanup on disconnect
    println!("Container {} disconnected, cleaning up", container_id);
    let mut rt = shared.runtime.lock().unwrap();
    if let Err(e) = rt.cleanup_container(container_id) {
        eprintln!("Cleanup failed for {}: {}", container_id, e);
    }
}

fn main() {
    let socket_path = std::env::var("APPLEGPU_SOCKET")
        .unwrap_or_else(|_| applegpu_core::ipc::default_socket_path());

    // Ensure socket directory exists and remove stale socket
    if let Some(parent) = std::path::Path::new(&socket_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_file(&socket_path);

    let device = Device::new().expect("No Metal GPU available");
    println!("GPU service started on {} (device: {})", socket_path, device.name());

    let shared = Arc::new(SharedState {
        runtime: Mutex::new(LazyRuntime::new()),
        device,
    });

    let listener = UnixListener::bind(&socket_path)
        .expect("Failed to bind Unix socket");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let shared = shared.clone();
                thread::spawn(move || handle_connection(shared, stream));
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }
}
```

- [ ] **Step 3: Build the service**

Run: `cargo build -p applegpu-service 2>&1`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add crates/gpu-service/Cargo.toml crates/gpu-service/src/main.rs
git commit -m "feat: rewrite gpu-service with thread-per-connection, handshake, shared LazyRuntime"
```

---

### Task 7: Multi-client integration test

**Files:**
- Create: `crates/core/tests/multi_client_integration.rs`

- [ ] **Step 1: Write integration test**

Create `crates/core/tests/multi_client_integration.rs`:

```rust
//! Tests the wire protocol end-to-end without the GPU service binary.
//! Uses Unix socketpairs to simulate the client-server protocol.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::thread;

use applegpu_wire::{
    self as wire,
    EvalRequest, EvalResponse, WireOpNode, WireOpKind, WireTensorData,
    HandshakeRequest, HandshakeResponse,
    HANDSHAKE_OK, PROTOCOL_VERSION, MAX_MESSAGE_SIZE,
};

fn mock_server(mut stream: UnixStream) {
    // Read handshake
    let msg = wire::read_message(&mut stream, 1024).unwrap();
    let req = HandshakeRequest::deserialize(&msg).unwrap();
    assert_eq!(req.protocol_version, PROTOCOL_VERSION);

    // Send handshake response
    let resp = HandshakeResponse {
        status: HANDSHAKE_OK,
        container_id: 42,
        granted_memory: req.requested_memory,
    };
    wire::write_message(&mut stream, &resp.serialize()).unwrap();

    // Read eval request
    let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
    let eval_req = EvalRequest::deserialize(&msg).unwrap();
    assert_eq!(eval_req.target_id, 100);

    // Send eval response (fake result)
    let data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
        .iter().flat_map(|f| f.to_le_bytes()).collect();
    let eval_resp = EvalResponse::Ok {
        tensor_id: 100,
        shape: vec![4],
        data,
    };
    wire::write_message(&mut stream, &eval_resp.serialize()).unwrap();
}

#[test]
fn wire_protocol_handshake_and_eval() {
    let (client_stream, server_stream) = UnixStream::pair().unwrap();

    let server = thread::spawn(move || mock_server(server_stream));

    let mut client = client_stream;

    // Client: send handshake
    let hs = HandshakeRequest {
        protocol_version: PROTOCOL_VERSION,
        requested_memory: 1024 * 1024,
    };
    wire::write_message(&mut client, &hs.serialize()).unwrap();

    // Client: read handshake response
    let msg = wire::read_message(&mut client, 1024).unwrap();
    let hs_resp = HandshakeResponse::deserialize(&msg).unwrap();
    assert_eq!(hs_resp.status, HANDSHAKE_OK);
    assert_eq!(hs_resp.container_id, 42);

    // Client: send eval request
    let data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
        .iter().flat_map(|f| f.to_le_bytes()).collect();
    let eval_req = EvalRequest {
        target_id: 100,
        tensors: vec![WireTensorData { id: 1, shape: vec![4], dtype: 0, data }],
        nodes: vec![WireOpNode {
            id: 100, op: WireOpKind::Neg, inputs: vec![1],
            out_shape: vec![4], out_dtype: 0,
        }],
    };
    wire::write_message(&mut client, &eval_req.serialize()).unwrap();

    // Client: read eval response
    let msg = wire::read_message(&mut client, MAX_MESSAGE_SIZE).unwrap();
    let eval_resp = EvalResponse::deserialize(&msg).unwrap();
    match eval_resp {
        EvalResponse::Ok { tensor_id, shape, data } => {
            assert_eq!(tensor_id, 100);
            assert_eq!(shape, vec![4]);
            let floats: Vec<f32> = data.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(floats, vec![1.0, 2.0, 3.0, 4.0]);
        }
        _ => panic!("Expected Ok response"),
    }

    server.join().unwrap();
}

#[test]
fn wire_protocol_multiple_concurrent_clients() {
    let num_clients = 4;
    let mut handles = Vec::new();

    for client_idx in 0..num_clients {
        let (client_stream, server_stream) = UnixStream::pair().unwrap();

        // Server thread for this client
        let server_handle = thread::spawn(move || {
            let mut stream = server_stream;
            // Handshake
            let msg = wire::read_message(&mut stream, 1024).unwrap();
            let _ = HandshakeRequest::deserialize(&msg).unwrap();
            let resp = HandshakeResponse {
                status: HANDSHAKE_OK,
                container_id: client_idx as u64 + 1,
                granted_memory: 1024 * 1024,
            };
            wire::write_message(&mut stream, &resp.serialize()).unwrap();

            // Handle 10 eval requests
            for _ in 0..10 {
                let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
                let _ = EvalRequest::deserialize(&msg).unwrap();
                let resp = EvalResponse::Ok {
                    tensor_id: 1,
                    shape: vec![2],
                    data: vec![0; 8],
                };
                wire::write_message(&mut stream, &resp.serialize()).unwrap();
            }
        });

        // Client thread
        let client_handle = thread::spawn(move || {
            let mut stream = client_stream;
            // Handshake
            let hs = HandshakeRequest {
                protocol_version: PROTOCOL_VERSION,
                requested_memory: 1024 * 1024,
            };
            wire::write_message(&mut stream, &hs.serialize()).unwrap();
            let msg = wire::read_message(&mut stream, 1024).unwrap();
            let resp = HandshakeResponse::deserialize(&msg).unwrap();
            assert_eq!(resp.status, HANDSHAKE_OK);

            // Send 10 eval requests
            for i in 0..10 {
                let req = EvalRequest {
                    target_id: i,
                    tensors: vec![],
                    nodes: vec![WireOpNode {
                        id: i, op: WireOpKind::Add, inputs: vec![],
                        out_shape: vec![2], out_dtype: 0,
                    }],
                };
                wire::write_message(&mut stream, &req.serialize()).unwrap();
                let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
                let resp = EvalResponse::deserialize(&msg).unwrap();
                assert!(matches!(resp, EvalResponse::Ok { .. }));
            }
        });

        handles.push(server_handle);
        handles.push(client_handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}
```

- [ ] **Step 2: Add container ID stamping test**

Add to the same file (`crates/core/tests/multi_client_integration.rs`):

```rust
/// Verify that the gpu-service stamps container_id on incoming nodes.
/// This tests the spec requirement: "The service rewrites node.container_id
/// to the session's assigned ContainerId before inserting into the shared graph."
#[test]
fn container_id_stamping() {
    use applegpu_core::graph::{OpKind, OpNode};
    use applegpu_core::lazy::LazyRuntime;
    use applegpu_core::scheduler::{ContainerId, ContainerConfig, Priority};
    use applegpu_core::serial::wire_node_to_core;
    use applegpu_wire::WireOpNode;

    let mut rt = LazyRuntime::new();

    // Register a container (simulates what the service does on handshake)
    let config = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 10 * 1024 * 1024,
        max_tensor_count: 100,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let cid = rt.scheduler.register_container(config).unwrap();
    assert_ne!(cid, ContainerId::DEFAULT);

    // Simulate receiving a wire node (container_id will be DEFAULT after conversion)
    let wire_node = WireOpNode {
        id: 200,
        op: applegpu_wire::WireOpKind::Add,
        inputs: vec![1, 2],
        out_shape: vec![4],
        out_dtype: 0,
    };
    let mut node = wire_node_to_core(&wire_node).unwrap();

    // Before stamping: should be DEFAULT
    assert_eq!(node.container_id, ContainerId::DEFAULT);

    // Stamp with session's container ID (what gpu-service does)
    node.container_id = cid;
    rt.record_op(node);

    // Verify the node in the graph has the stamped container ID
    let graph_node = rt.graph_node(200).unwrap();
    assert_eq!(graph_node.container_id, cid);
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p applegpu-core multi_client container_id_stamping 2>&1`
Expected: 3 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/core/tests/multi_client_integration.rs
git commit -m "test: add multi-client wire protocol and container ID stamping tests"
```

---

### Note: Swift vsock FFI (deferred)

The spec defines three Swift C ABI functions for vsock (`gpu_bridge_vsock_create_listener`, `gpu_bridge_vsock_accept`, `gpu_bridge_vsock_destroy_listener`). These are **deferred from this plan** because:
- `VZVirtioSocketListener` requires an active `VZVirtualMachine` instance to attach to
- We are not managing VM lifecycle — the container framework provides VMs
- Without a VM instance, the Swift code cannot be tested end-to-end
- The vsock transport on the client side (Linux `AF_VSOCK`) is implemented and testable via Unix socketpairs

The Swift vsock implementation is tracked in the backlog as "Apple Containerization Framework integration." When a VM instance is available (from the container framework), the Swift vsock listener code will be added and connected to the gpu-service's vsock listener thread.

---

## Chunk 4: Client Crate

### Task 8: Create `crates/client` with GpuClient

**Files:**
- Create: `crates/client/Cargo.toml`
- Create: `crates/client/src/lib.rs`
- Create: `crates/client/src/transport.rs`
- Modify: `Cargo.toml` (workspace)

- [ ] **Step 1: Create client crate Cargo.toml**

Create `crates/client/Cargo.toml`:

```toml
[package]
name = "applegpu-client"
version.workspace = true
edition.workspace = true

[lib]
name = "applegpu_client"

[dependencies]
applegpu-wire = { path = "../wire" }

[target.'cfg(unix)'.dependencies]
libc = "0.2"
```

- [ ] **Step 2: Add client to workspace**

In root `Cargo.toml`, update members:

```toml
[workspace]
members = ["crates/core", "crates/python", "crates/gpu-service", "crates/wire", "crates/client"]
```

- [ ] **Step 3: Create transport.rs**

Create `crates/client/src/transport.rs`:

```rust
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;

/// Trait for a bidirectional byte stream transport.
pub trait Transport: Read + Write + Send {}

impl Transport for UnixStream {}

/// Connect via Unix domain socket.
pub fn connect_unix(path: &str) -> io::Result<UnixStream> {
    UnixStream::connect(path)
}

/// Connect via vsock (AF_VSOCK). Only available on Linux.
/// CID 2 = host. Returns a UnixStream wrapping the vsock fd.
#[cfg(target_os = "linux")]
pub fn connect_vsock(cid: u32, port: u32) -> io::Result<UnixStream> {
    use std::os::unix::io::FromRawFd;

    // AF_VSOCK = 40, SOCK_STREAM = 1
    let fd = unsafe { libc::socket(40, libc::SOCK_STREAM, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }

    #[repr(C)]
    struct SockaddrVm {
        svm_family: u16,
        svm_reserved1: u16,
        svm_port: u32,
        svm_cid: u32,
        svm_zero: [u8; 4],
    }

    let addr = SockaddrVm {
        svm_family: 40, // AF_VSOCK
        svm_reserved1: 0,
        svm_port: port,
        svm_cid: cid,
        svm_zero: [0; 4],
    };

    let ret = unsafe {
        libc::connect(
            fd,
            &addr as *const SockaddrVm as *const libc::sockaddr,
            std::mem::size_of::<SockaddrVm>() as u32,
        )
    };
    if ret < 0 {
        unsafe { libc::close(fd) };
        return Err(io::Error::last_os_error());
    }

    let stream = unsafe { UnixStream::from_raw_fd(fd) };
    Ok(stream)
}

#[cfg(not(target_os = "linux"))]
pub fn connect_vsock(_cid: u32, _port: u32) -> io::Result<UnixStream> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "vsock is only available on Linux guests",
    ))
}
```

- [ ] **Step 4: Create lib.rs with GpuClient**

Create `crates/client/src/lib.rs`:

```rust
pub mod transport;

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;

use applegpu_wire::{
    self as wire,
    EvalRequest, EvalResponse, HandshakeRequest, HandshakeResponse,
    HANDSHAKE_OK, PROTOCOL_VERSION, MAX_MESSAGE_SIZE,
};

/// Error type for GPU client operations.
#[derive(Debug)]
pub enum ClientError {
    Io(io::Error),
    HandshakeRejected { status: u32 },
    Protocol(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::Io(e) => write!(f, "I/O error: {}", e),
            ClientError::HandshakeRejected { status } => write!(f, "Handshake rejected (status {})", status),
            ClientError::Protocol(msg) => write!(f, "Protocol error: {}", msg),
        }
    }
}

impl std::error::Error for ClientError {}

impl From<io::Error> for ClientError {
    fn from(e: io::Error) -> Self { ClientError::Io(e) }
}

pub type Result<T> = std::result::Result<T, ClientError>;

/// Default vsock port for the GPU service.
pub const DEFAULT_VSOCK_PORT: u32 = 5678;

/// GPU client for submitting compute workloads to the host GPU service.
pub struct GpuClient {
    stream: UnixStream,
    pub container_id: u64,
    pub granted_memory: u64,
}

impl GpuClient {
    /// Connect to the GPU service via Unix socket and perform handshake.
    pub fn connect_unix(path: &str, requested_memory: u64) -> Result<Self> {
        let stream = transport::connect_unix(path)?;
        Self::handshake(stream, requested_memory)
    }

    /// Connect to the GPU service via vsock and perform handshake.
    /// CID 2 = host. Only works on Linux guests.
    pub fn connect_vsock(port: u32, requested_memory: u64) -> Result<Self> {
        let stream = transport::connect_vsock(2, port)?;
        Self::handshake(stream, requested_memory)
    }

    fn handshake(mut stream: UnixStream, requested_memory: u64) -> Result<Self> {
        let hs = HandshakeRequest {
            protocol_version: PROTOCOL_VERSION,
            requested_memory,
        };
        wire::write_message(&mut stream, &hs.serialize())?;

        let msg = wire::read_message(&mut stream, 1024)?;
        let resp = HandshakeResponse::deserialize(&msg)?;

        if resp.status != HANDSHAKE_OK {
            return Err(ClientError::HandshakeRejected { status: resp.status });
        }

        Ok(GpuClient {
            stream,
            container_id: resp.container_id,
            granted_memory: resp.granted_memory,
        })
    }

    /// Submit an eval request and block for the result.
    pub fn eval(&mut self, request: &EvalRequest) -> Result<EvalResponse> {
        wire::write_message(&mut self.stream, &request.serialize())?;
        let msg = wire::read_message(&mut self.stream, MAX_MESSAGE_SIZE)?;
        EvalResponse::deserialize(&msg).map_err(|e| ClientError::Protocol(e.to_string()))
    }
}

impl Drop for GpuClient {
    fn drop(&mut self) {
        // Closing the stream triggers server-side cleanup
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn mock_gpu_service(mut stream: UnixStream) {
        // Handshake
        let msg = wire::read_message(&mut stream, 1024).unwrap();
        let req = HandshakeRequest::deserialize(&msg).unwrap();
        let resp = HandshakeResponse {
            status: HANDSHAKE_OK,
            container_id: 7,
            granted_memory: req.requested_memory,
        };
        wire::write_message(&mut stream, &resp.serialize()).unwrap();

        // One eval request
        let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
        let eval_req = EvalRequest::deserialize(&msg).unwrap();
        let result: Vec<u8> = vec![42.0f32]
            .iter().flat_map(|f| f.to_le_bytes()).collect();
        let eval_resp = EvalResponse::Ok {
            tensor_id: eval_req.target_id,
            shape: vec![1],
            data: result,
        };
        wire::write_message(&mut stream, &eval_resp.serialize()).unwrap();
    }

    #[test]
    fn gpu_client_connect_and_eval() {
        let (client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || mock_gpu_service(server_stream));

        let mut client = GpuClient::handshake(client_stream, 1024 * 1024).unwrap();
        assert_eq!(client.container_id, 7);

        let req = EvalRequest {
            target_id: 1,
            tensors: vec![],
            nodes: vec![],
        };
        let resp = client.eval(&req).unwrap();
        match resp {
            EvalResponse::Ok { tensor_id, .. } => assert_eq!(tensor_id, 1),
            _ => panic!("Expected Ok"),
        }

        server.join().unwrap();
    }

    #[test]
    fn gpu_client_handshake_rejected() {
        let (client_stream, mut server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let msg = wire::read_message(&mut server_stream, 1024).unwrap();
            let _ = HandshakeRequest::deserialize(&msg).unwrap();
            let resp = HandshakeResponse {
                status: 1, // rejected
                container_id: 0,
                granted_memory: 0,
            };
            wire::write_message(&mut server_stream, &resp.serialize()).unwrap();
        });

        let result = GpuClient::handshake(client_stream, 1024);
        assert!(result.is_err());

        server.join().unwrap();
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p applegpu-client 2>&1`
Expected: 2 tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/client/ Cargo.toml
git commit -m "feat: add client crate with GpuClient, Unix socket + vsock transport"
```

---

### Task 9: Run full test suite and verify backward compatibility

- [ ] **Step 1: Run all Rust tests**

Run: `cargo test -p applegpu-wire -p applegpu-core -p applegpu-client 2>&1`
Expected: All tests pass (wire: ~9, core: ~266+, client: ~2)

- [ ] **Step 2: Build the gpu-service**

Run: `cargo build -p applegpu-service 2>&1`
Expected: Build succeeds

- [ ] **Step 3: Run Python tests to verify backward compatibility**

Run: `uv run maturin develop && uv run pytest -v 2>&1`
Expected: All Python tests pass (MLX backend unaffected)

- [ ] **Step 4: Commit any remaining changes**

If any fixes were needed, commit them.

---

### Task 10: Update backlog and README

- [ ] **Step 1: Update backlog**

Mark Phase 7b items as complete in `docs/BACKLOG.md`:
- Wire protocol crate
- Multi-client GPU service
- Container client library
- Handshake protocol

Add remaining backlog items from the spec (ACF integration, Docker docs, client Python bindings, finer-grained locking, read timeouts, connection limits).

- [ ] **Step 2: Update README**

Add a "Container GPU Bridge" section to `README.md` showing:
- How to start the GPU service: `cargo run -p applegpu-service`
- How containers connect (Unix socket path)
- Architecture diagram update

- [ ] **Step 3: Commit**

```bash
git add docs/BACKLOG.md README.md
git commit -m "docs: update backlog and README for Phase 7b container GPU bridge"
```
