# Phase 7a: VM Backend — Graph Serialization, IPC, and Backend Routing

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the VM backend path where `APPLEGPU_BACKEND=vm` routes GPU operations through IPC to a separate host GPU service process, which deserializes the computation graph and executes it on Metal — proving the full remote execution round-trip works.

**Architecture:** A new `serde` module serializes `OpNode` graphs and tensor data to a binary format. A new `ipc` module communicates over Unix domain sockets between client (container) and server (host GPU service). When the VM backend is selected, `LazyRuntime::eval()` serializes the pending graph + input tensor data, sends it over IPC, and receives the materialized result back. The GPU service is a standalone Rust binary that listens for requests, deserializes them, and executes using the existing `LazyRuntime` + `KernelRegistry` on the host's Metal GPU. This validates the two-backend architecture without requiring actual AVF VM infrastructure (which is a follow-up plan).

**Tech Stack:** Rust (serde-like binary serialization, Unix domain sockets, `std::os::unix::net`), existing Metal compute infrastructure (unchanged)

**Scope:** This plan covers serialization, IPC protocol, the GPU service binary, and backend routing. Actual Apple Virtualization Framework VM creation is Phase 7b.

**Design notes:**
- Kernel fusion is NOT run client-side for the VM backend — graphs are sent unfused. The GPU service's `eval()` runs the fusion pass server-side, which is correct since the server compiles Metal kernels.
- The GPU service is single-threaded (one request at a time) for this proof-of-concept. Concurrent VM clients are a Phase 7b concern.
- Socket path defaults to `$HOME/.applegpu/runtime.sock` (not `/tmp`, for security).

---

## File Structure

### New Files — Rust Core
- `crates/core/src/serial.rs` — Serialize/deserialize OpNodes, tensor data, and eval requests to binary format
- `crates/core/src/ipc.rs` — IPC client (send request, receive result) over Unix domain sockets

### New Files — GPU Service Binary
- `crates/gpu-service/Cargo.toml` — Standalone binary crate for the host GPU service
- `crates/gpu-service/src/main.rs` — Listens on Unix socket, deserializes requests, executes on Metal, returns results

### Modified Files
- `crates/core/src/lazy.rs` — Add `eval_remote()` method that serializes and sends over IPC
- `crates/core/src/backend.rs` — Route eval to local or remote based on Backend selection
- `crates/core/src/graph.rs` — (no changes needed — existing API sufficient)
- `crates/core/src/lib.rs` — Add serial and ipc modules
- `crates/core/Cargo.toml` — No new deps needed (pure std)
- `Cargo.toml` — Add gpu-service to workspace members

### Test Files
- `crates/core/tests/serial_integration.rs` — Round-trip serialization tests
- `crates/core/tests/ipc_integration.rs` — IPC client-server integration test
- `python/tests/test_vm_backend.py` — Python end-to-end test with VM backend

---

## Chunk 1: Binary Serialization

### Task 1: Implement binary serialization for graph and tensor data

**Files:**
- Create: `crates/core/src/serial.rs`
- Modify: `crates/core/src/lib.rs`

The serialization format is simple and custom (no serde dependency — keeping deps minimal):

**Wire format for an EvalRequest:**
```
[4 bytes] magic: b"AGPU"
[4 bytes] version: 1u32
[8 bytes] target_id: u64
[4 bytes] num_tensors: u32
  for each tensor:
    [8 bytes] id: u64
    [4 bytes] num_dims: u32
    [num_dims * 8 bytes] dims: [u64]
    [4 bytes] dtype: u32 (0=f32)
    [4 bytes] data_len: u32 (in bytes)
    [data_len bytes] data
[4 bytes] num_nodes: u32
  for each node:
    [8 bytes] id: u64
    [4 bytes] op_type: u32 (enum discriminant)
    [4 bytes] num_inputs: u32
    [num_inputs * 8 bytes] inputs: [u64]
    [4 bytes] num_out_dims: u32
    [num_out_dims * 8 bytes] out_shape: [u64]
    [4 bytes] out_dtype: u32
    // if op_type == FUSED (10):
    [4 bytes] kernel_source_len: u32
    [kernel_source_len bytes] kernel_source
    [4 bytes] function_name_len: u32
    [function_name_len bytes] function_name
```

**Wire format for an EvalResponse:**
```
[4 bytes] magic: b"AGPR"
[4 bytes] status: 0=ok, 1=error
if ok:
  [8 bytes] tensor_id: u64
  [4 bytes] num_dims: u32
  [num_dims * 8 bytes] dims: [u64]
  [4 bytes] data_len: u32
  [data_len bytes] data (f32 values)
if error:
  [4 bytes] msg_len: u32
  [msg_len bytes] error message (UTF-8)
```

- [ ] **Step 1: Create serial.rs with serialize/deserialize**

Create `crates/core/src/serial.rs`:

```rust
use std::io::{self, Read, Write, Cursor};

use crate::graph::{OpKind, OpNode};
use crate::tensor::{DType, Shape};

const MAGIC_REQUEST: &[u8; 4] = b"AGPU";
const MAGIC_RESPONSE: &[u8; 4] = b"AGPR";
const VERSION: u32 = 1;

// Op type discriminants
const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_DIV: u32 = 3;
const OP_NEG: u32 = 4;
const OP_RELU: u32 = 5;
const OP_EXP: u32 = 6;
const OP_LOG: u32 = 7;
const OP_SQRT: u32 = 8;
const OP_MATMUL: u32 = 9;
const OP_FUSED: u32 = 10;

/// A serializable tensor (ID + shape + data).
#[derive(Debug, Clone)]
pub struct TensorData {
    pub id: u64,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

/// A request to evaluate a computation graph remotely.
#[derive(Debug)]
pub struct EvalRequest {
    pub target_id: u64,
    pub tensors: Vec<TensorData>,
    pub nodes: Vec<OpNode>,
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

// Helper: write/read primitives
fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn read_u32(r: &mut impl Read) -> io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn read_u64(r: &mut impl Read) -> io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }

fn op_to_discriminant(op: &OpKind) -> u32 {
    match op {
        OpKind::Add => OP_ADD,
        OpKind::Sub => OP_SUB,
        OpKind::Mul => OP_MUL,
        OpKind::Div => OP_DIV,
        OpKind::Neg => OP_NEG,
        OpKind::Relu => OP_RELU,
        OpKind::Exp => OP_EXP,
        OpKind::Log => OP_LOG,
        OpKind::Sqrt => OP_SQRT,
        OpKind::Matmul => OP_MATMUL,
        OpKind::FusedElementwise { .. } => OP_FUSED,
    }
}

fn discriminant_to_op(d: u32, r: &mut impl Read) -> io::Result<OpKind> {
    match d {
        OP_ADD => Ok(OpKind::Add),
        OP_SUB => Ok(OpKind::Sub),
        OP_MUL => Ok(OpKind::Mul),
        OP_DIV => Ok(OpKind::Div),
        OP_NEG => Ok(OpKind::Neg),
        OP_RELU => Ok(OpKind::Relu),
        OP_EXP => Ok(OpKind::Exp),
        OP_LOG => Ok(OpKind::Log),
        OP_SQRT => Ok(OpKind::Sqrt),
        OP_MATMUL => Ok(OpKind::Matmul),
        OP_FUSED => {
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

            Ok(OpKind::FusedElementwise { kernel_source, function_name })
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unknown op type: {}", d))),
    }
}

impl EvalRequest {
    /// Serialize to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_REQUEST).unwrap();
        write_u32(&mut buf, VERSION).unwrap();
        write_u64(&mut buf, self.target_id).unwrap();

        // Tensors
        write_u32(&mut buf, self.tensors.len() as u32).unwrap();
        for t in &self.tensors {
            write_u64(&mut buf, t.id).unwrap();
            write_u32(&mut buf, t.shape.len() as u32).unwrap();
            for &d in &t.shape {
                write_u64(&mut buf, d as u64).unwrap();
            }
            write_u32(&mut buf, 0).unwrap(); // dtype: 0 = f32
            write_u32(&mut buf, t.data.len() as u32).unwrap();
            buf.write_all(&t.data).unwrap();
        }

        // Nodes
        write_u32(&mut buf, self.nodes.len() as u32).unwrap();
        for node in &self.nodes {
            write_u64(&mut buf, node.id).unwrap();
            let disc = op_to_discriminant(&node.op);
            write_u32(&mut buf, disc).unwrap();
            write_u32(&mut buf, node.inputs.len() as u32).unwrap();
            for &inp in &node.inputs {
                write_u64(&mut buf, inp).unwrap();
            }
            write_u32(&mut buf, node.out_shape.dims().len() as u32).unwrap();
            for &d in node.out_shape.dims() {
                write_u64(&mut buf, d as u64).unwrap();
            }
            write_u32(&mut buf, 0).unwrap(); // out_dtype: 0 = f32

            if let OpKind::FusedElementwise { ref kernel_source, ref function_name } = node.op {
                write_u32(&mut buf, kernel_source.len() as u32).unwrap();
                buf.write_all(kernel_source.as_bytes()).unwrap();
                write_u32(&mut buf, function_name.len() as u32).unwrap();
                buf.write_all(function_name.as_bytes()).unwrap();
            }
        }

        buf
    }

    /// Deserialize from bytes.
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_REQUEST {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad magic"));
        }
        let version = read_u32(&mut r)?;
        if version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad version"));
        }

        let target_id = read_u64(&mut r)?;

        let num_tensors = read_u32(&mut r)? as usize;
        let mut tensors = Vec::with_capacity(num_tensors);
        for _ in 0..num_tensors {
            let id = read_u64(&mut r)?;
            let num_dims = read_u32(&mut r)? as usize;
            let mut shape = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                shape.push(read_u64(&mut r)? as usize);
            }
            let _dtype = read_u32(&mut r)?; // 0 = f32
            let data_len = read_u32(&mut r)? as usize;
            let mut data = vec![0u8; data_len];
            r.read_exact(&mut data)?;
            tensors.push(TensorData { id, shape, dtype: DType::Float32, data });
        }

        let num_nodes = read_u32(&mut r)? as usize;
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let id = read_u64(&mut r)?;
            let op_disc = read_u32(&mut r)?;
            let num_inputs = read_u32(&mut r)? as usize;
            let mut inputs = Vec::with_capacity(num_inputs);
            for _ in 0..num_inputs {
                inputs.push(read_u64(&mut r)?);
            }
            let num_out_dims = read_u32(&mut r)? as usize;
            let mut out_shape = Vec::with_capacity(num_out_dims);
            for _ in 0..num_out_dims {
                out_shape.push(read_u64(&mut r)? as usize);
            }
            let _out_dtype = read_u32(&mut r)?; // 0 = f32

            let op = discriminant_to_op(op_disc, &mut r)?;
            nodes.push(OpNode {
                id,
                op,
                inputs,
                out_shape: Shape::new(out_shape),
                out_dtype: DType::Float32,
            });
        }

        Ok(EvalRequest { target_id, tensors, nodes })
    }
}

impl EvalResponse {
    /// Serialize to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_RESPONSE).unwrap();
        match self {
            EvalResponse::Ok { tensor_id, shape, data } => {
                write_u32(&mut buf, 0).unwrap(); // status ok
                write_u64(&mut buf, *tensor_id).unwrap();
                write_u32(&mut buf, shape.len() as u32).unwrap();
                for &d in shape {
                    write_u64(&mut buf, d as u64).unwrap();
                }
                write_u32(&mut buf, data.len() as u32).unwrap();
                buf.write_all(data).unwrap();
            }
            EvalResponse::Err(msg) => {
                write_u32(&mut buf, 1).unwrap(); // status error
                let bytes = msg.as_bytes();
                write_u32(&mut buf, bytes.len() as u32).unwrap();
                buf.write_all(bytes).unwrap();
            }
        }
        buf
    }

    /// Deserialize from bytes.
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
            for _ in 0..num_dims {
                shape.push(read_u64(&mut r)? as usize);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_roundtrip() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let req = EvalRequest {
            target_id: 42,
            tensors: vec![
                TensorData { id: 1, shape: vec![4], dtype: DType::Float32, data: bytes },
            ],
            nodes: vec![
                OpNode {
                    id: 42,
                    op: OpKind::Neg,
                    inputs: vec![1],
                    out_shape: Shape::new(vec![4]),
                    out_dtype: DType::Float32,
                },
            ],
        };

        let serialized = req.serialize();
        let deserialized = EvalRequest::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.target_id, 42);
        assert_eq!(deserialized.tensors.len(), 1);
        assert_eq!(deserialized.tensors[0].id, 1);
        assert_eq!(deserialized.tensors[0].shape, vec![4]);
        assert_eq!(deserialized.nodes.len(), 1);
        assert_eq!(deserialized.nodes[0].id, 42);
        assert!(deserialized.nodes[0].op.is_unary());
    }

    #[test]
    fn request_roundtrip_with_fused() {
        let req = EvalRequest {
            target_id: 10,
            tensors: vec![],
            nodes: vec![
                OpNode {
                    id: 10,
                    op: OpKind::FusedElementwise {
                        kernel_source: "kernel void test() {}".to_string(),
                        function_name: "test".to_string(),
                    },
                    inputs: vec![1, 2],
                    out_shape: Shape::new(vec![4]),
                    out_dtype: DType::Float32,
                },
            ],
        };

        let serialized = req.serialize();
        let deserialized = EvalRequest::deserialize(&serialized).unwrap();

        assert!(deserialized.nodes[0].op.is_fused());
        if let OpKind::FusedElementwise { ref kernel_source, ref function_name } = deserialized.nodes[0].op {
            assert_eq!(kernel_source, "kernel void test() {}");
            assert_eq!(function_name, "test");
        }
    }

    #[test]
    fn response_ok_roundtrip() {
        let resp = EvalResponse::Ok {
            tensor_id: 42,
            shape: vec![2, 3],
            data: vec![1, 2, 3, 4],
        };
        let serialized = resp.serialize();
        let deserialized = EvalResponse::deserialize(&serialized).unwrap();
        match deserialized {
            EvalResponse::Ok { tensor_id, shape, data } => {
                assert_eq!(tensor_id, 42);
                assert_eq!(shape, vec![2, 3]);
                assert_eq!(data, vec![1, 2, 3, 4]);
            }
            _ => panic!("Expected Ok"),
        }
    }

    #[test]
    fn response_err_roundtrip() {
        let resp = EvalResponse::Err("something failed".to_string());
        let serialized = resp.serialize();
        let deserialized = EvalResponse::deserialize(&serialized).unwrap();
        match deserialized {
            EvalResponse::Err(msg) => assert_eq!(msg, "something failed"),
            _ => panic!("Expected Err"),
        }
    }

    #[test]
    fn request_roundtrip_matmul() {
        let req = EvalRequest {
            target_id: 5,
            tensors: vec![],
            nodes: vec![
                OpNode {
                    id: 5,
                    op: OpKind::Matmul,
                    inputs: vec![1, 2],
                    out_shape: Shape::new(vec![2, 2]),
                    out_dtype: DType::Float32,
                },
            ],
        };
        let serialized = req.serialize();
        let deserialized = EvalRequest::deserialize(&serialized).unwrap();
        assert!(deserialized.nodes[0].op.is_matmul());
        assert_eq!(deserialized.nodes[0].out_shape.dims(), &[2, 2]);
    }
}
```

- [ ] **Step 2: Add serial module to lib.rs**

Add `pub mod serial;` to `crates/core/src/lib.rs`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p applegpu-core serial 2>&1`
Expected: 5 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/serial.rs crates/core/src/lib.rs
git commit -m "feat: add binary serialization for graph eval requests and responses"
```

---

## Chunk 2: IPC Client + GPU Service Binary

### Task 2: Create IPC client module

**Files:**
- Create: `crates/core/src/ipc.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Create ipc.rs with Unix socket client**

```rust
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

use crate::error::{GpuError, Result};
use crate::serial::{EvalRequest, EvalResponse};

/// Default socket path for the GPU service.
/// Default socket path under user's home directory (avoids /tmp security concerns).
pub fn default_socket_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    format!("{}/.applegpu/runtime.sock", home)
}

/// Send an eval request to the GPU service and receive the result.
pub fn eval_remote(socket_path: &str, request: &EvalRequest) -> Result<EvalResponse> {
    let mut stream = UnixStream::connect(socket_path)
        .map_err(|e| GpuError::ComputeFailed(format!(
            "Failed to connect to GPU service at {}: {}",
            socket_path, e
        )))?;

    // Send request: [4 bytes length] + [payload]
    let payload = request.serialize();
    let len_bytes = (payload.len() as u32).to_le_bytes();
    stream.write_all(&len_bytes)
        .map_err(|e| GpuError::ComputeFailed(format!("IPC write failed: {}", e)))?;
    stream.write_all(&payload)
        .map_err(|e| GpuError::ComputeFailed(format!("IPC write failed: {}", e)))?;
    stream.flush()
        .map_err(|e| GpuError::ComputeFailed(format!("IPC flush failed: {}", e)))?;

    // Read response: [4 bytes length] + [payload]
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)
        .map_err(|e| GpuError::ComputeFailed(format!("IPC read failed: {}", e)))?;
    let resp_len = u32::from_le_bytes(len_buf) as usize;
    let mut resp_buf = vec![0u8; resp_len];
    stream.read_exact(&mut resp_buf)
        .map_err(|e| GpuError::ComputeFailed(format!("IPC read failed: {}", e)))?;

    EvalResponse::deserialize(&resp_buf)
        .map_err(|e| GpuError::ComputeFailed(format!("IPC deserialization failed: {}", e)))
}

// Note: No service_available() check — Path::exists is unreliable for sockets
// (stale files persist after crashes). The client just attempts to connect
// and gets a clear error if the service isn't running.
```

- [ ] **Step 2: Add ipc module to lib.rs**

Add `pub mod ipc;` to `crates/core/src/lib.rs`.

- [ ] **Step 3: Run cargo check**

Run: `cargo check -p applegpu-core 2>&1`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/ipc.rs crates/core/src/lib.rs
git commit -m "feat: add IPC client for remote GPU evaluation over Unix sockets"
```

---

### Task 3: Create the GPU service binary

**Files:**
- Create: `crates/gpu-service/Cargo.toml`
- Create: `crates/gpu-service/src/main.rs`
- Modify: `Cargo.toml` (workspace)

- [ ] **Step 1: Add gpu-service to workspace**

In root `Cargo.toml`, update members:

```toml
[workspace]
members = ["crates/core", "crates/python", "crates/gpu-service"]
```

- [ ] **Step 2: Create gpu-service Cargo.toml**

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
```

- [ ] **Step 3: Create main.rs — the GPU service**

```rust
use std::io::{Read, Write};
use std::os::unix::net::UnixListener;

use applegpu_core::device::Device;
use applegpu_core::graph::OpNode;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::serial::{EvalRequest, EvalResponse, TensorData};
use applegpu_core::tensor::Tensor;
use applegpu_core::buffer::Buffer;
use applegpu_core::ipc::default_socket_path();

fn handle_request(device: &Device, request: &EvalRequest) -> EvalResponse {
    let mut rt = LazyRuntime::new();

    // Load input tensors
    for td in &request.tensors {
        match Buffer::from_bytes(device, &td.data) {
            Ok(buffer) => {
                let tensor = Tensor::from_raw(td.id, td.shape.clone(), buffer);
                rt.insert_tensor(tensor);
            }
            Err(e) => return EvalResponse::Err(format!("Buffer allocation failed: {}", e)),
        }
    }

    // Load graph nodes
    for node in &request.nodes {
        rt.record_op(node.clone());
    }

    // Evaluate
    if let Err(e) = rt.eval(device, request.target_id) {
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

    let listener = UnixListener::bind(&socket_path)
        .expect("Failed to bind Unix socket");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                // Read request: [4 bytes length] + [payload]
                let mut len_buf = [0u8; 4];
                if stream.read_exact(&mut len_buf).is_err() {
                    continue;
                }
                let req_len = u32::from_le_bytes(len_buf) as usize;
                let mut req_buf = vec![0u8; req_len];
                if stream.read_exact(&mut req_buf).is_err() {
                    continue;
                }

                let response = match EvalRequest::deserialize(&req_buf) {
                    Ok(request) => handle_request(&device, &request),
                    Err(e) => EvalResponse::Err(format!("Deserialization failed: {}", e)),
                };

                // Send response: [4 bytes length] + [payload]
                let resp_payload = response.serialize();
                let len_bytes = (resp_payload.len() as u32).to_le_bytes();
                let _ = stream.write_all(&len_bytes);
                let _ = stream.write_all(&resp_payload);
                let _ = stream.flush();
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }
}
```

- [ ] **Step 4: Add `Tensor::from_raw` constructor and `record_op` taking owned OpNode**

The GPU service needs to construct tensors from raw buffers and IDs. Add to `crates/core/src/tensor.rs`:

```rust
    /// Create a tensor from a pre-existing buffer and explicit ID.
    /// Used by the GPU service to reconstruct tensors from serialized data.
    pub fn from_raw(id: u64, shape: Vec<usize>, buffer: Buffer) -> Self {
        Tensor {
            meta: TensorMeta {
                id,
                shape: Shape::new(shape),
                dtype: DType::Float32,
                location: TensorLocation::Shared,
            },
            buffer,
        }
    }
```

- [ ] **Step 5: Build the service**

Run: `cargo build -p applegpu-service 2>&1`
Expected: Build succeeds

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/gpu-service/ crates/core/src/tensor.rs
git commit -m "feat: add GPU service binary for remote Metal execution over IPC"
```

---

## Chunk 3: Backend Routing and End-to-End Tests

### Task 4: Wire VM backend into LazyRuntime eval

**Files:**
- Modify: `crates/core/src/lazy.rs`
- Modify: `crates/core/src/backend.rs`

- [ ] **Step 1: Add eval_remote method to LazyRuntime**

Add to `LazyRuntime` in `lazy.rs`:

```rust
    /// Evaluate a tensor via the remote GPU service (VM backend).
    /// Serializes the graph + input tensors, sends over IPC, receives result.
    pub fn eval_remote(&mut self, device: &Device, id: u64, socket_path: &str) -> Result<()> {
        if self.is_materialized(id) {
            return Ok(());
        }

        let order = self.graph.topo_sort(id)?;
        if order.is_empty() {
            return Err(GpuError::GraphError(format!("Tensor {} not found", id)));
        }

        // Collect input tensors needed by the graph
        let mut tensor_data = Vec::new();
        let mut needed_tensors = std::collections::HashSet::new();
        for &node_id in &order {
            if let Some(node) = self.graph.get_node(node_id) {
                for &input_id in &node.inputs {
                    if !self.graph.has_node(input_id) && !needed_tensors.contains(&input_id) {
                        needed_tensors.insert(input_id);
                        if let Some(t) = self.tensors.get(&input_id) {
                            let data = t.as_f32_slice();
                            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                            tensor_data.push(crate::serial::TensorData {
                                id: input_id,
                                shape: t.meta.shape.dims().to_vec(),
                                dtype: t.meta.dtype,
                                data: bytes,
                            });
                        }
                    }
                }
            }
        }

        // Collect graph nodes
        let nodes: Vec<crate::graph::OpNode> = order.iter()
            .filter_map(|&nid| self.graph.get_node(nid).cloned())
            .collect();

        let request = crate::serial::EvalRequest {
            target_id: id,
            tensors: tensor_data,
            nodes,
        };

        // Send to GPU service
        let response = crate::ipc::eval_remote(socket_path, &request)?;

        match response {
            crate::serial::EvalResponse::Ok { tensor_id, shape, data } => {
                // Reconstruct the result tensor from received bytes
                let buffer = crate::buffer::Buffer::from_bytes(device, &data)?;
                let tensor = Tensor::from_raw(tensor_id, shape, buffer);
                self.tensors.insert(tensor_id, tensor);

                // Remove evaluated nodes from graph
                for &nid in &order {
                    self.graph.remove_node(nid);
                }

                Ok(())
            }
            crate::serial::EvalResponse::Err(msg) => {
                Err(GpuError::ComputeFailed(format!("Remote eval failed: {}", msg)))
            }
        }
    }
```

- [ ] **Step 2: Add socket_path to Runtime and update Backend routing**

In `crates/core/src/backend.rs`, add socket path to `Runtime`:

```rust
pub struct Runtime {
    pub backend: Backend,
    pub device: Device,
    pub socket_path: Option<String>,
}
```

Update `init_backend()`:

```rust
pub fn init_backend() -> Result<&'static Runtime> {
    RUNTIME.get_or_try_init(|| {
        let backend = std::env::var("APPLEGPU_BACKEND")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(Backend::Mlx);

        let socket_path = if backend == Backend::Vm {
            Some(std::env::var("APPLEGPU_SOCKET")
                .unwrap_or_else(|_| crate::ipc::default_socket_path().to_string()))
        } else {
            None
        };

        let device = Device::new()?;

        Ok(Runtime { backend, device, socket_path })
    })
}
```

- [ ] **Step 3: Run cargo check**

Run: `cargo check -p applegpu-core 2>&1`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/lazy.rs crates/core/src/backend.rs
git commit -m "feat: add eval_remote to LazyRuntime and socket_path to Backend"
```

---

### Task 5: Wire Python to use VM backend when configured

**Files:**
- Modify: `crates/python/src/lib.rs`

- [ ] **Step 1: Update to_list to route through VM backend when active**

In `crates/python/src/lib.rs`, update the `to_list` method on `GpuTensor`:

```rust
    fn to_list(&self) -> PyResult<Vec<f32>> {
        let runtime = get_device_runtime()?;
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        if rt.is_pending(self.id) {
            if let Some(ref socket_path) = runtime.socket_path {
                // VM backend: send over IPC
                rt.eval_remote(&runtime.device, self.id, socket_path)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            } else {
                // MLX backend: execute locally
                rt.eval(&runtime.device, self.id)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
        }
        rt.read_f32(self.id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
```

Similarly update the `eval` method:

```rust
    fn eval(&self) -> PyResult<()> {
        let runtime = get_device_runtime()?;
        let mut rt = RUNTIME_LAZY.lock().unwrap();
        if let Some(ref socket_path) = runtime.socket_path {
            rt.eval_remote(&runtime.device, self.id, socket_path)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            rt.eval(&runtime.device, self.id)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }
```

And the module-level `eval` and `to_list` functions (which delegate to the methods — no change needed since they call `t.to_list()` and `t.eval()`).

- [ ] **Step 2: Build and verify existing tests still pass**

Run: `uv run maturin develop && uv run pytest -v 2>&1`
Expected: All 46 existing tests pass (MLX backend unaffected)

- [ ] **Step 3: Commit**

```bash
git add crates/python/src/lib.rs
git commit -m "feat: route eval through VM backend IPC when APPLEGPU_BACKEND=vm"
```

---

### Task 6: Integration tests and end-to-end verification

**Files:**
- Create: `crates/core/tests/serial_integration.rs`
- Modify: `Makefile` — add gpu-service build target

- [ ] **Step 1: Create serialization integration test**

```rust
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::ops;
use applegpu_core::serial::{EvalRequest, EvalResponse, TensorData};
use applegpu_core::tensor::Tensor;
use applegpu_core::buffer::Buffer;

#[test]
fn serialize_and_execute_remotely_in_process() {
    let device = match Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    // Client side: build graph
    let mut client_rt = LazyRuntime::new();
    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;

    // Serialize tensor data
    let a_bytes: Vec<u8> = a.as_f32_slice().iter().flat_map(|f| f.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.as_f32_slice().iter().flat_map(|f| f.to_le_bytes()).collect();

    client_rt.insert_tensor(a);
    client_rt.insert_tensor(b);

    let sum_id = ops::add(&mut client_rt, a_id, b_id).unwrap();
    let relu_id = ops::relu(&mut client_rt, sum_id).unwrap();

    // Build eval request (simulating what eval_remote does)
    let order = vec![sum_id, relu_id]; // manually since we know the graph
    let nodes: Vec<_> = order.iter()
        .filter_map(|&nid| client_rt.graph_node(nid).cloned())
        .collect();

    let request = EvalRequest {
        target_id: relu_id,
        tensors: vec![
            TensorData { id: a_id, shape: vec![4], dtype: applegpu_core::tensor::DType::Float32, data: a_bytes },
            TensorData { id: b_id, shape: vec![4], dtype: applegpu_core::tensor::DType::Float32, data: b_bytes },
        ],
        nodes,
    };

    // Serialize and deserialize (simulating wire transfer)
    let wire = request.serialize();
    let received = EvalRequest::deserialize(&wire).unwrap();

    // Server side: execute
    let mut server_rt = LazyRuntime::new();
    for td in &received.tensors {
        let buffer = Buffer::from_bytes(&device, &td.data).unwrap();
        let tensor = Tensor::from_raw(td.id, td.shape.clone(), buffer);
        server_rt.insert_tensor(tensor);
    }
    for node in &received.nodes {
        server_rt.record_op(node.clone());
    }

    server_rt.eval(&device, received.target_id).unwrap();

    let result = server_rt.read_f32(received.target_id).unwrap();
    // relu(add([1,2,3,4], [10,20,30,40])) = relu([11,22,33,44]) = [11,22,33,44]
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
}
```

Note: this test needs `LazyRuntime::graph_node()` — a public accessor for graph nodes. Add to `lazy.rs`:

```rust
    /// Get a graph node by ID (for serialization).
    pub fn graph_node(&self, id: u64) -> Option<&crate::graph::OpNode> {
        self.graph.get_node(id)
    }
```

- [ ] **Step 2: Update Makefile with gpu-service build**

Add to Makefile:

```makefile
build-service: build-rust
	cargo build -p applegpu-service
```

- [ ] **Step 3: Run all tests**

Run: `make clean && make test 2>&1`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/core/tests/serial_integration.rs crates/core/src/lazy.rs Makefile
git commit -m "test: add serialization integration test with simulated remote execution"
```

---

### Task 7: End-to-end verification and push

- [ ] **Step 1: Update backlog**

Mark IPC layer, graph serialization as complete. Add Phase 7b (AVF VM integration) as next.

- [ ] **Step 2: Update README**

Add VM backend section showing `APPLEGPU_BACKEND=vm` usage.

- [ ] **Step 3: Push**

```bash
git push origin main
```
