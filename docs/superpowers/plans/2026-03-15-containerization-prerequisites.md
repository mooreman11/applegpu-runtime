# Containerization Prerequisites Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix security, dtype, and wire protocol issues that block containerization integration.

**Architecture:** Five prerequisite fixes to the existing gpu-service and wire protocol. Each is independent and can be tested in isolation. No new crates or Swift code — all changes to existing Rust code.

**Tech Stack:** Rust (applegpu-core, applegpu-wire, applegpu-service)

**Spec:** `docs/superpowers/specs/2026-03-15-containerization-integration-design.md` (Prerequisites section)

---

## Chunk 1: Security + Wire Protocol

### Task 1: Reject FusedElementwise over wire protocol

**Files:**
- Modify: `crates/gpu-service/src/main.rs`

- [ ] **Step 1: Write test**

Add to gpu-service or as an integration test:
```rust
#[test]
fn test_fused_elementwise_rejected() {
    // Create a wire EvalRequest with a FusedElementwise node
    // Send to gpu-service handler
    // Verify it returns an error
}
```

- [ ] **Step 2: Add guard in handle_eval**

In `crates/gpu-service/src/main.rs`, in `handle_eval`, before converting wire nodes:
```rust
for node in &request.nodes {
    if matches!(&node.op, WireOpKind::FusedElementwise { .. }) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "FusedElementwise not allowed over wire protocol — fusion runs server-side",
        ));
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p applegpu-service -- --test-threads=1`

- [ ] **Step 4: Commit**

```bash
git add crates/gpu-service/
git commit -m "security: reject FusedElementwise over wire protocol

Prevents arbitrary MSL kernel injection from containers.
Fusion runs server-side from the unfused graph.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 2: Fix DType handling in gpu-service

**Files:**
- Modify: `crates/gpu-service/src/main.rs`
- Modify: `crates/wire/src/lib.rs` (if dtype mapping function needed)

- [ ] **Step 1: Write test**

```rust
#[test]
fn test_dtype_respected_in_eval() {
    // Create WireTensorData with dtype=1 (Float16)
    // Send eval request
    // Verify the created tensor has Float16 dtype
}
```

- [ ] **Step 2: Add dtype mapping in handle_eval**

Replace hardcoded `DType::Float32` with:
```rust
fn wire_dtype_to_core(d: u32) -> io::Result<DType> {
    match d {
        0 => Ok(DType::Float32),
        1 => Ok(DType::Float16),
        2 => Ok(DType::Float64),
        3 => Ok(DType::Int8),
        4 => Ok(DType::Int16),
        5 => Ok(DType::Int32),
        6 => Ok(DType::Int64),
        7 => Ok(DType::UInt8),
        8 => Ok(DType::UInt32),
        9 => Ok(DType::Bool),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unknown dtype: {}", d))),
    }
}
```

Use in `handle_eval`:
```rust
let dtype = wire_dtype_to_core(tensor_data.dtype)?;
let tensor = Tensor::from_data(&device, shape, dtype, &tensor_data.data)?;
```

Also fix `read_f32` to use `read_bytes` for generic dtype support.

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git add crates/gpu-service/ crates/wire/
git commit -m "fix: respect wire protocol dtype field in gpu-service

Was hardcoded to Float32 — now maps all 10 dtypes correctly.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 3: Add ReadTensorRequest/Response to wire protocol

**Files:**
- Modify: `crates/wire/src/lib.rs`
- Modify: `crates/gpu-service/src/main.rs`
- Modify: `crates/client/src/lib.rs`

- [ ] **Step 1: Add message types to wire crate**

```rust
pub struct ReadTensorRequest {
    pub tensor_id: u64,
}

pub enum ReadTensorResponse {
    Ok { tensor_id: u64, shape: Vec<usize>, dtype: u32, data: Vec<u8> },
    NotFound { tensor_id: u64 },
}
```

Add serialize/deserialize with a new magic: `AGRD` (read request), `AGRR` (read response).

- [ ] **Step 2: Add handler in gpu-service**

In `handle_connection`, after the eval handler, add a read handler that looks up the tensor in the runtime and returns its data.

- [ ] **Step 3: Add read_tensor to client**

```rust
impl GpuClient {
    pub fn read_tensor(&mut self, tensor_id: u64) -> io::Result<(Vec<usize>, u32, Vec<u8>)> {
        // Send ReadTensorRequest, receive ReadTensorResponse
    }
}
```

- [ ] **Step 4: Write roundtrip test**

- [ ] **Step 5: Commit**

```bash
git add crates/wire/ crates/gpu-service/ crates/client/
git commit -m "feat: add ReadTensorRequest/Response to wire protocol

Allows clients to fetch previously-computed tensor data from the server.
Needed for to_numpy()/to_list() on tensors evaluated in earlier requests.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 4: Deprecate legacy ipc.rs

**Files:**
- Modify: `crates/core/src/ipc.rs`

- [ ] **Step 1: Add deprecation attribute**

```rust
#[deprecated(since = "0.8.0", note = "Use applegpu-client crate instead")]
pub fn eval_remote(socket_path: &str, request: &EvalRequest) -> Result<EvalResponse> {
    // ... existing code unchanged
}
```

- [ ] **Step 2: Run tests, fix warnings**

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/ipc.rs
git commit -m "deprecate: mark eval_remote in ipc.rs as deprecated

Replaced by applegpu-client crate for container communication.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 5: gpu-service PID file + signal handling

**Files:**
- Modify: `crates/gpu-service/src/main.rs`

- [ ] **Step 1: Add PID file on startup**

Write PID to `~/.applegpu/gpu-service.pid` on bind. Remove on clean shutdown.

- [ ] **Step 2: Add SIGTERM handler**

Use `ctrlc` crate or `signal_hook` to catch SIGTERM/SIGINT. Set a flag, break the accept loop, drain connections.

- [ ] **Step 3: Add stale socket/PID detection**

Before binding, check if PID file exists and process is alive. If stale, clean up.

- [ ] **Step 4: Write test**

- [ ] **Step 5: Commit**

```bash
git add crates/gpu-service/
git commit -m "feat: gpu-service PID file + signal handling

Prevents double-start via PID file. SIGTERM triggers graceful shutdown.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 2: Backend Trait Abstraction

### Task 6: Create Backend trait + refactor PyO3 crate

**Files:**
- Create: `crates/python/src/backend.rs`
- Create: `crates/python/src/metal_backend.rs`
- Modify: `crates/python/src/lib.rs`

This is the foundation for the Linux socket backend (Plan B). For now, extract the existing Metal code into the trait pattern.

- [ ] **Step 1: Define Backend trait in backend.rs**

```rust
use applegpu_core::tensor::DType;
use applegpu_core::error::Result;

pub trait Backend: Send + Sync {
    fn init(&self) -> Result<()>;
    fn tensor_from_data(&self, data: &[u8], shape: Vec<usize>, dtype: DType) -> Result<u64>;
    fn eval(&self, id: u64) -> Result<()>;
    fn read_bytes(&self, id: u64) -> Result<Vec<u8>>;
    fn shape(&self, id: u64) -> Result<Vec<usize>>;
    fn dtype(&self, id: u64) -> Result<DType>;
    fn destroy(&self, id: u64) -> Result<()>;
    // Ops
    fn add(&self, a: u64, b: u64) -> Result<u64>;
    fn sub(&self, a: u64, b: u64) -> Result<u64>;
    fn mul(&self, a: u64, b: u64) -> Result<u64>;
    fn div(&self, a: u64, b: u64) -> Result<u64>;
    fn matmul(&self, a: u64, b: u64) -> Result<u64>;
    fn neg(&self, a: u64) -> Result<u64>;
    fn relu(&self, a: u64) -> Result<u64>;
    fn gelu(&self, a: u64) -> Result<u64>;
    fn softmax(&self, a: u64) -> Result<u64>;
    fn reshape(&self, a: u64, shape: Vec<usize>) -> Result<u64>;
    // ... all 38+ ops
}
```

- [ ] **Step 2: Implement MetalBackend in metal_backend.rs**

Move the existing `RUNTIME_LAZY` + `get_device_runtime()` + all op dispatch into `MetalBackend`. Each method wraps the existing logic.

- [ ] **Step 3: Refactor lib.rs to use Backend trait**

```rust
#[cfg(target_os = "macos")]
mod metal_backend;
#[cfg(target_os = "macos")]
use metal_backend::MetalBackend as ActiveBackend;

// Linux backend will be added in Plan B
#[cfg(target_os = "linux")]
mod socket_backend;
#[cfg(target_os = "linux")]
use socket_backend::SocketBackend as ActiveBackend;

static BACKEND: Lazy<ActiveBackend> = Lazy::new(|| ActiveBackend::new());
```

PyO3 functions call `BACKEND.add(a, b)` instead of direct `applegpu_core::ops::add()`.

- [ ] **Step 4: Verify all existing tests pass**

Run: `cargo test -p applegpu-python -- --test-threads=1`
Run: `uv run maturin develop && uv run pytest -v`

This is a refactor — zero behavior change. ALL existing tests must pass.

- [ ] **Step 5: Commit**

```bash
git add crates/python/
git commit -m "refactor: extract Backend trait, move Metal code to MetalBackend

No behavior change. Prepares for Linux SocketBackend in Plan B.
All existing tests pass unchanged.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
