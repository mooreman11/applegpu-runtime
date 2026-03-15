# Apple Containerization Framework Integration

**Date:** 2026-03-15
**Status:** Approved
**Scope:** Containerization framework integration (option B), gpu-container CLI, platform-specific Python wheels, security hardening, packaging.

## Overview

Integrate with Apple's Containerization framework to enable running PyTorch workloads inside OCI Linux containers on Apple Silicon, with GPU compute routed to the host's Metal GPU via the gpu-service over socket relay.

Users run: `gpu-container run pytorch:latest -- python train.py` and their PyTorch code uses Metal GPU transparently.

## Architecture

```
User
  │
  ├── gpu-container run pytorch:latest -- python train.py
  │   (or gpu.create_container("pytorch:latest") from Python)
  │
  ▼
gpu-container CLI (Swift, macOS 26+)
  │
  ├── 1. Auto-start gpu-service (with readiness probe)
  ├── 2. Pull OCI image (via Containerization framework)
  ├── 3. Create LinuxContainer with:
  │      - Socket relay: host ~/.applegpu/runtime.sock → container /var/run/applegpu.sock
  │      - Mount: inject applegpu client + Python package into /opt/applegpu/
  │      - Env: APPLEGPU_SOCKET=/var/run/applegpu.sock, PYTHONPATH=/opt/applegpu
  ├── 4. Start container, run user command
  ├── 5. Stream stdout/stderr
  └── 6. Cleanup on exit
```

Inside the container:
```python
import applegpu_runtime as gpu  # detects Linux → socket client backend
gpu.init_backend()              # connects to /var/run/applegpu.sock
model = gpu.to_applegpu(model)  # ops route over socket to host Metal GPU
```

## Prerequisites (must fix before implementation)

### P1: Disable FusedElementwise over wire protocol

The wire protocol accepts arbitrary MSL kernel source via `WireOpKind::FusedElementwise`. A malicious container could inject arbitrary Metal shaders.

**Fix:** In `crates/gpu-service/src/main.rs`, in the `handle_eval` function (around line 122), add a guard BEFORE the `wire_node_to_core` conversion:

```rust
// In handle_eval, after deserializing the EvalRequest:
for node in &request.nodes {
    if matches!(node.op, WireOpKind::FusedElementwise { .. }) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "FusedElementwise not allowed over wire protocol — fusion runs server-side"
        ));
    }
}
```

The guard is in `handle_eval` (not `wire_node_to_core`) because the conversion function is also used in local tests and the core crate where FusedElementwise is legitimate.

Fusion still works: the gpu-service receives the unfused graph, runs its own fusion pass, and generates kernels locally.

### P2: Fix DType handling in gpu-service

Currently gpu-service hardcodes `DType::Float32` when creating tensors from wire data (line 111 of `main.rs`). The wire protocol's `WireTensorData` struct DOES carry a `dtype: u32` field (line 504 of `crates/wire/src/lib.rs`) — it's just ignored by the gpu-service.

**Fix:** In `crates/gpu-service/src/main.rs`, in `handle_eval`, map `tensor_data.dtype` to `DType`:

```rust
let dtype = match tensor_data.dtype {
    0 => DType::Float32,
    1 => DType::Float16,
    2 => DType::Float64,
    3 => DType::Int8,
    4 => DType::Int16,
    5 => DType::Int32,
    6 => DType::Int64,
    7 => DType::UInt8,
    8 => DType::UInt32,
    9 => DType::Bool,
    _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Unknown dtype")),
};
let tensor = Tensor::from_data(&device, shape, dtype, &tensor_data.data)?;
```

Also update `read_f32` (line 136) to dispatch based on dtype: use `read_bytes` for the generic path.

### P3: Add ReadTensorRequest/Response to wire protocol

The current wire protocol only returns tensor data as part of `EvalResponse`. The Linux client needs to fetch previously-computed tensor data for `to_numpy()`, `to_list()`, etc.

**Fix:** Add message types:
- `ReadTensorRequest { tensor_id: u64 }` — client requests tensor data
- `ReadTensorResponse { tensor_id: u64, shape: Vec<usize>, dtype: DType, data: Vec<u8> }` — server returns data
- Or: `ReadTensorResponse::NotFound { tensor_id }` — tensor doesn't exist

### P4: Deprecate legacy ipc.rs path

The old `eval_remote` in `crates/core/src/ipc.rs` doesn't perform the handshake that the new gpu-service expects.

**Fix:** Add `#[deprecated(since = "0.8.0", note = "Use applegpu-client crate instead")]` to the `eval_remote` function. Update callers to use `crates/client` for new code.

### P5: gpu-service signal handling + PID file

- PID file at `~/.applegpu/gpu-service.pid` — prevents double-start
- SIGTERM handler — graceful shutdown (close listener, drain connections)
- Stale socket detection — check PID file before removing socket

## Container-side Python Package

### Single package, platform-specific wheels

`pip install applegpu-runtime` produces:
- **macOS wheel** — links `applegpu-core` (Metal FFI). Current behavior.
- **Linux wheel** — links `applegpu-client` + `applegpu-wire` (socket transport). New.

Same Python API everywhere: `tensor()`, `from_numpy()`, `eval()`, `add()`, etc.

### Trait-based backend abstraction

Split `crates/python/src/lib.rs` into:

```
crates/python/src/
  ├── lib.rs              — #[pymodule], thin dispatch via Backend trait
  ├── backend.rs          — trait Backend { fn tensor(), fn eval(), fn add(), ... }
  ├── metal_backend.rs    — #[cfg(target_os = "macos")] — wraps LazyRuntime
  └── socket_backend.rs   — #[cfg(target_os = "linux")] — wraps GpuClient
```

```rust
// backend.rs
pub trait Backend: Send + Sync {
    fn init(&self) -> Result<()>;
    fn tensor(&self, data: &[u8], shape: &[usize], dtype: DType) -> Result<u64>;
    fn eval(&self, id: u64) -> Result<()>;
    fn read_bytes(&self, id: u64) -> Result<Vec<u8>>;
    fn shape(&self, id: u64) -> Result<Vec<usize>>;
    fn dtype(&self, id: u64) -> Result<DType>;
    fn add(&self, a: u64, b: u64) -> Result<u64>;
    fn matmul(&self, a: u64, b: u64) -> Result<u64>;
    // ... all ops
    fn destroy(&self, id: u64) -> Result<()>;
}
```

```rust
// metal_backend.rs (macOS only)
#[cfg(target_os = "macos")]
pub struct MetalBackend {
    runtime: Mutex<LazyRuntime>,
    device_runtime: &'static Runtime,
}
// Implements Backend by delegating to LazyRuntime + applegpu_core ops
```

```rust
// socket_backend.rs (Linux only)
#[cfg(target_os = "linux")]
pub struct SocketBackend {
    client: Mutex<GpuClient>,
    local_tensors: Mutex<HashMap<u64, TensorMeta>>,  // local metadata cache
    pending_graph: Mutex<Graph>,                       // lazy ops before eval
}
// Implements Backend by serializing ops to wire protocol
```

### Cargo.toml conditional dependencies

```toml
[target.'cfg(target_os = "macos")'.dependencies]
applegpu-core = { path = "../core" }

[target.'cfg(target_os = "linux")'.dependencies]
applegpu-client = { path = "../client" }
applegpu-wire = { path = "../wire" }
```

### init_backend() behavior

```python
gpu.init_backend()
# macOS: initialize Metal device (current behavior)
# Linux: connect to APPLEGPU_SOCKET env var (default: /var/run/applegpu.sock)
#         perform handshake, receive ContainerId
```

### SocketBackend lazy evaluation

The Linux client mirrors the lazy evaluation model:
1. `tensor(data, shape, dtype)` — stores data locally, assigns local ID
2. `add(a, b)` / `matmul(a, b)` / etc. — records in local graph (same Graph struct from core, reused via wire crate)
3. `eval(id)` — serializes the subgraph + input tensor data into `EvalRequest`, sends to gpu-service, receives `EvalResponse` with result data
4. `to_numpy(id)` — if already evaluated (data cached locally), return it. If not, send `ReadTensorRequest` to server.

This means the wire crate's `EvalRequest` already handles batched graph submission — the Linux client collects ops lazily and sends them all at once, same as the macOS path does locally.

## gpu-container CLI

### Separate Swift package

```
swift/GPUContainer/
  ├── Package.swift         — depends on apple/containerization
  ├── Sources/
  │   ├── main.swift        — CLI entry (ArgumentParser)
  │   ├── ContainerManager.swift  — LinuxContainer lifecycle
  │   └── ServiceManager.swift    — gpu-service auto-start/stop/status
```

Separate from `swift/Package.swift` (AppleGPUBridge) because:
- Different platform requirement (macOS 26+ vs macOS 14+)
- Different purpose (container management vs Metal compute)
- Different dependency (Containerization framework vs Virtualization framework)

### Commands

```bash
gpu-container run IMAGE [--cpus N] [--memory SIZE] [-- COMMAND]
gpu-container stop CONTAINER_ID
gpu-container list
gpu-container status          # check gpu-service health
gpu-container service start   # manual gpu-service management
gpu-container service stop
```

### LinuxContainer Configuration

```swift
var config = LinuxContainer.Configuration()
config.cpus = cpuCount
config.memoryInBytes = memoryBytes
config.sockets = [
    UnixSocketConfiguration(
        source: URL(fileURLWithPath: "\(home)/.applegpu/runtime.sock"),
        destination: URL(fileURLWithPath: "/var/run/applegpu.sock"),
        direction: .into
    )
]
config.mounts.append(
    Mount(source: injectPath, destination: "/opt/applegpu", readOnly: true)
)
config.process.environment["APPLEGPU_SOCKET"] = "/var/run/applegpu.sock"
config.process.environment["PYTHONPATH"] = "/opt/applegpu"
```

### Auto-start logic

```swift
func ensureServiceRunning() async throws {
    let pidFile = "\(home)/.applegpu/gpu-service.pid"
    let socketPath = "\(home)/.applegpu/runtime.sock"

    // Check if already running
    if FileManager.default.fileExists(atPath: pidFile) {
        let pid = try String(contentsOfFile: pidFile).trimmingCharacters(in: .whitespacesAndNewlines)
        if let pidInt = Int(pid), processExists(pid: pidInt) {
            // Verify socket is connectable
            if try await testConnect(socketPath) {
                return  // Already running and healthy
            }
        }
    }

    // Start gpu-service
    let process = Process()
    process.executableURL = URL(fileURLWithPath: gpuServiceBinaryPath)
    process.arguments = ["--socket", socketPath, "--pid-file", pidFile]
    try process.run()

    // Readiness probe: poll until socket is connectable
    for _ in 0..<50 {  // 5 seconds max
        try await Task.sleep(nanoseconds: 100_000_000)  // 100ms
        if try await testConnect(socketPath) { return }
    }
    throw GPUContainerError.serviceStartTimeout
}
```

## Security

### v0.8.0

- FusedElementwise rejected over wire protocol (prevents arbitrary MSL injection)
- Per-container resource quotas enforced server-side
- MAX_MESSAGE_SIZE prevents oversized payloads
- Socket file permissions (0600) restrict access to current user

### Backlog (v0.9.0+)

- Token-based authentication for container connections
- Rate limiting per container
- Audit logging for GPU operations
- TLS for non-local transports

## Packaging

### PyPI wheel

`maturin build` produces platform wheels:
- `applegpu_runtime-0.8.0-cp311-cp311-macosx_14_0_arm64.whl` (Metal)
- `applegpu_runtime-0.8.0-cp311-cp311-manylinux_2_34_aarch64.whl` (socket client)

Cross-compilation for Linux: use `maturin build --target aarch64-unknown-linux-gnu` with zig cc or cross-rs.

### Container base image

```dockerfile
FROM python:3.11-slim
RUN pip install applegpu-runtime torch
# Client package auto-detects Linux → socket backend
ENV APPLEGPU_SOCKET=/var/run/applegpu.sock
```

Published to: `ghcr.io/mooreman11/applegpu-runtime:latest`

### GPU binaries

`gpu-container` and `gpu-service` distributed via:
- GitHub Releases (universal macOS binary)
- Install script: `curl -fsSL https://raw.githubusercontent.com/mooreman11/applegpu-runtime/main/install.sh | sh`

## Testing Strategy

### Prerequisites (P1-P5)
- Wire protocol rejects FusedElementwise
- gpu-service respects wire dtype field
- ReadTensorRequest/Response roundtrip
- Legacy ipc.rs marked deprecated
- PID file + signal handling

### Unit tests
- SocketBackend: tensor creation, lazy op recording, eval serialization
- MetalBackend: existing tests (unchanged)
- Backend trait: mock backend for testing dispatch layer

### Integration tests
- gpu-service → client → handshake → create tensor → eval → read result
- Multiple clients with different ContainerIds
- Resource quota enforcement across clients
- FusedElementwise rejection

### End-to-end tests
- `gpu-container run` with base image, execute `python -c "import applegpu_runtime; ..."`
- PyTorch inference inside container (ResNet-18 forward pass)
- PyTorch training inside container (MLP training, loss decreases)

### Packaging tests
- macOS wheel installs and imports on macOS
- Linux wheel installs and imports on Linux (docker test)
- Cross-compilation produces valid Linux aarch64 wheel

## Backlog: Raw Virtualization.framework (option A)

Deferred to post-v0.8.0. For users who need:
- Custom VM configurations not supported by Containerization framework
- macOS < 26 support
- Direct vsock without socket relay
- Custom Linux kernel configurations

Implementation: use `VZVirtualMachine` directly with `VZVirtioSocketListener` for vsock. The gpu-service already listens on Unix sockets, and the client already has vsock transport — the missing piece is VM lifecycle management.
