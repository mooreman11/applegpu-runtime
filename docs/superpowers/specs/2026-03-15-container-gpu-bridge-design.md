# Phase 7b: Container GPU Bridge

**Date:** 2026-03-15
**Status:** Draft
**Scope:** Multi-client GPU service with dual transport (virtio-vsock + Unix socket), connection-based container identity, and a Linux-compilable client crate. No container/VM lifecycle management — that is the container framework's responsibility.

## Overview

Containers running on macOS (via Apple Containerization Framework or Docker) cannot access the host's Metal GPU directly. Phase 7b builds the **GPU bridge layer**: a host-side GPU service that accepts concurrent connections from containers, assigns each connection a scheduler container with resource quotas, and executes Metal workloads on their behalf.

Phase 7a proved the serialization protocol and single-client IPC. Phase 7b upgrades this to production multi-tenancy with the transport layer containers actually use.

## Target Container Runtimes

| Runtime | Transport | How |
|---------|-----------|-----|
| Apple Containerization Framework (primary) | virtio-vsock | Guest connects to host CID=2, port 5678 |
| Docker on macOS (secondary, future) | Unix socket | Bind-mount `~/.applegpu/runtime.sock` into container |
| Local development / testing | Unix socket | Direct connection from host process |

## Architecture

```
┌─────────────────────────────────────┐
│  Container (Linux guest)            │
│                                     │
│  Python API  →  applegpu-client     │
│                 (serial + transport)│
│                      │              │
│              vsock connect(CID=2,   │
│                     port 5678)      │
└──────────────┬──────────────────────┘
               │ virtio-vsock
┌──────────────▼──────────────────────┐
│  Host (macOS)                       │
│                                     │
│  gpu-service                        │
│   ├── VsockListener (port 5678)     │
│   ├── UnixListener (runtime.sock)   │
│   ├── thread-per-connection         │
│   │    └── Session                  │
│   │         ├── ContainerId         │
│   │         ├── LazyRuntime (owned) │
│   │         └── read/eval/respond   │
│   ├── SharedState (Arc<Mutex<>>)    │
│   │    ├── Scheduler                │
│   │    ├── BufferPool               │
│   │    └── Device                   │
│   └── Metal GPU                     │
└─────────────────────────────────────┘
```

## Design Decisions

### Per-connection LazyRuntime vs shared LazyRuntime

**Decision: Shared LazyRuntime behind Arc<Mutex<>>.**

Each connection gets a `ContainerId` but shares a single `LazyRuntime` instance. This is necessary because:
- The `Scheduler` must see all containers' jobs to make fair scheduling decisions
- The `BufferPool` should be shared to maximize GPU memory reuse across containers
- The `Device` (Metal GPU) is a single global resource

The `Mutex<LazyRuntime>` pattern already exists in the Python layer and is proven. The GPU is the bottleneck, not lock contention — most time is spent waiting for Metal command buffers.

### Connection-based container identity

**Decision: No authentication tokens. Connection = container.**

- **Vsock**: Hypervisor-mediated — only VMs on this physical host can connect. The guest CID uniquely identifies the VM. No spoofing possible.
- **Unix socket**: `SO_PEERCRED` provides the connecting process's UID/PID. Sufficient for local dev.
- Each connection triggers `scheduler.register_container(config)` on connect and `scheduler.deregister_container(id)` on disconnect (including crashes).

### Session lifecycle

```
connect → handshake → [eval_request → eval_response]* → disconnect
```

**Handshake** (new, not in Phase 7a):
```
Client sends:  [4 bytes] magic: b"AGHI"
               [4 bytes] version: 1u32
               [4 bytes] max_memory_bytes: u32 (requested quota, 0 = default)

Server sends:  [4 bytes] magic: b"AGHO"
               [4 bytes] status: 0=ok, 1=rejected
               [8 bytes] container_id: u64
               [4 bytes] granted_memory_bytes: u32
```

After handshake, the existing `EvalRequest`/`EvalResponse` protocol from Phase 7a is used unchanged.

On disconnect (clean or crash), the service:
1. Deregisters the container from the scheduler
2. Frees all tensors owned by that container
3. Removes pending graph nodes for that container
4. Returns buffers to the pool

### Vsock listener (Swift side)

Apple's `Virtualization.framework` provides `VZVirtioSocketListener` for host-side vsock listening. This must be Swift code since the framework is Swift/ObjC only.

The Swift layer exports a C ABI function:
```c
// Start listening for vsock connections on the given port.
// Calls the provided callback with each accepted connection's file descriptor.
void gpu_bridge_vsock_listen(uint32_t port, void (*on_accept)(int fd, uint32_t guest_cid));
```

The Rust GPU service calls this via FFI. Each accepted connection yields a standard file descriptor that Rust can wrap in `std::os::unix::io::FromRawFd` and use with the existing `Read`/`Write` protocol code.

**Important**: `VZVirtioSocketListener` requires an active `VZVirtualMachine` instance to attach to. Since we are NOT managing VM lifecycle, the vsock listener must be attached to VM instances that the container framework creates. This means:
- For Phase 7b, we implement and test the vsock transport code but **cannot run it end-to-end** without a container framework providing VMs.
- The Unix socket path is fully testable end-to-end.
- The vsock code path is tested via unit tests with mock file descriptors and integration tests that verify the protocol over Unix socketpairs (same byte-stream semantics).

**Deferred to backlog**: Integration with Apple Containerization Framework's VM instances for vsock listener attachment.

### Multi-threaded GPU service

```rust
struct SharedState {
    runtime: Mutex<LazyRuntime>,  // graph, tensors, scheduler, pool
    device: Device,               // Metal GPU (thread-safe, Send+Sync)
}

fn main() {
    let shared = Arc::new(SharedState::new());

    // Listener threads
    let s1 = shared.clone();
    thread::spawn(move || unix_listener_loop(s1));

    // Vsock listener (when available)
    // let s2 = shared.clone();
    // thread::spawn(move || vsock_listener_loop(s2));

    // Main thread handles signals / shutdown
}

fn handle_connection(shared: Arc<SharedState>, stream: impl Read + Write) {
    // 1. Handshake → register container
    // 2. Loop: read EvalRequest → lock runtime → eval → unlock → send EvalResponse
    // 3. On disconnect → deregister container, cleanup
}
```

### Linux client crate

A new crate `crates/client` that compiles on Linux without Swift or Metal dependencies:

```
crates/client/
  Cargo.toml       # depends only on std (no applegpu-core)
  src/
    lib.rs
    serial.rs      # copy of core's serial.rs (or shared via a common crate)
    transport.rs   # vsock + unix socket connect/send/recv
```

**Decision: Extract `serial.rs` into a shared crate `crates/wire`** rather than duplicating.

```
crates/wire/          # no platform dependencies, compiles everywhere
  Cargo.toml
  src/lib.rs          # EvalRequest, EvalResponse, TensorData, handshake types

crates/core/          # depends on wire + Swift/Metal (macOS only)
  Cargo.toml
  src/serial.rs       # re-exports from wire, removed duplicated code

crates/client/        # depends on wire only (Linux + macOS)
  Cargo.toml
  src/lib.rs          # connect(), eval(), disconnect()
  src/transport.rs    # VsockTransport, UnixTransport
```

This avoids code duplication and ensures the wire format is always in sync between client and service.

## Components

### 1. `crates/wire` — Shared wire protocol (new crate)

**Contents extracted from `crates/core/src/serial.rs`:**
- `EvalRequest`, `EvalResponse`, `TensorData` structs
- `serialize()` / `deserialize()` methods
- Handshake message types (`HandshakeRequest`, `HandshakeResponse`)
- All op-type discriminants and conversion functions
- Length-prefixed framing helpers: `write_message(stream, payload)`, `read_message(stream)`

**No dependencies** beyond `std`. Compiles on Linux and macOS.

### 2. `crates/client` — Container-side client library (new crate)

**Purpose:** Minimal library that containers use to submit GPU workloads.

```rust
pub struct GpuClient {
    stream: Box<dyn ReadWrite>,   // vsock or unix socket
    container_id: u64,
}

impl GpuClient {
    /// Connect via Unix socket.
    pub fn connect_unix(path: &str) -> Result<Self>;

    /// Connect via vsock (CID=2 is host).
    pub fn connect_vsock(port: u32) -> Result<Self>;

    /// Submit an eval request and block for the result.
    pub fn eval(&mut self, request: &EvalRequest) -> Result<EvalResponse>;

    /// Disconnect and release resources.
    pub fn disconnect(self) -> Result<()>;
}
```

**Dependencies:** `crates/wire` only. No Metal, no Swift, no `applegpu-core`.

**Vsock on Linux:** Uses the `AF_VSOCK` socket family via raw syscalls (`libc::socket`, `libc::connect` with `sockaddr_vm`). No external crate needed — it's ~20 lines of unsafe socket code.

### 3. `crates/gpu-service` — Upgraded host service (modify existing)

**Changes from Phase 7a:**
- Thread-per-connection with `Arc<SharedState>`
- Handshake on connect → `scheduler.register_container()`
- Cleanup on disconnect → `scheduler.deregister_container()`
- Unix socket listener in a dedicated thread
- Vsock listener preparation (compiled behind `#[cfg(feature = "vsock")]` until container framework integration is ready)
- Graceful shutdown on SIGTERM/SIGINT

### 4. Swift vsock support (new FFI functions)

**New functions in `swift/Sources/AppleGPUBridge/`:**

```c
// bridge.h additions
void* gpu_bridge_vsock_create_listener(uint32_t port);
void gpu_bridge_vsock_destroy_listener(void* listener);
int gpu_bridge_vsock_accept(void* listener);  // blocks, returns fd
```

**Deferred to backlog:** Actual vsock listener requires a `VZVirtualMachine` to attach to. The Swift code will be structured to accept a VM handle, but the integration point with the container framework is out of scope.

### 5. `crates/core` changes (minimal)

- `serial.rs` → re-export from `crates/wire` (backward compatible)
- `ipc.rs` → update to use `crates/wire` framing helpers
- `lazy.rs` → add `cleanup_container(container_id)` method that removes all tensors and graph nodes belonging to a container
- `graph.rs` → `remove_nodes_for_container()` already exists

## Wire Protocol

### Framing (all transports)

```
[4 bytes] message_length: u32 (little-endian, payload only)
[message_length bytes] payload
```

### Handshake (new)

```
Client → Server:
  [4 bytes] magic: b"AGHI"
  [4 bytes] protocol_version: u32 = 1
  [4 bytes] requested_memory: u32 (bytes, 0 = use default)

Server → Client:
  [4 bytes] magic: b"AGHO"
  [4 bytes] status: u32 (0=accepted, 1=rejected_quota, 2=rejected_capacity)
  [8 bytes] container_id: u64
  [4 bytes] granted_memory: u32 (bytes)
```

### Eval (unchanged from Phase 7a)

```
Client → Server: EvalRequest (magic b"AGPU", version 2)
Server → Client: EvalResponse (magic b"AGPR")
```

## Error Handling

- **Connection refused**: GPU service not running → client returns clear error with instructions
- **Quota exceeded**: Scheduler rejects allocation → `EvalResponse::Err("memory quota exceeded for container N")`
- **Container crash**: Service detects broken pipe on read/write → cleanup thread runs deregister + free
- **Service crash**: Containers get connection reset → must reconnect and re-send any pending work (no server-side persistence)
- **Malformed messages**: Deserialization failure → log warning, send `EvalResponse::Err`, close connection

## Testing Strategy

1. **Unit tests** (`crates/wire`): Roundtrip serialization for handshake messages, backward compat with existing EvalRequest/EvalResponse tests
2. **Unit tests** (`crates/client`): Mock transport (in-memory byte buffer), verify protocol sequence
3. **Integration tests** (`crates/gpu-service`): Spawn service in-process, connect N Unix socket clients concurrently, verify fair scheduling and resource cleanup
4. **Stress test**: 8 concurrent clients each sending 100 eval requests, verify no deadlocks, all results correct, memory freed on disconnect
5. **Vsock transport**: Tested via Unix socketpair (identical byte-stream semantics) until container framework integration is available

## Backlog Items (deferred)

- **Apple Containerization Framework integration** — attach vsock listener to framework-managed VMs
- **Docker bind-mount documentation** — instructions for mounting the GPU service socket
- **Client Python bindings** — PyO3 wrapper around `crates/client` for use inside containers
- **Persistent tensor handles** — allow containers to keep tensors resident between eval requests (currently each request is independent)
- **TLS/authentication for TCP transport** — if TCP transport is ever added for remote containers
- **Health check endpoint** — service status query for monitoring
- **Metrics/observability** — per-container GPU utilization, queue depth, latency histograms
