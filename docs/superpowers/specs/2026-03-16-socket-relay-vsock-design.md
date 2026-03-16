# Socket Relay & vsock Transport Design

**Date:** 2026-03-16
**Status:** Approved
**Scope:** Replace TCP bridge with Unix socket relay (primary) and vsock transport (secondary). Support Containerization framework, Docker, and legacy `container` CLI.

## Overview

The current `gpu-container` CLI uses a TCP bridge to forward container traffic to the gpu-service Unix socket. This adds latency and complexity. Replace it with direct Unix socket mounting (primary) and vsock relay (secondary), while keeping TCP as a legacy fallback.

## Architecture

Three transport tiers, all converging on the same gpu-service wire protocol:

```
Tier 1 (primary): Unix socket mount
  Container (/var/run/applegpu.sock)
    → mounted from ~/.applegpu/runtime.sock
    → gpu-service (direct, no relay)
  Used by: Containerization framework (UnixSocketConfiguration), Docker (-v bind mount)

Tier 2: vsock via Containerization framework
  Container (AF_VSOCK CID=2:5678)
    → gpu-container Swift process (VZVirtioSocketDevice listener)
    → relay to ~/.applegpu/runtime.sock
    → gpu-service
  Used by: Containerization framework VMs when socket mount unavailable

Tier 3 (legacy): TCP bridge
  Container (TCP 192.168.64.1:7654)
    → TCPBridge relay in gpu-container
    → gpu-service Unix socket
  Used by: container CLI fallback
```

The wire protocol, handshake, message framing, and per-container isolation are unchanged across all tiers.

## Component Changes

### 1. gpu-service (Rust) — No changes needed

**File:** `crates/gpu-service/src/main.rs`

gpu-service stays Unix-socket-only. All three transport tiers ultimately connect to the gpu-service via its Unix socket (`~/.applegpu/runtime.sock`):
- Tier 1: Containerization framework or Docker mounts the socket directly into the container
- Tier 2: vsock relay in the Swift process connects to the Unix socket on behalf of the container
- Tier 3: TCP bridge connects to the Unix socket on behalf of the container

Since only `UnixStream` ever reaches `handle_connection`, no generalization is needed. The existing `handle_connection(shared: Arc<SharedState>, stream: UnixStream)` signature stays as-is.

### 2. Client transport (Rust) — Type-erased transport

**Files:** `crates/client/src/lib.rs`, `crates/client/src/transport.rs`

Replace the current `UnixStream::from_raw_fd` vsock hack with a proper `Transport` trait. Use type-erasure (`Box<dyn Transport>`) inside `GpuClient` to avoid cascading generics through `SocketBackend`.

The current `Transport` trait at `transport.rs:4-5` is `pub trait Transport: Read + Write + Send {}` with no methods. The current `GpuClient.stream` field at `lib.rs:42` is `UnixStream`. The current `GpuClient::Drop` calls `self.stream.shutdown(Shutdown::Both)` at `lib.rs:104`.

**Proposed changes:**

Add a `shutdown` method to the `Transport` trait (new requirement — currently not present):
```rust
// transport.rs — CHANGED from current empty trait
pub trait Transport: Read + Write + Send {
    fn shutdown(&self) -> io::Result<()>;
}

impl Transport for UnixStream {
    fn shutdown(&self) -> io::Result<()> {
        UnixStream::shutdown(self, std::net::Shutdown::Both)
    }
}

impl Transport for TcpStream {
    fn shutdown(&self) -> io::Result<()> {
        TcpStream::shutdown(self, std::net::Shutdown::Both)
    }
}
```

For vsock on Linux, use the `vsock` crate which provides `VsockStream` (implements `Read + Write`):
```rust
#[cfg(target_os = "linux")]
impl Transport for vsock::VsockStream {
    fn shutdown(&self) -> io::Result<()> {
        vsock::VsockStream::shutdown(self, std::net::Shutdown::Both)
    }
}
```

Change `GpuClient.stream` from `UnixStream` to `Box<dyn Transport>` (proposed change):
```rust
// lib.rs — CHANGED from UnixStream
pub struct GpuClient {
    stream: Box<dyn Transport>,
    pub container_id: u64,
    pub granted_memory: u64,
}
```

Add `connect_tcp(host: &str, port: u16)` for Tier 3 support:
```rust
pub fn connect_tcp(host: &str, port: u16, requested_memory: u64) -> Result<Self> {
    let stream = TcpStream::connect((host, port))?;
    Self::handshake(Box::new(stream), requested_memory)
}
```

Add `connect_auto()` that encapsulates the auto-detection order:

```rust
pub fn connect_auto(requested_memory: u64) -> Result<Self> {
    // 1. Check /dev/vsock exists → connect_vsock(CID=2, port=5678)
    // 2. Check APPLEGPU_SOCKET env → connect_unix(path)
    // 3. Check APPLEGPU_HOST + APPLEGPU_PORT env → connect_tcp(host, port)
    // 4. Try default /var/run/applegpu.sock → connect_unix
    // 5. None found → error with helpful message listing what was tried
}
```

The auto-detection order is a simple if/else chain, trivially reorderable after benchmarking.

### 3. gpu-container CLI (Swift) — Major refactor

**Files:** `swift/GPUContainer/Sources/Run.swift`, new `ContainerRunner.swift`

**Platform:** Bump `Package.swift` platform target from `.macOS(.v15)` to `.macOS(.v26)`. Use `#available(macOS 26, *)` guards for Containerization framework APIs. TCP bridge remains as runtime fallback. This drops macOS 15-25 support for the `gpu-container` binary (the gpu-service Rust binary and Python package are unaffected).

**Primary path (Containerization framework):**
- Create `LinuxContainer.Configuration()` with:
  - `UnixSocketConfiguration(source: host_socket, destination: "/var/run/applegpu.sock", direction: .into)`
  - Environment: `APPLEGPU_SOCKET=/var/run/applegpu.sock`
  - Optional: mount `/opt/applegpu` for Python package injection
- Start container, stream stdout/stderr, cleanup on exit

**Secondary path (vsock relay):**
- Register vsock listener via `VZVirtioSocketDevice.setSocketListenerForPort(5678)`
- On each connection: open Unix socket to `~/.applegpu/runtime.sock`, spawn bidirectional relay
- Container env: `APPLEGPU_TRANSPORT=vsock`, `APPLEGPU_VSOCK_PORT=5678`

**Legacy fallback:** Keep existing `container` CLI + TCP bridge code for when Containerization framework is unavailable. The `findContainerBinary()` function in `Run.swift` moves into the legacy fallback path only.

**Relay error handling:** If the gpu-service crashes or the Unix socket becomes unavailable during a vsock relay, the relay threads detect the read/write failure and shut down. The container-side client receives a broken pipe error from its vsock/TCP connection and can retry via `connect_auto()`. No automatic reconnection in the relay itself — the client is responsible for retry.

**Relay fix — double-close race:**
The current `TCPBridge.relay` has a double-close bug where both relay threads independently close the same file descriptors. Fix pattern:
1. When one direction's read returns 0 or error, call `shutdown(otherFd, SHUT_WR)`
2. Join both relay threads (use `DispatchGroup` or similar)
3. Close both fds once after both threads complete

### 4. SocketBackend integration

**File:** `crates/python/src/socket_backend.rs`

Update `SocketBackend::init()` to call `GpuClient::connect_auto()` instead of hardcoded `GpuClient::connect_unix()`. This makes the container-side Python package automatically detect the best available transport.

Also fix pre-existing bug: `VecDeque::new()` is called at line 170 but `VecDeque` is not imported (only `HashMap` and `HashSet` at line 11). The `stack` variable at line 170 is also unused — the actual DFS uses `visit_stack` at line 174. This is a compile error on Linux that needs fixing.

### 5. No changes to

- Wire crate (`crates/wire/src/lib.rs`) — protocol unchanged
- Python crate (`crates/python/src/lib.rs`) — dispatch layer unchanged
- Backend trait (`crates/python/src/backend.rs`) — interface unchanged
- MetalBackend (`crates/python/src/metal_backend.rs`) — macOS path unchanged

## Container-side Auto-detection

Inside the container, the Python client (via `SocketBackend` → `GpuClient::connect_auto()`) detects transport. This is the same order as the `connect_auto()` function in section 2:

1. `/dev/vsock` exists → `connect_vsock(CID=2, port=5678)` (Tier 2)
2. `APPLEGPU_SOCKET` env set → `connect_unix(path)` (Tier 1)
3. `APPLEGPU_HOST` + `APPLEGPU_PORT` env set → `connect_tcp(host, port)` (Tier 3)
4. Try default `/var/run/applegpu.sock` → `connect_unix` (Tier 1 fallback)
5. Nothing works → error with diagnostic message listing all transports tried

## Security

- **Unix socket mount**: File permissions (0600) restrict access to current user on host. Inside container, any process can connect — same as current model.
- **vsock**: Hypervisor-controlled channel between host and guest. No network exposure.
- **TCP bridge**: Currently listens on `INADDR_ANY` (`Run.swift:106`). In-scope fix: bind to container subnet (`192.168.64.1`) instead of all interfaces.
- **Future**: Token-based authentication in handshake for multi-tenant scenarios (backlog item, not blocking).

## Testing

### Rust unit tests
- `Transport` trait implementations (Unix, TCP, vsock mock)
- `connect_auto()` detection logic with mocked env vars and filesystem
- `GpuClient` with `Box<dyn Transport>` — verify handshake and eval still work

### Rust integration tests
- Existing `GpuClient` roundtrip tests pass after generalization
- TCP transport roundtrip test

### Swift tests
- ContainerRunner lifecycle (start/stop)
- vsock relay: connect, relay bytes, clean shutdown
- Double-close fix: verify fds closed exactly once

### End-to-end tests
- `gpu-container run` with Containerization framework + Unix socket mount
- `docker run -v ~/.applegpu/runtime.sock:/var/run/applegpu.sock` with `APPLEGPU_SOCKET` env
- Legacy: `container` CLI + TCP bridge still works

## Docker Documentation

With this design, Docker support is a one-liner:

```bash
docker run -v ~/.applegpu/runtime.sock:/var/run/applegpu.sock \
  -e APPLEGPU_SOCKET=/var/run/applegpu.sock \
  pytorch:latest python train.py
```

No TCP bridge, no port forwarding, no special networking. The gpu-service Unix socket is bind-mounted directly into the container.
