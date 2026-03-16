# Socket Relay & vsock Transport Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the TCP bridge with Unix socket mounting (primary) and vsock relay (secondary), making container GPU access direct and lower-latency.

**Architecture:** Three transport tiers converge on the same gpu-service wire protocol: (1) Unix socket mount via Containerization framework or Docker, (2) vsock relay through the Swift gpu-container process, (3) legacy TCP bridge. The client crate's `GpuClient` is type-erased over a `Transport` trait so all transports use the same client API.

**Tech Stack:** Rust (applegpu-client, applegpu-wire), Swift (GPUContainer CLI, Apple Containerization framework)

**Spec:** `docs/superpowers/specs/2026-03-16-socket-relay-vsock-design.md`

---

## Chunk 1: Client Transport Generalization (Rust)

### Task 1: Add `shutdown` method to Transport trait and implement for UnixStream

**Files:**
- Modify: `crates/client/src/transport.rs`

- [ ] **Step 1: Write failing test**

Add to `crates/client/src/transport.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixStream as StdUnixStream;

    #[test]
    fn unix_transport_shutdown() {
        let (a, _b) = StdUnixStream::pair().unwrap();
        let transport: Box<dyn Transport> = Box::new(a);
        assert!(transport.shutdown().is_ok());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-client -- transport::tests::unix_transport_shutdown`
Expected: FAIL — `shutdown` method does not exist on `Transport` trait

- [ ] **Step 3: Implement shutdown on Transport trait**

Replace the current `Transport` trait in `crates/client/src/transport.rs` (line 4-5):

```rust
/// Trait for a bidirectional byte stream transport.
pub trait Transport: Read + Write + Send {
    /// Shut down the transport (both read and write halves).
    fn shutdown(&self) -> io::Result<()>;
}

impl Transport for UnixStream {
    fn shutdown(&self) -> io::Result<()> {
        UnixStream::shutdown(self, std::net::Shutdown::Both)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p applegpu-client -- transport::tests::unix_transport_shutdown`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/client/src/transport.rs
git commit -m "feat: add shutdown method to Transport trait

Prepares for type-erased transport in GpuClient.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 2: Add TcpStream Transport implementation

**Files:**
- Modify: `crates/client/src/transport.rs`

- [ ] **Step 1: Write failing test**

Add to `crates/client/src/transport.rs` tests:
```rust
#[test]
fn tcp_transport_implements_trait() {
    use std::net::{TcpListener, TcpStream};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let stream = TcpStream::connect(addr).unwrap();
    let _accepted = listener.accept().unwrap();
    let transport: Box<dyn Transport> = Box::new(stream);
    assert!(transport.shutdown().is_ok());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-client -- transport::tests::tcp_transport_implements_trait`
Expected: FAIL — `TcpStream` does not implement `Transport`

- [ ] **Step 3: Implement Transport for TcpStream**

Add to `crates/client/src/transport.rs`:
```rust
use std::net::TcpStream;

impl Transport for TcpStream {
    fn shutdown(&self) -> io::Result<()> {
        TcpStream::shutdown(self, std::net::Shutdown::Both)
    }
}

/// Connect via TCP.
pub fn connect_tcp(host: &str, port: u16) -> io::Result<TcpStream> {
    TcpStream::connect((host, port))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p applegpu-client -- transport::tests::tcp_transport_implements_trait`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/client/src/transport.rs
git commit -m "feat: add TcpStream Transport impl and connect_tcp

Tier 3 (legacy TCP bridge) transport support.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 3: Replace vsock UnixStream hack with vsock crate

**Files:**
- Modify: `crates/client/Cargo.toml`
- Modify: `crates/client/src/transport.rs`

- [ ] **Step 1: Add vsock dependency**

Add to `crates/client/Cargo.toml`:
```toml
[target.'cfg(target_os = "linux")'.dependencies]
vsock = "0.4"
```

- [ ] **Step 2: Replace connect_vsock implementation**

Replace the entire `connect_vsock` function (lines 16-64 of `transport.rs`) with:

```rust
/// Connect via vsock (AF_VSOCK). Only available on Linux.
/// CID 2 = host. Returns a boxed Transport for uniform handling.
#[cfg(target_os = "linux")]
pub fn connect_vsock(cid: u32, port: u32) -> io::Result<Box<dyn Transport>> {
    let stream = vsock::VsockStream::connect_with_cid_port(cid, port)?;
    Ok(Box::new(stream))
}

#[cfg(target_os = "linux")]
impl Transport for vsock::VsockStream {
    fn shutdown(&self) -> io::Result<()> {
        // vsock crate may not expose shutdown directly; use raw fd fallback
        use std::os::unix::io::AsRawFd;
        let ret = unsafe { libc::shutdown(self.as_raw_fd(), libc::SHUT_RDWR) };
        if ret < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn connect_vsock(_cid: u32, _port: u32) -> io::Result<Box<dyn Transport>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "vsock is only available on Linux guests",
    ))
}
```

Keep `libc` in `Cargo.toml` — still needed for vsock shutdown on Linux. Update the target cfg:
```toml
[target.'cfg(target_os = "linux")'.dependencies]
libc = "0.2"
vsock = "0.4"
```

- [ ] **Step 3: Run all client tests**

Run: `cargo test -p applegpu-client`
Expected: All existing tests pass (vsock is Linux-only, tests run on macOS via Unix streams)

- [ ] **Step 4: Commit**

```bash
git add crates/client/
git commit -m "feat: replace vsock UnixStream hack with vsock crate

Removes hand-rolled AF_VSOCK/SockaddrVm code and libc dependency.
Uses vsock crate's VsockStream which properly implements Read + Write.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 4: Type-erase GpuClient stream field to Box<dyn Transport>

**Files:**
- Modify: `crates/client/src/lib.rs`

- [ ] **Step 1: Write failing test**

Add to `crates/client/src/lib.rs` tests:
```rust
#[test]
fn gpu_client_works_with_boxed_transport() {
    let (client_stream, server_stream) = UnixStream::pair().unwrap();
    let server = thread::spawn(move || mock_gpu_service(server_stream));

    // Test that handshake works with Box<dyn Transport>
    let boxed: Box<dyn transport::Transport> = Box::new(client_stream);
    let mut client = GpuClient::handshake(boxed, 1024 * 1024).unwrap();
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-client -- gpu_client_works_with_boxed_transport`
Expected: FAIL — `handshake` expects `UnixStream`, not `Box<dyn Transport>`

- [ ] **Step 3: Change GpuClient to use Box<dyn Transport>**

In `crates/client/src/lib.rs`, make these changes:

1. Change the import (line 4-5):
```rust
// Remove:
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
// Add:
use crate::transport::Transport;
```

2. Change the struct field (line 41-45):
```rust
pub struct GpuClient {
    stream: Box<dyn Transport>,
    pub container_id: u64,
    pub granted_memory: u64,
}
```

3. Change `handshake` signature (line 58):
```rust
fn handshake(mut stream: Box<dyn Transport>, requested_memory: u64) -> Result<Self> {
```
Body unchanged — `wire::write_message` and `wire::read_message` take `impl Write`/`impl Read`.

4. Change `connect_unix` (line 48-51):
```rust
pub fn connect_unix(path: &str, requested_memory: u64) -> Result<Self> {
    let stream = transport::connect_unix(path)?;
    Self::handshake(Box::new(stream), requested_memory)
}
```

5. Change `connect_vsock` (line 53-56). Note: `connect_vsock` now returns `Box<dyn Transport>` directly, so no `Box::new()` wrapping needed:
```rust
pub fn connect_vsock(port: u32, requested_memory: u64) -> Result<Self> {
    let stream = transport::connect_vsock(2, port)?;
    Self::handshake(stream, requested_memory) // already Box<dyn Transport>
}
```

6. Change `Drop` (line 102-106):
```rust
impl Drop for GpuClient {
    fn drop(&mut self) {
        let _ = self.stream.shutdown();
    }
}
```

7. Add `connect_tcp`:
```rust
pub fn connect_tcp(host: &str, port: u16, requested_memory: u64) -> Result<Self> {
    let stream = transport::connect_tcp(host, port)?;
    Self::handshake(Box::new(stream), requested_memory)
}
```

- [ ] **Step 4: Fix existing tests**

The existing tests use `UnixStream::pair()` and pass raw `UnixStream` to `handshake`. Update each test's `GpuClient::handshake(...)` call to wrap in `Box::new(...)`:

```rust
// Change all occurrences of:
let mut client = GpuClient::handshake(client_stream, 1024 * 1024).unwrap();
// To:
let mut client = GpuClient::handshake(Box::new(client_stream), 1024 * 1024).unwrap();
```

Also add `use std::os::unix::net::UnixStream;` to the test module since it's no longer in the top-level imports.

- [ ] **Step 5: Run all tests**

Run: `cargo test -p applegpu-client`
Expected: ALL tests pass (4 existing + 1 new)

- [ ] **Step 6: Commit**

```bash
git add crates/client/src/lib.rs
git commit -m "refactor: type-erase GpuClient stream to Box<dyn Transport>

GpuClient now works with any transport (Unix, TCP, vsock).
No cascading changes needed in SocketBackend — it uses GpuClient opaquely.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 5: Add connect_auto() with transport auto-detection

**Files:**
- Modify: `crates/client/src/lib.rs`

- [ ] **Step 1: Write failing test**

Add to `crates/client/src/lib.rs` tests:
```rust
#[test]
fn connect_auto_falls_back_to_unix_socket() {
    // connect_auto should try env vars and fallback paths.
    // With no env vars set and no /dev/vsock, it should try
    // the default /var/run/applegpu.sock (which won't exist in test).
    // We just verify it returns an error with a helpful message.
    std::env::remove_var("APPLEGPU_SOCKET");
    std::env::remove_var("APPLEGPU_HOST");
    std::env::remove_var("APPLEGPU_PORT");
    let result = GpuClient::connect_auto(1024 * 1024);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("No transport available") || err.contains("Connection refused") || err.contains("No such file"),
        "Expected transport error, got: {}", err
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p applegpu-client -- connect_auto_falls_back`
Expected: FAIL — `connect_auto` does not exist

- [ ] **Step 3: Implement connect_auto**

Add to `GpuClient` impl in `crates/client/src/lib.rs`:

```rust
/// Auto-detect the best available transport and connect.
///
/// Detection order:
/// 1. /dev/vsock exists → vsock (CID=2, port from APPLEGPU_VSOCK_PORT or 5678)
/// 2. APPLEGPU_SOCKET env → Unix socket at that path
/// 3. APPLEGPU_HOST + APPLEGPU_PORT env → TCP
/// 4. Default /var/run/applegpu.sock → Unix socket
/// 5. Error with diagnostic message
pub fn connect_auto(requested_memory: u64) -> Result<Self> {
    // 1. vsock
    if std::path::Path::new("/dev/vsock").exists() {
        let port: u32 = std::env::var("APPLEGPU_VSOCK_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_VSOCK_PORT);
        match Self::connect_vsock(port, requested_memory) {
            Ok(client) => return Ok(client),
            Err(_) => {} // fall through to next transport
        }
    }

    // 2. APPLEGPU_SOCKET env
    if let Ok(path) = std::env::var("APPLEGPU_SOCKET") {
        return Self::connect_unix(&path, requested_memory);
    }

    // 3. APPLEGPU_HOST + APPLEGPU_PORT env → TCP
    if let (Ok(host), Ok(port_str)) = (
        std::env::var("APPLEGPU_HOST"),
        std::env::var("APPLEGPU_PORT"),
    ) {
        if let Ok(port) = port_str.parse::<u16>() {
            return Self::connect_tcp(&host, port, requested_memory);
        }
    }

    // 4. Default Unix socket path
    let default_path = "/var/run/applegpu.sock";
    if std::path::Path::new(default_path).exists() {
        return Self::connect_unix(default_path, requested_memory);
    }

    // 5. Nothing found
    Err(ClientError::Protocol(
        "No transport available. Set APPLEGPU_SOCKET, APPLEGPU_HOST+APPLEGPU_PORT, \
         or ensure /var/run/applegpu.sock exists."
            .to_string(),
    ))
}
```

- [ ] **Step 4: Run all tests**

Run: `cargo test -p applegpu-client`
Expected: ALL tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/client/src/lib.rs
git commit -m "feat: add connect_auto() for transport auto-detection

Tries vsock → APPLEGPU_SOCKET → APPLEGPU_HOST:PORT → default path.
Order is trivially reorderable after benchmarking.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 6: Update SocketBackend to use connect_auto and fix VecDeque bug

**Files:**
- Modify: `crates/python/src/socket_backend.rs`

- [ ] **Step 1: Fix VecDeque compile bug**

In `crates/python/src/socket_backend.rs`:

Delete only line 170: `let mut stack = VecDeque::new();`

The `visit_stack` at line 174 is the actual DFS stack used. The `stack` variable is unused dead code, and `VecDeque` is not imported (only `HashMap` and `HashSet` at line 11). This would cause a compile error on Linux.

- [ ] **Step 2: Update init() to use connect_auto**

In `crates/python/src/socket_backend.rs`, change `init()` (lines 210-238):

Replace:
```rust
let socket_path = std::env::var("APPLEGPU_SOCKET")
    .unwrap_or_else(|_| "/var/run/applegpu.sock".to_string());

let requested_memory: u64 = std::env::var("APPLEGPU_MEMORY_MB")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(4096)
    * 1024
    * 1024;

let client = GpuClient::connect_unix(&socket_path, requested_memory)
    .map_err(|e| format!("Failed to connect to gpu-service at {}: {}", socket_path, e))?;
```

With:
```rust
let requested_memory: u64 = std::env::var("APPLEGPU_MEMORY_MB")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(4096)
    * 1024
    * 1024;

let client = GpuClient::connect_auto(requested_memory)
    .map_err(|e| format!("Failed to connect to gpu-service: {}", e))?;
```

Update the info map — remove the hardcoded `socket_path` key since transport is now auto-detected:
```rust
let mut info = HashMap::new();
info.insert("backend".to_string(), "socket".to_string());
info.insert("container_id".to_string(), client.container_id.to_string());
info.insert("granted_memory".to_string(), client.granted_memory.to_string());
```

- [ ] **Step 3: Verify Rust compilation**

Note: `socket_backend.rs` is `#[cfg(target_os = "linux")]` so it won't compile on macOS. Verify the client crate instead:

Run: `cargo test -p applegpu-client`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/python/src/socket_backend.rs
git commit -m "fix: use connect_auto in SocketBackend, remove VecDeque bug

SocketBackend now auto-detects transport (vsock/unix/tcp).
Removed unused VecDeque::new() that would fail to compile on Linux.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 2: Swift GPU Container Refactor

### Task 7: Refactor Run.swift — extract legacy path, add ContainerRunner skeleton

**Files:**
- Create: `swift/GPUContainer/Sources/ContainerRunner.swift`
- Modify: `swift/GPUContainer/Sources/Run.swift`
- Modify: `swift/GPUContainer/Package.swift`

> **Note:** The Containerization framework API is speculative — the exact types and method signatures depend on the macOS 26 SDK. This task creates the structure and wires up the `#available` guard. The `ContainerRunner` implementation will need adjustment once the SDK is available. The `Run.swift` refactor to extract `runWithContainerCLI` is independent and can proceed regardless.

- [ ] **Step 1a: Update swift-tools-version**

In `swift/GPUContainer/Package.swift` line 1, change:
```swift
// swift-tools-version: 6.0
```
to:
```swift
// swift-tools-version: 6.1
```

macOS 26 platform version requires Swift 6.1 toolchain.

- [ ] **Step 1b: Bump platform target and add Containerization dependency**

In `swift/GPUContainer/Package.swift`, change:
```swift
platforms: [.macOS(.v26)],
```

This drops macOS 15-25 support for the `gpu-container` binary. The gpu-service Rust binary and Python package are unaffected.

Add the Containerization framework dependency:
```swift
dependencies: [
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    .package(url: "https://github.com/apple/containerization.git", from: "0.1.0"),
],
targets: [
    .executableTarget(
        name: "gpu-container",
        dependencies: [
            .product(name: "ArgumentParser", package: "swift-argument-parser"),
            .product(name: "Containerization", package: "containerization"),
        ],
        path: "Sources"
    ),
],
```

- [ ] **Step 2: Create ContainerRunner.swift**

Create `swift/GPUContainer/Sources/ContainerRunner.swift`:

```swift
import Foundation
import Containerization

/// Runs containers using Apple's Containerization framework with Unix socket relay.
/// Primary transport: UnixSocketConfiguration mounts gpu-service socket into container.
@available(macOS 26, *)
enum ContainerRunner {
    static func run(
        image: String,
        cpus: Int,
        memory: Int,
        socketPath: String,
        command: [String]
    ) async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path

        var config = LinuxContainer.Configuration()
        config.cpus = cpus
        config.memoryInBytes = UInt64(memory) * 1024 * 1024

        // Socket relay: mount host gpu-service socket into container
        config.sockets = [
            UnixSocketConfiguration(
                source: URL(fileURLWithPath: socketPath),
                destination: URL(fileURLWithPath: "/var/run/applegpu.sock"),
                direction: .into
            )
        ]

        // Environment
        config.process.environment["APPLEGPU_SOCKET"] = "/var/run/applegpu.sock"

        // Set the command
        let filteredCommand = command.filter { $0 != "--" }
        if !filteredCommand.isEmpty {
            config.process.arguments = filteredCommand
        }

        print("Starting container from \(image)...")
        print("  CPUs: \(cpus), Memory: \(memory)MB")
        print("  GPU: \(socketPath) → /var/run/applegpu.sock (direct socket mount)")

        let container = try await LinuxContainer(image: image, configuration: config)
        try await container.start()
        try await container.waitUntilExit()
    }
}
```

Note: The exact Containerization API may differ — adjust types and method names to match the actual framework when building on macOS 26.

- [ ] **Step 3: Update Run.swift to try Containerization framework first**

In `swift/GPUContainer/Sources/Run.swift`, change the `run()` method to:

```swift
func run() async throws {
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    let socketPath = "\(home)/.applegpu/runtime.sock"

    // Auto-start gpu-service
    print("Ensuring gpu-service is running...")
    try await ServiceManager.ensureRunning(socketPath: socketPath)

    // Try Containerization framework first (macOS 26+, direct socket mount)
    if #available(macOS 26, *) {
        do {
            try await ContainerRunner.run(
                image: image,
                cpus: cpus,
                memory: memory,
                socketPath: socketPath,
                command: command
            )
            return
        } catch {
            print("Containerization framework unavailable, falling back to container CLI...")
            print("  Error: \(error)")
        }
    }

    // Fallback: container CLI + TCP bridge
    try await runWithContainerCLI(socketPath: socketPath)
}

/// Legacy fallback: uses Apple's `container` CLI with TCP bridge.
private func runWithContainerCLI(socketPath: String) async throws {
    print("Starting TCP bridge on port \(gpuPort)...")
    let bridge = try TCPBridge(port: gpuPort, socketPath: socketPath)
    bridge.start()

    var args: [String] = ["run", "--rm"]
    args += ["--cpus", "\(cpus)"]
    args += ["--memory", "\(memory)M"]
    args += ["--env", "APPLEGPU_HOST=192.168.64.1"]
    args += ["--env", "APPLEGPU_PORT=\(gpuPort)"]
    args += [image]

    let filteredCommand = command.filter { $0 != "--" }
    if !filteredCommand.isEmpty {
        args += filteredCommand
    }

    print("Starting container from \(image)...")
    print("  CPUs: \(cpus), Memory: \(memory)MB")
    print("  GPU bridge: 192.168.64.1:\(gpuPort) → \(socketPath)")

    guard let containerBin = findContainerBinary() else {
        throw GPUContainerError.containerCliNotFound
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: containerBin)
    process.arguments = args
    process.standardOutput = FileHandle.standardOutput
    process.standardError = FileHandle.standardError
    process.standardInput = FileHandle.standardInput

    try process.run()
    process.waitUntilExit()

    bridge.stop()

    let exitCode = process.terminationStatus
    if exitCode != 0 {
        throw ExitCode(exitCode)
    }
}
```

Move `findContainerBinary()` into the `runWithContainerCLI` scope or keep it as a private method (used only by the legacy path).

- [ ] **Step 4: Build**

Run: `cd swift/GPUContainer && swift build`
Expected: Compiles (may need macOS 26 SDK — if unavailable, the `#available` guard ensures the legacy path still compiles)

- [ ] **Step 5: Commit**

```bash
git add swift/GPUContainer/
git commit -m "feat: add ContainerRunner with Containerization framework socket relay

Primary path uses UnixSocketConfiguration to mount gpu-service socket
directly into the container. Falls back to container CLI + TCP bridge.
Requires macOS 26 for Containerization framework.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 8: Fix TCPBridge double-close race condition

**Files:**
- Modify: `swift/GPUContainer/Sources/Run.swift`

- [ ] **Step 1: Fix the relay method**

Replace the `relay` method in `TCPBridge` (lines 166-184 of `Run.swift`):

```swift
private func relay(from src: Int32, to dst: Int32, group: DispatchGroup) {
    group.enter()
    Thread.detachNewThread {
        defer { group.leave() }
        var buf = [UInt8](repeating: 0, count: 65536)
        while true {
            let n = read(src, &buf, buf.count)
            if n <= 0 {
                // Signal the other direction to stop
                Darwin.shutdown(dst, SHUT_WR)
                break
            }
            var written = 0
            while written < n {
                let w = buf.withUnsafeBufferPointer { ptr in
                    write(dst, ptr.baseAddress! + written, n - written)
                }
                if w <= 0 {
                    Darwin.shutdown(src, SHUT_RD)
                    return
                }
                written += w
            }
        }
    }
}
```

- [ ] **Step 2: Update the accept loop to use DispatchGroup for fd lifecycle**

In the `start()` method, replace the relay calls (lines 154-155) with:

```swift
// Bidirectional relay with proper shutdown
let group = DispatchGroup()
self.relay(from: clientFd, to: unixFd, group: group)
self.relay(from: unixFd, to: clientFd, group: group)

// Close both fds after both relay threads complete
Thread.detachNewThread {
    group.wait()
    close(clientFd)
    close(unixFd)
}
```

- [ ] **Step 3: Build and verify**

Run: `cd swift/GPUContainer && swift build`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add swift/GPUContainer/Sources/Run.swift
git commit -m "fix: TCPBridge double-close race condition

Use shutdown() to signal relay direction, DispatchGroup to join threads,
close fds exactly once after both directions complete.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 9: Tighten TCP bridge bind address

**Files:**
- Modify: `swift/GPUContainer/Sources/Run.swift`

- [ ] **Step 1: Change INADDR_ANY to container subnet**

In `TCPBridge.start()`, change line 106:

```swift
// Replace:
addr.sin_addr.s_addr = INADDR_ANY
// With:
addr.sin_addr.s_addr = inet_addr("192.168.64.1")
```

This binds only to the container subnet interface instead of all interfaces.

- [ ] **Step 2: Build**

Run: `cd swift/GPUContainer && swift build`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add swift/GPUContainer/Sources/Run.swift
git commit -m "security: bind TCP bridge to container subnet only

Was INADDR_ANY (all interfaces), now 192.168.64.1 (container subnet).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 3: vsock Relay and Documentation

### Task 10: Add vsock relay to ContainerRunner (DEFERRED — requires macOS 26 SDK)

> **DEFERRED:** This task requires the macOS 26 SDK and access to `VZVirtioSocketDevice` (from Apple's `Virtualization` framework, NOT the `Containerization` Swift package). The `VZ`-prefixed types use a delegate pattern (`VZVirtioSocketListenerDelegate`), not closures. Implementation depends on verifying the actual API surface. Skip this task until the SDK is available and add `import Virtualization` + the `com.apple.security.virtualization` entitlement when implementing.

**Files:**
- Modify: `swift/GPUContainer/Sources/ContainerRunner.swift`

- [ ] **Step 1: Verify VZVirtioSocketDevice API is available**

Check if macOS 26 SDK provides `VZVirtioSocketDevice` and `VZVirtioSocketListener`. If not available, skip this task.

- [ ] **Step 2: Implement vsock relay using VZVirtioSocketListenerDelegate**

The relay pattern is the same as the fixed TCPBridge relay (shutdown + DispatchGroup + single close), but accepting connections from `VZVirtioSocketDevice` instead of TCP. Use the delegate pattern:

```swift
import Virtualization

@available(macOS 26, *)
class VsockRelayListener: NSObject, VZVirtioSocketListenerDelegate {
    let socketPath: String

    init(socketPath: String) {
        self.socketPath = socketPath
    }

    func listener(_ listener: VZVirtioSocketListener, shouldAcceptNewConnection connection: VZVirtioSocketConnection, from device: VZVirtioSocketDevice) -> Bool {
        // Connect to gpu-service Unix socket and start relay
        // ... (same relay pattern as TCPBridge with DispatchGroup)
        return true
    }
}
```

- [ ] **Step 3: Build and test when SDK available**

- [ ] **Step 4: Commit**

```bash
git add swift/GPUContainer/Sources/ContainerRunner.swift
git commit -m "feat: add vsock relay to ContainerRunner

Tier 2 transport: VZVirtioSocketListener relays vsock connections
from container to gpu-service Unix socket.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 11: Add Docker documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add Docker GPU access section**

Add after the "Container GPU Access" section in `README.md`. The section should have:

- H3 heading: `### Docker GPU Access`
- Intro: "Docker containers can access Metal GPU by bind-mounting the gpu-service socket:"
- A bash code block with:
  - `cargo run -p applegpu-service` (start gpu-service)
  - `docker run -v ~/.applegpu/runtime.sock:/var/run/applegpu.sock -e APPLEGPU_SOCKET=/var/run/applegpu.sock pytorch:latest python -c "import applegpu_runtime as gpu; gpu.init_backend(); a = gpu.tensor([1.0, 2.0, 3.0]); print((a + a).to_list())"`
- Footer: "No TCP bridge, no port forwarding, no special networking required."

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Docker GPU access via socket bind-mount

One-liner docker run with -v socket mount, no TCP bridge needed.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 12: Update BACKLOG to reflect completed work

**Files:**
- Modify: `docs/BACKLOG.md`

- [ ] **Step 1: Update backlog**

In `docs/BACKLOG.md`, under "PRIORITY 1: Replace TCP bridge with Unix socket relay / vsock":
- Check off completed items (Unix socket relay, vsock transport, connect_auto, TCP bridge tightened)
- Move "Remove TCP bridge" to future cleanup (kept as legacy fallback)
- Check off "Docker bind-mount documentation"

- [ ] **Step 2: Commit**

```bash
git add docs/BACKLOG.md
git commit -m "docs: update backlog for socket relay implementation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
