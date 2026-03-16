# Vsock Socket Relay Design

**Date:** 2026-03-16
**Status:** Approved
**Scope:** Replace TCP bridge with Unix socket relay via Apple Containerization framework. Three-tier fallback. Extract shared socket helpers. Fix async blocking.

## Overview

The current `gpu-container run` shells out to the `container` CLI and sets up a TCP bridge (port 7654) to forward connections from the Linux container to the gpu-service Unix socket on the host. This adds ~10ms latency per round-trip and requires managing a TCP relay process.

With macOS 26.2 and the `apple/containerization` Swift package (v0.10.0), we can use `UnixSocketConfiguration` to relay Unix sockets directly over vsock — the framework handles the relay internally. The container-side Python connects to a Unix socket (same as Docker bind-mount), not TCP.

## Environment

- macOS 26.2, SDK 26.2, Swift 6.2
- `container` CLI 0.10.0 installed via Homebrew
- Container runtime running (`container system status` → running)
- `container run alpine echo "hello"` works
- Bind-mounting directories works, but Unix sockets inside bind-mounts are NOT connectable from guest (confirmed by testing — socket inode visible but connection fails)

## Three-Tier Fallback Architecture

```
Run.swift → try Tier 1 → catch → try Tier 2 → catch → Tier 3
```

### Tier 1: Full Programmatic (macOS 26+)

Use `apple/containerization` Swift package for everything:
1. Pull OCI image via framework's registry client
2. Create `LinuxContainer` with `UnixSocketConfiguration` for gpu-service socket
3. Start container, wait for exit

```swift
import Containerization

let config = LinuxContainer.Configuration()
config.cpus = cpus
config.memoryInBytes = UInt64(memory) * 1024 * 1024
config.sockets = [
    UnixSocketConfiguration(
        source: URL(fileURLWithPath: socketPath),
        destination: URL(fileURLWithPath: "/var/run/applegpu.sock"),
        direction: .into
    )
]
config.process.environment["APPLEGPU_SOCKET"] = "/var/run/applegpu.sock"
config.process.arguments = command

let container = try await LinuxContainer(image: image, configuration: config)
try await container.start()
let status = try await container.wait()
```

**Risk:** The image pull API may differ from what's sketched. If it requires explicit registry client setup, OCI manifest resolution, or rootfs unpacking, this tier becomes complex.

### Tier 2: Hybrid (macOS 26+)

Shell out to `container pull <image>` for image caching, then use programmatic API for container lifecycle + socket relay.

```swift
// Pull image via CLI
let pullProcess = Process()
pullProcess.executableURL = URL(fileURLWithPath: containerBinary)
pullProcess.arguments = ["pull", image]
try pullProcess.run()
pullProcess.waitUntilExit()

// Then create container programmatically with socket relay
// (same as Tier 1 but image is already cached)
```

This gets the vsock socket relay benefit without reimplementing OCI registry client code.

### Tier 3: CLI + TCP Bridge (pre-macOS 26)

Existing path, unchanged:
1. Shell out to `container run` with `--env APPLEGPU_HOST=192.168.64.1 --env APPLEGPU_PORT=7654`
2. `TCPBridge` relays TCP port 7654 → Unix socket

## Data Flow (Tier 1/2)

```
Container (Linux guest)         Host (macOS)
┌────────────────────┐          ┌───────────────────────┐
│ Python code        │          │  gpu-container         │
│  └─ applegpu       │──vsock──▶│  └─ ContainerRunner   │
│     (SocketBackend │  (auto)  │     └─ Containerization│
│      connects to   │          │        framework       │
│      /var/run/     │          │        relays to       │
│      applegpu.sock)│          │        ~/.applegpu/    │
└────────────────────┘          │        runtime.sock    │
                                └───────────────────────┘
```

Container-side Python uses `APPLEGPU_SOCKET=/var/run/applegpu.sock` — same env var as Docker bind-mount, same SocketBackend code path.

## Files Changed

### Modified

| File | Change |
|------|--------|
| `swift/GPUContainer/Package.swift` | Uncomment `apple/containerization` dep, bump swift-tools-version to 6.1 |
| `swift/GPUContainer/Sources/ContainerRunner.swift` | Implement Tiers 1 + 2 with real Containerization API |
| `swift/GPUContainer/Sources/Run.swift` | Fix `waitUntilExit()` async blocking, deduplicate `filteredCommand` |
| `swift/GPUContainer/Sources/VsockRelay.swift` | Add deprecation notice |

### Created

| File | Purpose |
|------|---------|
| `swift/GPUContainer/Sources/UnixSocketHelper.swift` | Shared `connectToUnixSocket()` + `relay()` extracted from TCPBridge, VsockRelay, ServiceManager |

## Specific Fixes

### Extract UnixSocketHelper

Three files duplicate the same relay and Unix socket connect code. Extract into shared helper:

```swift
enum UnixSocketHelper {
    /// Connect to a Unix domain socket, returning the fd.
    static func connect(to path: String) -> Int32? { ... }

    /// Bidirectional relay between two file descriptors.
    static func relay(from: Int32, to: Int32, group: DispatchGroup) { ... }
}
```

Update TCPBridge, VsockRelay, and ServiceManager to use the helper.

### Fix waitUntilExit() blocking async

In `Run.swift`, replace:
```swift
process.waitUntilExit()
```
With:
```swift
await withCheckedContinuation { continuation in
    process.terminationHandler = { _ in continuation.resume() }
}
```

### Make TCPBridge IP configurable

Replace hardcoded `192.168.64.1` with:
```swift
let hostIP = ProcessInfo.processInfo.environment["APPLEGPU_BRIDGE_HOST"] ?? "192.168.64.1"
```

### VsockRelay deprecation

Add to top of file:
```swift
/// @deprecated Use ContainerRunner with UnixSocketConfiguration instead.
/// Kept for potential AVF VM backend use. Remove once confirmed unnecessary.
```

## Testing Strategy

- **End-to-end test (Tier 1/2):** `gpu-container run python:3.11-slim -- python3 -c "import applegpu_runtime as gpu; ..."` with gpu-service running. Verify Add, Int32, Cast, Embedding work.
- **Fallback test:** Force Tier 3 by setting `APPLEGPU_FORCE_TCP=1` env var or by testing on a pre-macOS 26 system.
- **Socket helper unit tests (deferred):** Test `UnixSocketHelper.connect` and `relay` with temporary Unix sockets. No GPU needed.

## Not Included (deferred to backlog)

- Removing TCP bridge (kept for backward compat)
- Deleting VsockRelay.swift (kept with deprecation notice)
- Socket helper unit tests (no GPU needed, can add to CI later)

## Success Criteria

1. `gpu-container run python:3.11-slim -- python3 -c "import applegpu_runtime; ..."` works with vsock (no TCP bridge, no port)
2. Container connects via Unix socket at `/var/run/applegpu.sock`
3. F32 add, Int32 add, Cast+embedding, comparison ops all work from container
4. Falls back to TCP bridge gracefully on error
5. Existing Docker bind-mount path still works unchanged
