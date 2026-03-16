# Vsock Socket Relay Design

**Date:** 2026-03-16
**Updated:** 2026-03-16 (post API exploration)
**Status:** Approved
**Scope:** Replace TCP bridge with Unix socket relay via Apple Containerization framework. Three-tier fallback. Extract shared socket helpers.

## Overview

Replace the TCP bridge with direct Unix socket relay using `apple/containerization` Swift package. The `ContainerManager` provides a high-level API that handles image pulling, rootfs creation, VM setup, and container lifecycle. `UnixSocketConfiguration` relays Unix sockets over vsock.

## Environment

- macOS 26.2, SDK 26.2, Swift 6.2
- `container` CLI 0.10.0, runtime running
- Kernel: `vmlinux-6.18.5-177` at `~/Library/Application Support/com.apple.container/kernels/`
- Init image: `ghcr.io/apple/containerization/vminit:0.26.5`

## Verified API (from actual source at .build/checkouts/containerization/)

### ContainerManager — High-Level API

```swift
import Containerization

// 1. Create manager with kernel + initfs + network
var manager = try await ContainerManager(
    kernel: Kernel(
        path: URL(fileURLWithPath: kernelPath),
        platform: .linuxArm
    ),
    initfsReference: "ghcr.io/apple/containerization/vminit:0.26.5",
    network: try ContainerManager.VmnetNetwork()
)

// 2. Create container from image reference (handles pull + rootfs)
let container = try await manager.create(
    "gpu-run-\(UUID().uuidString.prefix(8))",
    reference: "python:3.11-slim",
    rootfsSizeInBytes: 8.gib()
) { @Sendable config in
    config.cpus = cpus
    config.memoryInBytes = UInt64(memory).mib()
    config.sockets = [
        UnixSocketConfiguration(
            source: URL(fileURLWithPath: socketPath),
            destination: URL(fileURLWithPath: "/var/run/applegpu.sock"),
            direction: .into
        )
    ]
    config.process.environmentVariables.append("APPLEGPU_SOCKET=/var/run/applegpu.sock")
    config.process.arguments = command
}

// 3. Lifecycle: create → start → wait → stop (stop MUST be called)
try await container.create()
try await container.start()
let status = try await container.wait()
try await container.stop()
try manager.delete(containerId)
```

### Key API Facts (verified against source)
- `LinuxProcessConfiguration.environmentVariables` is `[String]` (OCI-style `KEY=VALUE`), NOT a dictionary
- `container.stop()` MUST be called even after `wait()` returns (per doc comments)
- `manager.delete(id)` cleans up rootfs and resources
- `ContainerManager.VmnetNetwork()` provides NAT networking with vmnet
- Kernel path: `~/Library/Application Support/com.apple.container/kernels/default.kernel-arm64`
- Init image reference: `ghcr.io/apple/containerization/vminit:0.26.5`

## Three-Tier Fallback Architecture

### Tier 1: Full Programmatic (macOS 26+)

`ContainerManager` handles everything — image pull, rootfs, VMM, socket relay:

```swift
@available(macOS 26, *)
enum ContainerRunner {
    static func run(image: String, cpus: Int, memory: Int,
                    socketPath: String, command: [String]) async throws {
        let kernelPath = Self.findKernel()
        var manager = try await ContainerManager(
            kernel: Kernel(path: URL(fileURLWithPath: kernelPath), platform: .linuxArm),
            initfsReference: "ghcr.io/apple/containerization/vminit:0.26.5",
            network: try ContainerManager.VmnetNetwork()
        )

        let containerId = "gpu-\(UUID().uuidString.prefix(8))"
        let container = try await manager.create(
            containerId, reference: image
        ) { @Sendable config in
            config.cpus = cpus
            config.memoryInBytes = UInt64(memory) * 1024 * 1024
            config.sockets = [
                UnixSocketConfiguration(
                    source: URL(fileURLWithPath: socketPath),
                    destination: URL(fileURLWithPath: "/var/run/applegpu.sock"),
                    direction: .into
                )
            ]
            config.process.environmentVariables.append("APPLEGPU_SOCKET=/var/run/applegpu.sock")
            config.process.arguments = command.filter { $0 != "--" }
        }
        defer { try? manager.delete(containerId) }

        try await container.create()
        try await container.start()
        let status = try await container.wait()
        try await container.stop()

        if case .exited(let code) = status, code != 0 {
            throw GPUContainerError.containerExited(code: Int(code))
        }
    }

    static func findKernel() -> String {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let kernelPath = appSupport.appendingPathComponent("com.apple.container/kernels/default.kernel-arm64").path
        return kernelPath
    }
}
```

### Tier 2: Hybrid (macOS 26+, image pull fallback)

If Tier 1's image pull fails (registry auth, network, etc.), shell out to `container pull` then retry with cached image:

```swift
// In the catch block of Tier 1, if error is image-pull related:
let pullProcess = Process()
pullProcess.executableURL = URL(fileURLWithPath: containerBinary)
pullProcess.arguments = ["pull", image]
try pullProcess.run()
// ... then retry manager.create() which will find the cached image
```

### Tier 3: CLI + TCP Bridge (pre-macOS 26 fallback)

Existing path, unchanged. Shell out to `container run` with TCP bridge.

## Error Classification for Fallback

| Error Type | Fallback? | Rationale |
|-----------|-----------|-----------|
| Containerization framework not available | Yes → Tier 3 | Pre-macOS 26 |
| Image pull failed (registry, auth, network) | Yes → Tier 2 | Try CLI pull |
| Kernel not found | Yes → Tier 3 | container CLI handles its own kernel |
| Container exited non-zero | No | Real application error |
| Socket path not found | No | Configuration error |
| Permission denied | No | Real error |
| VmnetNetwork init failed | Yes → Tier 3 | Networking config issue |

## Files Changed

| File | Change |
|------|--------|
| `swift/GPUContainer/Package.swift` | swift-tools-version 6.1, macOS 26.0 platform, uncomment containerization dep |
| `swift/GPUContainer/Sources/ContainerRunner.swift` | Full implementation with Tier 1 + 2 |
| `swift/GPUContainer/Sources/Run.swift` | Fix async blocking, add `APPLEGPU_FORCE_TCP` env var |
| `swift/GPUContainer/Sources/VsockRelay.swift` | Add `@available(*, deprecated)` |
| `swift/GPUContainer/Sources/UnixSocketHelper.swift` | New — extracted relay + connect code |

## Other Fixes

- **Extract `UnixSocketHelper`** — deduplicate relay + connect across TCPBridge, VsockRelay, ServiceManager
- **Fix `waitUntilExit()` blocking** — set `terminationHandler` before `process.run()`
- **Make TCPBridge IP configurable** — `APPLEGPU_BRIDGE_HOST` env var (default `192.168.64.1`)
- **Use `@available(*, deprecated)`** on VsockRelay — proper Swift deprecation
- **Add `APPLEGPU_FORCE_TCP`** env var to force Tier 3 for testing

## Testing

1. `gpu-container run python:3.11-slim -- python3 -c "import applegpu_runtime as gpu; gpu.init_backend(); print((gpu.tensor([1,2,3]) + gpu.tensor([4,5,6])).to_list())"` — Tier 1 vsock
2. `APPLEGPU_FORCE_TCP=1 gpu-container run ...` — verify Tier 3 still works
3. Docker bind-mount path unchanged — verify still works

## Success Criteria

1. `gpu-container run` works without TCP bridge (vsock socket relay)
2. Container connects via Unix socket at `/var/run/applegpu.sock`
3. All ops work: F32 add, Int32, Cast+embedding, comparison
4. Falls back gracefully to TCP bridge when vsock unavailable
5. Docker bind-mount path still works unchanged
