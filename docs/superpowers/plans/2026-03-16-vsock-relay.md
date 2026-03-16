# Vsock Socket Relay Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the TCP bridge with Unix socket relay via Apple Containerization framework, with three-tier fallback (full programmatic → hybrid CLI pull → CLI + TCP bridge).

**Architecture:** `ContainerRunner` uses `ContainerManager` from `apple/containerization` to create containers with `UnixSocketConfiguration` for socket relay over vsock. `Run.swift` dispatches to `ContainerRunner` first, falls back to CLI + TCP bridge for pre-macOS 26. Shared socket code extracted to `UnixSocketHelper`.

**Tech Stack:** Swift 6.1, apple/containerization package, Virtualization framework, vmnet

**Spec:** `docs/superpowers/specs/2026-03-16-vsock-relay-design.md`

---

## Chunk 1: Package Setup + Shared Helpers

### Task 1: Update Package.swift for Containerization dependency

**Files:**
- Modify: `swift/GPUContainer/Package.swift`

- [ ] **Step 1: Update Package.swift**

```swift
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "GPUContainer",
    platforms: [.macOS("26.0")],
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
    ]
)
```

Key changes: swift-tools-version 6.1, platform macOS 26.0, containerization dependency uncommented, added to target dependencies.

- [ ] **Step 2: Resolve dependencies**

Run: `cd swift/GPUContainer && swift package resolve`
Expected: Dependencies fetched successfully

- [ ] **Step 3: Verify build compiles**

Run: `cd swift/GPUContainer && swift build 2>&1 | tail -5`
Expected: Build may have warnings about unused imports but should succeed

- [ ] **Step 4: Commit**

```bash
git add swift/GPUContainer/Package.swift swift/GPUContainer/Package.resolved
git commit -m "feat: add apple/containerization dependency, bump to Swift 6.1"
```

### Task 2: Extract UnixSocketHelper

**Files:**
- Create: `swift/GPUContainer/Sources/UnixSocketHelper.swift`
- Modify: `swift/GPUContainer/Sources/Run.swift` (TCPBridge uses helper)
- Modify: `swift/GPUContainer/Sources/VsockRelay.swift` (uses helper)
- Modify: `swift/GPUContainer/Sources/ServiceManager.swift` (uses helper)

- [ ] **Step 1: Create UnixSocketHelper.swift**

```swift
import Foundation

/// Shared low-level Unix socket utilities used by TCPBridge, VsockRelay, and ServiceManager.
enum UnixSocketHelper {
    /// Connect to a Unix domain socket. Returns the file descriptor, or nil on failure.
    static func connect(to path: String) -> Int32? {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = path.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            close(fd)
            return nil
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                for (i, byte) in pathBytes.enumerated() {
                    dest[i] = byte
                }
            }
        }

        let addrLen = socklen_t(MemoryLayout<sockaddr_un>.size)
        let result = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Foundation.connect(fd, sockPtr, addrLen)
            }
        }

        if result != 0 {
            close(fd)
            return nil
        }
        return fd
    }

    /// Test connectivity to a Unix socket (connect + close). Returns true if connectable.
    static func testConnect(to path: String) -> Bool {
        guard let fd = connect(to: path) else { return false }
        close(fd)
        return true
    }

    /// Bidirectional relay between two file descriptors.
    /// Each direction runs on a background thread; the group tracks completion.
    static func relay(from src: Int32, to dst: Int32, group: DispatchGroup) {
        group.enter()
        DispatchQueue.global().async {
            defer { group.leave() }
            var buf = [UInt8](repeating: 0, count: 8192)
            while true {
                let n = read(src, &buf, buf.count)
                if n <= 0 { break }
                var written = 0
                while written < n {
                    let w = write(dst, &buf + written, n - written)
                    if w <= 0 { return }
                    written += w
                }
            }
            // Signal EOF to peer
            shutdown(dst, SHUT_WR)
        }
    }
}
```

- [ ] **Step 2: Update TCPBridge in Run.swift to use helper**

Replace the `connect(to socketPath:)` method and `relay(from:to:group:)` in TCPBridge with calls to `UnixSocketHelper.connect(to:)` and `UnixSocketHelper.relay(from:to:group:)`. Remove the duplicated implementations.

- [ ] **Step 3: Update VsockRelay.swift to use helper**

Replace the duplicated `connectToUnixSocket()` and `relay(from:to:group:)` with `UnixSocketHelper` calls. Add deprecation:

```swift
@available(*, deprecated, message: "Use ContainerRunner with UnixSocketConfiguration instead")
```

- [ ] **Step 4: Update ServiceManager.swift to use helper**

Replace `testConnect()` with `UnixSocketHelper.testConnect(to:)`.

- [ ] **Step 5: Build and verify**

Run: `cd swift/GPUContainer && swift build`
Expected: BUILD SUCCEEDED

- [ ] **Step 6: Commit**

```bash
git add swift/GPUContainer/Sources/
git commit -m "refactor: extract UnixSocketHelper, deduplicate socket code"
```

---

## Chunk 2: ContainerRunner Implementation

### Task 3: Implement ContainerRunner with full Containerization API

**Files:**
- Modify: `swift/GPUContainer/Sources/ContainerRunner.swift`

- [ ] **Step 1: Rewrite ContainerRunner.swift**

```swift
import Containerization
import Foundation

/// Error types for GPU container operations.
enum GPUContainerError: Error, CustomStringConvertible {
    case containerizationNotAvailable
    case kernelNotFound(String)
    case containerExited(code: Int)
    case imagePullFailed(String)

    var description: String {
        switch self {
        case .containerizationNotAvailable:
            return "Containerization framework not available (requires macOS 26+)"
        case .kernelNotFound(let path):
            return "Linux kernel not found at \(path). Run: container system kernel set --recommended"
        case .containerExited(let code):
            return "Container exited with code \(code)"
        case .imagePullFailed(let msg):
            return "Image pull failed: \(msg)"
        }
    }
}

/// Runs containers using Apple's Containerization framework with Unix socket relay.
@available(macOS 26, *)
enum ContainerRunner {

    /// Run a container with direct Unix socket relay to gpu-service (Tier 1).
    /// Falls back to CLI image pull (Tier 2) if programmatic pull fails.
    static func run(
        image: String,
        cpus: Int,
        memory: Int,
        socketPath: String,
        command: [String]
    ) async throws {
        let kernelPath = try findKernel()
        let filteredCommand = command.filter { $0 != "--" }
        let containerId = "gpu-\(UUID().uuidString.prefix(8))"

        // Create container manager
        var manager = try await ContainerManager(
            kernel: Kernel(
                path: URL(fileURLWithPath: kernelPath),
                platform: .linuxArm
            ),
            initfsReference: "ghcr.io/apple/containerization/vminit:0.26.5",
            network: try ContainerManager.VmnetNetwork()
        )

        // Tier 1: Try full programmatic image pull + container creation
        let container: LinuxContainer
        do {
            container = try await manager.create(
                containerId,
                reference: image,
                rootfsSizeInBytes: 8 * 1024 * 1024 * 1024 // 8 GiB
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
                if !filteredCommand.isEmpty {
                    config.process.arguments = filteredCommand
                }
            }
        } catch {
            // Tier 2: Try CLI image pull, then programmatic container creation
            container = try await fallbackWithCLIPull(
                manager: &manager,
                containerId: containerId,
                image: image,
                cpus: cpus,
                memory: memory,
                socketPath: socketPath,
                command: filteredCommand,
                originalError: error
            )
        }

        // Lifecycle: create → start → wait → stop (stop MUST be called)
        defer { try? manager.delete(containerId) }
        try await container.create()
        try await container.start()
        let status = try await container.wait()
        try await container.stop()

        if case .exited(let code) = status, code != 0 {
            throw GPUContainerError.containerExited(code: Int(code))
        }
    }

    /// Tier 2: Pull image via `container pull` CLI, then create container programmatically.
    private static func fallbackWithCLIPull(
        manager: inout ContainerManager,
        containerId: String,
        image: String,
        cpus: Int,
        memory: Int,
        socketPath: String,
        command: [String],
        originalError: Error
    ) async throws -> LinuxContainer {
        // Find container CLI binary
        guard let containerBin = ServiceManager.findContainerBinary() else {
            throw originalError // No CLI available, propagate original error
        }

        // Pull image via CLI
        let pullProcess = Process()
        pullProcess.executableURL = URL(fileURLWithPath: containerBin)
        pullProcess.arguments = ["pull", image]
        pullProcess.standardOutput = FileHandle.standardError // Don't pollute stdout
        try pullProcess.run()
        pullProcess.waitUntilExit()

        guard pullProcess.terminationStatus == 0 else {
            throw GPUContainerError.imagePullFailed(
                "container pull \(image) exited with \(pullProcess.terminationStatus)"
            )
        }

        // Retry container creation (image should now be cached)
        return try await manager.create(
            containerId,
            reference: image,
            rootfsSizeInBytes: 8 * 1024 * 1024 * 1024
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
            if !command.isEmpty {
                config.process.arguments = command
            }
        }
    }

    /// Find the Linux kernel binary for VM boot.
    private static func findKernel() throws -> String {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let kernelDir = appSupport.appendingPathComponent("com.apple.container/kernels")
        let defaultKernel = kernelDir.appendingPathComponent("default.kernel-arm64").path

        if FileManager.default.fileExists(atPath: defaultKernel) {
            return defaultKernel
        }

        // Try any vmlinux file in the kernels directory
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: kernelDir.path),
           let vmlinux = contents.first(where: { $0.hasPrefix("vmlinux") }) {
            return kernelDir.appendingPathComponent(vmlinux).path
        }

        throw GPUContainerError.kernelNotFound(kernelDir.path)
    }
}
```

- [ ] **Step 2: Build and verify**

Run: `cd swift/GPUContainer && swift build`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add swift/GPUContainer/Sources/ContainerRunner.swift
git commit -m "feat: implement ContainerRunner with Containerization API + socket relay"
```

### Task 4: Update Run.swift — async fix, force-TCP flag, error classification

**Files:**
- Modify: `swift/GPUContainer/Sources/Run.swift`

- [ ] **Step 1: Add APPLEGPU_FORCE_TCP env var check**

At the top of the `run()` method in `Run`, before the `#available` check:

```swift
// Allow forcing TCP bridge for testing
let forceTCP = ProcessInfo.processInfo.environment["APPLEGPU_FORCE_TCP"] != nil
```

Then wrap the ContainerRunner dispatch:
```swift
if !forceTCP, #available(macOS 26, *) {
    do {
        try await ContainerRunner.run(...)
        return
    } catch let error as GPUContainerError {
        switch error {
        case .containerExited:
            throw error  // Real error, don't fall back
        default:
            print("ContainerRunner failed: \(error). Falling back to TCP bridge...")
        }
    } catch {
        print("ContainerRunner failed: \(error). Falling back to TCP bridge...")
    }
}
```

- [ ] **Step 2: Fix waitUntilExit() blocking async**

In `runWithContainerCLI`, replace:
```swift
try process.run()
process.waitUntilExit()
```
With:
```swift
try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
    process.terminationHandler = { _ in continuation.resume() }
    do {
        try process.run()
    } catch {
        continuation.resume(throwing: error)
    }
}
```

- [ ] **Step 3: Make TCPBridge IP configurable**

Replace hardcoded `"192.168.64.1"` with:
```swift
let hostIP = ProcessInfo.processInfo.environment["APPLEGPU_BRIDGE_HOST"] ?? "192.168.64.1"
```

Use `hostIP` in both the `inet_addr()` call and the `--env APPLEGPU_HOST=` argument.

- [ ] **Step 4: Build and verify**

Run: `cd swift/GPUContainer && swift build`
Expected: BUILD SUCCEEDED

- [ ] **Step 5: Commit**

```bash
git add swift/GPUContainer/Sources/Run.swift
git commit -m "feat: Run.swift — force-TCP flag, async fix, configurable bridge IP"
```

---

## Chunk 3: Testing + Validation

### Task 5: End-to-end test with vsock relay

- [ ] **Step 1: Build gpu-container release binary**

Run: `cd swift/GPUContainer && swift build -c release`
Expected: BUILD SUCCEEDED

- [ ] **Step 2: Ensure gpu-service is running**

```bash
# Kill any old gpu-service
kill $(pgrep -f gpu-service) 2>/dev/null; sleep 1
# Build and start fresh
cargo build -p applegpu-service --release
nohup target/release/gpu-service > /tmp/gpu-service.log 2>&1 &
sleep 2
cat /tmp/gpu-service.log
```

Expected: "GPU service started on ..."

- [ ] **Step 3: Build Linux wheel**

Run: `uv run maturin build --release --target aarch64-unknown-linux-gnu --zig -i python3.11`

- [ ] **Step 4: Run gpu-container with vsock relay**

```bash
WHEEL_DIR="$(pwd)/target/wheels"
WHEEL_NAME="applegpu_runtime-0.8.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"

swift/GPUContainer/.build/release/gpu-container run \
  --mount "type=bind,source=$WHEEL_DIR,target=/wheels" \
  python:3.11-slim -- bash -c "pip install /wheels/$WHEEL_NAME numpy 2>&1 | tail -1 && python3 -c '
import applegpu_runtime as gpu
gpu.init_backend()
print(\"Backend:\", gpu.device_name())
a = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
print(\"Add:\", (a + a).to_list())
x = gpu.tensor([1, 2, 3], shape=[3], dtype=\"int32\")
print(\"Int32:\", (x + x).to_list())
weights = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
idx = gpu.tensor([1, 0], shape=[2], dtype=\"int64\")
idx32 = gpu.cast(idx, \"int32\")
result = gpu.embedding(weights, idx32)
result.eval()
print(\"Embedding:\", result.to_list())
print(\"\\nVsock relay test PASSED!\")
'"
```

Expected:
```
Backend: gpu-service (remote)
Add: [2.0, 4.0, 6.0]
Int32: [2, 4, 6]
Embedding: [3.0, 4.0, 1.0, 2.0]

Vsock relay test PASSED!
```

- [ ] **Step 5: Verify TCP fallback still works**

```bash
APPLEGPU_FORCE_TCP=1 swift/GPUContainer/.build/release/gpu-container run \
  python:3.11-slim -- echo "TCP fallback works"
```

- [ ] **Step 6: Verify Docker path still works**

```bash
docker run --rm \
  -v ~/.applegpu/runtime.sock:/var/run/applegpu.sock \
  -e APPLEGPU_SOCKET=/var/run/applegpu.sock \
  -v "$(pwd)/target/wheels:/wheels" \
  python:3.11-slim \
  bash -c "pip install /wheels/*manylinux*.whl numpy 2>&1 | tail -1 && python3 -c '
import applegpu_runtime as gpu
gpu.init_backend()
print(\"Docker:\", (gpu.tensor([1.0,2.0,3.0]) + gpu.tensor([4.0,5.0,6.0])).to_list())
'"
```

- [ ] **Step 7: Commit milestone**

```bash
git commit --allow-empty -m "milestone: vsock socket relay working — TCP bridge replaced"
```

### Task 6: Update README container section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update container architecture diagram**

Replace the TCP bridge diagram with:

```
Container (Linux)              Host (macOS)
┌──────────────────┐           ┌──────────────────┐
│ Python code      │           │  gpu-container   │
│  └─ applegpu     │──vsock──▶ │  ├─ Containeriz. │
│     (socket      │  (auto)   │  │   framework   │
│      backend)    │           │  ├─ gpu-service  │
└──────────────────┘           │  ├─ Scheduler    │
                               │  ├─ BufferPool   │
                               │  └─ Metal GPU    │
                               └──────────────────┘
```

Update the numbered steps to reflect vsock relay instead of TCP bridge.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README container section for vsock relay"
```

```bash
git push origin main
```
