import Foundation

enum ServiceManager {
    static func ensureRunning(socketPath: String) async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let pidPath = "\(home)/.applegpu/gpu-service.pid"

        // Check if already running
        if FileManager.default.fileExists(atPath: pidPath),
           let pidStr = try? String(contentsOfFile: pidPath, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           let pid = Int32(pidStr)
        {
            // Check if process is alive
            if kill(pid, 0) == 0 {
                // Verify socket is connectable
                if UnixSocketHelper.testConnect(to: socketPath) {
                    return  // Already running and healthy
                }
            }
        }

        // Find gpu-service binary
        guard let servicePath = findServiceBinary() else {
            throw GPUContainerError.serviceNotFound
        }

        // Ensure directory exists
        let dir = (socketPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)

        // Start gpu-service
        let process = Process()
        process.executableURL = URL(fileURLWithPath: servicePath)
        process.arguments = []
        try process.run()

        // Readiness probe: wait up to 5 seconds
        for _ in 0..<50 {
            try await Task.sleep(nanoseconds: 100_000_000)  // 100ms
            if UnixSocketHelper.testConnect(to: socketPath) {
                print("  gpu-service is ready")
                return
            }
        }
        throw GPUContainerError.serviceStartTimeout
    }

    static func testConnect(_ socketPath: String) -> Bool {
        guard FileManager.default.fileExists(atPath: socketPath) else { return false }
        return UnixSocketHelper.testConnect(to: socketPath)
    }

    static func findContainerBinary() -> String? {
        ["/opt/homebrew/bin/container", "/usr/local/bin/container"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    static func findServiceBinary() -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let cwd = FileManager.default.currentDirectoryPath
        let candidates = [
            "/usr/local/bin/gpu-service",
            "\(home)/.applegpu/bin/gpu-service",
            // Development: cargo target
            "\(cwd)/target/release/gpu-service",
            "\(cwd)/target/debug/gpu-service",
            "\(cwd)/target/release/applegpu-service",
            "\(cwd)/target/debug/applegpu-service",
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }
}

enum GPUContainerError: Error, CustomStringConvertible {
    case serviceNotFound
    case serviceStartTimeout
    case containerCliNotFound
    case kernelNotFound(String)
    case containerizationNotAvailable
    case containerExited(code: Int)
    case imagePullFailed(String)

    var description: String {
        switch self {
        case .serviceNotFound:
            return
                "gpu-service binary not found. Install it or build with: cargo build -p applegpu-service --release"
        case .serviceStartTimeout:
            return "gpu-service failed to start within 5 seconds"
        case .containerCliNotFound:
            return "Apple container CLI not found. Install with: brew install container"
        case .kernelNotFound(let path):
            return
                "Linux kernel not found at \(path). Run: container system kernel set --recommended"
        case .containerizationNotAvailable:
            return "Containerization framework not available (requires macOS 26+)"
        case .containerExited(let code):
            return "Container exited with code \(code)"
        case .imagePullFailed(let msg):
            return "Image pull failed: \(msg)"
        }
    }
}
