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
                if testConnect(socketPath) {
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
            if testConnect(socketPath) {
                print("  gpu-service is ready")
                return
            }
        }
        throw GPUContainerError.serviceStartTimeout
    }

    static func testConnect(_ socketPath: String) -> Bool {
        guard FileManager.default.fileExists(atPath: socketPath) else { return false }
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        let maxLen = MemoryLayout.size(ofValue: addr.sun_path) - 1
        withUnsafeMutableBytes(of: &addr.sun_path) { dest in
            for i in 0..<min(pathBytes.count, maxLen) {
                dest[i] = UInt8(bitPattern: pathBytes[i])
            }
        }

        return withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Foundation.connect(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size)) == 0
            }
        }
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

    var description: String {
        switch self {
        case .serviceNotFound:
            return
                "gpu-service binary not found. Install it or build with: cargo build -p applegpu-service --release"
        case .serviceStartTimeout:
            return "gpu-service failed to start within 5 seconds"
        case .containerCliNotFound:
            return "Apple container CLI not found. Install with: brew install container"
        }
    }
}
