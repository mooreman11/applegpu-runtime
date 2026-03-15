import ArgumentParser
import Foundation

struct Run: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run a container with GPU access"
    )

    @Argument(help: "OCI image name (e.g., python:3.11-slim)")
    var image: String

    @Option(name: .long, help: "Number of CPUs")
    var cpus: Int = 4

    @Option(name: .long, help: "Memory in MB")
    var memory: Int = 4096

    @Option(name: .long, help: "GPU service port for container access")
    var gpuPort: Int = 7654

    @Argument(parsing: .captureForPassthrough, help: "Command to run")
    var command: [String] = []

    func run() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let socketPath = "\(home)/.applegpu/runtime.sock"

        // Auto-start gpu-service
        print("Ensuring gpu-service is running...")
        try await ServiceManager.ensureRunning(socketPath: socketPath)

        // Start TCP bridge: forwards TCP connections on gpuPort to the Unix socket
        print("Starting TCP bridge on port \(gpuPort)...")
        let bridge = try TCPBridge(port: gpuPort, socketPath: socketPath)
        bridge.start()

        // Build container run command via Apple's container CLI
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

    private func findContainerBinary() -> String? {
        ["/opt/homebrew/bin/container", "/usr/local/bin/container"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }
    }
}

/// Bridges TCP connections to a Unix domain socket.
/// Listens on a TCP port and forwards each connection to the gpu-service Unix socket.
class TCPBridge {
    let port: Int
    let socketPath: String
    private var serverFd: Int32 = -1
    private var running = false
    private var thread: Thread?

    init(port: Int, socketPath: String) throws {
        self.port = port
        self.socketPath = socketPath
    }

    func start() {
        serverFd = socket(AF_INET, SOCK_STREAM, 0)
        guard serverFd >= 0 else { return }

        var opt: Int32 = 1
        setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr.s_addr = INADDR_ANY

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                bind(serverFd, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            close(serverFd)
            return
        }

        listen(serverFd, 16)
        running = true

        thread = Thread {
            while self.running {
                var clientAddr = sockaddr_in()
                var clientLen = socklen_t(MemoryLayout<sockaddr_in>.size)
                let clientFd = withUnsafeMutablePointer(to: &clientAddr) { ptr in
                    ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                        accept(self.serverFd, sockPtr, &clientLen)
                    }
                }
                guard clientFd >= 0 else { continue }

                // Connect to Unix socket
                let unixFd = socket(AF_UNIX, SOCK_STREAM, 0)
                guard unixFd >= 0 else { close(clientFd); continue }

                var unixAddr = sockaddr_un()
                unixAddr.sun_family = sa_family_t(AF_UNIX)
                let pathBytes = self.socketPath.utf8CString
                let maxLen = MemoryLayout.size(ofValue: unixAddr.sun_path) - 1
                withUnsafeMutableBytes(of: &unixAddr.sun_path) { dest in
                    for i in 0..<min(pathBytes.count, maxLen) {
                        dest[i] = UInt8(bitPattern: pathBytes[i])
                    }
                }

                let connectResult = withUnsafePointer(to: &unixAddr) { ptr in
                    ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                        Foundation.connect(unixFd, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
                    }
                }
                guard connectResult == 0 else { close(clientFd); close(unixFd); continue }

                // Bidirectional relay in background threads
                self.relay(from: clientFd, to: unixFd)
                self.relay(from: unixFd, to: clientFd)
            }
        }
        thread?.start()
    }

    func stop() {
        running = false
        if serverFd >= 0 { close(serverFd) }
    }

    private func relay(from src: Int32, to dst: Int32) {
        Thread.detachNewThread {
            var buf = [UInt8](repeating: 0, count: 65536)
            while true {
                let n = read(src, &buf, buf.count)
                if n <= 0 { break }
                var written = 0
                while written < n {
                    let w = buf.withUnsafeBufferPointer { ptr in
                        write(dst, ptr.baseAddress! + written, n - written)
                    }
                    if w <= 0 { close(src); close(dst); return }
                    written += w
                }
            }
            close(src)
            close(dst)
        }
    }
}
