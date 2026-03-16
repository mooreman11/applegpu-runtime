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

        // Allow forcing TCP bridge for testing
        let forceTCP = ProcessInfo.processInfo.environment["APPLEGPU_FORCE_TCP"] != nil

        // Try Containerization framework first (macOS 26+, direct socket mount)
        if !forceTCP, #available(macOS 26, *) {
            do {
                try await ContainerRunner.run(
                    image: image,
                    cpus: cpus,
                    memory: memory,
                    socketPath: socketPath,
                    command: command
                )
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

        // Fallback: container CLI + TCP bridge
        try await runWithContainerCLI(socketPath: socketPath)
    }

    /// Legacy fallback: uses Apple's `container` CLI with TCP bridge.
    private func runWithContainerCLI(socketPath: String) async throws {
        let hostIP = ProcessInfo.processInfo.environment["APPLEGPU_BRIDGE_HOST"] ?? "192.168.64.1"

        print("Starting TCP bridge on port \(gpuPort)...")
        let bridge = try TCPBridge(port: gpuPort, socketPath: socketPath, hostIP: hostIP)
        bridge.start()

        var args: [String] = ["run", "--rm"]
        args += ["--cpus", "\(cpus)"]
        args += ["--memory", "\(memory)M"]
        args += ["--env", "APPLEGPU_HOST=\(hostIP)"]
        args += ["--env", "APPLEGPU_PORT=\(gpuPort)"]
        args += [image]

        let filteredCommand = command.filter { $0 != "--" }
        if !filteredCommand.isEmpty {
            args += filteredCommand
        }

        print("Starting container from \(image)...")
        print("  CPUs: \(cpus), Memory: \(memory)MB")
        print("  GPU bridge: \(hostIP):\(gpuPort) → \(socketPath)")

        guard let containerBin = ServiceManager.findContainerBinary() else {
            throw GPUContainerError.containerCliNotFound
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: containerBin)
        process.arguments = args
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        process.standardInput = FileHandle.standardInput

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            process.terminationHandler = { _ in continuation.resume() }
            do {
                try process.run()
            } catch {
                continuation.resume(throwing: error)
            }
        }

        bridge.stop()

        let exitCode = process.terminationStatus
        if exitCode != 0 {
            throw ExitCode(exitCode)
        }
    }
}

/// Bridges TCP connections to a Unix domain socket.
/// Listens on a TCP port and forwards each connection to the gpu-service Unix socket.
class TCPBridge {
    let port: Int
    let socketPath: String
    let hostIP: String
    private var serverFd: Int32 = -1
    private var running = false
    private var thread: Thread?

    init(port: Int, socketPath: String, hostIP: String = "192.168.64.1") throws {
        self.port = port
        self.socketPath = socketPath
        self.hostIP = hostIP
    }

    func start() {
        serverFd = socket(AF_INET, SOCK_STREAM, 0)
        guard serverFd >= 0 else { return }

        var opt: Int32 = 1
        setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr.s_addr = inet_addr(hostIP)

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
                guard let unixFd = UnixSocketHelper.connect(to: self.socketPath) else {
                    close(clientFd)
                    continue
                }

                // Bidirectional relay with proper shutdown
                let group = DispatchGroup()
                UnixSocketHelper.relay(from: clientFd, to: unixFd, group: group)
                UnixSocketHelper.relay(from: unixFd, to: clientFd, group: group)

                // Close both fds after both relay threads complete
                Thread.detachNewThread {
                    group.wait()
                    close(clientFd)
                    close(unixFd)
                }
            }
        }
        thread?.start()
    }

    func stop() {
        running = false
        if serverFd >= 0 { close(serverFd) }
    }

}
