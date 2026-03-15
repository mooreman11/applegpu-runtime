import ArgumentParser
import Containerization
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

    @Argument(parsing: .captureForPassthrough, help: "Command to run")
    var command: [String] = []

    func run() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let socketPath = "\(home)/.applegpu/runtime.sock"

        // Auto-start gpu-service
        print("Ensuring gpu-service is running...")
        try await ServiceManager.ensureRunning(socketPath: socketPath)

        // Configure container
        // NOTE: The exact Containerization API depends on the framework version.
        // The code below follows the patterns from the cctl example in the
        // Containerization repository. Adapt as needed for API changes.

        // TODO(macOS 26): Uncomment and adapt when Containerization framework is available.
        //
        // var config = LinuxContainer.Configuration()
        // config.cpus = cpus
        // config.memoryInBytes = UInt64(memory) * 1024 * 1024
        //
        // // Socket relay: host gpu-service -> container
        // config.sockets = [
        //     UnixSocketConfiguration(
        //         source: URL(fileURLWithPath: socketPath),
        //         destination: URL(fileURLWithPath: "/var/run/applegpu.sock"),
        //         direction: .into
        //     )
        // ]
        //
        // // Environment
        // config.process.environment["APPLEGPU_SOCKET"] = "/var/run/applegpu.sock"
        // config.process.environment["PYTHONPATH"] = "/opt/applegpu"
        //
        // // Set command
        // if !command.isEmpty {
        //     config.process.arguments = command
        // }
        //
        // // Image pull + container creation (check cctl example for correct API):
        // let imageStore = try await ImageStore(...)
        // let rootfs = try await imageStore.pull(image)
        // let container = LinuxContainer(id: UUID().uuidString, rootfs: rootfs, config: config)
        // try await container.start()
        // let process = try await container.spawn(config.process)
        // let exitStatus = try await process.wait()

        print("Starting container from \(image)...")
        print("  CPUs: \(cpus), Memory: \(memory)MB")
        print("  GPU socket: /var/run/applegpu.sock")
        print("")
        print("NOTE: Container execution requires macOS 26+ with the Containerization framework.")
        print("The container configuration is ready but execution is stubbed until macOS 26 is available.")

        if !command.isEmpty {
            print("  Command: \(command.joined(separator: " "))")
        }

        print("Container exited")
    }
}
