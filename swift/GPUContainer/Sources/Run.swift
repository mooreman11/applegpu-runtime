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

    @Argument(parsing: .captureForPassthrough, help: "Command to run")
    var command: [String] = []

    func run() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let socketPath = "\(home)/.applegpu/runtime.sock"

        // Auto-start gpu-service
        print("Ensuring gpu-service is running...")
        try await ServiceManager.ensureRunning(socketPath: socketPath)

        // Build the `container run` command
        // Apple's `container` CLI handles image pull, VM creation, and process execution.
        // We configure it with our gpu-service socket mount and environment.
        var args: [String] = ["run"]

        // Resource limits
        args += ["--cpus", "\(cpus)"]
        args += ["--memory", "\(memory)M"]

        // Mount the directory containing the gpu-service socket into the container
        let socketDir = (socketPath as NSString).deletingLastPathComponent
        let socketName = (socketPath as NSString).lastPathComponent
        args += ["-v", "\(socketDir):/var/run/applegpu"]
        args += ["--env", "APPLEGPU_SOCKET=/var/run/applegpu/\(socketName)"]

        // Image
        args += [image]

        // Command — passed directly after image (no -- separator)
        if !command.isEmpty {
            // Strip leading "--" if present (from ArgumentParser passthrough)
            let filtered = command.filter { $0 != "--" }
            args += filtered
        }

        print("Starting container from \(image)...")
        print("  CPUs: \(cpus), Memory: \(memory)MB")
        print("  GPU socket: /var/run/applegpu.sock")

        // Find container CLI
        guard let containerBin = findContainerBinary() else {
            throw GPUContainerError.containerCliNotFound
        }

        // Execute
        let process = Process()
        process.executableURL = URL(fileURLWithPath: containerBin)
        process.arguments = args
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        process.standardInput = FileHandle.standardInput

        try process.run()
        process.waitUntilExit()

        let exitCode = process.terminationStatus
        if exitCode != 0 {
            throw ExitCode(exitCode)
        }
    }

    private func findContainerBinary() -> String? {
        let candidates = [
            "/opt/homebrew/bin/container",
            "/usr/local/bin/container",
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }
}
