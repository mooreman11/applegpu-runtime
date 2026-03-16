import Containerization
import Foundation

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
        let normalizedImage = Self.normalizeImageReference(image)

        // Create container manager
        var manager = try await ContainerManager(
            kernel: Kernel(
                path: URL(fileURLWithPath: kernelPath),
                platform: .linuxArm
            ),
            initfsReference: "ghcr.io/apple/containerization/vminit:0.26.5",
            network: (try? ContainerManager.VmnetNetwork())  // nil if vmnet unavailable (no entitlement)
        )

        // Socket relay: mount the directory containing the host socket into the container.
        // We relay to /run/applegpu/ directory — the socket will appear as /run/applegpu/runtime.sock
        let socketDir = (socketPath as NSString).deletingLastPathComponent
        let socketFilename = (socketPath as NSString).lastPathComponent
        // Use /tmp/ — part of the rootfs (not a separate mount), exists in all images
        let containerSocketPath = "/tmp/\(socketFilename)"

        let socketConfig = UnixSocketConfiguration(
            source: URL(fileURLWithPath: socketPath),
            destination: URL(fileURLWithPath: containerSocketPath),
            direction: .into
        )

        // Capture values for the configuration closure
        let cpuCount = cpus
        let memoryMB = memory
        let cmd = filteredCommand

        // Tier 1: Try full programmatic image pull + container creation
        let container: LinuxContainer
        do {
            container = try await manager.create(
                containerId,
                reference: normalizedImage,
                rootfsSizeInBytes: 8 * 1024 * 1024 * 1024 // 8 GiB
            ) { config in
                config.cpus = cpuCount
                config.memoryInBytes = UInt64(memoryMB) * 1024 * 1024
                // Socket relay will be set up after container starts via vminitd
                config.sockets = [socketConfig]
                config.process.environmentVariables.append("APPLEGPU_SOCKET=\(containerSocketPath)")
                config.useInit = true  // Enable init wrapper for proper socket staging
                if !cmd.isEmpty {
                    config.process.arguments = cmd
                }
            }
        } catch {
            print("[ContainerRunner] Tier 1 failed: \(error)")
            // Tier 2: Try CLI image pull, then programmatic container creation
            container = try await fallbackWithCLIPull(
                manager: &manager,
                containerId: containerId,
                image: image,
                cpus: cpuCount,
                memory: memoryMB,
                socketConfig: socketConfig,
                command: cmd,
                originalError: error
            )
        }

        // Lifecycle: create → start → wait → stop (stop MUST be called)
        defer { try? manager.delete(containerId) }
        try await container.create()
        try await container.start()
        let status = try await container.wait()
        try await container.stop()

        if status.exitCode != 0 {
            throw GPUContainerError.containerExited(code: Int(status.exitCode))
        }
    }

    /// Tier 2: Pull image via `container pull` CLI, then create container programmatically.
    private static func fallbackWithCLIPull(
        manager: inout ContainerManager,
        containerId: String,
        image: String,
        cpus: Int,
        memory: Int,
        socketConfig: UnixSocketConfiguration,
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
        pullProcess.arguments = ["image", "pull", image]
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
        ) { config in
            config.cpus = cpus
            config.memoryInBytes = UInt64(memory) * 1024 * 1024
            config.sockets = [socketConfig]
            config.process.environmentVariables.append("APPLEGPU_SOCKET=/tmp/runtime.sock")
            if !command.isEmpty {
                config.process.arguments = command
            }
        }
    }

    /// Normalize a short image reference to a fully qualified one.
    /// "alpine" → "docker.io/library/alpine:latest"
    /// "python:3.11-slim" → "docker.io/library/python:3.11-slim"
    /// "ghcr.io/foo/bar:v1" → unchanged
    private static func normalizeImageReference(_ ref: String) -> String {
        var result = ref
        // Add default tag if missing
        if !result.contains(":") || (result.contains("/") && !result.split(separator: "/").last!.contains(":")) {
            if !result.contains(":") { result += ":latest" }
        }
        // Add docker.io/library/ prefix for short names
        if !result.contains("/") {
            result = "docker.io/library/" + result
        } else if !result.contains(".") {
            // user/repo but no domain → docker.io/
            result = "docker.io/" + result
        }
        return result
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
