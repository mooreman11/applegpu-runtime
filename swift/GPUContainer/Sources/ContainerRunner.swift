import Foundation

/// Runs containers using Apple's Containerization framework with Unix socket relay.
/// Primary transport: UnixSocketConfiguration mounts gpu-service socket into container.
///
/// NOTE: This is a skeleton. The exact Containerization framework API depends on
/// the macOS 26 SDK. Adjust types and method names when the SDK is available.
/// Import Containerization and uncomment the framework-specific code.
@available(macOS 26, *)
enum ContainerRunner {
    /// Run a container with direct Unix socket relay to gpu-service.
    static func run(
        image: String,
        cpus: Int,
        memory: Int,
        socketPath: String,
        command: [String]
    ) async throws {
        // TODO: Implement when macOS 26 SDK is available
        // var config = LinuxContainer.Configuration()
        // config.cpus = cpus
        // config.memoryInBytes = UInt64(memory) * 1024 * 1024
        // config.sockets = [
        //     UnixSocketConfiguration(
        //         source: URL(fileURLWithPath: socketPath),
        //         destination: URL(fileURLWithPath: "/var/run/applegpu.sock"),
        //         direction: .into
        //     )
        // ]
        // config.process.environment["APPLEGPU_SOCKET"] = "/var/run/applegpu.sock"
        //
        // let filteredCommand = command.filter { $0 != "--" }
        // if !filteredCommand.isEmpty {
        //     config.process.arguments = filteredCommand
        // }
        //
        // let container = try await LinuxContainer(image: image, configuration: config)
        // try await container.start()
        // try await container.waitUntilExit()

        throw GPUContainerError.containerizationNotAvailable
    }
}
