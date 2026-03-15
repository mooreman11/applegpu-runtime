import ArgumentParser
import Foundation

struct Stop: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Stop a running container"
    )

    @Argument(help: "Container ID to stop")
    var containerId: String

    func run() async throws {
        // TODO(macOS 26): Implement container stop using Containerization framework.
        // This will look up the running container by ID and send a graceful shutdown signal.
        print("TODO: Stop container \(containerId)")
    }
}
