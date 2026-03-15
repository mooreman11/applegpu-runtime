import ArgumentParser
import Foundation

struct List: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "List running GPU containers"
    )

    func run() async throws {
        // TODO(macOS 26): Implement container listing using Containerization framework.
        // This will enumerate all running LinuxContainer instances managed by gpu-container.
        print("TODO: List containers")
    }
}
