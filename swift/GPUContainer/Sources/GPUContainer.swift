import ArgumentParser
import Foundation

@main
struct GPUContainer: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gpu-container",
        abstract: "Run Linux containers with Metal GPU access on Apple Silicon",
        version: "0.9.0",
        subcommands: [Run.self, Stop.self, List.self, Status.self, Service.self]
    )
}
