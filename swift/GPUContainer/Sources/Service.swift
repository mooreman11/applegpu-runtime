import ArgumentParser
import Foundation

struct Service: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Manage gpu-service",
        subcommands: [ServiceStart.self, ServiceStop.self]
    )
}

struct ServiceStart: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "start",
        abstract: "Start gpu-service"
    )

    func run() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        try await ServiceManager.ensureRunning(
            socketPath: "\(home)/.applegpu/runtime.sock")
        print("gpu-service started")
    }
}

struct ServiceStop: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "stop",
        abstract: "Stop gpu-service"
    )

    func run() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let pidPath = "\(home)/.applegpu/gpu-service.pid"
        guard
            let pidStr = try? String(contentsOfFile: pidPath, encoding: .utf8),
            let pid = Int32(pidStr.trimmingCharacters(in: .whitespacesAndNewlines))
        else {
            print("gpu-service is not running")
            return
        }
        kill(pid, SIGTERM)
        print("Sent SIGTERM to gpu-service (PID \(pid))")
    }
}
