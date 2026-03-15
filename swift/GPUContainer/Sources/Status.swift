import ArgumentParser
import Foundation

struct Status: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Check gpu-service health"
    )

    func run() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let socketPath = "\(home)/.applegpu/runtime.sock"
        let pidPath = "\(home)/.applegpu/gpu-service.pid"

        if ServiceManager.testConnect(socketPath) {
            let pid =
                (try? String(contentsOfFile: pidPath, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines)) ?? "unknown"
            print("gpu-service: running (PID \(pid))")
            print("  socket: \(socketPath)")
        } else {
            print("gpu-service: not running")
        }
    }
}
