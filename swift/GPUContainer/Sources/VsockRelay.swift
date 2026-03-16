import Foundation
import Virtualization

/// Relays vsock connections from a container VM to the gpu-service Unix socket.
/// Each incoming vsock connection spawns a bidirectional byte relay to the
/// gpu-service at the given socketPath.
class VsockRelay: NSObject, VZVirtioSocketListenerDelegate {
    let socketPath: String
    private let listener: VZVirtioSocketListener

    init(socketPath: String) {
        self.socketPath = socketPath
        self.listener = VZVirtioSocketListener()
        super.init()
        self.listener.delegate = self
    }

    /// Register this relay on a vsock device for the given port.
    func attach(to device: VZVirtioSocketDevice, port: UInt32) {
        device.setSocketListener(listener, forPort: port)
    }

    /// Detach from a vsock device.
    func detach(from device: VZVirtioSocketDevice, port: UInt32) {
        device.removeSocketListener(forPort: port)
    }

    // MARK: - VZVirtioSocketListenerDelegate

    func listener(
        _ listener: VZVirtioSocketListener,
        shouldAcceptNewConnection connection: VZVirtioSocketConnection,
        from socketDevice: VZVirtioSocketDevice
    ) -> Bool {
        // Connect to gpu-service Unix socket
        let unixFd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard unixFd >= 0 else { return false }

        var unixAddr = sockaddr_un()
        unixAddr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        let maxLen = MemoryLayout.size(ofValue: unixAddr.sun_path) - 1
        withUnsafeMutableBytes(of: &unixAddr.sun_path) { dest in
            for i in 0..<min(pathBytes.count, maxLen) {
                dest[i] = UInt8(bitPattern: pathBytes[i])
            }
        }

        let connectResult = withUnsafePointer(to: &unixAddr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Foundation.connect(unixFd, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connectResult == 0 else {
            close(unixFd)
            return false
        }

        let vsockFd = connection.fileDescriptor

        // Bidirectional relay with proper shutdown
        let group = DispatchGroup()

        relay(from: vsockFd, to: unixFd, group: group)
        relay(from: unixFd, to: vsockFd, group: group)

        // Close both fds after both relay threads complete
        Thread.detachNewThread {
            group.wait()
            close(unixFd)
            // Note: vsock fd is owned by VZVirtioSocketConnection,
            // closed when the connection object is deallocated.
            // We do NOT close vsockFd here.
        }

        return true
    }

    // MARK: - Relay

    private func relay(from src: Int32, to dst: Int32, group: DispatchGroup) {
        group.enter()
        Thread.detachNewThread {
            defer { group.leave() }
            var buf = [UInt8](repeating: 0, count: 65536)
            while true {
                let n = read(src, &buf, buf.count)
                if n <= 0 {
                    Darwin.shutdown(dst, SHUT_WR)
                    break
                }
                var written = 0
                while written < n {
                    let w = buf.withUnsafeBufferPointer { ptr in
                        write(dst, ptr.baseAddress! + written, n - written)
                    }
                    if w <= 0 {
                        Darwin.shutdown(src, SHUT_RD)
                        return
                    }
                    written += w
                }
            }
        }
    }
}
