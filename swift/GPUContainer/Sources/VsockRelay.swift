import Foundation
import Virtualization

/// Relays vsock connections from a container VM to the gpu-service Unix socket.
/// Each incoming vsock connection spawns a bidirectional byte relay to the
/// gpu-service at the given socketPath.
@available(*, deprecated, message: "Use ContainerRunner with UnixSocketConfiguration instead")
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
        guard let unixFd = UnixSocketHelper.connect(to: socketPath) else {
            return false
        }

        let vsockFd = connection.fileDescriptor

        // Bidirectional relay with proper shutdown
        let group = DispatchGroup()

        UnixSocketHelper.relay(from: vsockFd, to: unixFd, group: group)
        UnixSocketHelper.relay(from: unixFd, to: vsockFd, group: group)

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
}
