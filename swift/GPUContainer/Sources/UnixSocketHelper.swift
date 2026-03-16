import Foundation

/// Shared low-level Unix socket utilities used by TCPBridge, VsockRelay, and ServiceManager.
enum UnixSocketHelper {
    /// Connect to a Unix domain socket. Returns the file descriptor, or nil on failure.
    static func connect(to path: String) -> Int32? {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = path.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            close(fd)
            return nil
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                for (i, byte) in pathBytes.enumerated() {
                    dest[i] = byte
                }
            }
        }

        let addrLen = socklen_t(MemoryLayout<sockaddr_un>.size)
        let result = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Foundation.connect(fd, sockPtr, addrLen)
            }
        }

        if result != 0 {
            close(fd)
            return nil
        }
        return fd
    }

    /// Test connectivity to a Unix socket (connect + close). Returns true if connectable.
    static func testConnect(to path: String) -> Bool {
        guard let fd = connect(to: path) else { return false }
        close(fd)
        return true
    }

    /// Bidirectional relay between two file descriptors.
    /// Each direction runs on a background thread; the group tracks completion.
    static func relay(from src: Int32, to dst: Int32, group: DispatchGroup) {
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
