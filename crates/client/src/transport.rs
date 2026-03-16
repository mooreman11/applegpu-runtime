use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::os::unix::net::UnixStream;

/// Trait for a bidirectional byte stream transport.
pub trait Transport: Read + Write + Send {
    /// Shut down the transport (both read and write halves).
    fn shutdown(&self) -> io::Result<()>;
}

impl Transport for UnixStream {
    fn shutdown(&self) -> io::Result<()> {
        UnixStream::shutdown(self, std::net::Shutdown::Both)
    }
}

/// Connect via Unix domain socket.
pub fn connect_unix(path: &str) -> io::Result<UnixStream> {
    UnixStream::connect(path)
}

impl Transport for TcpStream {
    fn shutdown(&self) -> io::Result<()> {
        TcpStream::shutdown(self, std::net::Shutdown::Both)
    }
}

/// Connect via TCP.
pub fn connect_tcp(host: &str, port: u16) -> io::Result<TcpStream> {
    TcpStream::connect((host, port))
}

/// Connect via vsock (AF_VSOCK). Only available on Linux.
/// CID 2 = host. Returns a UnixStream wrapping the vsock fd.
#[cfg(target_os = "linux")]
pub fn connect_vsock(cid: u32, port: u32) -> io::Result<UnixStream> {
    use std::os::unix::io::FromRawFd;

    let fd = unsafe { libc::socket(40, libc::SOCK_STREAM, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }

    #[repr(C)]
    struct SockaddrVm {
        svm_family: u16,
        svm_reserved1: u16,
        svm_port: u32,
        svm_cid: u32,
        svm_zero: [u8; 4],
    }

    let addr = SockaddrVm {
        svm_family: 40,
        svm_reserved1: 0,
        svm_port: port,
        svm_cid: cid,
        svm_zero: [0; 4],
    };

    let ret = unsafe {
        libc::connect(
            fd,
            &addr as *const SockaddrVm as *const libc::sockaddr,
            std::mem::size_of::<SockaddrVm>() as u32,
        )
    };
    if ret < 0 {
        unsafe { libc::close(fd) };
        return Err(io::Error::last_os_error());
    }

    let stream = unsafe { UnixStream::from_raw_fd(fd) };
    Ok(stream)
}

#[cfg(not(target_os = "linux"))]
pub fn connect_vsock(_cid: u32, _port: u32) -> io::Result<UnixStream> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "vsock is only available on Linux guests",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixStream as StdUnixStream;

    #[test]
    fn unix_transport_shutdown() {
        let (a, _b) = StdUnixStream::pair().unwrap();
        let transport: Box<dyn Transport> = Box::new(a);
        assert!(transport.shutdown().is_ok());
    }

    #[test]
    fn tcp_transport_implements_trait() {
        use std::net::{TcpListener, TcpStream};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let stream = TcpStream::connect(addr).unwrap();
        let _accepted = listener.accept().unwrap();
        let transport: Box<dyn Transport> = Box::new(stream);
        assert!(transport.shutdown().is_ok());
    }
}
