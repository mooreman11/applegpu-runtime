use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;

/// Trait for a bidirectional byte stream transport.
pub trait Transport: Read + Write + Send {}

impl Transport for UnixStream {}

/// Connect via Unix domain socket.
pub fn connect_unix(path: &str) -> io::Result<UnixStream> {
    UnixStream::connect(path)
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
