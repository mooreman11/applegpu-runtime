use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

use crate::error::{GpuError, Result};
use crate::serial::{EvalRequest, EvalResponse};

/// Default socket path under user's home directory (avoids /tmp security concerns).
pub fn default_socket_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    format!("{}/.applegpu/runtime.sock", home)
}

/// Send an eval request to the GPU service and receive the result.
#[deprecated(since = "0.8.0", note = "Use applegpu-client crate instead")]
pub fn eval_remote(socket_path: &str, request: &EvalRequest) -> Result<EvalResponse> {
    let mut stream = UnixStream::connect(socket_path)
        .map_err(|e| GpuError::ComputeFailed(format!(
            "Failed to connect to GPU service at {}: {}",
            socket_path, e
        )))?;

    // Send request: [4 bytes length] + [payload]
    let payload = request.serialize();
    let len_bytes = (payload.len() as u32).to_le_bytes();
    stream.write_all(&len_bytes)
        .map_err(|e| GpuError::ComputeFailed(format!("IPC write failed: {}", e)))?;
    stream.write_all(&payload)
        .map_err(|e| GpuError::ComputeFailed(format!("IPC write failed: {}", e)))?;
    stream.flush()
        .map_err(|e| GpuError::ComputeFailed(format!("IPC flush failed: {}", e)))?;

    // Read response: [4 bytes length] + [payload]
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)
        .map_err(|e| GpuError::ComputeFailed(format!("IPC read failed: {}", e)))?;
    let resp_len = u32::from_le_bytes(len_buf) as usize;
    let mut resp_buf = vec![0u8; resp_len];
    stream.read_exact(&mut resp_buf)
        .map_err(|e| GpuError::ComputeFailed(format!("IPC read failed: {}", e)))?;

    EvalResponse::deserialize(&resp_buf)
        .map_err(|e| GpuError::ComputeFailed(format!("IPC deserialization failed: {}", e)))
}

// Note: No service_available() check — Path::exists is unreliable for sockets
// (stale files persist after crashes). The client just attempts to connect
// and gets a clear error if the service isn't running.
