pub mod transport;

use std::io;
use std::net::Shutdown;
use std::os::unix::net::UnixStream;

use applegpu_wire::{
    self as wire,
    EvalRequest, EvalResponse, HandshakeRequest, HandshakeResponse,
    HANDSHAKE_OK, PROTOCOL_VERSION, MAX_MESSAGE_SIZE,
};

#[derive(Debug)]
pub enum ClientError {
    Io(io::Error),
    HandshakeRejected { status: u32 },
    Protocol(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::Io(e) => write!(f, "I/O error: {}", e),
            ClientError::HandshakeRejected { status } => write!(f, "Handshake rejected (status {})", status),
            ClientError::Protocol(msg) => write!(f, "Protocol error: {}", msg),
        }
    }
}

impl std::error::Error for ClientError {}

impl From<io::Error> for ClientError {
    fn from(e: io::Error) -> Self { ClientError::Io(e) }
}

pub type Result<T> = std::result::Result<T, ClientError>;

pub const DEFAULT_VSOCK_PORT: u32 = 5678;

pub struct GpuClient {
    stream: UnixStream,
    pub container_id: u64,
    pub granted_memory: u64,
}

impl GpuClient {
    pub fn connect_unix(path: &str, requested_memory: u64) -> Result<Self> {
        let stream = transport::connect_unix(path)?;
        Self::handshake(stream, requested_memory)
    }

    pub fn connect_vsock(port: u32, requested_memory: u64) -> Result<Self> {
        let stream = transport::connect_vsock(2, port)?;
        Self::handshake(stream, requested_memory)
    }

    fn handshake(mut stream: UnixStream, requested_memory: u64) -> Result<Self> {
        let hs = HandshakeRequest {
            protocol_version: PROTOCOL_VERSION,
            requested_memory,
        };
        wire::write_message(&mut stream, &hs.serialize())?;

        let msg = wire::read_message(&mut stream, 1024)?;
        let resp = HandshakeResponse::deserialize(&msg)?;

        if resp.status != HANDSHAKE_OK {
            return Err(ClientError::HandshakeRejected { status: resp.status });
        }

        Ok(GpuClient {
            stream,
            container_id: resp.container_id,
            granted_memory: resp.granted_memory,
        })
    }

    pub fn eval(&mut self, request: &EvalRequest) -> Result<EvalResponse> {
        wire::write_message(&mut self.stream, &request.serialize())?;
        let msg = wire::read_message(&mut self.stream, MAX_MESSAGE_SIZE)?;
        EvalResponse::deserialize(&msg).map_err(|e| ClientError::Protocol(e.to_string()))
    }
}

impl Drop for GpuClient {
    fn drop(&mut self) {
        let _ = self.stream.shutdown(Shutdown::Both);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn mock_gpu_service(mut stream: UnixStream) {
        let msg = wire::read_message(&mut stream, 1024).unwrap();
        let req = HandshakeRequest::deserialize(&msg).unwrap();
        let resp = HandshakeResponse {
            status: HANDSHAKE_OK,
            container_id: 7,
            granted_memory: req.requested_memory,
        };
        wire::write_message(&mut stream, &resp.serialize()).unwrap();

        let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
        let eval_req = EvalRequest::deserialize(&msg).unwrap();
        let result: Vec<u8> = vec![42.0f32].iter().flat_map(|f| f.to_le_bytes()).collect();
        let eval_resp = EvalResponse::Ok {
            tensor_id: eval_req.target_id,
            shape: vec![1],
            data: result,
        };
        wire::write_message(&mut stream, &eval_resp.serialize()).unwrap();
    }

    #[test]
    fn gpu_client_connect_and_eval() {
        let (client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || mock_gpu_service(server_stream));

        let mut client = GpuClient::handshake(client_stream, 1024 * 1024).unwrap();
        assert_eq!(client.container_id, 7);

        let req = EvalRequest {
            target_id: 1,
            tensors: vec![],
            nodes: vec![],
        };
        let resp = client.eval(&req).unwrap();
        match resp {
            EvalResponse::Ok { tensor_id, .. } => assert_eq!(tensor_id, 1),
            _ => panic!("Expected Ok"),
        }
        server.join().unwrap();
    }

    #[test]
    fn gpu_client_handshake_rejected() {
        let (client_stream, mut server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let msg = wire::read_message(&mut server_stream, 1024).unwrap();
            let _ = HandshakeRequest::deserialize(&msg).unwrap();
            let resp = HandshakeResponse {
                status: 1,
                container_id: 0,
                granted_memory: 0,
            };
            wire::write_message(&mut server_stream, &resp.serialize()).unwrap();
        });

        let result = GpuClient::handshake(client_stream, 1024);
        assert!(result.is_err());
        server.join().unwrap();
    }
}
