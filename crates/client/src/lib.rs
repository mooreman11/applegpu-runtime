pub mod transport;

use std::io;
use std::net::Shutdown;
use std::os::unix::net::UnixStream;

use applegpu_wire::{
    self as wire,
    EvalRequest, EvalResponse, HandshakeRequest, HandshakeResponse,
    ReadTensorRequest, ReadTensorResponse,
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

    /// Read a previously-computed tensor by ID.
    /// Returns `(shape, dtype, data)` on success.
    pub fn read_tensor(&mut self, tensor_id: u64) -> Result<(Vec<usize>, u32, Vec<u8>)> {
        let req = ReadTensorRequest { tensor_id };
        wire::write_message(&mut self.stream, &req.serialize())?;
        let msg = wire::read_message(&mut self.stream, MAX_MESSAGE_SIZE)?;
        let resp = ReadTensorResponse::deserialize(&msg)
            .map_err(|e| ClientError::Protocol(e.to_string()))?;
        match resp {
            ReadTensorResponse::Ok { shape, dtype, data, .. } => Ok((shape, dtype, data)),
            ReadTensorResponse::NotFound { tensor_id } => Err(ClientError::Protocol(
                format!("Tensor {} not found", tensor_id),
            )),
        }
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

    fn mock_gpu_service_with_read(mut stream: UnixStream) {
        // Handshake
        let msg = wire::read_message(&mut stream, 1024).unwrap();
        let req = HandshakeRequest::deserialize(&msg).unwrap();
        let resp = HandshakeResponse {
            status: HANDSHAKE_OK,
            container_id: 8,
            granted_memory: req.requested_memory,
        };
        wire::write_message(&mut stream, &resp.serialize()).unwrap();

        // Eval request
        let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
        let _eval_req = EvalRequest::deserialize(&msg).unwrap();
        let result: Vec<u8> = vec![10.0f32, 20.0f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let eval_resp = EvalResponse::Ok {
            tensor_id: 1,
            shape: vec![2],
            data: result.clone(),
        };
        wire::write_message(&mut stream, &eval_resp.serialize()).unwrap();

        // ReadTensor request
        let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
        let read_req = ReadTensorRequest::deserialize(&msg).unwrap();
        let read_resp = ReadTensorResponse::Ok {
            tensor_id: read_req.tensor_id,
            shape: vec![2],
            dtype: 0, // f32
            data: result,
        };
        wire::write_message(&mut stream, &read_resp.serialize()).unwrap();
    }

    #[test]
    fn gpu_client_read_tensor() {
        let (client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || mock_gpu_service_with_read(server_stream));

        let mut client = GpuClient::handshake(client_stream, 1024 * 1024).unwrap();
        assert_eq!(client.container_id, 8);

        // First do an eval
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

        // Now read the tensor back
        let (shape, dtype, data) = client.read_tensor(1).unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(dtype, 0);
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(floats, vec![10.0, 20.0]);

        server.join().unwrap();
    }

    fn mock_gpu_service_read_not_found(mut stream: UnixStream) {
        // Handshake
        let msg = wire::read_message(&mut stream, 1024).unwrap();
        let req = HandshakeRequest::deserialize(&msg).unwrap();
        let resp = HandshakeResponse {
            status: HANDSHAKE_OK,
            container_id: 9,
            granted_memory: req.requested_memory,
        };
        wire::write_message(&mut stream, &resp.serialize()).unwrap();

        // ReadTensor request — respond with NotFound
        let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
        let read_req = ReadTensorRequest::deserialize(&msg).unwrap();
        let read_resp = ReadTensorResponse::NotFound {
            tensor_id: read_req.tensor_id,
        };
        wire::write_message(&mut stream, &read_resp.serialize()).unwrap();
    }

    #[test]
    fn gpu_client_read_tensor_not_found() {
        let (client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || mock_gpu_service_read_not_found(server_stream));

        let mut client = GpuClient::handshake(client_stream, 1024 * 1024).unwrap();
        let result = client.read_tensor(999);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("999"), "Error should mention tensor ID: {}", err_msg);

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
