use std::io::{Read, Write};
use std::os::unix::net::UnixListener;

use applegpu_core::buffer::Buffer;
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::serial::{EvalRequest, EvalResponse};
use applegpu_core::tensor::Tensor;

fn handle_request(device: &Device, request: &EvalRequest) -> EvalResponse {
    let mut rt = LazyRuntime::new();

    // Load input tensors from serialized data
    for td in &request.tensors {
        match Buffer::from_bytes(device, &td.data) {
            Ok(buffer) => {
                let tensor = Tensor::from_raw(td.id, td.shape.clone(), buffer);
                rt.insert_tensor(tensor);
            }
            Err(e) => return EvalResponse::Err(format!("Buffer allocation failed: {}", e)),
        }
    }

    // Load graph nodes
    for node in &request.nodes {
        let _ = rt.record_op(node.clone());
    }

    // Evaluate (fusion runs automatically inside eval)
    if let Err(e) = rt.eval(device, request.target_id) {
        return EvalResponse::Err(format!("Eval failed: {}", e));
    }

    // Read result
    match rt.read_f32(request.target_id) {
        Ok(data) => {
            let shape = rt.shape(request.target_id).unwrap_or_default();
            let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            EvalResponse::Ok {
                tensor_id: request.target_id,
                shape,
                data: data_bytes,
            }
        }
        Err(e) => EvalResponse::Err(format!("Read failed: {}", e)),
    }
}

fn main() {
    let socket_path = std::env::var("APPLEGPU_SOCKET")
        .unwrap_or_else(|_| applegpu_core::ipc::default_socket_path());

    // Ensure socket directory exists and remove stale socket
    if let Some(parent) = std::path::Path::new(&socket_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_file(&socket_path);

    let device = Device::new().expect("No Metal GPU available");
    println!("GPU service started on {} (device: {})", socket_path, device.name());

    let listener = UnixListener::bind(&socket_path)
        .expect("Failed to bind Unix socket");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                // Read request: [4 bytes length] + [payload]
                let mut len_buf = [0u8; 4];
                if stream.read_exact(&mut len_buf).is_err() {
                    continue;
                }
                let req_len = u32::from_le_bytes(len_buf) as usize;

                // Safety: limit request size to 256MB
                if req_len > 256 * 1024 * 1024 {
                    let resp = EvalResponse::Err("Request too large".to_string());
                    let resp_payload = resp.serialize();
                    let len_bytes = (resp_payload.len() as u32).to_le_bytes();
                    let _ = stream.write_all(&len_bytes);
                    let _ = stream.write_all(&resp_payload);
                    continue;
                }

                let mut req_buf = vec![0u8; req_len];
                if stream.read_exact(&mut req_buf).is_err() {
                    continue;
                }

                let response = match EvalRequest::deserialize(&req_buf) {
                    Ok(request) => handle_request(&device, &request),
                    Err(e) => EvalResponse::Err(format!("Deserialization failed: {}", e)),
                };

                // Send response: [4 bytes length] + [payload]
                let resp_payload = response.serialize();
                let len_bytes = (resp_payload.len() as u32).to_le_bytes();
                let _ = stream.write_all(&len_bytes);
                let _ = stream.write_all(&resp_payload);
                let _ = stream.flush();
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }
}
