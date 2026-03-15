use std::io;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use applegpu_core::buffer::Buffer;
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::scheduler::{ContainerId, ContainerConfig, Priority};
use applegpu_core::serial::wire_node_to_core;
use applegpu_core::tensor::{DType, Tensor};

use applegpu_wire::{
    self as wire,
    EvalRequest, EvalResponse, HandshakeRequest, HandshakeResponse,
    ReadTensorRequest, ReadTensorResponse,
    HANDSHAKE_OK, HANDSHAKE_REJECTED_QUOTA, PROTOCOL_VERSION, MAX_MESSAGE_SIZE,
    MAGIC_REQUEST, MAGIC_READ_REQ,
};

/// Convert a wire protocol dtype discriminant to a core DType.
fn wire_dtype_to_core(d: u32) -> io::Result<DType> {
    match d {
        0 => Ok(DType::Float32),
        1 => Ok(DType::Float16),
        2 => Ok(DType::Float64),
        3 => Ok(DType::Int8),
        4 => Ok(DType::Int16),
        5 => Ok(DType::Int32),
        6 => Ok(DType::Int64),
        7 => Ok(DType::UInt8),
        8 => Ok(DType::UInt32),
        9 => Ok(DType::Bool),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown dtype: {}", d),
        )),
    }
}

/// Convert a core DType back to the wire protocol discriminant.
fn core_dtype_to_wire(d: DType) -> u32 {
    match d {
        DType::Float32 => 0,
        DType::Float16 => 1,
        DType::Float64 => 2,
        DType::Int8 => 3,
        DType::Int16 => 4,
        DType::Int32 => 5,
        DType::Int64 => 6,
        DType::UInt8 => 7,
        DType::UInt32 => 8,
        DType::Bool => 9,
        DType::BFloat16 => 10,
    }
}

struct SharedState {
    runtime: Mutex<LazyRuntime>,
    device: Device,
}

fn handle_handshake(
    shared: &SharedState,
    stream: &mut UnixStream,
) -> Option<ContainerId> {
    let msg = match wire::read_message(stream, 1024) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Handshake read failed: {}", e);
            return None;
        }
    };
    let req = match HandshakeRequest::deserialize(&msg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Handshake parse failed: {}", e);
            return None;
        }
    };

    if req.protocol_version != PROTOCOL_VERSION {
        eprintln!("Unsupported protocol version: {}", req.protocol_version);
        let resp = HandshakeResponse {
            status: 2, // HANDSHAKE_REJECTED_CAPACITY
            container_id: 0,
            granted_memory: 0,
        };
        let _ = wire::write_message(stream, &resp.serialize());
        return None;
    }

    let mut rt = shared.runtime.lock().unwrap();
    let default_memory = rt.scheduler.global_limits.max_total_memory_bytes;
    let default_tensors = rt.scheduler.global_limits.max_tensor_count;
    let requested = if req.requested_memory == 0 {
        default_memory / 4
    } else {
        req.requested_memory as usize
    };

    let config = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: requested,
        max_tensor_count: default_tensors / 4,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 64,
    };

    match rt.scheduler.register_container(config) {
        Ok(cid) => {
            let resp = HandshakeResponse {
                status: HANDSHAKE_OK,
                container_id: cid.0,
                granted_memory: requested as u64,
            };
            drop(rt);
            if wire::write_message(stream, &resp.serialize()).is_err() {
                return None;
            }
            println!("Container {} registered (memory: {} bytes)", cid, requested);
            Some(cid)
        }
        Err(e) => {
            let resp = HandshakeResponse {
                status: HANDSHAKE_REJECTED_QUOTA,
                container_id: 0,
                granted_memory: 0,
            };
            drop(rt);
            let _ = wire::write_message(stream, &resp.serialize());
            eprintln!("Container registration failed: {}", e);
            None
        }
    }
}

fn handle_eval(
    shared: &SharedState,
    container_id: ContainerId,
    request: &EvalRequest,
) -> EvalResponse {
    // Guard: reject FusedElementwise from clients (prevents MSL injection)
    for node in &request.nodes {
        if matches!(&node.op, applegpu_wire::WireOpKind::FusedElementwise { .. }) {
            return EvalResponse::Err(
                "FusedElementwise not allowed over wire protocol".to_string(),
            );
        }
    }

    let mut rt = shared.runtime.lock().unwrap();

    for td in &request.tensors {
        let dtype = match wire_dtype_to_core(td.dtype) {
            Ok(d) => d,
            Err(e) => return EvalResponse::Err(format!("Invalid dtype: {}", e)),
        };
        match Buffer::from_bytes(&shared.device, &td.data) {
            Ok(buffer) => {
                let tensor = Tensor::from_raw(
                    td.id,
                    td.shape.clone(),
                    dtype,
                    buffer,
                );
                if let Err(e) = rt.insert_tensor_for(tensor, container_id) {
                    return EvalResponse::Err(format!("Insert tensor failed: {}", e));
                }
            }
            Err(e) => return EvalResponse::Err(format!("Buffer allocation failed: {}", e)),
        }
    }

    for wire_node in &request.nodes {
        match wire_node_to_core(wire_node) {
            Ok(mut node) => {
                node.container_id = container_id;
                rt.record_op(node);
            }
            Err(e) => return EvalResponse::Err(format!("Node conversion failed: {}", e)),
        }
    }

    if let Err(e) = rt.eval(&shared.device, request.target_id) {
        return EvalResponse::Err(format!("Eval failed: {}", e));
    }

    match rt.read_bytes(request.target_id) {
        Ok(data) => {
            let shape = rt.shape(request.target_id).unwrap_or_default();
            EvalResponse::Ok {
                tensor_id: request.target_id,
                shape,
                data,
            }
        }
        Err(e) => EvalResponse::Err(format!("Read failed: {}", e)),
    }
}

fn handle_read_tensor(
    shared: &SharedState,
    _container_id: ContainerId,
    request: &ReadTensorRequest,
) -> ReadTensorResponse {
    let rt = shared.runtime.lock().unwrap();

    let shape = match rt.shape(request.tensor_id) {
        Ok(s) => s,
        Err(_) => return ReadTensorResponse::NotFound { tensor_id: request.tensor_id },
    };

    let dtype = match rt.dtype(request.tensor_id) {
        Ok(d) => core_dtype_to_wire(d),
        Err(_) => return ReadTensorResponse::NotFound { tensor_id: request.tensor_id },
    };

    match rt.read_bytes(request.tensor_id) {
        Ok(data) => ReadTensorResponse::Ok {
            tensor_id: request.tensor_id,
            shape,
            dtype,
            data,
        },
        Err(_) => ReadTensorResponse::NotFound { tensor_id: request.tensor_id },
    }
}

fn handle_connection(shared: Arc<SharedState>, mut stream: UnixStream) {
    let container_id = match handle_handshake(&shared, &mut stream) {
        Some(cid) => cid,
        None => return,
    };

    loop {
        let msg = match wire::read_message(&mut stream, MAX_MESSAGE_SIZE) {
            Ok(m) => m,
            Err(_) => break,
        };

        let magic = match wire::peek_magic(&msg) {
            Ok(m) => m,
            Err(_) => break,
        };

        let response_bytes = match &magic {
            MAGIC_REQUEST => {
                let response = match EvalRequest::deserialize(&msg) {
                    Ok(request) => handle_eval(&shared, container_id, &request),
                    Err(e) => EvalResponse::Err(format!("Deserialization failed: {}", e)),
                };
                response.serialize()
            }
            MAGIC_READ_REQ => {
                let response = match ReadTensorRequest::deserialize(&msg) {
                    Ok(request) => handle_read_tensor(&shared, container_id, &request),
                    Err(_) => ReadTensorResponse::NotFound { tensor_id: 0 },
                };
                response.serialize()
            }
            _ => {
                let response = EvalResponse::Err("Unknown message type".to_string());
                response.serialize()
            }
        };

        if wire::write_message(&mut stream, &response_bytes).is_err() {
            break;
        }
    }

    println!("Container {} disconnected, cleaning up", container_id);
    let mut rt = shared.runtime.lock().unwrap();
    if let Err(e) = rt.cleanup_container(container_id) {
        eprintln!("Cleanup failed for {}: {}", container_id, e);
    }
}

/// Default PID file path: ~/.applegpu/gpu-service.pid
fn default_pid_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    format!("{}/.applegpu/gpu-service.pid", home)
}

/// Check if a PID file exists and whether the referenced process is still alive.
/// If the process is alive, return an error (already running).
/// If the PID file is stale (process dead), remove it and return Ok.
/// If no PID file exists, return Ok.
fn check_stale_pid(pid_path: &str) -> io::Result<()> {
    if let Ok(content) = std::fs::read_to_string(pid_path) {
        if let Ok(pid) = content.trim().parse::<i32>() {
            // Check if process exists (signal 0 = no signal, just check)
            let result = unsafe { libc::kill(pid, 0) };
            if result == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!("gpu-service already running (PID {})", pid),
                ));
            }
        }
        // Stale PID file — remove it
        std::fs::remove_file(pid_path).ok();
    }
    Ok(())
}

/// Write the current process PID to the given path.
fn write_pid_file(pid_path: &str) -> io::Result<()> {
    if let Some(parent) = std::path::Path::new(pid_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(pid_path, format!("{}", std::process::id()))
}

/// Remove the PID file (best-effort).
fn remove_pid_file(pid_path: &str) {
    std::fs::remove_file(pid_path).ok();
}

fn main() {
    let socket_path = std::env::var("APPLEGPU_SOCKET")
        .unwrap_or_else(|_| applegpu_core::ipc::default_socket_path());
    let pid_path = std::env::var("APPLEGPU_PID_FILE")
        .unwrap_or_else(|_| default_pid_path());

    // Check for stale or active PID file before binding
    check_stale_pid(&pid_path).expect("PID check failed");
    write_pid_file(&pid_path).expect("Failed to write PID file");

    if let Some(parent) = std::path::Path::new(&socket_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_file(&socket_path);

    let device = Device::new().expect("No Metal GPU available");
    println!("GPU service started on {} (device: {})", socket_path, device.name());

    let shared = Arc::new(SharedState {
        runtime: Mutex::new(LazyRuntime::new()),
        device,
    });

    // Set up signal handling for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let listener = UnixListener::bind(&socket_path)
        .expect("Failed to bind Unix socket");

    // Set a timeout so we can periodically check the shutdown flag
    listener
        .set_nonblocking(false)
        .ok();

    // Use a timeout-based accept loop so we can check the `running` flag
    while running.load(Ordering::SeqCst) {
        // Set a short timeout for each accept cycle
        listener.set_nonblocking(true).ok();
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).ok();
                let shared = shared.clone();
                thread::spawn(move || handle_connection(shared, stream));
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                // No pending connection — sleep briefly and retry
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }

    println!("Shutting down gpu-service...");
    remove_pid_file(&pid_path);
    let _ = std::fs::remove_file(&socket_path);
}

#[cfg(test)]
mod tests {
    use super::*;
    use applegpu_wire::{WireOpKind, WireOpNode, WireTensorData, ReadTensorResponse};

    fn make_shared() -> SharedState {
        SharedState {
            runtime: Mutex::new(LazyRuntime::new()),
            device: Device::new().expect("No Metal GPU available"),
        }
    }

    #[test]
    fn reject_fused_elementwise_over_wire() {
        let shared = make_shared();
        let cid = {
            let mut rt = shared.runtime.lock().unwrap();
            let config = ContainerConfig {
                priority: Priority::Normal,
                max_memory_bytes: 1024 * 1024,
                max_tensor_count: 64,
                max_tensor_size_bytes: 0,
                max_pending_jobs: 64,
            };
            rt.scheduler.register_container(config).unwrap()
        };

        let request = EvalRequest {
            target_id: 10,
            tensors: vec![],
            nodes: vec![WireOpNode {
                id: 10,
                op: WireOpKind::FusedElementwise {
                    kernel_source: "kernel void evil() {}".to_string(),
                    function_name: "evil".to_string(),
                },
                inputs: vec![1, 2],
                out_shape: vec![4],
                out_dtype: 0,
            }],
        };

        let response = handle_eval(&shared, cid, &request);
        match response {
            EvalResponse::Err(msg) => {
                assert!(
                    msg.contains("FusedElementwise not allowed"),
                    "Expected rejection message, got: {}",
                    msg
                );
            }
            _ => panic!("Expected EvalResponse::Err for FusedElementwise, got Ok"),
        }
    }

    #[test]
    fn wire_dtype_to_core_maps_all_known_variants() {
        assert_eq!(wire_dtype_to_core(0).unwrap(), DType::Float32);
        assert_eq!(wire_dtype_to_core(1).unwrap(), DType::Float16);
        assert_eq!(wire_dtype_to_core(2).unwrap(), DType::Float64);
        assert_eq!(wire_dtype_to_core(3).unwrap(), DType::Int8);
        assert_eq!(wire_dtype_to_core(4).unwrap(), DType::Int16);
        assert_eq!(wire_dtype_to_core(5).unwrap(), DType::Int32);
        assert_eq!(wire_dtype_to_core(6).unwrap(), DType::Int64);
        assert_eq!(wire_dtype_to_core(7).unwrap(), DType::UInt8);
        assert_eq!(wire_dtype_to_core(8).unwrap(), DType::UInt32);
        assert_eq!(wire_dtype_to_core(9).unwrap(), DType::Bool);
    }

    #[test]
    fn wire_dtype_to_core_rejects_unknown() {
        assert!(wire_dtype_to_core(10).is_err());
        assert!(wire_dtype_to_core(255).is_err());
    }

    #[test]
    fn eval_respects_wire_dtype_int32() {
        let shared = make_shared();
        let cid = {
            let mut rt = shared.runtime.lock().unwrap();
            let config = ContainerConfig {
                priority: Priority::Normal,
                max_memory_bytes: 1024 * 1024,
                max_tensor_count: 64,
                max_tensor_size_bytes: 0,
                max_pending_jobs: 64,
            };
            rt.scheduler.register_container(config).unwrap()
        };

        // Create Int32 tensor data (dtype=5)
        let data: Vec<u8> = vec![1i32, -2, 3, -4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let request = EvalRequest {
            target_id: 42,
            tensors: vec![WireTensorData {
                id: 1,
                shape: vec![4],
                dtype: 5, // Int32
                data,
            }],
            nodes: vec![WireOpNode {
                id: 42,
                op: WireOpKind::Neg,
                inputs: vec![1],
                out_shape: vec![4],
                out_dtype: 5,
            }],
        };

        let response = handle_eval(&shared, cid, &request);
        match &response {
            EvalResponse::Err(msg) => {
                // If the backend doesn't support Int32 neg, at least it shouldn't
                // fail with "Unknown dtype" — that would mean the mapping is broken
                assert!(
                    !msg.contains("Unknown dtype") && !msg.contains("Invalid dtype"),
                    "DType mapping failed: {}",
                    msg,
                );
            }
            EvalResponse::Ok { .. } => { /* success — dtype was accepted */ }
        }
    }

    #[test]
    fn eval_rejects_unknown_dtype() {
        let shared = make_shared();
        let cid = {
            let mut rt = shared.runtime.lock().unwrap();
            let config = ContainerConfig {
                priority: Priority::Normal,
                max_memory_bytes: 1024 * 1024,
                max_tensor_count: 64,
                max_tensor_size_bytes: 0,
                max_pending_jobs: 64,
            };
            rt.scheduler.register_container(config).unwrap()
        };

        let request = EvalRequest {
            target_id: 42,
            tensors: vec![WireTensorData {
                id: 1,
                shape: vec![4],
                dtype: 99, // invalid dtype
                data: vec![0u8; 16],
            }],
            nodes: vec![],
        };

        let response = handle_eval(&shared, cid, &request);
        match response {
            EvalResponse::Err(msg) => {
                assert!(
                    msg.contains("Invalid dtype") || msg.contains("Unknown dtype"),
                    "Expected dtype error, got: {}",
                    msg,
                );
            }
            _ => panic!("Expected error for unknown dtype 99"),
        }
    }

    #[test]
    fn allow_safe_ops_over_wire() {
        let shared = make_shared();
        let cid = {
            let mut rt = shared.runtime.lock().unwrap();
            let config = ContainerConfig {
                priority: Priority::Normal,
                max_memory_bytes: 1024 * 1024,
                max_tensor_count: 64,
                max_tensor_size_bytes: 0,
                max_pending_jobs: 64,
            };
            rt.scheduler.register_container(config).unwrap()
        };

        let data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let request = EvalRequest {
            target_id: 42,
            tensors: vec![WireTensorData {
                id: 1,
                shape: vec![4],
                dtype: 0,
                data,
            }],
            nodes: vec![WireOpNode {
                id: 42,
                op: WireOpKind::Neg,
                inputs: vec![1],
                out_shape: vec![4],
                out_dtype: 0,
            }],
        };

        // Should NOT be rejected by the FusedElementwise guard
        let response = handle_eval(&shared, cid, &request);
        match &response {
            EvalResponse::Err(msg) => {
                assert!(
                    !msg.contains("FusedElementwise not allowed"),
                    "Safe op Neg was incorrectly rejected as FusedElementwise",
                );
            }
            EvalResponse::Ok { .. } => { /* success */ }
        }
    }

    #[test]
    fn read_tensor_not_found() {
        let shared = make_shared();
        let cid = {
            let mut rt = shared.runtime.lock().unwrap();
            let config = ContainerConfig {
                priority: Priority::Normal,
                max_memory_bytes: 1024 * 1024,
                max_tensor_count: 64,
                max_tensor_size_bytes: 0,
                max_pending_jobs: 64,
            };
            rt.scheduler.register_container(config).unwrap()
        };

        let req = ReadTensorRequest { tensor_id: 9999 };
        let resp = handle_read_tensor(&shared, cid, &req);
        match resp {
            ReadTensorResponse::NotFound { tensor_id } => assert_eq!(tensor_id, 9999),
            _ => panic!("Expected NotFound for non-existent tensor"),
        }
    }

    #[test]
    fn read_tensor_after_eval() {
        let shared = make_shared();
        let cid = {
            let mut rt = shared.runtime.lock().unwrap();
            let config = ContainerConfig {
                priority: Priority::Normal,
                max_memory_bytes: 1024 * 1024,
                max_tensor_count: 64,
                max_tensor_size_bytes: 0,
                max_pending_jobs: 64,
            };
            rt.scheduler.register_container(config).unwrap()
        };

        let data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let eval_req = EvalRequest {
            target_id: 42,
            tensors: vec![WireTensorData {
                id: 1,
                shape: vec![4],
                dtype: 0,
                data: data.clone(),
            }],
            nodes: vec![WireOpNode {
                id: 42,
                op: WireOpKind::Neg,
                inputs: vec![1],
                out_shape: vec![4],
                out_dtype: 0,
            }],
        };

        // Eval first
        let eval_resp = handle_eval(&shared, cid, &eval_req);
        assert!(matches!(eval_resp, EvalResponse::Ok { .. }));

        // Now read the result tensor back
        let read_req = ReadTensorRequest { tensor_id: 42 };
        let read_resp = handle_read_tensor(&shared, cid, &read_req);
        match read_resp {
            ReadTensorResponse::Ok { tensor_id, shape, dtype, data } => {
                assert_eq!(tensor_id, 42);
                assert_eq!(shape, vec![4]);
                assert_eq!(dtype, 0); // Float32
                assert_eq!(data.len(), 16); // 4 * f32
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                assert_eq!(floats, vec![-1.0, -2.0, -3.0, -4.0]);
            }
            ReadTensorResponse::NotFound { .. } => {
                panic!("Expected Ok after eval, got NotFound");
            }
        }
    }

    #[test]
    fn core_dtype_to_wire_roundtrip() {
        // Verify that wire_dtype_to_core and core_dtype_to_wire are inverses
        for wire_val in 0u32..=9 {
            let core = wire_dtype_to_core(wire_val).unwrap();
            let back = core_dtype_to_wire(core);
            assert_eq!(back, wire_val, "Roundtrip failed for wire dtype {}", wire_val);
        }
    }

    // --- PID file tests ---

    #[test]
    fn pid_file_created_on_write() {
        let dir = std::env::temp_dir().join("applegpu_test_pid_create");
        let _ = std::fs::remove_dir_all(&dir);
        let pid_path = dir.join("gpu-service.pid");
        let pid_str = pid_path.to_str().unwrap();

        write_pid_file(pid_str).expect("Failed to write PID file");
        assert!(pid_path.exists(), "PID file should exist after write");

        let content = std::fs::read_to_string(&pid_path).unwrap();
        let pid: u32 = content.trim().parse().expect("PID should be a number");
        assert_eq!(pid, std::process::id());

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_pid_detection_cleans_up() {
        let dir = std::env::temp_dir().join("applegpu_test_stale_pid");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let pid_path = dir.join("gpu-service.pid");
        let pid_str = pid_path.to_str().unwrap();

        // Write a PID that almost certainly doesn't exist (very high number)
        std::fs::write(&pid_path, "999999999").unwrap();
        assert!(pid_path.exists());

        // check_stale_pid should detect the stale PID and remove the file
        check_stale_pid(pid_str).expect("Should succeed for stale PID");
        assert!(!pid_path.exists(), "Stale PID file should be removed");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn double_start_detection_returns_error() {
        let dir = std::env::temp_dir().join("applegpu_test_double_start");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let pid_path = dir.join("gpu-service.pid");
        let pid_str = pid_path.to_str().unwrap();

        // Write our own PID (which is definitely alive)
        std::fs::write(&pid_path, format!("{}", std::process::id())).unwrap();

        // check_stale_pid should return an error because the process is alive
        let result = check_stale_pid(pid_str);
        assert!(result.is_err(), "Should error when process is still alive");
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AddrInUse);
        assert!(
            err.to_string().contains("already running"),
            "Error message should mention 'already running', got: {}",
            err
        );

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_stale_pid_no_file_is_ok() {
        let dir = std::env::temp_dir().join("applegpu_test_no_pid");
        let _ = std::fs::remove_dir_all(&dir);
        let pid_path = dir.join("gpu-service.pid");
        let pid_str = pid_path.to_str().unwrap();

        // No PID file exists — should be fine
        let result = check_stale_pid(pid_str);
        assert!(result.is_ok(), "No PID file should be fine");
    }

    #[test]
    fn remove_pid_file_cleans_up() {
        let dir = std::env::temp_dir().join("applegpu_test_remove_pid");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let pid_path = dir.join("gpu-service.pid");
        let pid_str = pid_path.to_str().unwrap();

        write_pid_file(pid_str).unwrap();
        assert!(pid_path.exists());

        remove_pid_file(pid_str);
        assert!(!pid_path.exists(), "PID file should be removed after cleanup");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}
