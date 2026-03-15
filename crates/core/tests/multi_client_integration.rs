use std::os::unix::net::UnixStream;
use std::thread;

use applegpu_wire::{
    self as wire,
    EvalRequest, EvalResponse, WireOpNode, WireOpKind, WireTensorData,
    HandshakeRequest, HandshakeResponse,
    HANDSHAKE_OK, PROTOCOL_VERSION, MAX_MESSAGE_SIZE,
};

fn mock_server(mut stream: UnixStream) {
    let msg = wire::read_message(&mut stream, 1024).unwrap();
    let req = HandshakeRequest::deserialize(&msg).unwrap();
    assert_eq!(req.protocol_version, PROTOCOL_VERSION);

    let resp = HandshakeResponse {
        status: HANDSHAKE_OK,
        container_id: 42,
        granted_memory: req.requested_memory,
    };
    wire::write_message(&mut stream, &resp.serialize()).unwrap();

    let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
    let eval_req = EvalRequest::deserialize(&msg).unwrap();
    assert_eq!(eval_req.target_id, 100);

    let data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
        .iter().flat_map(|f| f.to_le_bytes()).collect();
    let eval_resp = EvalResponse::Ok {
        tensor_id: 100,
        shape: vec![4],
        data,
    };
    wire::write_message(&mut stream, &eval_resp.serialize()).unwrap();
}

#[test]
fn wire_protocol_handshake_and_eval() {
    let (client_stream, server_stream) = UnixStream::pair().unwrap();

    let server = thread::spawn(move || mock_server(server_stream));

    let mut client = client_stream;

    let hs = HandshakeRequest {
        protocol_version: PROTOCOL_VERSION,
        requested_memory: 1024 * 1024,
    };
    wire::write_message(&mut client, &hs.serialize()).unwrap();

    let msg = wire::read_message(&mut client, 1024).unwrap();
    let hs_resp = HandshakeResponse::deserialize(&msg).unwrap();
    assert_eq!(hs_resp.status, HANDSHAKE_OK);
    assert_eq!(hs_resp.container_id, 42);

    let data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
        .iter().flat_map(|f| f.to_le_bytes()).collect();
    let eval_req = EvalRequest {
        target_id: 100,
        tensors: vec![WireTensorData { id: 1, shape: vec![4], dtype: 0, data }],
        nodes: vec![WireOpNode {
            id: 100, op: WireOpKind::Neg, inputs: vec![1],
            out_shape: vec![4], out_dtype: 0,
        }],
    };
    wire::write_message(&mut client, &eval_req.serialize()).unwrap();

    let msg = wire::read_message(&mut client, MAX_MESSAGE_SIZE).unwrap();
    let eval_resp = EvalResponse::deserialize(&msg).unwrap();
    match eval_resp {
        EvalResponse::Ok { tensor_id, shape, data } => {
            assert_eq!(tensor_id, 100);
            assert_eq!(shape, vec![4]);
            let floats: Vec<f32> = data.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(floats, vec![1.0, 2.0, 3.0, 4.0]);
        }
        _ => panic!("Expected Ok response"),
    }

    server.join().unwrap();
}

#[test]
fn wire_protocol_multiple_concurrent_clients() {
    let num_clients = 4;
    let mut handles = Vec::new();

    for client_idx in 0..num_clients {
        let (client_stream, server_stream) = UnixStream::pair().unwrap();

        let server_handle = thread::spawn(move || {
            let mut stream = server_stream;
            let msg = wire::read_message(&mut stream, 1024).unwrap();
            let _ = HandshakeRequest::deserialize(&msg).unwrap();
            let resp = HandshakeResponse {
                status: HANDSHAKE_OK,
                container_id: client_idx as u64 + 1,
                granted_memory: 1024 * 1024,
            };
            wire::write_message(&mut stream, &resp.serialize()).unwrap();

            for _ in 0..10 {
                let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
                let _ = EvalRequest::deserialize(&msg).unwrap();
                let resp = EvalResponse::Ok {
                    tensor_id: 1,
                    shape: vec![2],
                    data: vec![0; 8],
                };
                wire::write_message(&mut stream, &resp.serialize()).unwrap();
            }
        });

        let client_handle = thread::spawn(move || {
            let mut stream = client_stream;
            let hs = HandshakeRequest {
                protocol_version: PROTOCOL_VERSION,
                requested_memory: 1024 * 1024,
            };
            wire::write_message(&mut stream, &hs.serialize()).unwrap();
            let msg = wire::read_message(&mut stream, 1024).unwrap();
            let resp = HandshakeResponse::deserialize(&msg).unwrap();
            assert_eq!(resp.status, HANDSHAKE_OK);

            for i in 0..10u64 {
                let req = EvalRequest {
                    target_id: i,
                    tensors: vec![],
                    nodes: vec![WireOpNode {
                        id: i, op: WireOpKind::Add, inputs: vec![],
                        out_shape: vec![2], out_dtype: 0,
                    }],
                };
                wire::write_message(&mut stream, &req.serialize()).unwrap();
                let msg = wire::read_message(&mut stream, MAX_MESSAGE_SIZE).unwrap();
                let resp = EvalResponse::deserialize(&msg).unwrap();
                assert!(matches!(resp, EvalResponse::Ok { .. }));
            }
        });

        handles.push(server_handle);
        handles.push(client_handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn container_id_stamping() {
    use applegpu_core::lazy::LazyRuntime;
    use applegpu_core::scheduler::{ContainerId, ContainerConfig, Priority};
    use applegpu_core::serial::wire_node_to_core;

    let mut rt = LazyRuntime::new();

    let config = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 10 * 1024 * 1024,
        max_tensor_count: 100,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let cid = rt.scheduler.register_container(config).unwrap();
    assert_ne!(cid, ContainerId::DEFAULT);

    let wire_node = WireOpNode {
        id: 200,
        op: WireOpKind::Add,
        inputs: vec![1, 2],
        out_shape: vec![4],
        out_dtype: 0,
    };
    let mut node = wire_node_to_core(&wire_node).unwrap();

    assert_eq!(node.container_id, ContainerId::DEFAULT);

    node.container_id = cid;
    rt.record_op(node);

    let graph_node = rt.graph_node(200).unwrap();
    assert_eq!(graph_node.container_id, cid);
}
