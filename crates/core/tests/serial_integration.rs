use applegpu_core::buffer::Buffer;
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::ops;
use applegpu_core::serial::{EvalRequest, TensorData};
use applegpu_core::tensor::{DType, Tensor};

/// Simulates the full remote execution flow in-process:
/// client builds graph → serializes → deserializes → server executes → returns result
#[test]
fn serialize_and_execute_remotely_in_process() {
    let device = match Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    // === Client side: build graph ===
    let mut client_rt = LazyRuntime::new();
    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;

    let a_bytes: Vec<u8> = a.as_f32_slice().iter().flat_map(|f| f.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.as_f32_slice().iter().flat_map(|f| f.to_le_bytes()).collect();

    client_rt.insert_tensor(a).unwrap();
    client_rt.insert_tensor(b).unwrap();

    let sum_id = ops::add(&mut client_rt, a_id, b_id).unwrap();
    let relu_id = ops::relu(&mut client_rt, sum_id).unwrap();

    // Build eval request
    let nodes: Vec<_> = [sum_id, relu_id]
        .iter()
        .filter_map(|&nid| client_rt.graph_node(nid).cloned())
        .collect();

    let request = EvalRequest {
        target_id: relu_id,
        tensors: vec![
            TensorData { id: a_id, shape: vec![4], dtype: DType::Float32, data: a_bytes },
            TensorData { id: b_id, shape: vec![4], dtype: DType::Float32, data: b_bytes },
        ],
        nodes,
    };

    // === Simulate wire transfer ===
    let wire = request.serialize();
    let received = EvalRequest::deserialize(&wire).unwrap();

    // === Server side: execute ===
    let mut server_rt = LazyRuntime::new();
    for td in &received.tensors {
        let buffer = Buffer::from_bytes(&device, &td.data).unwrap();
        let tensor = Tensor::from_raw(td.id, td.shape.clone(), td.dtype, buffer);
        server_rt.insert_tensor(tensor).unwrap();
    }
    for node in &received.nodes {
        let _ = server_rt.record_op(node.clone());
    }

    server_rt.eval(&device, received.target_id).unwrap();

    let result = server_rt.read_f32(received.target_id).unwrap();
    // relu(add([1,2,3,4], [10,20,30,40])) = relu([11,22,33,44]) = [11,22,33,44]
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn serialize_matmul_remotely() {
    let device = match Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    let mut client_rt = LazyRuntime::new();
    let a = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;

    let a_bytes: Vec<u8> = a.as_f32_slice().iter().flat_map(|f| f.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.as_f32_slice().iter().flat_map(|f| f.to_le_bytes()).collect();

    client_rt.insert_tensor(a).unwrap();
    client_rt.insert_tensor(b).unwrap();

    let c_id = ops::matmul(&mut client_rt, a_id, b_id).unwrap();

    let nodes: Vec<_> = [c_id]
        .iter()
        .filter_map(|&nid| client_rt.graph_node(nid).cloned())
        .collect();

    let request = EvalRequest {
        target_id: c_id,
        tensors: vec![
            TensorData { id: a_id, shape: vec![2, 2], dtype: DType::Float32, data: a_bytes },
            TensorData { id: b_id, shape: vec![2, 2], dtype: DType::Float32, data: b_bytes },
        ],
        nodes,
    };

    let wire = request.serialize();
    let received = EvalRequest::deserialize(&wire).unwrap();

    let mut server_rt = LazyRuntime::new();
    for td in &received.tensors {
        let buffer = Buffer::from_bytes(&device, &td.data).unwrap();
        let tensor = Tensor::from_raw(td.id, td.shape.clone(), td.dtype, buffer);
        server_rt.insert_tensor(tensor).unwrap();
    }
    for node in &received.nodes {
        let _ = server_rt.record_op(node.clone());
    }

    server_rt.eval(&device, received.target_id).unwrap();

    let result = server_rt.read_f32(received.target_id).unwrap();
    assert_eq!(result, &[19.0, 22.0, 43.0, 50.0]);
}
