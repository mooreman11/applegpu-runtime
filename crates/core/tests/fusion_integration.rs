use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::ops;
use applegpu_core::tensor::Tensor;

#[test]
fn fused_add_relu() {
    let device = match Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, -2.0, 3.0, -4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    // add -> relu should be fused into a single kernel
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();
    let relu_id = ops::relu(&mut rt, sum_id).unwrap();

    rt.eval(&device, relu_id).unwrap();
    // relu(add([1,-2,3,-4], [10,20,30,40])) = relu([11,18,33,36]) = [11,18,33,36]
    assert_eq!(rt.read_f32(relu_id).unwrap(), &[11.0, 18.0, 33.0, 36.0]);
}

#[test]
fn fused_chain_of_three() {
    let device = match Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, 4.0, 9.0, 16.0]).unwrap();
    let a_id = a.meta.id;
    rt.insert_tensor(a).unwrap();

    // sqrt -> neg -> relu should be fused
    let sqrt_id = ops::sqrt(&mut rt, a_id).unwrap();
    let neg_id = ops::neg(&mut rt, sqrt_id).unwrap();
    let relu_id = ops::relu(&mut rt, neg_id).unwrap();

    rt.eval(&device, relu_id).unwrap();
    // sqrt([1,4,9,16]) = [1,2,3,4]
    // neg([1,2,3,4]) = [-1,-2,-3,-4]
    // relu([-1,-2,-3,-4]) = [0,0,0,0]
    assert_eq!(rt.read_f32(relu_id).unwrap(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn unfused_matmul_not_affected() {
    let device = match Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    // matmul can't be fused but should still work
    let c_id = ops::matmul(&mut rt, a_id, b_id).unwrap();
    let relu_id = ops::relu(&mut rt, c_id).unwrap();

    rt.eval(&device, relu_id).unwrap();
    assert_eq!(rt.read_f32(relu_id).unwrap(), &[19.0, 22.0, 43.0, 50.0]);
}
