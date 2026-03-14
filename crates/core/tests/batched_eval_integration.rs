use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::ops;
use applegpu_core::tensor::Tensor;

fn get_device() -> Option<Device> {
    Device::new().ok()
}

#[test]
fn batched_eval_add_mul_relu_chain() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    // Chain: (a + b) * a -> relu
    // (a + b) = [11, 22, 33, 44]
    // (a + b) * a = [11, 44, 99, 176]
    // relu(...) = [11, 44, 99, 176] (all positive)
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();
    let prod_id = ops::mul(&mut rt, sum_id, a_id).unwrap();
    let result_id = ops::relu(&mut rt, prod_id).unwrap();

    rt.eval(&device, result_id).unwrap();
    let result = rt.read_f32(result_id).unwrap();
    assert_eq!(result, &[11.0, 44.0, 99.0, 176.0]);
}

#[test]
fn batched_eval_with_negative_relu() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, -2.0, 3.0, -4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    // sub then relu: relu(a - b) = relu([-9, -22, -27, -44]) = [0, 0, 0, 0]
    let diff_id = ops::sub(&mut rt, a_id, b_id).unwrap();
    let result_id = ops::relu(&mut rt, diff_id).unwrap();

    rt.eval(&device, result_id).unwrap();
    let result = rt.read_f32(result_id).unwrap();
    assert_eq!(result, &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn batched_eval_long_chain() {
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[4.0, 9.0, 16.0, 25.0]).unwrap();
    let a_id = a.meta.id;
    rt.insert_tensor(a).unwrap();

    // sqrt -> neg -> neg -> sqrt of the original should NOT work as sqrt(neg(...)) is NaN
    // Instead: sqrt -> relu -> sqrt is fine: sqrt([4,9,16,25]) = [2,3,4,5], relu = same, sqrt = [~1.41, ~1.73, 2, ~2.24]
    // Actually let's do a simple chain: neg -> neg = identity
    let neg1 = ops::neg(&mut rt, a_id).unwrap();
    let neg2 = ops::neg(&mut rt, neg1).unwrap();

    rt.eval(&device, neg2).unwrap();
    let result = rt.read_f32(neg2).unwrap();
    assert_eq!(result, &[4.0, 9.0, 16.0, 25.0]);
}

#[test]
fn batched_eval_single_op() {
    // Ensure single-op graphs still work (last_cb is the only cb)
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, sum_id).unwrap();
    let result = rt.read_f32(sum_id).unwrap();
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
}
