use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::ops;
use applegpu_core::tensor::{Tensor, DType};

fn get_device() -> Option<Device> {
    Device::new().ok()
}

#[test]
fn preallocated_tensor_is_materialized() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();
    let buf = rt.pool.acquire(&device, 4 * 4).unwrap();
    let id = applegpu_core::tensor::next_tensor_id();
    let tensor = Tensor::from_raw(id, vec![4], DType::Float32, buf);
    rt.insert_preallocated(tensor).unwrap();
    assert!(rt.is_materialized(id));
    assert!(!rt.is_pending(id));
}

#[test]
fn preallocated_tensor_usable_as_op_input() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();
    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_preallocated(a).unwrap();
    rt.insert_preallocated(b).unwrap();
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, sum_id).unwrap();
    let result = rt.read_f32(sum_id).unwrap();
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn preallocated_buffer_stored_for_later_use() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();
    let buf = rt.pool.acquire(&device, 4 * 4).unwrap();
    let id = applegpu_core::tensor::next_tensor_id();
    rt.insert_preallocated_buffer(id, buf);
    assert!(!rt.is_materialized(id)); // buffer stored but not yet a full tensor
    assert!(!rt.is_pending(id));
    assert!(rt.has_preallocated_buffer(id));
}
