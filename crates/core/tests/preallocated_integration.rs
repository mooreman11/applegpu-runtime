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

#[test]
fn eval_writes_into_preallocated_output_buffer() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    // Create input tensors normally
    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    // Record an add op — this returns a new tensor ID for the output
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();

    // Pre-allocate the OUTPUT buffer BEFORE eval
    let out_buf = rt.pool.acquire(&device, 4 * 4).unwrap(); // 4 floats
    let out_ptr = out_buf.contents() as usize;
    rt.insert_preallocated_buffer(sum_id, out_buf);

    // Eval should write into the pre-allocated buffer
    rt.eval(&device, sum_id).unwrap();

    // Verify correct result
    let result = rt.read_f32(sum_id).unwrap();
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);

    // Verify it used the same buffer (same pointer)
    let final_ptr = rt.get_tensor_ptr(sum_id).unwrap();
    assert_eq!(out_ptr, final_ptr, "eval should write into pre-allocated buffer, not allocate new");
}

#[test]
fn deferred_free_keeps_tensor_alive_until_eval() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_preallocated(a).unwrap();
    rt.insert_preallocated(b).unwrap();

    // Record an op that depends on a and b
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();

    // Try to free a — should be deferred because sum_id's graph node depends on it
    let freed = rt.try_deferred_free(a_id);
    assert!(!freed, "Should not free: tensor has graph dependents");
    assert!(rt.is_materialized(a_id), "Tensor should still be alive");

    // Eval the dependent op
    rt.eval(&device, sum_id).unwrap();
    let result = rt.read_f32(sum_id).unwrap();
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);

    // Now free should succeed (no more graph dependents)
    let freed2 = rt.try_deferred_free(a_id);
    assert!(freed2, "Should free: no more dependents after eval");
    assert!(!rt.is_materialized(a_id), "Tensor should be gone");
}

#[test]
fn deferred_free_processes_after_eval() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![3], &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![3], &[4.0, 5.0, 6.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_preallocated(a).unwrap();
    rt.insert_preallocated(b).unwrap();

    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();

    // Defer free of both inputs
    assert!(!rt.try_deferred_free(a_id));
    assert!(!rt.try_deferred_free(b_id));

    // Eval processes deferred frees automatically
    rt.eval(&device, sum_id).unwrap();

    // Both inputs should be freed after eval (they were deferred)
    assert!(!rt.is_materialized(a_id), "a should be freed after eval");
    assert!(!rt.is_materialized(b_id), "b should be freed after eval");
}
