use applegpu_core::compute;
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::ops;
use applegpu_core::tensor::Tensor;
use std::sync::Mutex;

// Streaming batch uses global static state, so tests must run serially.
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn get_device() -> Option<Device> {
    Device::new().ok()
}

#[test]
fn streaming_batch_basic_lifecycle() {
    let _guard = TEST_LOCK.lock().unwrap();
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let queue = compute::get_shared_queue(&device);
    assert!(!queue.is_null());
    assert!(!compute::streaming_is_active());
    compute::begin_streaming_batch(queue);
    assert!(compute::streaming_is_active());
    compute::flush_streaming_batch();
    assert!(compute::streaming_is_active());
    compute::end_streaming_batch();
    assert!(!compute::streaming_is_active());
}

#[test]
fn streaming_batch_idempotent_begin() {
    let _guard = TEST_LOCK.lock().unwrap();
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);
    compute::begin_streaming_batch(queue); // no-op
    assert!(compute::streaming_is_active());
    compute::end_streaming_batch();
    assert!(!compute::streaming_is_active());
}

#[test]
fn streaming_batch_end_when_inactive() {
    let _guard = TEST_LOCK.lock().unwrap();
    assert!(!compute::streaming_is_active());
    compute::end_streaming_batch();
    assert!(!compute::streaming_is_active());
}

#[test]
fn streaming_batch_flush_when_inactive() {
    let _guard = TEST_LOCK.lock().unwrap();
    assert!(!compute::streaming_is_active());
    compute::flush_streaming_batch();
    assert!(!compute::streaming_is_active());
}

#[test]
fn streaming_eval_chain_then_read() {
    let _guard = TEST_LOCK.lock().unwrap();
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
    let prod_id = ops::mul(&mut rt, sum_id, a_id).unwrap();

    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);
    rt.eval(&device, prod_id).unwrap();
    let result = rt.read_f32(prod_id).unwrap();
    assert_eq!(result, &[11.0, 44.0, 99.0, 176.0]);
    compute::end_streaming_batch();
}

#[test]
fn streaming_multiple_evals_then_read() {
    let _guard = TEST_LOCK.lock().unwrap();
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();
    let a = Tensor::from_f32(&device, vec![3], &[1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![3], &[4.0, 5.0, 6.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();
    let prod_id = ops::mul(&mut rt, a_id, b_id).unwrap();

    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);
    rt.eval(&device, sum_id).unwrap();
    rt.eval(&device, prod_id).unwrap();
    let sum_result = rt.read_f32(sum_id).unwrap();
    let prod_result = rt.read_f32(prod_id).unwrap();
    assert_eq!(sum_result, &[5.0, 7.0, 9.0]);
    assert_eq!(prod_result, &[4.0, 10.0, 18.0]);
    compute::end_streaming_batch();
}

#[test]
fn streaming_10_ops_chain_correctness() {
    let _guard = TEST_LOCK.lock().unwrap();
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();
    let one = Tensor::from_f32(&device, vec![1], &[1.0]).unwrap();
    let one_id = one.meta.id;
    rt.insert_tensor(one).unwrap();
    let mut current_id = one_id;
    for _ in 0..10 {
        let inc = Tensor::from_f32(&device, vec![1], &[1.0]).unwrap();
        let inc_id = inc.meta.id;
        rt.insert_tensor(inc).unwrap();
        current_id = ops::add(&mut rt, current_id, inc_id).unwrap();
    }
    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);
    rt.eval(&device, current_id).unwrap();
    let result = rt.read_f32(current_id).unwrap();
    assert_eq!(result, &[11.0]);
    compute::end_streaming_batch();
}

#[test]
fn streaming_error_recovery() {
    let _guard = TEST_LOCK.lock().unwrap();
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let mut rt = LazyRuntime::new();
    let a = Tensor::from_f32(&device, vec![3], &[1.0, 2.0, 3.0]).unwrap();
    let a_id = a.meta.id;
    rt.insert_tensor(a).unwrap();

    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);

    let b = Tensor::from_f32(&device, vec![3], &[4.0, 5.0, 6.0]).unwrap();
    let b_id = b.meta.id;
    rt.insert_tensor(b).unwrap();
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, sum_id).unwrap();

    // Eval a non-existent tensor — should error
    let bad_result = rt.eval(&device, 999999);
    assert!(bad_result.is_err());
    assert!(compute::streaming_is_active());

    // Subsequent valid ops should work
    let c = Tensor::from_f32(&device, vec![3], &[10.0, 20.0, 30.0]).unwrap();
    let c_id = c.meta.id;
    rt.insert_tensor(c).unwrap();
    let sum2_id = ops::add(&mut rt, a_id, c_id).unwrap();
    rt.eval(&device, sum2_id).unwrap();
    let result = rt.read_f32(sum2_id).unwrap();
    assert_eq!(result, &[11.0, 22.0, 33.0]);
    compute::end_streaming_batch();
}
