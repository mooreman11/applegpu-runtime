use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::scheduler::ContainerId;
use applegpu_core::tensor::Tensor;

fn get_device() -> Option<Device> {
    Device::new().ok()
}

#[test]
fn test_eval_reuses_pooled_buffers() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = applegpu_core::ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();
    assert_eq!(rt.read_f32(c_id).unwrap(), &[11.0, 22.0, 33.0, 44.0]);
    let stats1 = rt.pool.stats();
    assert!(stats1.misses > 0);

    rt.destroy(c_id).unwrap();
    assert!(rt.pool.pooled_bytes() > 0);

    let c2_id = applegpu_core::ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c2_id).unwrap();
    assert_eq!(rt.read_f32(c2_id).unwrap(), &[11.0, 22.0, 33.0, 44.0]);
    let stats2 = rt.pool.stats();
    assert!(stats2.hits > stats1.hits);
}

#[test]
fn test_destroy_returns_buffer_to_pool() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = applegpu_core::ops::add(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();

    let before = rt.pool.pooled_bytes();
    rt.destroy(c_id).unwrap();
    assert!(rt.pool.pooled_bytes() > before);
}

#[test]
fn test_pool_respects_scheduler_limits() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let id = t.meta.id;
    rt.insert_tensor(t).unwrap();

    let (bytes, count) = rt.scheduler.global_usage();
    assert_eq!(bytes, 16);
    assert_eq!(count, 1);

    rt.destroy(id).unwrap();
    assert_eq!(rt.scheduler.global_usage(), (0, 0));
}

#[test]
fn test_backward_compat_matmul() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = applegpu_core::ops::matmul(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();
    assert_eq!(rt.read_f32(c_id).unwrap(), &[19.0, 22.0, 43.0, 50.0]);
}
