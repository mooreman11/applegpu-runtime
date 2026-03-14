use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::scheduler::{ContainerId, ContainerConfig, Priority};
use applegpu_core::tensor::Tensor;

fn get_device() -> Option<Device> {
    Device::new().ok()
}

#[test]
fn test_default_container_backward_compat() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();
    let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let id = t.meta.id;
    rt.insert_tensor(t).unwrap();
    assert!(rt.is_materialized(id));
    // 4 f32s = 16 bytes
    let (bytes, count) = rt.scheduler.container_usage(ContainerId::DEFAULT).unwrap();
    assert!(bytes >= 16, "expected at least 16 bytes, got {}", bytes);
    assert_eq!(count, 1);
}

#[test]
fn test_resource_tracking_through_lifecycle() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let id = t.meta.id;
    rt.insert_tensor(t).unwrap();

    let (bytes, count) = rt.scheduler.global_usage();
    assert!(bytes > 0);
    assert_eq!(count, 1);

    rt.destroy(id).unwrap();
    assert_eq!(rt.scheduler.global_usage(), (0, 0));
}

#[test]
fn test_multi_container_round_robin() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let config_a = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 100,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let config_b = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 1024 * 1024,
        max_tensor_count: 100,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };

    let a = rt.scheduler.register_container(config_a).unwrap();
    let b = rt.scheduler.register_container(config_b).unwrap();

    let t1 = Tensor::from_f32(&device, vec![2], &[1.0, 2.0]).unwrap();
    let t2 = Tensor::from_f32(&device, vec![2], &[3.0, 4.0]).unwrap();
    let t1_id = t1.meta.id;
    let t2_id = t2.meta.id;
    rt.insert_tensor(t1).unwrap();
    rt.insert_tensor(t2).unwrap();

    let c1 = applegpu_core::ops::add(&mut rt, t1_id, t2_id).unwrap();
    let c2 = applegpu_core::ops::mul(&mut rt, t1_id, t2_id).unwrap();

    rt.scheduler.submit(a, c1).unwrap();
    rt.scheduler.submit(b, c2).unwrap();

    let j1 = rt.run_next(&device).unwrap();
    assert!(j1.is_some());
    let j2 = rt.run_next(&device).unwrap();
    assert!(j2.is_some());

    assert!(rt.is_materialized(c1));
    assert!(rt.is_materialized(c2));
}

#[test]
fn test_eval_attributes_intermediates_to_container() {
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let a_id = a.meta.id;
    rt.insert_tensor(a).unwrap();

    let neg_id = applegpu_core::ops::neg(&mut rt, a_id).unwrap();
    let relu_id = applegpu_core::ops::relu(&mut rt, neg_id).unwrap();

    rt.eval(&device, relu_id).unwrap();
    assert!(rt.is_materialized(relu_id));

    // Verify container attribution
    assert_eq!(rt.scheduler.tensor_owner(relu_id), Some(ContainerId::DEFAULT));
}
