use applegpu_core::lazy::LazyRuntime;
use applegpu_core::scheduler::{ContainerConfig, Priority};
use applegpu_core::graph::{OpKind, OpNode};
use applegpu_core::tensor::{DType, Shape};

#[test]
fn cleanup_container_removes_tensors_and_nodes() {
    let mut rt = LazyRuntime::new();
    let config = ContainerConfig {
        priority: Priority::Normal,
        max_memory_bytes: 10 * 1024 * 1024,
        max_tensor_count: 100,
        max_tensor_size_bytes: 0,
        max_pending_jobs: 10,
    };
    let cid = rt.scheduler.register_container(config).unwrap();
    let node = OpNode {
        id: 100,
        op: OpKind::Add,
        inputs: vec![1, 2],
        out_shape: Shape::new(vec![4]).unwrap(),
        out_dtype: DType::Float32,
        container_id: cid,
    };
    rt.record_op(node);
    assert!(rt.is_pending(100));
    rt.cleanup_container(cid).unwrap();
    assert!(!rt.is_pending(100));
    assert!(rt.scheduler.container_usage(cid).is_none());
}
