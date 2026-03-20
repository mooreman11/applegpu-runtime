use applegpu_core::device::Device;
use applegpu_core::eager::EagerRuntime;
use applegpu_core::tensor::DType;

#[test]
fn test_eager_register_and_query() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    let (id, ptr) = rt.alloc(&device, &[2, 3], DType::Float32).unwrap();
    assert!(!ptr.is_null());
    assert_eq!(rt.shape(id).unwrap(), vec![2, 3]);
    assert_eq!(rt.dtype(id).unwrap(), DType::Float32);
    assert!(rt.is_contiguous(id).unwrap());
}

#[test]
fn test_eager_view_shares_buffer() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    let (base_id, _) = rt.alloc(&device, &[4, 8], DType::Float32).unwrap();

    // Create a transposed view: shape [8,4], strides [1,8] (non-contiguous)
    let view_id = rt.create_view(base_id, &[8, 4], &[1, 8], 0).unwrap();

    let base_t = rt.get(base_id).unwrap();
    let view_t = rt.get(view_id).unwrap();
    // Same buffer, same data pointer (offset=0 for both)
    assert_eq!(base_t.data_ptr(), view_t.data_ptr());
    assert_eq!(rt.shape(view_id).unwrap(), vec![8, 4]);
    assert!(!rt.is_contiguous(view_id).unwrap());

    // Freeing the base should not affect the view (Arc keeps buffer alive)
    rt.free(base_id);
    assert!(rt.get(base_id).is_err());
    assert!(rt.get(view_id).is_ok());
    rt.free(view_id);
}

#[test]
fn test_eager_pool_recycles_buffers() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    let (id1, _) = rt.alloc(&device, &[256], DType::Float32).unwrap();
    rt.free(id1);
    let stats_after_free = rt.pool_stats();
    let (id2, _) = rt.alloc(&device, &[256], DType::Float32).unwrap();
    let stats_after_reuse = rt.pool_stats();
    assert!(stats_after_reuse.hits > stats_after_free.hits);
    rt.free(id2);
}
