use applegpu_core::device::Device;
use applegpu_core::eager::EagerRuntime;
use applegpu_core::tensor::DType;
use std::sync::Mutex;

// Streaming batch uses global static state, so tests must run serially.
static STREAMING_LOCK: Mutex<()> = Mutex::new(());

// ── Binary op tests ──────────────────────────────────────────────────

#[test]
fn test_eager_add_contiguous() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    let (b_id, b_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 4)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        std::slice::from_raw_parts_mut(b_ptr as *mut f32, 4)
            .copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
    }

    let (out_id, out_ptr) = rt.binary_op(&device, "elementwise_add", a_id, b_id).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 4) };
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);

    // Verify output tensor metadata
    assert_eq!(rt.shape(out_id).unwrap(), vec![4]);
    assert_eq!(rt.dtype(out_id).unwrap(), DType::Float32);

    rt.end_streaming();
}

#[test]
fn test_eager_add_broadcast() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[2, 3], DType::Float32).unwrap();
    let (b_id, b_ptr) = rt.alloc(&device, &[3], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 6)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        std::slice::from_raw_parts_mut(b_ptr as *mut f32, 3)
            .copy_from_slice(&[10.0, 20.0, 30.0]);
    }

    let (_, out_ptr) = rt.binary_op(&device, "elementwise_add", a_id, b_id).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 6) };
    assert_eq!(result, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

    rt.end_streaming();
}

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

#[test]
fn test_eager_streaming_lifecycle() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);
    assert!(rt.is_streaming());
    rt.flush_and_wait();
    assert!(rt.is_streaming()); // still active after flush
    rt.end_streaming();
    assert!(!rt.is_streaming());
}
