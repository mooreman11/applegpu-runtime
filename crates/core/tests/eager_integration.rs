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

// ── Unary op tests ──────────────────────────────────────────────────

#[test]
fn test_eager_relu() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 4)
            .copy_from_slice(&[-1.0, 2.0, -3.0, 4.0]);
    }

    let (out_id, out_ptr) = rt.unary_op(&device, "elementwise_relu", a_id).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 4) };
    assert_eq!(result, &[0.0, 2.0, 0.0, 4.0]);

    assert_eq!(rt.shape(out_id).unwrap(), vec![4]);
    assert_eq!(rt.dtype(out_id).unwrap(), DType::Float32);

    rt.end_streaming();
}

#[test]
fn test_eager_neg() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 4)
            .copy_from_slice(&[-1.0, 2.0, -3.0, 4.0]);
    }

    let (_, out_ptr) = rt.unary_op(&device, "elementwise_neg", a_id).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 4) };
    assert_eq!(result, &[1.0, -2.0, 3.0, -4.0]);

    rt.end_streaming();
}

// ── Matmul tests ────────────────────────────────────────────────────

#[test]
fn test_eager_matmul() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    // [2, 3] @ [3, 4] → [2, 4]
    let (a_id, a_ptr) = rt.alloc(&device, &[2, 3], DType::Float32).unwrap();
    let (b_id, b_ptr) = rt.alloc(&device, &[3, 4], DType::Float32).unwrap();
    unsafe {
        // First two rows of 3x3 identity
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 6)
            .copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        std::slice::from_raw_parts_mut(b_ptr as *mut f32, 12)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    let (out_id, out_ptr) = rt.matmul(&device, a_id, b_id).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 8) };
    // Row 0: [1,0,0] @ B = [1,2,3,4]
    // Row 1: [0,1,0] @ B = [5,6,7,8]
    assert_eq!(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    assert_eq!(rt.shape(out_id).unwrap(), vec![2, 4]);
    assert_eq!(rt.dtype(out_id).unwrap(), DType::Float32);

    rt.end_streaming();
}

// ── make_contiguous tests ────────────────────────────────────────────

#[test]
fn test_eager_make_contiguous_noop_for_contiguous() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    let _lock = STREAMING_LOCK.lock().unwrap();
    rt.begin_streaming(&device);

    let (id, ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    let (contig_id, contig_ptr) = rt.make_contiguous(&device, id).unwrap();
    // Should return same id and pointer for already-contiguous tensor
    assert_eq!(contig_id, id);
    assert_eq!(contig_ptr, ptr);

    rt.end_streaming();
}

#[test]
fn test_eager_make_contiguous() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    let _lock = STREAMING_LOCK.lock().unwrap();
    rt.begin_streaming(&device);

    // Create [2, 3] and transpose to [3, 2] (non-contiguous)
    let (base_id, base_ptr) = rt.alloc(&device, &[2, 3], DType::Float32).unwrap();
    unsafe {
        // Row-major: [[1,2,3],[4,5,6]]
        std::slice::from_raw_parts_mut(base_ptr as *mut f32, 6)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // Transposed view: shape [3, 2], strides [1, 3]
    let view_id = rt.create_view(base_id, &[3, 2], &[1, 3], 0).unwrap();
    assert!(!rt.is_contiguous(view_id).unwrap());

    let (contig_id, contig_ptr) = rt.make_contiguous(&device, view_id).unwrap();
    assert!(rt.is_contiguous(contig_id).unwrap());
    assert_ne!(contig_id, view_id);

    // Transposed [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]] → row-major: [1,4,2,5,3,6]
    let result = unsafe { std::slice::from_raw_parts(contig_ptr as *const f32, 6) };
    assert_eq!(result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    rt.end_streaming();
}

// ── In-place binary op tests ────────────────────────────────────────

#[test]
fn test_eager_inplace_add() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    let (b_id, _) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 4)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        std::slice::from_raw_parts_mut(rt.get(b_id).unwrap().data_ptr() as *mut f32, 4)
            .copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
    }

    rt.inplace_binary_op(&device, "elementwise_add", a_id, b_id).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, 4) };
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    rt.end_streaming();
}

#[test]
fn test_eager_inplace_on_non_contiguous_errors() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    let _lock = STREAMING_LOCK.lock().unwrap();
    rt.begin_streaming(&device);

    let (base_id, _) = rt.alloc(&device, &[4, 4], DType::Float32).unwrap();
    let view_id = rt.create_view(base_id, &[4, 4], &[1, 4], 0).unwrap();
    let (b_id, _) = rt.alloc(&device, &[4, 4], DType::Float32).unwrap();

    let result = rt.inplace_binary_op(&device, "elementwise_add", view_id, b_id);
    assert!(result.is_err());

    rt.end_streaming();
}

// ── Scalar mul tests ────────────────────────────────────────────────

#[test]
fn test_eager_scalar_mul() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 4)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    }

    let (out_id, out_ptr) = rt.scalar_mul(&device, a_id, 10.0).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 4) };
    assert_eq!(result, &[10.0, 20.0, 30.0, 40.0]);
    assert_eq!(rt.shape(out_id).unwrap(), vec![4]);

    rt.end_streaming();
}

#[test]
fn test_eager_scalar_mul_negative() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[3], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 3)
            .copy_from_slice(&[1.0, -2.0, 3.0]);
    }

    let (_, out_ptr) = rt.scalar_mul(&device, a_id, -0.5).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 3) };
    assert_eq!(result, &[-0.5, 1.0, -1.5]);

    rt.end_streaming();
}

// ── Mean all tests ──────────────────────────────────────────────────

#[test]
fn test_eager_mean_all() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 4)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    }

    let (out_id, out_ptr) = rt.mean_all(&device, a_id).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 1) };
    assert_eq!(result[0], 2.5);
    assert_eq!(rt.shape(out_id).unwrap(), vec![1]);

    rt.end_streaming();
}

#[test]
fn test_eager_mean_all_2d() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[2, 3], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 6)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    let (_, out_ptr) = rt.mean_all(&device, a_id).unwrap();
    rt.flush_and_wait(); // GPU must finish before CPU reads

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 1) };
    assert!((result[0] - 3.5).abs() < 0.001, "expected 3.5, got {}", result[0]);

    rt.end_streaming();
}

// ── Threshold backward tests ────────────────────────────────────────

#[test]
fn test_eager_threshold_backward() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (grad_id, grad_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    let (input_id, input_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(grad_ptr as *mut f32, 4)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        std::slice::from_raw_parts_mut(input_ptr as *mut f32, 4)
            .copy_from_slice(&[-1.0, 0.5, -0.5, 2.0]);
    }

    let (out_id, out_ptr) = rt.threshold_backward(&device, grad_id, input_id, 0.0).unwrap();
    rt.flush_and_wait(); // GPU must finish before CPU reads

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 4) };
    assert_eq!(result, &[0.0, 2.0, 0.0, 4.0]);
    assert_eq!(rt.shape(out_id).unwrap(), vec![4]);

    rt.end_streaming();
}

#[test]
fn test_eager_threshold_backward_nonzero_threshold() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (grad_id, grad_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    let (input_id, input_ptr) = rt.alloc(&device, &[4], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(grad_ptr as *mut f32, 4)
            .copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
        std::slice::from_raw_parts_mut(input_ptr as *mut f32, 4)
            .copy_from_slice(&[0.5, 1.5, 0.9, 2.0]);
    }

    // threshold = 1.0: only input[1]=1.5 and input[3]=2.0 pass
    let (_, out_ptr) = rt.threshold_backward(&device, grad_id, input_id, 1.0).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, 4) };
    assert_eq!(result, &[0.0, 20.0, 0.0, 40.0]);

    rt.end_streaming();
}

// ── Add scaled in-place tests ───────────────────────────────────────

#[test]
fn test_eager_add_scaled_inplace() {
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

    // a += -0.1 * b → [1-1, 2-2, 3-3, 4-4] = [0, 0, 0, 0]
    rt.add_scaled_inplace(&device, a_id, b_id, -0.1).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, 4) };
    for (i, &v) in result.iter().enumerate() {
        assert!((v - 0.0).abs() < 1e-5, "element {} = {}, expected 0.0", i, v);
    }

    rt.end_streaming();
}

#[test]
fn test_eager_add_scaled_inplace_positive_alpha() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();
    rt.begin_streaming(&device);

    let (a_id, a_ptr) = rt.alloc(&device, &[3], DType::Float32).unwrap();
    let (b_id, b_ptr) = rt.alloc(&device, &[3], DType::Float32).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(a_ptr as *mut f32, 3)
            .copy_from_slice(&[1.0, 2.0, 3.0]);
        std::slice::from_raw_parts_mut(b_ptr as *mut f32, 3)
            .copy_from_slice(&[10.0, 10.0, 10.0]);
    }

    // a += 0.5 * b → [1+5, 2+5, 3+5] = [6, 7, 8]
    rt.add_scaled_inplace(&device, a_id, b_id, 0.5).unwrap();
    rt.flush_and_wait();

    let result = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, 3) };
    assert_eq!(result, &[6.0, 7.0, 8.0]);

    rt.end_streaming();
}

// ── Matmul error tests ──────────────────────────────────────────────

#[test]
fn test_eager_matmul_non_contiguous_errors() {
    let device = Device::new().unwrap();
    let mut rt = EagerRuntime::new();

    let (base_id, _) = rt.alloc(&device, &[4, 4], DType::Float32).unwrap();
    // Create a transposed (non-contiguous) view
    let view_id = rt.create_view(base_id, &[4, 4], &[1, 4], 0).unwrap();
    let (b_id, _) = rt.alloc(&device, &[4, 4], DType::Float32).unwrap();

    let result = rt.matmul(&device, view_id, b_id);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("contiguous"), "Expected contiguity error, got: {}", err_msg);
}

// ── Debug test: chained mean reduction ──

#[test]
fn test_mean_chain_debug() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut pool = applegpu_core::pool::BufferPool::new(64 * 1024 * 1024);
    let registry = applegpu_core::compute::KernelRegistry::new();

    let queue = applegpu_core::compute::get_shared_queue(&device);
    applegpu_core::compute::begin_streaming_batch(queue);

    // [2,3] = [[1,2,3],[4,5,6]]
    let buf_in = pool.acquire(&device, 6 * 4).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(buf_in.contents() as *mut f32, 6)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // Step 1: [2,3] → [2], rows=2 cols=3
    let buf_mid = pool.acquire(&device, 2 * 4).unwrap();
    let _cb1 = registry.dispatch_mean_typed_nb(
        &device, DType::Float32, queue, &buf_in, &buf_mid, 2, 3,
    ).unwrap();
    applegpu_core::compute::streaming_tick();

    // Flush to ensure step 1 completes
    applegpu_core::compute::flush_streaming_batch();

    // Read intermediate
    let mid = unsafe { std::slice::from_raw_parts(buf_mid.contents() as *const f32, 2) };
    eprintln!("Step 1 result (expected [2.0, 5.0]): {:?}", mid);

    // Step 2: [2] → [1], rows=1 cols=2
    let buf_out = pool.acquire(&device, 1 * 4).unwrap();
    let _cb2 = registry.dispatch_mean_typed_nb(
        &device, DType::Float32, queue, &buf_mid, &buf_out, 1, 2,
    ).unwrap();
    applegpu_core::compute::streaming_tick();
    applegpu_core::compute::flush_streaming_batch();

    let result = unsafe { *(buf_out.contents() as *const f32) };
    eprintln!("Step 2 result (expected 3.5): {}", result);

    applegpu_core::compute::end_streaming_batch();

    assert!((mid[0] - 2.0).abs() < 0.01, "Step 1[0]: expected 2.0, got {}", mid[0]);
    assert!((mid[1] - 5.0).abs() < 0.01, "Step 1[1]: expected 5.0, got {}", mid[1]);
    assert!((result - 3.5).abs() < 0.01, "Step 2: expected 3.5, got {}", result);
}

#[test]
fn test_mean_chain_no_flush() {
    let _lock = STREAMING_LOCK.lock().unwrap();
    let device = Device::new().unwrap();
    let mut pool = applegpu_core::pool::BufferPool::new(64 * 1024 * 1024);
    let registry = applegpu_core::compute::KernelRegistry::new();

    let queue = applegpu_core::compute::get_shared_queue(&device);
    applegpu_core::compute::begin_streaming_batch(queue);

    let buf_in = pool.acquire(&device, 6 * 4).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(buf_in.contents() as *mut f32, 6)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // Step 1 and 2 WITHOUT flush between them
    let buf_mid = pool.acquire(&device, 2 * 4).unwrap();
    let _cb1 = registry.dispatch_mean_typed_nb(
        &device, DType::Float32, queue, &buf_in, &buf_mid, 2, 3,
    ).unwrap();

    let buf_out = pool.acquire(&device, 1 * 4).unwrap();
    let _cb2 = registry.dispatch_mean_typed_nb(
        &device, DType::Float32, queue, &buf_mid, &buf_out, 1, 2,
    ).unwrap();

    // Only flush at the end
    applegpu_core::compute::flush_streaming_batch();

    let result = unsafe { *(buf_out.contents() as *const f32) };
    eprintln!("No-flush result (expected 3.5): {}", result);

    applegpu_core::compute::end_streaming_batch();
    assert!((result - 3.5).abs() < 0.01, "Expected 3.5, got {}", result);
}
