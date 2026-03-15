use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::ops;
use applegpu_core::tensor::Tensor;

fn get_device() -> Option<Device> {
    Device::new().ok()
}

// ── Batched Matmul Tests ────────────────────────────────────────────────────

#[test]
fn test_batched_matmul_3d() {
    // [2, 3, 4] @ [2, 4, 5] -> [2, 3, 5]
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    // Batch 0: 3x4 identity-like * 4x5 sequential
    // Batch 1: different values
    #[rustfmt::skip]
    let a_data: Vec<f32> = vec![
        // batch 0: 3x4
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        // batch 1: 3x4
        2.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 0.0,
    ];
    #[rustfmt::skip]
    let b_data: Vec<f32> = vec![
        // batch 0: 4x5
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
        // batch 1: 4x5
        1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0,
    ];

    let a = Tensor::from_f32(&device, vec![2, 3, 4], &a_data).unwrap();
    let b = Tensor::from_f32(&device, vec![2, 4, 5], &b_data).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = ops::matmul(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();

    let result = rt.read_f32(c_id).unwrap();
    let shape = rt.shape(c_id).unwrap();
    assert_eq!(shape, vec![2, 3, 5]);

    // Batch 0: identity-like * B = first 3 rows of B
    assert_eq!(&result[0..5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(&result[5..10], &[6.0, 7.0, 8.0, 9.0, 10.0]);
    assert_eq!(&result[10..15], &[11.0, 12.0, 13.0, 14.0, 15.0]);

    // Batch 1: 2*identity-like * identity-like = 2*identity (first 3 rows)
    assert_eq!(&result[15..20], &[2.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&result[20..25], &[0.0, 2.0, 0.0, 0.0, 0.0]);
    assert_eq!(&result[25..30], &[0.0, 0.0, 2.0, 0.0, 0.0]);
}

#[test]
fn test_batched_matmul_broadcast() {
    // [2, 3, 4] @ [4, 5] -> [2, 3, 5]
    // B broadcasts across batch dimension
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    #[rustfmt::skip]
    let a_data: Vec<f32> = vec![
        // batch 0: 3x4
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        // batch 1: 3x4
        2.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 0.0,
    ];
    #[rustfmt::skip]
    let b_data: Vec<f32> = vec![
        // 4x5 (shared across batches)
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
    ];

    let a = Tensor::from_f32(&device, vec![2, 3, 4], &a_data).unwrap();
    let b = Tensor::from_f32(&device, vec![4, 5], &b_data).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = ops::matmul(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();

    let result = rt.read_f32(c_id).unwrap();
    let shape = rt.shape(c_id).unwrap();
    assert_eq!(shape, vec![2, 3, 5]);

    // Batch 0: identity-like * B = first 3 rows of B
    assert_eq!(&result[0..5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(&result[5..10], &[6.0, 7.0, 8.0, 9.0, 10.0]);
    assert_eq!(&result[10..15], &[11.0, 12.0, 13.0, 14.0, 15.0]);

    // Batch 1: 2*identity-like * B = 2 * first 3 rows of B
    assert_eq!(&result[15..20], &[2.0, 4.0, 6.0, 8.0, 10.0]);
    assert_eq!(&result[20..25], &[12.0, 14.0, 16.0, 18.0, 20.0]);
    assert_eq!(&result[25..30], &[22.0, 24.0, 26.0, 28.0, 30.0]);
}

#[test]
fn test_existing_2d_matmul_still_works() {
    // [2, 3] @ [3, 4] -> [2, 4]
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = ops::matmul(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();

    let result = rt.read_f32(c_id).unwrap();
    let shape = rt.shape(c_id).unwrap();
    assert_eq!(shape, vec![2, 2]);
    // [1,2,3]@[7,8;9,10;11,12] = [58,64], [4,5,6]@... = [139,154]
    assert_eq!(result, &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_batched_matmul_4d() {
    // [2, 2, 2, 3] @ [2, 2, 3, 2] -> [2, 2, 2, 2]
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    // Simple: all ones, result should be K=3 everywhere
    let a_data: Vec<f32> = vec![1.0; 2 * 2 * 2 * 3];
    let b_data: Vec<f32> = vec![1.0; 2 * 2 * 3 * 2];

    let a = Tensor::from_f32(&device, vec![2, 2, 2, 3], &a_data).unwrap();
    let b = Tensor::from_f32(&device, vec![2, 2, 3, 2], &b_data).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a).unwrap();
    rt.insert_tensor(b).unwrap();

    let c_id = ops::matmul(&mut rt, a_id, b_id).unwrap();
    rt.eval(&device, c_id).unwrap();

    let result = rt.read_f32(c_id).unwrap();
    let shape = rt.shape(c_id).unwrap();
    assert_eq!(shape, vec![2, 2, 2, 2]);
    // Every element should be 3.0 (sum of 3 ones)
    for &v in &result {
        assert_eq!(v, 3.0);
    }
}

// ── Batched Softmax Tests ───────────────────────────────────────────────────

#[test]
fn test_batched_softmax_3d() {
    // [2, 3, 4] -> softmax over last dim (4), for each of 6 rows
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    // Use uniform values so softmax = 1/4 = 0.25 for each
    let data: Vec<f32> = vec![0.0; 2 * 3 * 4];
    let input = Tensor::from_f32(&device, vec![2, 3, 4], &data).unwrap();
    let input_id = input.meta.id;
    rt.insert_tensor(input).unwrap();

    let out_id = ops::softmax(&mut rt, input_id).unwrap();
    rt.eval(&device, out_id).unwrap();

    let result = rt.read_f32(out_id).unwrap();
    let shape = rt.shape(out_id).unwrap();
    assert_eq!(shape, vec![2, 3, 4]);
    for &v in &result {
        assert!((v - 0.25).abs() < 1e-5, "expected 0.25, got {}", v);
    }
}

#[test]
fn test_batched_softmax_1d() {
    // [5] -> softmax over 5 elements (total_rows=1)
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input = Tensor::from_f32(&device, vec![5], &data).unwrap();
    let input_id = input.meta.id;
    rt.insert_tensor(input).unwrap();

    let out_id = ops::softmax(&mut rt, input_id).unwrap();
    rt.eval(&device, out_id).unwrap();

    let result = rt.read_f32(out_id).unwrap();
    let shape = rt.shape(out_id).unwrap();
    assert_eq!(shape, vec![5]);

    // Check that it sums to 1
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Check monotonically increasing
    for i in 0..4 {
        assert!(result[i] < result[i + 1]);
    }
}

#[test]
fn test_batched_softmax_2d_still_works() {
    // [2, 3] -> softmax over last dim
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let data = [1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0];
    let input = Tensor::from_f32(&device, vec![2, 3], &data).unwrap();
    let input_id = input.meta.id;
    rt.insert_tensor(input).unwrap();

    let out_id = ops::softmax(&mut rt, input_id).unwrap();
    rt.eval(&device, out_id).unwrap();

    let result = rt.read_f32(out_id).unwrap();
    let shape = rt.shape(out_id).unwrap();
    assert_eq!(shape, vec![2, 3]);

    // Both rows should be identical softmax of [1,2,3]
    assert!((result[0] - result[3]).abs() < 1e-5);
    assert!((result[1] - result[4]).abs() < 1e-5);
    assert!((result[2] - result[5]).abs() < 1e-5);

    // Each row sums to 1
    let row0_sum: f32 = result[0..3].iter().sum();
    let row1_sum: f32 = result[3..6].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5);
    assert!((row1_sum - 1.0).abs() < 1e-5);
}

// ── Batched Softmax Causal Tests ────────────────────────────────────────────

#[test]
fn test_batched_softmax_causal_3d() {
    // [2, 3, 3] -> causal softmax, batch dim 2
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    // Both batches use same data: all zeros
    let data: Vec<f32> = vec![0.0; 2 * 3 * 3];
    let input = Tensor::from_f32(&device, vec![2, 3, 3], &data).unwrap();
    let input_id = input.meta.id;
    rt.insert_tensor(input).unwrap();

    let out_id = ops::softmax_causal(&mut rt, input_id).unwrap();
    rt.eval(&device, out_id).unwrap();

    let result = rt.read_f32(out_id).unwrap();
    let shape = rt.shape(out_id).unwrap();
    assert_eq!(shape, vec![2, 3, 3]);

    // For both batches, check causal mask pattern:
    // row 0: [1.0, 0.0, 0.0]  (only first element)
    // row 1: [0.5, 0.5, 0.0]  (first two elements)
    // row 2: [0.333, 0.333, 0.333]  (all three)
    for batch in 0..2 {
        let offset = batch * 9;
        // Row 0
        assert!((result[offset] - 1.0).abs() < 1e-5, "batch {} row 0 col 0: {}", batch, result[offset]);
        assert!((result[offset + 1]).abs() < 1e-5, "batch {} row 0 col 1: {}", batch, result[offset + 1]);
        assert!((result[offset + 2]).abs() < 1e-5, "batch {} row 0 col 2: {}", batch, result[offset + 2]);
        // Row 1
        assert!((result[offset + 3] - 0.5).abs() < 1e-5);
        assert!((result[offset + 4] - 0.5).abs() < 1e-5);
        assert!((result[offset + 5]).abs() < 1e-5);
        // Row 2
        let third = 1.0 / 3.0;
        assert!((result[offset + 6] - third).abs() < 1e-5);
        assert!((result[offset + 7] - third).abs() < 1e-5);
        assert!((result[offset + 8] - third).abs() < 1e-5);
    }
}

#[test]
fn test_batched_softmax_causal_2d_still_works() {
    // [3, 3] -> causal softmax (backward compat)
    let device = match get_device() { Some(d) => d, None => return };
    let mut rt = LazyRuntime::new();

    let data: Vec<f32> = vec![0.0; 9];
    let input = Tensor::from_f32(&device, vec![3, 3], &data).unwrap();
    let input_id = input.meta.id;
    rt.insert_tensor(input).unwrap();

    let out_id = ops::softmax_causal(&mut rt, input_id).unwrap();
    rt.eval(&device, out_id).unwrap();

    let result = rt.read_f32(out_id).unwrap();
    assert_eq!(rt.shape(out_id).unwrap(), vec![3, 3]);

    // Row 0: [1.0, 0.0, 0.0]
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1]).abs() < 1e-5);
    assert!((result[2]).abs() < 1e-5);
    // Row 1: [0.5, 0.5, 0.0]
    assert!((result[3] - 0.5).abs() < 1e-5);
    assert!((result[4] - 0.5).abs() < 1e-5);
    assert!((result[5]).abs() < 1e-5);
}
