use applegpu_core::backend_ffi;

#[test]
fn ffi_init_succeeds() {
    let ok = backend_ffi::applegpu_ffi_init();
    assert!(ok, "FFI init should succeed on macOS with Metal");
}

#[test]
fn ffi_alloc_free_roundtrip() {
    backend_ffi::applegpu_ffi_init();
    let mut tensor_id: u64 = 0;
    let ptr = backend_ffi::applegpu_ffi_alloc(
        256, // 256 bytes
        0,   // dtype: Float32
        &mut tensor_id,
    );
    assert!(!ptr.is_null(), "alloc should return non-null pointer");
    assert!(tensor_id > 0, "should assign a valid tensor_id");

    // Free should not crash
    backend_ffi::applegpu_ffi_free(tensor_id);
}

#[test]
fn ffi_alloc_returns_writable_memory() {
    backend_ffi::applegpu_ffi_init();
    let mut tensor_id: u64 = 0;
    let ptr = backend_ffi::applegpu_ffi_alloc(
        16, // 4 floats
        0,  // Float32
        &mut tensor_id,
    );
    assert!(!ptr.is_null());

    // Write data through the pointer (storageModeShared = CPU writable)
    unsafe {
        let floats = ptr as *mut f32;
        *floats.add(0) = 1.0;
        *floats.add(1) = 2.0;
        *floats.add(2) = 3.0;
        *floats.add(3) = 4.0;
    }

    backend_ffi::applegpu_ffi_free(tensor_id);
}

#[test]
fn ffi_add_eval_readback() {
    backend_ffi::applegpu_ffi_init();

    // Allocate two tensors
    let mut a_id: u64 = 0;
    let mut b_id: u64 = 0;
    let a_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut a_id);
    let b_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut b_id);

    // Write data
    unsafe {
        let a = a_ptr as *mut f32;
        let b = b_ptr as *mut f32;
        for i in 0..4 {
            *a.add(i) = (i + 1) as f32;        // [1, 2, 3, 4]
            *b.add(i) = ((i + 1) * 10) as f32; // [10, 20, 30, 40]
        }
    }

    // Register tensors with shape metadata
    let dims: [u64; 1] = [4];
    assert_eq!(backend_ffi::applegpu_ffi_register_tensor(a_id, dims.as_ptr(), 1, 0), 0);
    assert_eq!(backend_ffi::applegpu_ffi_register_tensor(b_id, dims.as_ptr(), 1, 0), 0);

    // Record add op
    let result_id = backend_ffi::applegpu_ffi_add(a_id, b_id);
    assert!(result_id > 0, "add should return valid tensor_id");

    // Eval
    let rc = backend_ffi::applegpu_ffi_eval(result_id);
    assert_eq!(rc, 0, "eval should succeed");

    // Readback
    let mut out = [0.0f32; 4];
    let rc = backend_ffi::applegpu_ffi_read_f32(result_id, out.as_mut_ptr(), 4);
    assert_eq!(rc, 0, "read_f32 should succeed");
    assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);

    backend_ffi::applegpu_ffi_free(a_id);
    backend_ffi::applegpu_ffi_free(b_id);
    backend_ffi::applegpu_ffi_free(result_id);
}

#[test]
fn ffi_matmul_eval_readback() {
    backend_ffi::applegpu_ffi_init();

    // 2x2 matmul
    let mut a_id: u64 = 0;
    let mut b_id: u64 = 0;
    let a_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut a_id);
    let b_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut b_id);

    unsafe {
        let a = a_ptr as *mut f32;
        let b = b_ptr as *mut f32;
        // A = [[1, 2], [3, 4]]
        *a.add(0) = 1.0; *a.add(1) = 2.0; *a.add(2) = 3.0; *a.add(3) = 4.0;
        // B = [[5, 6], [7, 8]]
        *b.add(0) = 5.0; *b.add(1) = 6.0; *b.add(2) = 7.0; *b.add(3) = 8.0;
    }

    let dims: [u64; 2] = [2, 2];
    assert_eq!(backend_ffi::applegpu_ffi_register_tensor(a_id, dims.as_ptr(), 2, 0), 0);
    assert_eq!(backend_ffi::applegpu_ffi_register_tensor(b_id, dims.as_ptr(), 2, 0), 0);

    let result_id = backend_ffi::applegpu_ffi_matmul(a_id, b_id);
    assert!(result_id > 0);

    assert_eq!(backend_ffi::applegpu_ffi_eval(result_id), 0);

    let mut out = [0.0f32; 4];
    assert_eq!(backend_ffi::applegpu_ffi_read_f32(result_id, out.as_mut_ptr(), 4), 0);
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);

    backend_ffi::applegpu_ffi_free(a_id);
    backend_ffi::applegpu_ffi_free(b_id);
    backend_ffi::applegpu_ffi_free(result_id);
}

#[test]
fn ffi_relu_eval_readback() {
    backend_ffi::applegpu_ffi_init();

    let mut a_id: u64 = 0;
    let a_ptr = backend_ffi::applegpu_ffi_alloc(16, 0, &mut a_id);

    unsafe {
        let a = a_ptr as *mut f32;
        *a.add(0) = -2.0; *a.add(1) = 0.0; *a.add(2) = 3.0; *a.add(3) = -1.0;
    }

    let dims: [u64; 1] = [4];
    assert_eq!(backend_ffi::applegpu_ffi_register_tensor(a_id, dims.as_ptr(), 1, 0), 0);

    let result_id = backend_ffi::applegpu_ffi_relu(a_id);
    assert!(result_id > 0);

    assert_eq!(backend_ffi::applegpu_ffi_eval(result_id), 0);

    let mut out = [0.0f32; 4];
    assert_eq!(backend_ffi::applegpu_ffi_read_f32(result_id, out.as_mut_ptr(), 4), 0);
    assert_eq!(out, [0.0, 0.0, 3.0, 0.0]);

    backend_ffi::applegpu_ffi_free(a_id);
    backend_ffi::applegpu_ffi_free(result_id);
}
