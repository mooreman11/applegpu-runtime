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
