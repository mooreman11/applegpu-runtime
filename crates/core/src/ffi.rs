/// Rust-to-Swift FFI bridge.
///
/// Functions implemented in Swift via `@_cdecl`, callable from Rust as `extern "C"`.
/// Resolved at link time against libAppleGPUBridge.a.
///
/// Currently commented out so `cargo test` passes without the Swift library linked.
/// Uncomment once the Swift static lib is built and linked via build.rs.

/// Opaque handle to a GPU device from the Swift side.
#[repr(C)]
pub struct GPUDeviceHandle {
    _opaque: [u8; 0],
}

// extern "C" {
//     pub fn gpu_bridge_create_device() -> *mut GPUDeviceHandle;
//     pub fn gpu_bridge_destroy_device(device: *mut GPUDeviceHandle);
//     pub fn gpu_bridge_device_name(device: *const GPUDeviceHandle) -> *const std::ffi::c_char;
// }

#[cfg(test)]
mod tests {
    #[test]
    fn ffi_types_compile() {
        assert_eq!(std::mem::size_of::<super::GPUDeviceHandle>(), 0);
    }
}
