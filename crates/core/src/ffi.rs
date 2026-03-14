use std::ffi::CStr;

/// Opaque handle to a GPU device from the Swift side.
#[repr(C)]
pub struct GPUDeviceHandle {
    _opaque: [u8; 0],
}

extern "C" {
    pub fn gpu_bridge_create_device() -> *mut GPUDeviceHandle;
    pub fn gpu_bridge_destroy_device(device: *mut GPUDeviceHandle);
    pub fn gpu_bridge_device_name(device: *const GPUDeviceHandle) -> *const std::ffi::c_char;
}

/// Safe wrapper: create a Metal device. Returns None if no GPU available.
pub fn create_device() -> Option<*mut GPUDeviceHandle> {
    let ptr = unsafe { gpu_bridge_create_device() };
    if ptr.is_null() {
        None
    } else {
        Some(ptr)
    }
}

/// Safe wrapper: destroy a Metal device.
///
/// # Safety
/// The pointer must have been returned by `create_device()` and not yet destroyed.
pub fn destroy_device(device: *mut GPUDeviceHandle) {
    unsafe { gpu_bridge_destroy_device(device) }
}

/// Safe wrapper: get the device name as a Rust string.
///
/// # Safety
/// The pointer must be a valid device handle.
pub fn device_name(device: *const GPUDeviceHandle) -> Option<String> {
    let name_ptr = unsafe { gpu_bridge_device_name(device) };
    if name_ptr.is_null() {
        None
    } else {
        let c_str = unsafe { CStr::from_ptr(name_ptr) };
        Some(c_str.to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_query_device() {
        if let Some(device) = create_device() {
            let name = device_name(device);
            assert!(name.is_some());
            assert!(!name.unwrap().is_empty());
            destroy_device(device);
        }
        // If no Metal GPU (CI), test passes by skipping
    }
}
