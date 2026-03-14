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

/// Opaque handle to a GPU buffer from the Swift side.
#[repr(C)]
pub struct GPUBufferHandle {
    _opaque: [u8; 0],
}

extern "C" {
    pub fn gpu_bridge_create_buffer(
        device: *const GPUDeviceHandle,
        size_bytes: u64,
    ) -> *mut GPUBufferHandle;

    pub fn gpu_bridge_create_buffer_with_data(
        device: *const GPUDeviceHandle,
        data: *const std::ffi::c_void,
        size_bytes: u64,
    ) -> *mut GPUBufferHandle;

    pub fn gpu_bridge_destroy_buffer(buffer: *mut GPUBufferHandle);

    pub fn gpu_bridge_buffer_contents(buffer: *const GPUBufferHandle) -> *mut std::ffi::c_void;

    pub fn gpu_bridge_buffer_length(buffer: *const GPUBufferHandle) -> u64;
}

/// Opaque handle to a compute context from the Swift side.
#[repr(C)]
pub struct GPUComputeHandle {
    _opaque: [u8; 0],
}

extern "C" {
    pub fn gpu_bridge_create_compute(
        device: *const GPUDeviceHandle,
        kernel_source: *const std::ffi::c_char,
        function_name: *const std::ffi::c_char,
    ) -> *mut GPUComputeHandle;

    pub fn gpu_bridge_destroy_compute(compute: *mut GPUComputeHandle);

    pub fn gpu_bridge_compute_elementwise(
        compute: *mut GPUComputeHandle,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        element_count: u64,
    ) -> i32;

    pub fn gpu_bridge_compute_unary(
        compute: *mut GPUComputeHandle,
        buf_input: *const GPUBufferHandle,
        buf_out: *mut GPUBufferHandle,
        element_count: u64,
    ) -> i32;

    pub fn gpu_bridge_compute_matmul(
        compute: *mut GPUComputeHandle,
        buf_a: *const GPUBufferHandle,
        buf_b: *const GPUBufferHandle,
        buf_c: *mut GPUBufferHandle,
        m: u32,
        n: u32,
        k: u32,
    ) -> i32;

    pub fn gpu_bridge_compute_fused(
        compute: *mut GPUComputeHandle,
        input_buffers: *const *const GPUBufferHandle,
        buffer_count: u32,
        output: *mut GPUBufferHandle,
        element_count: u64,
    ) -> i32;
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
