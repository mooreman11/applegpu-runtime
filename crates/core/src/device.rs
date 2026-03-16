use crate::error::{GpuError, Result};
use crate::ffi;

/// A Metal GPU device. Wraps the Swift-side device handle with RAII.
pub struct Device {
    handle: *mut ffi::GPUDeviceHandle,
}

// Safety: the Swift GPUDevice is thread-safe (MTLDevice is thread-safe)
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Create a new Metal GPU device.
    pub fn new() -> Result<Self> {
        ffi::create_device()
            .map(|handle| Device { handle })
            .ok_or(GpuError::DeviceNotAvailable)
    }

    /// Get the raw FFI handle. Used internally by buffer and compute modules.
    pub(crate) fn raw_handle(&self) -> *const ffi::GPUDeviceHandle {
        self.handle as *const _
    }

    /// Returns true if this device supports Apple GPU family 9+ (M3/M4),
    /// which is required for Int64 (`long`) MSL kernels.
    pub fn supports_int64(&self) -> bool {
        unsafe { ffi::gpu_bridge_supports_apple9(self.handle) }
    }

    /// Get the device name (e.g. "Apple M1 Pro").
    pub fn name(&self) -> String {
        ffi::device_name(self.handle)
            .unwrap_or_else(|| "Unknown".to_string())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        ffi::destroy_device(self.handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_lifecycle() {
        match Device::new() {
            Ok(device) => {
                let name = device.name();
                assert!(!name.is_empty());
                assert_ne!(name, "Unknown");
                // Drop cleans up automatically
            }
            Err(GpuError::DeviceNotAvailable) => {
                // No Metal GPU (CI) — pass
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn device_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Device>();
    }
}
