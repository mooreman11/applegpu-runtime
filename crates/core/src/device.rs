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
