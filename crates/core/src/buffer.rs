use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::ffi;

/// A Metal GPU buffer. Wraps an MTLBuffer via the Swift bridge.
/// Uses storageModeShared for zero-copy CPU/GPU access.
pub struct Buffer {
    handle: *mut ffi::GPUBufferHandle,
    len: usize,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Allocate an empty buffer of `size_bytes` on the GPU.
    pub fn new(device: &Device, size_bytes: usize) -> Result<Self> {
        let handle = unsafe {
            ffi::gpu_bridge_create_buffer(device.raw_handle(), size_bytes as u64)
        };
        if handle.is_null() {
            Err(GpuError::BufferAllocationFailed(size_bytes))
        } else {
            Ok(Buffer {
                handle,
                len: size_bytes,
            })
        }
    }

    /// Create a buffer initialized with data from a byte slice.
    pub fn from_bytes(device: &Device, data: &[u8]) -> Result<Self> {
        let handle = unsafe {
            ffi::gpu_bridge_create_buffer_with_data(
                device.raw_handle(),
                data.as_ptr() as *const _,
                data.len() as u64,
            )
        };
        if handle.is_null() {
            Err(GpuError::BufferAllocationFailed(data.len()))
        } else {
            Ok(Buffer {
                handle,
                len: data.len(),
            })
        }
    }

    /// Get a raw pointer to the buffer contents (shared memory).
    pub fn contents(&self) -> *mut u8 {
        unsafe { ffi::gpu_bridge_buffer_contents(self.handle) as *mut u8 }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Read buffer contents into a Vec<u8>.
    pub fn read_bytes(&self) -> Vec<u8> {
        let ptr = self.contents();
        let mut data = vec![0u8; self.len];
        unsafe { std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), self.len) };
        data
    }

    /// Read buffer contents as a slice of T (zero-copy view).
    /// WARNING: Returns the FULL physical buffer. With pooled buffers, this may
    /// be larger than logical tensor data. Use Tensor::as_f32_slice() instead.
    /// # Safety
    /// Caller must ensure the buffer contains valid data of type T
    /// and that the buffer length is a multiple of size_of::<T>().
    pub unsafe fn as_slice<T: Copy>(&self) -> &[T] {
        let count = self.len / std::mem::size_of::<T>();
        std::slice::from_raw_parts(self.contents() as *const T, count)
    }

    /// Zero all bytes in the buffer (shared memory, so this is a CPU memset).
    pub fn zero(&self) -> Result<()> {
        let ptr = self.contents();
        if ptr.is_null() {
            return Err(GpuError::BufferAllocationFailed(0));
        }
        unsafe { std::ptr::write_bytes(ptr, 0, self.len) };
        Ok(())
    }

    /// Get the raw FFI handle. Used internally by compute module.
    pub(crate) fn raw_handle(&self) -> *mut ffi::GPUBufferHandle {
        self.handle
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { ffi::gpu_bridge_destroy_buffer(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_alloc_and_length() {
        let device = match Device::new() {
            Ok(d) => d,
            Err(_) => return, // No GPU
        };
        let buf = Buffer::new(&device, 1024).unwrap();
        assert_eq!(buf.len(), 1024);
    }

    #[test]
    fn buffer_from_bytes_roundtrip() {
        let device = match Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        let buf = Buffer::from_bytes(&device, bytes).unwrap();
        assert_eq!(buf.len(), 16);

        let result = unsafe { buf.as_slice::<f32>() };
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn buffer_write_via_contents() {
        let device = match Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let buf = Buffer::new(&device, 4 * std::mem::size_of::<f32>()).unwrap();
        let ptr = buf.contents() as *mut f32;
        unsafe {
            *ptr.add(0) = 10.0;
            *ptr.add(1) = 20.0;
            *ptr.add(2) = 30.0;
            *ptr.add(3) = 40.0;
        }
        let result = unsafe { buf.as_slice::<f32>() };
        assert_eq!(result, &[10.0, 20.0, 30.0, 40.0]);
    }
}
