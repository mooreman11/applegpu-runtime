use std::ffi::CString;

use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::ffi;

/// Metal Shading Language source for element-wise add.
const ADD_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        out[id] = a[id] + b[id];
    }
}
"#;

/// A Metal compute pipeline. Wraps command queue + pipeline state.
pub struct ComputePipeline {
    handle: *mut ffi::GPUComputeHandle,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    /// Create a compute pipeline from MSL source and function name.
    pub fn new(device: &Device, kernel_source: &str, function_name: &str) -> Result<Self> {
        let source = CString::new(kernel_source).map_err(|_| {
            GpuError::ComputeFailed("Invalid kernel source (null byte)".to_string())
        })?;
        let name = CString::new(function_name).map_err(|_| {
            GpuError::ComputeFailed("Invalid function name (null byte)".to_string())
        })?;

        let handle = unsafe {
            ffi::gpu_bridge_create_compute(
                device.raw_handle(),
                source.as_ptr(),
                name.as_ptr(),
            )
        };

        if handle.is_null() {
            Err(GpuError::ComputeFailed(format!(
                "Failed to create compute pipeline for '{}'",
                function_name
            )))
        } else {
            Ok(ComputePipeline { handle })
        }
    }

    /// Create a compute pipeline for element-wise add.
    pub fn add(device: &Device) -> Result<Self> {
        Self::new(device, ADD_KERNEL_SOURCE, "elementwise_add")
    }

    /// Dispatch element-wise operation: out = op(a, b).
    pub fn dispatch_elementwise(
        &self,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let result = unsafe {
            ffi::gpu_bridge_compute_elementwise(
                self.handle,
                buf_a.raw_handle() as *const _,
                buf_b.raw_handle() as *const _,
                buf_out.raw_handle(),
                element_count as u64,
            )
        };
        if result == 0 {
            Ok(())
        } else {
            Err(GpuError::ComputeFailed(
                "Kernel dispatch failed".to_string(),
            ))
        }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { ffi::gpu_bridge_destroy_compute(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_pipeline_creates() {
        let device = match Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let pipeline = ComputePipeline::add(&device);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn elementwise_add_computes_correctly() {
        let device = match Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let pipeline = ComputePipeline::add(&device).unwrap();

        let a_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let b_data: [f32; 4] = [10.0, 20.0, 30.0, 40.0];

        let bytes_a = unsafe {
            std::slice::from_raw_parts(a_data.as_ptr() as *const u8, 16)
        };
        let bytes_b = unsafe {
            std::slice::from_raw_parts(b_data.as_ptr() as *const u8, 16)
        };

        let buf_a = Buffer::from_bytes(&device, bytes_a).unwrap();
        let buf_b = Buffer::from_bytes(&device, bytes_b).unwrap();
        let buf_out = Buffer::new(&device, 16).unwrap();

        pipeline.dispatch_elementwise(&buf_a, &buf_b, &buf_out, 4).unwrap();

        let result = unsafe { buf_out.as_slice::<f32>() };
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }
}
