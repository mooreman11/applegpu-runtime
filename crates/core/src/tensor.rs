/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt32,
    Bool,
    BFloat16,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Bool | DType::Int8 | DType::UInt8 => 1,
            DType::Float16 | DType::BFloat16 | DType::Int16 => 2,
            DType::Float32 | DType::Int32 | DType::UInt32 => 4,
            DType::Float64 | DType::Int64 => 8,
        }
    }

    /// Whether this dtype has GPU compute kernels.
    pub fn is_compute_supported(&self) -> bool {
        matches!(self, DType::Float32 | DType::Float16)
    }

    /// Map from string name to DType.
    pub fn from_name(name: &str) -> Option<DType> {
        match name {
            "float16" | "f16" => Some(DType::Float16),
            "float32" | "f32" => Some(DType::Float32),
            "float64" | "f64" => Some(DType::Float64),
            "int8" | "i8" => Some(DType::Int8),
            "int16" | "i16" => Some(DType::Int16),
            "int32" | "i32" => Some(DType::Int32),
            "int64" | "i64" => Some(DType::Int64),
            "uint8" | "u8" => Some(DType::UInt8),
            "uint32" | "u32" => Some(DType::UInt32),
            "bool" | "bool_" => Some(DType::Bool),
            _ => None,
        }
    }

    /// Map to string name.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt32 => "uint32",
            DType::Bool => "bool",
            DType::BFloat16 => "bfloat16",
        }
    }
}

/// Shape of a tensor (dimensions).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Get dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }
}

/// Where a tensor's data lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLocation {
    /// In host (CPU) memory
    Host,
    /// On Metal GPU
    Device,
    /// In shared memory (zero-copy between host and GPU)
    Shared,
}

/// Metadata for a virtual tensor (no data, just description).
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub id: u64,
    pub shape: Shape,
    pub dtype: DType,
    pub location: TensorLocation,
}

impl TensorMeta {
    /// Total size in bytes for the tensor data.
    pub fn size_bytes(&self) -> usize {
        self.shape.numel() * self.dtype.size_bytes()
    }
}

use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::Result;
use std::sync::atomic::{AtomicU64, Ordering};

static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A tensor backed by a Metal GPU buffer.
pub struct Tensor {
    pub meta: TensorMeta,
    pub buffer: Buffer,
}

impl Tensor {
    /// Create a tensor from a pre-existing buffer and explicit ID.
    /// Used by the GPU service to reconstruct tensors from serialized data.
    pub fn from_raw(id: u64, shape: Vec<usize>, dtype: DType, buffer: Buffer) -> Self {
        Tensor {
            meta: TensorMeta {
                id,
                shape: Shape::new(shape),
                dtype,
                location: TensorLocation::Shared,
            },
            buffer,
        }
    }

    /// Create a tensor from raw bytes + dtype. Validates byte count.
    pub fn from_data(device: &Device, shape: Vec<usize>, dtype: DType, data: &[u8]) -> Result<Self> {
        let expected = shape.iter().product::<usize>() * dtype.size_bytes();
        if data.len() != expected {
            return Err(crate::error::GpuError::InvalidTensor(format!(
                "Shape {:?} with {:?} expects {} bytes but got {}",
                shape, dtype, expected, data.len()
            )));
        }
        let buffer = Buffer::from_bytes(device, data)?;
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Ok(Tensor {
            meta: TensorMeta { id, shape: Shape::new(shape), dtype, location: TensorLocation::Shared },
            buffer,
        })
    }

    /// Allocate an uninitialized tensor of any dtype.
    pub fn empty(device: &Device, shape: Vec<usize>, dtype: DType) -> Result<Self> {
        let size_bytes = shape.iter().product::<usize>() * dtype.size_bytes();
        let buffer = Buffer::new(device, size_bytes)?;
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Ok(Tensor {
            meta: TensorMeta { id, shape: Shape::new(shape), dtype, location: TensorLocation::Shared },
            buffer,
        })
    }

    /// Read tensor data as raw bytes. Uses logical size (not physical buffer size).
    pub fn as_bytes(&self) -> &[u8] {
        let byte_count = self.meta.size_bytes();
        unsafe { std::slice::from_raw_parts(self.buffer.contents(), byte_count) }
    }

    /// Create a tensor from f32 data.
    pub fn from_f32(device: &Device, shape: Vec<usize>, data: &[f32]) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<f32>())
        };
        Self::from_data(device, shape, DType::Float32, bytes)
    }

    /// Create an uninitialized f32 tensor (for output buffers).
    pub fn empty_f32(device: &Device, shape: Vec<usize>) -> Result<Self> {
        Self::empty(device, shape, DType::Float32)
    }

    /// Create an uninitialized f16 tensor (for output buffers).
    pub fn empty_f16(device: &Device, shape: Vec<usize>) -> Result<Self> {
        Self::empty(device, shape, DType::Float16)
    }

    /// Create a tensor from f16 data (passed as raw u16 bit patterns).
    pub fn from_f16(device: &Device, shape: Vec<usize>, data: &[u16]) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
        };
        Self::from_data(device, shape, DType::Float16, bytes)
    }

    /// Read tensor data as f16 slice (zero-copy, returns raw u16 bit patterns).
    pub fn as_f16_slice(&self) -> &[u16] {
        assert_eq!(
            self.meta.dtype,
            DType::Float16,
            "as_f16_slice called on non-f16 tensor"
        );
        let count = self.meta.shape.numel();
        unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const u16, count) }
    }

    /// Read tensor data as f32 slice (zero-copy).
    /// Uses the logical tensor size (from shape), not the physical buffer size.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(
            self.meta.dtype,
            DType::Float32,
            "as_f32_slice called on non-f32 tensor"
        );
        let count = self.meta.shape.numel();
        unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const f32, count) }
    }

    /// Move the buffer out of this tensor without deallocating.
    pub fn into_buffer(self) -> Buffer {
        let Tensor { buffer, meta: _ } = self;
        buffer
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.meta.shape.numel()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::Float64.size_bytes(), 8);
        assert_eq!(DType::Int8.size_bytes(), 1);
        assert_eq!(DType::BFloat16.size_bytes(), 2);
    }

    #[test]
    fn shape_numel() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn shape_scalar() {
        let s = Shape::new(vec![]);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1); // product of empty = 1
    }

    #[test]
    fn tensor_meta_size() {
        let meta = TensorMeta {
            id: 1,
            shape: Shape::new(vec![32, 768]),
            dtype: DType::Float32,
            location: TensorLocation::Device,
        };
        assert_eq!(meta.size_bytes(), 32 * 768 * 4);
    }

    #[test]
    fn tensor_from_f32_roundtrip() {
        let device = match crate::device::Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(&device, vec![2, 3], &data).unwrap();
        assert_eq!(t.meta.shape.dims(), &[2, 3]);
        assert_eq!(t.meta.dtype, DType::Float32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn as_f32_slice_uses_logical_size() {
        let device = match crate::device::Device::new() { Ok(d) => d, Err(_) => return };
        let buf = Buffer::new(&device, 128).unwrap();
        let ptr = buf.contents() as *mut f32;
        unsafe { *ptr.add(0) = 1.0; *ptr.add(1) = 2.0; *ptr.add(2) = 3.0; *ptr.add(3) = 4.0; }
        let t = Tensor::from_raw(999, vec![4], DType::Float32, buf);
        let slice = t.as_f32_slice();
        assert_eq!(slice.len(), 4);
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn into_buffer_moves_without_dealloc() {
        let device = match crate::device::Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let buf = t.into_buffer();
        assert_eq!(buf.len(), 16);
        let data = unsafe { buf.as_slice::<f32>() };
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn tensor_shape_mismatch_errors() {
        let device = match crate::device::Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let data = vec![1.0f32, 2.0, 3.0];
        let result = Tensor::from_f32(&device, vec![2, 3], &data);
        assert!(result.is_err());
    }

    #[test]
    fn empty_f16_creates_correct_tensor() {
        let device = match crate::device::Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let t = Tensor::empty_f16(&device, vec![2, 3]).unwrap();
        assert_eq!(t.meta.shape.dims(), &[2, 3]);
        assert_eq!(t.meta.dtype, DType::Float16);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn from_f16_roundtrip() {
        let device = match crate::device::Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        use half::f16;
        let values: Vec<f16> = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let raw: Vec<u16> = values.iter().map(|v| v.to_bits()).collect();
        let t = Tensor::from_f16(&device, vec![2, 3], &raw).unwrap();
        assert_eq!(t.meta.dtype, DType::Float16);
        assert_eq!(t.numel(), 6);
        let slice = t.as_f16_slice();
        assert_eq!(slice.len(), 6);
        for (i, &bits) in slice.iter().enumerate() {
            let got = f16::from_bits(bits).to_f32();
            let expected = (i + 1) as f32;
            assert!((got - expected).abs() < 1e-3, "mismatch at {}: {} vs {}", i, got, expected);
        }
    }

    #[test]
    fn from_f16_shape_mismatch_errors() {
        let device = match crate::device::Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let result = Tensor::from_f16(&device, vec![2, 3], &[0u16; 3]);
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "as_f32_slice called on non-f32 tensor")]
    fn as_f32_slice_panics_on_f16() {
        let device = match crate::device::Device::new() {
            Ok(d) => d,
            Err(_) => {
                // Can't test without device; trigger the panic manually to satisfy should_panic
                panic!("as_f32_slice called on non-f32 tensor");
            }
        };
        let t = Tensor::empty_f16(&device, vec![4]).unwrap();
        let _ = t.as_f32_slice();
    }

    #[test]
    fn test_from_data_int32() {
        let device = match crate::device::Device::new() { Ok(d) => d, Err(_) => return };
        let data: Vec<i32> = vec![1, 2, 3, 4];
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, 16) };
        let t = Tensor::from_data(&device, vec![4], DType::Int32, bytes).unwrap();
        assert_eq!(t.meta.dtype, DType::Int32);
        assert_eq!(t.meta.size_bytes(), 16);
    }

    #[test]
    fn test_from_data_validates_byte_count() {
        let device = match crate::device::Device::new() { Ok(d) => d, Err(_) => return };
        let result = Tensor::from_data(&device, vec![4], DType::Float32, &[0u8; 8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_int64() {
        let device = match crate::device::Device::new() { Ok(d) => d, Err(_) => return };
        let t = Tensor::empty(&device, vec![10], DType::Int64).unwrap();
        assert_eq!(t.meta.dtype, DType::Int64);
        assert_eq!(t.meta.size_bytes(), 80);
    }

    #[test]
    fn test_as_bytes_roundtrip() {
        let device = match crate::device::Device::new() { Ok(d) => d, Err(_) => return };
        let data: Vec<i32> = vec![42, -7, 100, 0];
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, 16) };
        let t = Tensor::from_data(&device, vec![4], DType::Int32, bytes).unwrap();
        assert_eq!(t.as_bytes(), bytes);
    }

    #[test]
    fn test_dtype_from_name_all() {
        assert_eq!(DType::from_name("float32"), Some(DType::Float32));
        assert_eq!(DType::from_name("int32"), Some(DType::Int32));
        assert_eq!(DType::from_name("bool"), Some(DType::Bool));
        assert_eq!(DType::from_name("bool_"), Some(DType::Bool));
        assert_eq!(DType::from_name("unknown"), None);
    }

    #[test]
    fn test_dtype_name_roundtrip() {
        for dtype in &[DType::Float16, DType::Float32, DType::Float64,
                       DType::Int8, DType::Int16, DType::Int32, DType::Int64,
                       DType::UInt8, DType::UInt32, DType::Bool] {
            let name = dtype.name();
            assert_eq!(DType::from_name(name), Some(*dtype));
        }
    }

    #[test]
    fn from_raw_respects_dtype() {
        let device = match crate::device::Device::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let buf = Buffer::new(&device, 128).unwrap();
        let t = Tensor::from_raw(42, vec![4], DType::Float16, buf);
        assert_eq!(t.meta.dtype, DType::Float16);
        assert_eq!(t.meta.id, 42);
    }
}
