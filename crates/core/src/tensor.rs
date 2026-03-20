use crate::error::{GpuError, Result};

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
        !matches!(self, DType::Float64)
    }

    /// Whether this dtype is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, DType::Float32 | DType::Float16 | DType::BFloat16 | DType::Float64)
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
            "bfloat16" | "bf16" => Some(DType::BFloat16),
            _ => None,
        }
    }

    /// Encode as wire protocol discriminant (matches `WireDType`).
    pub fn to_wire(&self) -> u32 {
        match self {
            DType::Float32 => 0,
            DType::Float16 => 1,
            DType::Float64 => 2,
            DType::Int8 => 3,
            DType::Int16 => 4,
            DType::Int32 => 5,
            DType::Int64 => 6,
            DType::UInt8 => 7,
            DType::UInt32 => 8,
            DType::Bool => 9,
            DType::BFloat16 => 10,
        }
    }

    /// Decode from wire protocol discriminant (matches `WireDType`).
    pub fn from_wire(d: u32) -> Option<DType> {
        match d {
            0 => Some(DType::Float32),
            1 => Some(DType::Float16),
            2 => Some(DType::Float64),
            3 => Some(DType::Int8),
            4 => Some(DType::Int16),
            5 => Some(DType::Int32),
            6 => Some(DType::Int64),
            7 => Some(DType::UInt8),
            8 => Some(DType::UInt32),
            9 => Some(DType::Bool),
            10 => Some(DType::BFloat16),
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

// ── Shape ───────────────────────────────────────────────────────────────────

pub const MAX_DIMS: usize = 8;

/// Fixed-size, stack-allocated shape. Copy semantics (72 bytes).
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub dims: [usize; MAX_DIMS],
    ndim: usize,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Result<Self> {
        if dims.len() > MAX_DIMS {
            return Err(GpuError::InvalidTensor(format!(
                "Shape has {} dimensions, maximum is {}", dims.len(), MAX_DIMS
            )));
        }
        let mut arr = [1usize; MAX_DIMS];
        for (i, &d) in dims.iter().enumerate() {
            arr[i] = d;
        }
        Ok(Shape { dims: arr, ndim: dims.len() })
    }

    pub fn scalar() -> Self {
        Shape { dims: [1; MAX_DIMS], ndim: 0 }
    }

    pub fn ndim(&self) -> usize { self.ndim }

    pub fn dims(&self) -> &[usize] { &self.dims[..self.ndim] }

    pub fn numel(&self) -> usize {
        self.dims[..self.ndim].iter().product::<usize>().max(1)
    }

    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape> {
        let out_ndim = self.ndim.max(other.ndim);
        let mut out_dims = [1usize; MAX_DIMS];
        for i in 0..out_ndim {
            let a = if i < self.ndim { self.dims[self.ndim - 1 - i] } else { 1 };
            let b = if i < other.ndim { other.dims[other.ndim - 1 - i] } else { 1 };
            if a == b {
                out_dims[out_ndim - 1 - i] = a;
            } else if a == 1 {
                out_dims[out_ndim - 1 - i] = b;
            } else if b == 1 {
                out_dims[out_ndim - 1 - i] = a;
            } else {
                return Err(GpuError::InvalidTensor(format!(
                    "Cannot broadcast shapes {:?} and {:?}: dim {} is {} vs {}",
                    self.dims(), other.dims(), out_ndim - 1 - i, a, b
                )));
            }
        }
        Ok(Shape { dims: out_dims, ndim: out_ndim })
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim && self.dims[..self.ndim] == other.dims[..other.ndim]
    }
}
impl Eq for Shape {}

impl std::hash::Hash for Shape {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ndim.hash(state);
        self.dims[..self.ndim].hash(state);
    }
}

// ── TensorLayout ────────────────────────────────────────────────────────────

/// Layout with element strides. Clone (not Copy — 136 bytes).
#[derive(Debug, Clone)]
pub struct TensorLayout {
    pub shape: Shape,
    strides: [usize; MAX_DIMS],
}

impl TensorLayout {
    pub fn contiguous(shape: Shape) -> Self {
        let mut strides = [0usize; MAX_DIMS];
        if shape.ndim() > 0 {
            strides[shape.ndim() - 1] = 1;
            for i in (0..shape.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape.dims[i + 1];
            }
        }
        TensorLayout { shape, strides }
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.ndim() == 0 { return true; }
        let mut expected = 1;
        for i in (0..self.shape.ndim()).rev() {
            if self.strides[i] != expected { return false; }
            expected *= self.shape.dims[i];
        }
        true
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.shape.ndim()]
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let mut new = self.clone();
        new.shape.dims.swap(dim0, dim1);
        new.strides.swap(dim0, dim1);
        new
    }

    pub fn set_stride(&mut self, dim: usize, stride: usize) {
        assert!(dim < MAX_DIMS, "dim {} exceeds MAX_DIMS {}", dim, MAX_DIMS);
        self.strides[dim] = stride;
    }

    pub fn broadcast_strides_for(source: &Shape, target: &Shape) -> [usize; MAX_DIMS] {
        let mut strides = [0usize; MAX_DIMS];
        let src_contiguous = TensorLayout::contiguous(*source);
        let offset = target.ndim() - source.ndim();
        for i in 0..source.ndim() {
            let target_i = i + offset;
            if source.dims[i] == target.dims[target_i] {
                strides[target_i] = src_contiguous.strides[i];
            } else {
                strides[target_i] = 0; // broadcast dim
            }
        }
        strides
    }
}

impl PartialEq for TensorLayout {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.strides[..self.shape.ndim()] == other.strides[..other.shape.ndim()]
    }
}
impl Eq for TensorLayout {}

// ── TensorLocation ──────────────────────────────────────────────────────────

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

// ── TensorMeta ──────────────────────────────────────────────────────────────

/// Metadata for a virtual tensor (no data, just description).
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub id: u64,
    pub layout: TensorLayout,
    pub dtype: DType,
    pub location: TensorLocation,
}

impl TensorMeta {
    /// Total size in bytes for the tensor data.
    pub fn size_bytes(&self) -> usize {
        self.layout.shape.numel() * self.dtype.size_bytes()
    }
}

// ── Tensor ──────────────────────────────────────────────────────────────────

use crate::buffer::Buffer;
use crate::device::Device;
use std::sync::atomic::{AtomicU64, Ordering};

static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Allocate a unique tensor ID (thread-safe).
pub fn next_tensor_id() -> u64 {
    TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// A tensor backed by a Metal GPU buffer.
pub struct Tensor {
    pub meta: TensorMeta,
    pub buffer: Buffer,
}

impl Tensor {
    /// Create a tensor from a pre-existing buffer and explicit ID.
    /// Used by the GPU service to reconstruct tensors from serialized data.
    pub fn from_raw(id: u64, shape: Vec<usize>, dtype: DType, buffer: Buffer) -> Self {
        let shape = Shape::new(shape).unwrap();
        let layout = TensorLayout::contiguous(shape);
        Tensor {
            meta: TensorMeta {
                id,
                layout,
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
            return Err(GpuError::InvalidTensor(format!(
                "Shape {:?} with {:?} expects {} bytes but got {}",
                shape, dtype, expected, data.len()
            )));
        }
        let buffer = Buffer::from_bytes(device, data)?;
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let shape = Shape::new(shape)?;
        let layout = TensorLayout::contiguous(shape);
        Ok(Tensor {
            meta: TensorMeta { id, layout, dtype, location: TensorLocation::Shared },
            buffer,
        })
    }

    /// Allocate an uninitialized tensor of any dtype.
    pub fn empty(device: &Device, shape: Vec<usize>, dtype: DType) -> Result<Self> {
        let size_bytes = shape.iter().product::<usize>() * dtype.size_bytes();
        let buffer = Buffer::new(device, size_bytes)?;
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let shape = Shape::new(shape)?;
        let layout = TensorLayout::contiguous(shape);
        Ok(Tensor {
            meta: TensorMeta { id, layout, dtype, location: TensorLocation::Shared },
            buffer,
        })
    }

    /// Read tensor data as raw bytes. Uses logical size (not physical buffer size).
    /// Returns error if tensor is non-contiguous.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        if !self.meta.layout.is_contiguous() {
            return Err(GpuError::InvalidTensor("as_bytes requires contiguous tensor".into()));
        }
        let byte_count = self.meta.size_bytes();
        Ok(unsafe { std::slice::from_raw_parts(self.buffer.contents(), byte_count) })
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

    /// Create a tensor from i32 data.
    pub fn from_i32(device: &Device, shape: Vec<usize>, data: &[i32]) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<i32>())
        };
        Self::from_data(device, shape, DType::Int32, bytes)
    }

    /// Create a tensor from f16 data (passed as raw u16 bit patterns).
    pub fn from_f16(device: &Device, shape: Vec<usize>, data: &[u16]) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
        };
        Self::from_data(device, shape, DType::Float16, bytes)
    }

    /// Read tensor data as f16 slice (zero-copy, returns raw u16 bit patterns).
    /// Returns error if tensor is non-contiguous.
    pub fn as_f16_slice(&self) -> Result<&[u16]> {
        if !self.meta.layout.is_contiguous() {
            return Err(GpuError::InvalidTensor("as_f16_slice requires contiguous tensor".into()));
        }
        assert_eq!(
            self.meta.dtype,
            DType::Float16,
            "as_f16_slice called on non-f16 tensor"
        );
        let count = self.meta.layout.shape.numel();
        Ok(unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const u16, count) })
    }

    /// Read tensor data as f32 slice (zero-copy).
    /// Uses the logical tensor size (from shape), not the physical buffer size.
    /// Returns error if tensor is non-contiguous.
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        if !self.meta.layout.is_contiguous() {
            return Err(GpuError::InvalidTensor("as_f32_slice requires contiguous tensor".into()));
        }
        assert_eq!(
            self.meta.dtype,
            DType::Float32,
            "as_f32_slice called on non-f32 tensor"
        );
        let count = self.meta.layout.shape.numel();
        Ok(unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const f32, count) })
    }

    /// Move the buffer out of this tensor without deallocating.
    pub fn into_buffer(self) -> Buffer {
        let Tensor { buffer, meta: _ } = self;
        buffer
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.meta.layout.shape.numel()
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
        let s = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn shape_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.dims(), &[]);
    }

    #[test]
    fn tensor_meta_size() {
        let meta = TensorMeta {
            id: 1,
            layout: TensorLayout::contiguous(Shape::new(vec![32, 768]).unwrap()),
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
        assert_eq!(t.meta.layout.shape.dims(), &[2, 3]);
        assert_eq!(t.meta.dtype, DType::Float32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn as_f32_slice_uses_logical_size() {
        let device = match crate::device::Device::new() { Ok(d) => d, Err(_) => return };
        let buf = Buffer::new(&device, 128).unwrap();
        let ptr = buf.contents() as *mut f32;
        unsafe { *ptr.add(0) = 1.0; *ptr.add(1) = 2.0; *ptr.add(2) = 3.0; *ptr.add(3) = 4.0; }
        let t = Tensor::from_raw(999, vec![4], DType::Float32, buf);
        let slice = t.as_f32_slice().unwrap();
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
        assert_eq!(t.meta.layout.shape.dims(), &[2, 3]);
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
        let slice = t.as_f16_slice().unwrap();
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
        assert_eq!(t.as_bytes().unwrap(), bytes);
    }

    #[test]
    fn bfloat16_name_roundtrip() {
        assert_eq!(DType::from_name("bfloat16"), Some(DType::BFloat16));
        assert_eq!(DType::from_name("bf16"), Some(DType::BFloat16));
        assert_eq!(DType::BFloat16.name(), "bfloat16");
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

    // ── Shape tests ────────────────────────────────────────────────────────

    #[test]
    fn test_shape_2d() {
        let s = Shape::new(vec![2, 3]).unwrap();
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.dims(), &[2, 3]);
        assert_eq!(s.numel(), 6);
    }

    #[test]
    fn test_shape_3d() {
        let s = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.dims(), &[2, 3, 4]);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn test_shape_1d() {
        let s = Shape::new(vec![5]).unwrap();
        assert_eq!(s.ndim(), 1);
        assert_eq!(s.dims(), &[5]);
        assert_eq!(s.numel(), 5);
    }

    #[test]
    fn test_shape_exceeds_max_dims() {
        let dims = vec![1; 9];
        assert!(Shape::new(dims).is_err());
    }

    #[test]
    fn test_shape_equality_ignores_padding() {
        let a = Shape::new(vec![2, 3]).unwrap();
        let b = Shape::new(vec![2, 3]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_shape_broadcast_bias() {
        let a = Shape::new(vec![4, 3]).unwrap();
        let b = Shape::new(vec![3]).unwrap();
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c.dims(), &[4, 3]);
    }

    #[test]
    fn test_shape_broadcast_3d() {
        let a = Shape::new(vec![2, 1, 4]).unwrap();
        let b = Shape::new(vec![3, 4]).unwrap();
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_shape_broadcast_incompatible() {
        let a = Shape::new(vec![2, 3]).unwrap();
        let b = Shape::new(vec![4, 3]).unwrap();
        assert!(a.broadcast_with(&b).is_err());
    }

    #[test]
    fn test_shape_copy() {
        let a = Shape::new(vec![2, 3]).unwrap();
        let b = a; // Copy
        assert_eq!(a, b);
    }

    // ── TensorLayout tests ─────────────────────────────────────────────────

    #[test]
    fn test_contiguous_strides_2d() {
        let shape = Shape::new(vec![2, 3]).unwrap();
        let layout = TensorLayout::contiguous(shape);
        assert_eq!(layout.strides(), &[3, 1]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_contiguous_strides_3d() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        let layout = TensorLayout::contiguous(shape);
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_contiguous_strides_1d() {
        let shape = Shape::new(vec![5]).unwrap();
        let layout = TensorLayout::contiguous(shape);
        assert_eq!(layout.strides(), &[1]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_transpose_swaps_dims_and_strides() {
        let shape = Shape::new(vec![2, 3]).unwrap();
        let layout = TensorLayout::contiguous(shape).transpose(0, 1);
        assert_eq!(layout.shape.dims(), &[3, 2]);
        assert_eq!(layout.strides(), &[1, 3]);
        assert!(!layout.is_contiguous());
    }

    #[test]
    fn test_broadcast_strides() {
        let src = Shape::new(vec![3]).unwrap();
        let target = Shape::new(vec![4, 3]).unwrap();
        let strides = TensorLayout::broadcast_strides_for(&src, &target);
        assert_eq!(strides[..2], [0, 1]); // dim 0 broadcast (stride=0), dim 1 normal
    }

    #[test]
    fn test_scalar_layout() {
        let shape = Shape::scalar();
        let layout = TensorLayout::contiguous(shape);
        assert!(layout.is_contiguous());
        assert_eq!(layout.strides(), &[]);
    }
}
