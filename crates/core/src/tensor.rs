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
}
