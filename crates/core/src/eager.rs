use std::collections::HashMap;
use std::sync::Arc;

use crate::buffer::Buffer;
use crate::compute::KernelRegistry;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::pool::BufferPool;
use crate::tensor::{DType, Shape, TensorLayout, next_tensor_id, MAX_DIMS};

/// A tensor in the eager runtime — owns a reference-counted Metal buffer
/// plus stride-aware layout metadata. Views share the underlying `Arc<Buffer>`.
pub struct EagerTensor {
    pub buffer: Arc<Buffer>,
    pub layout: TensorLayout,
    pub dtype: DType,
    /// Byte offset into the buffer (non-zero for views with storage offset).
    pub offset: usize,
}

impl EagerTensor {
    /// Raw pointer to this tensor's data (buffer base + byte offset).
    pub fn data_ptr(&self) -> *mut u8 {
        unsafe { self.buffer.contents().add(self.offset) }
    }

    /// True if offset is zero and strides are contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.layout.is_contiguous()
    }

    /// Total bytes for the logical tensor data (numel * element size).
    pub fn nbytes(&self) -> usize {
        self.layout.shape.numel() * self.dtype.size_bytes()
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.layout.shape.numel()
    }

    /// Shape as a Vec (heap-allocated, for FFI / Python interop).
    pub fn shape_vec(&self) -> Vec<usize> {
        self.layout.shape.dims().to_vec()
    }

    /// Strides as u32 array (for Metal kernel parameters).
    pub fn strides_u32(&self) -> [u32; MAX_DIMS] {
        let mut out = [0u32; MAX_DIMS];
        for (i, &s) in self.layout.strides().iter().enumerate() {
            out[i] = s as u32;
        }
        out
    }

    /// Shape as u32 array (for Metal kernel parameters).
    pub fn shape_u32(&self) -> [u32; MAX_DIMS] {
        let mut out = [0u32; MAX_DIMS];
        for (i, &d) in self.layout.shape.dims().iter().enumerate() {
            out[i] = d as u32;
        }
        out
    }
}

/// Eager runtime: stride-aware tensor registry with `Arc<Buffer>` for view sharing.
/// This is the data layer for eager Metal dispatch — no GPU ops yet, just tensor
/// metadata and buffer management via the pool.
pub struct EagerRuntime {
    tensors: HashMap<u64, EagerTensor>,
    pool: BufferPool,
    pub(crate) registry: KernelRegistry,
}

impl EagerRuntime {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            pool: BufferPool::new(256 * 1024 * 1024),
            registry: KernelRegistry::new(),
        }
    }

    /// Allocate a new contiguous tensor. Returns (tensor_id, raw data pointer).
    pub fn alloc(
        &mut self,
        device: &Device,
        shape: &[usize],
        dtype: DType,
    ) -> Result<(u64, *mut u8)> {
        let s = Shape::new(shape.to_vec())?;
        let layout = TensorLayout::contiguous(s);
        let nbytes = s.numel() * dtype.size_bytes();
        let alloc_bytes = if nbytes == 0 { 4 } else { nbytes };
        let buffer = Arc::new(self.pool.acquire(device, alloc_bytes)?);
        let ptr = buffer.contents();
        let id = next_tensor_id();
        self.tensors.insert(id, EagerTensor {
            buffer,
            layout,
            dtype,
            offset: 0,
        });
        Ok((id, ptr))
    }

    /// Create a view of an existing tensor with different shape/strides/offset.
    /// The view shares the same underlying `Arc<Buffer>`.
    pub fn create_view(
        &mut self,
        base_id: u64,
        shape: &[usize],
        strides: &[usize],
        offset_elements: usize,
    ) -> Result<u64> {
        let base = self.tensors.get(&base_id)
            .ok_or_else(|| GpuError::InvalidTensor(format!("tensor {} not found", base_id)))?;
        let s = Shape::new(shape.to_vec())?;
        let mut layout = TensorLayout::contiguous(s);
        for (i, &stride) in strides.iter().enumerate() {
            layout.set_stride(i, stride);
        }
        let byte_offset = base.offset + offset_elements * base.dtype.size_bytes();
        let buffer = Arc::clone(&base.buffer);
        let dtype = base.dtype;
        let id = next_tensor_id();
        self.tensors.insert(id, EagerTensor {
            buffer,
            layout,
            dtype,
            offset: byte_offset,
        });
        Ok(id)
    }

    /// Free a tensor. If this was the last reference to the underlying buffer,
    /// the buffer is returned to the pool for reuse.
    pub fn free(&mut self, id: u64) {
        if let Some(tensor) = self.tensors.remove(&id) {
            if let Some(buffer) = Arc::into_inner(tensor.buffer) {
                self.pool.release(buffer);
            }
        }
    }

    /// Look up a tensor by ID.
    pub fn get(&self, id: u64) -> Result<&EagerTensor> {
        self.tensors.get(&id)
            .ok_or_else(|| GpuError::InvalidTensor(format!("tensor {} not found", id)))
    }

    /// Mutable access to the tensor map (for ops that need to insert results).
    pub fn tensors_mut(&mut self) -> &mut HashMap<u64, EagerTensor> {
        &mut self.tensors
    }

    /// Get shape of a tensor.
    pub fn shape(&self, id: u64) -> Result<Vec<usize>> {
        Ok(self.get(id)?.shape_vec())
    }

    /// Get dtype of a tensor.
    pub fn dtype(&self, id: u64) -> Result<DType> {
        Ok(self.get(id)?.dtype)
    }

    /// Check if a tensor is contiguous.
    pub fn is_contiguous(&self, id: u64) -> Result<bool> {
        Ok(self.get(id)?.is_contiguous())
    }

    /// Number of tensors currently tracked.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Pool statistics (hits, misses, pooled bytes, bucket count).
    pub fn pool_stats(&self) -> crate::pool::PoolStats {
        self.pool.stats()
    }
}
