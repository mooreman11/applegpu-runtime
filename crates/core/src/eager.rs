use std::collections::HashMap;
use std::sync::Arc;

use crate::buffer::Buffer;
use crate::compute::{self, ComputePipeline, KernelRegistry};
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

    // ── Streaming command buffer management ──────────────────────────

    /// Begin a streaming command buffer session. No-op if already active.
    pub fn begin_streaming(&mut self, device: &Device) {
        if !compute::streaming_is_active() {
            let queue = compute::get_shared_queue(device);
            compute::begin_streaming_batch(queue);
        }
    }

    /// Flush the current command buffer (commit + wait), then reopen a new one.
    pub fn flush_and_wait(&self) {
        if compute::streaming_is_active() {
            compute::flush_streaming_batch();
        }
    }

    /// End the streaming session (commit final CB + wait + clear state).
    pub fn end_streaming(&self) {
        if compute::streaming_is_active() {
            compute::end_streaming_batch();
        }
    }

    /// True if a streaming command buffer session is active.
    pub fn is_streaming(&self) -> bool {
        compute::streaming_is_active()
    }

    // ── Binary ops ────────────────────────────────────────────────────

    /// Encode a binary element-wise op (add, sub, mul, div) into the streaming
    /// command buffer. Returns (output_tensor_id, raw data pointer). Data is
    /// only valid after `flush_and_wait()`.
    pub fn binary_op(
        &mut self,
        device: &Device,
        base_kernel: &str,
        a_id: u64,
        b_id: u64,
    ) -> Result<(u64, *mut u8)> {
        // 1. Extract data from immutable borrows (releases borrow before mutable ops)
        let (a_shape, a_dtype, a_buf, b_shape, b_buf) = {
            let a = self.get(a_id)?;
            let b = self.get(b_id)?;
            if a.dtype != b.dtype {
                return Err(GpuError::InvalidTensor(format!(
                    "dtype mismatch: {:?} vs {:?}", a.dtype, b.dtype
                )));
            }
            (a.layout.shape, a.dtype, Arc::clone(&a.buffer),
             b.layout.shape, Arc::clone(&b.buffer))
        };

        // 2. Compute output shape (supports broadcasting)
        let out_shape = a_shape.broadcast_with(&b_shape)?;
        let out_layout = TensorLayout::contiguous(out_shape);
        let numel = out_shape.numel();
        let ndim = out_shape.ndim();

        // 3. Compute broadcast strides
        let a_bcast = TensorLayout::broadcast_strides_for(&a_shape, &out_shape);
        let b_bcast = TensorLayout::broadcast_strides_for(&b_shape, &out_shape);

        let mut a_strides = [0u32; MAX_DIMS];
        let mut b_strides = [0u32; MAX_DIMS];
        let mut shape_u32 = [0u32; MAX_DIMS];
        for i in 0..ndim {
            a_strides[i] = a_bcast[i] as u32;
            b_strides[i] = b_bcast[i] as u32;
            shape_u32[i] = out_shape.dims()[i] as u32;
        }

        // 4. Allocate output buffer (mutable borrow of pool)
        let nbytes = numel * a_dtype.size_bytes();
        let alloc_bytes = if nbytes == 0 { 4 } else { nbytes };
        let out_buffer = Arc::new(self.pool.acquire(device, alloc_bytes)?);
        let out_ptr = out_buffer.contents();

        // 5. Get pipeline and dispatch into streaming CB
        let pipeline = self.get_pipeline(device, base_kernel, a_dtype)?;
        let queue = compute::get_shared_queue(device);
        let _cb = pipeline.dispatch_binary_nd_nb(
            queue,
            &*a_buf, &a_strides,
            &*b_buf, &b_strides,
            &*out_buffer, &shape_u32,
            ndim as u32, numel as u32,
        )?;
        compute::streaming_tick();

        // 6. Register output tensor
        let out_id = next_tensor_id();
        self.tensors.insert(out_id, EagerTensor {
            buffer: out_buffer,
            layout: out_layout,
            dtype: a_dtype,
            offset: 0,
        });

        Ok((out_id, out_ptr))
    }

    // ── Unary ops ─────────────────────────────────────────────────────

    /// Encode a unary element-wise op (relu, neg, abs, tanh, sigmoid, etc.) into
    /// the streaming command buffer. Returns (output_tensor_id, raw data pointer).
    /// Data is only valid after `flush_and_wait()`.
    pub fn unary_op(
        &mut self,
        device: &Device,
        base_kernel: &str,
        input_id: u64,
    ) -> Result<(u64, *mut u8)> {
        // 1. Extract data from immutable borrow
        let (shape, dtype, in_buf, in_strides_u32) = {
            let t = self.get(input_id)?;
            (t.layout.shape, t.dtype, Arc::clone(&t.buffer), t.strides_u32())
        };

        let numel = shape.numel();
        let ndim = shape.ndim();
        let mut shape_u32 = [0u32; MAX_DIMS];
        for i in 0..ndim {
            shape_u32[i] = shape.dims()[i] as u32;
        }

        // 2. Allocate output buffer
        let nbytes = numel * dtype.size_bytes();
        let alloc_bytes = if nbytes == 0 { 4 } else { nbytes };
        let out_buffer = Arc::new(self.pool.acquire(device, alloc_bytes)?);
        let out_ptr = out_buffer.contents();

        // 3. Dispatch into streaming CB
        let pipeline = self.get_pipeline(device, base_kernel, dtype)?;
        let queue = compute::get_shared_queue(device);
        let _cb = pipeline.dispatch_unary_nd_nb(
            queue,
            &*in_buf, &in_strides_u32,
            &*out_buffer, &shape_u32,
            ndim as u32, numel as u32,
        )?;
        compute::streaming_tick();

        // 4. Register output tensor
        let out_id = next_tensor_id();
        let out_layout = TensorLayout::contiguous(shape);
        self.tensors.insert(out_id, EagerTensor {
            buffer: out_buffer,
            layout: out_layout,
            dtype,
            offset: 0,
        });

        Ok((out_id, out_ptr))
    }

    // ── Matmul ───────────────────────────────────────────────────────

    /// Encode a matmul [M,K] @ [K,N] → [M,N] (or batched) into the streaming
    /// command buffer. Inputs must be contiguous. Returns (output_tensor_id,
    /// raw data pointer). Data is only valid after `flush_and_wait()`.
    pub fn matmul(
        &mut self,
        device: &Device,
        a_id: u64,
        b_id: u64,
    ) -> Result<(u64, *mut u8)> {
        // 1. Extract data from immutable borrows
        let (a_shape, a_dtype, a_buf, a_contig, b_shape, b_buf, b_contig) = {
            let a = self.get(a_id)?;
            let b = self.get(b_id)?;
            if a.dtype != b.dtype {
                return Err(GpuError::InvalidTensor(format!(
                    "dtype mismatch: {:?} vs {:?}", a.dtype, b.dtype
                )));
            }
            (a.layout.shape, a.dtype, Arc::clone(&a.buffer), a.is_contiguous(),
             b.layout.shape, Arc::clone(&b.buffer), b.is_contiguous())
        };

        // 2. Validate contiguity (D5 will add make_contiguous)
        if !a_contig || !b_contig {
            return Err(GpuError::InvalidTensor("matmul requires contiguous inputs".into()));
        }

        // 3. Extract dimensions
        let a_dims = a_shape.dims();
        let b_dims = b_shape.dims();
        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(GpuError::InvalidTensor("matmul requires 2D+ tensors".into()));
        }

        let m = a_dims[a_dims.len() - 2];
        let k = a_dims[a_dims.len() - 1];
        let n = b_dims[b_dims.len() - 1];
        if k != b_dims[b_dims.len() - 2] {
            return Err(GpuError::InvalidTensor(format!(
                "matmul inner dim mismatch: {} vs {}", k, b_dims[b_dims.len() - 2]
            )));
        }

        let batch_size: usize = a_dims[..a_dims.len() - 2].iter().product::<usize>().max(1);

        // 4. Compute output shape
        let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 2].to_vec();
        out_dims.push(m);
        out_dims.push(n);
        let out_shape = Shape::new(out_dims)?;
        let out_layout = TensorLayout::contiguous(out_shape);

        // 5. Allocate output buffer
        let nbytes = out_shape.numel() * a_dtype.size_bytes();
        let alloc_bytes = if nbytes == 0 { 4 } else { nbytes };
        let out_buffer = Arc::new(self.pool.acquire(device, alloc_bytes)?);
        let out_ptr = out_buffer.contents();

        // 6. Dispatch into streaming CB
        let pipeline = self.get_pipeline(device, "matmul", a_dtype)?;
        let queue = compute::get_shared_queue(device);
        let _cb = pipeline.dispatch_matmul_batched_nb(
            queue,
            &*a_buf, &*b_buf, &*out_buffer,
            m, n, k,
            batch_size, m * k, k * n,
        )?;
        compute::streaming_tick();

        // 7. Register output tensor
        let out_id = next_tensor_id();
        self.tensors.insert(out_id, EagerTensor {
            buffer: out_buffer,
            layout: out_layout,
            dtype: a_dtype,
            offset: 0,
        });

        Ok((out_id, out_ptr))
    }

    // ── make_contiguous ────────────────────────────────────────────────

    /// If the tensor is already contiguous, return its id and pointer unchanged.
    /// Otherwise, flush the GPU, do a CPU-side strided copy on shared memory,
    /// and return a new contiguous tensor. (D2 will add a GPU copy kernel.)
    pub fn make_contiguous(&mut self, device: &Device, id: u64) -> Result<(u64, *mut u8)> {
        if self.is_contiguous(id)? {
            let ptr = self.get(id)?.data_ptr();
            return Ok((id, ptr));
        }

        // CPU-side strided copy — must flush first so shared memory is up to date
        self.flush_and_wait();

        let (shape, dtype, strides, src_ptr, offset) = {
            let src = self.get(id)?;
            (
                src.layout.shape,
                src.dtype,
                src.layout.strides().to_vec(),
                src.buffer.contents(),
                src.offset,
            )
        };

        let numel = shape.numel();
        let ndim = shape.ndim();
        let elem_size = dtype.size_bytes();

        let (out_id, out_ptr) = self.alloc(device, shape.dims(), dtype)?;

        for linear in 0..numel {
            let mut remaining = linear;
            let mut src_byte_offset = offset;
            for d in (0..ndim).rev() {
                let idx = remaining % shape.dims()[d];
                remaining /= shape.dims()[d];
                src_byte_offset += idx * strides[d] * elem_size;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(src_byte_offset),
                    out_ptr.add(linear * elem_size),
                    elem_size,
                );
            }
        }

        Ok((out_id, out_ptr))
    }

    // ── In-place binary ops ──────────────────────────────────────────────

    /// Dispatch a binary op where the output buffer IS self's buffer (in-place).
    /// Self must be contiguous (Safety Invariant #2). Returns error if self is
    /// non-contiguous — D2 will support automatic make_contiguous + copy-back.
    pub fn inplace_binary_op(
        &mut self,
        device: &Device,
        base_kernel: &str,
        self_id: u64,
        other_id: u64,
    ) -> Result<()> {
        // Safety Invariant #2: self must be contiguous for in-place output
        if !self.is_contiguous(self_id)? {
            return Err(GpuError::InvalidTensor(
                "in-place op on non-contiguous tensor not yet supported".into(),
            ));
        }

        // Extract data from immutable borrows
        let (self_shape, self_dtype, self_buf, self_strides, other_buf, other_shape) = {
            let s = self.get(self_id)?;
            let o = self.get(other_id)?;
            if s.dtype != o.dtype {
                return Err(GpuError::InvalidTensor(format!(
                    "dtype mismatch: {:?} vs {:?}",
                    s.dtype, o.dtype
                )));
            }
            (
                s.layout.shape,
                s.dtype,
                Arc::clone(&s.buffer),
                s.strides_u32(),
                Arc::clone(&o.buffer),
                o.layout.shape,
            )
        };

        let numel = self_shape.numel();
        let ndim = self_shape.ndim();

        let mut out_shape_u32 = [0u32; MAX_DIMS];
        for i in 0..ndim {
            out_shape_u32[i] = self_shape.dims()[i] as u32;
        }

        // Broadcast strides for other
        let other_bcast = TensorLayout::broadcast_strides_for(&other_shape, &self_shape);
        let mut other_strides = [0u32; MAX_DIMS];
        for i in 0..ndim {
            other_strides[i] = other_bcast[i] as u32;
        }

        // Dispatch: output buffer = self's buffer (in-place)
        let pipeline = self.get_pipeline(device, base_kernel, self_dtype)?;
        let queue = compute::get_shared_queue(device);
        let _cb = pipeline.dispatch_binary_nd_nb(
            queue,
            &*self_buf,
            &self_strides,
            &*other_buf,
            &other_strides,
            &*self_buf, // output IS self
            &out_shape_u32,
            ndim as u32,
            numel as u32,
        )?;
        compute::streaming_tick();

        Ok(())
    }

    // ── Scalar mul ────────────────────────────────────────────────────

    /// Multiply every element of `input_id` by a float scalar `scale`.
    /// Implemented via broadcast binary mul: input * scalar_tensor([scale]).
    pub fn scalar_mul(
        &mut self,
        device: &Device,
        input_id: u64,
        scale: f32,
    ) -> Result<(u64, *mut u8)> {
        let dtype = self.dtype(input_id)?;
        let (scalar_id, scalar_ptr) = self.alloc(device, &[1], dtype)?;
        unsafe { *(scalar_ptr as *mut f32) = scale; }
        let result = self.binary_op(device, "elementwise_mul", input_id, scalar_id);
        self.free(scalar_id);
        result
    }

    // ── Mean all ─────────────────────────────────────────────────────

    /// Full reduction to mean (scalar [1]).
    /// Chains last-dim mean reductions on GPU until scalar.
    pub fn mean_all(
        &mut self,
        device: &Device,
        input_id: u64,
    ) -> Result<(u64, *mut u8)> {
        let (shape_vec, dtype) = {
            let t = self.get(input_id)?;
            (t.shape_vec(), t.dtype)
        };
        let ndim = shape_vec.len();
        if ndim == 0 {
            return Err(GpuError::InvalidTensor("mean_all requires at least 1D tensor".into()));
        }

        // Chain mean reductions: reduce last dim at each step
        let mut current_id = input_id;
        let mut current_shape = shape_vec.clone();

        for step in 0..ndim {
            // Flush between chained reductions — the mean kernel reads the
            // previous output, so the GPU must finish writing it first.
            if step > 0 {
                self.flush_and_wait();
            }
            let cols = *current_shape.last().unwrap();
            let rows: usize = current_shape[..current_shape.len() - 1].iter().product::<usize>().max(1);

            // Output shape: drop last dim (or [1] if scalar)
            let mut out_shape = current_shape[..current_shape.len() - 1].to_vec();
            if out_shape.is_empty() { out_shape = vec![1]; }

            let out_numel: usize = out_shape.iter().product();
            let nbytes = out_numel * dtype.size_bytes();
            let alloc_bytes = if nbytes == 0 { 4 } else { nbytes };

            let cur_buf = {
                let t = self.get(current_id)?;
                Arc::clone(&t.buffer)
            };

            let out_buffer = Arc::new(self.pool.acquire(device, alloc_bytes)?);
            let out_ptr = out_buffer.contents();

            let queue = compute::get_shared_queue(device);
            let _cb = self.registry.dispatch_mean_typed_nb(
                device, dtype, queue, &*cur_buf, &*out_buffer, rows, cols,
            )?;
            compute::streaming_tick();

            let out_id = next_tensor_id();
            let out_s = Shape::new(out_shape.clone())?;
            self.tensors.insert(out_id, EagerTensor {
                buffer: out_buffer,
                layout: TensorLayout::contiguous(out_s),
                dtype,
                offset: 0,
            });

            current_id = out_id;
            current_shape = out_shape;
        }

        let ptr = self.get(current_id)?.data_ptr();
        Ok((current_id, ptr))
    }

    // ── Threshold backward ───────────────────────────────────────────

    /// threshold_backward: grad * (input > threshold). ReLU backward.
    /// GPU-native via dispatch_cnn_3d_nb.
    pub fn threshold_backward(
        &mut self,
        device: &Device,
        grad_id: u64,
        input_id: u64,
        threshold: f32,
    ) -> Result<(u64, *mut u8)> {
        let (shape, dtype, numel, grad_buf, input_buf) = {
            let g = self.get(grad_id)?;
            let i = self.get(input_id)?;
            if g.dtype != i.dtype {
                return Err(GpuError::InvalidTensor(format!(
                    "dtype mismatch: {:?} vs {:?}", g.dtype, i.dtype
                )));
            }
            (g.layout.shape, g.dtype, g.numel(),
             Arc::clone(&g.buffer), Arc::clone(&i.buffer))
        };

        let nbytes = numel * dtype.size_bytes();
        let alloc_bytes = if nbytes == 0 { 4 } else { nbytes };
        let out_buffer = Arc::new(self.pool.acquire(device, alloc_bytes)?);
        let out_ptr = out_buffer.contents();

        let (k_src, k_fn) = KernelRegistry::resolve_kernel("threshold_backward", dtype);
        let queue = compute::get_shared_queue(device);
        let _cb = self.registry.dispatch_cnn_3d_nb(
            device, &k_src, &k_fn, queue,
            &[&*grad_buf, &*input_buf], &*out_buffer,
            &[numel as u32], &[threshold],
            (numel as u32, 1, 1),
        )?;
        compute::streaming_tick();

        let out_id = next_tensor_id();
        self.tensors.insert(out_id, EagerTensor {
            buffer: out_buffer,
            layout: TensorLayout::contiguous(shape),
            dtype,
            offset: 0,
        });

        Ok((out_id, out_ptr))
    }

    // ── Sum along dimension ──────────────────────────────────────────

    /// Sum reduction along a single dimension.
    /// GPU-native via dispatch_sum_typed_nb (reduces last dim).
    /// For non-last dims, reshapes so target dim is last, sums, then reshapes back.
    pub fn sum_dim(
        &mut self,
        device: &Device,
        input_id: u64,
        dim: i64,
        keepdim: bool,
    ) -> Result<(u64, *mut u8)> {
        let (shape_vec, dtype, ndim) = {
            let t = self.get(input_id)?;
            (t.shape_vec(), t.dtype, t.layout.shape.ndim())
        };

        // Normalize negative dim
        let dim = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
        if dim >= ndim {
            return Err(GpuError::InvalidTensor(format!(
                "sum dim {} out of range for {}-D tensor", dim, ndim
            )));
        }

        // If reducing last dim, dispatch directly
        if dim == ndim - 1 {
            let cols = shape_vec[ndim - 1];
            let rows: usize = shape_vec[..ndim - 1].iter().product::<usize>().max(1);

            let mut out_shape = shape_vec[..ndim - 1].to_vec();
            if keepdim { out_shape.push(1); }
            if out_shape.is_empty() { out_shape = vec![1]; }

            let out_numel: usize = out_shape.iter().product();
            let nbytes = out_numel * dtype.size_bytes();
            let alloc_bytes = if nbytes == 0 { 4 } else { nbytes };

            let in_buf = Arc::clone(&self.get(input_id)?.buffer);
            let out_buffer = Arc::new(self.pool.acquire(device, alloc_bytes)?);
            let out_ptr = out_buffer.contents();

            let queue = compute::get_shared_queue(device);
            let _cb = self.registry.dispatch_sum_typed_nb(
                device, dtype, queue, &*in_buf, &*out_buffer, rows, cols,
            )?;
            compute::streaming_tick();

            let out_id = next_tensor_id();
            let out_s = Shape::new(out_shape)?;
            self.tensors.insert(out_id, EagerTensor {
                buffer: out_buffer,
                layout: TensorLayout::contiguous(out_s),
                dtype,
                offset: 0,
            });
            return Ok((out_id, out_ptr));
        }

        // For non-last dim: reshape so target dim is last, sum, reshape back.
        // For 2D [rows, cols] with dim=0: treat as rows=cols, cols=rows (transposed sum).
        // General case: flatten dims before target as batch, target dim as cols.
        //
        // Example: [A, B, C] sum(dim=1) → reshape to [A, C, B] → sum last → [A, C]
        // But we can avoid reshape by computing rows/cols directly:
        //   cols = shape[dim]
        //   rows = product of all other dims
        //   But the data layout matters — we need contiguous dim as the summed axis.
        //
        // For the common case (bias gradient): [batch, hidden] sum(dim=0) → [hidden]
        // This is: for each column j, sum across rows i: out[j] = sum_i(in[i*hidden + j])
        // The sum kernel reduces along the LAST (contiguous) dim.
        // So we need to transpose: [batch, hidden] → [hidden, batch] → sum → [hidden]
        //
        // Use CPU-via-shared-memory for now. The data is already in shared memory.
        self.flush_and_wait();

        let (data_ptr, strides) = {
            let t = self.get(input_id)?;
            (t.data_ptr(), t.layout.strides().to_vec())
        };

        // Compute output shape
        let mut out_shape: Vec<usize> = shape_vec.iter().enumerate()
            .filter(|&(i, _)| i != dim || keepdim)
            .map(|(i, &d)| if i == dim && keepdim { 1 } else { d })
            .collect();
        if out_shape.is_empty() { out_shape = vec![1]; }

        let out_numel: usize = out_shape.iter().product();
        let (out_id, out_ptr) = self.alloc(device, &out_shape, dtype)?;

        if dtype == DType::Float32 {
            let data = unsafe { std::slice::from_raw_parts(data_ptr as *const f32, shape_vec.iter().product()) };
            let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_numel) };

            // Initialize output to zero
            for v in out.iter_mut() { *v = 0.0; }

            // Iterate over all input elements, accumulate into output
            let in_numel: usize = shape_vec.iter().product();
            for linear in 0..in_numel {
                // Convert linear index to multi-dim
                let mut remaining = linear;
                let mut src_offset = 0usize;
                let mut out_idx = 0usize;
                let mut out_stride = 1usize;

                // Build output index (skip the summed dim)
                let mut out_indices = vec![0usize; ndim];
                for d in (0..ndim).rev() {
                    let idx = remaining % shape_vec[d];
                    remaining /= shape_vec[d];
                    out_indices[d] = idx;
                    src_offset += idx * strides[d];
                }

                // Compute output linear index
                let mut out_linear = 0;
                let mut out_mult = 1;
                for d in (0..ndim).rev() {
                    if d == dim {
                        if keepdim {
                            // This dim is 1 in output, contributes 0
                            out_mult *= 1;
                        }
                        continue;
                    }
                    out_linear += out_indices[d] * out_mult;
                    out_mult *= if d == dim && keepdim { 1 } else { shape_vec[d] };
                }

                let elem_size = dtype.size_bytes();
                let val = unsafe { *(data_ptr.add(src_offset * elem_size) as *const f32) };
                out[out_linear] += val;
            }
        }

        Ok((out_id, out_ptr))
    }

    // ── Add scaled in-place ──────────────────────────────────────────

    /// self += alpha * other. SGD optimizer parameter update.
    pub fn add_scaled_inplace(
        &mut self,
        device: &Device,
        self_id: u64,
        other_id: u64,
        alpha: f32,
    ) -> Result<()> {
        let (scaled_id, _) = self.scalar_mul(device, other_id, alpha)?;
        self.inplace_binary_op(device, "elementwise_add", self_id, scaled_id)?;
        self.free(scaled_id);
        Ok(())
    }

    // ── Kernel pipeline resolution ───────────────────────────────────

    /// Resolve a kernel by base name + dtype, compile/cache the pipeline.
    pub(crate) fn get_pipeline(
        &self,
        device: &Device,
        base_name: &str,
        dtype: DType,
    ) -> Result<Arc<ComputePipeline>> {
        let (source, func_name) = compute::KernelRegistry::resolve_kernel(base_name, dtype);
        self.registry.get_or_create(device, &source, &func_name)
    }
}
