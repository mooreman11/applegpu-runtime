use std::collections::HashMap;

use crate::compute::KernelRegistry;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::ffi;
use crate::graph::{Graph, OpNode};
use crate::limits::ResourceLimits;
use crate::pool::BufferPool;
use crate::scheduler::{ContainerId, Scheduler};
use crate::tensor::{DType, Tensor, TensorLayout, Shape, MAX_DIMS};

use once_cell::sync::Lazy;

static REGISTRY: Lazy<KernelRegistry> = Lazy::new(KernelRegistry::new);

/// Storage for lazy tensors — holds both materialized data and pending graph.
pub struct LazyRuntime {
    /// Materialized tensors (have actual GPU buffers).
    tensors: HashMap<u64, Tensor>,
    /// Pending computation graph.
    graph: Graph,
    /// Multi-container scheduler (single resource authority).
    pub scheduler: Scheduler,
    /// Buffer pool for reusing GPU allocations.
    pub pool: BufferPool,
}

impl LazyRuntime {
    pub fn new() -> Self {
        LazyRuntime {
            tensors: HashMap::new(),
            graph: Graph::new(),
            scheduler: Scheduler::new(ResourceLimits::from_env()),
            pool: BufferPool::new(256 * 1024 * 1024),
        }
    }

    /// Store a materialized tensor (e.g., from user data).
    /// Checks resource limits before inserting. Uses default container.
    pub fn insert_tensor(&mut self, tensor: Tensor) -> Result<()> {
        let size = tensor.meta.size_bytes();
        let id = tensor.meta.id;
        self.scheduler.allocate_tensor(ContainerId::DEFAULT, id, size)?;
        self.tensors.insert(id, tensor);
        Ok(())
    }

    /// Insert a tensor with custom memory accounting size.
    /// Used for shared (zero-copy) tensors where memory belongs to Python, not Metal.
    pub fn insert_tensor_with_size(&mut self, tensor: Tensor, accounting_size: usize) -> Result<()> {
        let id = tensor.meta.id;
        self.scheduler.allocate_tensor(ContainerId::DEFAULT, id, accounting_size)?;
        self.tensors.insert(id, tensor);
        Ok(())
    }

    /// Insert a tensor attributed to a specific container.
    pub fn insert_tensor_for(&mut self, tensor: Tensor, container_id: ContainerId) -> Result<()> {
        let size = tensor.meta.size_bytes();
        let id = tensor.meta.id;
        self.scheduler.allocate_tensor(container_id, id, size)?;
        self.tensors.insert(id, tensor);
        Ok(())
    }

    /// Record a lazy operation. Returns the output node ID.
    pub fn record_op(&mut self, node: OpNode) -> u64 {
        let id = node.id;
        self.graph.add_node(node);
        id
    }

    /// Check if a tensor ID is materialized.
    pub fn is_materialized(&self, id: u64) -> bool {
        self.tensors.contains_key(&id)
    }

    /// Check if a tensor ID is pending (in the graph).
    pub fn is_pending(&self, id: u64) -> bool {
        self.graph.has_node(id)
    }

    /// Get a graph node by ID (for serialization).
    pub fn graph_node(&self, id: u64) -> Option<&crate::graph::OpNode> {
        self.graph.get_node(id)
    }

    /// Check if an ID exists (either materialized or pending).
    pub fn exists(&self, id: u64) -> bool {
        self.is_materialized(id) || self.is_pending(id)
    }

    /// Get shape of a tensor (materialized or pending).
    pub fn shape(&self, id: u64) -> Result<Vec<usize>> {
        if let Some(t) = self.tensors.get(&id) {
            return Ok(t.meta.layout.shape.dims().to_vec());
        }
        if let Some(node) = self.graph.get_node(id) {
            return Ok(node.out_shape.dims().to_vec());
        }
        Err(GpuError::GraphError(format!("Tensor {} not found", id)))
    }

    /// Evaluate a tensor: execute all pending ops needed to materialize it.
    /// After evaluation, the tensor is in `self.tensors` and its graph nodes are removed.
    ///
    /// Dispatches independent ops across multiple Metal command queues (up to 4) using
    /// `parallel_levels()` to identify ops with no data dependencies between them.
    /// Each level's command buffers are committed and waited on before proceeding to
    /// the next level, providing correct per-level synchronization without MTLEvent.
    /// Falls back to the single command buffer path for linear graphs (zero overhead).
    ///
    /// TODO: Phase II could use MTLSharedEvent (which supports multiple signalers via
    /// monotonically increasing values from different encoders) instead of commit-and-wait.
    /// Plain MTLEvent is unsuitable because ANY encoder signaling value N marks it done,
    /// even if other encoders at the same level haven't finished yet.
    pub fn eval(&mut self, device: &Device, id: u64) -> Result<()> {
        if self.is_materialized(id) {
            return Ok(()); // already done
        }

        let container_id = self.resolve_container(id);

        // Run fusion BEFORE computing parallel levels (fusion mutates graph topology)
        let order = self.graph.topo_sort(id)?;
        if order.is_empty() {
            return Err(GpuError::GraphError(format!("Tensor {} not found", id)));
        }
        let _fused_order = crate::fusion::optimize(&mut self.graph, &order);

        let levels = self.graph.parallel_levels(id)?;

        // Fast path: linear graph -> existing single-CB path
        if levels.iter().all(|l| l.len() == 1) {
            return self.eval_single_cb(device, container_id, levels);
        }

        let num_queues = std::cmp::min(
            levels.iter().map(|l| l.len()).max().unwrap_or(1),
            4,
        );
        let queues: Vec<_> = (0..num_queues)
            .map(|i| crate::compute::get_queue(device, i as u32))
            .collect();

        let loop_result: Result<()> = (|| {
            for (_level_idx, level) in levels.iter().enumerate() {
                // Create ONE command buffer per QUEUE (not per node).
                // Multiple nodes sharing a queue share a CB.
                let mut queue_cbs: HashMap<usize, *mut std::ffi::c_void> = HashMap::new();
                for (i, _) in level.iter().enumerate() {
                    let queue_idx = i % num_queues;
                    queue_cbs.entry(queue_idx).or_insert_with(|| {
                        let ctx_id = (queue_idx + 1) as u32;
                        let cb = crate::compute::set_batch_context(ctx_id, queues[queue_idx]);
                        cb // caller checks for null below
                    });
                }

                // Check for any null command buffers (set_batch_context failure)
                if queue_cbs.values().any(|cb| cb.is_null()) {
                    return Err(GpuError::GraphError(
                        "Failed to create batch context for one or more queues".to_string(),
                    ));
                }

                // Encode ops for this level — each node uses its queue's CB
                for (node_idx, &node_id) in level.iter().enumerate() {
                    if self.is_materialized(node_id) { continue; }
                    let node = self.graph.remove_node(node_id)
                        .ok_or_else(|| GpuError::GraphError(format!("Node {} not found", node_id)))?;
                    let out_buf = self.pool.acquire(device, node.out_shape.numel() * node.out_dtype.size_bytes())?;
                    let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);

                    // Set active context to this node's queue
                    let queue_idx = node_idx % num_queues;
                    crate::compute::set_active_context((queue_idx + 1) as u32);
                    self.execute_node_nb(device, queues[queue_idx], &node, &out)?;

                    self.scheduler.allocate_tensor(container_id, node_id, node.out_shape.numel() * node.out_dtype.size_bytes())?;
                    self.tensors.insert(node_id, out);
                }

                // Commit all CBs for this level, then wait for all to complete
                // before proceeding to the next level (correct synchronization).
                let mut level_cbs: Vec<*mut std::ffi::c_void> = Vec::new();
                for (&queue_idx, &_cb) in &queue_cbs {
                    let ctx_id = (queue_idx + 1) as u32;
                    let committed = crate::compute::commit_batch_context(ctx_id);
                    level_cbs.push(committed);
                }
                for cb in &level_cbs {
                    if !cb.is_null() {
                        crate::compute::wait_command_buffer(*cb);
                    }
                }
            }
            Ok(())
        })();

        // Reset active context to default
        crate::compute::set_active_context(0);

        // On error, clean up any uncommitted batch contexts
        if loop_result.is_err() {
            for queue_idx in 0..num_queues {
                let ctx_id = (queue_idx + 1) as u32;
                let cb = crate::compute::commit_batch_context(ctx_id);
                if !cb.is_null() {
                    crate::compute::wait_command_buffer(cb);
                }
            }
        }

        loop_result
    }

    /// Single command buffer eval path — used for linear graphs (no parallelism)
    /// and as fallback when MTLEvent creation fails.
    fn eval_single_cb(&mut self, device: &Device, container_id: ContainerId, levels: Vec<Vec<u64>>) -> Result<()> {
        let order: Vec<u64> = levels.into_iter().flat_map(|l| l.into_iter()).collect();
        let queue = crate::compute::get_shared_queue(device);
        let batch_cb = crate::compute::begin_batch(queue);
        let use_batch = !batch_cb.is_null();
        let mut last_cb: Option<*mut std::ffi::c_void> = None;

        let loop_result: Result<()> = (|| {
            for node_id in order {
                if self.is_materialized(node_id) { continue; }
                let node = self.graph.remove_node(node_id).ok_or_else(|| {
                    GpuError::GraphError(format!("Node {} not found in graph", node_id))
                })?;
                let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();
                let out_buf = self.pool.acquire(device, out_size)?;
                let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
                let cb = self.execute_node_nb(device, queue, &node, &out)?;
                if !use_batch { last_cb = Some(cb); }
                let size = node.out_shape.numel() * node.out_dtype.size_bytes();
                self.scheduler.allocate_tensor(container_id, node_id, size)?;
                self.tensors.insert(node_id, out);
            }
            Ok(())
        })();

        if use_batch {
            if loop_result.is_ok() {
                let cb = crate::compute::end_batch();
                if !cb.is_null() {
                    crate::compute::wait_command_buffer(cb);
                }
            } else {
                crate::compute::abort_batch();
            }
        } else if let Some(cb) = last_cb {
            crate::compute::wait_command_buffer(cb);
        }
        loop_result
    }

    /// Resolve the container that owns (or will own) a tensor.
    fn resolve_container(&self, id: u64) -> ContainerId {
        if let Some(cid) = self.scheduler.tensor_owner(id) {
            return cid;
        }
        if let Some(node) = self.graph.get_node(id) {
            return node.container_id;
        }
        ContainerId::DEFAULT
    }

    /// Execute a single graph node synchronously, producing a materialized Tensor.
    /// Acquires output buffer from pool FIRST (mutable borrow on pool),
    /// then reads input tensors (shared borrow on tensors) to satisfy the borrow checker.
    /// Retained for eval_remote and any path that needs synchronous single-op execution.
    #[allow(dead_code)]
    fn execute_node(&mut self, device: &Device, node: &OpNode) -> Result<Tensor> {
        // Int64 kernels require Apple9+ GPU (M3/M4)
        if matches!(node.out_dtype, DType::Int64) && !device.supports_int64() {
            return Err(GpuError::UnsupportedDtype(
                "Int64 requires Apple9+ GPU (M3/M4). This device does not support it.".to_string(),
            ));
        }

        let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();

        // Pre-check: if the node references an existing tensor as output, reject borrowed buffers
        if let Some(existing) = self.tensors.get(&node.id) {
            if existing.buffer.kind.is_borrowed() {
                return Err(GpuError::ImmutableBuffer(existing.meta.id));
            }
        }

        // Handle fused kernels first (N-D stride-based dispatch)
        if let crate::graph::OpKind::FusedElementwise { ref kernel_source, ref function_name } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input_tensors: Vec<&Tensor> = node.inputs.iter()
                .map(|&id| self.get_tensor(id))
                .collect::<Result<Vec<_>>>()?;
            let input_buffers: Vec<&crate::buffer::Buffer> = input_tensors.iter()
                .map(|t| &t.buffer)
                .collect();

            // Compute per-input strides (contiguous for fused chains)
            let stride_arrays: Vec<[u32; MAX_DIMS]> = input_tensors.iter()
                .map(|t| {
                    let strides = TensorLayout::broadcast_strides_for(&t.meta.layout.shape, &node.out_shape);
                    Self::to_u32_array(&strides)
                })
                .collect();
            let stride_refs: Vec<&[u32; MAX_DIMS]> = stride_arrays.iter().collect();
            let out_shape_u32 = Self::shape_to_u32(&node.out_shape);
            let ndim = node.out_shape.ndim() as u32;
            let numel = node.out_shape.numel() as u32;

            REGISTRY.dispatch_fused_nd(
                device,
                kernel_source,
                function_name,
                &input_buffers,
                &out.buffer,
                &stride_refs,
                &out_shape_u32,
                ndim,
                numel,
            )?;
            return Ok(out);
        }

        // Comparison ops: output is Bool but kernel is resolved by INPUT dtype
        if node.op.is_comparison() {
            let (a_strides, b_strides, out_shape_u32, ndim, numel) = self.binary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            let input_dtype = a.meta.dtype;
            REGISTRY.dispatch_binary_nd_typed(
                device,
                node.op.kernel_name(),
                input_dtype,
                &a.buffer,
                &a_strides,
                &b.buffer,
                &b_strides,
                &out.buffer,
                &out_shape_u32,
                ndim,
                numel,
            )?;
            return Ok(out);
        }

        let dtype = node.out_dtype;

        if node.op.is_softmax() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let cols = dims[dims.len() - 1];
            let total_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
            REGISTRY.dispatch_softmax_typed(device, dtype, &input.buffer, &out.buffer, total_rows, cols)?;
            return Ok(out);
        }

        if node.op.is_softmax_backward() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let grad_output = self.get_tensor(node.inputs[0])?;
            let output = self.get_tensor(node.inputs[1])?;
            let dims = grad_output.meta.layout.shape.dims();
            let cols = dims[dims.len() - 1];
            let total_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
            REGISTRY.dispatch_softmax_backward_typed(device, dtype, &grad_output.buffer, &output.buffer, &out.buffer, total_rows, cols)?;
            return Ok(out);
        }

        if node.op.is_layer_norm_backward() {
            let eps = match node.op { crate::graph::OpKind::LayerNormBackward { eps } => eps, _ => unreachable!() };
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let grad_output = self.get_tensor(node.inputs[0])?;
            let input = self.get_tensor(node.inputs[1])?;
            let gamma = self.get_tensor(node.inputs[2])?;
            let dims = grad_output.meta.layout.shape.dims();
            let cols = dims[dims.len() - 1];
            let total_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
            REGISTRY.dispatch_layer_norm_backward_typed(device, dtype, &grad_output.buffer, &input.buffer, &gamma.buffer, &out.buffer, total_rows, cols, eps)?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Conv2dBackwardInput { stride, padding } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let grad_output = self.get_tensor(node.inputs[0])?;
            let weight = self.get_tensor(node.inputs[1])?;
            let go_dims = grad_output.meta.layout.shape.dims();
            let w_dims = weight.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let uint_params: Vec<u32> = vec![
                go_dims[0] as u32,   // batch
                w_dims[1] as u32,    // in_channels
                go_dims[1] as u32,   // out_channels
                out_dims[2] as u32,  // in_h
                out_dims[3] as u32,  // in_w
                go_dims[2] as u32,   // out_h
                go_dims[3] as u32,   // out_w
                w_dims[2] as u32,    // kh
                w_dims[3] as u32,    // kw
                stride.0 as u32,     // stride_h
                stride.1 as u32,     // stride_w
                padding.0 as u32,    // pad_h
                padding.1 as u32,    // pad_w
            ];
            let grid_y = out_dims[2] as u32 * w_dims[1] as u32;
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("conv2d_backward_input", dtype);
            REGISTRY.dispatch_cnn_3d(
                device, &k_src, &k_fn,
                &[&grad_output.buffer, &weight.buffer], &out.buffer,
                &uint_params, &[], (out_dims[3] as u32, grid_y, go_dims[0] as u32),
            )?;
            return Ok(out);
        }

        if node.op.is_embedding_backward() {
            let out_buf = self.pool.acquire(device, out_size)?;
            // Zero the output buffer (embedding backward accumulates into zeroed buffer)
            out_buf.zero()?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let grad_output = self.get_tensor(node.inputs[0])?;
            let indices = self.get_tensor(node.inputs[1])?;
            let seq_len = indices.meta.layout.shape.numel();
            let embed_dim = node.out_shape.dims()[1];
            let uint_params: Vec<u32> = vec![seq_len as u32, embed_dim as u32];
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("embedding_backward", dtype);
            REGISTRY.dispatch_cnn_3d(
                device, &k_src, &k_fn,
                &[&grad_output.buffer, &indices.buffer], &out.buffer,
                &uint_params, &[], (embed_dim as u32, seq_len as u32, 1),
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::BatchNormBackward { eps } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let grad_output = self.get_tensor(node.inputs[0])?;
            let weight = self.get_tensor(node.inputs[1])?;
            let running_var = self.get_tensor(node.inputs[2])?;
            let in_dims = grad_output.meta.layout.shape.dims();
            let batch = in_dims[0];
            let channels = in_dims[1];
            let spatial: usize = in_dims[2..].iter().product();
            let uint_params: Vec<u32> = vec![batch as u32, channels as u32, spatial as u32];
            let float_params: Vec<f32> = vec![eps];
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("batch_norm_backward", dtype);
            REGISTRY.dispatch_cnn_3d(
                device, &k_src, &k_fn,
                &[&grad_output.buffer, &weight.buffer, &running_var.buffer],
                &out.buffer, &uint_params, &float_params,
                (spatial as u32, channels as u32, batch as u32),
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Transpose { dim0, dim1 } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let ndim = dims.len();

            // Fast path: swapping last two dims uses optimized batched kernel
            if dim0 == ndim - 2 && dim1 == ndim - 1 {
                let rows = dims[ndim - 2];
                let cols = dims[ndim - 1];
                let batch_size: usize = dims[..ndim - 2].iter().product::<usize>().max(1);
                if batch_size <= 1 {
                    REGISTRY.dispatch_transpose_typed(device, dtype, &input.buffer, &out.buffer, rows, cols)?;
                } else {
                    REGISTRY.dispatch_transpose_batched_typed(device, dtype, &input.buffer, &out.buffer, batch_size, rows, cols)?;
                }
            } else {
                // General path: compute transposed strides, use strided copy kernel
                let in_strides = self.transposed_strides(dims, dim0, dim1);
                let out_shape_u32 = Self::shape_to_u32(&node.out_shape);
                let ndim_u32 = ndim as u32;
                let numel = node.out_shape.numel() as u32;
                REGISTRY.dispatch_unary_nd_typed(
                    device, "copy_strided", dtype,
                    &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim_u32, numel,
                )?;
            }
            return Ok(out);
        }

        if let crate::graph::OpKind::ScalarMul(ref sv) = node.op {
            let scale = sv.as_f64() as f32;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_scalar_mul_typed(device, dtype, &input.buffer, &out.buffer, scale, input.numel())?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Pow { ref exponent } = node.op {
            let exp_f32 = exponent.as_f64() as f32;
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_pow_nd_typed(
                device, dtype, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel, exp_f32,
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Clamp { ref min_val, ref max_val } = node.op {
            let min_f32 = min_val.as_f64() as f32;
            let max_f32 = max_val.as_f64() as f32;
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_clamp_nd_typed(
                device, dtype, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel, min_f32, max_f32,
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Shl { shift } = node.op {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_pow_nd_typed(
                device, dtype, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel, shift as f32,
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Shr { shift } = node.op {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_pow_nd_typed(
                device, dtype, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel, shift as f32,
            )?;
            return Ok(out);
        }

        if node.op.is_where() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let cond = self.get_tensor(node.inputs[0])?;
            let x = self.get_tensor(node.inputs[1])?;
            let y = self.get_tensor(node.inputs[2])?;
            let cond_strides = Self::to_u32_array(&TensorLayout::broadcast_strides_for(&cond.meta.layout.shape, &node.out_shape));
            let x_strides = Self::to_u32_array(&TensorLayout::broadcast_strides_for(&x.meta.layout.shape, &node.out_shape));
            let y_strides = Self::to_u32_array(&TensorLayout::broadcast_strides_for(&y.meta.layout.shape, &node.out_shape));
            let out_shape_u32 = Self::shape_to_u32(&node.out_shape);
            let ndim = node.out_shape.ndim() as u32;
            let numel = node.out_shape.numel() as u32;
            REGISTRY.dispatch_where_nd_typed(
                device, dtype,
                &cond.buffer, &cond_strides,
                &x.buffer, &x_strides,
                &y.buffer, &y_strides,
                &out.buffer, &out_shape_u32, ndim, numel,
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::MaskedFill { ref value } = node.op {
            let fill_f32 = value.as_f64() as f32;
            let (a_strides, b_strides, out_shape_u32, ndim, numel) = self.binary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let mask = self.get_tensor(node.inputs[1])?;
            REGISTRY.dispatch_masked_fill_nd_typed(
                device, dtype,
                &input.buffer, &a_strides,
                &mask.buffer, &b_strides,
                &out.buffer, &out_shape_u32, ndim, numel, fill_f32,
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Triu { diagonal } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let ndim = dims.len();
            let rows = dims[ndim - 2];
            let cols = dims[ndim - 1];
            let batch_size: usize = dims[..ndim - 2].iter().product::<usize>().max(1);
            REGISTRY.dispatch_triu_typed(device, dtype, &input.buffer, &out.buffer, batch_size, rows, cols, diagonal)?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Tril { diagonal } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let ndim = dims.len();
            let rows = dims[ndim - 2];
            let cols = dims[ndim - 1];
            let batch_size: usize = dims[..ndim - 2].iter().product::<usize>().max(1);
            REGISTRY.dispatch_tril_typed(device, dtype, &input.buffer, &out.buffer, batch_size, rows, cols, diagonal)?;
            return Ok(out);
        }

        if node.op.is_gelu() {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_unary_nd_typed(
                device, "gelu", dtype,
                &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel,
            )?;
            return Ok(out);
        }

        if node.op.is_layer_norm() {
            let eps = match node.op { crate::graph::OpKind::LayerNorm { eps } => eps, _ => unreachable!() };
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let gamma = self.get_tensor(node.inputs[1])?;
            let beta = self.get_tensor(node.inputs[2])?;
            let dims = input.meta.layout.shape.dims();
            let cols = dims[dims.len() - 1];
            let total_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
            REGISTRY.dispatch_layer_norm_typed(device, dtype, &input.buffer, &gamma.buffer, &beta.buffer, &out.buffer, total_rows, cols, eps)?;
            return Ok(out);
        }

        if node.op.is_embedding() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let weights = self.get_tensor(node.inputs[0])?;
            let indices = self.get_tensor(node.inputs[1])?;
            let seq_len = indices.meta.layout.shape.numel();
            let embed_dim = weights.meta.layout.shape.dims()[1];
            REGISTRY.dispatch_embedding_typed(device, dtype, &weights.buffer, &indices.buffer, &out.buffer, seq_len, embed_dim)?;
            return Ok(out);
        }

        // ── CNN ops ────────────────────────────────────────────────────────

        if let crate::graph::OpKind::Conv1d { stride, padding } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let weight = self.get_tensor(node.inputs[1])?;
            let in_dims = input.meta.layout.shape.dims();
            let w_dims = weight.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let uint_params: Vec<u32> = vec![
                in_dims[0] as u32,  // batch
                in_dims[1] as u32,  // in_channels
                w_dims[0] as u32,   // out_channels
                in_dims[2] as u32,  // in_length
                out_dims[2] as u32, // out_length
                w_dims[2] as u32,   // kernel_size
                stride as u32,
                padding as u32,
            ];
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("conv1d", dtype);
            REGISTRY.dispatch_cnn_3d(
                device, &k_src, &k_fn,
                &[&input.buffer, &weight.buffer], &out.buffer,
                &uint_params, &[], (out_dims[2] as u32, w_dims[0] as u32, in_dims[0] as u32),
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Conv2d { stride, padding } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let weight = self.get_tensor(node.inputs[1])?;
            let in_dims = input.meta.layout.shape.dims();
            let w_dims = weight.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let uint_params: Vec<u32> = vec![
                in_dims[0] as u32,  // batch
                in_dims[1] as u32,  // in_channels
                w_dims[0] as u32,   // out_channels
                in_dims[2] as u32,  // in_h
                in_dims[3] as u32,  // in_w
                out_dims[2] as u32, // out_h
                out_dims[3] as u32, // out_w
                w_dims[2] as u32,   // kh
                w_dims[3] as u32,   // kw
                stride.0 as u32,    // stride_h
                stride.1 as u32,    // stride_w
                padding.0 as u32,   // pad_h
                padding.1 as u32,   // pad_w
            ];
            // Grid: (out_w, out_h * out_channels, batch)
            let grid_y = out_dims[2] as u32 * w_dims[0] as u32;
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("conv2d", dtype);
            REGISTRY.dispatch_cnn_3d(
                device, &k_src, &k_fn,
                &[&input.buffer, &weight.buffer], &out.buffer,
                &uint_params, &[], (out_dims[3] as u32, grid_y, in_dims[0] as u32),
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::BatchNorm { eps } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let mean = self.get_tensor(node.inputs[1])?;
            let var = self.get_tensor(node.inputs[2])?;
            let weight = self.get_tensor(node.inputs[3])?;
            let bias = self.get_tensor(node.inputs[4])?;
            let in_dims = input.meta.layout.shape.dims();
            let batch = in_dims[0];
            let channels = in_dims[1];
            let spatial: usize = in_dims[2..].iter().product();
            let uint_params: Vec<u32> = vec![
                batch as u32,
                channels as u32,
                spatial as u32,
            ];
            let float_params: Vec<f32> = vec![eps];
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("batch_norm", dtype);
            REGISTRY.dispatch_cnn_3d(
                device, &k_src, &k_fn,
                &[&input.buffer, &mean.buffer, &var.buffer, &weight.buffer, &bias.buffer],
                &out.buffer, &uint_params, &float_params,
                (spatial as u32, channels as u32, batch as u32),
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::MaxPool2d { kernel_size, stride, padding } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let in_dims = input.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let uint_params: Vec<u32> = vec![
                in_dims[0] as u32,  // batch
                in_dims[1] as u32,  // channels
                in_dims[2] as u32,  // in_h
                in_dims[3] as u32,  // in_w
                out_dims[2] as u32, // out_h
                out_dims[3] as u32, // out_w
                kernel_size.0 as u32,
                kernel_size.1 as u32,
                stride.0 as u32,
                stride.1 as u32,
                padding.0 as u32,
                padding.1 as u32,
            ];
            let grid_y = out_dims[2] as u32 * in_dims[1] as u32;
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("max_pool2d", dtype);
            REGISTRY.dispatch_cnn_3d(
                device, &k_src, &k_fn,
                &[&input.buffer], &out.buffer,
                &uint_params, &[], (out_dims[3] as u32, grid_y, in_dims[0] as u32),
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::AvgPool2d { kernel_size, stride, padding } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let in_dims = input.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let uint_params: Vec<u32> = vec![
                in_dims[0] as u32,  // batch
                in_dims[1] as u32,  // channels
                in_dims[2] as u32,  // in_h
                in_dims[3] as u32,  // in_w
                out_dims[2] as u32, // out_h
                out_dims[3] as u32, // out_w
                kernel_size.0 as u32,
                kernel_size.1 as u32,
                stride.0 as u32,
                stride.1 as u32,
                padding.0 as u32,
                padding.1 as u32,
            ];
            let grid_y = out_dims[2] as u32 * in_dims[1] as u32;
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("avg_pool2d", dtype);
            REGISTRY.dispatch_cnn_3d(
                device, &k_src, &k_fn,
                &[&input.buffer], &out.buffer,
                &uint_params, &[], (out_dims[3] as u32, grid_y, in_dims[0] as u32),
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Gather { dim } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let indices = self.get_tensor(node.inputs[1])?;
            let in_dims = input.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let kernel_base = if dim == 0 { "gather_dim0" } else { "gather_dim1" };
            REGISTRY.dispatch_gather_typed(
                device, dtype, kernel_base,
                &input.buffer, &indices.buffer, &out.buffer,
                out_dims[0], in_dims[1], out_dims[1],
            )?;
            return Ok(out);
        }

        if let crate::graph::OpKind::IndexSelect { dim } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let indices = self.get_tensor(node.inputs[1])?;
            let in_dims = input.meta.layout.shape.dims();
            let num_indices = indices.meta.layout.shape.numel();
            if dim == 0 {
                REGISTRY.dispatch_index_select_dim0_typed(
                    device, dtype,
                    &input.buffer, &indices.buffer, &out.buffer,
                    num_indices, in_dims[1],
                )?;
            } else {
                REGISTRY.dispatch_index_select_dim1_typed(
                    device, dtype,
                    &input.buffer, &indices.buffer, &out.buffer,
                    in_dims[0], in_dims[1], num_indices,
                )?;
            }
            return Ok(out);
        }

        if node.op.is_reshape() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let size_bytes = input.meta.size_bytes();
            let queue = crate::compute::get_shared_queue(device);
            let cb = unsafe {
                ffi::gpu_bridge_blit_copy_nb(
                    device.raw_handle() as *mut _,
                    queue,
                    input.buffer.raw_handle(),
                    out.buffer.raw_handle(),
                    size_bytes as u64,
                )
            };
            if cb.is_null() {
                return Err(GpuError::ComputeFailed("Blit copy failed".to_string()));
            }
            crate::compute::wait_command_buffer(cb);
            return Ok(out);
        }

        if let crate::graph::OpKind::Slice { dim, start, .. } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let in_dims = input.meta.layout.shape.dims();
            if dim == 0 {
                let cols = in_dims[1];
                let out_rows = node.out_shape.dims()[0];
                REGISTRY.dispatch_slice_dim0_typed(device, dtype, &input.buffer, &out.buffer, cols, start, out_rows)?;
            } else {
                let rows = in_dims[0];
                let in_cols = in_dims[1];
                let out_cols = node.out_shape.dims()[1];
                REGISTRY.dispatch_slice_dim1_typed(device, dtype, &input.buffer, &out.buffer, in_cols, out_cols, start, rows)?;
            }
            return Ok(out);
        }

        if let crate::graph::OpKind::Concat { dim } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            let a_dims = a.meta.layout.shape.dims();
            if dim == 0 {
                let rows_a = a_dims[0];
                let cols = a_dims[1];
                let total_rows = node.out_shape.dims()[0];
                REGISTRY.dispatch_concat_dim0_typed(device, dtype, &a.buffer, &b.buffer, &out.buffer, rows_a, cols, total_rows)?;
            } else {
                let rows = a_dims[0];
                let cols_a = a_dims[1];
                let cols_b = b.meta.layout.shape.dims()[1];
                REGISTRY.dispatch_concat_dim1_typed(device, dtype, &a.buffer, &b.buffer, &out.buffer, rows, cols_a, cols_b)?;
            }
            return Ok(out);
        }

        if node.op.is_add_bias() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let bias = self.get_tensor(node.inputs[1])?;
            let dims = input.meta.layout.shape.dims();
            let (rows, cols) = (dims[0], dims[1]);
            REGISTRY.dispatch_add_bias_typed(device, dtype, &input.buffer, &bias.buffer, &out.buffer, rows, cols)?;
            return Ok(out);
        }

        if node.op.is_softmax_causal() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let ndim = dims.len();
            let rows = dims[ndim - 2];
            let cols = dims[ndim - 1];
            let batch_size: usize = dims[..ndim - 2].iter().product::<usize>().max(1);
            REGISTRY.dispatch_softmax_causal_typed(device, dtype, &input.buffer, &out.buffer, batch_size, rows, cols)?;
            return Ok(out);
        }

        if node.op.is_argmax() {
            // Argmax: cross-dtype output. Input is f32/f16, output is Int32.
            // Acquire output buffer first (pool borrow), then read input (tensor borrow).
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), DType::Int32, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let input_dtype = input.meta.dtype;
            let in_dims = input.meta.layout.shape.dims();
            let (rows, cols) = if in_dims.len() == 2 {
                (in_dims[0], in_dims[1])
            } else {
                (1, in_dims[0])
            };
            REGISTRY.dispatch_argmax_typed(device, input_dtype, &input.buffer, &out.buffer, rows, cols)?;
            return Ok(out);
        }

        if node.op.is_sum() || node.op.is_mean() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let cols = dims[dims.len() - 1];
            let total_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
            if node.op.is_sum() {
                REGISTRY.dispatch_sum_typed(device, dtype, &input.buffer, &out.buffer, total_rows, cols)?;
            } else {
                REGISTRY.dispatch_mean_typed(device, dtype, &input.buffer, &out.buffer, total_rows, cols)?;
            }
            return Ok(out);
        }

        if let crate::graph::OpKind::Cast { target_dtype } = node.op {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let src_dtype = input.meta.dtype;
            let (source, func) = KernelRegistry::resolve_cast_kernel(src_dtype, target_dtype);
            let pipeline = REGISTRY.get_or_create(device, &source, &func)?;
            pipeline.dispatch_unary_nd(&input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel)?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Quantize { scale, zero_point, target_dtype } = node.op {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let src_dtype = input.meta.dtype;
            let (source, func) = KernelRegistry::resolve_quantize_kernel(src_dtype, target_dtype, scale, zero_point);
            let pipeline = REGISTRY.get_or_create(device, &source, &func)?;
            pipeline.dispatch_unary_nd(&input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel)?;
            return Ok(out);
        }

        if let crate::graph::OpKind::Dequantize { scale, zero_point, target_dtype } = node.op {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let src_dtype = input.meta.dtype;
            let (source, func) = KernelRegistry::resolve_dequantize_kernel(src_dtype, target_dtype, scale, zero_point);
            let pipeline = REGISTRY.get_or_create(device, &source, &func)?;
            pipeline.dispatch_unary_nd(&input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel)?;
            return Ok(out);
        }

        if node.op.is_unary() {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_unary_nd_typed(
                device,
                node.op.kernel_name(),
                dtype,
                &input.buffer,
                &in_strides,
                &out.buffer,
                &out_shape_u32,
                ndim,
                numel,
            )?;
            Ok(out)
        } else if node.op.is_matmul() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            let a_dims = a.meta.layout.shape.dims();
            let b_dims = b.meta.layout.shape.dims();
            let a_ndim = a_dims.len();
            let b_ndim = b_dims.len();
            let m = a_dims[a_ndim - 2];
            let k = a_dims[a_ndim - 1];
            let n = b_dims[b_ndim - 1];
            let out_dims = node.out_shape.dims();
            let out_ndim = out_dims.len();
            let batch_size: usize = out_dims[..out_ndim - 2].iter().product::<usize>().max(1);
            let a_batch_stride = if a_ndim > 2 { m * k } else { 0 };
            let b_batch_stride = if b_ndim > 2 { k * n } else { 0 };
            if batch_size <= 1 {
                REGISTRY.dispatch_matmul_typed(device, dtype, &a.buffer, &b.buffer, &out.buffer, m, n, k)?;
            } else {
                REGISTRY.dispatch_matmul_batched_typed(device, dtype, &a.buffer, &b.buffer, &out.buffer, m, n, k, batch_size, a_batch_stride, b_batch_stride)?;
            }
            Ok(out)
        } else {
            // Binary element-wise (N-D with broadcasting)
            let (a_strides, b_strides, out_shape_u32, ndim, numel) = self.binary_nd_params(node)?;
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            REGISTRY.dispatch_binary_nd_typed(
                device,
                node.op.kernel_name(),
                dtype,
                &a.buffer,
                &a_strides,
                &b.buffer,
                &b_strides,
                &out.buffer,
                &out_shape_u32,
                ndim,
                numel,
            )?;
            Ok(out)
        }
    }

    /// Execute a single graph node without blocking. The output tensor is pre-allocated
    /// by the caller. Returns the command buffer handle for deferred waiting.
    fn execute_node_nb(
        &self,
        device: &Device,
        queue: *mut std::ffi::c_void,
        node: &OpNode,
        out: &Tensor,
    ) -> Result<*mut std::ffi::c_void> {
        // Int64 kernels require Apple9+ GPU (M3/M4)
        if matches!(node.out_dtype, DType::Int64) && !device.supports_int64() {
            return Err(GpuError::UnsupportedDtype(
                "Int64 requires Apple9+ GPU (M3/M4). This device does not support it.".to_string(),
            ));
        }

        // Reject borrowed (immutable) buffers as output targets
        if out.buffer.kind.is_borrowed() {
            return Err(GpuError::ImmutableBuffer(out.meta.id));
        }

        // Handle fused kernels first (N-D stride-based dispatch)
        if let crate::graph::OpKind::FusedElementwise { ref kernel_source, ref function_name } = node.op {
            let input_tensors: Vec<&Tensor> = node.inputs.iter()
                .map(|&id| self.get_tensor(id))
                .collect::<Result<Vec<_>>>()?;
            let input_buffers: Vec<&crate::buffer::Buffer> = input_tensors.iter()
                .map(|t| &t.buffer)
                .collect();

            // Compute per-input strides (contiguous for fused chains)
            let stride_arrays: Vec<[u32; MAX_DIMS]> = input_tensors.iter()
                .map(|t| {
                    let strides = TensorLayout::broadcast_strides_for(&t.meta.layout.shape, &node.out_shape);
                    Self::to_u32_array(&strides)
                })
                .collect();
            let stride_refs: Vec<&[u32; MAX_DIMS]> = stride_arrays.iter().collect();
            let out_shape_u32 = Self::shape_to_u32(&node.out_shape);
            let ndim = node.out_shape.ndim() as u32;
            let numel = node.out_shape.numel() as u32;

            return REGISTRY.dispatch_fused_nd_nb(
                device,
                kernel_source,
                function_name,
                queue,
                &input_buffers,
                &out.buffer,
                &stride_refs,
                &out_shape_u32,
                ndim,
                numel,
            );
        }

        // Comparison ops: output is Bool but kernel is resolved by INPUT dtype
        if node.op.is_comparison() {
            let (a_strides, b_strides, out_shape_u32, ndim, numel) = self.binary_nd_params(node)?;
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            let input_dtype = a.meta.dtype;
            return REGISTRY.dispatch_binary_nd_typed_nb(
                device,
                node.op.kernel_name(),
                input_dtype,
                queue,
                &a.buffer,
                &a_strides,
                &b.buffer,
                &b_strides,
                &out.buffer,
                &out_shape_u32,
                ndim,
                numel,
            );
        }

        let dtype = node.out_dtype;

        if node.op.is_softmax() {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let cols = dims[dims.len() - 1];
            let total_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
            return REGISTRY.dispatch_softmax_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, total_rows, cols);
        }

        if node.op.is_softmax_backward() {
            let grad_output = self.get_tensor(node.inputs[0])?;
            let output = self.get_tensor(node.inputs[1])?;
            let dims = grad_output.meta.layout.shape.dims();
            let cols = dims[dims.len() - 1];
            let total_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
            return REGISTRY.dispatch_softmax_backward_typed_nb(device, dtype, queue, &grad_output.buffer, &output.buffer, &out.buffer, total_rows, cols);
        }

        if node.op.is_layer_norm_backward() {
            let eps = match node.op { crate::graph::OpKind::LayerNormBackward { eps } => eps, _ => unreachable!() };
            let grad_output = self.get_tensor(node.inputs[0])?;
            let input = self.get_tensor(node.inputs[1])?;
            let gamma = self.get_tensor(node.inputs[2])?;
            let dims = grad_output.meta.layout.shape.dims();
            let cols = dims[dims.len() - 1];
            let total_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
            return REGISTRY.dispatch_layer_norm_backward_typed_nb(device, dtype, queue, &grad_output.buffer, &input.buffer, &gamma.buffer, &out.buffer, total_rows, cols, eps);
        }

        if let crate::graph::OpKind::Conv2dBackwardInput { stride, padding } = node.op {
            let grad_output = self.get_tensor(node.inputs[0])?;
            let weight = self.get_tensor(node.inputs[1])?;
            let go_dims = grad_output.meta.layout.shape.dims();
            let w_dims = weight.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let uint_params: Vec<u32> = vec![
                go_dims[0] as u32, w_dims[1] as u32, go_dims[1] as u32,
                out_dims[2] as u32, out_dims[3] as u32,
                go_dims[2] as u32, go_dims[3] as u32,
                w_dims[2] as u32, w_dims[3] as u32,
                stride.0 as u32, stride.1 as u32,
                padding.0 as u32, padding.1 as u32,
            ];
            let grid_y = out_dims[2] as u32 * w_dims[1] as u32;
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("conv2d_backward_input", dtype);
            return REGISTRY.dispatch_cnn_3d_nb(
                device, &k_src, &k_fn, queue,
                &[&grad_output.buffer, &weight.buffer], &out.buffer,
                &uint_params, &[], (out_dims[3] as u32, grid_y, go_dims[0] as u32),
            );
        }

        if node.op.is_embedding_backward() {
            // Zero the output buffer (embedding backward accumulates into zeroed buffer)
            out.buffer.zero()?;
            let grad_output = self.get_tensor(node.inputs[0])?;
            let indices = self.get_tensor(node.inputs[1])?;
            let seq_len = indices.meta.layout.shape.numel();
            let embed_dim = node.out_shape.dims()[1];
            let uint_params: Vec<u32> = vec![seq_len as u32, embed_dim as u32];
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("embedding_backward", dtype);
            return REGISTRY.dispatch_cnn_3d_nb(
                device, &k_src, &k_fn, queue,
                &[&grad_output.buffer, &indices.buffer], &out.buffer,
                &uint_params, &[], (embed_dim as u32, seq_len as u32, 1),
            );
        }

        if let crate::graph::OpKind::BatchNormBackward { eps } = node.op {
            let grad_output = self.get_tensor(node.inputs[0])?;
            let weight = self.get_tensor(node.inputs[1])?;
            let running_var = self.get_tensor(node.inputs[2])?;
            let in_dims = grad_output.meta.layout.shape.dims();
            let batch = in_dims[0];
            let channels = in_dims[1];
            let spatial: usize = in_dims[2..].iter().product();
            let uint_params: Vec<u32> = vec![batch as u32, channels as u32, spatial as u32];
            let float_params: Vec<f32> = vec![eps];
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("batch_norm_backward", dtype);
            return REGISTRY.dispatch_cnn_3d_nb(
                device, &k_src, &k_fn, queue,
                &[&grad_output.buffer, &weight.buffer, &running_var.buffer],
                &out.buffer, &uint_params, &float_params,
                (spatial as u32, channels as u32, batch as u32),
            );
        }

        if let crate::graph::OpKind::Transpose { dim0, dim1 } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let ndim = dims.len();

            // Fast path: swapping last two dims uses optimized batched kernel
            if dim0 == ndim - 2 && dim1 == ndim - 1 {
                let rows = dims[ndim - 2];
                let cols = dims[ndim - 1];
                let batch_size: usize = dims[..ndim - 2].iter().product::<usize>().max(1);
                if batch_size <= 1 {
                    return REGISTRY.dispatch_transpose_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, rows, cols);
                } else {
                    return REGISTRY.dispatch_transpose_batched_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, batch_size, rows, cols);
                }
            } else {
                // General path: strided copy
                let in_strides = self.transposed_strides(dims, dim0, dim1);
                let out_shape_u32 = Self::shape_to_u32(&node.out_shape);
                let ndim_u32 = ndim as u32;
                let numel = node.out_shape.numel() as u32;
                return REGISTRY.dispatch_unary_nd_typed_nb(
                    device, "copy_strided", dtype, queue,
                    &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim_u32, numel,
                );
            }
        }

        if let crate::graph::OpKind::ScalarMul(ref sv) = node.op {
            let scale = sv.as_f64() as f32;
            let input = self.get_tensor(node.inputs[0])?;
            return REGISTRY.dispatch_scalar_mul_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, scale, input.numel());
        }

        if let crate::graph::OpKind::Pow { ref exponent } = node.op {
            let exp_f32 = exponent.as_f64() as f32;
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let input = self.get_tensor(node.inputs[0])?;
            return REGISTRY.dispatch_pow_nd_typed_nb(
                device, dtype, queue, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel, exp_f32,
            );
        }

        if let crate::graph::OpKind::Clamp { ref min_val, ref max_val } = node.op {
            let min_f32 = min_val.as_f64() as f32;
            let max_f32 = max_val.as_f64() as f32;
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let input = self.get_tensor(node.inputs[0])?;
            return REGISTRY.dispatch_clamp_nd_typed_nb(
                device, dtype, queue, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel, min_f32, max_f32,
            );
        }

        if let crate::graph::OpKind::Shl { shift } = node.op {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let input = self.get_tensor(node.inputs[0])?;
            return REGISTRY.dispatch_pow_nd_typed_nb(
                device, dtype, queue, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel, shift as f32,
            );
        }

        if let crate::graph::OpKind::Shr { shift } = node.op {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let input = self.get_tensor(node.inputs[0])?;
            return REGISTRY.dispatch_pow_nd_typed_nb(
                device, dtype, queue, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel, shift as f32,
            );
        }

        if node.op.is_where() {
            let cond = self.get_tensor(node.inputs[0])?;
            let x = self.get_tensor(node.inputs[1])?;
            let y = self.get_tensor(node.inputs[2])?;
            let cond_strides = Self::to_u32_array(&TensorLayout::broadcast_strides_for(&cond.meta.layout.shape, &node.out_shape));
            let x_strides = Self::to_u32_array(&TensorLayout::broadcast_strides_for(&x.meta.layout.shape, &node.out_shape));
            let y_strides = Self::to_u32_array(&TensorLayout::broadcast_strides_for(&y.meta.layout.shape, &node.out_shape));
            let out_shape_u32 = Self::shape_to_u32(&node.out_shape);
            let ndim = node.out_shape.ndim() as u32;
            let numel = node.out_shape.numel() as u32;
            return REGISTRY.dispatch_where_nd_typed_nb(
                device, dtype, queue,
                &cond.buffer, &cond_strides,
                &x.buffer, &x_strides,
                &y.buffer, &y_strides,
                &out.buffer, &out_shape_u32, ndim, numel,
            );
        }

        if let crate::graph::OpKind::MaskedFill { ref value } = node.op {
            let fill_f32 = value.as_f64() as f32;
            let (a_strides, b_strides, out_shape_u32, ndim, numel) = self.binary_nd_params(node)?;
            let input = self.get_tensor(node.inputs[0])?;
            let mask = self.get_tensor(node.inputs[1])?;
            return REGISTRY.dispatch_masked_fill_nd_typed_nb(
                device, dtype, queue,
                &input.buffer, &a_strides,
                &mask.buffer, &b_strides,
                &out.buffer, &out_shape_u32, ndim, numel, fill_f32,
            );
        }

        if let crate::graph::OpKind::Triu { diagonal } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let ndim = dims.len();
            let rows = dims[ndim - 2];
            let cols = dims[ndim - 1];
            let batch_size: usize = dims[..ndim - 2].iter().product::<usize>().max(1);
            return REGISTRY.dispatch_triu_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, batch_size, rows, cols, diagonal);
        }

        if let crate::graph::OpKind::Tril { diagonal } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let ndim = dims.len();
            let rows = dims[ndim - 2];
            let cols = dims[ndim - 1];
            let batch_size: usize = dims[..ndim - 2].iter().product::<usize>().max(1);
            return REGISTRY.dispatch_tril_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, batch_size, rows, cols, diagonal);
        }

        if node.op.is_gelu() {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let input = self.get_tensor(node.inputs[0])?;
            return REGISTRY.dispatch_unary_nd_typed_nb(
                device, "gelu", dtype, queue,
                &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel,
            );
        }

        if node.op.is_layer_norm() {
            let eps = match node.op { crate::graph::OpKind::LayerNorm { eps } => eps, _ => unreachable!() };
            let input = self.get_tensor(node.inputs[0])?;
            let gamma = self.get_tensor(node.inputs[1])?;
            let beta = self.get_tensor(node.inputs[2])?;
            let dims = input.meta.layout.shape.dims();
            let cols = dims[dims.len() - 1];
            let total_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
            return REGISTRY.dispatch_layer_norm_typed_nb(device, dtype, queue, &input.buffer, &gamma.buffer, &beta.buffer, &out.buffer, total_rows, cols, eps);
        }

        if node.op.is_embedding() {
            let weights = self.get_tensor(node.inputs[0])?;
            let indices = self.get_tensor(node.inputs[1])?;
            let seq_len = indices.meta.layout.shape.numel();
            let embed_dim = weights.meta.layout.shape.dims()[1];
            return REGISTRY.dispatch_embedding_typed_nb(device, dtype, queue, &weights.buffer, &indices.buffer, &out.buffer, seq_len, embed_dim);
        }

        // ── CNN ops (non-blocking) ──────────────────────────────────────

        if let crate::graph::OpKind::Conv1d { stride, padding } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let weight = self.get_tensor(node.inputs[1])?;
            let in_dims = input.meta.layout.shape.dims();
            let w_dims = weight.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let uint_params: Vec<u32> = vec![
                in_dims[0] as u32, in_dims[1] as u32, w_dims[0] as u32,
                in_dims[2] as u32, out_dims[2] as u32, w_dims[2] as u32,
                stride as u32, padding as u32,
            ];
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("conv1d", dtype);
            return REGISTRY.dispatch_cnn_3d_nb(
                device, &k_src, &k_fn, queue,
                &[&input.buffer, &weight.buffer], &out.buffer,
                &uint_params, &[], (out_dims[2] as u32, w_dims[0] as u32, in_dims[0] as u32),
            );
        }

        if let crate::graph::OpKind::Conv2d { stride, padding } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let weight = self.get_tensor(node.inputs[1])?;
            let in_dims = input.meta.layout.shape.dims();
            let w_dims = weight.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let uint_params: Vec<u32> = vec![
                in_dims[0] as u32, in_dims[1] as u32, w_dims[0] as u32,
                in_dims[2] as u32, in_dims[3] as u32,
                out_dims[2] as u32, out_dims[3] as u32,
                w_dims[2] as u32, w_dims[3] as u32,
                stride.0 as u32, stride.1 as u32,
                padding.0 as u32, padding.1 as u32,
            ];
            let grid_y = out_dims[2] as u32 * w_dims[0] as u32;
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("conv2d", dtype);
            return REGISTRY.dispatch_cnn_3d_nb(
                device, &k_src, &k_fn, queue,
                &[&input.buffer, &weight.buffer], &out.buffer,
                &uint_params, &[], (out_dims[3] as u32, grid_y, in_dims[0] as u32),
            );
        }

        if let crate::graph::OpKind::BatchNorm { eps } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let mean = self.get_tensor(node.inputs[1])?;
            let var = self.get_tensor(node.inputs[2])?;
            let weight = self.get_tensor(node.inputs[3])?;
            let bias = self.get_tensor(node.inputs[4])?;
            let in_dims = input.meta.layout.shape.dims();
            let batch = in_dims[0];
            let channels = in_dims[1];
            let spatial: usize = in_dims[2..].iter().product();
            let uint_params: Vec<u32> = vec![batch as u32, channels as u32, spatial as u32];
            let float_params: Vec<f32> = vec![eps];
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("batch_norm", dtype);
            return REGISTRY.dispatch_cnn_3d_nb(
                device, &k_src, &k_fn, queue,
                &[&input.buffer, &mean.buffer, &var.buffer, &weight.buffer, &bias.buffer],
                &out.buffer, &uint_params, &float_params,
                (spatial as u32, channels as u32, batch as u32),
            );
        }

        if let crate::graph::OpKind::MaxPool2d { kernel_size, stride, padding } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let in_dims = input.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let uint_params: Vec<u32> = vec![
                in_dims[0] as u32, in_dims[1] as u32, in_dims[2] as u32, in_dims[3] as u32,
                out_dims[2] as u32, out_dims[3] as u32,
                kernel_size.0 as u32, kernel_size.1 as u32,
                stride.0 as u32, stride.1 as u32,
                padding.0 as u32, padding.1 as u32,
            ];
            let grid_y = out_dims[2] as u32 * in_dims[1] as u32;
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("max_pool2d", dtype);
            return REGISTRY.dispatch_cnn_3d_nb(
                device, &k_src, &k_fn, queue,
                &[&input.buffer], &out.buffer,
                &uint_params, &[], (out_dims[3] as u32, grid_y, in_dims[0] as u32),
            );
        }

        if let crate::graph::OpKind::AvgPool2d { kernel_size, stride, padding } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let in_dims = input.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let uint_params: Vec<u32> = vec![
                in_dims[0] as u32, in_dims[1] as u32, in_dims[2] as u32, in_dims[3] as u32,
                out_dims[2] as u32, out_dims[3] as u32,
                kernel_size.0 as u32, kernel_size.1 as u32,
                stride.0 as u32, stride.1 as u32,
                padding.0 as u32, padding.1 as u32,
            ];
            let grid_y = out_dims[2] as u32 * in_dims[1] as u32;
            let (k_src, k_fn) = KernelRegistry::resolve_kernel("avg_pool2d", dtype);
            return REGISTRY.dispatch_cnn_3d_nb(
                device, &k_src, &k_fn, queue,
                &[&input.buffer], &out.buffer,
                &uint_params, &[], (out_dims[3] as u32, grid_y, in_dims[0] as u32),
            );
        }

        if let crate::graph::OpKind::Gather { dim } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let indices = self.get_tensor(node.inputs[1])?;
            let in_dims = input.meta.layout.shape.dims();
            let out_dims = node.out_shape.dims();
            let kernel_base = if dim == 0 { "gather_dim0" } else { "gather_dim1" };
            return REGISTRY.dispatch_gather_typed_nb(
                device, dtype, kernel_base, queue,
                &input.buffer, &indices.buffer, &out.buffer,
                out_dims[0], in_dims[1], out_dims[1],
            );
        }

        if let crate::graph::OpKind::IndexSelect { dim } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let indices = self.get_tensor(node.inputs[1])?;
            let in_dims = input.meta.layout.shape.dims();
            let num_indices = indices.meta.layout.shape.numel();
            if dim == 0 {
                return REGISTRY.dispatch_index_select_dim0_typed_nb(
                    device, dtype, queue,
                    &input.buffer, &indices.buffer, &out.buffer,
                    num_indices, in_dims[1],
                );
            } else {
                return REGISTRY.dispatch_index_select_dim1_typed_nb(
                    device, dtype, queue,
                    &input.buffer, &indices.buffer, &out.buffer,
                    in_dims[0], in_dims[1], num_indices,
                );
            }
        }

        if node.op.is_reshape() {
            let input = self.get_tensor(node.inputs[0])?;
            let size_bytes = input.meta.size_bytes();
            let cb = unsafe {
                ffi::gpu_bridge_blit_copy_nb(
                    device.raw_handle() as *mut _,
                    queue,
                    input.buffer.raw_handle(),
                    out.buffer.raw_handle(),
                    size_bytes as u64,
                )
            };
            if cb.is_null() {
                return Err(GpuError::ComputeFailed("Blit copy failed".to_string()));
            }
            return Ok(cb);
        }

        if let crate::graph::OpKind::Slice { dim, start, .. } = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let in_dims = input.meta.layout.shape.dims();
            if dim == 0 {
                let cols = in_dims[1];
                let out_rows = node.out_shape.dims()[0];
                return REGISTRY.dispatch_slice_dim0_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, cols, start, out_rows);
            } else {
                let rows = in_dims[0];
                let in_cols = in_dims[1];
                let out_cols = node.out_shape.dims()[1];
                return REGISTRY.dispatch_slice_dim1_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, in_cols, out_cols, start, rows);
            }
        }

        if let crate::graph::OpKind::Concat { dim } = node.op {
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            let a_dims = a.meta.layout.shape.dims();
            if dim == 0 {
                let rows_a = a_dims[0];
                let cols = a_dims[1];
                let total_rows = node.out_shape.dims()[0];
                return REGISTRY.dispatch_concat_dim0_typed_nb(device, dtype, queue, &a.buffer, &b.buffer, &out.buffer, rows_a, cols, total_rows);
            } else {
                let rows = a_dims[0];
                let cols_a = a_dims[1];
                let cols_b = b.meta.layout.shape.dims()[1];
                return REGISTRY.dispatch_concat_dim1_typed_nb(device, dtype, queue, &a.buffer, &b.buffer, &out.buffer, rows, cols_a, cols_b);
            }
        }

        if node.op.is_add_bias() {
            let input = self.get_tensor(node.inputs[0])?;
            let bias = self.get_tensor(node.inputs[1])?;
            let dims = input.meta.layout.shape.dims();
            let (rows, cols) = (dims[0], dims[1]);
            return REGISTRY.dispatch_add_bias_typed_nb(device, dtype, queue, &input.buffer, &bias.buffer, &out.buffer, rows, cols);
        }

        if node.op.is_softmax_causal() {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let ndim = dims.len();
            let rows = dims[ndim - 2];
            let cols = dims[ndim - 1];
            let batch_size: usize = dims[..ndim - 2].iter().product::<usize>().max(1);
            return REGISTRY.dispatch_softmax_causal_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, batch_size, rows, cols);
        }

        if node.op.is_argmax() {
            let input = self.get_tensor(node.inputs[0])?;
            let input_dtype = input.meta.dtype;
            let in_dims = input.meta.layout.shape.dims();
            let (rows, cols) = if in_dims.len() == 2 {
                (in_dims[0], in_dims[1])
            } else {
                (1, in_dims[0])
            };
            return REGISTRY.dispatch_argmax_typed_nb(device, input_dtype, queue, &input.buffer, &out.buffer, rows, cols);
        }

        if node.op.is_sum() || node.op.is_mean() {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.layout.shape.dims();
            let cols = dims[dims.len() - 1];
            let total_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
            if node.op.is_sum() {
                return REGISTRY.dispatch_sum_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, total_rows, cols);
            } else {
                return REGISTRY.dispatch_mean_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, total_rows, cols);
            }
        }

        if let crate::graph::OpKind::Cast { target_dtype } = node.op {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let input = self.get_tensor(node.inputs[0])?;
            let src_dtype = input.meta.dtype;
            let (source, func) = KernelRegistry::resolve_cast_kernel(src_dtype, target_dtype);
            let pipeline = REGISTRY.get_or_create(device, &source, &func)?;
            return pipeline.dispatch_unary_nd_nb(queue, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel);
        }

        if let crate::graph::OpKind::Quantize { scale, zero_point, target_dtype } = node.op {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let input = self.get_tensor(node.inputs[0])?;
            let src_dtype = input.meta.dtype;
            let (source, func) = KernelRegistry::resolve_quantize_kernel(src_dtype, target_dtype, scale, zero_point);
            let pipeline = REGISTRY.get_or_create(device, &source, &func)?;
            return pipeline.dispatch_unary_nd_nb(queue, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel);
        }

        if let crate::graph::OpKind::Dequantize { scale, zero_point, target_dtype } = node.op {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let input = self.get_tensor(node.inputs[0])?;
            let src_dtype = input.meta.dtype;
            let (source, func) = KernelRegistry::resolve_dequantize_kernel(src_dtype, target_dtype, scale, zero_point);
            let pipeline = REGISTRY.get_or_create(device, &source, &func)?;
            return pipeline.dispatch_unary_nd_nb(queue, &input.buffer, &in_strides, &out.buffer, &out_shape_u32, ndim, numel);
        }

        if node.op.is_unary() {
            let (in_strides, out_shape_u32, ndim, numel) = self.unary_nd_params(node)?;
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_unary_nd_typed_nb(
                device,
                node.op.kernel_name(),
                dtype,
                queue,
                &input.buffer,
                &in_strides,
                &out.buffer,
                &out_shape_u32,
                ndim,
                numel,
            )
        } else if node.op.is_matmul() {
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            let a_dims = a.meta.layout.shape.dims();
            let b_dims = b.meta.layout.shape.dims();
            let a_ndim = a_dims.len();
            let b_ndim = b_dims.len();
            let m = a_dims[a_ndim - 2];
            let k = a_dims[a_ndim - 1];
            let n = b_dims[b_ndim - 1];
            let out_dims = node.out_shape.dims();
            let out_ndim = out_dims.len();
            let batch_size: usize = out_dims[..out_ndim - 2].iter().product::<usize>().max(1);
            let a_batch_stride = if a_ndim > 2 { m * k } else { 0 };
            let b_batch_stride = if b_ndim > 2 { k * n } else { 0 };
            if batch_size <= 1 {
                REGISTRY.dispatch_matmul_typed_nb(device, dtype, queue, &a.buffer, &b.buffer, &out.buffer, m, n, k)
            } else {
                REGISTRY.dispatch_matmul_batched_typed_nb(device, dtype, queue, &a.buffer, &b.buffer, &out.buffer, m, n, k, batch_size, a_batch_stride, b_batch_stride)
            }
        } else {
            // Binary element-wise (N-D with broadcasting)
            let (a_strides, b_strides, out_shape_u32, ndim, numel) = self.binary_nd_params(node)?;
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            REGISTRY.dispatch_binary_nd_typed_nb(
                device,
                node.op.kernel_name(),
                dtype,
                queue,
                &a.buffer,
                &a_strides,
                &b.buffer,
                &b_strides,
                &out.buffer,
                &out_shape_u32,
                ndim,
                numel,
            )
        }
    }

    /// Convert a [usize; MAX_DIMS] array to [u32; MAX_DIMS] for Metal dispatch.
    fn to_u32_array(arr: &[usize; MAX_DIMS]) -> [u32; MAX_DIMS] {
        let mut out = [0u32; MAX_DIMS];
        for i in 0..MAX_DIMS {
            out[i] = arr[i] as u32;
        }
        out
    }

    /// Convert a Shape's dims to [u32; MAX_DIMS] for Metal dispatch.
    fn shape_to_u32(shape: &Shape) -> [u32; MAX_DIMS] {
        Self::to_u32_array(&shape.dims)
    }

    /// Compute contiguous strides for the input, then swap dim0/dim1
    /// so that reading the input with these strides produces the transposed output.
    fn transposed_strides(&self, in_dims: &[usize], dim0: usize, dim1: usize) -> [u32; MAX_DIMS] {
        let ndim = in_dims.len();
        // Compute contiguous strides for the input
        let mut strides = [0usize; MAX_DIMS];
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * in_dims[i + 1];
            }
        }
        // Swap strides for dim0 and dim1
        strides.swap(dim0, dim1);
        let mut out = [0u32; MAX_DIMS];
        for i in 0..MAX_DIMS {
            out[i] = strides[i] as u32;
        }
        out
    }

    /// Compute broadcast strides for a binary op, converting to u32.
    fn binary_nd_params(
        &self, node: &OpNode,
    ) -> Result<([u32; MAX_DIMS], [u32; MAX_DIMS], [u32; MAX_DIMS], u32, u32)> {
        let a_shape = {
            if let Some(t) = self.tensors.get(&node.inputs[0]) {
                t.meta.layout.shape
            } else if let Some(n) = self.graph.get_node(node.inputs[0]) {
                n.out_shape
            } else {
                return Err(GpuError::GraphError(format!("Tensor {} not found", node.inputs[0])));
            }
        };
        let b_shape = {
            if let Some(t) = self.tensors.get(&node.inputs[1]) {
                t.meta.layout.shape
            } else if let Some(n) = self.graph.get_node(node.inputs[1]) {
                n.out_shape
            } else {
                return Err(GpuError::GraphError(format!("Tensor {} not found", node.inputs[1])));
            }
        };

        let a_strides = TensorLayout::broadcast_strides_for(&a_shape, &node.out_shape);
        let b_strides = TensorLayout::broadcast_strides_for(&b_shape, &node.out_shape);

        let a_strides_u32 = Self::to_u32_array(&a_strides);
        let b_strides_u32 = Self::to_u32_array(&b_strides);
        let out_shape_u32 = Self::shape_to_u32(&node.out_shape);
        let ndim = node.out_shape.ndim() as u32;
        let numel = node.out_shape.numel() as u32;

        Ok((a_strides_u32, b_strides_u32, out_shape_u32, ndim, numel))
    }

    /// Compute contiguous strides for a unary op input, converting to u32.
    fn unary_nd_params(
        &self, node: &OpNode,
    ) -> Result<([u32; MAX_DIMS], [u32; MAX_DIMS], u32, u32)> {
        let in_shape = {
            if let Some(t) = self.tensors.get(&node.inputs[0]) {
                t.meta.layout.shape
            } else if let Some(n) = self.graph.get_node(node.inputs[0]) {
                n.out_shape
            } else {
                return Err(GpuError::GraphError(format!("Tensor {} not found", node.inputs[0])));
            }
        };

        let in_strides = TensorLayout::broadcast_strides_for(&in_shape, &node.out_shape);
        let in_strides_u32 = Self::to_u32_array(&in_strides);
        let out_shape_u32 = Self::shape_to_u32(&node.out_shape);
        let ndim = node.out_shape.ndim() as u32;
        let numel = node.out_shape.numel() as u32;

        Ok((in_strides_u32, out_shape_u32, ndim, numel))
    }

    /// Get a reference to a materialized tensor. Errors if not materialized.
    fn get_tensor(&self, id: u64) -> Result<&Tensor> {
        self.tensors.get(&id).ok_or_else(|| {
            GpuError::GraphError(format!(
                "Tensor {} not materialized (eval required first)",
                id
            ))
        })
    }

    /// Get the dtype of a tensor (materialized or pending).
    pub fn dtype(&self, id: u64) -> Result<DType> {
        if let Some(t) = self.tensors.get(&id) {
            return Ok(t.meta.dtype);
        }
        if let Some(node) = self.graph.get_node(id) {
            return Ok(node.out_dtype);
        }
        Err(GpuError::GraphError(format!("Tensor {} not found", id)))
    }

    /// Read tensor data as raw bytes. Requires the tensor to be materialized.
    pub fn read_bytes(&self, id: u64) -> Result<Vec<u8>> {
        let t = self.get_tensor(id)?;
        Ok(t.as_bytes()?.to_vec())
    }

    /// Read tensor data as f16 slice (raw u16 bit patterns). Requires the tensor to be materialized.
    pub fn read_f16(&self, id: u64) -> Result<Vec<u16>> {
        let t = self.get_tensor(id)?;
        Ok(t.as_f16_slice()?.to_vec())
    }

    /// Read tensor data as f32 slice. Requires the tensor to be materialized.
    pub fn read_f32(&self, id: u64) -> Result<Vec<f32>> {
        let t = self.get_tensor(id)?;
        Ok(t.as_f32_slice()?.to_vec())
    }

    /// Evaluate a tensor via the remote GPU service (VM backend).
    /// Serializes the graph + input tensors, sends over IPC, receives result.
    pub fn eval_remote(&mut self, device: &Device, id: u64, socket_path: &str) -> Result<()> {
        if self.is_materialized(id) {
            return Ok(());
        }

        let order = self.graph.topo_sort(id)?;
        if order.is_empty() {
            return Err(GpuError::GraphError(format!("Tensor {} not found", id)));
        }

        // Collect input tensors needed by the graph
        let mut tensor_data = Vec::new();
        let mut needed_tensors = std::collections::HashSet::new();
        for &node_id in &order {
            if let Some(node) = self.graph.get_node(node_id) {
                for &input_id in &node.inputs {
                    if !self.graph.has_node(input_id) && !needed_tensors.contains(&input_id) {
                        needed_tensors.insert(input_id);
                        if let Some(t) = self.tensors.get(&input_id) {
                            let data = t.as_f32_slice()?;
                            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                            tensor_data.push(crate::serial::TensorData {
                                id: input_id,
                                shape: t.meta.layout.shape.dims().to_vec(),
                                dtype: t.meta.dtype,
                                data: bytes,
                            });
                        }
                    }
                }
            }
        }

        // Collect graph nodes (unfused — fusion runs server-side)
        let nodes: Vec<crate::graph::OpNode> = order.iter()
            .filter_map(|&nid| self.graph.get_node(nid).cloned())
            .collect();

        let request = crate::serial::EvalRequest {
            target_id: id,
            tensors: tensor_data,
            nodes,
        };

        // Send to GPU service
        let response = crate::ipc::eval_remote(socket_path, &request)?;

        match response {
            crate::serial::EvalResponse::Ok { tensor_id, shape, data } => {
                let buffer = crate::buffer::Buffer::from_bytes(device, &data)?;
                let size = shape.iter().product::<usize>() * 4;
                let tensor = Tensor::from_raw(tensor_id, shape, DType::Float32, buffer);
                let container_id = self.resolve_container(id);
                self.scheduler.allocate_tensor(container_id, tensor_id, size)?;
                self.tensors.insert(tensor_id, tensor);

                // Remove evaluated nodes from graph
                for &nid in &order {
                    self.graph.remove_node(nid);
                }

                Ok(())
            }
            crate::serial::EvalResponse::Err(msg) => {
                Err(GpuError::ComputeFailed(format!("Remote eval failed: {}", msg)))
            }
        }
    }

    /// Destroy a tensor, freeing its GPU buffer.
    /// Errors if any pending graph node depends on this tensor.
    pub fn destroy(&mut self, id: u64) -> Result<()> {
        for node in self.graph.iter_nodes() {
            if node.inputs.contains(&id) {
                return Err(GpuError::GraphError(format!(
                    "Cannot destroy tensor {} while pending op {} depends on it",
                    id, node.id
                )));
            }
        }
        if let Some(tensor) = self.tensors.remove(&id) {
            self.scheduler.free_tensor(id, tensor.meta.size_bytes());
            self.pool.release(tensor.into_buffer());
        }
        self.graph.remove_node(id);
        Ok(())
    }

    pub fn memory_usage(&self) -> usize {
        self.scheduler.global_usage().0
    }

    pub fn live_tensor_count(&self) -> usize {
        self.scheduler.global_usage().1
    }

    pub fn set_limits(&mut self, limits: ResourceLimits) {
        self.scheduler.update_global_limits(limits);
        self.pool.drain();
    }

    /// Clean up all resources belonging to a container.
    /// Removes tensors, graph nodes, and deregisters from the scheduler.
    pub fn cleanup_container(&mut self, container_id: ContainerId) -> Result<()> {
        let owned_tensors = self.scheduler.deregister_container(container_id)?;
        for tid in &owned_tensors {
            self.remove_tensor_raw(*tid);
        }
        self.graph.remove_nodes_for_container(container_id);
        Ok(())
    }

    /// Remove a tensor without scheduler tracking (for deregister cleanup).
    pub fn remove_tensor_raw(&mut self, id: u64) {
        if let Some(tensor) = self.tensors.remove(&id) {
            self.pool.release(tensor.into_buffer());
        }
        self.graph.remove_node(id);
    }

    /// Dequeue the next scheduled job, evaluate it, and complete it.
    pub fn run_next(&mut self, device: &Device) -> Result<Option<crate::scheduler::JobId>> {
        let job = match self.scheduler.next_job() {
            Some(j) => j,
            None => return Ok(None),
        };

        let job_id = job.id;
        let target_id = job.target_tensor_id;
        let start = std::time::Instant::now();

        match self.eval(device, target_id) {
            Ok(()) => {
                let elapsed = start.elapsed().as_nanos() as u64;
                self.scheduler.complete_job(job_id, elapsed)?;
                Ok(Some(job_id))
            }
            Err(e) => {
                self.scheduler.fail_job(job_id, e.to_string())?;
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpKind;
    use crate::scheduler::ContainerId;
    use crate::tensor::{DType, Shape};

    fn get_device() -> Option<Device> {
        Device::new().ok()
    }

    #[test]
    fn insert_tensor_tracks_logical_size() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let logical = t.meta.size_bytes();
        rt.insert_tensor(t).unwrap();
        let (bytes, _) = rt.scheduler.container_usage(ContainerId::DEFAULT).unwrap();
        assert_eq!(bytes, logical);
    }

    #[test]
    fn lazy_input_tensor_is_materialized() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        assert!(rt.is_materialized(id));
        assert!(!rt.is_pending(id));
    }

    #[test]
    fn lazy_add_defers_execution() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();

        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(500_000);
        let c_id = COUNTER.fetch_add(1, Ordering::Relaxed);

        rt.record_op(OpNode {
            id: c_id,
            op: OpKind::Add,
            inputs: vec![a_id, b_id],
            out_shape: Shape::new(vec![4]).unwrap(),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        assert!(rt.is_pending(c_id));
        assert!(!rt.is_materialized(c_id));

        rt.eval(&device, c_id).unwrap();

        assert!(rt.is_materialized(c_id));
        assert!(!rt.is_pending(c_id));
        assert_eq!(rt.read_f32(c_id).unwrap(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn lazy_chain_defers_until_eval() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![4], &[1.0, -2.0, 3.0, -4.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();

        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(600_000);

        let neg_id = COUNTER.fetch_add(1, Ordering::Relaxed);
        rt.record_op(OpNode {
            id: neg_id,
            op: OpKind::Neg,
            inputs: vec![a_id],
            out_shape: Shape::new(vec![4]).unwrap(),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let relu_id = COUNTER.fetch_add(1, Ordering::Relaxed);
        rt.record_op(OpNode {
            id: relu_id,
            op: OpKind::Relu,
            inputs: vec![neg_id],
            out_shape: Shape::new(vec![4]).unwrap(),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        assert!(rt.is_pending(neg_id));
        assert!(rt.is_pending(relu_id));

        rt.eval(&device, relu_id).unwrap();

        // relu(neg([1,-2,3,-4])) = relu([-1,2,-3,4]) = [0,2,0,4]
        assert_eq!(rt.read_f32(relu_id).unwrap(), &[0.0, 2.0, 0.0, 4.0]);
        // Note: neg_id may or may not be materialized — fusion may combine neg+relu
        // into a single kernel, skipping the intermediate. The result is correct either way.
    }

    #[test]
    fn lazy_destroy_frees_tensor() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        assert!(rt.exists(id));
        rt.destroy(id).unwrap();
        assert!(!rt.exists(id));
    }

    #[test]
    fn test_run_next_dequeue_eval_complete() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();

        let c_id = crate::ops::add(&mut rt, a_id, b_id).unwrap();

        let job_id = rt.scheduler.submit(ContainerId::DEFAULT, c_id).unwrap();
        assert_eq!(rt.scheduler.queue_depth(), 1);

        let result = rt.run_next(&device).unwrap();
        assert_eq!(result, Some(job_id));
        assert!(rt.is_materialized(c_id));
        assert_eq!(rt.read_f32(c_id).unwrap(), &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(rt.scheduler.queue_depth(), 0);
    }
}
