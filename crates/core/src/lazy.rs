use std::collections::HashMap;

use crate::compute::KernelRegistry;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::graph::{Graph, OpNode};
use crate::limits::ResourceLimits;
use crate::pool::BufferPool;
use crate::scheduler::{ContainerId, Scheduler};
use crate::tensor::{DType, Tensor};

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
            return Ok(t.meta.shape.dims().to_vec());
        }
        if let Some(node) = self.graph.get_node(id) {
            return Ok(node.out_shape.dims().to_vec());
        }
        Err(GpuError::GraphError(format!("Tensor {} not found", id)))
    }

    /// Evaluate a tensor: execute all pending ops needed to materialize it.
    /// After evaluation, the tensor is in `self.tensors` and its graph nodes are removed.
    ///
    /// Uses command buffer batching: all ops are submitted non-blocking to a
    /// shared command queue, then we wait only on the last command buffer.
    /// Metal guarantees in-order execution on the same queue, so all prior
    /// command buffers complete before the last one.
    pub fn eval(&mut self, device: &Device, id: u64) -> Result<()> {
        if self.is_materialized(id) {
            return Ok(()); // already done
        }

        let container_id = self.resolve_container(id);

        let mut order = self.graph.topo_sort(id)?;
        if order.is_empty() {
            return Err(GpuError::GraphError(format!("Tensor {} not found", id)));
        }

        // Run fusion optimization pass
        order = crate::fusion::optimize(&mut self.graph, &order);

        let queue = crate::compute::get_shared_queue(device);
        let mut last_cb: Option<*mut std::ffi::c_void> = None;

        for node_id in order {
            if self.is_materialized(node_id) {
                continue; // already evaluated (shared subexpression)
            }

            let node = self.graph.remove_node(node_id).ok_or_else(|| {
                GpuError::GraphError(format!("Node {} not found in graph", node_id))
            })?;

            let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);

            let cb = self.execute_node_nb(device, queue, &node, &out)?;
            last_cb = Some(cb);

            let size = node.out_shape.numel() * node.out_dtype.size_bytes();
            self.scheduler.allocate_tensor(container_id, node_id, size)?;
            self.tensors.insert(node_id, out);
        }

        // Wait only on the last command buffer — Metal in-order queue guarantees
        // all prior submissions are complete when this one finishes.
        if let Some(cb) = last_cb {
            crate::compute::wait_command_buffer(cb);
        }

        Ok(())
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
        let out_size = node.out_shape.numel() * node.out_dtype.size_bytes();

        // Handle fused kernels first
        if let crate::graph::OpKind::FusedElementwise { ref kernel_source, ref function_name } = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input_tensors: Vec<&Tensor> = node.inputs.iter()
                .map(|&id| self.get_tensor(id))
                .collect::<Result<Vec<_>>>()?;
            let input_buffers: Vec<&crate::buffer::Buffer> = input_tensors.iter()
                .map(|t| &t.buffer)
                .collect();
            let numel = input_tensors[0].numel();
            REGISTRY.dispatch_fused(
                device,
                kernel_source,
                function_name,
                &input_buffers,
                &out.buffer,
                numel,
            )?;
            return Ok(out);
        }

        let dtype = node.out_dtype;

        if node.op.is_softmax() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.shape.dims();
            let (rows, cols) = (dims[0], dims[1]);
            REGISTRY.dispatch_softmax_typed(device, dtype, &input.buffer, &out.buffer, rows, cols)?;
            return Ok(out);
        }

        if node.op.is_transpose() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.shape.dims();
            let (rows, cols) = (dims[0], dims[1]);
            REGISTRY.dispatch_transpose_typed(device, dtype, &input.buffer, &out.buffer, rows, cols)?;
            return Ok(out);
        }

        if let crate::graph::OpKind::ScalarMul(scale) = node.op {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_scalar_mul_typed(device, dtype, &input.buffer, &out.buffer, scale, input.numel())?;
            return Ok(out);
        }

        if node.op.is_unary() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_unary_typed(
                device,
                node.op.kernel_name(),
                dtype,
                &input.buffer,
                &out.buffer,
                input.numel(),
            )?;
            Ok(out)
        } else if node.op.is_matmul() {
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            let a_dims = a.meta.shape.dims();
            let b_dims = b.meta.shape.dims();
            let (m, k) = (a_dims[0], a_dims[1]);
            let n = b_dims[1];
            REGISTRY.dispatch_matmul_typed(device, dtype, &a.buffer, &b.buffer, &out.buffer, m, n, k)?;
            Ok(out)
        } else {
            // Binary element-wise
            let out_buf = self.pool.acquire(device, out_size)?;
            let out = Tensor::from_raw(node.id, node.out_shape.dims().to_vec(), node.out_dtype, out_buf);
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            REGISTRY.dispatch_binary_typed(
                device,
                node.op.kernel_name(),
                dtype,
                &a.buffer,
                &b.buffer,
                &out.buffer,
                a.numel(),
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
        // Handle fused kernels first
        if let crate::graph::OpKind::FusedElementwise { ref kernel_source, ref function_name } = node.op {
            let input_tensors: Vec<&Tensor> = node.inputs.iter()
                .map(|&id| self.get_tensor(id))
                .collect::<Result<Vec<_>>>()?;
            let input_buffers: Vec<&crate::buffer::Buffer> = input_tensors.iter()
                .map(|t| &t.buffer)
                .collect();
            let numel = input_tensors[0].numel();
            return REGISTRY.dispatch_fused_nb(
                device,
                kernel_source,
                function_name,
                queue,
                &input_buffers,
                &out.buffer,
                numel,
            );
        }

        let dtype = node.out_dtype;

        if node.op.is_softmax() {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.shape.dims();
            let (rows, cols) = (dims[0], dims[1]);
            return REGISTRY.dispatch_softmax_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, rows, cols);
        }

        if node.op.is_transpose() {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.shape.dims();
            let (rows, cols) = (dims[0], dims[1]);
            return REGISTRY.dispatch_transpose_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, rows, cols);
        }

        if let crate::graph::OpKind::ScalarMul(scale) = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            return REGISTRY.dispatch_scalar_mul_typed_nb(device, dtype, queue, &input.buffer, &out.buffer, scale, input.numel());
        }

        if node.op.is_unary() {
            let input = self.get_tensor(node.inputs[0])?;
            REGISTRY.dispatch_unary_typed_nb(
                device,
                node.op.kernel_name(),
                dtype,
                queue,
                &input.buffer,
                &out.buffer,
                input.numel(),
            )
        } else if node.op.is_matmul() {
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            let a_dims = a.meta.shape.dims();
            let b_dims = b.meta.shape.dims();
            let (m, k) = (a_dims[0], a_dims[1]);
            let n = b_dims[1];
            REGISTRY.dispatch_matmul_typed_nb(device, dtype, queue, &a.buffer, &b.buffer, &out.buffer, m, n, k)
        } else {
            // Binary element-wise
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            REGISTRY.dispatch_binary_typed_nb(
                device,
                node.op.kernel_name(),
                dtype,
                queue,
                &a.buffer,
                &b.buffer,
                &out.buffer,
                a.numel(),
            )
        }
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

    /// Read tensor data as f16 slice (raw u16 bit patterns). Requires the tensor to be materialized.
    pub fn read_f16(&self, id: u64) -> Result<Vec<u16>> {
        let t = self.get_tensor(id)?;
        Ok(t.as_f16_slice().to_vec())
    }

    /// Read tensor data as f32 slice. Requires the tensor to be materialized.
    pub fn read_f32(&self, id: u64) -> Result<Vec<f32>> {
        let t = self.get_tensor(id)?;
        Ok(t.as_f32_slice().to_vec())
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
                            let data = t.as_f32_slice();
                            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                            tensor_data.push(crate::serial::TensorData {
                                id: input_id,
                                shape: t.meta.shape.dims().to_vec(),
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
            out_shape: Shape::new(vec![4]),
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
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let relu_id = COUNTER.fetch_add(1, Ordering::Relaxed);
        rt.record_op(OpNode {
            id: relu_id,
            op: OpKind::Relu,
            inputs: vec![neg_id],
            out_shape: Shape::new(vec![4]),
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
