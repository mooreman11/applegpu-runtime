use std::collections::HashMap;

use crate::compute::KernelRegistry;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::graph::{Graph, OpNode};
use crate::limits::{MemoryTracker, ResourceLimits};
use crate::tensor::Tensor;

use once_cell::sync::Lazy;

static REGISTRY: Lazy<KernelRegistry> = Lazy::new(KernelRegistry::new);

/// Storage for lazy tensors — holds both materialized data and pending graph.
pub struct LazyRuntime {
    /// Materialized tensors (have actual GPU buffers).
    tensors: HashMap<u64, Tensor>,
    /// Pending computation graph.
    graph: Graph,
    /// Resource limits.
    pub limits: ResourceLimits,
    /// Memory usage tracker.
    pub tracker: MemoryTracker,
}

impl LazyRuntime {
    pub fn new() -> Self {
        LazyRuntime {
            tensors: HashMap::new(),
            graph: Graph::new(),
            limits: ResourceLimits::from_env(),
            tracker: MemoryTracker::new(),
        }
    }

    /// Store a materialized tensor (e.g., from user data).
    /// Checks resource limits before inserting.
    pub fn insert_tensor(&mut self, tensor: Tensor) -> Result<()> {
        let size = tensor.buffer.len();
        self.tracker.check_allocation(size, &self.limits)?;
        self.tracker.track_alloc(size);
        self.tensors.insert(tensor.meta.id, tensor);
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
    pub fn eval(&mut self, device: &Device, id: u64) -> Result<()> {
        if self.is_materialized(id) {
            return Ok(()); // already done
        }

        let mut order = self.graph.topo_sort(id)?;
        if order.is_empty() {
            return Err(GpuError::GraphError(format!("Tensor {} not found", id)));
        }

        // Run fusion optimization pass
        order = crate::fusion::optimize(&mut self.graph, &order);

        for node_id in order {
            if self.is_materialized(node_id) {
                continue; // already evaluated (shared subexpression)
            }

            let node = self.graph.remove_node(node_id).ok_or_else(|| {
                GpuError::GraphError(format!("Node {} not found in graph", node_id))
            })?;

            let result = self.execute_node(device, &node)?;
            let size = result.buffer.len();
            self.tracker.check_allocation(size, &self.limits)?;
            self.tracker.track_alloc(size);
            self.tensors.insert(node_id, result);
        }

        Ok(())
    }

    /// Execute a single graph node, producing a materialized Tensor.
    fn execute_node(&self, device: &Device, node: &OpNode) -> Result<Tensor> {
        // Handle fused kernels first
        if let crate::graph::OpKind::FusedElementwise { ref kernel_source, ref function_name } = node.op {
            let input_tensors: Vec<&Tensor> = node.inputs.iter()
                .map(|&id| self.get_tensor(id))
                .collect::<Result<Vec<_>>>()?;
            let input_buffers: Vec<&crate::buffer::Buffer> = input_tensors.iter()
                .map(|t| &t.buffer)
                .collect();
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
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

        if node.op.is_softmax() {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.shape.dims();
            let (rows, cols) = (dims[0], dims[1]);
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
            REGISTRY.dispatch_softmax(device, &input.buffer, &out.buffer, rows, cols)?;
            return Ok(out);
        }

        if node.op.is_transpose() {
            let input = self.get_tensor(node.inputs[0])?;
            let dims = input.meta.shape.dims();
            let (rows, cols) = (dims[0], dims[1]);
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
            REGISTRY.dispatch_transpose(device, &input.buffer, &out.buffer, rows, cols)?;
            return Ok(out);
        }

        if let crate::graph::OpKind::ScalarMul(scale) = node.op {
            let input = self.get_tensor(node.inputs[0])?;
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
            REGISTRY.dispatch_scalar_mul(device, &input.buffer, &out.buffer, scale, input.numel())?;
            return Ok(out);
        }

        if node.op.is_unary() {
            let input = self.get_tensor(node.inputs[0])?;
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
            REGISTRY.dispatch_unary(
                device,
                node.op.kernel_name(),
                &input.buffer,
                &out.buffer,
                input.numel(),
            )?;
            Ok(out)
        } else if node.op.is_matmul() {
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            let a_dims = a.meta.shape.dims();
            let b_dims = b.meta.shape.dims();
            let (m, k) = (a_dims[0], a_dims[1]);
            let n = b_dims[1];
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
            REGISTRY.dispatch_matmul(device, &a.buffer, &b.buffer, &out.buffer, m, n, k)?;
            Ok(out)
        } else {
            // Binary element-wise
            let a = self.get_tensor(node.inputs[0])?;
            let b = self.get_tensor(node.inputs[1])?;
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
            REGISTRY.dispatch_binary(
                device,
                node.op.kernel_name(),
                &a.buffer,
                &b.buffer,
                &out.buffer,
                a.numel(),
            )?;
            Ok(out)
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
                let size = buffer.len();
                let tensor = Tensor::from_raw(tensor_id, shape, buffer);
                self.tracker.track_alloc(size);
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
            self.tracker.track_free(tensor.buffer.len());
        }
        self.graph.remove_node(id);
        Ok(())
    }

    pub fn memory_usage(&self) -> usize {
        self.tracker.memory_usage()
    }

    pub fn live_tensor_count(&self) -> usize {
        self.tracker.tensor_count()
    }

    pub fn set_limits(&mut self, limits: ResourceLimits) {
        self.limits = limits;
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
}
