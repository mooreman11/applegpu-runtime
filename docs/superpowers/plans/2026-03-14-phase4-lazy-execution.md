# Phase 4: Lazy Execution & Graph Capture Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace eager GPU execution with lazy graph capture — ops build a DAG instead of dispatching immediately, and computation happens only when results are materialized via `gpu.eval()` or `gpu.to_list()`.

**Architecture:** A new `graph` module defines `OpNode` (operation + input refs + output metadata) and `Graph` (DAG of nodes). `LazyTensor` wraps either a materialized `Tensor` (for input data) or an unevaluated `OpNode` ID. When materialized, the graph is topologically sorted and executed in order, reusing the existing `KernelRegistry` for GPU dispatch. The Python API is unchanged — ops still return opaque u64 IDs, but now they're lazy until `to_list()` or `eval()` is called. Tensor cleanup (`gpu.destroy()`) is included since the new storage model makes it natural.

**Tech Stack:** Rust (DAG, topological sort), Python (PyO3), existing Metal compute infrastructure (unchanged)

**Breaking changes:** The `ops` module API changes from eager (`fn add(&Device, &Tensor, &Tensor) -> Result<Tensor>`) to lazy (`fn add(&mut LazyRuntime, u64, u64) -> Result<u64>`). No external Rust consumers exist today. Python API is backward compatible — `to_list()` auto-evaluates lazy tensors.

**ID space:** Input tensors use `TENSOR_ID_COUNTER` (starts at 1), lazy op nodes use `OP_ID_COUNTER` (starts at 100,000). When a lazy node is evaluated, the resulting `Tensor` is stored under the op's ID (not the tensor's auto-assigned `meta.id`). This is an internal detail — the Python layer only sees the op ID.

---

## File Structure

### New Files
- `crates/core/src/graph.rs` — `OpNode`, `OpKind`, `Graph` (DAG with topological sort and execution)
- `crates/core/src/lazy.rs` — `LazyTensor` enum (Materialized | Pending), materialization logic
- `python/tests/test_lazy.py` — Tests for lazy evaluation, graph capture, eval()

### Modified Files
- `crates/core/src/ops.rs` — Ops return `LazyTensor` instead of `Tensor`, delegate to `Graph`
- `crates/core/src/error.rs` — Add `GraphError` variant
- `crates/core/src/lib.rs` — Add graph and lazy modules
- `crates/python/src/lib.rs` — Replace `TENSORS` HashMap with lazy storage, add `eval()` and `destroy()`
- `python/applegpu_runtime/__init__.py` — Export `eval` and `destroy`
- `python/tests/test_compute.py` — Existing tests still pass (backward compatible)

---

## Chunk 1: Graph Data Structure

### Task 1: Define OpKind, OpNode, and Graph

**Files:**
- Create: `crates/core/src/graph.rs`
- Modify: `crates/core/src/error.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Add GraphError variant to error.rs**

In `crates/core/src/error.rs`, add to the `GpuError` enum:

```rust
    /// Graph evaluation error
    GraphError(String),
```

And update the `Display` impl:

```rust
            GpuError::GraphError(msg) => write!(f, "Graph error: {}", msg),
```

- [ ] **Step 2: Create graph.rs with OpKind, OpNode, and Graph**

Create `crates/core/src/graph.rs`:

```rust
use std::collections::HashMap;

use crate::tensor::{DType, Shape};

/// The kind of operation a graph node represents.
#[derive(Debug, Clone)]
pub enum OpKind {
    // Binary element-wise
    Add,
    Sub,
    Mul,
    Div,
    // Unary element-wise
    Neg,
    Relu,
    Exp,
    Log,
    Sqrt,
    // Matrix multiply
    Matmul,
}

impl OpKind {
    /// Map to the MSL kernel function name.
    pub fn kernel_name(&self) -> &'static str {
        match self {
            OpKind::Add => "elementwise_add",
            OpKind::Sub => "elementwise_sub",
            OpKind::Mul => "elementwise_mul",
            OpKind::Div => "elementwise_div",
            OpKind::Neg => "elementwise_neg",
            OpKind::Relu => "elementwise_relu",
            OpKind::Exp => "elementwise_exp",
            OpKind::Log => "elementwise_log",
            OpKind::Sqrt => "elementwise_sqrt",
            OpKind::Matmul => "matmul_f32",
        }
    }

    pub fn is_unary(&self) -> bool {
        matches!(self, OpKind::Neg | OpKind::Relu | OpKind::Exp | OpKind::Log | OpKind::Sqrt)
    }

    pub fn is_matmul(&self) -> bool {
        matches!(self, OpKind::Matmul)
    }
}

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Unique ID for this node (matches the LazyTensor ID).
    pub id: u64,
    /// The operation to perform.
    pub op: OpKind,
    /// Input tensor IDs (1 for unary, 2 for binary/matmul).
    pub inputs: Vec<u64>,
    /// Output shape (computed at graph-build time).
    pub out_shape: Shape,
    /// Output dtype.
    pub out_dtype: DType,
}

/// A computation graph (DAG of operations).
/// Nodes are stored by ID. Leaf tensors (materialized data) are not in the graph.
pub struct Graph {
    nodes: HashMap<u64, OpNode>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
        }
    }

    /// Add an operation node to the graph.
    pub fn add_node(&mut self, node: OpNode) {
        self.nodes.insert(node.id, node);
    }

    /// Check if a node ID exists in the graph (i.e., is a pending operation).
    pub fn has_node(&self, id: u64) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Remove a node from the graph (after it has been evaluated).
    pub fn remove_node(&mut self, id: u64) -> Option<OpNode> {
        self.nodes.remove(&id)
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: u64) -> Option<&OpNode> {
        self.nodes.get(&id)
    }

    /// Return topologically sorted node IDs needed to evaluate `target_id`.
    /// Only includes nodes that are pending (in the graph), not materialized tensors.
    /// Returns Err if a cycle is detected.
    pub fn topo_sort(&self, target_id: u64) -> crate::error::Result<Vec<u64>> {
        let mut visited = std::collections::HashSet::new();
        let mut in_stack = std::collections::HashSet::new();
        let mut order = Vec::new();
        self.topo_visit(target_id, &mut visited, &mut in_stack, &mut order)?;
        Ok(order)
    }

    fn topo_visit(
        &self,
        id: u64,
        visited: &mut std::collections::HashSet<u64>,
        in_stack: &mut std::collections::HashSet<u64>,
        order: &mut Vec<u64>,
    ) -> crate::error::Result<()> {
        if in_stack.contains(&id) {
            return Err(crate::error::GpuError::GraphError(
                format!("Cycle detected at node {}", id),
            ));
        }
        if visited.contains(&id) {
            return Ok(());
        }
        in_stack.insert(id);

        if let Some(node) = self.nodes.get(&id) {
            for &input_id in &node.inputs {
                self.topo_visit(input_id, visited, in_stack, order)?;
            }
            in_stack.remove(&id);
            visited.insert(id);
            order.push(id);
        }
        // If not in graph, it's a materialized tensor — skip (already available)
        Ok(())
    }

    /// Iterate over all nodes.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &OpNode> {
        self.nodes.values()
    }

    /// Number of pending nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_kind_kernel_names() {
        assert_eq!(OpKind::Add.kernel_name(), "elementwise_add");
        assert_eq!(OpKind::Matmul.kernel_name(), "matmul_f32");
        assert!(OpKind::Neg.is_unary());
        assert!(!OpKind::Add.is_unary());
        assert!(OpKind::Matmul.is_matmul());
    }

    #[test]
    fn graph_topo_sort_linear() {
        // a(1) -> add(3) -> neg(4)
        //         b(2) /
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3,
            op: OpKind::Add,
            inputs: vec![1, 2], // 1, 2 are materialized (not in graph)
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 4,
            op: OpKind::Neg,
            inputs: vec![3],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
        });

        let order = g.topo_sort(4).unwrap();
        assert_eq!(order, vec![3, 4]);
    }

    #[test]
    fn graph_topo_sort_diamond() {
        // a(1) -> add(3) -> mul(5)
        // b(2) -> sub(4) /
        // a(1) /
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3,
            op: OpKind::Add,
            inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 4,
            op: OpKind::Sub,
            inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 5,
            op: OpKind::Mul,
            inputs: vec![3, 4],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
        });

        let order = g.topo_sort(5).unwrap();
        // 3 and 4 must come before 5, order between 3 and 4 doesn't matter
        assert_eq!(order.len(), 3);
        assert_eq!(*order.last().unwrap(), 5);
        assert!(order.contains(&3));
        assert!(order.contains(&4));
    }

    #[test]
    fn graph_partial_eval() {
        // Only nodes needed for target are returned
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3,
            op: OpKind::Add,
            inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 4,
            op: OpKind::Neg,
            inputs: vec![1], // independent of node 3
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
        });

        let order = g.topo_sort(4).unwrap();
        assert_eq!(order, vec![4]); // only node 4, not node 3
    }
}
```

- [ ] **Step 3: Add graph module to lib.rs**

Add `pub mod graph;` to `crates/core/src/lib.rs`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p applegpu-core graph 2>&1`
Expected: 4 tests pass (op_kind_kernel_names, topo_sort_linear, topo_sort_diamond, partial_eval)

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/graph.rs crates/core/src/error.rs crates/core/src/lib.rs
git commit -m "feat: add computation graph with OpNode, OpKind, and topological sort"
```

---

## Chunk 2: LazyTensor and Graph Execution

### Task 2: Create LazyTensor and graph execution engine

**Files:**
- Create: `crates/core/src/lazy.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Create lazy.rs with LazyTensor and evaluation**

Create `crates/core/src/lazy.rs`:

```rust
use std::collections::HashMap;

use crate::compute::KernelRegistry;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::graph::{Graph, OpNode};
use crate::tensor::Tensor;

use once_cell::sync::Lazy;

static REGISTRY: Lazy<KernelRegistry> = Lazy::new(KernelRegistry::new);

/// Storage for lazy tensors — holds both materialized data and pending graph.
pub struct LazyRuntime {
    /// Materialized tensors (have actual GPU buffers).
    tensors: HashMap<u64, Tensor>,
    /// Pending computation graph.
    graph: Graph,
}

impl LazyRuntime {
    pub fn new() -> Self {
        LazyRuntime {
            tensors: HashMap::new(),
            graph: Graph::new(),
        }
    }

    /// Store a materialized tensor (e.g., from user data).
    pub fn insert_tensor(&mut self, tensor: Tensor) {
        self.tensors.insert(tensor.meta.id, tensor);
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

        let order = self.graph.topo_sort(id)?;
        if order.is_empty() {
            return Err(GpuError::GraphError(format!("Tensor {} not found", id)));
        }

        for node_id in order {
            if self.is_materialized(node_id) {
                continue; // already evaluated (shared subexpression)
            }

            let node = self.graph.remove_node(node_id).ok_or_else(|| {
                GpuError::GraphError(format!("Node {} not found in graph", node_id))
            })?;

            let result = self.execute_node(device, &node)?;
            self.tensors.insert(node_id, result);
        }

        Ok(())
    }

    /// Execute a single graph node, producing a materialized Tensor.
    fn execute_node(&self, device: &Device, node: &OpNode) -> Result<Tensor> {
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

    /// Destroy a tensor, freeing its GPU buffer.
    /// Errors if any pending graph node depends on this tensor.
    pub fn destroy(&mut self, id: u64) -> Result<()> {
        // Check no pending node depends on this tensor
        for node in self.graph.iter_nodes() {
            if node.inputs.contains(&id) {
                return Err(GpuError::GraphError(format!(
                    "Cannot destroy tensor {} while pending op {} depends on it",
                    id, node.id
                )));
            }
        }
        self.tensors.remove(&id);
        self.graph.remove_node(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> Option<Device> {
        Device::new().ok()
    }

    #[test]
    fn lazy_input_tensor_is_materialized() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t);
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
        rt.insert_tensor(a);
        rt.insert_tensor(b);

        // Record lazy add — no GPU execution yet
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1000);
        let c_id = COUNTER.fetch_add(1, Ordering::Relaxed);

        rt.record_op(OpNode {
            id: c_id,
            op: OpKind::Add,
            inputs: vec![a_id, b_id],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
        });

        assert!(rt.is_pending(c_id));
        assert!(!rt.is_materialized(c_id));

        // Evaluate
        rt.eval(&device, c_id).unwrap();

        assert!(rt.is_materialized(c_id));
        assert!(!rt.is_pending(c_id));
        assert_eq!(rt.read_f32(c_id).unwrap(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn lazy_chain_defers_until_eval() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a);

        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(2000);

        // neg(a) -> relu(neg(a))
        let neg_id = COUNTER.fetch_add(1, Ordering::Relaxed);
        rt.record_op(OpNode {
            id: neg_id,
            op: OpKind::Neg,
            inputs: vec![a_id],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
        });

        let relu_id = COUNTER.fetch_add(1, Ordering::Relaxed);
        rt.record_op(OpNode {
            id: relu_id,
            op: OpKind::Relu,
            inputs: vec![neg_id],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
        });

        // Neither is materialized yet
        assert!(rt.is_pending(neg_id));
        assert!(rt.is_pending(relu_id));

        // Evaluate the final node — should cascade through neg first
        rt.eval(&device, relu_id).unwrap();

        // relu(neg([1,2,3,4])) = relu([-1,-2,-3,-4]) = [0,0,0,0]
        assert_eq!(rt.read_f32(relu_id).unwrap(), &[0.0, 0.0, 0.0, 0.0]);
        // neg was also materialized as a side effect
        assert!(rt.is_materialized(neg_id));
    }

    #[test]
    fn lazy_destroy_frees_tensor() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t);
        assert!(rt.exists(id));
        rt.destroy(id).unwrap();
        assert!(!rt.exists(id));
    }
}
```

- [ ] **Step 2: Add lazy module to lib.rs**

Add `pub mod lazy;` to `crates/core/src/lib.rs`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p applegpu-core lazy 2>&1`
Expected: 4 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/lazy.rs crates/core/src/lib.rs
git commit -m "feat: add LazyRuntime with deferred execution and graph evaluation"
```

---

## Chunk 3: Rewire Ops and Python to Use Lazy Execution

### Task 3: Rewire ops.rs to build graph nodes instead of executing eagerly

**Files:**
- Modify: `crates/core/src/ops.rs`

- [ ] **Step 1: Rewrite ops.rs to return LazyTensor IDs via LazyRuntime**

The ops module needs to work with `LazyRuntime` instead of directly dispatching. Since `LazyRuntime` is stateful (owned by the Python layer), ops functions will take a `&mut LazyRuntime` parameter instead of using a global.

Replace `crates/core/src/ops.rs`:

```rust
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::graph::{OpKind, OpNode};
use crate::lazy::LazyRuntime;
use crate::tensor::{DType, Shape};

use std::sync::atomic::{AtomicU64, Ordering};

static OP_ID_COUNTER: AtomicU64 = AtomicU64::new(100_000);

fn next_id() -> u64 {
    OP_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Record a binary element-wise op in the graph.
fn lazy_binary_op(rt: &mut LazyRuntime, a_id: u64, b_id: u64, op: OpKind) -> Result<u64> {
    let a_shape = rt.shape(a_id)?;
    let b_shape = rt.shape(b_id)?;

    if a_shape != b_shape {
        return Err(GpuError::InvalidTensor(format!(
            "Shape mismatch: {:?} vs {:?}",
            a_shape, b_shape
        )));
    }

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op,
        inputs: vec![a_id, b_id],
        out_shape: Shape::new(a_shape),
        out_dtype: DType::Float32,
    });
    Ok(out_id)
}

/// Record a unary element-wise op in the graph.
fn lazy_unary_op(rt: &mut LazyRuntime, input_id: u64, op: OpKind) -> Result<u64> {
    let shape = rt.shape(input_id)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op,
        inputs: vec![input_id],
        out_shape: Shape::new(shape),
        out_dtype: DType::Float32,
    });
    Ok(out_id)
}

pub fn add(rt: &mut LazyRuntime, a_id: u64, b_id: u64) -> Result<u64> {
    lazy_binary_op(rt, a_id, b_id, OpKind::Add)
}

pub fn sub(rt: &mut LazyRuntime, a_id: u64, b_id: u64) -> Result<u64> {
    lazy_binary_op(rt, a_id, b_id, OpKind::Sub)
}

pub fn mul(rt: &mut LazyRuntime, a_id: u64, b_id: u64) -> Result<u64> {
    lazy_binary_op(rt, a_id, b_id, OpKind::Mul)
}

pub fn div(rt: &mut LazyRuntime, a_id: u64, b_id: u64) -> Result<u64> {
    lazy_binary_op(rt, a_id, b_id, OpKind::Div)
}

pub fn neg(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::Neg)
}

pub fn relu(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::Relu)
}

pub fn exp(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::Exp)
}

pub fn log(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::Log)
}

pub fn sqrt(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::Sqrt)
}

/// Record a matmul op. Validates 2D shapes and inner dimension match.
pub fn matmul(rt: &mut LazyRuntime, a_id: u64, b_id: u64) -> Result<u64> {
    let a_shape = rt.shape(a_id)?;
    let b_shape = rt.shape(b_id)?;

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul requires 2D tensors, got {:?} and {:?}",
            a_shape, b_shape
        )));
    }

    let (m, k1) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);

    if k1 != k2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul inner dimensions mismatch: A[{},{}] * B[{},{}]",
            m, k1, k2, n
        )));
    }

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Matmul,
        inputs: vec![a_id, b_id],
        out_shape: Shape::new(vec![m, n]),
        out_dtype: DType::Float32,
    });
    Ok(out_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::Tensor;

    fn get_device() -> Option<Device> {
        Device::new().ok()
    }

    #[test]
    fn lazy_ops_add_eval() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a);
        rt.insert_tensor(b);

        let c_id = add(&mut rt, a_id, b_id).unwrap();
        assert!(rt.is_pending(c_id));

        rt.eval(&device, c_id).unwrap();
        assert_eq!(rt.read_f32(c_id).unwrap(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn lazy_ops_chain() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a);
        rt.insert_tensor(b);

        // (a + b) * a = [11, 22, 33, 44] * [1, 2, 3, 4] = [11, 44, 99, 176]
        let sum_id = add(&mut rt, a_id, b_id).unwrap();
        let prod_id = mul(&mut rt, sum_id, a_id).unwrap();

        rt.eval(&device, prod_id).unwrap();
        assert_eq!(rt.read_f32(prod_id).unwrap(), &[11.0, 44.0, 99.0, 176.0]);
    }

    #[test]
    fn lazy_ops_matmul() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a);
        rt.insert_tensor(b);

        let c_id = matmul(&mut rt, a_id, b_id).unwrap();
        rt.eval(&device, c_id).unwrap();
        assert_eq!(rt.read_f32(c_id).unwrap(), &[19.0, 22.0, 43.0, 50.0]);
        assert_eq!(rt.shape(c_id).unwrap(), vec![2, 2]);
    }

    #[test]
    fn lazy_ops_shape_mismatch() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a);
        rt.insert_tensor(b);

        let result = add(&mut rt, a_id, b_id);
        assert!(result.is_err());
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p applegpu-core ops 2>&1`
Expected: 4 new lazy ops tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/ops.rs
git commit -m "feat: rewrite ops to build lazy graph nodes instead of executing eagerly"
```

---

### Task 4: Rewire Python bindings to use LazyRuntime

**Files:**
- Modify: `crates/python/src/lib.rs`
- Modify: `python/applegpu_runtime/__init__.py`
- Create: `python/tests/test_lazy.py`

- [ ] **Step 1: Write failing Python tests for lazy behavior**

Create `python/tests/test_lazy.py`:

```python
import applegpu_runtime as gpu


def test_eval_materializes():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    c = gpu.add(a, b)
    # c is lazy — eval materializes it
    gpu.eval(c)
    assert gpu.to_list(c) == [11.0, 22.0, 33.0, 44.0]


def test_to_list_auto_evals():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    c = gpu.add(a, b)
    # to_list should auto-eval
    assert gpu.to_list(c) == [11.0, 22.0, 33.0, 44.0]


def test_lazy_chain():
    gpu.init_backend()
    a = gpu.tensor([1.0, -2.0, 3.0, -4.0], shape=[4])
    b = gpu.neg(a)
    c = gpu.relu(b)  # relu(neg(a)) = relu([-1, 2, -3, 4]) = [0, 2, 0, 4]
    assert gpu.to_list(c) == [0.0, 2.0, 0.0, 4.0]


def test_destroy_frees_memory():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0], shape=[2])
    gpu.destroy(a)
    try:
        gpu.to_list(a)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_shape_works_on_lazy():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    c = gpu.add(a, b)
    # shape should work even before eval
    assert gpu.shape(c) == [2, 3]


def test_lazy_matmul_chain():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = gpu.matmul(a, b)
    d = gpu.neg(c)
    assert gpu.to_list(d) == [-19.0, -22.0, -43.0, -50.0]
```

- [ ] **Step 2: Rewrite Python bindings to use LazyRuntime**

Replace `crates/python/src/lib.rs`:

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::sync::Mutex;

use applegpu_core::lazy::LazyRuntime;
use applegpu_core::tensor::Tensor;

/// Global lazy runtime.
static RUNTIME_LAZY: once_cell::sync::Lazy<Mutex<LazyRuntime>> =
    once_cell::sync::Lazy::new(|| Mutex::new(LazyRuntime::new()));

/// Helper: run a binary op (records in graph, does not execute).
fn binary_op_py(a_id: u64, b_id: u64, op: fn(&mut LazyRuntime, u64, u64) -> applegpu_core::error::Result<u64>) -> PyResult<u64> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    op(&mut rt, a_id, b_id).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Helper: run a unary op (records in graph, does not execute).
fn unary_op_py(input_id: u64, op: fn(&mut LazyRuntime, u64) -> applegpu_core::error::Result<u64>) -> PyResult<u64> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    op(&mut rt, input_id).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn version() -> &'static str {
    applegpu_core::version()
}

#[pyfunction]
fn init_backend() -> PyResult<HashMap<String, String>> {
    let runtime = applegpu_core::backend::init_backend()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut info = HashMap::new();
    info.insert("backend".to_string(), format!("{:?}", runtime.backend).to_lowercase());
    info.insert("device".to_string(), runtime.device.name());
    Ok(info)
}

#[pyfunction]
fn device_name() -> PyResult<String> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(runtime.device.name())
}

#[pyfunction]
fn dtype_size(name: &str) -> PyResult<usize> {
    use applegpu_core::tensor::DType;
    let dt = match name {
        "float16" | "f16" => DType::Float16,
        "float32" | "f32" => DType::Float32,
        "float64" | "f64" => DType::Float64,
        "bfloat16" | "bf16" => DType::BFloat16,
        "int8" | "i8" => DType::Int8,
        "int16" | "i16" => DType::Int16,
        "int32" | "i32" => DType::Int32,
        "int64" | "i64" => DType::Int64,
        "uint8" | "u8" => DType::UInt8,
        "uint32" | "u32" => DType::UInt32,
        "bool" => DType::Bool,
        _ => return Err(PyValueError::new_err(format!("Unknown dtype: {}", name))),
    };
    Ok(dt.size_bytes())
}

/// Create a tensor from data (immediately materialized — this is input data).
#[pyfunction]
fn tensor(data: Vec<f32>, shape: Vec<usize>) -> PyResult<u64> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let t = Tensor::from_f32(&runtime.device, shape, &data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let id = t.meta.id;
    RUNTIME_LAZY.lock().unwrap().insert_tensor(t);
    Ok(id)
}

/// Explicitly evaluate a lazy tensor, materializing its result on the GPU.
#[pyfunction]
fn eval(tensor_id: u64) -> PyResult<()> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.eval(&runtime.device, tensor_id)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Read tensor data. Auto-evaluates if the tensor is lazy.
#[pyfunction]
fn to_list(tensor_id: u64) -> PyResult<Vec<f32>> {
    let runtime = applegpu_core::backend::get_runtime()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut rt = RUNTIME_LAZY.lock().unwrap();

    // Auto-eval if pending
    if rt.is_pending(tensor_id) {
        rt.eval(&runtime.device, tensor_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    rt.read_f32(tensor_id)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Get shape (works on both materialized and lazy tensors).
#[pyfunction]
fn shape(tensor_id: u64) -> PyResult<Vec<usize>> {
    let rt = RUNTIME_LAZY.lock().unwrap();
    rt.shape(tensor_id)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Destroy a tensor, freeing its GPU buffer.
/// Errors if pending graph ops depend on this tensor.
#[pyfunction]
fn destroy(tensor_id: u64) -> PyResult<()> {
    let mut rt = RUNTIME_LAZY.lock().unwrap();
    rt.destroy(tensor_id)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// Binary ops (lazy — just record in graph)
#[pyfunction]
fn add(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::add) }
#[pyfunction]
fn sub(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::sub) }
#[pyfunction]
fn mul(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::mul) }
#[pyfunction]
fn div(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::div) }
#[pyfunction]
fn matmul(a_id: u64, b_id: u64) -> PyResult<u64> { binary_op_py(a_id, b_id, applegpu_core::ops::matmul) }

// Unary ops (lazy)
#[pyfunction]
fn neg(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::neg) }
#[pyfunction]
fn relu(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::relu) }
#[pyfunction]
fn exp(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::exp) }
#[pyfunction]
fn log(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::log) }
#[pyfunction]
fn sqrt(input_id: u64) -> PyResult<u64> { unary_op_py(input_id, applegpu_core::ops::sqrt) }

#[pymodule]
fn applegpu_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(init_backend, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_size, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(eval, m)?)?;
    m.add_function(wrap_pyfunction!(to_list, m)?)?;
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(destroy, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(neg, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
```

- [ ] **Step 3: Update Python __init__.py**

```python
"""Apple GPU Runtime - Unified API for GPU operations on Apple Silicon."""

from applegpu_runtime.applegpu_runtime import (
    version,
    init_backend,
    device_name,
    dtype_size,
    tensor,
    eval,
    to_list,
    shape,
    destroy,
    add,
    sub,
    mul,
    div,
    neg,
    relu,
    exp,
    log,
    sqrt,
    matmul,
)

__version__ = version()
__all__ = [
    "version",
    "init_backend",
    "device_name",
    "dtype_size",
    "tensor",
    "eval",
    "to_list",
    "shape",
    "destroy",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "relu",
    "exp",
    "log",
    "sqrt",
    "matmul",
]
```

- [ ] **Step 4: Rebuild and run ALL Python tests**

Run: `uv run maturin develop && uv run pytest -v 2>&1`
Expected: All existing tests pass (backward compatible — `to_list` auto-evals) AND 6 new lazy tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/python/src/lib.rs python/applegpu_runtime/__init__.py python/tests/test_lazy.py
git commit -m "feat: rewire Python to use lazy execution with auto-eval on to_list()"
```

---

### Task 5: End-to-end verification and push

- [ ] **Step 1: Run full test suite from clean**

Run: `make clean && make test 2>&1`
Expected: All tests pass across all three layers

- [ ] **Step 2: Update backlog**

Mark Phase 4 items as complete in `docs/BACKLOG.md`. Mark Phase 5 tensor cleanup as complete (destroy() is now implemented).

- [ ] **Step 3: Update README**

Add lazy execution example and `gpu.eval()` / `gpu.destroy()` to README.

- [ ] **Step 4: Push**

```bash
git push origin main
```
