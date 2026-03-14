use std::collections::HashMap;

use crate::scheduler::ContainerId;
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
    /// A fused chain of element-wise ops. Contains runtime-generated MSL.
    FusedElementwise {
        kernel_source: String,
        function_name: String,
    },
    // Reduction ops
    Softmax,
    // Shape ops
    Transpose,
    // Scalar multiply (carries the scalar value)
    ScalarMul(f32),
    // Transformer ops
    Gelu,
    LayerNorm { eps: f32 },
    Embedding,
}

impl OpKind {
    /// Map to the MSL kernel function name.
    pub fn kernel_name(&self) -> &str {
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
            OpKind::FusedElementwise { ref function_name, .. } => function_name.as_str(),
            OpKind::Softmax => "softmax_f32",
            OpKind::Transpose => "transpose_f32",
            OpKind::ScalarMul(_) => "scalar_mul_f32",
            OpKind::Gelu => "gelu_f32",
            OpKind::LayerNorm { .. } => "layer_norm_f32",
            OpKind::Embedding => "embedding_f32",
        }
    }

    pub fn is_unary(&self) -> bool {
        matches!(self, OpKind::Neg | OpKind::Relu | OpKind::Exp | OpKind::Log | OpKind::Sqrt | OpKind::Gelu)
    }

    pub fn is_matmul(&self) -> bool {
        matches!(self, OpKind::Matmul)
    }

    pub fn is_fused(&self) -> bool {
        matches!(self, OpKind::FusedElementwise { .. })
    }

    pub fn is_elementwise(&self) -> bool {
        matches!(self, OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div |
                       OpKind::Neg | OpKind::Relu | OpKind::Exp | OpKind::Log | OpKind::Sqrt | OpKind::Gelu)
    }

    pub fn is_softmax(&self) -> bool {
        matches!(self, OpKind::Softmax)
    }

    pub fn is_transpose(&self) -> bool {
        matches!(self, OpKind::Transpose)
    }

    pub fn is_scalar_mul(&self) -> bool {
        matches!(self, OpKind::ScalarMul(_))
    }

    pub fn is_gelu(&self) -> bool {
        matches!(self, OpKind::Gelu)
    }

    pub fn is_layer_norm(&self) -> bool {
        matches!(self, OpKind::LayerNorm { .. })
    }

    pub fn is_embedding(&self) -> bool {
        matches!(self, OpKind::Embedding)
    }
}

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Unique ID for this node.
    pub id: u64,
    /// The operation to perform.
    pub op: OpKind,
    /// Input tensor IDs (1 for unary, 2 for binary/matmul).
    pub inputs: Vec<u64>,
    /// Output shape (computed at graph-build time).
    pub out_shape: Shape,
    /// Output dtype.
    pub out_dtype: DType,
    /// Container that owns this operation. Defaults to ContainerId::DEFAULT.
    pub container_id: ContainerId,
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

    /// Iterate over all nodes.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &OpNode> {
        self.nodes.values()
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
        if visited.contains(&id) {
            return Ok(());
        }

        if let Some(node) = self.nodes.get(&id) {
            // Only graph nodes can form cycles — leaf tensors can't
            if in_stack.contains(&id) {
                return Err(crate::error::GpuError::GraphError(
                    format!("Cycle detected at node {}", id),
                ));
            }
            in_stack.insert(id);

            for &input_id in &node.inputs {
                self.topo_visit(input_id, visited, in_stack, order)?;
            }

            in_stack.remove(&id);
            visited.insert(id);
            order.push(id);
        }
        // If not in graph, it's a materialized tensor — skip
        Ok(())
    }

    /// Count how many nodes in the graph consume a given node's output.
    pub fn ref_count(&self, id: u64) -> usize {
        self.nodes.values()
            .filter(|node| node.inputs.contains(&id))
            .count()
    }

    /// Number of pending nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Remove all nodes belonging to a container. Returns their IDs.
    pub fn remove_nodes_for_container(&mut self, container_id: ContainerId) -> Vec<u64> {
        let ids: Vec<u64> = self.nodes.iter()
            .filter(|(_, node)| node.container_id == container_id)
            .map(|(&id, _)| id)
            .collect();
        for &id in &ids {
            self.nodes.remove(&id);
        }
        ids
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
            inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4,
            op: OpKind::Neg,
            inputs: vec![3],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let order = g.topo_sort(4).unwrap();
        assert_eq!(order, vec![3, 4]);
    }

    #[test]
    fn graph_topo_sort_diamond() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3,
            op: OpKind::Add,
            inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4,
            op: OpKind::Sub,
            inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 5,
            op: OpKind::Mul,
            inputs: vec![3, 4],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let order = g.topo_sort(5).unwrap();
        assert_eq!(order.len(), 3);
        assert_eq!(*order.last().unwrap(), 5);
        assert!(order.contains(&3));
        assert!(order.contains(&4));
    }

    #[test]
    fn graph_partial_eval() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3,
            op: OpKind::Add,
            inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4,
            op: OpKind::Neg,
            inputs: vec![1],
            out_shape: Shape::new(vec![4]),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let order = g.topo_sort(4).unwrap();
        assert_eq!(order, vec![4]); // only node 4, not node 3
    }

    #[test]
    fn remove_nodes_for_container() {
        let c1 = ContainerId(1);
        let c2 = ContainerId(2);
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 10, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: c1,
        });
        g.add_node(OpNode {
            id: 11, op: OpKind::Neg, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: c2,
        });
        g.add_node(OpNode {
            id: 12, op: OpKind::Relu, inputs: vec![10],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: c1,
        });

        let removed = g.remove_nodes_for_container(c1);
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&10));
        assert!(removed.contains(&12));
        assert!(g.has_node(11));
        assert!(!g.has_node(10));
        assert!(!g.has_node(12));
    }
}
