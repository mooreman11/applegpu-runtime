use std::collections::HashMap;

use crate::scheduler::ContainerId;
use crate::tensor::{DType, Shape};

/// A scalar value that can represent any dtype's scalar.
/// Used by ops that carry scalar parameters (Pow, ScalarMul, Clamp, MaskedFill).
#[derive(Debug, Clone, Copy)]
pub enum ScalarValue {
    Float(f64),
    Int(i64),
    UInt(u64),
    Bool(bool),
}

impl ScalarValue {
    pub fn as_f64(&self) -> f64 {
        match self {
            ScalarValue::Float(v) => *v,
            ScalarValue::Int(v) => *v as f64,
            ScalarValue::UInt(v) => *v as f64,
            ScalarValue::Bool(v) => if *v { 1.0 } else { 0.0 },
        }
    }

    pub fn to_msl_literal(&self) -> String {
        match self {
            ScalarValue::Float(v) => format!("{}", v),
            ScalarValue::Int(v) => format!("{}", v),
            ScalarValue::UInt(v) => format!("{}u", v),
            ScalarValue::Bool(v) => if *v { "true".to_string() } else { "false".to_string() },
        }
    }

    pub fn from_f32(v: f32) -> Self {
        ScalarValue::Float(v as f64)
    }
}

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
    LogSoftmax,
    // Shape ops — general transpose swapping two dimensions
    Transpose { dim0: usize, dim1: usize },
    // Scalar multiply (carries the scalar value)
    ScalarMul(ScalarValue),
    // Transformer ops
    Tanh,
    Sin,
    Cos,
    Gelu,
    Sigmoid,
    LayerNorm { eps: f32 },
    Embedding,
    // Shape ops (data copy, no compute kernel)
    Reshape { new_shape: Vec<usize> },
    // Slice along a dimension
    Slice { dim: usize, start: usize, end: usize },
    // Concatenate along a dimension (binary)
    Concat { dim: usize },
    // Add bias: 2D input + 1D bias broadcast
    AddBias,
    // Softmax with causal (upper-triangle) mask
    SoftmaxCausal,
    // Argmax reduction (output is always Int32)
    Argmax,
    // Sum reduction along last dim
    Sum,
    // Mean reduction along last dim
    Mean,
    // Variance reduction along last dim (with Bessel's correction parameter)
    Var { correction: u32 },
    // Element-wise absolute value
    Abs,
    // Element-wise sign (-1, 0, 1)
    Sign,
    // Element-wise power by scalar exponent
    Pow { exponent: ScalarValue },
    // Element-wise clamp to [min, max]
    Clamp { min_val: ScalarValue, max_val: ScalarValue },
    // Ternary conditional: where(cond, x, y) — select x where cond != 0, else y
    Where,
    // Masked fill: set elements to value where mask is true
    MaskedFill { value: ScalarValue },
    // Upper triangular: zero below diagonal
    Triu { diagonal: i32 },
    // Lower triangular: zero above diagonal
    Tril { diagonal: i32 },
    // Gather values from input along dim using index tensor
    Gather { dim: usize },
    // Select rows/columns by 1D index tensor
    IndexSelect { dim: usize },
    // CNN ops
    Conv1d { stride: usize, padding: usize, groups: usize },
    Conv2d { stride: (usize, usize), padding: (usize, usize), groups: usize },
    BatchNorm { eps: f32 },
    MaxPool2d { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
    MaxPool2dWithIndices { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize), indices_id: u64 },
    AvgPool2d { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
    // Backward ops
    SoftmaxBackward,
    LayerNormBackward { eps: f32 },
    Conv2dBackwardInput { stride: (usize, usize), padding: (usize, usize), groups: usize },
    Conv2dBackwardWeight { stride: (usize, usize), padding: (usize, usize), groups: usize },
    Conv1dBackwardInput { stride: usize, padding: usize, groups: usize },
    EmbeddingBackward,
    BatchNormBackward { eps: f32 },
    ThresholdBackward { threshold: f32 },
    TanhBackward,
    SigmoidBackward,
    GeluBackward,
    // Exact GELU forward (uses erf)
    GeluExact,
    // Exact GELU backward (derivative of exact GELU)
    GeluExactBackward,
    // Tanh GELU backward (same math as GeluBackward, but explicit name for clarity)
    GeluTanhBackward,
    MaxPool2dBackward,
    // Comparison ops (output is always Bool)
    Lt, Gt, Le, Ge, Eq, Ne,
    // Bitwise ops (integer + Bool)
    BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot,
    Shl { shift: u32 }, Shr { shift: u32 },
    // Modulo (integer only)
    Mod,
    // Element-wise min/max
    ElemMin, ElemMax,
    // Logical NOT (Bool only)
    LogicalNot,
    // Type conversion
    Cast { target_dtype: DType },
    // Quantize: float → int8/uint8 with scale and zero_point
    Quantize { scale: f32, zero_point: i32, target_dtype: DType },
    // Dequantize: int8/uint8 → float with scale and zero_point
    Dequantize { scale: f32, zero_point: i32, target_dtype: DType },
    // Scatter write: copy values into output at indices (no accumulation)
    ScatterWrite,
    // Scatter add: atomically add values into output at indices
    ScatterAdd,
    // Absolute max reduction along last dim (for L-inf norm)
    Amax,
}

impl OpKind {
    /// Map to the MSL kernel base name (without dtype suffix).
    /// The dtype suffix is appended by resolve_kernel at dispatch time.
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
            OpKind::Matmul => "matmul",
            OpKind::FusedElementwise { ref function_name, .. } => function_name.as_str(),
            OpKind::Softmax => "softmax",
            OpKind::LogSoftmax => "log_softmax",
            OpKind::Transpose { .. } => "transpose",
            OpKind::ScalarMul(_) => "scalar_mul",
            OpKind::Tanh => "elementwise_tanh",
            OpKind::Sin => "elementwise_sin",
            OpKind::Cos => "elementwise_cos",
            OpKind::Gelu => "gelu",
            OpKind::Sigmoid => "sigmoid",
            OpKind::LayerNorm { .. } => "layer_norm",
            OpKind::Embedding => "embedding",
            OpKind::Reshape { .. } => "reshape",
            OpKind::Slice { dim, .. } => if *dim == 0 { "slice_dim0" } else { "slice_dim1" },
            OpKind::Concat { dim } => if *dim == 0 { "concat_dim0" } else { "concat_dim1" },
            OpKind::AddBias => "add_bias",
            OpKind::SoftmaxCausal => "softmax_causal",
            OpKind::Argmax => "argmax",
            OpKind::Sum => "sum",
            OpKind::Mean => "mean",
            OpKind::Var { .. } => "var",
            OpKind::Abs => "elementwise_abs",
            OpKind::Sign => "elementwise_sign",
            OpKind::Pow { .. } => "pow",
            OpKind::Clamp { .. } => "clamp",
            OpKind::Where => "where",
            OpKind::MaskedFill { .. } => "masked_fill",
            OpKind::Triu { .. } => "triu",
            OpKind::Tril { .. } => "tril",
            OpKind::Gather { dim } => if *dim == 0 { "gather_dim0" } else { "gather_dim1" },
            OpKind::IndexSelect { dim } => if *dim == 0 { "index_select_dim0" } else { "index_select_dim1" },
            OpKind::Conv1d { .. } => "conv1d",
            OpKind::Conv2d { .. } => "conv2d",
            OpKind::BatchNorm { .. } => "batch_norm",
            OpKind::MaxPool2d { .. } => "max_pool2d",
            OpKind::MaxPool2dWithIndices { .. } => "max_pool2d_idx",
            OpKind::AvgPool2d { .. } => "avg_pool2d",
            OpKind::SoftmaxBackward => "softmax_backward",
            OpKind::LayerNormBackward { .. } => "layer_norm_backward",
            OpKind::Conv2dBackwardInput { .. } => "conv2d_backward_input",
            OpKind::Conv2dBackwardWeight { .. } => "conv2d_backward_weight",
            OpKind::Conv1dBackwardInput { .. } => "conv1d_backward_input",
            OpKind::EmbeddingBackward => "embedding_backward",
            OpKind::BatchNormBackward { .. } => "batch_norm_backward",
            OpKind::ThresholdBackward { .. } => "threshold_backward",
            OpKind::TanhBackward => "tanh_backward",
            OpKind::SigmoidBackward => "sigmoid_backward",
            OpKind::GeluBackward => "gelu_backward",
            OpKind::GeluExact => "gelu_exact",
            OpKind::GeluExactBackward => "gelu_exact_backward",
            OpKind::GeluTanhBackward => "gelu_tanh_backward",
            OpKind::MaxPool2dBackward => "max_pool2d_backward",
            OpKind::Lt => "lt",
            OpKind::Gt => "gt",
            OpKind::Le => "le",
            OpKind::Ge => "ge",
            OpKind::Eq => "eq",
            OpKind::Ne => "ne",
            OpKind::BitwiseAnd => "bitwise_and",
            OpKind::BitwiseOr => "bitwise_or",
            OpKind::BitwiseXor => "bitwise_xor",
            OpKind::BitwiseNot => "bitwise_not",
            OpKind::Shl { .. } => "shl",
            OpKind::Shr { .. } => "shr",
            OpKind::Mod => "mod",
            OpKind::ElemMin => "elem_min",
            OpKind::ElemMax => "elem_max",
            OpKind::LogicalNot => "logical_not",
            OpKind::Cast { .. } => "cast",
            OpKind::Quantize { .. } => "quantize",
            OpKind::Dequantize { .. } => "dequantize",
            OpKind::ScatterWrite => "scatter_write",
            OpKind::ScatterAdd => "scatter_add",
            OpKind::Amax => "amax",
        }
    }

    pub fn is_comparison(&self) -> bool {
        matches!(self, OpKind::Lt | OpKind::Gt | OpKind::Le | OpKind::Ge | OpKind::Eq | OpKind::Ne)
    }

    pub fn is_unary(&self) -> bool {
        matches!(self, OpKind::Neg | OpKind::Relu | OpKind::Exp | OpKind::Log | OpKind::Sqrt | OpKind::Tanh | OpKind::Sin | OpKind::Cos | OpKind::Gelu | OpKind::GeluExact | OpKind::Sigmoid | OpKind::Abs | OpKind::Sign | OpKind::BitwiseNot | OpKind::LogicalNot)
    }

    pub fn is_matmul(&self) -> bool {
        matches!(self, OpKind::Matmul)
    }

    pub fn is_fused(&self) -> bool {
        matches!(self, OpKind::FusedElementwise { .. })
    }

    pub fn is_elementwise(&self) -> bool {
        matches!(self, OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div |
                       OpKind::Neg | OpKind::Relu | OpKind::Exp | OpKind::Log | OpKind::Sqrt | OpKind::Tanh | OpKind::Sin | OpKind::Cos | OpKind::Gelu | OpKind::Sigmoid |
                       OpKind::Abs | OpKind::Sign)
        // Note: GeluExact is NOT fusible because it requires a custom erf_approx function
        // that would need to be injected into the fused kernel source.

    }

    pub fn is_softmax(&self) -> bool {
        matches!(self, OpKind::Softmax)
    }

    pub fn is_log_softmax(&self) -> bool {
        matches!(self, OpKind::LogSoftmax)
    }

    pub fn is_transpose(&self) -> bool {
        matches!(self, OpKind::Transpose { .. })
    }

    pub fn is_scalar_mul(&self) -> bool {
        matches!(self, OpKind::ScalarMul(_))
    }

    pub fn is_tanh(&self) -> bool {
        matches!(self, OpKind::Tanh)
    }

    pub fn is_gelu(&self) -> bool {
        matches!(self, OpKind::Gelu)
    }

    pub fn is_sigmoid(&self) -> bool {
        matches!(self, OpKind::Sigmoid)
    }

    pub fn is_layer_norm(&self) -> bool {
        matches!(self, OpKind::LayerNorm { .. })
    }

    pub fn is_embedding(&self) -> bool {
        matches!(self, OpKind::Embedding)
    }

    pub fn is_reshape(&self) -> bool {
        matches!(self, OpKind::Reshape { .. })
    }

    pub fn is_slice(&self) -> bool {
        matches!(self, OpKind::Slice { .. })
    }

    pub fn is_concat(&self) -> bool {
        matches!(self, OpKind::Concat { .. })
    }

    pub fn is_add_bias(&self) -> bool {
        matches!(self, OpKind::AddBias)
    }

    pub fn is_softmax_causal(&self) -> bool {
        matches!(self, OpKind::SoftmaxCausal)
    }

    pub fn is_argmax(&self) -> bool {
        matches!(self, OpKind::Argmax)
    }

    pub fn is_sum(&self) -> bool {
        matches!(self, OpKind::Sum)
    }

    pub fn is_mean(&self) -> bool {
        matches!(self, OpKind::Mean)
    }

    pub fn is_var(&self) -> bool {
        matches!(self, OpKind::Var { .. })
    }

    pub fn is_amax(&self) -> bool {
        matches!(self, OpKind::Amax)
    }

    pub fn is_abs(&self) -> bool {
        matches!(self, OpKind::Abs)
    }

    pub fn is_sign(&self) -> bool {
        matches!(self, OpKind::Sign)
    }

    pub fn is_pow(&self) -> bool {
        matches!(self, OpKind::Pow { .. })
    }

    pub fn is_clamp(&self) -> bool {
        matches!(self, OpKind::Clamp { .. })
    }

    pub fn is_where(&self) -> bool {
        matches!(self, OpKind::Where)
    }

    pub fn is_masked_fill(&self) -> bool {
        matches!(self, OpKind::MaskedFill { .. })
    }

    pub fn is_triu(&self) -> bool {
        matches!(self, OpKind::Triu { .. })
    }

    pub fn is_tril(&self) -> bool {
        matches!(self, OpKind::Tril { .. })
    }

    pub fn is_gather(&self) -> bool {
        matches!(self, OpKind::Gather { .. })
    }

    pub fn is_index_select(&self) -> bool {
        matches!(self, OpKind::IndexSelect { .. })
    }

    pub fn is_conv1d(&self) -> bool {
        matches!(self, OpKind::Conv1d { .. })
    }

    pub fn is_conv2d(&self) -> bool {
        matches!(self, OpKind::Conv2d { .. })
    }

    pub fn is_batch_norm(&self) -> bool {
        matches!(self, OpKind::BatchNorm { .. })
    }

    pub fn is_max_pool2d(&self) -> bool {
        matches!(self, OpKind::MaxPool2d { .. })
    }

    pub fn is_max_pool2d_with_indices(&self) -> bool {
        matches!(self, OpKind::MaxPool2dWithIndices { .. })
    }

    pub fn is_avg_pool2d(&self) -> bool {
        matches!(self, OpKind::AvgPool2d { .. })
    }

    pub fn is_softmax_backward(&self) -> bool {
        matches!(self, OpKind::SoftmaxBackward)
    }

    pub fn is_layer_norm_backward(&self) -> bool {
        matches!(self, OpKind::LayerNormBackward { .. })
    }

    pub fn is_conv2d_backward_input(&self) -> bool {
        matches!(self, OpKind::Conv2dBackwardInput { .. })
    }

    pub fn is_conv2d_backward_weight(&self) -> bool {
        matches!(self, OpKind::Conv2dBackwardWeight { .. })
    }

    pub fn is_conv1d_backward_input(&self) -> bool {
        matches!(self, OpKind::Conv1dBackwardInput { .. })
    }

    pub fn is_embedding_backward(&self) -> bool {
        matches!(self, OpKind::EmbeddingBackward)
    }

    pub fn is_batch_norm_backward(&self) -> bool {
        matches!(self, OpKind::BatchNormBackward { .. })
    }

    pub fn is_threshold_backward(&self) -> bool {
        matches!(self, OpKind::ThresholdBackward { .. })
    }

    pub fn is_tanh_backward(&self) -> bool {
        matches!(self, OpKind::TanhBackward)
    }

    pub fn is_sigmoid_backward(&self) -> bool {
        matches!(self, OpKind::SigmoidBackward)
    }

    pub fn is_gelu_backward(&self) -> bool {
        matches!(self, OpKind::GeluBackward)
    }

    pub fn is_gelu_exact(&self) -> bool {
        matches!(self, OpKind::GeluExact)
    }

    pub fn is_gelu_exact_backward(&self) -> bool {
        matches!(self, OpKind::GeluExactBackward)
    }

    pub fn is_gelu_tanh_backward(&self) -> bool {
        matches!(self, OpKind::GeluTanhBackward)
    }

    pub fn is_max_pool2d_backward(&self) -> bool {
        matches!(self, OpKind::MaxPool2dBackward)
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

    /// Partition a topo-sorted subgraph into parallel depth levels.
    /// Nodes at the same level have no data dependencies on each other.
    pub fn parallel_levels(&self, target_id: u64) -> crate::error::Result<Vec<Vec<u64>>> {
        let order = self.topo_sort(target_id)?;
        if order.is_empty() {
            return Ok(vec![]);
        }
        let mut depth: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();

        for &id in &order {
            let node = self.get_node(id).ok_or_else(|| {
                crate::error::GpuError::GraphError(format!("Node {} not in graph", id))
            })?;
            let d = node.inputs.iter()
                .filter_map(|&inp| depth.get(&inp))
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
            depth.insert(id, d);
        }

        let max_depth = depth.values().copied().max().unwrap_or(0);
        let mut levels = vec![Vec::new(); max_depth + 1];
        for &id in &order {
            levels[depth[&id]].push(id);
        }
        Ok(levels)
    }

    /// Check if a tensor ID is referenced as input by any pending graph node.
    pub fn is_referenced(&self, id: u64) -> bool {
        self.nodes.values().any(|node| node.inputs.contains(&id))
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
    fn scalar_value_to_f64() {
        assert_eq!(ScalarValue::Float(3.14).as_f64(), 3.14);
        assert_eq!(ScalarValue::Int(42).as_f64(), 42.0);
        assert_eq!(ScalarValue::UInt(255).as_f64(), 255.0);
        assert_eq!(ScalarValue::Bool(true).as_f64(), 1.0);
    }

    #[test]
    fn scalar_value_to_msl_literal() {
        assert_eq!(ScalarValue::Float(1.5).to_msl_literal(), "1.5");
        assert_eq!(ScalarValue::Int(-3).to_msl_literal(), "-3");
        assert_eq!(ScalarValue::UInt(0).to_msl_literal(), "0u");
        assert_eq!(ScalarValue::Bool(true).to_msl_literal(), "true");
    }

    #[test]
    fn op_kind_kernel_names() {
        assert_eq!(OpKind::Add.kernel_name(), "elementwise_add");
        assert_eq!(OpKind::Matmul.kernel_name(), "matmul");
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
            out_shape: Shape::new(vec![4]).unwrap(),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4,
            op: OpKind::Neg,
            inputs: vec![3],
            out_shape: Shape::new(vec![4]).unwrap(),
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
            out_shape: Shape::new(vec![4]).unwrap(),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4,
            op: OpKind::Sub,
            inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]).unwrap(),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 5,
            op: OpKind::Mul,
            inputs: vec![3, 4],
            out_shape: Shape::new(vec![4]).unwrap(),
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
            out_shape: Shape::new(vec![4]).unwrap(),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4,
            op: OpKind::Neg,
            inputs: vec![1],
            out_shape: Shape::new(vec![4]).unwrap(),
            out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let order = g.topo_sort(4).unwrap();
        assert_eq!(order, vec![4]); // only node 4, not node 3
    }

    #[test]
    fn test_parallel_levels_diamond() {
        // A -> B, A -> C, B -> D, C -> D
        let mut g = Graph::new();
        let a = OpNode { id: 1, inputs: vec![], out_shape: Shape::new(vec![2]).unwrap(), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT };
        let b = OpNode { id: 2, inputs: vec![1], out_shape: Shape::new(vec![2]).unwrap(), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT };
        let c = OpNode { id: 3, inputs: vec![1], out_shape: Shape::new(vec![2]).unwrap(), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT };
        let d = OpNode { id: 4, inputs: vec![2, 3], out_shape: Shape::new(vec![2]).unwrap(), out_dtype: DType::Float32, op: OpKind::Add, container_id: ContainerId::DEFAULT };
        g.add_node(a); g.add_node(b); g.add_node(c); g.add_node(d);

        let levels = g.parallel_levels(4).unwrap();
        assert_eq!(levels.len(), 3); // level 0: [A], level 1: [B, C], level 2: [D]
        assert_eq!(levels[0].len(), 1);
        assert_eq!(levels[1].len(), 2);
        assert_eq!(levels[2].len(), 1);
        assert!(levels[1].contains(&2) && levels[1].contains(&3));
    }

    #[test]
    fn test_parallel_levels_linear() {
        // A -> B -> C (all linear, each level has 1 node)
        let mut g = Graph::new();
        g.add_node(OpNode { id: 1, inputs: vec![], out_shape: Shape::new(vec![2]).unwrap(), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT });
        g.add_node(OpNode { id: 2, inputs: vec![1], out_shape: Shape::new(vec![2]).unwrap(), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT });
        g.add_node(OpNode { id: 3, inputs: vec![2], out_shape: Shape::new(vec![2]).unwrap(), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT });

        let levels = g.parallel_levels(3).unwrap();
        assert_eq!(levels.len(), 3);
        assert!(levels.iter().all(|l| l.len() == 1)); // all linear
    }

    #[test]
    fn test_parallel_levels_wide() {
        // 4 independent nodes (no deps) -> all at level 0
        let mut g = Graph::new();
        for i in 1..=4 {
            g.add_node(OpNode { id: i, inputs: vec![], out_shape: Shape::new(vec![2]).unwrap(), out_dtype: DType::Float32, op: OpKind::Relu, container_id: ContainerId::DEFAULT });
        }
        // Fan-in: node 5 depends on all four
        g.add_node(OpNode { id: 5, inputs: vec![1, 2, 3, 4], out_shape: Shape::new(vec![2]).unwrap(), out_dtype: DType::Float32, op: OpKind::Add, container_id: ContainerId::DEFAULT });

        let levels = g.parallel_levels(5).unwrap();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 4); // all independent at level 0
        assert_eq!(levels[1].len(), 1); // fan-in at level 1
    }

    #[test]
    fn remove_nodes_for_container() {
        let c1 = ContainerId(1);
        let c2 = ContainerId(2);
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 10, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]).unwrap(), out_dtype: DType::Float32,
            container_id: c1,
        });
        g.add_node(OpNode {
            id: 11, op: OpKind::Neg, inputs: vec![3],
            out_shape: Shape::new(vec![4]).unwrap(), out_dtype: DType::Float32,
            container_id: c2,
        });
        g.add_node(OpNode {
            id: 12, op: OpKind::Relu, inputs: vec![10],
            out_shape: Shape::new(vec![4]).unwrap(), out_dtype: DType::Float32,
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
