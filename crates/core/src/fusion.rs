use crate::graph::{Graph, OpKind, OpNode};
use crate::scheduler::ContainerId;
use crate::tensor::{DType, Shape};

use std::sync::atomic::{AtomicU64, Ordering};

static FUSED_ID_COUNTER: AtomicU64 = AtomicU64::new(200_000);

/// MSL expression for a unary op applied to an expression string.
fn unary_msl(op: &OpKind, expr: &str, dtype: DType) -> String {
    match op {
        OpKind::Neg => format!("(-{})", expr),
        OpKind::Relu => {
            let zero = if dtype == DType::Float16 { "(half)0" } else { "0.0f" };
            format!("max({}, {})", expr, zero)
        }
        OpKind::Exp => format!("exp({})", expr),
        OpKind::Log => format!("log({})", expr),
        OpKind::Sqrt => format!("sqrt({})", expr),
        OpKind::Gelu => {
            format!("({expr} * 0.5f * (1.0f + tanh(clamp(0.7978845608f * ({expr} + 0.044715f * {expr} * {expr} * {expr}), -10.0f, 10.0f))))")
        }
        _ => unreachable!("Not a unary op"),
    }
}

/// MSL expression for a binary op.
fn binary_msl(op: &OpKind, lhs: &str, rhs: &str) -> String {
    match op {
        OpKind::Add => format!("({} + {})", lhs, rhs),
        OpKind::Sub => format!("({} - {})", lhs, rhs),
        OpKind::Mul => format!("({} * {})", lhs, rhs),
        OpKind::Div => format!("({} / {})", lhs, rhs),
        _ => unreachable!("Not a binary elementwise op"),
    }
}

/// A fusible chain of element-wise ops.
struct FusionChain {
    /// Node IDs in the chain, in execution order.
    node_ids: Vec<u64>,
    /// IDs of leaf inputs that need buffer bindings (not intermediate results).
    leaf_inputs: Vec<u64>,
    /// Output shape.
    out_shape: Shape,
    /// Output dtype.
    out_dtype: DType,
}

/// Attempt to find fusible element-wise chains in the execution order.
/// Returns chains of length >= 2 (single ops aren't worth fusing).
fn find_fusible_chains(graph: &Graph, exec_order: &[u64]) -> Vec<FusionChain> {
    let mut chains: Vec<FusionChain> = Vec::new();
    let mut consumed: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for &node_id in exec_order {
        if consumed.contains(&node_id) {
            continue;
        }
        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };
        if !node.op.is_elementwise() {
            continue;
        }

        // Try to grow a chain starting from this node
        let mut chain_ids = vec![node_id];
        let mut current_id = node_id;

        // Walk forward: find successors that are elementwise and only consume current
        loop {
            let successor = exec_order.iter().find(|&&next_id| {
                if consumed.contains(&next_id) || chain_ids.contains(&next_id) {
                    return false;
                }
                if let Some(next_node) = graph.get_node(next_id) {
                    if !next_node.op.is_elementwise() {
                        return false;
                    }
                    if !next_node.inputs.contains(&current_id) {
                        return false;
                    }
                    // current_id must have ref_count == 1 (only consumed by this successor)
                    if graph.ref_count(current_id) != 1 {
                        return false;
                    }
                    true
                } else {
                    false
                }
            });

            match successor {
                Some(&next_id) => {
                    chain_ids.push(next_id);
                    current_id = next_id;
                }
                None => break,
            }
        }

        if chain_ids.len() < 2 {
            continue; // not worth fusing a single op
        }

        // Collect leaf inputs (inputs to chain nodes that aren't chain-internal)
        let mut leaf_inputs = Vec::new();
        for &cid in &chain_ids {
            let n = graph.get_node(cid).unwrap();
            for &input_id in &n.inputs {
                if !chain_ids.contains(&input_id) && !leaf_inputs.contains(&input_id) {
                    leaf_inputs.push(input_id);
                }
            }
        }

        let last_node = graph.get_node(*chain_ids.last().unwrap()).unwrap();

        for &cid in &chain_ids {
            consumed.insert(cid);
        }

        chains.push(FusionChain {
            node_ids: chain_ids,
            leaf_inputs,
            out_shape: last_node.out_shape.clone(),
            out_dtype: last_node.out_dtype,
        });
    }

    chains
}

/// Generate MSL kernel source for a fusion chain.
/// Returns (kernel_source, function_name, leaf_input_ids_in_buffer_order).
fn generate_fused_msl(graph: &Graph, chain: &FusionChain) -> (String, String, Vec<u64>) {
    let func_name = format!("fused_{}", FUSED_ID_COUNTER.fetch_add(1, Ordering::Relaxed));
    let leaf_inputs = &chain.leaf_inputs;
    let dtype = chain.out_dtype;
    let metal_type = if dtype == DType::Float16 { "half" } else { "float" };

    // Map each leaf input to a buffer name
    let mut expr_map: std::collections::HashMap<u64, String> = std::collections::HashMap::new();

    for (i, &leaf_id) in leaf_inputs.iter().enumerate() {
        expr_map.insert(leaf_id, format!("in{}[id]", i));
    }

    // Build expressions for each chain node in order
    for &node_id in &chain.node_ids {
        let node = graph.get_node(node_id).unwrap();
        let expr = if node.op.is_unary() {
            let input_expr = expr_map.get(&node.inputs[0]).unwrap().clone();
            unary_msl(&node.op, &input_expr, dtype)
        } else {
            let lhs = expr_map.get(&node.inputs[0]).unwrap().clone();
            let rhs = expr_map.get(&node.inputs[1]).unwrap().clone();
            binary_msl(&node.op, &lhs, &rhs)
        };
        expr_map.insert(node_id, expr);
    }

    let last_id = *chain.node_ids.last().unwrap();
    let final_expr = expr_map.get(&last_id).unwrap();

    // Generate buffer parameters
    let mut params = Vec::new();
    for (i, _) in leaf_inputs.iter().enumerate() {
        params.push(format!("    device const {}* in{} [[buffer({})]]", metal_type, i, i));
    }
    let out_idx = leaf_inputs.len();
    params.push(format!("    device {}* out [[buffer({})]]", metal_type, out_idx));
    params.push(format!("    constant uint& count [[buffer({})]]", out_idx + 1));
    params.push("    uint id [[thread_position_in_grid]]".to_string());

    let kernel_source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {}(
{}
) {{
    if (id < count) {{
        out[id] = {};
    }}
}}"#,
        func_name,
        params.join(",\n"),
        final_expr
    );

    (kernel_source, func_name, leaf_inputs.clone())
}

/// Run the fusion optimization pass on the graph.
/// Modifies the graph in-place, replacing fusible chains with FusedElementwise nodes.
/// Returns the new execution order.
pub fn optimize(graph: &mut Graph, exec_order: &[u64]) -> Vec<u64> {
    let chains = find_fusible_chains(graph, exec_order);

    if chains.is_empty() {
        return exec_order.to_vec();
    }

    let mut new_order = exec_order.to_vec();

    for chain in &chains {
        let (kernel_source, function_name, leaf_inputs) = generate_fused_msl(graph, chain);

        // Use last chain node's ID — this is what callers expect from eval(target_id)
        let fused_id = *chain.node_ids.last().unwrap();
        let fused_node = OpNode {
            id: fused_id,
            op: OpKind::FusedElementwise {
                kernel_source,
                function_name,
            },
            inputs: leaf_inputs,
            out_shape: chain.out_shape.clone(),
            out_dtype: chain.out_dtype,
            container_id: ContainerId::DEFAULT,
        };

        // Remove all chain nodes from graph
        for &cid in &chain.node_ids {
            graph.remove_node(cid);
        }

        // Add the fused node
        graph.add_node(fused_node);

        // Update execution order: keep fused_id (last chain node) in place, remove all others
        let chain_set: std::collections::HashSet<u64> = chain.node_ids.iter().copied().collect();
        new_order.retain(|&id| {
            id == fused_id || !chain_set.contains(&id)
        });
    }

    new_order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_chain_add_relu() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Relu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let chains = find_fusible_chains(&g, &[3, 4]);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].node_ids, vec![3, 4]);
        assert_eq!(chains[0].leaf_inputs, vec![1, 2]);
    }

    #[test]
    fn generate_msl_add_relu() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Relu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let chains = find_fusible_chains(&g, &[3, 4]);
        let (source, name, inputs) = generate_fused_msl(&g, &chains[0]);

        assert!(source.contains("max((in0[id] + in1[id]), 0.0f)"));
        assert!(source.contains(&name));
        assert_eq!(inputs, vec![1, 2]);
    }

    #[test]
    fn optimize_replaces_chain() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Relu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let new_order = optimize(&mut g, &[3, 4]);
        assert_eq!(new_order.len(), 1);
        assert_eq!(new_order[0], 4); // uses last chain node's ID
        let fused = g.get_node(4).unwrap();
        assert!(fused.op.is_fused());
        assert_eq!(fused.inputs, vec![1, 2]);
    }

    #[test]
    fn no_fusion_for_single_op() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let new_order = optimize(&mut g, &[3]);
        assert_eq!(new_order, vec![3]);
        assert!(!g.get_node(3).unwrap().op.is_fused());
    }

    #[test]
    fn no_fusion_across_matmul() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![2, 2]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Matmul, inputs: vec![3, 2],
            out_shape: Shape::new(vec![2, 2]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 5, op: OpKind::Relu, inputs: vec![4],
            out_shape: Shape::new(vec![2, 2]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let new_order = optimize(&mut g, &[3, 4, 5]);
        assert_eq!(new_order.len(), 3);
    }

    #[test]
    fn generate_msl_add_relu_f16() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float16,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Relu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float16,
            container_id: ContainerId::DEFAULT,
        });

        let chains = find_fusible_chains(&g, &[3, 4]);
        let (source, _name, inputs) = generate_fused_msl(&g, &chains[0]);

        assert!(source.contains("half"), "f16 fused kernel should use half type");
        assert!(source.contains("max((in0[id] + in1[id]), (half)0)"));
        assert_eq!(inputs, vec![1, 2]);
    }

    #[test]
    fn find_chain_add_gelu() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Gelu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let chains = find_fusible_chains(&g, &[3, 4]);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].node_ids, vec![3, 4]);
    }

    #[test]
    fn generate_msl_add_gelu() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Gelu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let chains = find_fusible_chains(&g, &[3, 4]);
        let (source, _name, inputs) = generate_fused_msl(&g, &chains[0]);

        assert!(source.contains("tanh"));
        assert!(source.contains("0.7978845608f"));
        assert!(source.contains("0.044715f"));
        assert_eq!(inputs, vec![1, 2]);
    }

    #[test]
    fn no_fusion_when_intermediate_has_multiple_consumers() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Relu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });
        g.add_node(OpNode {
            id: 5, op: OpKind::Mul, inputs: vec![3, 1],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
            container_id: ContainerId::DEFAULT,
        });

        let chains = find_fusible_chains(&g, &[3, 4, 5]);
        assert!(chains.is_empty());
    }
}
