use crate::error::{GpuError, Result};
use crate::graph::{OpKind, OpNode};
use crate::lazy::LazyRuntime;
use crate::scheduler::ContainerId;
use crate::tensor::{DType, Shape};

use std::sync::atomic::{AtomicU64, Ordering};

fn validate_compute_dtype(dtype: DType) -> Result<()> {
    if !dtype.is_compute_supported() {
        return Err(GpuError::InvalidTensor(format!(
            "No compute kernel for {:?}. Supported: Float32, Float16.", dtype
        )));
    }
    Ok(())
}

static OP_ID_COUNTER: AtomicU64 = AtomicU64::new(100_000);

fn next_id() -> u64 {
    OP_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Record a binary element-wise op in the graph.
fn lazy_binary_op(rt: &mut LazyRuntime, a_id: u64, b_id: u64, op: OpKind) -> Result<u64> {
    let a_dtype = rt.dtype(a_id)?;
    validate_compute_dtype(a_dtype)?;
    let b_dtype = rt.dtype(b_id)?;
    validate_compute_dtype(b_dtype)?;

    let a_shape = rt.shape(a_id)?;
    let b_shape = rt.shape(b_id)?;

    if a_shape != b_shape {
        return Err(GpuError::InvalidTensor(format!(
            "Shape mismatch: {:?} vs {:?}",
            a_shape, b_shape
        )));
    }

    if a_dtype != b_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "Dtype mismatch: {:?} vs {:?}",
            a_dtype, b_dtype
        )));
    }

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op,
        inputs: vec![a_id, b_id],
        out_shape: Shape::new(a_shape),
        out_dtype: a_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Record a unary element-wise op in the graph.
fn lazy_unary_op(rt: &mut LazyRuntime, input_id: u64, op: OpKind) -> Result<u64> {
    let out_dtype = rt.dtype(input_id)?;
    validate_compute_dtype(out_dtype)?;
    let shape = rt.shape(input_id)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op,
        inputs: vec![input_id],
        out_shape: Shape::new(shape),
        out_dtype,
        container_id: ContainerId::DEFAULT,
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
    let a_dtype = rt.dtype(a_id)?;
    validate_compute_dtype(a_dtype)?;
    let b_dtype = rt.dtype(b_id)?;
    validate_compute_dtype(b_dtype)?;

    let a_shape = rt.shape(a_id)?;
    let b_shape = rt.shape(b_id)?;

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul requires 2D tensors, got {:?} and {:?}",
            a_shape, b_shape
        )));
    }

    if a_dtype != b_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "Dtype mismatch: {:?} vs {:?}",
            a_dtype, b_dtype
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
        out_dtype: a_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Softmax along last dimension. Input must be 2D [rows, cols].
pub fn softmax(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    if shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "softmax requires 2D tensor, got {:?}", shape
        )));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Softmax,
        inputs: vec![input_id],
        out_shape: Shape::new(shape),
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Transpose a 2D tensor: [rows, cols] → [cols, rows].
pub fn transpose(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    if shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "transpose requires 2D tensor, got {:?}", shape
        )));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Transpose,
        inputs: vec![input_id],
        out_shape: Shape::new(vec![shape[1], shape[0]]),
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Multiply every element by a scalar.
pub fn scalar_mul(rt: &mut LazyRuntime, input_id: u64, scale: f32) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::ScalarMul(scale),
        inputs: vec![input_id],
        out_shape: Shape::new(shape),
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Reshape a tensor without changing its data. Validates element count matches.
pub fn reshape(rt: &mut LazyRuntime, input_id: u64, new_shape: Vec<usize>) -> Result<u64> {
    let old_shape = rt.shape(input_id)?;
    let old_numel: usize = old_shape.iter().product();
    let new_numel: usize = new_shape.iter().product();
    if old_numel != new_numel {
        return Err(GpuError::InvalidTensor(format!(
            "Cannot reshape: old shape {:?} has {} elements, new shape {:?} has {}",
            old_shape, old_numel, new_shape, new_numel
        )));
    }
    let dtype = rt.dtype(input_id)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Reshape { new_shape: new_shape.clone() },
        inputs: vec![input_id],
        out_shape: Shape::new(new_shape),
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

pub fn gelu(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::Gelu)
}

pub fn layer_norm(rt: &mut LazyRuntime, input_id: u64, gamma_id: u64, beta_id: u64, eps: f32) -> Result<u64> {
    let input_shape = rt.shape(input_id)?;
    let input_dtype = rt.dtype(input_id)?;
    validate_compute_dtype(input_dtype)?;

    if input_shape.len() != 2 {
        return Err(GpuError::InvalidTensor("layer_norm requires 2D input".to_string()));
    }
    let cols = input_shape[1];

    let gamma_shape = rt.shape(gamma_id)?;
    if gamma_shape != vec![cols] {
        return Err(GpuError::InvalidTensor(format!(
            "gamma shape {:?} must be [{}]", gamma_shape, cols
        )));
    }
    let beta_shape = rt.shape(beta_id)?;
    if beta_shape != vec![cols] {
        return Err(GpuError::InvalidTensor(format!(
            "beta shape {:?} must be [{}]", beta_shape, cols
        )));
    }

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::LayerNorm { eps },
        inputs: vec![input_id, gamma_id, beta_id],
        out_shape: Shape::new(input_shape),
        out_dtype: input_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

pub fn embedding(rt: &mut LazyRuntime, weights_id: u64, indices_id: u64) -> Result<u64> {
    let weights_shape = rt.shape(weights_id)?;
    let weights_dtype = rt.dtype(weights_id)?;
    let indices_shape = rt.shape(indices_id)?;
    let indices_dtype = rt.dtype(indices_id)?;

    validate_compute_dtype(weights_dtype)?;

    if weights_shape.len() != 2 {
        return Err(GpuError::InvalidTensor("embedding weights must be 2D [vocab_size, embed_dim]".to_string()));
    }
    if indices_shape.len() != 1 {
        return Err(GpuError::InvalidTensor("embedding indices must be 1D [seq_len]".to_string()));
    }
    if indices_dtype != DType::Int32 {
        return Err(GpuError::InvalidTensor(format!(
            "embedding indices must be Int32, got {:?}", indices_dtype
        )));
    }

    let seq_len = indices_shape[0];
    let embed_dim = weights_shape[1];

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Embedding,
        inputs: vec![weights_id, indices_id],
        out_shape: Shape::new(vec![seq_len, embed_dim]),
        out_dtype: weights_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Slice: extract a sub-tensor along a given dimension.
/// dim=0 slices rows, dim=1 slices columns.
pub fn slice(rt: &mut LazyRuntime, input_id: u64, dim: usize, start: usize, end: usize) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;

    if dim >= shape.len() {
        return Err(GpuError::InvalidTensor(format!(
            "slice dim {} >= ndim {}", dim, shape.len()
        )));
    }
    if start >= end {
        return Err(GpuError::InvalidTensor(format!(
            "slice start {} >= end {}", start, end
        )));
    }
    if end > shape[dim] {
        return Err(GpuError::InvalidTensor(format!(
            "slice end {} > shape[{}] = {}", end, dim, shape[dim]
        )));
    }

    let mut out_shape = shape.clone();
    out_shape[dim] = end - start;

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Slice { dim, start, end },
        inputs: vec![input_id],
        out_shape: Shape::new(out_shape),
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Concat: concatenate two tensors along a given dimension.
pub fn concat(rt: &mut LazyRuntime, a_id: u64, b_id: u64, dim: usize) -> Result<u64> {
    let a_dtype = rt.dtype(a_id)?;
    validate_compute_dtype(a_dtype)?;
    let b_dtype = rt.dtype(b_id)?;
    if a_dtype != b_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "concat dtype mismatch: {:?} vs {:?}", a_dtype, b_dtype
        )));
    }

    let a_shape = rt.shape(a_id)?;
    let b_shape = rt.shape(b_id)?;

    if a_shape.len() != b_shape.len() {
        return Err(GpuError::InvalidTensor(format!(
            "concat ndim mismatch: {:?} vs {:?}", a_shape, b_shape
        )));
    }
    if dim >= a_shape.len() {
        return Err(GpuError::InvalidTensor(format!(
            "concat dim {} >= ndim {}", dim, a_shape.len()
        )));
    }

    for i in 0..a_shape.len() {
        if i != dim && a_shape[i] != b_shape[i] {
            return Err(GpuError::InvalidTensor(format!(
                "concat shape mismatch on dim {}: {} vs {}", i, a_shape[i], b_shape[i]
            )));
        }
    }

    let mut out_shape = a_shape.clone();
    out_shape[dim] = a_shape[dim] + b_shape[dim];

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Concat { dim },
        inputs: vec![a_id, b_id],
        out_shape: Shape::new(out_shape),
        out_dtype: a_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// AddBias: add a 1D bias to each row of a 2D tensor.
/// input: [rows, cols], bias: [cols] -> output: [rows, cols]
pub fn add_bias(rt: &mut LazyRuntime, input_id: u64, bias_id: u64) -> Result<u64> {
    let input_dtype = rt.dtype(input_id)?;
    validate_compute_dtype(input_dtype)?;
    let bias_dtype = rt.dtype(bias_id)?;
    if input_dtype != bias_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "add_bias dtype mismatch: {:?} vs {:?}", input_dtype, bias_dtype
        )));
    }

    let input_shape = rt.shape(input_id)?;
    let bias_shape = rt.shape(bias_id)?;

    if input_shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "add_bias requires 2D input, got {:?}", input_shape
        )));
    }
    if bias_shape.len() != 1 {
        return Err(GpuError::InvalidTensor(format!(
            "add_bias requires 1D bias, got {:?}", bias_shape
        )));
    }
    if bias_shape[0] != input_shape[1] {
        return Err(GpuError::InvalidTensor(format!(
            "add_bias bias length {} != input cols {}", bias_shape[0], input_shape[1]
        )));
    }

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::AddBias,
        inputs: vec![input_id, bias_id],
        out_shape: Shape::new(input_shape),
        out_dtype: input_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Softmax with causal (upper-triangle) mask.
/// For position (row, col) where col > row, value is treated as -inf.
pub fn softmax_causal(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    if shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "softmax_causal requires 2D tensor, got {:?}", shape
        )));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::SoftmaxCausal,
        inputs: vec![input_id],
        out_shape: Shape::new(shape),
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Argmax along last dimension. Output dtype is always Int32.
/// 2D [rows, cols] -> [rows]. 1D [cols] -> [1].
pub fn argmax(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let input_dtype = rt.dtype(input_id)?;
    validate_compute_dtype(input_dtype)?;
    let shape = rt.shape(input_id)?;

    let (out_shape, _rows, _cols) = if shape.len() == 2 {
        (vec![shape[0]], shape[0], shape[1])
    } else if shape.len() == 1 {
        (vec![1], 1, shape[0])
    } else {
        return Err(GpuError::InvalidTensor(format!(
            "argmax requires 1D or 2D tensor, got {:?}", shape
        )));
    };

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Argmax,
        inputs: vec![input_id],
        out_shape: Shape::new(out_shape),
        out_dtype: DType::Int32,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
/// Q: [q_len, d_k], K: [kv_len, d_k], V: [kv_len, d_v]
/// Output: [q_len, d_v]
pub fn attention(rt: &mut LazyRuntime, q_id: u64, k_id: u64, v_id: u64) -> Result<u64> {
    let q_shape = rt.shape(q_id)?;
    let k_shape = rt.shape(k_id)?;
    let v_shape = rt.shape(v_id)?;

    if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
        return Err(GpuError::InvalidTensor(
            "attention requires 2D tensors for Q, K, V".to_string()
        ));
    }

    let d_k = q_shape[1];
    if k_shape[1] != d_k {
        return Err(GpuError::InvalidTensor(format!(
            "Q and K must have same d_k: Q[{},{}] K[{},{}]",
            q_shape[0], q_shape[1], k_shape[0], k_shape[1]
        )));
    }
    if k_shape[0] != v_shape[0] {
        return Err(GpuError::InvalidTensor(format!(
            "K and V must have same seq_len: K[{},{}] V[{},{}]",
            k_shape[0], k_shape[1], v_shape[0], v_shape[1]
        )));
    }

    // K^T: [kv_len, d_k] → [d_k, kv_len]
    let kt_id = transpose(rt, k_id)?;
    // scores = Q @ K^T: [q_len, d_k] @ [d_k, kv_len] → [q_len, kv_len]
    let scores_id = matmul(rt, q_id, kt_id)?;
    // Scale by 1/sqrt(d_k)
    let scale = 1.0 / (d_k as f32).sqrt();
    let scaled_scores_id = scalar_mul(rt, scores_id, scale)?;
    // softmax along last dimension
    let attn_weights_id = softmax(rt, scaled_scores_id)?;
    // output = attn_weights @ V: [q_len, kv_len] @ [kv_len, d_v] → [q_len, d_v]
    let output_id = matmul(rt, attn_weights_id, v_id)?;

    Ok(output_id)
}

/// Causal scaled dot-product attention: softmax_causal(Q @ K^T / sqrt(d_k)) @ V
/// Q: [q_len, d_k], K: [kv_len, d_k], V: [kv_len, d_v]
/// Output: [q_len, d_v]
pub fn attention_causal(rt: &mut LazyRuntime, q_id: u64, k_id: u64, v_id: u64) -> Result<u64> {
    let q_shape = rt.shape(q_id)?;
    let k_shape = rt.shape(k_id)?;
    let v_shape = rt.shape(v_id)?;

    if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
        return Err(GpuError::InvalidTensor(
            "attention_causal requires 2D tensors".to_string()
        ));
    }

    let d_k = q_shape[1];
    if k_shape[1] != d_k {
        return Err(GpuError::InvalidTensor(format!(
            "Q and K must have same d_k: Q[{},{}] K[{},{}]",
            q_shape[0], q_shape[1], k_shape[0], k_shape[1]
        )));
    }
    if k_shape[0] != v_shape[0] {
        return Err(GpuError::InvalidTensor(format!(
            "K and V must have same seq_len: K[{},{}] V[{},{}]",
            k_shape[0], k_shape[1], v_shape[0], v_shape[1]
        )));
    }

    // K^T: [kv_len, d_k] → [d_k, kv_len]
    let kt_id = transpose(rt, k_id)?;
    // scores = Q @ K^T: [q_len, d_k] @ [d_k, kv_len] → [q_len, kv_len]
    let scores_id = matmul(rt, q_id, kt_id)?;
    // Scale by 1/sqrt(d_k)
    let scale = 1.0 / (d_k as f32).sqrt();
    let scaled_scores_id = scalar_mul(rt, scores_id, scale)?;
    // Causal softmax (masks future positions)
    let attn_weights_id = softmax_causal(rt, scaled_scores_id)?;
    // output = attn_weights @ V: [q_len, kv_len] @ [kv_len, d_v] → [q_len, d_v]
    let output_id = matmul(rt, attn_weights_id, v_id)?;

    Ok(output_id)
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
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();

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
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();

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
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();

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
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();

        let result = add(&mut rt, a_id, b_id);
        assert!(result.is_err());
    }

    #[test]
    fn lazy_ops_softmax() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();

        let s_id = softmax(&mut rt, a_id).unwrap();
        rt.eval(&device, s_id).unwrap();
        let result = rt.read_f32(s_id).unwrap();

        assert!((result[0] - 0.0900).abs() < 0.001);
        assert!((result[1] - 0.2447).abs() < 0.001);
        assert!((result[2] - 0.6652).abs() < 0.001);
        assert!((result[3] - 0.3333).abs() < 0.001);
    }

    #[test]
    fn lazy_ops_transpose() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();

        let t_id = transpose(&mut rt, a_id).unwrap();
        assert_eq!(rt.shape(t_id).unwrap(), vec![3, 2]);

        rt.eval(&device, t_id).unwrap();
        let result = rt.read_f32(t_id).unwrap();
        assert_eq!(result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn f16_add_eval() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        use half::f16;
        let data: Vec<u16> = vec![f16::from_f32(1.0).to_bits(); 4];
        let a = Tensor::from_f16(&device, vec![4], &data).unwrap();
        let b = Tensor::from_f16(&device, vec![4], &data).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();
        let c_id = add(&mut rt, a_id, b_id).unwrap();
        rt.eval(&device, c_id).unwrap();
        let result = rt.read_f16(c_id).unwrap();
        assert_eq!(f16::from_bits(result[0]).to_f32(), 2.0);
        assert_eq!(f16::from_bits(result[1]).to_f32(), 2.0);
    }

    #[test]
    fn mixed_dtype_errors() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![4], &[1.0; 4]).unwrap();
        let b = Tensor::from_f16(&device, vec![4], &[0u16; 4]).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();
        assert!(add(&mut rt, a_id, b_id).is_err());
    }

    #[test]
    fn f16_matmul_eval() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        use half::f16;
        let a_data: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let b_data: Vec<u16> = [5.0f32, 6.0, 7.0, 8.0].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let a = Tensor::from_f16(&device, vec![2, 2], &a_data).unwrap();
        let b = Tensor::from_f16(&device, vec![2, 2], &b_data).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();
        let c_id = matmul(&mut rt, a_id, b_id).unwrap();
        rt.eval(&device, c_id).unwrap();
        let result = rt.read_f16(c_id).unwrap();
        // Expected: [[19, 22], [43, 50]]
        assert!((f16::from_bits(result[0]).to_f32() - 19.0).abs() < 0.5);
        assert!((f16::from_bits(result[1]).to_f32() - 22.0).abs() < 0.5);
        assert!((f16::from_bits(result[2]).to_f32() - 43.0).abs() < 0.5);
        assert!((f16::from_bits(result[3]).to_f32() - 50.0).abs() < 0.5);
    }

    #[test]
    fn f16_unary_relu_eval() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        use half::f16;
        let data: Vec<u16> = [-1.0f32, 2.0, -3.0, 4.0].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let a = Tensor::from_f16(&device, vec![4], &data).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        let r_id = relu(&mut rt, a_id).unwrap();
        rt.eval(&device, r_id).unwrap();
        let result = rt.read_f16(r_id).unwrap();
        assert_eq!(f16::from_bits(result[0]).to_f32(), 0.0);
        assert_eq!(f16::from_bits(result[1]).to_f32(), 2.0);
        assert_eq!(f16::from_bits(result[2]).to_f32(), 0.0);
        assert_eq!(f16::from_bits(result[3]).to_f32(), 4.0);
    }

    #[test]
    fn f16_softmax_eval() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        use half::f16;
        let data: Vec<u16> = [1.0f32, 2.0, 3.0, 1.0, 1.0, 1.0].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let a = Tensor::from_f16(&device, vec![2, 3], &data).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        let s_id = softmax(&mut rt, a_id).unwrap();
        rt.eval(&device, s_id).unwrap();
        let result = rt.read_f16(s_id).unwrap();
        assert!((f16::from_bits(result[0]).to_f32() - 0.0900).abs() < 0.01);
        assert!((f16::from_bits(result[1]).to_f32() - 0.2447).abs() < 0.01);
        assert!((f16::from_bits(result[2]).to_f32() - 0.6652).abs() < 0.01);
        assert!((f16::from_bits(result[3]).to_f32() - 0.3333).abs() < 0.01);
    }

    #[test]
    fn f16_transpose_eval() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        use half::f16;
        let data: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let a = Tensor::from_f16(&device, vec![2, 3], &data).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        let t_id = transpose(&mut rt, a_id).unwrap();
        rt.eval(&device, t_id).unwrap();
        let result = rt.read_f16(t_id).unwrap();
        let result_f32: Vec<f32> = result.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
        assert_eq!(result_f32, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn f16_scalar_mul_eval() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        use half::f16;
        let data: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let a = Tensor::from_f16(&device, vec![4], &data).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        let s_id = scalar_mul(&mut rt, a_id, 3.0).unwrap();
        rt.eval(&device, s_id).unwrap();
        let result = rt.read_f16(s_id).unwrap();
        let result_f32: Vec<f32> = result.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
        assert_eq!(result_f32, &[3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn lazy_ops_attention() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let q = Tensor::from_f32(&device, vec![2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = Tensor::from_f32(&device, vec![2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let v = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let q_id = q.meta.id;
        let k_id = k.meta.id;
        let v_id = v.meta.id;
        rt.insert_tensor(q).unwrap();
        rt.insert_tensor(k).unwrap();
        rt.insert_tensor(v).unwrap();

        let out_id = attention(&mut rt, q_id, k_id, v_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result.len(), 4);
        // With Q=K=I, scores = I/sqrt(2), softmax gives weighted mix of V rows
        for &v in &result {
            assert!(v.is_finite());
            assert!(v >= 0.0 && v <= 10.0);
        }
    }

    #[test]
    fn lazy_ops_gelu_f32() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let a = Tensor::from_f32(&device, vec![4], &[0.0, 1.0, -1.0, 2.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();

        let g_id = gelu(&mut rt, a_id).unwrap();
        rt.eval(&device, g_id).unwrap();
        let result = rt.read_f32(g_id).unwrap();

        // gelu(0) = 0
        assert!((result[0] - 0.0).abs() < 0.001);
        // gelu(1) ≈ 0.8412
        assert!((result[1] - 0.8412).abs() < 0.01);
        // gelu(-1) ≈ -0.1588
        assert!((result[2] - (-0.1588)).abs() < 0.01);
        // gelu(2) ≈ 1.9545
        assert!((result[3] - 1.9545).abs() < 0.01);
    }

    #[test]
    fn lazy_ops_gelu_f16() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        use half::f16;
        let data: Vec<u16> = [0.0f32, 1.0, -1.0, 2.0].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let a = Tensor::from_f16(&device, vec![4], &data).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();

        let g_id = gelu(&mut rt, a_id).unwrap();
        rt.eval(&device, g_id).unwrap();
        let result = rt.read_f16(g_id).unwrap();

        assert!((f16::from_bits(result[0]).to_f32() - 0.0).abs() < 0.05);
        assert!((f16::from_bits(result[1]).to_f32() - 0.8412).abs() < 0.05);
        assert!((f16::from_bits(result[2]).to_f32() - (-0.1588)).abs() < 0.05);
        assert!((f16::from_bits(result[3]).to_f32() - 1.9545).abs() < 0.05);
    }

    #[test]
    fn lazy_ops_layer_norm_f32() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // 2x4 input
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma_data = [1.0, 1.0, 1.0, 1.0]; // scale = 1
        let beta_data = [0.0, 0.0, 0.0, 0.0]; // bias = 0

        let input = Tensor::from_f32(&device, vec![2, 4], &input_data).unwrap();
        let gamma = Tensor::from_f32(&device, vec![4], &gamma_data).unwrap();
        let beta = Tensor::from_f32(&device, vec![4], &beta_data).unwrap();
        let input_id = input.meta.id;
        let gamma_id = gamma.meta.id;
        let beta_id = beta.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(gamma).unwrap();
        rt.insert_tensor(beta).unwrap();

        let out_id = layer_norm(&mut rt, input_id, gamma_id, beta_id, 1e-5).unwrap();
        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();

        // For row [1,2,3,4]: mean=2.5, var=1.25, std=sqrt(1.25+1e-5)
        // normalized = (x-2.5)/std -> [-1.3416, -0.4472, 0.4472, 1.3416] approx
        assert!((result[0] - (-1.3416)).abs() < 0.01);
        assert!((result[1] - (-0.4472)).abs() < 0.01);
        assert!((result[2] - 0.4472).abs() < 0.01);
        assert!((result[3] - 1.3416).abs() < 0.01);

        // Second row should also be normalized
        assert!((result[4] - (-1.3416)).abs() < 0.01);
    }

    #[test]
    fn lazy_ops_layer_norm_f16() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        use half::f16;

        let input_data: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let gamma_data: Vec<u16> = [1.0f32; 4].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let beta_data: Vec<u16> = [0.0f32; 4].iter().map(|&x| f16::from_f32(x).to_bits()).collect();

        let input = Tensor::from_f16(&device, vec![1, 4], &input_data).unwrap();
        let gamma = Tensor::from_f16(&device, vec![4], &gamma_data).unwrap();
        let beta = Tensor::from_f16(&device, vec![4], &beta_data).unwrap();
        let input_id = input.meta.id;
        let gamma_id = gamma.meta.id;
        let beta_id = beta.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(gamma).unwrap();
        rt.insert_tensor(beta).unwrap();

        let out_id = layer_norm(&mut rt, input_id, gamma_id, beta_id, 1e-5).unwrap();
        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f16(out_id).unwrap();

        assert!((f16::from_bits(result[0]).to_f32() - (-1.3416)).abs() < 0.1);
        assert!((f16::from_bits(result[3]).to_f32() - 1.3416).abs() < 0.1);
    }

    #[test]
    fn lazy_ops_embedding_f32() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Weights: 3 vocab x 2 embed_dim
        let weights_data = [
            10.0, 11.0,  // row 0
            20.0, 21.0,  // row 1
            30.0, 31.0,  // row 2
        ];
        let indices_data: [i32; 3] = [2, 0, 1];

        let weights = Tensor::from_f32(&device, vec![3, 2], &weights_data).unwrap();
        let indices_bytes = unsafe {
            std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, 12)
        };
        let indices = Tensor::from_data(&device, vec![3], DType::Int32, indices_bytes).unwrap();
        let w_id = weights.meta.id;
        let i_id = indices.meta.id;
        rt.insert_tensor(weights).unwrap();
        rt.insert_tensor(indices).unwrap();

        let out_id = embedding(&mut rt, w_id, i_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![3, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();

        // indices [2, 0, 1] -> rows [30,31], [10,11], [20,21]
        assert_eq!(result, &[30.0, 31.0, 10.0, 11.0, 20.0, 21.0]);
    }

    #[test]
    fn lazy_ops_embedding_f16() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        use half::f16;

        // Weights: 3 vocab x 2 embed_dim
        let weights_data: Vec<u16> = [10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0]
            .iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let indices_data: [i32; 2] = [1, 2];

        let weights = Tensor::from_f16(&device, vec![3, 2], &weights_data).unwrap();
        let indices_bytes = unsafe {
            std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, 8)
        };
        let indices = Tensor::from_data(&device, vec![2], DType::Int32, indices_bytes).unwrap();
        let w_id = weights.meta.id;
        let i_id = indices.meta.id;
        rt.insert_tensor(weights).unwrap();
        rt.insert_tensor(indices).unwrap();

        let out_id = embedding(&mut rt, w_id, i_id).unwrap();
        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f16(out_id).unwrap();
        let result_f32: Vec<f32> = result.iter().map(|&b| f16::from_bits(b).to_f32()).collect();

        // indices [1, 2] -> rows [20,21], [30,31]
        assert_eq!(result_f32, &[20.0, 21.0, 30.0, 31.0]);
    }

    #[test]
    fn embedding_rejects_non_int32_indices() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let weights = Tensor::from_f32(&device, vec![3, 2], &[0.0; 6]).unwrap();
        let indices = Tensor::from_f32(&device, vec![3], &[0.0, 1.0, 2.0]).unwrap();
        let w_id = weights.meta.id;
        let i_id = indices.meta.id;
        rt.insert_tensor(weights).unwrap();
        rt.insert_tensor(indices).unwrap();

        let result = embedding(&mut rt, w_id, i_id);
        assert!(result.is_err());
    }

    #[test]
    fn layer_norm_rejects_non_2d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let gamma = Tensor::from_f32(&device, vec![4], &[1.0; 4]).unwrap();
        let beta = Tensor::from_f32(&device, vec![4], &[0.0; 4]).unwrap();
        let i_id = input.meta.id;
        let g_id = gamma.meta.id;
        let b_id = beta.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(gamma).unwrap();
        rt.insert_tensor(beta).unwrap();

        let result = layer_norm(&mut rt, i_id, g_id, b_id, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_preserves_data() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let reshaped_id = crate::ops::reshape(&mut rt, id, vec![2, 3]).unwrap();
        rt.eval(&device, reshaped_id).unwrap();
        assert_eq!(rt.shape(reshaped_id).unwrap(), vec![2, 3]);
        assert_eq!(rt.read_f32(reshaped_id).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_validates_numel() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![6], &[1.0; 6]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        assert!(crate::ops::reshape(&mut rt, id, vec![2, 2]).is_err());
    }

    #[test]
    fn test_reshape_f16() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        use half::f16;
        let data: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let t = Tensor::from_f16(&device, vec![6], &data).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let reshaped_id = crate::ops::reshape(&mut rt, id, vec![3, 2]).unwrap();
        rt.eval(&device, reshaped_id).unwrap();
        assert_eq!(rt.shape(reshaped_id).unwrap(), vec![3, 2]);
        let result = rt.read_f16(reshaped_id).unwrap();
        let result_f32: Vec<f32> = result.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
        assert_eq!(result_f32, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_chain_with_op() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        // reshape [6] -> [2, 3], then negate
        let reshaped_id = crate::ops::reshape(&mut rt, id, vec![2, 3]).unwrap();
        let neg_id = crate::ops::neg(&mut rt, reshaped_id).unwrap();
        rt.eval(&device, neg_id).unwrap();
        assert_eq!(rt.shape(neg_id).unwrap(), vec![2, 3]);
        assert_eq!(rt.read_f32(neg_id).unwrap(), &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
    }

    #[test]
    fn test_compute_rejects_int32() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let data = [0i32; 4];
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, 16) };
        let a = Tensor::from_data(&device, vec![4], crate::tensor::DType::Int32, bytes).unwrap();
        let b = Tensor::from_data(&device, vec![4], crate::tensor::DType::Int32, bytes).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();
        assert!(crate::ops::add(&mut rt, a_id, b_id).is_err());
    }

    // ── Slice tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_slice_dim1() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // 2x4 tensor: [[1,2,3,4],[5,6,7,8]]
        let t = Tensor::from_f32(&device, vec![2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        // Slice columns [1, 3) -> [[2,3],[6,7]]
        let s_id = crate::ops::slice(&mut rt, id, 1, 1, 3).unwrap();
        assert_eq!(rt.shape(s_id).unwrap(), vec![2, 2]);
        rt.eval(&device, s_id).unwrap();
        assert_eq!(rt.read_f32(s_id).unwrap(), &[2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn test_slice_dim0() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // 3x2 tensor: [[1,2],[3,4],[5,6]]
        let t = Tensor::from_f32(&device, vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        // Slice rows [1, 3) -> [[3,4],[5,6]]
        let s_id = crate::ops::slice(&mut rt, id, 0, 1, 3).unwrap();
        assert_eq!(rt.shape(s_id).unwrap(), vec![2, 2]);
        rt.eval(&device, s_id).unwrap();
        assert_eq!(rt.read_f32(s_id).unwrap(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_slice_validates() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![2, 4], &[0.0; 8]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        // dim out of range
        assert!(crate::ops::slice(&mut rt, id, 2, 0, 1).is_err());
        // start >= end
        assert!(crate::ops::slice(&mut rt, id, 0, 2, 1).is_err());
        // end > shape[dim]
        assert!(crate::ops::slice(&mut rt, id, 0, 0, 5).is_err());
    }

    // ── Concat tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_concat_dim1() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![2, 3], &[3.0, 4.0, 0.0, 7.0, 8.0, 0.0]).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();
        let c_id = crate::ops::concat(&mut rt, a_id, b_id, 1).unwrap();
        assert_eq!(rt.shape(c_id).unwrap(), vec![2, 5]);
        rt.eval(&device, c_id).unwrap();
        assert_eq!(rt.read_f32(c_id).unwrap(), &[1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0, 8.0, 0.0]);
    }

    #[test]
    fn test_concat_dim0() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![1, 3], &[7.0, 8.0, 9.0]).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();
        let c_id = crate::ops::concat(&mut rt, a_id, b_id, 0).unwrap();
        assert_eq!(rt.shape(c_id).unwrap(), vec![3, 3]);
        rt.eval(&device, c_id).unwrap();
        assert_eq!(rt.read_f32(c_id).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    // ── AddBias tests ────────────────────────────────────────────────────────

    #[test]
    fn test_add_bias() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // 2x3 input + 3-element bias
        let input = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let bias = Tensor::from_f32(&device, vec![3], &[10.0, 20.0, 30.0]).unwrap();
        let i_id = input.meta.id;
        let b_id = bias.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(bias).unwrap();
        let out_id = crate::ops::add_bias(&mut rt, i_id, b_id).unwrap();
        rt.eval(&device, out_id).unwrap();
        assert_eq!(rt.read_f32(out_id).unwrap(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_add_bias_validates() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let input = Tensor::from_f32(&device, vec![4], &[1.0; 4]).unwrap();
        let bias = Tensor::from_f32(&device, vec![4], &[1.0; 4]).unwrap();
        let i_id = input.meta.id;
        let b_id = bias.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(bias).unwrap();
        // input is 1D, not 2D
        assert!(crate::ops::add_bias(&mut rt, i_id, b_id).is_err());
    }

    // ── SoftmaxCausal tests ──────────────────────────────────────────────────

    #[test]
    fn test_softmax_causal() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // 3x3 input of ones
        let t = Tensor::from_f32(&device, vec![3, 3], &[1.0; 9]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let s_id = crate::ops::softmax_causal(&mut rt, id).unwrap();
        rt.eval(&device, s_id).unwrap();
        let result = rt.read_f32(s_id).unwrap();
        // Row 0: only col 0 visible -> [1.0, 0.0, 0.0]
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 0.0).abs() < 0.001);
        assert!((result[2] - 0.0).abs() < 0.001);
        // Row 1: cols 0,1 visible -> [0.5, 0.5, 0.0]
        assert!((result[3] - 0.5).abs() < 0.001);
        assert!((result[4] - 0.5).abs() < 0.001);
        assert!((result[5] - 0.0).abs() < 0.001);
        // Row 2: all visible -> [0.333, 0.333, 0.333]
        assert!((result[6] - 0.3333).abs() < 0.001);
        assert!((result[7] - 0.3333).abs() < 0.001);
        assert!((result[8] - 0.3333).abs() < 0.001);
        // Each row sums to 1
        for r in 0..3 {
            let sum: f32 = (0..3).map(|c| result[r * 3 + c]).sum();
            assert!((sum - 1.0).abs() < 0.001);
        }
    }

    // ── Argmax tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_argmax_f32() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // 2x4 tensor
        let t = Tensor::from_f32(&device, vec![2, 4], &[1.0, 3.0, 2.0, 0.0, 5.0, 1.0, 9.0, 2.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let a_id = crate::ops::argmax(&mut rt, id).unwrap();
        assert_eq!(rt.shape(a_id).unwrap(), vec![2]);
        assert_eq!(rt.dtype(a_id).unwrap(), DType::Int32);
        rt.eval(&device, a_id).unwrap();
        let bytes = rt.read_bytes(a_id).unwrap();
        let indices: &[i32] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const i32, 2)
        };
        assert_eq!(indices[0], 1); // max of [1,3,2,0] at index 1
        assert_eq!(indices[1], 2); // max of [5,1,9,2] at index 2
    }

    #[test]
    fn test_argmax_returns_int32() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![1, 3], &[0.0, 5.0, 2.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let a_id = crate::ops::argmax(&mut rt, id).unwrap();
        assert_eq!(rt.dtype(a_id).unwrap(), DType::Int32);
        rt.eval(&device, a_id).unwrap();
        let bytes = rt.read_bytes(a_id).unwrap();
        let idx = i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_argmax_1d_input() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let t = Tensor::from_f32(&device, vec![4], &[1.0, 0.0, 7.0, 3.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let a_id = crate::ops::argmax(&mut rt, id).unwrap();
        assert_eq!(rt.shape(a_id).unwrap(), vec![1]);
        assert_eq!(rt.dtype(a_id).unwrap(), DType::Int32);
        rt.eval(&device, a_id).unwrap();
        let bytes = rt.read_bytes(a_id).unwrap();
        let idx = i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(idx, 2); // max at index 2
    }

    // ── AttentionCausal tests ──────────────────────────────────────────────

    #[test]
    fn test_attention_causal() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let q = Tensor::from_f32(&device, vec![2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = Tensor::from_f32(&device, vec![2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let v = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let q_id = q.meta.id;
        let k_id = k.meta.id;
        let v_id = v.meta.id;
        rt.insert_tensor(q).unwrap();
        rt.insert_tensor(k).unwrap();
        rt.insert_tensor(v).unwrap();

        let out_id = attention_causal(&mut rt, q_id, k_id, v_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result.len(), 4);
        // Row 0 can only attend to position 0 (causal), so output[0] ≈ v[0] = [1.0, 2.0]
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_attention_causal_rejects_non_2d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let q = Tensor::from_f32(&device, vec![4], &[1.0; 4]).unwrap();
        let k = Tensor::from_f32(&device, vec![4], &[1.0; 4]).unwrap();
        let v = Tensor::from_f32(&device, vec![4], &[1.0; 4]).unwrap();
        let q_id = q.meta.id;
        let k_id = k.meta.id;
        let v_id = v.meta.id;
        rt.insert_tensor(q).unwrap();
        rt.insert_tensor(k).unwrap();
        rt.insert_tensor(v).unwrap();

        assert!(attention_causal(&mut rt, q_id, k_id, v_id).is_err());
    }

    #[test]
    fn test_gelu_large_values_no_nan() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // Values that previously caused NaN due to tanh overflow
        let t = Tensor::from_f32(&device, vec![6], &[-20.0, -10.0, -5.0, 5.0, 10.0, 20.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let out_id = gelu(&mut rt, id).unwrap();
        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        // All values should be finite (no NaN or Inf)
        for &v in &result {
            assert!(v.is_finite(), "GELU produced non-finite value: {}", v);
        }
        // GELU(-20) ≈ 0.0, GELU(20) ≈ 20.0
        assert!(result[0].abs() < 0.01);
        assert!((result[5] - 20.0).abs() < 0.01);
    }
}
