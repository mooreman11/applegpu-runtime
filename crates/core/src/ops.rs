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

    let a_shape_vec = rt.shape(a_id)?;
    let b_shape_vec = rt.shape(b_id)?;

    if a_dtype != b_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "Dtype mismatch: {:?} vs {:?}",
            a_dtype, b_dtype
        )));
    }

    let a_shape_obj = Shape::new(a_shape_vec)?;
    let b_shape_obj = Shape::new(b_shape_vec)?;
    let out_shape = a_shape_obj.broadcast_with(&b_shape_obj)?;

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op,
        inputs: vec![a_id, b_id],
        out_shape,
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
        out_shape: Shape::new(shape)?,
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

pub fn abs(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::Abs)
}

pub fn sign(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    lazy_unary_op(rt, input_id, OpKind::Sign)
}

/// Element-wise power by scalar exponent.
pub fn pow(rt: &mut LazyRuntime, input_id: u64, exponent: f32) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Pow { exponent },
        inputs: vec![input_id],
        out_shape: Shape::new(shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Element-wise clamp to [min_val, max_val].
pub fn clamp(rt: &mut LazyRuntime, input_id: u64, min_val: f32, max_val: f32) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Clamp { min_val, max_val },
        inputs: vec![input_id],
        out_shape: Shape::new(shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Record a matmul op. Validates 2D shapes and inner dimension match.
pub fn matmul(rt: &mut LazyRuntime, a_id: u64, b_id: u64) -> Result<u64> {
    let a_dtype = rt.dtype(a_id)?;
    validate_compute_dtype(a_dtype)?;
    let b_dtype = rt.dtype(b_id)?;
    validate_compute_dtype(b_dtype)?;

    let a_shape = rt.shape(a_id)?;
    let b_shape = rt.shape(b_id)?;

    if a_shape.len() < 2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul requires at least 2D tensors, got {:?}",
            a_shape
        )));
    }
    if b_shape.len() < 2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul requires at least 2D tensors, got {:?}",
            b_shape
        )));
    }

    if a_dtype != b_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "Dtype mismatch: {:?} vs {:?}",
            a_dtype, b_dtype
        )));
    }

    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();

    let m = a_shape[a_ndim - 2];
    let k1 = a_shape[a_ndim - 1];
    let k2 = b_shape[b_ndim - 2];
    let n = b_shape[b_ndim - 1];

    if k1 != k2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul inner dimensions mismatch: A[...,{},{}] * B[...,{},{}]",
            m, k1, k2, n
        )));
    }

    // Broadcast batch dimensions
    let a_batch = &a_shape[..a_ndim - 2];
    let b_batch = &b_shape[..b_ndim - 2];
    let a_batch_shape = Shape::new(if a_batch.is_empty() { vec![1] } else { a_batch.to_vec() })?;
    let b_batch_shape = Shape::new(if b_batch.is_empty() { vec![1] } else { b_batch.to_vec() })?;
    let broadcast_batch = a_batch_shape.broadcast_with(&b_batch_shape)?;

    // Build output shape: broadcast_batch_dims + [M, N]
    let mut out_dims: Vec<usize> = broadcast_batch.dims().to_vec();
    // If both inputs were 2D (no batch dims), out is 2D
    if a_batch.is_empty() && b_batch.is_empty() {
        out_dims.clear();
    }
    out_dims.push(m);
    out_dims.push(n);

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Matmul,
        inputs: vec![a_id, b_id],
        out_shape: Shape::new(out_dims)?,
        out_dtype: a_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Softmax along last dimension. Supports any shape with at least 1 dim.
/// Leading dims are treated as independent rows (flattened).
pub fn softmax(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    if shape.is_empty() {
        return Err(GpuError::InvalidTensor(
            "softmax requires at least 1D tensor".to_string()
        ));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Softmax,
        inputs: vec![input_id],
        out_shape: Shape::new(shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Transpose last two dimensions: [..., rows, cols] → [..., cols, rows].
/// General transpose: swap dimensions dim0 and dim1.
pub fn transpose_dims(rt: &mut LazyRuntime, input_id: u64, dim0: usize, dim1: usize) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    let ndim = shape.len();

    if dim0 >= ndim || dim1 >= ndim {
        return Err(GpuError::InvalidTensor(format!(
            "transpose dims ({}, {}) out of range for {}D tensor", dim0, dim1, ndim
        )));
    }
    if dim0 == dim1 {
        // No-op: return the input tensor unchanged
        return Ok(input_id);
    }

    let mut out_shape = shape.clone();
    out_shape.swap(dim0, dim1);
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Transpose { dim0, dim1 },
        inputs: vec![input_id],
        out_shape: Shape::new(out_shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Transpose: swap the last two dimensions (backward-compatible shorthand).
pub fn transpose(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let shape = rt.shape(input_id)?;
    let ndim = shape.len();
    if ndim < 2 {
        return Err(GpuError::InvalidTensor(format!(
            "transpose requires at least 2D tensor, got {:?}", shape
        )));
    }
    transpose_dims(rt, input_id, ndim - 2, ndim - 1)
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
        out_shape: Shape::new(shape)?,
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
        out_shape: Shape::new(new_shape)?,
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

    if input_shape.is_empty() {
        return Err(GpuError::InvalidTensor("layer_norm requires at least 1D input".to_string()));
    }
    let cols = input_shape[input_shape.len() - 1];

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
        out_shape: Shape::new(input_shape)?,
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
    if indices_shape.is_empty() {
        return Err(GpuError::InvalidTensor("embedding indices must have at least 1 dimension".to_string()));
    }
    if indices_dtype != DType::Int32 {
        return Err(GpuError::InvalidTensor(format!(
            "embedding indices must be Int32, got {:?}", indices_dtype
        )));
    }

    let embed_dim = weights_shape[1];

    // Output shape = indices_shape + [embed_dim]
    let mut out_dims = indices_shape.clone();
    out_dims.push(embed_dim);

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Embedding,
        inputs: vec![weights_id, indices_id],
        out_shape: Shape::new(out_dims)?,
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
        out_shape: Shape::new(out_shape)?,
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
        out_shape: Shape::new(out_shape)?,
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
        out_shape: Shape::new(input_shape)?,
        out_dtype: input_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Softmax with causal (upper-triangle) mask.
/// For position (row, col) where col > row, value is treated as -inf.
/// Softmax causal along last two dimensions. Supports N-D with batch dims.
/// Requires at least 2 dims. Last 2 dims are [rows, cols] for the causal mask.
pub fn softmax_causal(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    if shape.len() < 2 {
        return Err(GpuError::InvalidTensor(format!(
            "softmax_causal requires at least 2D tensor, got {:?}", shape
        )));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::SoftmaxCausal,
        inputs: vec![input_id],
        out_shape: Shape::new(shape)?,
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
        out_shape: Shape::new(out_shape)?,
        out_dtype: DType::Int32,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Sum reduction along last dimension. N-D supported: [..., cols] -> [...].
/// For 1D input [n] -> [1].
pub fn sum(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let shape = rt.shape(input_id)?;
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    if shape.is_empty() {
        return Err(GpuError::InvalidTensor("sum requires at least 1D tensor".into()));
    }
    let mut out_shape = shape[..shape.len() - 1].to_vec();
    if out_shape.is_empty() { out_shape = vec![1]; }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Sum,
        inputs: vec![input_id],
        out_shape: Shape::new(out_shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Mean reduction along last dimension. N-D supported: [..., cols] -> [...].
/// For 1D input [n] -> [1].
pub fn mean(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let shape = rt.shape(input_id)?;
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    if shape.is_empty() {
        return Err(GpuError::InvalidTensor("mean requires at least 1D tensor".into()));
    }
    let mut out_shape = shape[..shape.len() - 1].to_vec();
    if out_shape.is_empty() { out_shape = vec![1]; }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Mean,
        inputs: vec![input_id],
        out_shape: Shape::new(out_shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
/// Q: [..., q_len, d_k], K: [..., kv_len, d_k], V: [..., kv_len, d_v]
/// Output: [..., q_len, d_v]
pub fn attention(rt: &mut LazyRuntime, q_id: u64, k_id: u64, v_id: u64) -> Result<u64> {
    let q_shape = rt.shape(q_id)?;
    let k_shape = rt.shape(k_id)?;
    let v_shape = rt.shape(v_id)?;

    if q_shape.len() < 2 || k_shape.len() < 2 || v_shape.len() < 2 {
        return Err(GpuError::InvalidTensor(
            "attention requires at least 2D tensors for Q, K, V".to_string()
        ));
    }

    let q_ndim = q_shape.len();
    let k_ndim = k_shape.len();
    let v_ndim = v_shape.len();

    let d_k = q_shape[q_ndim - 1];
    if k_shape[k_ndim - 1] != d_k {
        return Err(GpuError::InvalidTensor(format!(
            "Q and K must have same d_k: {} vs {}", d_k, k_shape[k_ndim - 1]
        )));
    }
    let kv_len = k_shape[k_ndim - 2];
    if v_shape[v_ndim - 2] != kv_len {
        return Err(GpuError::InvalidTensor(format!(
            "K and V must have same kv_len: {} vs {}", kv_len, v_shape[v_ndim - 2]
        )));
    }

    // K^T: [..., kv_len, d_k] → [..., d_k, kv_len]
    let kt_id = transpose(rt, k_id)?;
    // scores = Q @ K^T: [..., q_len, d_k] @ [..., d_k, kv_len] → [..., q_len, kv_len]
    let scores_id = matmul(rt, q_id, kt_id)?;
    // Scale by 1/sqrt(d_k)
    let scale = 1.0 / (d_k as f32).sqrt();
    let scaled_scores_id = scalar_mul(rt, scores_id, scale)?;
    // softmax along last dimension
    let attn_weights_id = softmax(rt, scaled_scores_id)?;
    // output = attn_weights @ V: [..., q_len, kv_len] @ [..., kv_len, d_v] → [..., q_len, d_v]
    let output_id = matmul(rt, attn_weights_id, v_id)?;

    Ok(output_id)
}

/// Causal scaled dot-product attention: softmax_causal(Q @ K^T / sqrt(d_k)) @ V
/// Q: [..., q_len, d_k], K: [..., kv_len, d_k], V: [..., kv_len, d_v]
/// Output: [..., q_len, d_v]
pub fn attention_causal(rt: &mut LazyRuntime, q_id: u64, k_id: u64, v_id: u64) -> Result<u64> {
    let q_shape = rt.shape(q_id)?;
    let k_shape = rt.shape(k_id)?;
    let v_shape = rt.shape(v_id)?;

    if q_shape.len() < 2 || k_shape.len() < 2 || v_shape.len() < 2 {
        return Err(GpuError::InvalidTensor(
            "attention_causal requires at least 2D tensors".to_string()
        ));
    }

    let q_ndim = q_shape.len();
    let k_ndim = k_shape.len();
    let v_ndim = v_shape.len();

    let d_k = q_shape[q_ndim - 1];
    if k_shape[k_ndim - 1] != d_k {
        return Err(GpuError::InvalidTensor(format!(
            "Q and K must have same d_k: {} vs {}", d_k, k_shape[k_ndim - 1]
        )));
    }
    let kv_len = k_shape[k_ndim - 2];
    if v_shape[v_ndim - 2] != kv_len {
        return Err(GpuError::InvalidTensor(format!(
            "K and V must have same kv_len: {} vs {}", kv_len, v_shape[v_ndim - 2]
        )));
    }

    // K^T: [..., kv_len, d_k] → [..., d_k, kv_len]
    let kt_id = transpose(rt, k_id)?;
    // scores = Q @ K^T: [..., q_len, d_k] @ [..., d_k, kv_len] → [..., q_len, kv_len]
    let scores_id = matmul(rt, q_id, kt_id)?;
    // Scale by 1/sqrt(d_k)
    let scale = 1.0 / (d_k as f32).sqrt();
    let scaled_scores_id = scalar_mul(rt, scores_id, scale)?;
    // Causal softmax (masks future positions)
    let attn_weights_id = softmax_causal(rt, scaled_scores_id)?;
    // output = attn_weights @ V: [..., q_len, kv_len] @ [..., kv_len, d_v] → [..., q_len, d_v]
    let output_id = matmul(rt, attn_weights_id, v_id)?;

    Ok(output_id)
}

/// Ternary conditional select: where(cond, x, y) — select x where cond != 0, else y.
/// All three inputs broadcast together.
pub fn where_cond(rt: &mut LazyRuntime, cond_id: u64, x_id: u64, y_id: u64) -> Result<u64> {
    let cond_dtype = rt.dtype(cond_id)?;
    validate_compute_dtype(cond_dtype)?;
    let x_dtype = rt.dtype(x_id)?;
    validate_compute_dtype(x_dtype)?;
    let y_dtype = rt.dtype(y_id)?;
    validate_compute_dtype(y_dtype)?;

    if x_dtype != y_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "where: x and y dtype mismatch: {:?} vs {:?}", x_dtype, y_dtype
        )));
    }

    let cond_shape = Shape::new(rt.shape(cond_id)?)?;
    let x_shape = Shape::new(rt.shape(x_id)?)?;
    let y_shape = Shape::new(rt.shape(y_id)?)?;

    // Broadcast all three shapes together
    let out_shape = cond_shape.broadcast_with(&x_shape)?;
    let out_shape = out_shape.broadcast_with(&y_shape)?;

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Where,
        inputs: vec![cond_id, x_id, y_id],
        out_shape,
        out_dtype: x_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Masked fill: set elements to `value` where mask is true (nonzero).
/// Input and mask broadcast together.
pub fn masked_fill(rt: &mut LazyRuntime, input_id: u64, mask_id: u64, value: f32) -> Result<u64> {
    let in_dtype = rt.dtype(input_id)?;
    validate_compute_dtype(in_dtype)?;
    let mask_dtype = rt.dtype(mask_id)?;
    validate_compute_dtype(mask_dtype)?;

    let in_shape = Shape::new(rt.shape(input_id)?)?;
    let mask_shape = Shape::new(rt.shape(mask_id)?)?;
    let out_shape = in_shape.broadcast_with(&mask_shape)?;

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::MaskedFill { value },
        inputs: vec![input_id, mask_id],
        out_shape,
        out_dtype: in_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Upper triangular: zero elements below the diagonal.
/// Input must be at least 2D. Operates on last two dims.
pub fn triu(rt: &mut LazyRuntime, input_id: u64, diagonal: i32) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    if shape.len() < 2 {
        return Err(GpuError::InvalidTensor(
            "triu requires at least 2D tensor".to_string()
        ));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Triu { diagonal },
        inputs: vec![input_id],
        out_shape: Shape::new(shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Lower triangular: zero elements above the diagonal.
/// Input must be at least 2D. Operates on last two dims.
pub fn tril(rt: &mut LazyRuntime, input_id: u64, diagonal: i32) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;
    let shape = rt.shape(input_id)?;
    if shape.len() < 2 {
        return Err(GpuError::InvalidTensor(
            "tril requires at least 2D tensor".to_string()
        ));
    }
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Tril { diagonal },
        inputs: vec![input_id],
        out_shape: Shape::new(shape)?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Gather values from input along a dimension using index tensor.
/// Input and index must be 2D. Index must be Int32.
/// Output has same shape as index, same dtype as input.
pub fn gather(rt: &mut LazyRuntime, input_id: u64, dim: usize, index_id: u64) -> Result<u64> {
    let input_dtype = rt.dtype(input_id)?;
    validate_compute_dtype(input_dtype)?;
    let input_shape = rt.shape(input_id)?;
    let index_shape = rt.shape(index_id)?;
    let index_dtype = rt.dtype(index_id)?;

    if index_dtype != DType::Int32 {
        return Err(GpuError::InvalidTensor(format!(
            "gather indices must be Int32, got {:?}", index_dtype
        )));
    }
    if input_shape.len() != 2 {
        return Err(GpuError::InvalidTensor(
            "gather currently supports only 2D tensors".to_string()
        ));
    }
    if index_shape.len() != 2 {
        return Err(GpuError::InvalidTensor(
            "gather index must be 2D (same ndim as input)".to_string()
        ));
    }
    if dim > 1 {
        return Err(GpuError::InvalidTensor(format!(
            "gather dim {} not supported for 2D tensors (must be 0 or 1)", dim
        )));
    }
    // For dim=0: index rows must match input rows (the non-gather dim)
    // Actually: for gather, index shape determines output shape.
    // The non-gather dimensions must match between input and index.
    if dim == 0 && index_shape[1] != input_shape[1] {
        return Err(GpuError::InvalidTensor(format!(
            "gather dim=0: index cols {} must match input cols {}", index_shape[1], input_shape[1]
        )));
    }
    if dim == 1 && index_shape[0] != input_shape[0] {
        return Err(GpuError::InvalidTensor(format!(
            "gather dim=1: index rows {} must match input rows {}", index_shape[0], input_shape[0]
        )));
    }

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Gather { dim },
        inputs: vec![input_id, index_id],
        out_shape: Shape::new(index_shape)?,
        out_dtype: input_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Select rows/columns from input by 1D index tensor.
/// Input must be 2D, index must be 1D Int32.
pub fn index_select(rt: &mut LazyRuntime, input_id: u64, dim: usize, index_id: u64) -> Result<u64> {
    let input_dtype = rt.dtype(input_id)?;
    validate_compute_dtype(input_dtype)?;
    let input_shape = rt.shape(input_id)?;
    let index_shape = rt.shape(index_id)?;
    let index_dtype = rt.dtype(index_id)?;

    if index_dtype != DType::Int32 {
        return Err(GpuError::InvalidTensor(format!(
            "index_select indices must be Int32, got {:?}", index_dtype
        )));
    }
    if input_shape.len() != 2 {
        return Err(GpuError::InvalidTensor(
            "index_select currently supports only 2D tensors".to_string()
        ));
    }
    if index_shape.len() != 1 {
        return Err(GpuError::InvalidTensor(
            "index_select index must be 1D".to_string()
        ));
    }
    if dim > 1 {
        return Err(GpuError::InvalidTensor(format!(
            "index_select dim {} not supported for 2D tensors (must be 0 or 1)", dim
        )));
    }

    let num_indices = index_shape[0];
    let out_shape = if dim == 0 {
        vec![num_indices, input_shape[1]]
    } else {
        vec![input_shape[0], num_indices]
    };

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::IndexSelect { dim },
        inputs: vec![input_id, index_id],
        out_shape: Shape::new(out_shape)?,
        out_dtype: input_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

// ── CNN ops ─────────────────────────────────────────────────────────────────

/// 1D convolution.
/// input: [batch, in_channels, length], weight: [out_channels, in_channels, kernel_size]
/// output: [batch, out_channels, out_length]
pub fn conv1d(rt: &mut LazyRuntime, input_id: u64, weight_id: u64, stride: usize, padding: usize) -> Result<u64> {
    let in_dtype = rt.dtype(input_id)?;
    validate_compute_dtype(in_dtype)?;
    let w_dtype = rt.dtype(weight_id)?;
    if in_dtype != w_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "conv1d dtype mismatch: {:?} vs {:?}", in_dtype, w_dtype
        )));
    }

    let in_shape = rt.shape(input_id)?;
    let w_shape = rt.shape(weight_id)?;

    if in_shape.len() != 3 {
        return Err(GpuError::InvalidTensor(format!(
            "conv1d input must be 3D [B,C,L], got {:?}", in_shape
        )));
    }
    if w_shape.len() != 3 {
        return Err(GpuError::InvalidTensor(format!(
            "conv1d weight must be 3D [OC,IC,K], got {:?}", w_shape
        )));
    }
    if in_shape[1] != w_shape[1] {
        return Err(GpuError::InvalidTensor(format!(
            "conv1d in_channels mismatch: input {} vs weight {}", in_shape[1], w_shape[1]
        )));
    }

    let batch = in_shape[0];
    let in_length = in_shape[2];
    let out_channels = w_shape[0];
    let kernel_size = w_shape[2];

    if in_length + 2 * padding < kernel_size {
        return Err(GpuError::InvalidTensor(
            "conv1d: input too small for given kernel_size and padding".to_string()
        ));
    }
    let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Conv1d { stride, padding },
        inputs: vec![input_id, weight_id],
        out_shape: Shape::new(vec![batch, out_channels, out_length])?,
        out_dtype: in_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// 2D convolution.
/// input: [batch, in_channels, height, width], weight: [out_channels, in_channels, kh, kw]
/// output: [batch, out_channels, out_h, out_w]
pub fn conv2d(rt: &mut LazyRuntime, input_id: u64, weight_id: u64, stride: (usize, usize), padding: (usize, usize)) -> Result<u64> {
    let in_dtype = rt.dtype(input_id)?;
    validate_compute_dtype(in_dtype)?;
    let w_dtype = rt.dtype(weight_id)?;
    if in_dtype != w_dtype {
        return Err(GpuError::InvalidTensor(format!(
            "conv2d dtype mismatch: {:?} vs {:?}", in_dtype, w_dtype
        )));
    }

    let in_shape = rt.shape(input_id)?;
    let w_shape = rt.shape(weight_id)?;

    if in_shape.len() != 4 {
        return Err(GpuError::InvalidTensor(format!(
            "conv2d input must be 4D [B,C,H,W], got {:?}", in_shape
        )));
    }
    if w_shape.len() != 4 {
        return Err(GpuError::InvalidTensor(format!(
            "conv2d weight must be 4D [OC,IC,KH,KW], got {:?}", w_shape
        )));
    }
    if in_shape[1] != w_shape[1] {
        return Err(GpuError::InvalidTensor(format!(
            "conv2d in_channels mismatch: input {} vs weight {}", in_shape[1], w_shape[1]
        )));
    }

    let batch = in_shape[0];
    let in_h = in_shape[2];
    let in_w = in_shape[3];
    let out_channels = w_shape[0];
    let kh = w_shape[2];
    let kw = w_shape[3];

    if in_h + 2 * padding.0 < kh || in_w + 2 * padding.1 < kw {
        return Err(GpuError::InvalidTensor(
            "conv2d: input too small for given kernel_size and padding".to_string()
        ));
    }
    let out_h = (in_h + 2 * padding.0 - kh) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - kw) / stride.1 + 1;

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Conv2d { stride, padding },
        inputs: vec![input_id, weight_id],
        out_shape: Shape::new(vec![batch, out_channels, out_h, out_w])?,
        out_dtype: in_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Batch normalization (inference mode).
/// input: [batch, channels, ...], mean/var/weight/bias: [channels]
pub fn batch_norm(rt: &mut LazyRuntime, input_id: u64, mean_id: u64, var_id: u64, weight_id: u64, bias_id: u64, eps: f32) -> Result<u64> {
    let in_dtype = rt.dtype(input_id)?;
    validate_compute_dtype(in_dtype)?;

    let in_shape = rt.shape(input_id)?;
    if in_shape.len() < 2 {
        return Err(GpuError::InvalidTensor(format!(
            "batch_norm input must be >= 2D, got {:?}", in_shape
        )));
    }
    let channels = in_shape[1];

    // Validate all param shapes are [channels]
    for (name, id) in &[("mean", mean_id), ("var", var_id), ("weight", weight_id), ("bias", bias_id)] {
        let s = rt.shape(*id)?;
        let d = rt.dtype(*id)?;
        if d != in_dtype {
            return Err(GpuError::InvalidTensor(format!(
                "batch_norm {} dtype {:?} != input dtype {:?}", name, d, in_dtype
            )));
        }
        if s != vec![channels] {
            return Err(GpuError::InvalidTensor(format!(
                "batch_norm {} shape {:?} must be [{}]", name, s, channels
            )));
        }
    }

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::BatchNorm { eps },
        inputs: vec![input_id, mean_id, var_id, weight_id, bias_id],
        out_shape: Shape::new(in_shape)?,
        out_dtype: in_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Max pooling 2D.
/// input: [batch, channels, height, width]
pub fn max_pool2d(rt: &mut LazyRuntime, input_id: u64, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;

    let in_shape = rt.shape(input_id)?;
    if in_shape.len() != 4 {
        return Err(GpuError::InvalidTensor(format!(
            "max_pool2d input must be 4D [B,C,H,W], got {:?}", in_shape
        )));
    }

    let in_h = in_shape[2];
    let in_w = in_shape[3];
    if in_h + 2 * padding.0 < kernel_size.0 || in_w + 2 * padding.1 < kernel_size.1 {
        return Err(GpuError::InvalidTensor(
            "max_pool2d: input too small for given kernel_size and padding".to_string()
        ));
    }
    let out_h = (in_h + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::MaxPool2d { kernel_size, stride, padding },
        inputs: vec![input_id],
        out_shape: Shape::new(vec![in_shape[0], in_shape[1], out_h, out_w])?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Average pooling 2D.
/// input: [batch, channels, height, width]
pub fn avg_pool2d(rt: &mut LazyRuntime, input_id: u64, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Result<u64> {
    let dtype = rt.dtype(input_id)?;
    validate_compute_dtype(dtype)?;

    let in_shape = rt.shape(input_id)?;
    if in_shape.len() != 4 {
        return Err(GpuError::InvalidTensor(format!(
            "avg_pool2d input must be 4D [B,C,H,W], got {:?}", in_shape
        )));
    }

    let in_h = in_shape[2];
    let in_w = in_shape[3];
    if in_h + 2 * padding.0 < kernel_size.0 || in_w + 2 * padding.1 < kernel_size.1 {
        return Err(GpuError::InvalidTensor(
            "avg_pool2d: input too small for given kernel_size and padding".to_string()
        ));
    }
    let out_h = (in_h + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::AvgPool2d { kernel_size, stride, padding },
        inputs: vec![input_id],
        out_shape: Shape::new(vec![in_shape[0], in_shape[1], out_h, out_w])?,
        out_dtype: dtype,
        container_id: ContainerId::DEFAULT,
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
    fn layer_norm_1d_input_works() {
        // 1D input should now be accepted (normalize over the single dim)
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
        assert!(result.is_ok());
    }

    #[test]
    fn layer_norm_rejects_gamma_mismatch() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![2, 4], &[1.0; 8]).unwrap();
        let gamma = Tensor::from_f32(&device, vec![2], &[1.0; 2]).unwrap(); // wrong size
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

    // ── Sum tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_sum_2d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // [[1,2,3],[4,5,6]] -> sum over last dim -> [6, 15]
        let t = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let s_id = crate::ops::sum(&mut rt, id).unwrap();
        assert_eq!(rt.shape(s_id).unwrap(), vec![2]);
        rt.eval(&device, s_id).unwrap();
        let result = rt.read_f32(s_id).unwrap();
        assert!((result[0] - 6.0).abs() < 0.001);
        assert!((result[1] - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_sum_1d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // [1,2,3,4] -> sum -> [10]
        let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let s_id = crate::ops::sum(&mut rt, id).unwrap();
        assert_eq!(rt.shape(s_id).unwrap(), vec![1]);
        rt.eval(&device, s_id).unwrap();
        let result = rt.read_f32(s_id).unwrap();
        assert!((result[0] - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_sum_3d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // [2,3,4] -> sum over last dim -> [2,3]
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let t = Tensor::from_f32(&device, vec![2, 3, 4], &data).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let s_id = crate::ops::sum(&mut rt, id).unwrap();
        assert_eq!(rt.shape(s_id).unwrap(), vec![2, 3]);
        rt.eval(&device, s_id).unwrap();
        let result = rt.read_f32(s_id).unwrap();
        // Row 0: 1+2+3+4=10
        assert!((result[0] - 10.0).abs() < 0.001);
        // Row 1: 5+6+7+8=26
        assert!((result[1] - 26.0).abs() < 0.001);
        // Row 5: 21+22+23+24=90
        assert!((result[5] - 90.0).abs() < 0.001);
    }

    // ── Mean tests ────────────────────────────────────────────────────────

    #[test]
    fn test_mean_2d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // [[1,2,3],[4,5,6]] -> mean over last dim -> [2.0, 5.0]
        let t = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let m_id = crate::ops::mean(&mut rt, id).unwrap();
        assert_eq!(rt.shape(m_id).unwrap(), vec![2]);
        rt.eval(&device, m_id).unwrap();
        let result = rt.read_f32(m_id).unwrap();
        assert!((result[0] - 2.0).abs() < 0.001);
        assert!((result[1] - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_mean_1d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // [1,2,3,4] -> mean -> [2.5]
        let t = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let m_id = crate::ops::mean(&mut rt, id).unwrap();
        assert_eq!(rt.shape(m_id).unwrap(), vec![1]);
        rt.eval(&device, m_id).unwrap();
        let result = rt.read_f32(m_id).unwrap();
        assert!((result[0] - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_mean_3d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        // [2,3,4] -> mean over last dim -> [2,3]
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let t = Tensor::from_f32(&device, vec![2, 3, 4], &data).unwrap();
        let id = t.meta.id;
        rt.insert_tensor(t).unwrap();
        let m_id = crate::ops::mean(&mut rt, id).unwrap();
        assert_eq!(rt.shape(m_id).unwrap(), vec![2, 3]);
        rt.eval(&device, m_id).unwrap();
        let result = rt.read_f32(m_id).unwrap();
        // Row 0: (1+2+3+4)/4=2.5
        assert!((result[0] - 2.5).abs() < 0.001);
        // Row 1: (5+6+7+8)/4=6.5
        assert!((result[1] - 6.5).abs() < 0.001);
        // Row 5: (21+22+23+24)/4=22.5
        assert!((result[5] - 22.5).abs() < 0.001);
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

    // ── N-D stride-based kernel tests ─────────────────────────────────────

    #[test]
    fn test_add_3d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Create two [2,3,4] f32 tensors with distinct values
        let a_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..24).map(|i| (100 + i) as f32).collect();
        let a = Tensor::from_f32(&device, vec![2, 3, 4], &a_data).unwrap();
        let b = Tensor::from_f32(&device, vec![2, 3, 4], &b_data).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();

        let c_id = add(&mut rt, a_id, b_id).unwrap();
        rt.eval(&device, c_id).unwrap();

        let result = rt.read_f32(c_id).unwrap();
        assert_eq!(result.len(), 24);
        for i in 0..24 {
            assert_eq!(result[i], i as f32 + (100 + i) as f32,
                "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_add_broadcast_bias() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // [4,3] + [3] -> [4,3] (bias addition via broadcasting)
        let input_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let bias_data = vec![100.0, 200.0, 300.0];

        let input = Tensor::from_f32(&device, vec![4, 3], &input_data).unwrap();
        let bias = Tensor::from_f32(&device, vec![3], &bias_data).unwrap();
        let input_id = input.meta.id;
        let bias_id = bias.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(bias).unwrap();

        let c_id = add(&mut rt, input_id, bias_id).unwrap();

        // Verify output shape is [4,3]
        let shape = rt.shape(c_id).unwrap();
        assert_eq!(shape, vec![4, 3]);

        rt.eval(&device, c_id).unwrap();

        let result = rt.read_f32(c_id).unwrap();
        assert_eq!(result.len(), 12);
        // Each row gets the bias added: row[i] = [i*3+0+100, i*3+1+200, i*3+2+300]
        for row in 0..4 {
            for col in 0..3 {
                let idx = row * 3 + col;
                let expected = idx as f32 + bias_data[col];
                assert_eq!(result[idx], expected,
                    "Mismatch at [{},{}]: got {}, expected {}", row, col, result[idx], expected);
            }
        }
    }

    #[test]
    fn test_relu_3d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Create [2,3,4] tensor with mixed pos/neg values
        let data: Vec<f32> = (0..24).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();
        let t = Tensor::from_f32(&device, vec![2, 3, 4], &data).unwrap();
        let t_id = t.meta.id;
        rt.insert_tensor(t).unwrap();

        let r_id = relu(&mut rt, t_id).unwrap();
        rt.eval(&device, r_id).unwrap();

        let result = rt.read_f32(r_id).unwrap();
        assert_eq!(result.len(), 24);
        for i in 0..24 {
            let expected = if data[i] > 0.0 { data[i] } else { 0.0 };
            assert_eq!(result[i], expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_gelu_3d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Create [2,3] tensor with specific values
        let data = vec![0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5];
        let t = Tensor::from_f32(&device, vec![2, 3], &data).unwrap();
        let t_id = t.meta.id;
        rt.insert_tensor(t).unwrap();

        let g_id = gelu(&mut rt, t_id).unwrap();
        rt.eval(&device, g_id).unwrap();

        let result = rt.read_f32(g_id).unwrap();
        assert_eq!(result.len(), 6);

        // GELU(0) = 0
        assert!(result[0].abs() < 0.01, "GELU(0) = {}", result[0]);
        // GELU(1) ≈ 0.841
        assert!((result[1] - 0.841).abs() < 0.02, "GELU(1) = {}", result[1]);
        // GELU(-1) ≈ -0.159
        assert!((result[2] - (-0.159)).abs() < 0.02, "GELU(-1) = {}", result[2]);
    }

    #[test]
    fn test_mul_broadcast_3d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // [2,1,4] * [3,4] -> [2,3,4]
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_data: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect();

        let a = Tensor::from_f32(&device, vec![2, 1, 4], &a_data).unwrap();
        let b = Tensor::from_f32(&device, vec![3, 4], &b_data).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();

        let c_id = mul(&mut rt, a_id, b_id).unwrap();

        // Verify output shape is [2,3,4]
        let shape = rt.shape(c_id).unwrap();
        assert_eq!(shape, vec![2, 3, 4]);

        rt.eval(&device, c_id).unwrap();

        let result = rt.read_f32(c_id).unwrap();
        assert_eq!(result.len(), 24);

        // Manual computation:
        // a[2,1,4] means a[0,:,:] = [1,2,3,4] broadcast over dim 1
        // a[1,:,:] = [5,6,7,8] broadcast over dim 1
        // b[3,4] = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        // result[0,0,:] = [1*1, 2*2, 3*3, 4*4] = [1, 4, 9, 16]
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 4.0);
        assert_eq!(result[2], 9.0);
        assert_eq!(result[3], 16.0);
        // result[0,1,:] = [1*5, 2*6, 3*7, 4*8] = [5, 12, 21, 32]
        assert_eq!(result[4], 5.0);
        assert_eq!(result[5], 12.0);
        assert_eq!(result[6], 21.0);
        assert_eq!(result[7], 32.0);
    }

    #[test]
    fn test_sub_broadcast_scalar() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // [2,3] - [1] (scalar broadcast)
        let a_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let b_data: Vec<f32> = vec![5.0];

        let a = Tensor::from_f32(&device, vec![2, 3], &a_data).unwrap();
        let b = Tensor::from_f32(&device, vec![1], &b_data).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();

        let c_id = sub(&mut rt, a_id, b_id).unwrap();
        rt.eval(&device, c_id).unwrap();

        let result = rt.read_f32(c_id).unwrap();
        assert_eq!(result, &[5.0, 15.0, 25.0, 35.0, 45.0, 55.0]);
    }

    #[test]
    fn test_existing_2d_ops_unchanged() {
        // Verify that all existing 2D ops still work after the N-D kernel rewrite
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // 2D add + matmul chain
        let a = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let a_id = a.meta.id;
        let b_id = b.meta.id;
        rt.insert_tensor(a).unwrap();
        rt.insert_tensor(b).unwrap();

        // add
        let sum_id = add(&mut rt, a_id, b_id).unwrap();
        rt.eval(&device, sum_id).unwrap();
        assert_eq!(rt.read_f32(sum_id).unwrap(), &[6.0, 8.0, 10.0, 12.0]);

        // matmul
        let mm_id = matmul(&mut rt, a_id, b_id).unwrap();
        rt.eval(&device, mm_id).unwrap();
        assert_eq!(rt.read_f32(mm_id).unwrap(), &[19.0, 22.0, 43.0, 50.0]);
    }

    // ── Batched transpose tests ──────────────────────────────────────────

    #[test]
    fn test_batched_transpose_3d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // [2, 3, 4] -> transpose last 2 dims -> [2, 4, 3]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::from_f32(&device, vec![2, 3, 4], &data).unwrap();
        let t_id = t.meta.id;
        rt.insert_tensor(t).unwrap();

        let out_id = transpose(&mut rt, t_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 4, 3]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result.len(), 24);

        // Batch 0: input [3,4] row-major: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
        // Transposed [4,3]: [[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
        assert_eq!(&result[0..12], &[0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]);
        // Batch 1: input [[12,13,14,15],[16,17,18,19],[20,21,22,23]]
        // Transposed: [[12,16,20],[13,17,21],[14,18,22],[15,19,23]]
        assert_eq!(&result[12..24], &[12.0, 16.0, 20.0, 13.0, 17.0, 21.0, 14.0, 18.0, 22.0, 15.0, 19.0, 23.0]);
    }

    #[test]
    fn test_existing_2d_transpose_unchanged() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let t = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t_id = t.meta.id;
        rt.insert_tensor(t).unwrap();

        let out_id = transpose(&mut rt, t_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![3, 2]);

        rt.eval(&device, out_id).unwrap();
        assert_eq!(rt.read_f32(out_id).unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // ── Batched layer_norm tests ─────────────────────────────────────────

    #[test]
    fn test_batched_layer_norm_3d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // [2, 3, 4] -> layer_norm over last dim (4)
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let input = Tensor::from_f32(&device, vec![2, 3, 4], &data).unwrap();
        let gamma = Tensor::from_f32(&device, vec![4], &[1.0; 4]).unwrap();
        let beta = Tensor::from_f32(&device, vec![4], &[0.0; 4]).unwrap();
        let input_id = input.meta.id;
        let gamma_id = gamma.meta.id;
        let beta_id = beta.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(gamma).unwrap();
        rt.insert_tensor(beta).unwrap();

        let out_id = layer_norm(&mut rt, input_id, gamma_id, beta_id, 1e-5).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 3, 4]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result.len(), 24);

        // Each group of 4 should be normalized to mean~0, std~1
        // First row: [0,1,2,3] -> mean=1.5, std=~1.118
        // Normalized: [-1.342, -0.447, 0.447, 1.342]
        assert!((result[0] - (-1.342)).abs() < 0.05, "got {}", result[0]);
        assert!((result[1] - (-0.447)).abs() < 0.05, "got {}", result[1]);
        assert!((result[2] - 0.447).abs() < 0.05, "got {}", result[2]);
        assert!((result[3] - 1.342).abs() < 0.05, "got {}", result[3]);
    }

    #[test]
    fn test_existing_2d_layer_norm_unchanged() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![2, 4], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
        let gamma = Tensor::from_f32(&device, vec![4], &[1.0; 4]).unwrap();
        let beta = Tensor::from_f32(&device, vec![4], &[0.0; 4]).unwrap();
        let input_id = input.meta.id;
        let gamma_id = gamma.meta.id;
        let beta_id = beta.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(gamma).unwrap();
        rt.insert_tensor(beta).unwrap();

        let out_id = layer_norm(&mut rt, input_id, gamma_id, beta_id, 1e-5).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 4]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result.len(), 8);
        // Row 0: [0,1,2,3] -> normalized
        assert!((result[0] - (-1.342)).abs() < 0.05);
        assert!((result[3] - 1.342).abs() < 0.05);
    }

    // ── Batched embedding tests ──────────────────────────────────────────

    #[test]
    fn test_batched_embedding_2d_indices() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // weights [5, 3], indices [2, 4] -> [2, 4, 3]
        let weights_data: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let weights = Tensor::from_f32(&device, vec![5, 3], &weights_data).unwrap();
        // indices: [[0,1,2,3],[4,0,1,2]]
        let indices_data: Vec<i32> = vec![0, 1, 2, 3, 4, 0, 1, 2];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, indices_data.len() * 4)
        };
        let indices = crate::tensor::Tensor::from_data(&device, vec![2, 4], DType::Int32, indices_bytes).unwrap();

        let w_id = weights.meta.id;
        let i_id = indices.meta.id;
        rt.insert_tensor(weights).unwrap();
        rt.insert_tensor(indices).unwrap();

        let out_id = embedding(&mut rt, w_id, i_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 4, 3]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result.len(), 24);

        // Index 0 -> weights[0] = [0,1,2]
        assert_eq!(&result[0..3], &[0.0, 1.0, 2.0]);
        // Index 1 -> weights[1] = [3,4,5]
        assert_eq!(&result[3..6], &[3.0, 4.0, 5.0]);
        // Index 4 (second batch, first element) -> weights[4] = [12,13,14]
        assert_eq!(&result[12..15], &[12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_existing_1d_embedding_unchanged() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let weights_data: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let weights = Tensor::from_f32(&device, vec![5, 3], &weights_data).unwrap();
        let indices_data: Vec<i32> = vec![0, 2, 4];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, indices_data.len() * 4)
        };
        let indices = crate::tensor::Tensor::from_data(&device, vec![3], DType::Int32, indices_bytes).unwrap();

        let w_id = weights.meta.id;
        let i_id = indices.meta.id;
        rt.insert_tensor(weights).unwrap();
        rt.insert_tensor(indices).unwrap();

        let out_id = embedding(&mut rt, w_id, i_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![3, 3]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result.len(), 9);
        assert_eq!(&result[0..3], &[0.0, 1.0, 2.0]);   // index 0
        assert_eq!(&result[3..6], &[6.0, 7.0, 8.0]);   // index 2
        assert_eq!(&result[6..9], &[12.0, 13.0, 14.0]); // index 4
    }

    // ── Batched attention tests ──────────────────────────────────────────

    #[test]
    fn test_batched_attention_3d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Q [2, 3, 4], K [2, 3, 4], V [2, 3, 4] -> [2, 3, 4]
        // 2 batch elements, each seq=3, d_k=d_v=4
        let q_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
        let v_data: Vec<f32> = (0..24).map(|i| i as f32).collect();

        let q = Tensor::from_f32(&device, vec![2, 3, 4], &q_data).unwrap();
        let k = Tensor::from_f32(&device, vec![2, 3, 4], &k_data).unwrap();
        let v = Tensor::from_f32(&device, vec![2, 3, 4], &v_data).unwrap();
        let q_id = q.meta.id;
        let k_id = k.meta.id;
        let v_id = v.meta.id;
        rt.insert_tensor(q).unwrap();
        rt.insert_tensor(k).unwrap();
        rt.insert_tensor(v).unwrap();

        let out_id = attention(&mut rt, q_id, k_id, v_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 3, 4]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result.len(), 24);
        // Just verify all values are finite (no NaN/Inf)
        for &v in &result {
            assert!(v.is_finite(), "Attention produced non-finite value: {}", v);
        }
    }

    #[test]
    fn test_batched_attention_causal_3d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Batched causal attention: Q,K,V [2, 3, 4]
        let q_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
        let v_data: Vec<f32> = (0..24).map(|i| i as f32).collect();

        let q = Tensor::from_f32(&device, vec![2, 3, 4], &q_data).unwrap();
        let k = Tensor::from_f32(&device, vec![2, 3, 4], &k_data).unwrap();
        let v = Tensor::from_f32(&device, vec![2, 3, 4], &v_data).unwrap();
        let q_id = q.meta.id;
        let k_id = k.meta.id;
        let v_id = v.meta.id;
        rt.insert_tensor(q).unwrap();
        rt.insert_tensor(k).unwrap();
        rt.insert_tensor(v).unwrap();

        let out_id = attention_causal(&mut rt, q_id, k_id, v_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 3, 4]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result.len(), 24);
        for &v in &result {
            assert!(v.is_finite(), "Causal attention produced non-finite value: {}", v);
        }
    }

    #[test]
    fn test_attention_rejects_1d() {
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

        assert!(attention(&mut rt, q_id, k_id, v_id).is_err());
    }

    #[test]
    fn test_abs() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![3], &[-1.0, 2.0, -3.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        let r_id = abs(&mut rt, a_id).unwrap();
        rt.eval(&device, r_id).unwrap();
        assert_eq!(rt.read_f32(r_id).unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sign() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![3], &[-5.0, 0.0, 3.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        let r_id = sign(&mut rt, a_id).unwrap();
        rt.eval(&device, r_id).unwrap();
        assert_eq!(rt.read_f32(r_id).unwrap(), &[-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_pow() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![3], &[2.0, 3.0, 4.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        let r_id = pow(&mut rt, a_id, 2.0).unwrap();
        rt.eval(&device, r_id).unwrap();
        assert_eq!(rt.read_f32(r_id).unwrap(), &[4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_clamp() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![3], &[1.0, 5.0, 10.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        let r_id = clamp(&mut rt, a_id, 2.0, 8.0).unwrap();
        rt.eval(&device, r_id).unwrap();
        assert_eq!(rt.read_f32(r_id).unwrap(), &[2.0, 5.0, 8.0]);
    }

    #[test]
    fn test_where() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let cond = Tensor::from_f32(&device, vec![3], &[1.0, 0.0, 1.0]).unwrap();
        let x = Tensor::from_f32(&device, vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let y = Tensor::from_f32(&device, vec![3], &[10.0, 20.0, 30.0]).unwrap();
        let cond_id = cond.meta.id;
        let x_id = x.meta.id;
        let y_id = y.meta.id;
        rt.insert_tensor(cond).unwrap();
        rt.insert_tensor(x).unwrap();
        rt.insert_tensor(y).unwrap();
        let r_id = where_cond(&mut rt, cond_id, x_id, y_id).unwrap();
        rt.eval(&device, r_id).unwrap();
        assert_eq!(rt.read_f32(r_id).unwrap(), &[1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_masked_fill() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let input = Tensor::from_f32(&device, vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let mask = Tensor::from_f32(&device, vec![3], &[1.0, 0.0, 1.0]).unwrap();
        let input_id = input.meta.id;
        let mask_id = mask.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(mask).unwrap();
        let r_id = masked_fill(&mut rt, input_id, mask_id, f32::NEG_INFINITY).unwrap();
        rt.eval(&device, r_id).unwrap();
        let result = rt.read_f32(r_id).unwrap();
        assert!(result[0].is_infinite() && result[0] < 0.0);
        assert_eq!(result[1], 2.0);
        assert!(result[2].is_infinite() && result[2] < 0.0);
    }

    #[test]
    fn test_triu() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![3, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        let r_id = triu(&mut rt, a_id, 0).unwrap();
        rt.eval(&device, r_id).unwrap();
        assert_eq!(rt.read_f32(r_id).unwrap(), &[
            1.0, 2.0, 3.0,
            0.0, 5.0, 6.0,
            0.0, 0.0, 9.0,
        ]);
    }

    #[test]
    fn test_tril() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![3, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        let r_id = tril(&mut rt, a_id, 0).unwrap();
        rt.eval(&device, r_id).unwrap();
        assert_eq!(rt.read_f32(r_id).unwrap(), &[
            1.0, 0.0, 0.0,
            4.0, 5.0, 0.0,
            7.0, 8.0, 9.0,
        ]);
    }

    #[test]
    fn test_triu_diagonal_offset() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![3, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        // diagonal=1: zero on and below main diagonal
        let r_id = triu(&mut rt, a_id, 1).unwrap();
        rt.eval(&device, r_id).unwrap();
        assert_eq!(rt.read_f32(r_id).unwrap(), &[
            0.0, 2.0, 3.0,
            0.0, 0.0, 6.0,
            0.0, 0.0, 0.0,
        ]);
    }

    #[test]
    fn test_triu_rejects_1d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let a = Tensor::from_f32(&device, vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let a_id = a.meta.id;
        rt.insert_tensor(a).unwrap();
        assert!(triu(&mut rt, a_id, 0).is_err());
    }

    // ── Gather tests ─────────────────────────────────────────────────────

    #[test]
    fn test_gather_dim1() {
        // input [[1,2,3],[4,5,6]], indices [[0,2],[1,0]] → [[1,3],[5,4]]
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices_data: [i32; 4] = [0, 2, 1, 0];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, 16)
        };
        let indices = Tensor::from_data(&device, vec![2, 2], DType::Int32, indices_bytes).unwrap();
        let i_id = input.meta.id;
        let idx_id = indices.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(indices).unwrap();

        let out_id = gather(&mut rt, i_id, 1, idx_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result, &[1.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn test_gather_dim0() {
        // input [[1,2],[3,4],[5,6]], indices [[0,2],[1,0]] → [[1,6],[3,2]]
        // dim=0: output[i][j] = input[indices[i][j]][j]
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices_data: [i32; 4] = [0, 2, 1, 0];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, 16)
        };
        let indices = Tensor::from_data(&device, vec![2, 2], DType::Int32, indices_bytes).unwrap();
        let i_id = input.meta.id;
        let idx_id = indices.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(indices).unwrap();

        let out_id = gather(&mut rt, i_id, 0, idx_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result, &[1.0, 6.0, 3.0, 2.0]);
    }

    #[test]
    fn test_gather_rejects_non_int32_indices() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let input = Tensor::from_f32(&device, vec![2, 3], &[1.0; 6]).unwrap();
        let bad_indices = Tensor::from_f32(&device, vec![2, 2], &[0.0; 4]).unwrap();
        let i_id = input.meta.id;
        let idx_id = bad_indices.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(bad_indices).unwrap();
        assert!(gather(&mut rt, i_id, 1, idx_id).is_err());
    }

    #[test]
    fn test_gather_rejects_1d() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let input = Tensor::from_f32(&device, vec![6], &[1.0; 6]).unwrap();
        let indices_data: [i32; 2] = [0, 1];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, 8)
        };
        let indices = Tensor::from_data(&device, vec![2], DType::Int32, indices_bytes).unwrap();
        let i_id = input.meta.id;
        let idx_id = indices.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(indices).unwrap();
        assert!(gather(&mut rt, i_id, 0, idx_id).is_err());
    }

    // ── IndexSelect tests ────────────────────────────────────────────────

    #[test]
    fn test_index_select_dim0() {
        // input [[1,2,3],[4,5,6],[7,8,9]], indices [0,2] → [[1,2,3],[7,8,9]]
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![3, 3], &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
        ]).unwrap();
        let indices_data: [i32; 2] = [0, 2];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, 8)
        };
        let indices = Tensor::from_data(&device, vec![2], DType::Int32, indices_bytes).unwrap();
        let i_id = input.meta.id;
        let idx_id = indices.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(indices).unwrap();

        let out_id = index_select(&mut rt, i_id, 0, idx_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 3]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result, &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_index_select_dim1() {
        // input [[1,2,3],[4,5,6]], indices [0,2] → [[1,3],[4,6]]
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices_data: [i32; 2] = [0, 2];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, 8)
        };
        let indices = Tensor::from_data(&device, vec![2], DType::Int32, indices_bytes).unwrap();
        let i_id = input.meta.id;
        let idx_id = indices.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(indices).unwrap();

        let out_id = index_select(&mut rt, i_id, 1, idx_id).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![2, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result, &[1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_index_select_rejects_non_int32() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let input = Tensor::from_f32(&device, vec![3, 2], &[1.0; 6]).unwrap();
        let bad_indices = Tensor::from_f32(&device, vec![2], &[0.0, 1.0]).unwrap();
        let i_id = input.meta.id;
        let idx_id = bad_indices.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(bad_indices).unwrap();
        assert!(index_select(&mut rt, i_id, 0, idx_id).is_err());
    }

    #[test]
    fn test_index_select_rejects_2d_index() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();
        let input = Tensor::from_f32(&device, vec![3, 2], &[1.0; 6]).unwrap();
        let indices_data: [i32; 4] = [0, 1, 0, 1];
        let indices_bytes = unsafe {
            std::slice::from_raw_parts(indices_data.as_ptr() as *const u8, 16)
        };
        let indices = Tensor::from_data(&device, vec![2, 2], DType::Int32, indices_bytes).unwrap();
        let i_id = input.meta.id;
        let idx_id = indices.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(indices).unwrap();
        assert!(index_select(&mut rt, i_id, 0, idx_id).is_err());
    }

    // ── CNN ops tests ─────────────────────────────────────────────────────

    #[test]
    fn test_conv1d() {
        // input: [1,1,5], weight: [1,1,3], stride=1, pad=0 -> [1,1,3]
        // input = [1,2,3,4,5], weight = [1,1,1]
        // out[0] = 1*1 + 2*1 + 3*1 = 6
        // out[1] = 2*1 + 3*1 + 4*1 = 9
        // out[2] = 3*1 + 4*1 + 5*1 = 12
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let weight = Tensor::from_f32(&device, vec![1, 1, 3], &[1.0, 1.0, 1.0]).unwrap();
        let in_id = input.meta.id;
        let w_id = weight.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(weight).unwrap();

        let out_id = conv1d(&mut rt, in_id, w_id, 1, 0).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![1, 1, 3]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result, &[6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_conv1d_stride2() {
        // input: [1,1,6], weight: [1,1,3], stride=2, pad=0 -> [1,1,2]
        // out_length = (6 + 0 - 3) / 2 + 1 = 2
        // out[0] = 1+2+3 = 6, out[1] = 3+4+5 = 12
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![1, 1, 6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let weight = Tensor::from_f32(&device, vec![1, 1, 3], &[1.0, 1.0, 1.0]).unwrap();
        let in_id = input.meta.id;
        let w_id = weight.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(weight).unwrap();

        let out_id = conv1d(&mut rt, in_id, w_id, 2, 0).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![1, 1, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result, &[6.0, 12.0]);
    }

    #[test]
    fn test_conv1d_multichannel() {
        // input: [1,2,3], weight: [1,2,2], stride=1, pad=0 -> [1,1,2]
        // 2 input channels, 1 output channel, kernel_size=2
        // input ch0 = [1,2,3], ch1 = [4,5,6]
        // weight ch0 = [1,0], ch1 = [0,1]
        // out[0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
        // out[1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let weight = Tensor::from_f32(&device, vec![1, 2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let in_id = input.meta.id;
        let w_id = weight.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(weight).unwrap();

        let out_id = conv1d(&mut rt, in_id, w_id, 1, 0).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![1, 1, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result, &[6.0, 8.0]);
    }

    #[test]
    fn test_conv1d_shape_validation() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Input must be 3D
        let input_2d = Tensor::from_f32(&device, vec![2, 3], &[1.0; 6]).unwrap();
        let weight = Tensor::from_f32(&device, vec![1, 3, 2], &[1.0; 6]).unwrap();
        let in_id = input_2d.meta.id;
        let w_id = weight.meta.id;
        rt.insert_tensor(input_2d).unwrap();
        rt.insert_tensor(weight).unwrap();
        assert!(conv1d(&mut rt, in_id, w_id, 1, 0).is_err());

        // in_channels mismatch
        let input = Tensor::from_f32(&device, vec![1, 2, 5], &[1.0; 10]).unwrap();
        let weight2 = Tensor::from_f32(&device, vec![1, 3, 2], &[1.0; 6]).unwrap();
        let in_id = input.meta.id;
        let w_id = weight2.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(weight2).unwrap();
        assert!(conv1d(&mut rt, in_id, w_id, 1, 0).is_err());
    }

    #[test]
    fn test_conv2d() {
        // input: [1,1,3,3], weight: [1,1,2,2], stride=1, pad=0 -> [1,1,2,2]
        // input = [[1,2,3],[4,5,6],[7,8,9]]
        // weight = [[1,0],[0,1]]
        // out[0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
        // out[0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
        // out[1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
        // out[1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![1, 1, 3, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let weight = Tensor::from_f32(&device, vec![1, 1, 2, 2],
            &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let in_id = input.meta.id;
        let w_id = weight.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(weight).unwrap();

        let out_id = conv2d(&mut rt, in_id, w_id, (1, 1), (0, 0)).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![1, 1, 2, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result, &[6.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn test_conv2d_all_ones() {
        // input: [1,1,3,3] all ones, weight: [1,1,2,2] all ones -> out = 4.0 for all
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![1, 1, 3, 3], &[1.0; 9]).unwrap();
        let weight = Tensor::from_f32(&device, vec![1, 1, 2, 2], &[1.0; 4]).unwrap();
        let in_id = input.meta.id;
        let w_id = weight.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(weight).unwrap();

        let out_id = conv2d(&mut rt, in_id, w_id, (1, 1), (0, 0)).unwrap();
        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result, &[4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_conv2d_shape_validation() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Input must be 4D
        let input_3d = Tensor::from_f32(&device, vec![1, 1, 3], &[1.0; 3]).unwrap();
        let weight = Tensor::from_f32(&device, vec![1, 1, 2, 2], &[1.0; 4]).unwrap();
        let in_id = input_3d.meta.id;
        let w_id = weight.meta.id;
        rt.insert_tensor(input_3d).unwrap();
        rt.insert_tensor(weight).unwrap();
        assert!(conv2d(&mut rt, in_id, w_id, (1, 1), (0, 0)).is_err());
    }

    #[test]
    fn test_batch_norm() {
        // input: [1,2,3] (batch=1, channels=2, spatial=3)
        // channel 0: [1,2,3] with mean=2, var=1, weight=1, bias=0
        //   -> normalized: [(1-2)/sqrt(1+1e-5)*1+0, (2-2)/..., (3-2)/...]
        //   = [-1/sqrt(1.00001), 0, 1/sqrt(1.00001)]
        //   ~ [-0.99999, 0.0, 0.99999]
        // channel 1: [4,5,6] with mean=5, var=1, weight=2, bias=1
        //   -> normalized: [(4-5)/sqrt(1+1e-5)*2+1, (5-5)*2+1, (6-5)*2+1]
        //   = [-2/sqrt(1.00001)+1, 1, 2/sqrt(1.00001)+1]
        //   ~ [-1.0, 1.0, 3.0]
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input = Tensor::from_f32(&device, vec![1, 2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mean = Tensor::from_f32(&device, vec![2], &[2.0, 5.0]).unwrap();
        let var = Tensor::from_f32(&device, vec![2], &[1.0, 1.0]).unwrap();
        let weight = Tensor::from_f32(&device, vec![2], &[1.0, 2.0]).unwrap();
        let bias = Tensor::from_f32(&device, vec![2], &[0.0, 1.0]).unwrap();
        let in_id = input.meta.id;
        let m_id = mean.meta.id;
        let v_id = var.meta.id;
        let w_id = weight.meta.id;
        let b_id = bias.meta.id;
        rt.insert_tensor(input).unwrap();
        rt.insert_tensor(mean).unwrap();
        rt.insert_tensor(var).unwrap();
        rt.insert_tensor(weight).unwrap();
        rt.insert_tensor(bias).unwrap();

        let out_id = batch_norm(&mut rt, in_id, m_id, v_id, w_id, b_id, 1e-5).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![1, 2, 3]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        // Channel 0: [-1, 0, 1] (approx)
        assert!((result[0] - (-1.0)).abs() < 0.001, "got {}", result[0]);
        assert!((result[1] - 0.0).abs() < 0.001, "got {}", result[1]);
        assert!((result[2] - 1.0).abs() < 0.001, "got {}", result[2]);
        // Channel 1: [-1, 1, 3] (approx)
        assert!((result[3] - (-1.0)).abs() < 0.001, "got {}", result[3]);
        assert!((result[4] - 1.0).abs() < 0.001, "got {}", result[4]);
        assert!((result[5] - 3.0).abs() < 0.001, "got {}", result[5]);
    }

    #[test]
    fn test_batch_norm_shape_validation() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Input must be >= 2D
        let input_1d = Tensor::from_f32(&device, vec![3], &[1.0; 3]).unwrap();
        let mean = Tensor::from_f32(&device, vec![3], &[0.0; 3]).unwrap();
        let var = Tensor::from_f32(&device, vec![3], &[1.0; 3]).unwrap();
        let w = Tensor::from_f32(&device, vec![3], &[1.0; 3]).unwrap();
        let b = Tensor::from_f32(&device, vec![3], &[0.0; 3]).unwrap();
        let ids: Vec<u64> = vec![input_1d.meta.id, mean.meta.id, var.meta.id, w.meta.id, b.meta.id];
        rt.insert_tensor(input_1d).unwrap();
        rt.insert_tensor(mean).unwrap();
        rt.insert_tensor(var).unwrap();
        rt.insert_tensor(w).unwrap();
        rt.insert_tensor(b).unwrap();
        assert!(batch_norm(&mut rt, ids[0], ids[1], ids[2], ids[3], ids[4], 1e-5).is_err());
    }

    #[test]
    fn test_max_pool2d() {
        // input: [1,1,4,4], kernel=2, stride=2, pad=0 -> [1,1,2,2]
        // [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        // pool[0,0] = max(1,2,5,6) = 6
        // pool[0,1] = max(3,4,7,8) = 8
        // pool[1,0] = max(9,10,13,14) = 14
        // pool[1,1] = max(11,12,15,16) = 16
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let input = Tensor::from_f32(&device, vec![1, 1, 4, 4], &data).unwrap();
        let in_id = input.meta.id;
        rt.insert_tensor(input).unwrap();

        let out_id = max_pool2d(&mut rt, in_id, (2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![1, 1, 2, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert_eq!(result, &[6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_max_pool2d_shape_validation() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        // Input must be 4D
        let input_3d = Tensor::from_f32(&device, vec![1, 4, 4], &[1.0; 16]).unwrap();
        let in_id = input_3d.meta.id;
        rt.insert_tensor(input_3d).unwrap();
        assert!(max_pool2d(&mut rt, in_id, (2, 2), (2, 2), (0, 0)).is_err());
    }

    #[test]
    fn test_avg_pool2d() {
        // input: [1,1,4,4], kernel=2, stride=2, pad=0 -> [1,1,2,2]
        // pool[0,0] = avg(1,2,5,6) = 3.5
        // pool[0,1] = avg(3,4,7,8) = 5.5
        // pool[1,0] = avg(9,10,13,14) = 11.5
        // pool[1,1] = avg(11,12,15,16) = 13.5
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let input = Tensor::from_f32(&device, vec![1, 1, 4, 4], &data).unwrap();
        let in_id = input.meta.id;
        rt.insert_tensor(input).unwrap();

        let out_id = avg_pool2d(&mut rt, in_id, (2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(rt.shape(out_id).unwrap(), vec![1, 1, 2, 2]);

        rt.eval(&device, out_id).unwrap();
        let result = rt.read_f32(out_id).unwrap();
        assert!((result[0] - 3.5).abs() < 0.001);
        assert!((result[1] - 5.5).abs() < 0.001);
        assert!((result[2] - 11.5).abs() < 0.001);
        assert!((result[3] - 13.5).abs() < 0.001);
    }

    #[test]
    fn test_avg_pool2d_shape_validation() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut rt = LazyRuntime::new();

        let input_3d = Tensor::from_f32(&device, vec![1, 4, 4], &[1.0; 16]).unwrap();
        let in_id = input_3d.meta.id;
        rt.insert_tensor(input_3d).unwrap();
        assert!(avg_pool2d(&mut rt, in_id, (2, 2), (2, 2), (0, 0)).is_err());
    }
}
