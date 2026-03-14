use crate::error::{GpuError, Result};
use crate::graph::{OpKind, OpNode};
use crate::lazy::LazyRuntime;
use crate::scheduler::ContainerId;
use crate::tensor::Shape;

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

    let a_dtype = rt.dtype(a_id)?;
    let b_dtype = rt.dtype(b_id)?;
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
    let shape = rt.shape(input_id)?;
    let out_dtype = rt.dtype(input_id)?;
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
    let a_shape = rt.shape(a_id)?;
    let b_shape = rt.shape(b_id)?;

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul requires 2D tensors, got {:?} and {:?}",
            a_shape, b_shape
        )));
    }

    let a_dtype = rt.dtype(a_id)?;
    let b_dtype = rt.dtype(b_id)?;
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
    let shape = rt.shape(input_id)?;
    if shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "softmax requires 2D tensor, got {:?}", shape
        )));
    }
    let out_dtype = rt.dtype(input_id)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Softmax,
        inputs: vec![input_id],
        out_shape: Shape::new(shape),
        out_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Transpose a 2D tensor: [rows, cols] → [cols, rows].
pub fn transpose(rt: &mut LazyRuntime, input_id: u64) -> Result<u64> {
    let shape = rt.shape(input_id)?;
    if shape.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "transpose requires 2D tensor, got {:?}", shape
        )));
    }
    let out_dtype = rt.dtype(input_id)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::Transpose,
        inputs: vec![input_id],
        out_shape: Shape::new(vec![shape[1], shape[0]]),
        out_dtype,
        container_id: ContainerId::DEFAULT,
    });
    Ok(out_id)
}

/// Multiply every element by a scalar.
pub fn scalar_mul(rt: &mut LazyRuntime, input_id: u64, scale: f32) -> Result<u64> {
    let shape = rt.shape(input_id)?;
    let out_dtype = rt.dtype(input_id)?;
    let out_id = next_id();
    rt.record_op(OpNode {
        id: out_id,
        op: OpKind::ScalarMul(scale),
        inputs: vec![input_id],
        out_shape: Shape::new(shape),
        out_dtype,
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
}
