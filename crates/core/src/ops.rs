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
