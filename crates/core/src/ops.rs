use once_cell::sync::Lazy;

use crate::compute::KernelRegistry;
use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::tensor::Tensor;

static REGISTRY: Lazy<KernelRegistry> = Lazy::new(KernelRegistry::new);

/// Element-wise binary operation on two tensors. Shapes must match.
fn binary_op(device: &Device, a: &Tensor, b: &Tensor, kernel_name: &str) -> Result<Tensor> {
    if a.meta.shape != b.meta.shape {
        return Err(GpuError::InvalidTensor(format!(
            "Shape mismatch: {:?} vs {:?}",
            a.meta.shape.dims(),
            b.meta.shape.dims()
        )));
    }
    let out = Tensor::empty_f32(device, a.meta.shape.dims().to_vec())?;
    REGISTRY.dispatch_binary(device, kernel_name, &a.buffer, &b.buffer, &out.buffer, a.numel())?;
    Ok(out)
}

/// Element-wise unary operation on a tensor.
fn unary_op(device: &Device, input: &Tensor, kernel_name: &str) -> Result<Tensor> {
    let out = Tensor::empty_f32(device, input.meta.shape.dims().to_vec())?;
    REGISTRY.dispatch_unary(device, kernel_name, &input.buffer, &out.buffer, input.numel())?;
    Ok(out)
}

pub fn add(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(device, a, b, "elementwise_add")
}

pub fn sub(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(device, a, b, "elementwise_sub")
}

pub fn mul(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(device, a, b, "elementwise_mul")
}

pub fn div(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(device, a, b, "elementwise_div")
}

pub fn neg(device: &Device, input: &Tensor) -> Result<Tensor> {
    unary_op(device, input, "elementwise_neg")
}

pub fn relu(device: &Device, input: &Tensor) -> Result<Tensor> {
    unary_op(device, input, "elementwise_relu")
}

pub fn exp(device: &Device, input: &Tensor) -> Result<Tensor> {
    unary_op(device, input, "elementwise_exp")
}

pub fn log(device: &Device, input: &Tensor) -> Result<Tensor> {
    unary_op(device, input, "elementwise_log")
}

pub fn sqrt(device: &Device, input: &Tensor) -> Result<Tensor> {
    unary_op(device, input, "elementwise_sqrt")
}

/// Matrix multiplication: C[M,N] = A[M,K] * B[K,N].
/// A must be 2D with shape [M,K], B must be 2D with shape [K,N].
pub fn matmul(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.meta.shape.dims();
    let b_dims = b.meta.shape.dims();

    if a_dims.len() != 2 || b_dims.len() != 2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul requires 2D tensors, got {:?} and {:?}",
            a_dims, b_dims
        )));
    }

    let (m, k1) = (a_dims[0], a_dims[1]);
    let (k2, n) = (b_dims[0], b_dims[1]);

    if k1 != k2 {
        return Err(GpuError::InvalidTensor(format!(
            "matmul inner dimensions mismatch: A[{},{}] * B[{},{}]",
            m, k1, k2, n
        )));
    }

    let out = Tensor::empty_f32(device, vec![m, n])?;
    REGISTRY.dispatch_matmul(device, &a.buffer, &b.buffer, &out.buffer, m, n, k1)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> Option<Device> {
        Device::new().ok()
    }

    #[test]
    fn ops_add_sub_roundtrip() {
        let device = match get_device() { Some(d) => d, None => return };
        let a = Tensor::from_f32(&device, vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let c = add(&device, &a, &b).unwrap();
        assert_eq!(c.as_f32_slice(), &[11.0, 22.0, 33.0, 44.0]);
        let d = sub(&device, &c, &b).unwrap();
        assert_eq!(d.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ops_mul_div() {
        let device = match get_device() { Some(d) => d, None => return };
        let a = Tensor::from_f32(&device, vec![4], &[2.0, 4.0, 6.0, 8.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![4], &[2.0, 2.0, 2.0, 2.0]).unwrap();
        let c = mul(&device, &a, &b).unwrap();
        assert_eq!(c.as_f32_slice(), &[4.0, 8.0, 12.0, 16.0]);
        let d = div(&device, &c, &b).unwrap();
        assert_eq!(d.as_f32_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn ops_neg_relu() {
        let device = match get_device() { Some(d) => d, None => return };
        let a = Tensor::from_f32(&device, vec![4], &[1.0, -2.0, 3.0, -4.0]).unwrap();
        let b = neg(&device, &a).unwrap();
        assert_eq!(b.as_f32_slice(), &[-1.0, 2.0, -3.0, 4.0]);
        let c = relu(&device, &a).unwrap();
        assert_eq!(c.as_f32_slice(), &[1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn ops_matmul_2x2() {
        let device = match get_device() { Some(d) => d, None => return };
        let a = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_f32(&device, vec![2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = matmul(&device, &a, &b).unwrap();
        assert_eq!(c.meta.shape.dims(), &[2, 2]);
        assert_eq!(c.as_f32_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn ops_matmul_shape_validation() {
        let device = match get_device() { Some(d) => d, None => return };
        let a = Tensor::from_f32(&device, vec![2, 3], &[1.0; 6]).unwrap();
        let b = Tensor::from_f32(&device, vec![2, 2], &[1.0; 4]).unwrap();
        let result = matmul(&device, &a, &b);
        assert!(result.is_err()); // inner dimensions 3 != 2
    }
}
