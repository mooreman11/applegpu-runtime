use applegpu_core::tensor::DType;
use std::collections::HashMap;

pub type BackendResult<T> = std::result::Result<T, String>;

/// Backend trait abstracting over Metal (macOS) and future Socket (Linux) implementations.
///
/// All tensor operations go through this trait. Tensor IDs (u64) are opaque handles
/// managed by the backend. The trait is Send + Sync to allow a global static instance.
pub trait Backend: Send + Sync {
    // Lifecycle
    fn init(&self) -> BackendResult<HashMap<String, String>>;
    fn device_name(&self) -> BackendResult<String>;

    // Tensor creation
    fn tensor_from_data(&self, data: &[u8], shape: Vec<usize>, dtype: DType) -> BackendResult<u64>;
    fn insert_tensor_from_raw(&self, data: &[u8], shape: Vec<usize>, dtype: DType) -> BackendResult<u64>;
    fn destroy(&self, id: u64) -> BackendResult<()>;
    fn try_destroy(&self, id: u64);

    // Tensor queries
    fn shape(&self, id: u64) -> BackendResult<Vec<usize>>;
    fn dtype(&self, id: u64) -> BackendResult<DType>;
    fn read_bytes(&self, id: u64) -> BackendResult<Vec<u8>>;
    fn eval(&self, id: u64) -> BackendResult<()>;
    fn is_materialized(&self, id: u64) -> bool;
    fn is_pending(&self, id: u64) -> bool;

    // Binary ops
    fn add(&self, a: u64, b: u64) -> BackendResult<u64>;
    fn sub(&self, a: u64, b: u64) -> BackendResult<u64>;
    fn mul(&self, a: u64, b: u64) -> BackendResult<u64>;
    fn div(&self, a: u64, b: u64) -> BackendResult<u64>;
    fn matmul(&self, a: u64, b: u64) -> BackendResult<u64>;

    // Unary ops
    fn neg(&self, a: u64) -> BackendResult<u64>;
    fn relu(&self, a: u64) -> BackendResult<u64>;
    fn gelu(&self, a: u64) -> BackendResult<u64>;
    fn exp(&self, a: u64) -> BackendResult<u64>;
    fn log(&self, a: u64) -> BackendResult<u64>;
    fn sqrt(&self, a: u64) -> BackendResult<u64>;
    fn abs(&self, a: u64) -> BackendResult<u64>;
    fn sign(&self, a: u64) -> BackendResult<u64>;
    fn tanh(&self, a: u64) -> BackendResult<u64>;

    // Parameterized ops
    fn scalar_mul(&self, a: u64, scale: f32) -> BackendResult<u64>;
    fn pow(&self, a: u64, exponent: f32) -> BackendResult<u64>;
    fn clamp(&self, a: u64, min: f32, max: f32) -> BackendResult<u64>;

    // Reduction ops
    fn softmax(&self, a: u64) -> BackendResult<u64>;
    fn softmax_causal(&self, a: u64) -> BackendResult<u64>;
    fn argmax(&self, a: u64) -> BackendResult<u64>;
    fn sum(&self, a: u64) -> BackendResult<u64>;
    fn mean(&self, a: u64) -> BackendResult<u64>;

    // Shape ops
    fn reshape(&self, a: u64, shape: Vec<usize>) -> BackendResult<u64>;
    fn transpose(&self, a: u64) -> BackendResult<u64>;
    fn transpose_dims(&self, a: u64, dim0: usize, dim1: usize) -> BackendResult<u64>;
    fn slice(&self, a: u64, dim: usize, start: usize, end: usize) -> BackendResult<u64>;
    fn concat(&self, a: u64, b: u64, dim: usize) -> BackendResult<u64>;

    // Conditional ops
    fn where_cond(&self, cond: u64, x: u64, y: u64) -> BackendResult<u64>;
    fn masked_fill(&self, input: u64, mask: u64, value: f32) -> BackendResult<u64>;
    fn triu(&self, a: u64, diagonal: i32) -> BackendResult<u64>;
    fn tril(&self, a: u64, diagonal: i32) -> BackendResult<u64>;

    // Indexing
    fn gather(&self, input: u64, dim: usize, index: u64) -> BackendResult<u64>;
    fn index_select(&self, input: u64, dim: usize, index: u64) -> BackendResult<u64>;

    // Complex ops
    fn layer_norm(&self, input: u64, gamma: u64, beta: u64, eps: f32) -> BackendResult<u64>;
    fn embedding(&self, weights: u64, indices: u64) -> BackendResult<u64>;
    fn attention(&self, q: u64, k: u64, v: u64) -> BackendResult<u64>;
    fn attention_causal(&self, q: u64, k: u64, v: u64) -> BackendResult<u64>;
    fn add_bias(&self, input: u64, bias: u64) -> BackendResult<u64>;

    // CNN ops
    fn conv1d(&self, input: u64, weight: u64, stride: usize, padding: usize) -> BackendResult<u64>;
    fn conv2d(&self, input: u64, weight: u64, stride: (usize, usize), padding: (usize, usize)) -> BackendResult<u64>;
    fn batch_norm(&self, input: u64, mean: u64, var: u64, weight: u64, bias: u64, eps: f32) -> BackendResult<u64>;
    fn max_pool2d(&self, input: u64, kernel: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> BackendResult<u64>;
    fn avg_pool2d(&self, input: u64, kernel: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> BackendResult<u64>;

    // Backward ops
    fn softmax_backward(&self, grad: u64, output: u64) -> BackendResult<u64>;
    fn layer_norm_backward(&self, grad: u64, input: u64, gamma: u64, eps: f32) -> BackendResult<u64>;
    fn conv2d_backward_input(&self, grad: u64, weight: u64, in_h: usize, in_w: usize, stride: (usize, usize), padding: (usize, usize)) -> BackendResult<u64>;
    fn embedding_backward(&self, grad: u64, indices: u64, num_weights: usize) -> BackendResult<u64>;
    fn batch_norm_backward(&self, grad: u64, weight: u64, var: u64, eps: f32) -> BackendResult<u64>;

    // Resource management
    fn set_limits(&self, max_tensor_size_mb: usize, max_memory_mb: usize, max_tensors: usize);
    fn memory_usage(&self) -> usize;
    fn tensor_count(&self) -> usize;
    fn pool_stats(&self) -> HashMap<String, usize>;
    fn pool_drain(&self);
    fn set_pool_watermark(&self, mb: usize);

    // Scheduler
    fn register_container(&self, priority: &str, max_memory_mb: usize, max_tensors: usize, max_pending: usize) -> BackendResult<u64>;
    fn deregister_container(&self, container_id: u64) -> BackendResult<Vec<u64>>;
    fn pause_container(&self, container_id: u64) -> BackendResult<()>;
    fn resume_container(&self, container_id: u64) -> BackendResult<()>;
    fn submit_job(&self, container_id: u64, tensor_id: u64) -> BackendResult<u64>;
    fn run_next(&self) -> BackendResult<Option<u64>>;
    fn job_status(&self, job_id: u64) -> BackendResult<String>;
    fn container_usage(&self, container_id: u64) -> BackendResult<(usize, usize)>;
    fn global_usage(&self) -> (usize, usize);
    fn queue_depth(&self) -> usize;
}
