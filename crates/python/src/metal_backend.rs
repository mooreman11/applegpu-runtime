#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use applegpu_core::lazy::LazyRuntime;
use applegpu_core::tensor::{DType, Tensor};
use applegpu_core::scheduler::{ContainerId, ContainerConfig, Priority, JobId, JobStatus};

use crate::backend::{Backend, BackendResult};

/// C-compatible deallocator matching GPUDeallocator typedef.
/// Called by Metal (via Swift) when the MTLBuffer is released.
/// `context` is a PyObject* that was Py_IncRef'd during from_numpy_shared.
/// Must acquire the GIL before calling Py_DecRef.
unsafe extern "C" fn buffer_deallocator(
    _ptr: *mut c_void,
    _len: u64,
    context: *mut c_void,
) {
    if !context.is_null() {
        pyo3::Python::with_gil(|_py| {
            pyo3::ffi::Py_DecRef(context as *mut pyo3::ffi::PyObject);
        });
    }
}

/// Metal backend for macOS — wraps LazyRuntime with a Mutex for thread safety.
pub struct MetalBackend {
    runtime: Mutex<LazyRuntime>,
}

impl MetalBackend {
    pub fn new() -> Self {
        MetalBackend {
            runtime: Mutex::new(LazyRuntime::new()),
        }
    }
}

/// Helper: get the backend device runtime.
fn get_device_runtime() -> BackendResult<&'static applegpu_core::backend::Runtime> {
    applegpu_core::backend::get_runtime()
        .map_err(|e| e.to_string())
}

/// Helper: auto-evaluate a tensor if it is pending (lazy).
fn auto_eval(rt: &mut LazyRuntime, id: u64) -> BackendResult<()> {
    if rt.is_pending(id) {
        let runtime = get_device_runtime()?;
        if let Some(ref socket_path) = runtime.socket_path {
            rt.eval_remote(&runtime.device, id, socket_path)
                .map_err(|e| e.to_string())?;
        } else {
            rt.eval(&runtime.device, id)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

macro_rules! map_err {
    ($expr:expr) => {
        $expr.map_err(|e| e.to_string())
    };
}

impl Backend for MetalBackend {
    fn init(&self) -> BackendResult<HashMap<String, String>> {
        let runtime = map_err!(applegpu_core::backend::init_backend())?;
        let mut info = HashMap::new();
        info.insert("backend".to_string(), format!("{:?}", runtime.backend).to_lowercase());
        info.insert("device".to_string(), runtime.device.name());
        Ok(info)
    }

    fn device_name(&self) -> BackendResult<String> {
        let runtime = get_device_runtime()?;
        Ok(runtime.device.name())
    }

    fn tensor_from_data(&self, data: &[u8], shape: Vec<usize>, dtype: DType) -> BackendResult<u64> {
        let runtime = get_device_runtime()?;
        let t = map_err!(Tensor::from_data(&runtime.device, shape, dtype, data))?;
        let id = t.meta.id;
        map_err!(self.runtime.lock().unwrap().insert_tensor(t))?;
        Ok(id)
    }

    fn tensor_from_ptr_no_copy(
        &self, ptr: *mut u8, len: usize, shape: Vec<usize>,
        dtype: DType, release_context: *mut c_void,
    ) -> BackendResult<u64> {
        let runtime = get_device_runtime()?;
        let buffer = applegpu_core::buffer::Buffer::from_ptr_no_copy(
            &runtime.device, ptr, len, release_context,
            Some(buffer_deallocator),
        ).map_err(|e| e.to_string())?;
        let id = applegpu_core::tensor::next_tensor_id();
        let tensor = Tensor::from_raw(id, shape, dtype, buffer);
        let mut rt = self.runtime.lock().unwrap();
        // size=0 for memory accounting (memory belongs to Python)
        map_err!(rt.insert_tensor_with_size(tensor, 0))?;
        Ok(id)
    }

    fn insert_tensor_from_raw(&self, data: &[u8], shape: Vec<usize>, dtype: DType) -> BackendResult<u64> {
        self.tensor_from_data(data, shape, dtype)
    }

    fn destroy(&self, id: u64) -> BackendResult<()> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(rt.destroy(id))
    }

    fn try_destroy(&self, id: u64) {
        if let Ok(mut rt) = self.runtime.try_lock() {
            let _ = rt.destroy(id);
        }
    }

    fn shape(&self, id: u64) -> BackendResult<Vec<usize>> {
        let rt = self.runtime.lock().unwrap();
        map_err!(rt.shape(id))
    }

    fn dtype(&self, id: u64) -> BackendResult<DType> {
        let rt = self.runtime.lock().unwrap();
        map_err!(rt.dtype(id))
    }

    fn read_bytes(&self, id: u64) -> BackendResult<Vec<u8>> {
        let mut rt = self.runtime.lock().unwrap();
        auto_eval(&mut rt, id)?;
        map_err!(rt.read_bytes(id))
    }

    fn eval(&self, id: u64) -> BackendResult<()> {
        let mut rt = self.runtime.lock().unwrap();
        auto_eval(&mut rt, id)
    }

    fn is_materialized(&self, id: u64) -> bool {
        let rt = self.runtime.lock().unwrap();
        rt.is_materialized(id)
    }

    fn is_pending(&self, id: u64) -> bool {
        let rt = self.runtime.lock().unwrap();
        rt.is_pending(id)
    }

    // Binary ops
    fn add(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::add(&mut rt, a, b))
    }

    fn sub(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::sub(&mut rt, a, b))
    }

    fn mul(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::mul(&mut rt, a, b))
    }

    fn div(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::div(&mut rt, a, b))
    }

    fn matmul(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::matmul(&mut rt, a, b))
    }

    // Unary ops
    fn neg(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::neg(&mut rt, a))
    }

    fn relu(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::relu(&mut rt, a))
    }

    fn gelu(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::gelu(&mut rt, a))
    }

    fn sigmoid(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::sigmoid(&mut rt, a))
    }

    fn var(&self, a: u64, correction: u32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::var(&mut rt, a, correction))
    }

    fn std_dev(&self, a: u64, correction: u32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::std_dev(&mut rt, a, correction))
    }

    fn exp(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::exp(&mut rt, a))
    }

    fn log(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::log(&mut rt, a))
    }

    fn sqrt(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::sqrt(&mut rt, a))
    }

    fn abs(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::abs(&mut rt, a))
    }

    fn sign(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::sign(&mut rt, a))
    }

    fn tanh(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::tanh(&mut rt, a))
    }

    fn sin(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::sin(&mut rt, a))
    }

    fn cos(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::cos(&mut rt, a))
    }

    // Parameterized ops
    fn scalar_mul(&self, a: u64, scale: f32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::scalar_mul(&mut rt, a, scale))
    }

    fn pow(&self, a: u64, exponent: f32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::pow(&mut rt, a, exponent))
    }

    fn clamp(&self, a: u64, min: f32, max: f32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::clamp(&mut rt, a, min, max))
    }

    // Reduction ops
    fn softmax(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::softmax(&mut rt, a))
    }

    fn log_softmax(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::log_softmax(&mut rt, a))
    }

    fn softmax_causal(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::softmax_causal(&mut rt, a))
    }

    fn argmax(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::argmax(&mut rt, a))
    }

    fn sum(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::sum(&mut rt, a))
    }

    fn mean(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::mean(&mut rt, a))
    }

    // Comparison ops
    fn lt(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::lt(&mut rt, a, b))
    }
    fn gt(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::gt(&mut rt, a, b))
    }
    fn le(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::le(&mut rt, a, b))
    }
    fn ge(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::ge(&mut rt, a, b))
    }
    fn eq_op(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::eq_op(&mut rt, a, b))
    }
    fn ne_op(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::ne_op(&mut rt, a, b))
    }

    // Bitwise ops
    fn bitwise_and(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::bitwise_and(&mut rt, a, b))
    }
    fn bitwise_or(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::bitwise_or(&mut rt, a, b))
    }
    fn bitwise_xor(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::bitwise_xor(&mut rt, a, b))
    }
    fn bitwise_not(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::bitwise_not(&mut rt, a))
    }
    fn shl(&self, a: u64, shift: u32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::shl(&mut rt, a, shift))
    }
    fn shr(&self, a: u64, shift: u32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::shr(&mut rt, a, shift))
    }

    // Modulo
    fn mod_op(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::mod_op(&mut rt, a, b))
    }

    // Element-wise min/max
    fn elem_min(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::elem_min(&mut rt, a, b))
    }
    fn elem_max(&self, a: u64, b: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::elem_max(&mut rt, a, b))
    }

    // Logical NOT
    fn logical_not(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::logical_not(&mut rt, a))
    }

    fn cast(&self, a: u64, target_dtype: &str) -> BackendResult<u64> {
        let dt = applegpu_core::tensor::DType::from_name(target_dtype)
            .ok_or_else(|| format!("Unknown dtype: {}", target_dtype))?;
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::cast(&mut rt, a, dt))
    }

    fn quantize(&self, a: u64, target_dtype: &str, scale: f32, zero_point: i32) -> BackendResult<u64> {
        let dt = applegpu_core::tensor::DType::from_name(target_dtype)
            .ok_or_else(|| format!("Unknown dtype: {}", target_dtype))?;
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::quantize(&mut rt, a, dt, scale, zero_point))
    }

    fn dequantize(&self, a: u64, target_dtype: &str, scale: f32, zero_point: i32) -> BackendResult<u64> {
        let dt = applegpu_core::tensor::DType::from_name(target_dtype)
            .ok_or_else(|| format!("Unknown dtype: {}", target_dtype))?;
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::dequantize(&mut rt, a, dt, scale, zero_point))
    }

    // Shape ops
    fn reshape(&self, a: u64, shape: Vec<usize>) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::reshape(&mut rt, a, shape))
    }

    fn transpose(&self, a: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::transpose(&mut rt, a))
    }

    fn transpose_dims(&self, a: u64, dim0: usize, dim1: usize) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::transpose_dims(&mut rt, a, dim0, dim1))
    }

    fn slice(&self, a: u64, dim: usize, start: usize, end: usize) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::slice(&mut rt, a, dim, start, end))
    }

    fn concat(&self, a: u64, b: u64, dim: usize) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::concat(&mut rt, a, b, dim))
    }

    // Conditional ops
    fn where_cond(&self, cond: u64, x: u64, y: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::where_cond(&mut rt, cond, x, y))
    }

    fn masked_fill(&self, input: u64, mask: u64, value: f32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::masked_fill(&mut rt, input, mask, value))
    }

    fn triu(&self, a: u64, diagonal: i32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::triu(&mut rt, a, diagonal))
    }

    fn tril(&self, a: u64, diagonal: i32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::tril(&mut rt, a, diagonal))
    }

    // Indexing
    fn gather(&self, input: u64, dim: usize, index: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::gather(&mut rt, input, dim, index))
    }

    fn index_select(&self, input: u64, dim: usize, index: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::index_select(&mut rt, input, dim, index))
    }

    // Complex ops
    fn layer_norm(&self, input: u64, gamma: u64, beta: u64, eps: f32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::layer_norm(&mut rt, input, gamma, beta, eps))
    }

    fn embedding(&self, weights: u64, indices: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::embedding(&mut rt, weights, indices))
    }

    fn attention(&self, q: u64, k: u64, v: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::attention(&mut rt, q, k, v))
    }

    fn attention_causal(&self, q: u64, k: u64, v: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::attention_causal(&mut rt, q, k, v))
    }

    fn add_bias(&self, input: u64, bias: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::add_bias(&mut rt, input, bias))
    }

    // CNN ops
    fn conv1d(&self, input: u64, weight: u64, stride: usize, padding: usize) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::conv1d(&mut rt, input, weight, stride, padding))
    }

    fn conv2d(&self, input: u64, weight: u64, stride: (usize, usize), padding: (usize, usize)) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::conv2d(&mut rt, input, weight, stride, padding))
    }

    fn batch_norm(&self, input: u64, mean: u64, var: u64, weight: u64, bias: u64, eps: f32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::batch_norm(&mut rt, input, mean, var, weight, bias, eps))
    }

    fn max_pool2d(&self, input: u64, kernel: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::max_pool2d(&mut rt, input, kernel, stride, padding))
    }

    fn avg_pool2d(&self, input: u64, kernel: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::avg_pool2d(&mut rt, input, kernel, stride, padding))
    }

    // Backward ops
    fn softmax_backward(&self, grad: u64, output: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::softmax_backward(&mut rt, grad, output))
    }

    fn layer_norm_backward(&self, grad: u64, input: u64, gamma: u64, eps: f32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::layer_norm_backward(&mut rt, grad, input, gamma, eps))
    }

    fn conv2d_backward_input(&self, grad: u64, weight: u64, in_h: usize, in_w: usize, stride: (usize, usize), padding: (usize, usize)) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::conv2d_backward_input(&mut rt, grad, weight, in_h, in_w, stride, padding))
    }

    fn embedding_backward(&self, grad: u64, indices: u64, num_weights: usize) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::embedding_backward(&mut rt, grad, indices, num_weights))
    }

    fn batch_norm_backward(&self, grad: u64, weight: u64, var: u64, eps: f32) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(applegpu_core::ops::batch_norm_backward(&mut rt, grad, weight, var, eps))
    }

    // Resource management
    fn set_limits(&self, max_tensor_size_mb: usize, max_memory_mb: usize, max_tensors: usize) {
        let mut rt = self.runtime.lock().unwrap();
        rt.set_limits(applegpu_core::limits::ResourceLimits {
            max_tensor_size_bytes: if max_tensor_size_mb > 0 { max_tensor_size_mb * 1024 * 1024 } else { 0 },
            max_total_memory_bytes: if max_memory_mb > 0 { max_memory_mb * 1024 * 1024 } else { 0 },
            max_tensor_count: max_tensors,
        });
    }

    fn memory_usage(&self) -> usize {
        let rt = self.runtime.lock().unwrap();
        rt.memory_usage()
    }

    fn tensor_count(&self) -> usize {
        let rt = self.runtime.lock().unwrap();
        rt.live_tensor_count()
    }

    fn pool_stats(&self) -> HashMap<String, usize> {
        let rt = self.runtime.lock().unwrap();
        let stats = rt.pool.stats();
        let mut map = HashMap::new();
        map.insert("hits".to_string(), stats.hits as usize);
        map.insert("misses".to_string(), stats.misses as usize);
        map.insert("pooled_bytes".to_string(), stats.pooled_bytes);
        map.insert("bucket_count".to_string(), stats.bucket_count);
        map
    }

    fn pool_drain(&self) {
        let mut rt = self.runtime.lock().unwrap();
        rt.pool.drain();
    }

    fn set_pool_watermark(&self, mb: usize) {
        let mut rt = self.runtime.lock().unwrap();
        rt.pool.set_max_pooled_bytes(mb * 1024 * 1024);
    }

    // Scheduler
    fn register_container(&self, priority: &str, max_memory_mb: usize, max_tensors: usize, max_pending: usize) -> BackendResult<u64> {
        let priority = match priority {
            "high" => Priority::High,
            "normal" => Priority::Normal,
            "low" => Priority::Low,
            _ => return Err(format!("Invalid priority: {}. Use 'high', 'normal', or 'low'", priority)),
        };
        let config = ContainerConfig {
            priority,
            max_memory_bytes: max_memory_mb * 1024 * 1024,
            max_tensor_count: max_tensors,
            max_tensor_size_bytes: 0,
            max_pending_jobs: max_pending,
        };
        let mut rt = self.runtime.lock().unwrap();
        let id = map_err!(rt.scheduler.register_container(config))?;
        Ok(id.0)
    }

    fn deregister_container(&self, container_id: u64) -> BackendResult<Vec<u64>> {
        let mut rt = self.runtime.lock().unwrap();
        let tensors = map_err!(rt.scheduler.deregister_container(ContainerId(container_id)))?;
        for &tid in &tensors {
            rt.remove_tensor_raw(tid);
        }
        Ok(tensors)
    }

    fn pause_container(&self, container_id: u64) -> BackendResult<()> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(rt.scheduler.pause_container(ContainerId(container_id)))
    }

    fn resume_container(&self, container_id: u64) -> BackendResult<()> {
        let mut rt = self.runtime.lock().unwrap();
        map_err!(rt.scheduler.resume_container(ContainerId(container_id)))
    }

    fn submit_job(&self, container_id: u64, tensor_id: u64) -> BackendResult<u64> {
        let mut rt = self.runtime.lock().unwrap();
        let job_id = map_err!(rt.scheduler.submit(ContainerId(container_id), tensor_id))?;
        Ok(job_id.0)
    }

    fn run_next(&self) -> BackendResult<Option<u64>> {
        let runtime = get_device_runtime()?;
        let mut rt = self.runtime.lock().unwrap();
        let result = map_err!(rt.run_next(&runtime.device))?;
        Ok(result.map(|j| j.0))
    }

    fn job_status(&self, job_id: u64) -> BackendResult<String> {
        let rt = self.runtime.lock().unwrap();
        match rt.scheduler.job_status(JobId(job_id)) {
            Some((_, JobStatus::Queued)) => Ok("queued".to_string()),
            Some((_, JobStatus::Running { .. })) => Ok("running".to_string()),
            Some((_, JobStatus::Completed { .. })) => Ok("completed".to_string()),
            Some((_, JobStatus::Failed { .. })) => Ok("failed".to_string()),
            None => Err(format!("Job {} not found", job_id)),
        }
    }

    fn container_usage(&self, container_id: u64) -> BackendResult<(usize, usize)> {
        let rt = self.runtime.lock().unwrap();
        rt.scheduler.container_usage(ContainerId(container_id))
            .ok_or_else(|| format!("Container {} not found", container_id))
    }

    fn global_usage(&self) -> (usize, usize) {
        let rt = self.runtime.lock().unwrap();
        rt.scheduler.global_usage()
    }

    fn queue_depth(&self) -> usize {
        let rt = self.runtime.lock().unwrap();
        rt.scheduler.queue_depth()
    }
}
