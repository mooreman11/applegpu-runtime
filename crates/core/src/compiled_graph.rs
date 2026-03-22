//! Compiled graph executor for torch.compile backend.
//!
//! Python serializes the FX graph into a compact binary format at compile time.
//! At runtime, a single FFI call executes the entire graph on the EagerRuntime,
//! encoding ops into the streaming Metal command buffer.
//!
//! Wire format per op (little-endian):
//!   op_code: u8, n_inputs: u8, inputs: [u16; n], out_ndim: u8,
//!   out_dims: [u64; ndim], out_dtype: u8, params_len: u8, params: [f32; plen]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::compute;
use crate::device::Device;
use crate::eager::EagerRuntime;
use crate::error::{GpuError, Result};
use crate::tensor::{DType, next_tensor_id, Shape};

/// Thread-safe wrapper for MPSGraph handle pointers.
struct GraphHandle(*mut std::ffi::c_void);
unsafe impl Send for GraphHandle {}
unsafe impl Sync for GraphHandle {}

/// Cache for compiled MPSGraph handles. Keyed by (bytecode_hash, input_shapes_hash).
static MPSGRAPH_CACHE: Mutex<Option<HashMap<u64, GraphHandle>>> = Mutex::new(None);

fn mpsgraph_cache_key(ops_data: &[u8], shapes_flat: &[i64]) -> u64 {
    // FNV-1a hash
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in ops_data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    for &s in shapes_flat {
        for b in s.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

// Op codes — keep in sync with compile_backend.py
pub const OP_ADD: u8 = 0;
pub const OP_SUB: u8 = 1;
pub const OP_MUL: u8 = 2;
pub const OP_DIV: u8 = 3;
pub const OP_MATMUL: u8 = 4;
pub const OP_RELU: u8 = 5;
pub const OP_NEG: u8 = 6;
pub const OP_THRESHOLD_BACKWARD: u8 = 7;
pub const OP_SCALAR_MUL: u8 = 8;
pub const OP_MEAN_ALL: u8 = 9;
pub const OP_SUM_DIM: u8 = 10;
pub const OP_TRANSPOSE: u8 = 11;
pub const OP_VIEW: u8 = 12;
pub const OP_ADDMM: u8 = 13;
pub const OP_IDENTITY: u8 = 255;

/// Execute a compiled graph on the EagerRuntime.
pub fn execute(
    rt: &mut EagerRuntime,
    device: &Device,
    ops_data: &[u8],
    input_tids: &[u64],
    output_indices: &[u16],
    out_tids: &mut [u64],
    out_ptrs: &mut [*mut u8],
) -> Result<usize> {
    rt.begin_streaming(device);

    // Node outputs: index → (tensor_id, data_ptr, is_contiguous)
    let mut nodes: Vec<(u64, *mut u8, bool)> = input_tids.iter()
        .map(|&tid| {
            let ptr = rt.get(tid).map(|t| t.data_ptr()).unwrap_or(std::ptr::null_mut());
            let contig = rt.is_contiguous(tid).unwrap_or(true);
            (tid, ptr, contig)
        })
        .collect();

    let mut cursor = 0;
    while cursor < ops_data.len() {
        let op_code = ops_data[cursor]; cursor += 1;
        let n_inputs = ops_data[cursor] as usize; cursor += 1;

        let mut inp_indices = Vec::with_capacity(n_inputs);
        for _ in 0..n_inputs {
            let idx = u16::from_le_bytes([ops_data[cursor], ops_data[cursor + 1]]) as usize;
            cursor += 2;
            inp_indices.push(idx);
        }

        let out_ndim = ops_data[cursor] as usize; cursor += 1;
        let mut out_dims = Vec::with_capacity(out_ndim);
        for _ in 0..out_ndim {
            let dim = u64::from_le_bytes([
                ops_data[cursor], ops_data[cursor+1], ops_data[cursor+2], ops_data[cursor+3],
                ops_data[cursor+4], ops_data[cursor+5], ops_data[cursor+6], ops_data[cursor+7],
            ]) as usize;
            cursor += 8;
            out_dims.push(dim);
        }

        let _dtype_wire = ops_data[cursor]; cursor += 1;

        let params_len = ops_data[cursor] as usize; cursor += 1;
        let mut params = Vec::with_capacity(params_len);
        for _ in 0..params_len {
            let val = f32::from_le_bytes([
                ops_data[cursor], ops_data[cursor+1], ops_data[cursor+2], ops_data[cursor+3],
            ]);
            cursor += 4;
            params.push(val);
        }

        if op_code == OP_IDENTITY {
            let (tid, ptr, contig) = nodes[inp_indices[0]];
            nodes.push((tid, ptr, contig));
            continue;
        }

        let inp = |i: usize| -> (u64, *mut u8, bool) { nodes[inp_indices[i]] };

        let (out_tid, out_ptr, out_contig) = match op_code {
            OP_ADD => { let (a,_,_) = inp(0); let (b,_,_) = inp(1);
                let (id, ptr) = rt.binary_op(device, "elementwise_add", a, b)?; (id, ptr, true) }
            OP_SUB => { let (a,_,_) = inp(0); let (b,_,_) = inp(1);
                let (id, ptr) = rt.binary_op(device, "elementwise_sub", a, b)?; (id, ptr, true) }
            OP_MUL => { let (a,_,_) = inp(0); let (b,_,_) = inp(1);
                let (id, ptr) = rt.binary_op(device, "elementwise_mul", a, b)?; (id, ptr, true) }
            OP_DIV => { let (a,_,_) = inp(0); let (b,_,_) = inp(1);
                let (id, ptr) = rt.binary_op(device, "elementwise_div", a, b)?; (id, ptr, true) }
            OP_MATMUL => {
                let (mut a, _, ac) = inp(0); let (mut b, _, bc) = inp(1);
                if !ac { let (ca, _) = rt.scalar_mul(device, a, 1.0)?; a = ca; }
                if !bc { let (cb, _) = rt.scalar_mul(device, b, 1.0)?; b = cb; }
                let (id, ptr) = rt.matmul(device, a, b)?; (id, ptr, true)
            }
            OP_ADDMM => {
                let (bias, _, _) = inp(0);
                let (mut m1, _, m1c) = inp(1); let (mut m2, _, m2c) = inp(2);
                if !m1c { let (cm, _) = rt.scalar_mul(device, m1, 1.0)?; m1 = cm; }
                if !m2c { let (cm, _) = rt.scalar_mul(device, m2, 1.0)?; m2 = cm; }
                let (mm_id, _) = rt.matmul(device, m1, m2)?;
                let (id, ptr) = rt.binary_op(device, "elementwise_add", mm_id, bias)?; (id, ptr, true)
            }
            OP_RELU => { let (a,_,_) = inp(0);
                let (id, ptr) = rt.unary_op(device, "elementwise_relu", a)?; (id, ptr, true) }
            OP_NEG => { let (a,_,_) = inp(0);
                let (id, ptr) = rt.unary_op(device, "elementwise_neg", a)?; (id, ptr, true) }
            OP_THRESHOLD_BACKWARD => {
                let (grad,_,_) = inp(0); let (input,_,_) = inp(1);
                let threshold = params.first().copied().unwrap_or(0.0);
                let (id, ptr) = rt.threshold_backward(device, grad, input, threshold)?; (id, ptr, true)
            }
            OP_SCALAR_MUL => { let (a,_,_) = inp(0);
                let scale = params.first().copied().unwrap_or(1.0);
                let (id, ptr) = rt.scalar_mul(device, a, scale)?; (id, ptr, true) }
            OP_MEAN_ALL => { let (a,_,_) = inp(0);
                let (id, ptr) = rt.mean_all(device, a)?; (id, ptr, true) }
            OP_SUM_DIM => { let (a,_,_) = inp(0);
                let dim = params.first().copied().unwrap_or(0.0) as i64;
                let keepdim = params.get(1).copied().unwrap_or(0.0) != 0.0;
                let (id, ptr) = rt.sum_dim(device, a, dim, keepdim)?; (id, ptr, true) }
            OP_TRANSPOSE => { let (a,_,_) = inp(0);
                let shape = rt.shape(a)?; let ndim = shape.len();
                if ndim < 2 { return Err(GpuError::InvalidTensor("transpose requires 2D+".into())); }
                let mut new_shape = shape.clone(); new_shape.swap(ndim-2, ndim-1);
                let strides = { let t = rt.get(a)?; let s = t.layout.strides();
                    let mut ns = s.to_vec(); ns.swap(ndim-2, ndim-1); ns };
                let id = rt.create_view(a, &new_shape, &strides, 0)?;
                let ptr = rt.get(id)?.data_ptr(); (id, ptr, false)
            }
            OP_VIEW => { let (a,_,_) = inp(0);
                let mut strides = vec![1usize; out_dims.len()];
                for i in (0..out_dims.len().saturating_sub(1)).rev() { strides[i] = strides[i+1] * out_dims[i+1]; }
                let id = rt.create_view(a, &out_dims, &strides, 0)?;
                let ptr = rt.get(id)?.data_ptr(); (id, ptr, true)
            }
            _ => return Err(GpuError::InvalidTensor(format!("unsupported op: {}", op_code))),
        };

        nodes.push((out_tid, out_ptr, out_contig));
    }

    let n_out = output_indices.len().min(out_tids.len()).min(out_ptrs.len());
    for i in 0..n_out {
        let idx = output_indices[i] as usize;
        if idx >= nodes.len() {
            return Err(GpuError::InvalidTensor(format!("output {} out of range", idx)));
        }
        let (tid, ptr, _) = nodes[idx];
        out_tids[i] = tid;
        out_ptrs[i] = ptr;
    }

    Ok(n_out)
}

/// Try to execute the compiled graph via MPSGraph (fused execution).
/// Returns Ok(n_outputs) on success, or Err if MPSGraph build failed
/// (unsupported op — caller should fall back to per-op execute()).
pub fn execute_mpsgraph(
    rt: &mut EagerRuntime,
    device: &Device,
    ops_data: &[u8],
    input_tids: &[u64],
    output_indices: &[u16],
    out_tids: &mut [u64],
    out_ptrs: &mut [*mut u8],
) -> Result<usize> {
    let n_inputs = input_tids.len();
    let n_outputs = output_indices.len();

    // Collect input shapes, ndims, dtypes, buffer handles for FFI
    let mut shapes_flat: Vec<i64> = Vec::new();
    let mut ndims: Vec<u32> = Vec::new();
    let mut dtypes: Vec<u32> = Vec::new();
    let mut input_buf_handles: Vec<*const crate::ffi::GPUBufferHandle> = Vec::new();

    for &tid in input_tids {
        let tensor = rt.get(tid)?;
        let shape = tensor.shape_vec();
        ndims.push(shape.len() as u32);
        for &d in &shape {
            shapes_flat.push(d as i64);
        }
        dtypes.push(tensor.dtype.to_wire());
        input_buf_handles.push(tensor.buffer.raw_handle() as *const _);
    }

    // Build (or retrieve cached) MPSGraph
    // Only pass non-placeholder output indices to MPSGraph
    let graph_output_indices: Vec<u16> = output_indices.iter()
        .filter(|&&idx| (idx as usize) >= n_inputs)
        .copied()
        .collect();

    if graph_output_indices.is_empty() {
        for (i, &idx) in output_indices.iter().enumerate() {
            let input_tid = input_tids[idx as usize];
            let tensor = rt.get(input_tid)?;
            out_tids[i] = input_tid;
            out_ptrs[i] = tensor.data_ptr();
        }
        return Ok(n_outputs);
    }

    // Cache lookup: reuse compiled MPSGraph if we've seen this bytecode + shapes before
    let cache_key = mpsgraph_cache_key(ops_data, &shapes_flat);
    let graph_handle = {
        let mut cache = MPSGRAPH_CACHE.lock().unwrap();
        let map = cache.get_or_insert_with(HashMap::new);
        if let Some(cached) = map.get(&cache_key) {
            cached.0 // Cache hit — reuse compiled graph
        } else {
            // Cache miss — build new MPSGraph
            let handle = unsafe {
                crate::ffi::gpu_bridge_mpsgraph_build(
                    device.raw_handle() as *mut _,
                    ops_data.as_ptr(), ops_data.len() as u32,
                    n_inputs as u32,
                    shapes_flat.as_ptr(),
                    ndims.as_ptr(),
                    dtypes.as_ptr(),
                    graph_output_indices.len() as u32,
                    graph_output_indices.as_ptr(),
                )
            };
            if handle.is_null() {
                return Err(GpuError::ComputeFailed(
                    "MPSGraph build failed (unsupported op?) — falling back to per-op".into()));
            }
            if std::env::var("APPLEGPU_LOG_MPSGRAPH").is_ok() {
                eprintln!("[mpsgraph] built and cached: key={:#x}, {} inputs, {} graph outputs",
                          cache_key, n_inputs, graph_output_indices.len());
            }
            map.insert(cache_key, GraphHandle(handle));
            handle
        }
    };

    // Parse output shapes from bytecode to pre-allocate output buffers.
    // Outputs that reference placeholders are pass-throughs (no MPSGraph needed).
    let output_metas = parse_output_shapes(ops_data, input_tids.len(), output_indices)?;

    // Separate MPSGraph outputs from placeholder pass-throughs
    let mut mpsgraph_out_indices: Vec<u16> = Vec::new();
    let mut mpsgraph_buf_handles: Vec<*mut crate::ffi::GPUBufferHandle> = Vec::new();

    for (i, meta) in output_metas.iter().enumerate() {
        match meta {
            Some((shape, dtype)) => {
                // Computed output — allocate buffer for MPSGraph to write into
                let nbytes = shape.iter().product::<usize>() * dtype.size_bytes();
                let alloc_bytes = if nbytes == 0 { 4 } else { nbytes };
                let buffer = Arc::new(rt.pool_acquire(device, alloc_bytes)?);
                let ptr = buffer.contents();
                mpsgraph_buf_handles.push(buffer.raw_handle());
                mpsgraph_out_indices.push(output_indices[i]);

                let out_id = next_tensor_id();
                let out_shape = Shape::new(shape.clone())?;
                let out_layout = crate::tensor::TensorLayout::contiguous(out_shape);
                rt.insert_eager_tensor(out_id, crate::eager::EagerTensor {
                    buffer,
                    layout: out_layout,
                    dtype: *dtype,
                    offset: 0,
                });
                out_tids[i] = out_id;
                out_ptrs[i] = ptr;
            }
            None => {
                // Placeholder pass-through — just copy the input tensor reference
                let ph_idx = output_indices[i] as usize;
                let input_tid = input_tids[ph_idx];
                let tensor = rt.get(input_tid)?;
                out_tids[i] = input_tid;
                out_ptrs[i] = tensor.data_ptr();
            }
        }
    }

    // Flush any pending streaming work before MPSGraph runs
    rt.flush_and_wait();

    // If no computed outputs (all pass-throughs), skip MPSGraph execution
    if mpsgraph_buf_handles.is_empty() {
        return Ok(n_outputs);
    }

    // Execute MPSGraph with only the computed outputs
    let queue = compute::get_shared_queue(device);
    let rc = unsafe {
        crate::ffi::gpu_bridge_mpsgraph_run(
            graph_handle,
            queue,
            input_buf_handles.as_ptr(),
            n_inputs as u32,
            mpsgraph_buf_handles.as_ptr(),
            mpsgraph_buf_handles.len() as u32,
        )
    };

    if rc != 0 {
        return Err(GpuError::ComputeFailed(format!("MPSGraph run failed: rc={}", rc)));
    }

    Ok(n_outputs)
}

/// Parse output shapes from the bytecode for the given output indices.
fn parse_output_shapes(
    ops_data: &[u8],
    n_placeholders: usize,
    output_indices: &[u16],
) -> Result<Vec<Option<(Vec<usize>, DType)>>> {
    // Walk the bytecode, recording (shape, dtype) for each node
    let mut node_meta: Vec<(Vec<usize>, DType)> = Vec::new();

    // Placeholder nodes don't appear in bytecode — their metadata comes from inputs
    // We need to fill these from the caller. For now, skip them and index correctly.

    let mut cursor = 0;
    while cursor < ops_data.len() {
        let _op_code = ops_data[cursor]; cursor += 1;
        let n_inputs = ops_data[cursor] as usize; cursor += 1;
        cursor += n_inputs * 2; // skip input indices

        let out_ndim = ops_data[cursor] as usize; cursor += 1;
        let mut shape = Vec::with_capacity(out_ndim);
        for _ in 0..out_ndim {
            let dim = u64::from_le_bytes([
                ops_data[cursor], ops_data[cursor+1], ops_data[cursor+2], ops_data[cursor+3],
                ops_data[cursor+4], ops_data[cursor+5], ops_data[cursor+6], ops_data[cursor+7],
            ]) as usize;
            cursor += 8;
            shape.push(dim);
        }

        let dtype_wire = ops_data[cursor]; cursor += 1;
        let dtype = DType::from_wire(dtype_wire as u32).unwrap_or(DType::Float32);

        let params_len = ops_data[cursor] as usize; cursor += 1;
        cursor += params_len * 4; // skip params

        node_meta.push((shape, dtype));
    }

    // Map output indices to shapes.
    // Outputs that reference placeholders (idx < n_placeholders) get None —
    // the caller should pass these through from input buffers directly.
    let mut results = Vec::with_capacity(output_indices.len());
    for &idx in output_indices {
        let i = idx as usize;
        if i < n_placeholders {
            // Placeholder pass-through — no shape/dtype info needed from bytecode
            results.push(None);
        } else {
            let node_idx = i - n_placeholders;
            if node_idx >= node_meta.len() {
                return Err(GpuError::InvalidTensor(format!(
                    "MPSGraph output index {} out of range", idx)));
            }
            results.push(Some(node_meta[node_idx].clone()));
        }
    }

    Ok(results)
}
