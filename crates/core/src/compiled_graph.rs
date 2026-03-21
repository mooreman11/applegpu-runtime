//! Compiled graph executor for torch.compile backend.
//!
//! Python serializes the FX graph into a compact binary format at compile time.
//! At runtime, a single FFI call executes the entire graph on the EagerRuntime,
//! encoding ops into the streaming Metal command buffer.
//!
//! Wire format per op (little-endian):
//!   op_code: u8, n_inputs: u8, inputs: [u16; n], out_ndim: u8,
//!   out_dims: [u64; ndim], out_dtype: u8, params_len: u8, params: [f32; plen]

use crate::device::Device;
use crate::eager::EagerRuntime;
use crate::error::{GpuError, Result};

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
