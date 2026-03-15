use std::io::{self, Read, Write, Cursor};

use crate::graph::{OpKind, OpNode};
use crate::scheduler::ContainerId;
use crate::tensor::{DType, Shape};

const MAGIC_REQUEST: &[u8; 4] = b"AGPU";
const MAGIC_RESPONSE: &[u8; 4] = b"AGPR";
const VERSION: u32 = 2;

const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_DIV: u32 = 3;
const OP_NEG: u32 = 4;
const OP_RELU: u32 = 5;
const OP_EXP: u32 = 6;
const OP_LOG: u32 = 7;
const OP_SQRT: u32 = 8;
const OP_MATMUL: u32 = 9;
const OP_FUSED: u32 = 10;

/// A serializable tensor (ID + shape + data).
#[derive(Debug, Clone)]
pub struct TensorData {
    pub id: u64,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

/// A request to evaluate a computation graph remotely.
#[derive(Debug)]
pub struct EvalRequest {
    pub target_id: u64,
    pub tensors: Vec<TensorData>,
    pub nodes: Vec<OpNode>,
}

/// Response from remote evaluation.
#[derive(Debug)]
pub enum EvalResponse {
    Ok {
        tensor_id: u64,
        shape: Vec<usize>,
        data: Vec<u8>,
    },
    Err(String),
}

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn read_u32(r: &mut impl Read) -> io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn read_u64(r: &mut impl Read) -> io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }

fn op_to_discriminant(op: &OpKind) -> u32 {
    match op {
        OpKind::Add => OP_ADD,
        OpKind::Sub => OP_SUB,
        OpKind::Mul => OP_MUL,
        OpKind::Div => OP_DIV,
        OpKind::Neg => OP_NEG,
        OpKind::Relu => OP_RELU,
        OpKind::Exp => OP_EXP,
        OpKind::Log => OP_LOG,
        OpKind::Sqrt => OP_SQRT,
        OpKind::Matmul => OP_MATMUL,
        OpKind::FusedElementwise { .. } => OP_FUSED,
        OpKind::Softmax => 11,
        OpKind::Transpose { .. } => 12,
        OpKind::ScalarMul(_) => 13,
        OpKind::Gelu => 14,
        OpKind::LayerNorm { .. } => 15,
        OpKind::Embedding => 16,
        OpKind::Reshape { .. } => 17,
        OpKind::Slice { .. } => 18,
        OpKind::Concat { .. } => 19,
        OpKind::AddBias => 20,
        OpKind::SoftmaxCausal => 21,
        OpKind::Argmax => 22,
        OpKind::Sum => 23,
        OpKind::Mean => 24,
        OpKind::Abs => 25,
        OpKind::Sign => 26,
        OpKind::Pow { .. } => 27,
        OpKind::Clamp { .. } => 28,
        OpKind::Where => 29,
        OpKind::MaskedFill { .. } => 30,
        OpKind::Triu { .. } => 31,
        OpKind::Tril { .. } => 32,
        OpKind::Gather { .. } => 33,
        OpKind::IndexSelect { .. } => 34,
        OpKind::Conv1d { .. } => 35,
        OpKind::Conv2d { .. } => 36,
        OpKind::BatchNorm { .. } => 37,
        OpKind::MaxPool2d { .. } => 38,
        OpKind::AvgPool2d { .. } => 39,
        OpKind::Tanh => 40,
        OpKind::SoftmaxBackward => 41,
        OpKind::LayerNormBackward { .. } => 42,
        OpKind::Conv2dBackwardInput { .. } => 43,
        OpKind::EmbeddingBackward => 44,
        OpKind::BatchNormBackward { .. } => 45,
    }
}

fn discriminant_to_op(d: u32, r: &mut impl Read) -> io::Result<OpKind> {
    match d {
        OP_ADD => Ok(OpKind::Add),
        OP_SUB => Ok(OpKind::Sub),
        OP_MUL => Ok(OpKind::Mul),
        OP_DIV => Ok(OpKind::Div),
        OP_NEG => Ok(OpKind::Neg),
        OP_RELU => Ok(OpKind::Relu),
        OP_EXP => Ok(OpKind::Exp),
        OP_LOG => Ok(OpKind::Log),
        OP_SQRT => Ok(OpKind::Sqrt),
        OP_MATMUL => Ok(OpKind::Matmul),
        OP_FUSED => {
            let src_len = read_u32(r)? as usize;
            let mut src = vec![0u8; src_len];
            r.read_exact(&mut src)?;
            let kernel_source = String::from_utf8(src)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let name_len = read_u32(r)? as usize;
            let mut name = vec![0u8; name_len];
            r.read_exact(&mut name)?;
            let function_name = String::from_utf8(name)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            Ok(OpKind::FusedElementwise { kernel_source, function_name })
        }
        11 => Ok(OpKind::Softmax),
        12 => {
            let dim0 = read_u32(r)? as usize;
            let dim1 = read_u32(r)? as usize;
            Ok(OpKind::Transpose { dim0, dim1 })
        }
        13 => {
            let mut scale_bytes = [0u8; 4];
            r.read_exact(&mut scale_bytes)?;
            Ok(OpKind::ScalarMul(f32::from_le_bytes(scale_bytes)))
        }
        14 => Ok(OpKind::Gelu),
        15 => {
            let mut eps_bytes = [0u8; 4];
            r.read_exact(&mut eps_bytes)?;
            Ok(OpKind::LayerNorm { eps: f32::from_le_bytes(eps_bytes) })
        }
        16 => Ok(OpKind::Embedding),
        17 => {
            let ndims = read_u32(r)? as usize;
            let mut new_shape = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                new_shape.push(read_u64(r)? as usize);
            }
            Ok(OpKind::Reshape { new_shape })
        }
        18 => {
            let dim = read_u32(r)? as usize;
            let start = read_u64(r)? as usize;
            let end = read_u64(r)? as usize;
            Ok(OpKind::Slice { dim, start, end })
        }
        19 => {
            let dim = read_u32(r)? as usize;
            Ok(OpKind::Concat { dim })
        }
        20 => Ok(OpKind::AddBias),
        21 => Ok(OpKind::SoftmaxCausal),
        22 => Ok(OpKind::Argmax),
        23 => Ok(OpKind::Sum),
        24 => Ok(OpKind::Mean),
        25 => Ok(OpKind::Abs),
        26 => Ok(OpKind::Sign),
        27 => {
            let mut exp_bytes = [0u8; 4];
            r.read_exact(&mut exp_bytes)?;
            Ok(OpKind::Pow { exponent: f32::from_le_bytes(exp_bytes) })
        }
        28 => {
            let mut min_bytes = [0u8; 4];
            r.read_exact(&mut min_bytes)?;
            let mut max_bytes = [0u8; 4];
            r.read_exact(&mut max_bytes)?;
            Ok(OpKind::Clamp { min_val: f32::from_le_bytes(min_bytes), max_val: f32::from_le_bytes(max_bytes) })
        }
        29 => Ok(OpKind::Where),
        30 => {
            let mut val_bytes = [0u8; 4];
            r.read_exact(&mut val_bytes)?;
            Ok(OpKind::MaskedFill { value: f32::from_le_bytes(val_bytes) })
        }
        31 => {
            let mut diag_bytes = [0u8; 4];
            r.read_exact(&mut diag_bytes)?;
            Ok(OpKind::Triu { diagonal: i32::from_le_bytes(diag_bytes) })
        }
        32 => {
            let mut diag_bytes = [0u8; 4];
            r.read_exact(&mut diag_bytes)?;
            Ok(OpKind::Tril { diagonal: i32::from_le_bytes(diag_bytes) })
        }
        33 => {
            let dim = read_u32(r)? as usize;
            Ok(OpKind::Gather { dim })
        }
        34 => {
            let dim = read_u32(r)? as usize;
            Ok(OpKind::IndexSelect { dim })
        }
        35 => {
            let stride = read_u64(r)? as usize;
            let padding = read_u64(r)? as usize;
            Ok(OpKind::Conv1d { stride, padding })
        }
        36 => {
            let s0 = read_u64(r)? as usize;
            let s1 = read_u64(r)? as usize;
            let p0 = read_u64(r)? as usize;
            let p1 = read_u64(r)? as usize;
            Ok(OpKind::Conv2d { stride: (s0, s1), padding: (p0, p1) })
        }
        37 => {
            let mut eps_bytes = [0u8; 4];
            r.read_exact(&mut eps_bytes)?;
            Ok(OpKind::BatchNorm { eps: f32::from_le_bytes(eps_bytes) })
        }
        38 => {
            let k0 = read_u64(r)? as usize;
            let k1 = read_u64(r)? as usize;
            let s0 = read_u64(r)? as usize;
            let s1 = read_u64(r)? as usize;
            let p0 = read_u64(r)? as usize;
            let p1 = read_u64(r)? as usize;
            Ok(OpKind::MaxPool2d { kernel_size: (k0, k1), stride: (s0, s1), padding: (p0, p1) })
        }
        39 => {
            let k0 = read_u64(r)? as usize;
            let k1 = read_u64(r)? as usize;
            let s0 = read_u64(r)? as usize;
            let s1 = read_u64(r)? as usize;
            let p0 = read_u64(r)? as usize;
            let p1 = read_u64(r)? as usize;
            Ok(OpKind::AvgPool2d { kernel_size: (k0, k1), stride: (s0, s1), padding: (p0, p1) })
        }
        40 => Ok(OpKind::Tanh),
        41 => Ok(OpKind::SoftmaxBackward),
        42 => {
            let mut eps_bytes = [0u8; 4];
            r.read_exact(&mut eps_bytes)?;
            Ok(OpKind::LayerNormBackward { eps: f32::from_le_bytes(eps_bytes) })
        }
        43 => {
            let s0 = read_u64(r)? as usize;
            let s1 = read_u64(r)? as usize;
            let p0 = read_u64(r)? as usize;
            let p1 = read_u64(r)? as usize;
            Ok(OpKind::Conv2dBackwardInput { stride: (s0, s1), padding: (p0, p1) })
        }
        44 => Ok(OpKind::EmbeddingBackward),
        45 => {
            let mut eps_bytes = [0u8; 4];
            r.read_exact(&mut eps_bytes)?;
            Ok(OpKind::BatchNormBackward { eps: f32::from_le_bytes(eps_bytes) })
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unknown op type: {}", d))),
    }
}

impl EvalRequest {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_REQUEST).unwrap();
        write_u32(&mut buf, VERSION).unwrap();
        write_u64(&mut buf, self.target_id).unwrap();

        write_u32(&mut buf, self.tensors.len() as u32).unwrap();
        for t in &self.tensors {
            write_u64(&mut buf, t.id).unwrap();
            write_u32(&mut buf, t.shape.len() as u32).unwrap();
            for &d in &t.shape {
                write_u64(&mut buf, d as u64).unwrap();
            }
            write_u32(&mut buf, 0).unwrap(); // dtype: 0 = f32
            write_u32(&mut buf, t.data.len() as u32).unwrap();
            buf.write_all(&t.data).unwrap();
        }

        write_u32(&mut buf, self.nodes.len() as u32).unwrap();
        for node in &self.nodes {
            write_u64(&mut buf, node.id).unwrap();
            let disc = op_to_discriminant(&node.op);
            write_u32(&mut buf, disc).unwrap();
            write_u32(&mut buf, node.inputs.len() as u32).unwrap();
            for &inp in &node.inputs {
                write_u64(&mut buf, inp).unwrap();
            }
            write_u32(&mut buf, node.out_shape.dims().len() as u32).unwrap();
            for &d in node.out_shape.dims() {
                write_u64(&mut buf, d as u64).unwrap();
            }
            write_u32(&mut buf, 0).unwrap(); // out_dtype: 0 = f32

            if let OpKind::FusedElementwise { ref kernel_source, ref function_name } = node.op {
                write_u32(&mut buf, kernel_source.len() as u32).unwrap();
                buf.write_all(kernel_source.as_bytes()).unwrap();
                write_u32(&mut buf, function_name.len() as u32).unwrap();
                buf.write_all(function_name.as_bytes()).unwrap();
            }
            if let OpKind::ScalarMul(scale) = node.op {
                buf.write_all(&scale.to_le_bytes()).unwrap();
            }
            if let OpKind::LayerNorm { eps } = node.op {
                buf.write_all(&eps.to_le_bytes()).unwrap();
            }
            if let OpKind::Reshape { ref new_shape } = node.op {
                write_u32(&mut buf, new_shape.len() as u32).unwrap();
                for &d in new_shape {
                    write_u64(&mut buf, d as u64).unwrap();
                }
            }
            if let OpKind::Slice { dim, start, end } = node.op {
                write_u32(&mut buf, dim as u32).unwrap();
                write_u64(&mut buf, start as u64).unwrap();
                write_u64(&mut buf, end as u64).unwrap();
            }
            if let OpKind::Transpose { dim0, dim1 } = node.op {
                write_u32(&mut buf, dim0 as u32).unwrap();
                write_u32(&mut buf, dim1 as u32).unwrap();
            }
            if let OpKind::Concat { dim } = node.op {
                write_u32(&mut buf, dim as u32).unwrap();
            }
            if let OpKind::Pow { exponent } = node.op {
                buf.write_all(&exponent.to_le_bytes()).unwrap();
            }
            if let OpKind::Clamp { min_val, max_val } = node.op {
                buf.write_all(&min_val.to_le_bytes()).unwrap();
                buf.write_all(&max_val.to_le_bytes()).unwrap();
            }
            if let OpKind::MaskedFill { value } = node.op {
                buf.write_all(&value.to_le_bytes()).unwrap();
            }
            if let OpKind::Triu { diagonal } = node.op {
                buf.write_all(&diagonal.to_le_bytes()).unwrap();
            }
            if let OpKind::Tril { diagonal } = node.op {
                buf.write_all(&diagonal.to_le_bytes()).unwrap();
            }
            if let OpKind::Gather { dim } = node.op {
                write_u32(&mut buf, dim as u32).unwrap();
            }
            if let OpKind::IndexSelect { dim } = node.op {
                write_u32(&mut buf, dim as u32).unwrap();
            }
            if let OpKind::Conv1d { stride, padding } = node.op {
                write_u64(&mut buf, stride as u64).unwrap();
                write_u64(&mut buf, padding as u64).unwrap();
            }
            if let OpKind::Conv2d { stride, padding } = node.op {
                write_u64(&mut buf, stride.0 as u64).unwrap();
                write_u64(&mut buf, stride.1 as u64).unwrap();
                write_u64(&mut buf, padding.0 as u64).unwrap();
                write_u64(&mut buf, padding.1 as u64).unwrap();
            }
            if let OpKind::BatchNorm { eps } = node.op {
                buf.write_all(&eps.to_le_bytes()).unwrap();
            }
            if let OpKind::MaxPool2d { kernel_size, stride, padding } = node.op {
                write_u64(&mut buf, kernel_size.0 as u64).unwrap();
                write_u64(&mut buf, kernel_size.1 as u64).unwrap();
                write_u64(&mut buf, stride.0 as u64).unwrap();
                write_u64(&mut buf, stride.1 as u64).unwrap();
                write_u64(&mut buf, padding.0 as u64).unwrap();
                write_u64(&mut buf, padding.1 as u64).unwrap();
            }
            if let OpKind::AvgPool2d { kernel_size, stride, padding } = node.op {
                write_u64(&mut buf, kernel_size.0 as u64).unwrap();
                write_u64(&mut buf, kernel_size.1 as u64).unwrap();
                write_u64(&mut buf, stride.0 as u64).unwrap();
                write_u64(&mut buf, stride.1 as u64).unwrap();
                write_u64(&mut buf, padding.0 as u64).unwrap();
                write_u64(&mut buf, padding.1 as u64).unwrap();
            }
            if let OpKind::LayerNormBackward { eps } = node.op {
                buf.write_all(&eps.to_le_bytes()).unwrap();
            }
            if let OpKind::Conv2dBackwardInput { stride, padding } = node.op {
                write_u64(&mut buf, stride.0 as u64).unwrap();
                write_u64(&mut buf, stride.1 as u64).unwrap();
                write_u64(&mut buf, padding.0 as u64).unwrap();
                write_u64(&mut buf, padding.1 as u64).unwrap();
            }
            if let OpKind::BatchNormBackward { eps } = node.op {
                buf.write_all(&eps.to_le_bytes()).unwrap();
            }
        }

        buf
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_REQUEST {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad magic"));
        }
        let version = read_u32(&mut r)?;
        if version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad version"));
        }

        let target_id = read_u64(&mut r)?;

        let num_tensors = read_u32(&mut r)? as usize;
        let mut tensors = Vec::with_capacity(num_tensors);
        for _ in 0..num_tensors {
            let id = read_u64(&mut r)?;
            let num_dims = read_u32(&mut r)? as usize;
            let mut shape = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                shape.push(read_u64(&mut r)? as usize);
            }
            let _dtype = read_u32(&mut r)?;
            let data_len = read_u32(&mut r)? as usize;
            let mut data = vec![0u8; data_len];
            r.read_exact(&mut data)?;
            tensors.push(TensorData { id, shape, dtype: DType::Float32, data });
        }

        let num_nodes = read_u32(&mut r)? as usize;
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let id = read_u64(&mut r)?;
            let op_disc = read_u32(&mut r)?;
            let num_inputs = read_u32(&mut r)? as usize;
            let mut inputs = Vec::with_capacity(num_inputs);
            for _ in 0..num_inputs {
                inputs.push(read_u64(&mut r)?);
            }
            let num_out_dims = read_u32(&mut r)? as usize;
            let mut out_shape = Vec::with_capacity(num_out_dims);
            for _ in 0..num_out_dims {
                out_shape.push(read_u64(&mut r)? as usize);
            }
            let _out_dtype = read_u32(&mut r)?;

            let op = discriminant_to_op(op_disc, &mut r)?;
            let out_shape = Shape::new(out_shape)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            nodes.push(OpNode {
                id,
                op,
                inputs,
                out_shape,
                out_dtype: DType::Float32,
                container_id: ContainerId::DEFAULT,
            });
        }

        Ok(EvalRequest { target_id, tensors, nodes })
    }
}

impl EvalResponse {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_RESPONSE).unwrap();
        match self {
            EvalResponse::Ok { tensor_id, shape, data } => {
                write_u32(&mut buf, 0).unwrap();
                write_u64(&mut buf, *tensor_id).unwrap();
                write_u32(&mut buf, shape.len() as u32).unwrap();
                for &d in shape {
                    write_u64(&mut buf, d as u64).unwrap();
                }
                write_u32(&mut buf, data.len() as u32).unwrap();
                buf.write_all(data).unwrap();
            }
            EvalResponse::Err(msg) => {
                write_u32(&mut buf, 1).unwrap();
                let bytes = msg.as_bytes();
                write_u32(&mut buf, bytes.len() as u32).unwrap();
                buf.write_all(bytes).unwrap();
            }
        }
        buf
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_RESPONSE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad response magic"));
        }

        let status = read_u32(&mut r)?;
        if status == 0 {
            let tensor_id = read_u64(&mut r)?;
            let num_dims = read_u32(&mut r)? as usize;
            let mut shape = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                shape.push(read_u64(&mut r)? as usize);
            }
            let data_len = read_u32(&mut r)? as usize;
            let mut data = vec![0u8; data_len];
            r.read_exact(&mut data)?;
            Ok(EvalResponse::Ok { tensor_id, shape, data })
        } else {
            let msg_len = read_u32(&mut r)? as usize;
            let mut msg = vec![0u8; msg_len];
            r.read_exact(&mut msg)?;
            let msg = String::from_utf8(msg)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(EvalResponse::Err(msg))
        }
    }
}

// ---------------------------------------------------------------------------
// Wire type conversions
// ---------------------------------------------------------------------------

use applegpu_wire::{WireOpKind, WireOpNode};

impl From<&OpKind> for WireOpKind {
    fn from(op: &OpKind) -> Self {
        match op {
            OpKind::Add => WireOpKind::Add,
            OpKind::Sub => WireOpKind::Sub,
            OpKind::Mul => WireOpKind::Mul,
            OpKind::Div => WireOpKind::Div,
            OpKind::Neg => WireOpKind::Neg,
            OpKind::Relu => WireOpKind::Relu,
            OpKind::Exp => WireOpKind::Exp,
            OpKind::Log => WireOpKind::Log,
            OpKind::Sqrt => WireOpKind::Sqrt,
            OpKind::Matmul => WireOpKind::Matmul,
            OpKind::FusedElementwise { kernel_source, function_name } => {
                WireOpKind::FusedElementwise {
                    kernel_source: kernel_source.clone(),
                    function_name: function_name.clone(),
                }
            }
            OpKind::Softmax => WireOpKind::Softmax,
            OpKind::Transpose { dim0, dim1 } => WireOpKind::Transpose { dim0: *dim0, dim1: *dim1 },
            OpKind::ScalarMul(scale) => WireOpKind::ScalarMul(*scale),
            OpKind::Gelu => WireOpKind::Gelu,
            OpKind::LayerNorm { eps } => WireOpKind::LayerNorm { eps: *eps },
            OpKind::Embedding => WireOpKind::Embedding,
            OpKind::Reshape { new_shape } => WireOpKind::Reshape { new_shape: new_shape.clone() },
            OpKind::Slice { dim, start, end } => WireOpKind::Slice { dim: *dim, start: *start, end: *end },
            OpKind::Concat { dim } => WireOpKind::Concat { dim: *dim },
            OpKind::AddBias => WireOpKind::AddBias,
            OpKind::SoftmaxCausal => WireOpKind::SoftmaxCausal,
            OpKind::Argmax => WireOpKind::Argmax,
            OpKind::Sum => WireOpKind::Sum,
            OpKind::Mean => WireOpKind::Mean,
            OpKind::Abs => WireOpKind::Abs,
            OpKind::Sign => WireOpKind::Sign,
            OpKind::Pow { exponent } => WireOpKind::Pow { exponent: *exponent },
            OpKind::Clamp { min_val, max_val } => WireOpKind::Clamp { min_val: *min_val, max_val: *max_val },
            OpKind::Where => WireOpKind::Where,
            OpKind::MaskedFill { value } => WireOpKind::MaskedFill { value: *value },
            OpKind::Triu { diagonal } => WireOpKind::Triu { diagonal: *diagonal },
            OpKind::Tril { diagonal } => WireOpKind::Tril { diagonal: *diagonal },
            OpKind::Gather { dim } => WireOpKind::Gather { dim: *dim },
            OpKind::IndexSelect { dim } => WireOpKind::IndexSelect { dim: *dim },
            OpKind::Conv1d { stride, padding } => WireOpKind::Conv1d { stride: *stride, padding: *padding },
            OpKind::Conv2d { stride, padding } => WireOpKind::Conv2d { stride: *stride, padding: *padding },
            OpKind::BatchNorm { eps } => WireOpKind::BatchNorm { eps: *eps },
            OpKind::MaxPool2d { kernel_size, stride, padding } => {
                WireOpKind::MaxPool2d { kernel_size: *kernel_size, stride: *stride, padding: *padding }
            }
            OpKind::AvgPool2d { kernel_size, stride, padding } => {
                WireOpKind::AvgPool2d { kernel_size: *kernel_size, stride: *stride, padding: *padding }
            }
            OpKind::Tanh => WireOpKind::Tanh,
        }
    }
}

impl From<&OpNode> for WireOpNode {
    fn from(node: &OpNode) -> Self {
        WireOpNode {
            id: node.id,
            op: WireOpKind::from(&node.op),
            inputs: node.inputs.clone(),
            out_shape: node.out_shape.dims().to_vec(),
            out_dtype: 0,
        }
    }
}

pub fn wire_op_to_core(wire: &WireOpKind) -> OpKind {
    match wire {
        WireOpKind::Add => OpKind::Add,
        WireOpKind::Sub => OpKind::Sub,
        WireOpKind::Mul => OpKind::Mul,
        WireOpKind::Div => OpKind::Div,
        WireOpKind::Neg => OpKind::Neg,
        WireOpKind::Relu => OpKind::Relu,
        WireOpKind::Exp => OpKind::Exp,
        WireOpKind::Log => OpKind::Log,
        WireOpKind::Sqrt => OpKind::Sqrt,
        WireOpKind::Matmul => OpKind::Matmul,
        WireOpKind::FusedElementwise { kernel_source, function_name } => {
            OpKind::FusedElementwise {
                kernel_source: kernel_source.clone(),
                function_name: function_name.clone(),
            }
        }
        WireOpKind::Softmax => OpKind::Softmax,
        WireOpKind::Transpose { dim0, dim1 } => OpKind::Transpose { dim0: *dim0, dim1: *dim1 },
        WireOpKind::ScalarMul(scale) => OpKind::ScalarMul(*scale),
        WireOpKind::Gelu => OpKind::Gelu,
        WireOpKind::LayerNorm { eps } => OpKind::LayerNorm { eps: *eps },
        WireOpKind::Embedding => OpKind::Embedding,
        WireOpKind::Reshape { new_shape } => OpKind::Reshape { new_shape: new_shape.clone() },
        WireOpKind::Slice { dim, start, end } => OpKind::Slice { dim: *dim, start: *start, end: *end },
        WireOpKind::Concat { dim } => OpKind::Concat { dim: *dim },
        WireOpKind::AddBias => OpKind::AddBias,
        WireOpKind::SoftmaxCausal => OpKind::SoftmaxCausal,
        WireOpKind::Argmax => OpKind::Argmax,
        WireOpKind::Sum => OpKind::Sum,
        WireOpKind::Mean => OpKind::Mean,
        WireOpKind::Abs => OpKind::Abs,
        WireOpKind::Sign => OpKind::Sign,
        WireOpKind::Pow { exponent } => OpKind::Pow { exponent: *exponent },
        WireOpKind::Clamp { min_val, max_val } => OpKind::Clamp { min_val: *min_val, max_val: *max_val },
        WireOpKind::Where => OpKind::Where,
        WireOpKind::MaskedFill { value } => OpKind::MaskedFill { value: *value },
        WireOpKind::Triu { diagonal } => OpKind::Triu { diagonal: *diagonal },
        WireOpKind::Tril { diagonal } => OpKind::Tril { diagonal: *diagonal },
        WireOpKind::Gather { dim } => OpKind::Gather { dim: *dim },
        WireOpKind::IndexSelect { dim } => OpKind::IndexSelect { dim: *dim },
        WireOpKind::Conv1d { stride, padding } => OpKind::Conv1d { stride: *stride, padding: *padding },
        WireOpKind::Conv2d { stride, padding } => OpKind::Conv2d { stride: *stride, padding: *padding },
        WireOpKind::BatchNorm { eps } => OpKind::BatchNorm { eps: *eps },
        WireOpKind::MaxPool2d { kernel_size, stride, padding } => {
            OpKind::MaxPool2d { kernel_size: *kernel_size, stride: *stride, padding: *padding }
        }
        WireOpKind::AvgPool2d { kernel_size, stride, padding } => {
            OpKind::AvgPool2d { kernel_size: *kernel_size, stride: *stride, padding: *padding }
        }
        WireOpKind::Tanh => OpKind::Tanh,
        // Backward ops exist in wire protocol but not yet in OpKind on this branch.
        // Map to closest forward op for now; will be 1:1 once OpKind adds backward variants.
        WireOpKind::SoftmaxBackward => OpKind::Softmax,
        WireOpKind::LayerNormBackward { eps } => OpKind::LayerNorm { eps: *eps },
        WireOpKind::Conv2dBackwardInput { stride, padding } => OpKind::Conv2d { stride: *stride, padding: *padding },
        WireOpKind::EmbeddingBackward => OpKind::Embedding,
        WireOpKind::BatchNormBackward { eps } => OpKind::BatchNorm { eps: *eps },
    }
}

pub fn wire_node_to_core(wire: &WireOpNode) -> std::result::Result<OpNode, std::io::Error> {
    let out_shape = Shape::new(wire.out_shape.clone())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(OpNode {
        id: wire.id,
        op: wire_op_to_core(&wire.op),
        inputs: wire.inputs.clone(),
        out_shape,
        out_dtype: DType::Float32,
        container_id: ContainerId::DEFAULT,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_roundtrip() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let req = EvalRequest {
            target_id: 42,
            tensors: vec![
                TensorData { id: 1, shape: vec![4], dtype: DType::Float32, data: bytes },
            ],
            nodes: vec![
                OpNode {
                    id: 42,
                    op: OpKind::Neg,
                    inputs: vec![1],
                    out_shape: Shape::new(vec![4]).unwrap(),
                    out_dtype: DType::Float32,
                    container_id: ContainerId::DEFAULT,
                },
            ],
        };

        let serialized = req.serialize();
        let deserialized = EvalRequest::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.target_id, 42);
        assert_eq!(deserialized.tensors.len(), 1);
        assert_eq!(deserialized.tensors[0].id, 1);
        assert_eq!(deserialized.tensors[0].shape, vec![4]);
        assert_eq!(deserialized.nodes.len(), 1);
        assert_eq!(deserialized.nodes[0].id, 42);
        assert!(deserialized.nodes[0].op.is_unary());
    }

    #[test]
    fn request_roundtrip_with_fused() {
        let req = EvalRequest {
            target_id: 10,
            tensors: vec![],
            nodes: vec![
                OpNode {
                    id: 10,
                    op: OpKind::FusedElementwise {
                        kernel_source: "kernel void test() {}".to_string(),
                        function_name: "test".to_string(),
                    },
                    inputs: vec![1, 2],
                    out_shape: Shape::new(vec![4]).unwrap(),
                    out_dtype: DType::Float32,
                    container_id: ContainerId::DEFAULT,
                },
            ],
        };

        let serialized = req.serialize();
        let deserialized = EvalRequest::deserialize(&serialized).unwrap();

        assert!(deserialized.nodes[0].op.is_fused());
        if let OpKind::FusedElementwise { ref kernel_source, ref function_name } = deserialized.nodes[0].op {
            assert_eq!(kernel_source, "kernel void test() {}");
            assert_eq!(function_name, "test");
        }
    }

    #[test]
    fn response_ok_roundtrip() {
        let resp = EvalResponse::Ok {
            tensor_id: 42,
            shape: vec![2, 3],
            data: vec![1, 2, 3, 4],
        };
        let serialized = resp.serialize();
        let deserialized = EvalResponse::deserialize(&serialized).unwrap();
        match deserialized {
            EvalResponse::Ok { tensor_id, shape, data } => {
                assert_eq!(tensor_id, 42);
                assert_eq!(shape, vec![2, 3]);
                assert_eq!(data, vec![1, 2, 3, 4]);
            }
            _ => panic!("Expected Ok"),
        }
    }

    #[test]
    fn response_err_roundtrip() {
        let resp = EvalResponse::Err("something failed".to_string());
        let serialized = resp.serialize();
        let deserialized = EvalResponse::deserialize(&serialized).unwrap();
        match deserialized {
            EvalResponse::Err(msg) => assert_eq!(msg, "something failed"),
            _ => panic!("Expected Err"),
        }
    }

    #[test]
    fn request_roundtrip_matmul() {
        let req = EvalRequest {
            target_id: 5,
            tensors: vec![],
            nodes: vec![
                OpNode {
                    id: 5,
                    op: OpKind::Matmul,
                    inputs: vec![1, 2],
                    out_shape: Shape::new(vec![2, 2]).unwrap(),
                    out_dtype: DType::Float32,
                    container_id: ContainerId::DEFAULT,
                },
            ],
        };
        let serialized = req.serialize();
        let deserialized = EvalRequest::deserialize(&serialized).unwrap();
        assert!(deserialized.nodes[0].op.is_matmul());
        assert_eq!(deserialized.nodes[0].out_shape.dims(), &[2, 2]);
    }
}
