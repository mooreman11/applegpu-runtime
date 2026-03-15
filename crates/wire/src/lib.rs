//! Wire protocol types for the GPU service.
//!
//! Zero external dependencies — only `std`. Compiles on both Linux and macOS.

use std::io::{self, Cursor, Read, Write};

// ---------------------------------------------------------------------------
// Primitive helpers (all little-endian)
// ---------------------------------------------------------------------------

pub fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn write_f32(w: &mut impl Write, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn write_i32(w: &mut impl Write, v: i32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

pub fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

pub fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

// ---------------------------------------------------------------------------
// Length-prefixed message framing
// ---------------------------------------------------------------------------

/// Maximum allowed message size: 256 MiB.
pub const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

/// Write a length-prefixed message: `[u32 len][payload]`, then flush.
pub fn write_message(w: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    write_u32(w, payload.len() as u32)?;
    w.write_all(payload)?;
    w.flush()
}

/// Read a length-prefixed message. Rejects if `len > max_size`.
pub fn read_message(r: &mut impl Read, max_size: usize) -> io::Result<Vec<u8>> {
    let len = read_u32(r)? as usize;
    if len > max_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Message size {} exceeds max {}", len, max_size),
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Handshake constants
// ---------------------------------------------------------------------------

pub const MAGIC_HANDSHAKE_REQ: &[u8; 4] = b"AGHI";
pub const MAGIC_HANDSHAKE_RESP: &[u8; 4] = b"AGHO";
pub const HANDSHAKE_OK: u32 = 0;
pub const HANDSHAKE_REJECTED_QUOTA: u32 = 1;
pub const HANDSHAKE_REJECTED_CAPACITY: u32 = 2;
pub const PROTOCOL_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Handshake types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct HandshakeRequest {
    pub protocol_version: u32,
    pub requested_memory: u64,
}

impl HandshakeRequest {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_HANDSHAKE_REQ).unwrap();
        write_u32(&mut buf, self.protocol_version).unwrap();
        write_u64(&mut buf, self.requested_memory).unwrap();
        buf
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_HANDSHAKE_REQ {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad handshake request magic"));
        }
        let protocol_version = read_u32(&mut r)?;
        let requested_memory = read_u64(&mut r)?;
        Ok(HandshakeRequest { protocol_version, requested_memory })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HandshakeResponse {
    pub status: u32,
    pub container_id: u64,
    pub granted_memory: u64,
}

impl HandshakeResponse {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_HANDSHAKE_RESP).unwrap();
        write_u32(&mut buf, self.status).unwrap();
        write_u64(&mut buf, self.container_id).unwrap();
        write_u64(&mut buf, self.granted_memory).unwrap();
        buf
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_HANDSHAKE_RESP {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad handshake response magic"));
        }
        let status = read_u32(&mut r)?;
        let container_id = read_u64(&mut r)?;
        let granted_memory = read_u64(&mut r)?;
        Ok(HandshakeResponse { status, container_id, granted_memory })
    }
}

// ---------------------------------------------------------------------------
// Eval constants
// ---------------------------------------------------------------------------

pub const MAGIC_REQUEST: &[u8; 4] = b"AGPU";
pub const MAGIC_RESPONSE: &[u8; 4] = b"AGPR";
pub const EVAL_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// WireOpKind — 46 variants (discriminants 0–45)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum WireOpKind {
    // 0–9: basic ops
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Relu,
    Exp,
    Log,
    Sqrt,
    Matmul,
    // 10
    FusedElementwise { kernel_source: String, function_name: String },
    // 11–14
    Softmax,
    Transpose { dim0: usize, dim1: usize },
    ScalarMul(f32),
    Gelu,
    // 15–20
    LayerNorm { eps: f32 },
    Embedding,
    Reshape { new_shape: Vec<usize> },
    Slice { dim: usize, start: usize, end: usize },
    Concat { dim: usize },
    AddBias,
    // 21–26
    SoftmaxCausal,
    Argmax,
    Sum,
    Mean,
    Abs,
    Sign,
    // 27–32
    Pow { exponent: f32 },
    Clamp { min_val: f32, max_val: f32 },
    Where,
    MaskedFill { value: f32 },
    Triu { diagonal: i32 },
    Tril { diagonal: i32 },
    // 33–34
    Gather { dim: usize },
    IndexSelect { dim: usize },
    // 35–39
    Conv1d { stride: usize, padding: usize },
    Conv2d { stride: (usize, usize), padding: (usize, usize) },
    BatchNorm { eps: f32 },
    MaxPool2d { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
    AvgPool2d { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
    // 40–45: backward ops
    Tanh,
    SoftmaxBackward,
    LayerNormBackward { eps: f32 },
    Conv2dBackwardInput { stride: (usize, usize), padding: (usize, usize) },
    EmbeddingBackward,
    BatchNormBackward { eps: f32 },
}

impl WireOpKind {
    pub fn discriminant(&self) -> u32 {
        match self {
            WireOpKind::Add => 0,
            WireOpKind::Sub => 1,
            WireOpKind::Mul => 2,
            WireOpKind::Div => 3,
            WireOpKind::Neg => 4,
            WireOpKind::Relu => 5,
            WireOpKind::Exp => 6,
            WireOpKind::Log => 7,
            WireOpKind::Sqrt => 8,
            WireOpKind::Matmul => 9,
            WireOpKind::FusedElementwise { .. } => 10,
            WireOpKind::Softmax => 11,
            WireOpKind::Transpose { .. } => 12,
            WireOpKind::ScalarMul(_) => 13,
            WireOpKind::Gelu => 14,
            WireOpKind::LayerNorm { .. } => 15,
            WireOpKind::Embedding => 16,
            WireOpKind::Reshape { .. } => 17,
            WireOpKind::Slice { .. } => 18,
            WireOpKind::Concat { .. } => 19,
            WireOpKind::AddBias => 20,
            WireOpKind::SoftmaxCausal => 21,
            WireOpKind::Argmax => 22,
            WireOpKind::Sum => 23,
            WireOpKind::Mean => 24,
            WireOpKind::Abs => 25,
            WireOpKind::Sign => 26,
            WireOpKind::Pow { .. } => 27,
            WireOpKind::Clamp { .. } => 28,
            WireOpKind::Where => 29,
            WireOpKind::MaskedFill { .. } => 30,
            WireOpKind::Triu { .. } => 31,
            WireOpKind::Tril { .. } => 32,
            WireOpKind::Gather { .. } => 33,
            WireOpKind::IndexSelect { .. } => 34,
            WireOpKind::Conv1d { .. } => 35,
            WireOpKind::Conv2d { .. } => 36,
            WireOpKind::BatchNorm { .. } => 37,
            WireOpKind::MaxPool2d { .. } => 38,
            WireOpKind::AvgPool2d { .. } => 39,
            WireOpKind::Tanh => 40,
            WireOpKind::SoftmaxBackward => 41,
            WireOpKind::LayerNormBackward { .. } => 42,
            WireOpKind::Conv2dBackwardInput { .. } => 43,
            WireOpKind::EmbeddingBackward => 44,
            WireOpKind::BatchNormBackward { .. } => 45,
        }
    }

    /// Write the op-specific payload (after the discriminant has been written).
    pub fn write_payload(&self, w: &mut impl Write) -> io::Result<()> {
        match self {
            // No payload for simple ops
            WireOpKind::Add | WireOpKind::Sub | WireOpKind::Mul | WireOpKind::Div
            | WireOpKind::Neg | WireOpKind::Relu | WireOpKind::Exp | WireOpKind::Log
            | WireOpKind::Sqrt | WireOpKind::Matmul | WireOpKind::Softmax
            | WireOpKind::Gelu | WireOpKind::Embedding | WireOpKind::AddBias
            | WireOpKind::SoftmaxCausal | WireOpKind::Argmax | WireOpKind::Sum
            | WireOpKind::Mean | WireOpKind::Abs | WireOpKind::Sign | WireOpKind::Where
            | WireOpKind::Tanh | WireOpKind::SoftmaxBackward
            | WireOpKind::EmbeddingBackward => Ok(()),

            WireOpKind::FusedElementwise { kernel_source, function_name } => {
                write_u32(w, kernel_source.len() as u32)?;
                w.write_all(kernel_source.as_bytes())?;
                write_u32(w, function_name.len() as u32)?;
                w.write_all(function_name.as_bytes())
            }
            WireOpKind::Transpose { dim0, dim1 } => {
                write_u32(w, *dim0 as u32)?;
                write_u32(w, *dim1 as u32)
            }
            WireOpKind::ScalarMul(scale) => write_f32(w, *scale),
            WireOpKind::LayerNorm { eps } => write_f32(w, *eps),
            WireOpKind::Reshape { new_shape } => {
                write_u32(w, new_shape.len() as u32)?;
                for &d in new_shape {
                    write_u64(w, d as u64)?;
                }
                Ok(())
            }
            WireOpKind::Slice { dim, start, end } => {
                write_u32(w, *dim as u32)?;
                write_u64(w, *start as u64)?;
                write_u64(w, *end as u64)
            }
            WireOpKind::Concat { dim } => write_u32(w, *dim as u32),
            WireOpKind::Pow { exponent } => write_f32(w, *exponent),
            WireOpKind::Clamp { min_val, max_val } => {
                write_f32(w, *min_val)?;
                write_f32(w, *max_val)
            }
            WireOpKind::MaskedFill { value } => write_f32(w, *value),
            WireOpKind::Triu { diagonal } => write_i32(w, *diagonal),
            WireOpKind::Tril { diagonal } => write_i32(w, *diagonal),
            WireOpKind::Gather { dim } => write_u32(w, *dim as u32),
            WireOpKind::IndexSelect { dim } => write_u32(w, *dim as u32),
            WireOpKind::Conv1d { stride, padding } => {
                write_u64(w, *stride as u64)?;
                write_u64(w, *padding as u64)
            }
            WireOpKind::Conv2d { stride, padding } => {
                write_u64(w, stride.0 as u64)?;
                write_u64(w, stride.1 as u64)?;
                write_u64(w, padding.0 as u64)?;
                write_u64(w, padding.1 as u64)
            }
            WireOpKind::BatchNorm { eps } => write_f32(w, *eps),
            WireOpKind::MaxPool2d { kernel_size, stride, padding } => {
                write_u64(w, kernel_size.0 as u64)?;
                write_u64(w, kernel_size.1 as u64)?;
                write_u64(w, stride.0 as u64)?;
                write_u64(w, stride.1 as u64)?;
                write_u64(w, padding.0 as u64)?;
                write_u64(w, padding.1 as u64)
            }
            WireOpKind::AvgPool2d { kernel_size, stride, padding } => {
                write_u64(w, kernel_size.0 as u64)?;
                write_u64(w, kernel_size.1 as u64)?;
                write_u64(w, stride.0 as u64)?;
                write_u64(w, stride.1 as u64)?;
                write_u64(w, padding.0 as u64)?;
                write_u64(w, padding.1 as u64)
            }
            WireOpKind::LayerNormBackward { eps } => write_f32(w, *eps),
            WireOpKind::Conv2dBackwardInput { stride, padding } => {
                write_u64(w, stride.0 as u64)?;
                write_u64(w, stride.1 as u64)?;
                write_u64(w, padding.0 as u64)?;
                write_u64(w, padding.1 as u64)
            }
            WireOpKind::BatchNormBackward { eps } => write_f32(w, *eps),
        }
    }

    /// Read the op from a discriminant + reader (reader positioned after the discriminant).
    pub fn read_from(disc: u32, r: &mut impl Read) -> io::Result<Self> {
        match disc {
            0 => Ok(WireOpKind::Add),
            1 => Ok(WireOpKind::Sub),
            2 => Ok(WireOpKind::Mul),
            3 => Ok(WireOpKind::Div),
            4 => Ok(WireOpKind::Neg),
            5 => Ok(WireOpKind::Relu),
            6 => Ok(WireOpKind::Exp),
            7 => Ok(WireOpKind::Log),
            8 => Ok(WireOpKind::Sqrt),
            9 => Ok(WireOpKind::Matmul),
            10 => {
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
                Ok(WireOpKind::FusedElementwise { kernel_source, function_name })
            }
            11 => Ok(WireOpKind::Softmax),
            12 => {
                let dim0 = read_u32(r)? as usize;
                let dim1 = read_u32(r)? as usize;
                Ok(WireOpKind::Transpose { dim0, dim1 })
            }
            13 => Ok(WireOpKind::ScalarMul(read_f32(r)?)),
            14 => Ok(WireOpKind::Gelu),
            15 => Ok(WireOpKind::LayerNorm { eps: read_f32(r)? }),
            16 => Ok(WireOpKind::Embedding),
            17 => {
                let ndims = read_u32(r)? as usize;
                let mut new_shape = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    new_shape.push(read_u64(r)? as usize);
                }
                Ok(WireOpKind::Reshape { new_shape })
            }
            18 => {
                let dim = read_u32(r)? as usize;
                let start = read_u64(r)? as usize;
                let end = read_u64(r)? as usize;
                Ok(WireOpKind::Slice { dim, start, end })
            }
            19 => Ok(WireOpKind::Concat { dim: read_u32(r)? as usize }),
            20 => Ok(WireOpKind::AddBias),
            21 => Ok(WireOpKind::SoftmaxCausal),
            22 => Ok(WireOpKind::Argmax),
            23 => Ok(WireOpKind::Sum),
            24 => Ok(WireOpKind::Mean),
            25 => Ok(WireOpKind::Abs),
            26 => Ok(WireOpKind::Sign),
            27 => Ok(WireOpKind::Pow { exponent: read_f32(r)? }),
            28 => {
                let min_val = read_f32(r)?;
                let max_val = read_f32(r)?;
                Ok(WireOpKind::Clamp { min_val, max_val })
            }
            29 => Ok(WireOpKind::Where),
            30 => Ok(WireOpKind::MaskedFill { value: read_f32(r)? }),
            31 => Ok(WireOpKind::Triu { diagonal: read_i32(r)? }),
            32 => Ok(WireOpKind::Tril { diagonal: read_i32(r)? }),
            33 => Ok(WireOpKind::Gather { dim: read_u32(r)? as usize }),
            34 => Ok(WireOpKind::IndexSelect { dim: read_u32(r)? as usize }),
            35 => {
                let stride = read_u64(r)? as usize;
                let padding = read_u64(r)? as usize;
                Ok(WireOpKind::Conv1d { stride, padding })
            }
            36 => {
                let s0 = read_u64(r)? as usize;
                let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize;
                let p1 = read_u64(r)? as usize;
                Ok(WireOpKind::Conv2d { stride: (s0, s1), padding: (p0, p1) })
            }
            37 => Ok(WireOpKind::BatchNorm { eps: read_f32(r)? }),
            38 => {
                let k0 = read_u64(r)? as usize;
                let k1 = read_u64(r)? as usize;
                let s0 = read_u64(r)? as usize;
                let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize;
                let p1 = read_u64(r)? as usize;
                Ok(WireOpKind::MaxPool2d {
                    kernel_size: (k0, k1), stride: (s0, s1), padding: (p0, p1),
                })
            }
            39 => {
                let k0 = read_u64(r)? as usize;
                let k1 = read_u64(r)? as usize;
                let s0 = read_u64(r)? as usize;
                let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize;
                let p1 = read_u64(r)? as usize;
                Ok(WireOpKind::AvgPool2d {
                    kernel_size: (k0, k1), stride: (s0, s1), padding: (p0, p1),
                })
            }
            40 => Ok(WireOpKind::Tanh),
            41 => Ok(WireOpKind::SoftmaxBackward),
            42 => Ok(WireOpKind::LayerNormBackward { eps: read_f32(r)? }),
            43 => {
                let s0 = read_u64(r)? as usize;
                let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize;
                let p1 = read_u64(r)? as usize;
                Ok(WireOpKind::Conv2dBackwardInput { stride: (s0, s1), padding: (p0, p1) })
            }
            44 => Ok(WireOpKind::EmbeddingBackward),
            45 => Ok(WireOpKind::BatchNormBackward { eps: read_f32(r)? }),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown op discriminant: {}", disc),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Wire graph node and tensor data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct WireOpNode {
    pub id: u64,
    pub op: WireOpKind,
    pub inputs: Vec<u64>,
    pub out_shape: Vec<usize>,
    pub out_dtype: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WireTensorData {
    pub id: u64,
    pub shape: Vec<usize>,
    pub dtype: u32,
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// EvalRequest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct EvalRequest {
    pub target_id: u64,
    pub tensors: Vec<WireTensorData>,
    pub nodes: Vec<WireOpNode>,
}

impl EvalRequest {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_REQUEST).unwrap();
        write_u32(&mut buf, EVAL_VERSION).unwrap();
        write_u64(&mut buf, self.target_id).unwrap();

        // Tensors
        write_u32(&mut buf, self.tensors.len() as u32).unwrap();
        for t in &self.tensors {
            write_u64(&mut buf, t.id).unwrap();
            write_u32(&mut buf, t.shape.len() as u32).unwrap();
            for &d in &t.shape {
                write_u64(&mut buf, d as u64).unwrap();
            }
            write_u32(&mut buf, t.dtype).unwrap();
            write_u32(&mut buf, t.data.len() as u32).unwrap();
            buf.write_all(&t.data).unwrap();
        }

        // Nodes
        write_u32(&mut buf, self.nodes.len() as u32).unwrap();
        for node in &self.nodes {
            write_u64(&mut buf, node.id).unwrap();
            write_u32(&mut buf, node.op.discriminant()).unwrap();
            write_u32(&mut buf, node.inputs.len() as u32).unwrap();
            for &inp in &node.inputs {
                write_u64(&mut buf, inp).unwrap();
            }
            write_u32(&mut buf, node.out_shape.len() as u32).unwrap();
            for &d in &node.out_shape {
                write_u64(&mut buf, d as u64).unwrap();
            }
            write_u32(&mut buf, node.out_dtype).unwrap();
            node.op.write_payload(&mut buf).unwrap();
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
        if version != EVAL_VERSION {
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
            let dtype = read_u32(&mut r)?;
            let data_len = read_u32(&mut r)? as usize;
            let mut data = vec![0u8; data_len];
            r.read_exact(&mut data)?;
            tensors.push(WireTensorData { id, shape, dtype, data });
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
            let out_dtype = read_u32(&mut r)?;
            let op = WireOpKind::read_from(op_disc, &mut r)?;
            nodes.push(WireOpNode { id, op, inputs, out_shape, out_dtype });
        }

        Ok(EvalRequest { target_id, tensors, nodes })
    }
}

// ---------------------------------------------------------------------------
// EvalResponse
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum EvalResponse {
    Ok {
        tensor_id: u64,
        shape: Vec<usize>,
        data: Vec<u8>,
    },
    Err(String),
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Task 1: framing tests ---

    #[test]
    fn framing_roundtrip() {
        let payload = b"hello wire protocol";
        let mut buf = Vec::new();
        write_message(&mut buf, payload).unwrap();

        let mut cursor = Cursor::new(&buf);
        let got = read_message(&mut cursor, MAX_MESSAGE_SIZE).unwrap();
        assert_eq!(got, payload);
    }

    #[test]
    fn framing_rejects_oversized() {
        // Fabricate a header claiming a huge message
        let mut buf = Vec::new();
        write_u32(&mut buf, 1_000_000).unwrap(); // 1 MB claim
        buf.extend_from_slice(&[0u8; 100]); // only 100 bytes of actual data

        let mut cursor = Cursor::new(&buf);
        let result = read_message(&mut cursor, 512); // max = 512 bytes
        assert!(result.is_err());
    }

    // --- Task 2: handshake tests ---

    #[test]
    fn handshake_request_roundtrip() {
        let req = HandshakeRequest {
            protocol_version: PROTOCOL_VERSION,
            requested_memory: 8 * 1024 * 1024 * 1024, // 8 GiB
        };
        let bytes = req.serialize();
        let got = HandshakeRequest::deserialize(&bytes).unwrap();
        assert_eq!(req, got);
    }

    #[test]
    fn handshake_response_roundtrip() {
        let resp = HandshakeResponse {
            status: HANDSHAKE_OK,
            container_id: 42,
            granted_memory: 4 * 1024 * 1024 * 1024,
        };
        let bytes = resp.serialize();
        let got = HandshakeResponse::deserialize(&bytes).unwrap();
        assert_eq!(resp, got);
    }

    #[test]
    fn handshake_response_rejected() {
        let resp = HandshakeResponse {
            status: HANDSHAKE_REJECTED_QUOTA,
            container_id: 0,
            granted_memory: 0,
        };
        let bytes = resp.serialize();
        let got = HandshakeResponse::deserialize(&bytes).unwrap();
        assert_eq!(got.status, HANDSHAKE_REJECTED_QUOTA);
        assert_eq!(got.container_id, 0);
        assert_eq!(got.granted_memory, 0);
    }

    // --- Task 3: eval tests ---

    #[test]
    fn eval_request_roundtrip_basic() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let req = EvalRequest {
            target_id: 42,
            tensors: vec![WireTensorData {
                id: 1,
                shape: vec![4],
                dtype: 0, // f32
                data: bytes,
            }],
            nodes: vec![WireOpNode {
                id: 42,
                op: WireOpKind::Neg,
                inputs: vec![1],
                out_shape: vec![4],
                out_dtype: 0,
            }],
        };

        let serialized = req.serialize();
        let got = EvalRequest::deserialize(&serialized).unwrap();
        assert_eq!(req, got);
    }

    #[test]
    fn eval_request_roundtrip_fused() {
        let req = EvalRequest {
            target_id: 10,
            tensors: vec![],
            nodes: vec![WireOpNode {
                id: 10,
                op: WireOpKind::FusedElementwise {
                    kernel_source: "kernel void test() {}".to_string(),
                    function_name: "test".to_string(),
                },
                inputs: vec![1, 2],
                out_shape: vec![4],
                out_dtype: 0,
            }],
        };

        let serialized = req.serialize();
        let got = EvalRequest::deserialize(&serialized).unwrap();
        assert_eq!(req, got);
    }

    #[test]
    fn eval_response_ok_roundtrip() {
        let resp = EvalResponse::Ok {
            tensor_id: 42,
            shape: vec![2, 3],
            data: vec![1, 2, 3, 4],
        };
        let serialized = resp.serialize();
        let got = EvalResponse::deserialize(&serialized).unwrap();
        assert_eq!(resp, got);
    }

    #[test]
    fn eval_response_err_roundtrip() {
        let resp = EvalResponse::Err("something failed".to_string());
        let serialized = resp.serialize();
        let got = EvalResponse::deserialize(&serialized).unwrap();
        assert_eq!(resp, got);
    }
}
