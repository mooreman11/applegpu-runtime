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
pub const PROTOCOL_VERSION: u32 = 2;

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
    Sigmoid,
    Var { correction: u32 },
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
    Conv1d { stride: usize, padding: usize, groups: usize },
    Conv2d { stride: (usize, usize), padding: (usize, usize), groups: usize },
    BatchNorm { eps: f32 },
    MaxPool2d { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
    AvgPool2d { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
    // 40–45: backward ops
    Tanh,
    SoftmaxBackward,
    LayerNormBackward { eps: f32 },
    Conv2dBackwardInput { stride: (usize, usize), padding: (usize, usize), groups: usize },
    Conv2dBackwardWeight { stride: (usize, usize), padding: (usize, usize), groups: usize },
    Conv1dBackwardInput { stride: usize, padding: usize, groups: usize },
    EmbeddingBackward,
    BatchNormBackward { eps: f32 },
    // 46: type conversion
    Cast { target_dtype: u8 },
    // 47–52: comparison (output Bool)
    Lt, Gt, Le, Ge, Eq, Ne,
    // 53–56: bitwise
    BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot,
    // 57–58: shifts
    Shl { shift: u32 }, Shr { shift: u32 },
    // 59–62: utility
    Mod, ElemMin, ElemMax, LogicalNot,
    // 63–64: quantization
    Quantize { scale: f32, zero_point: i32, target_dtype: u8 },
    Dequantize { scale: f32, zero_point: i32, target_dtype: u8 },
    // 65–67: new ops
    Sin,
    Cos,
    LogSoftmax,
    // 70: threshold backward
    ThresholdBackward { threshold: f32 },
    // 71: tanh backward
    TanhBackward,
    // 72: sigmoid backward
    SigmoidBackward,
    // 73: gelu backward
    GeluBackward,
    // 75: max_pool2d backward
    MaxPool2dBackward,
    // 76: max_pool2d with indices (dual output)
    MaxPool2dWithIndices { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize), indices_id: u64 },
    // 77-79: exact GELU and tanh GELU backward
    GeluExact,
    GeluExactBackward,
    GeluTanhBackward,
    // 81-82: scatter ops
    ScatterWrite,
    ScatterAdd,
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
            WireOpKind::Conv2dBackwardWeight { .. } => 80,
            WireOpKind::EmbeddingBackward => 44,
            WireOpKind::BatchNormBackward { .. } => 45,
            WireOpKind::Cast { .. } => 46,
            WireOpKind::Lt => 47,
            WireOpKind::Gt => 48,
            WireOpKind::Le => 49,
            WireOpKind::Ge => 50,
            WireOpKind::Eq => 51,
            WireOpKind::Ne => 52,
            WireOpKind::BitwiseAnd => 53,
            WireOpKind::BitwiseOr => 54,
            WireOpKind::BitwiseXor => 55,
            WireOpKind::BitwiseNot => 56,
            WireOpKind::Shl { .. } => 57,
            WireOpKind::Shr { .. } => 58,
            WireOpKind::Mod => 59,
            WireOpKind::ElemMin => 60,
            WireOpKind::ElemMax => 61,
            WireOpKind::LogicalNot => 62,
            WireOpKind::Quantize { .. } => 63,
            WireOpKind::Dequantize { .. } => 64,
            WireOpKind::Sin => 65,
            WireOpKind::Cos => 66,
            WireOpKind::LogSoftmax => 67,
            WireOpKind::Sigmoid => 68,
            WireOpKind::Var { .. } => 69,
            WireOpKind::ThresholdBackward { .. } => 70,
            WireOpKind::TanhBackward => 71,
            WireOpKind::SigmoidBackward => 72,
            WireOpKind::GeluBackward => 73,
            WireOpKind::Conv1dBackwardInput { .. } => 74,
            WireOpKind::MaxPool2dBackward => 75,
            WireOpKind::MaxPool2dWithIndices { .. } => 76,
            WireOpKind::GeluExact => 77,
            WireOpKind::GeluExactBackward => 78,
            WireOpKind::GeluTanhBackward => 79,
            WireOpKind::ScatterWrite => 81,
            WireOpKind::ScatterAdd => 82,
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
            | WireOpKind::Tanh | WireOpKind::Sin | WireOpKind::Cos | WireOpKind::Sigmoid
            | WireOpKind::SoftmaxBackward
            | WireOpKind::EmbeddingBackward
            | WireOpKind::LogSoftmax
            | WireOpKind::Lt | WireOpKind::Gt | WireOpKind::Le | WireOpKind::Ge
            | WireOpKind::Eq | WireOpKind::Ne
            | WireOpKind::BitwiseAnd | WireOpKind::BitwiseOr | WireOpKind::BitwiseXor
            | WireOpKind::BitwiseNot
            | WireOpKind::Mod | WireOpKind::ElemMin | WireOpKind::ElemMax
            | WireOpKind::LogicalNot => Ok(()),

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
            WireOpKind::Var { correction } => write_u32(w, *correction),
            WireOpKind::Clamp { min_val, max_val } => {
                write_f32(w, *min_val)?;
                write_f32(w, *max_val)
            }
            WireOpKind::MaskedFill { value } => write_f32(w, *value),
            WireOpKind::Triu { diagonal } => write_i32(w, *diagonal),
            WireOpKind::Tril { diagonal } => write_i32(w, *diagonal),
            WireOpKind::Gather { dim } => write_u32(w, *dim as u32),
            WireOpKind::IndexSelect { dim } => write_u32(w, *dim as u32),
            WireOpKind::Conv1d { stride, padding, groups } => {
                write_u64(w, *stride as u64)?;
                write_u64(w, *padding as u64)?;
                write_u64(w, *groups as u64)
            }
            WireOpKind::Conv2d { stride, padding, groups } => {
                write_u64(w, stride.0 as u64)?;
                write_u64(w, stride.1 as u64)?;
                write_u64(w, padding.0 as u64)?;
                write_u64(w, padding.1 as u64)?;
                write_u64(w, *groups as u64)
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
            WireOpKind::Conv2dBackwardInput { stride, padding, groups } => {
                write_u64(w, stride.0 as u64)?;
                write_u64(w, stride.1 as u64)?;
                write_u64(w, padding.0 as u64)?;
                write_u64(w, padding.1 as u64)?;
                write_u64(w, *groups as u64)
            }
            WireOpKind::Conv2dBackwardWeight { stride, padding, groups } => {
                write_u64(w, stride.0 as u64)?;
                write_u64(w, stride.1 as u64)?;
                write_u64(w, padding.0 as u64)?;
                write_u64(w, padding.1 as u64)?;
                write_u64(w, *groups as u64)
            }
            WireOpKind::Conv1dBackwardInput { stride, padding, groups } => {
                write_u32(w, *stride as u32)?;
                write_u32(w, *padding as u32)?;
                write_u32(w, *groups as u32)
            }
            WireOpKind::BatchNormBackward { eps } => write_f32(w, *eps),
            WireOpKind::Cast { target_dtype } => w.write_all(&[*target_dtype]),
            WireOpKind::Shl { shift } => write_u32(w, *shift),
            WireOpKind::Shr { shift } => write_u32(w, *shift),
            WireOpKind::Quantize { scale, zero_point, target_dtype } => {
                write_f32(w, *scale)?;
                write_i32(w, *zero_point)?;
                w.write_all(&[*target_dtype])
            }
            WireOpKind::Dequantize { scale, zero_point, target_dtype } => {
                write_f32(w, *scale)?;
                write_i32(w, *zero_point)?;
                w.write_all(&[*target_dtype])
            }
            WireOpKind::ThresholdBackward { threshold } => write_f32(w, *threshold),
            WireOpKind::TanhBackward | WireOpKind::SigmoidBackward | WireOpKind::GeluBackward |
            WireOpKind::MaxPool2dBackward |
            WireOpKind::GeluExact | WireOpKind::GeluExactBackward | WireOpKind::GeluTanhBackward |
            WireOpKind::ScatterWrite | WireOpKind::ScatterAdd => Ok(()),
            WireOpKind::MaxPool2dWithIndices { kernel_size, stride, padding, indices_id } => {
                write_u64(w, kernel_size.0 as u64)?;
                write_u64(w, kernel_size.1 as u64)?;
                write_u64(w, stride.0 as u64)?;
                write_u64(w, stride.1 as u64)?;
                write_u64(w, padding.0 as u64)?;
                write_u64(w, padding.1 as u64)?;
                write_u64(w, *indices_id)
            }
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
                let groups = read_u64(r)? as usize;
                Ok(WireOpKind::Conv1d { stride, padding, groups })
            }
            36 => {
                let s0 = read_u64(r)? as usize;
                let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize;
                let p1 = read_u64(r)? as usize;
                let groups = read_u64(r)? as usize;
                Ok(WireOpKind::Conv2d { stride: (s0, s1), padding: (p0, p1), groups })
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
                let groups = read_u64(r)? as usize;
                Ok(WireOpKind::Conv2dBackwardInput { stride: (s0, s1), padding: (p0, p1), groups })
            }
            44 => Ok(WireOpKind::EmbeddingBackward),
            45 => Ok(WireOpKind::BatchNormBackward { eps: read_f32(r)? }),
            46 => {
                let mut b = [0u8; 1];
                r.read_exact(&mut b)?;
                Ok(WireOpKind::Cast { target_dtype: b[0] })
            }
            47 => Ok(WireOpKind::Lt),
            48 => Ok(WireOpKind::Gt),
            49 => Ok(WireOpKind::Le),
            50 => Ok(WireOpKind::Ge),
            51 => Ok(WireOpKind::Eq),
            52 => Ok(WireOpKind::Ne),
            53 => Ok(WireOpKind::BitwiseAnd),
            54 => Ok(WireOpKind::BitwiseOr),
            55 => Ok(WireOpKind::BitwiseXor),
            56 => Ok(WireOpKind::BitwiseNot),
            57 => Ok(WireOpKind::Shl { shift: read_u32(r)? }),
            58 => Ok(WireOpKind::Shr { shift: read_u32(r)? }),
            59 => Ok(WireOpKind::Mod),
            60 => Ok(WireOpKind::ElemMin),
            61 => Ok(WireOpKind::ElemMax),
            62 => Ok(WireOpKind::LogicalNot),
            63 => {
                let scale = read_f32(r)?;
                let zero_point = read_i32(r)?;
                let mut b = [0u8; 1];
                r.read_exact(&mut b)?;
                Ok(WireOpKind::Quantize { scale, zero_point, target_dtype: b[0] })
            }
            64 => {
                let scale = read_f32(r)?;
                let zero_point = read_i32(r)?;
                let mut b = [0u8; 1];
                r.read_exact(&mut b)?;
                Ok(WireOpKind::Dequantize { scale, zero_point, target_dtype: b[0] })
            }
            65 => Ok(WireOpKind::Sin),
            66 => Ok(WireOpKind::Cos),
            67 => Ok(WireOpKind::LogSoftmax),
            68 => Ok(WireOpKind::Sigmoid),
            69 => Ok(WireOpKind::Var { correction: read_u32(r)? }),
            70 => Ok(WireOpKind::ThresholdBackward { threshold: read_f32(r)? }),
            71 => Ok(WireOpKind::TanhBackward),
            72 => Ok(WireOpKind::SigmoidBackward),
            73 => Ok(WireOpKind::GeluBackward),
            74 => {
                let stride = read_u32(r)? as usize;
                let padding = read_u32(r)? as usize;
                let groups = read_u32(r)? as usize;
                Ok(WireOpKind::Conv1dBackwardInput { stride, padding, groups })
            }
            75 => Ok(WireOpKind::MaxPool2dBackward),
            76 => {
                let k0 = read_u64(r)? as usize;
                let k1 = read_u64(r)? as usize;
                let s0 = read_u64(r)? as usize;
                let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize;
                let p1 = read_u64(r)? as usize;
                let indices_id = read_u64(r)?;
                Ok(WireOpKind::MaxPool2dWithIndices {
                    kernel_size: (k0, k1), stride: (s0, s1), padding: (p0, p1), indices_id,
                })
            }
            77 => Ok(WireOpKind::GeluExact),
            78 => Ok(WireOpKind::GeluExactBackward),
            79 => Ok(WireOpKind::GeluTanhBackward),
            80 => {
                let s0 = read_u64(r)? as usize;
                let s1 = read_u64(r)? as usize;
                let p0 = read_u64(r)? as usize;
                let p1 = read_u64(r)? as usize;
                let groups = read_u64(r)? as usize;
                Ok(WireOpKind::Conv2dBackwardWeight { stride: (s0, s1), padding: (p0, p1), groups })
            }
            81 => Ok(WireOpKind::ScatterWrite),
            82 => Ok(WireOpKind::ScatterAdd),
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

// ---------------------------------------------------------------------------
// ReadTensor constants
// ---------------------------------------------------------------------------

pub const MAGIC_READ_REQ: &[u8; 4] = b"AGRD";
pub const MAGIC_READ_RESP: &[u8; 4] = b"AGRR";
pub const READ_TENSOR_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// ReadTensorRequest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ReadTensorRequest {
    pub tensor_id: u64,
}

impl ReadTensorRequest {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_READ_REQ).unwrap();
        write_u32(&mut buf, READ_TENSOR_VERSION).unwrap();
        write_u64(&mut buf, self.tensor_id).unwrap();
        buf
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_READ_REQ {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad ReadTensorRequest magic"));
        }
        let version = read_u32(&mut r)?;
        if version != READ_TENSOR_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad ReadTensorRequest version"));
        }
        let tensor_id = read_u64(&mut r)?;
        Ok(ReadTensorRequest { tensor_id })
    }
}

// ---------------------------------------------------------------------------
// ReadTensorResponse
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum ReadTensorResponse {
    Ok {
        tensor_id: u64,
        shape: Vec<usize>,
        dtype: u32,
        data: Vec<u8>,
    },
    NotFound {
        tensor_id: u64,
    },
}

impl ReadTensorResponse {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_READ_RESP).unwrap();
        match self {
            ReadTensorResponse::Ok { tensor_id, shape, dtype, data } => {
                write_u32(&mut buf, 0).unwrap(); // status: ok
                write_u64(&mut buf, *tensor_id).unwrap();
                write_u32(&mut buf, shape.len() as u32).unwrap();
                for &d in shape {
                    write_u64(&mut buf, d as u64).unwrap();
                }
                write_u32(&mut buf, *dtype).unwrap();
                write_u32(&mut buf, data.len() as u32).unwrap();
                buf.write_all(data).unwrap();
            }
            ReadTensorResponse::NotFound { tensor_id } => {
                write_u32(&mut buf, 1).unwrap(); // status: not found
                write_u64(&mut buf, *tensor_id).unwrap();
            }
        }
        buf
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC_READ_RESP {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad ReadTensorResponse magic"));
        }
        let status = read_u32(&mut r)?;
        if status == 0 {
            let tensor_id = read_u64(&mut r)?;
            let num_dims = read_u32(&mut r)? as usize;
            let mut shape = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                shape.push(read_u64(&mut r)? as usize);
            }
            let dtype = read_u32(&mut r)?;
            let data_len = read_u32(&mut r)? as usize;
            let mut data = vec![0u8; data_len];
            r.read_exact(&mut data)?;
            Ok(ReadTensorResponse::Ok { tensor_id, shape, dtype, data })
        } else {
            let tensor_id = read_u64(&mut r)?;
            Ok(ReadTensorResponse::NotFound { tensor_id })
        }
    }
}

// ---------------------------------------------------------------------------
// peek_magic — peek at the first 4 bytes of a message to determine its type
// ---------------------------------------------------------------------------

/// Peek at the first 4 bytes of a message buffer to determine the message type.
/// Returns the 4-byte magic, or an error if the buffer is too short.
pub fn peek_magic(data: &[u8]) -> io::Result<[u8; 4]> {
    if data.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Message too short to contain magic bytes",
        ));
    }
    let mut magic = [0u8; 4];
    magic.copy_from_slice(&data[..4]);
    Ok(magic)
}

// ---------------------------------------------------------------------------
// WireDType — platform-independent dtype enum for socket backend
// ---------------------------------------------------------------------------

/// Data type for tensor elements (wire-protocol level).
///
/// Mirrors `applegpu_core::tensor::DType` but lives in the wire crate so it
/// can be used on Linux without pulling in Metal/Swift dependencies.
/// Wire discriminants match the dtype field in `WireTensorData`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WireDType {
    Float32  = 0,
    Float16  = 1,
    Float64  = 2,
    Int8     = 3,
    Int16    = 4,
    Int32    = 5,
    Int64    = 6,
    UInt8    = 7,
    UInt32   = 8,
    Bool     = 9,
    BFloat16 = 10,
}

impl WireDType {
    pub fn from_discriminant(d: u32) -> Option<Self> {
        match d {
            0 => Some(WireDType::Float32),
            1 => Some(WireDType::Float16),
            2 => Some(WireDType::Float64),
            3 => Some(WireDType::Int8),
            4 => Some(WireDType::Int16),
            5 => Some(WireDType::Int32),
            6 => Some(WireDType::Int64),
            7 => Some(WireDType::UInt8),
            8 => Some(WireDType::UInt32),
            9 => Some(WireDType::Bool),
            10 => Some(WireDType::BFloat16),
            _ => None,
        }
    }

    pub fn discriminant(&self) -> u32 {
        *self as u32
    }

    /// Size of one element in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            WireDType::Bool | WireDType::Int8 | WireDType::UInt8 => 1,
            WireDType::Float16 | WireDType::BFloat16 | WireDType::Int16 => 2,
            WireDType::Float32 | WireDType::Int32 | WireDType::UInt32 => 4,
            WireDType::Float64 | WireDType::Int64 => 8,
        }
    }

    /// Map from string name to WireDType.
    pub fn from_name(name: &str) -> Option<WireDType> {
        match name {
            "float16" | "f16" => Some(WireDType::Float16),
            "float32" | "f32" => Some(WireDType::Float32),
            "float64" | "f64" => Some(WireDType::Float64),
            "int8" | "i8" => Some(WireDType::Int8),
            "int16" | "i16" => Some(WireDType::Int16),
            "int32" | "i32" => Some(WireDType::Int32),
            "int64" | "i64" => Some(WireDType::Int64),
            "uint8" | "u8" => Some(WireDType::UInt8),
            "uint32" | "u32" => Some(WireDType::UInt32),
            "bool" | "bool_" => Some(WireDType::Bool),
            "bfloat16" | "bf16" => Some(WireDType::BFloat16),
            _ => None,
        }
    }

    /// Map to string name.
    pub fn name(&self) -> &'static str {
        match self {
            WireDType::Float16 => "float16",
            WireDType::Float32 => "float32",
            WireDType::Float64 => "float64",
            WireDType::Int8 => "int8",
            WireDType::Int16 => "int16",
            WireDType::Int32 => "int32",
            WireDType::Int64 => "int64",
            WireDType::UInt8 => "uint8",
            WireDType::UInt32 => "uint32",
            WireDType::Bool => "bool",
            WireDType::BFloat16 => "bfloat16",
        }
    }
}

// ---------------------------------------------------------------------------
// Shape inference — platform-independent, used by SocketBackend on Linux
// ---------------------------------------------------------------------------

/// Broadcast two shapes element-wise (NumPy-style broadcasting).
/// Returns the output shape, or an error if shapes are incompatible.
pub fn infer_broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let ndim = a.len().max(b.len());
    let mut result = vec![0usize; ndim];
    for i in 0..ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da == db {
            result[ndim - 1 - i] = da;
        } else if da == 1 {
            result[ndim - 1 - i] = db;
        } else if db == 1 {
            result[ndim - 1 - i] = da;
        } else {
            return Err(format!(
                "Shapes {:?} and {:?} are not broadcast-compatible", a, b
            ));
        }
    }
    Ok(result)
}

/// Infer the output shape of matmul(a, b).
/// Supports 2D ([M,K] @ [K,N] -> [M,N]) and batched matmul.
pub fn infer_matmul_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    if a.len() < 2 || b.len() < 2 {
        return Err(format!(
            "Matmul requires at least 2D tensors, got {:?} and {:?}", a, b
        ));
    }
    let m = a[a.len() - 2];
    let k1 = a[a.len() - 1];
    let k2 = b[b.len() - 2];
    let n = b[b.len() - 1];
    if k1 != k2 {
        return Err(format!(
            "Matmul inner dimensions mismatch: {:?} vs {:?}", a, b
        ));
    }
    // Batch dimensions
    let a_batch = &a[..a.len() - 2];
    let b_batch = &b[..b.len() - 2];
    let mut batch = infer_broadcast_shape(a_batch, b_batch)?;
    batch.push(m);
    batch.push(n);
    Ok(batch)
}

/// Infer the output shape of a transpose (swap last two dims).
pub fn infer_transpose_shape(a: &[usize]) -> Result<Vec<usize>, String> {
    if a.len() < 2 {
        return Err("Transpose requires at least 2D tensor".to_string());
    }
    let mut out = a.to_vec();
    let n = out.len();
    out.swap(n - 2, n - 1);
    Ok(out)
}

/// Infer the output shape of transpose_dims(a, dim0, dim1).
pub fn infer_transpose_dims_shape(a: &[usize], dim0: usize, dim1: usize) -> Result<Vec<usize>, String> {
    if dim0 >= a.len() || dim1 >= a.len() {
        return Err(format!(
            "Transpose dims {} and {} out of range for {:?}", dim0, dim1, a
        ));
    }
    let mut out = a.to_vec();
    out.swap(dim0, dim1);
    Ok(out)
}

/// Infer the output shape of a reduction to scalar (e.g. sum, mean).
pub fn infer_reduce_shape(_a: &[usize]) -> Vec<usize> {
    vec![1]
}

/// Infer the output shape of argmax (reduces to 1D of length batch_size or 1).
pub fn infer_argmax_shape(a: &[usize]) -> Vec<usize> {
    // Argmax over the last dim, returns shape without last dim (or [1] for 1D)
    if a.len() <= 1 {
        vec![1]
    } else {
        a[..a.len() - 1].to_vec()
    }
}

/// Infer the output shape of slice(a, dim, start, end).
pub fn infer_slice_shape(a: &[usize], dim: usize, start: usize, end: usize) -> Result<Vec<usize>, String> {
    if dim >= a.len() {
        return Err(format!("Slice dim {} out of range for {:?}", dim, a));
    }
    if end > a[dim] || start > end {
        return Err(format!(
            "Invalid slice range [{}..{}) for dim {} of {:?}", start, end, dim, a
        ));
    }
    let mut out = a.to_vec();
    out[dim] = end - start;
    Ok(out)
}

/// Infer the output shape of concat(a, b, dim).
pub fn infer_concat_shape(a: &[usize], b: &[usize], dim: usize) -> Result<Vec<usize>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Concat requires same number of dims, got {:?} and {:?}", a, b
        ));
    }
    if dim >= a.len() {
        return Err(format!("Concat dim {} out of range for {:?}", dim, a));
    }
    for i in 0..a.len() {
        if i != dim && a[i] != b[i] {
            return Err(format!(
                "Concat shapes {:?} and {:?} differ on non-concat dim {}", a, b, i
            ));
        }
    }
    let mut out = a.to_vec();
    out[dim] = a[dim] + b[dim];
    Ok(out)
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

    // --- ReadTensor tests ---

    #[test]
    fn read_tensor_request_roundtrip() {
        let req = ReadTensorRequest { tensor_id: 99 };
        let bytes = req.serialize();
        let got = ReadTensorRequest::deserialize(&bytes).unwrap();
        assert_eq!(req, got);
    }

    #[test]
    fn read_tensor_request_bad_magic() {
        let mut bytes = ReadTensorRequest { tensor_id: 1 }.serialize();
        bytes[0] = b'X'; // corrupt magic
        assert!(ReadTensorRequest::deserialize(&bytes).is_err());
    }

    #[test]
    fn read_tensor_response_ok_roundtrip() {
        let resp = ReadTensorResponse::Ok {
            tensor_id: 42,
            shape: vec![2, 3],
            dtype: 0,
            data: vec![1, 2, 3, 4, 5, 6],
        };
        let bytes = resp.serialize();
        let got = ReadTensorResponse::deserialize(&bytes).unwrap();
        assert_eq!(resp, got);
    }

    #[test]
    fn read_tensor_response_not_found_roundtrip() {
        let resp = ReadTensorResponse::NotFound { tensor_id: 77 };
        let bytes = resp.serialize();
        let got = ReadTensorResponse::deserialize(&bytes).unwrap();
        assert_eq!(resp, got);
    }

    #[test]
    fn read_tensor_response_bad_magic() {
        let mut bytes = ReadTensorResponse::NotFound { tensor_id: 1 }.serialize();
        bytes[0] = b'Z';
        assert!(ReadTensorResponse::deserialize(&bytes).is_err());
    }

    #[test]
    fn peek_magic_works() {
        let req = ReadTensorRequest { tensor_id: 1 };
        let bytes = req.serialize();
        let magic = peek_magic(&bytes).unwrap();
        assert_eq!(&magic, MAGIC_READ_REQ);
    }

    #[test]
    fn peek_magic_eval_request() {
        let req = EvalRequest {
            target_id: 1,
            tensors: vec![],
            nodes: vec![],
        };
        let bytes = req.serialize();
        let magic = peek_magic(&bytes).unwrap();
        assert_eq!(&magic, MAGIC_REQUEST);
    }

    #[test]
    fn peek_magic_too_short() {
        assert!(peek_magic(&[0u8; 3]).is_err());
    }

    // --- WireDType tests ---

    #[test]
    fn wire_dtype_roundtrip() {
        for d in 0..=10u32 {
            let dt = WireDType::from_discriminant(d).unwrap();
            assert_eq!(dt.discriminant(), d);
        }
        assert!(WireDType::from_discriminant(99).is_none());
    }

    #[test]
    fn wire_dtype_from_name() {
        assert_eq!(WireDType::from_name("float32"), Some(WireDType::Float32));
        assert_eq!(WireDType::from_name("bool"), Some(WireDType::Bool));
        assert_eq!(WireDType::from_name("int64"), Some(WireDType::Int64));
        assert!(WireDType::from_name("unknown").is_none());
    }

    #[test]
    fn wire_dtype_size_bytes() {
        assert_eq!(WireDType::Float32.size_bytes(), 4);
        assert_eq!(WireDType::Float16.size_bytes(), 2);
        assert_eq!(WireDType::Bool.size_bytes(), 1);
        assert_eq!(WireDType::Int64.size_bytes(), 8);
    }

    // --- Shape inference tests ---

    #[test]
    fn broadcast_same_shape() {
        assert_eq!(infer_broadcast_shape(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn broadcast_scalar() {
        assert_eq!(infer_broadcast_shape(&[2, 3], &[1]).unwrap(), vec![2, 3]);
        assert_eq!(infer_broadcast_shape(&[1], &[4, 5]).unwrap(), vec![4, 5]);
    }

    #[test]
    fn broadcast_expand() {
        assert_eq!(infer_broadcast_shape(&[1, 3], &[2, 1]).unwrap(), vec![2, 3]);
        assert_eq!(infer_broadcast_shape(&[3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn broadcast_incompatible() {
        assert!(infer_broadcast_shape(&[2, 3], &[4]).is_err());
    }

    #[test]
    fn matmul_2d() {
        assert_eq!(infer_matmul_shape(&[2, 3], &[3, 4]).unwrap(), vec![2, 4]);
    }

    #[test]
    fn matmul_batched() {
        assert_eq!(
            infer_matmul_shape(&[5, 2, 3], &[5, 3, 4]).unwrap(),
            vec![5, 2, 4]
        );
    }

    #[test]
    fn matmul_inner_mismatch() {
        assert!(infer_matmul_shape(&[2, 3], &[4, 5]).is_err());
    }

    #[test]
    fn matmul_1d_fails() {
        assert!(infer_matmul_shape(&[3], &[3, 4]).is_err());
    }

    #[test]
    fn transpose_shape() {
        assert_eq!(infer_transpose_shape(&[2, 3]).unwrap(), vec![3, 2]);
        assert_eq!(infer_transpose_shape(&[5, 2, 3]).unwrap(), vec![5, 3, 2]);
    }

    #[test]
    fn transpose_dims_shape() {
        assert_eq!(
            infer_transpose_dims_shape(&[2, 3, 4], 0, 2).unwrap(),
            vec![4, 3, 2]
        );
    }

    #[test]
    fn slice_shape() {
        assert_eq!(infer_slice_shape(&[4, 5], 0, 1, 3).unwrap(), vec![2, 5]);
        assert_eq!(infer_slice_shape(&[4, 5], 1, 0, 2).unwrap(), vec![4, 2]);
    }

    #[test]
    fn concat_shape() {
        assert_eq!(
            infer_concat_shape(&[2, 3], &[2, 4], 1).unwrap(),
            vec![2, 7]
        );
    }

    #[test]
    fn concat_shape_mismatch() {
        assert!(infer_concat_shape(&[2, 3], &[4, 3], 1).is_err());
    }
}
