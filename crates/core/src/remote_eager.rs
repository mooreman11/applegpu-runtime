//! Remote eager runtime — records ops locally with proxy buffers,
//! dispatches to gpu-service via wire protocol on flush.
//!
//! This module enables the PrivateUse1 C++ backend to work inside
//! Docker containers where Metal GPU is not available. Ops are
//! recorded locally and batched; on `flush_and_wait()`, the batch
//! is serialized as a wire protocol EvalRequest and sent to
//! gpu-service running on the host.
//!
//! Key invariant: each tensor's local buffer pointer is stable for
//! its entire lifetime (Box<[u8]> never reallocates).

use std::collections::HashMap;
use std::pin::Pin;

use crate::tensor::DType;
use applegpu_wire::{self as wire, EvalRequest, EvalResponse, WireOpNode, WireTensorData};

/// A tensor stored locally with a stable-pointer proxy buffer.
pub(crate) struct RemoteTensor {
    /// Pinned local buffer — its pointer is given to PyTorch and must not move.
    buf: Pin<Box<[u8]>>,
    shape: Vec<usize>,
    dtype: DType,
    strides: Vec<usize>,
    offset: usize,
    /// True if local buf contains valid data (written by CPU or received from server).
    materialized: bool,
    /// For views: the base tensor ID that owns the actual buffer.
    base_id: Option<u64>,
}

impl RemoteTensor {
    fn byte_size(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.size_bytes()
    }
}

/// A pending operation to be sent to the server on flush.
struct PendingOp {
    output_id: u64,
    op: wire::WireOpKind,
    inputs: Vec<u64>,
    output_shape: Vec<usize>,
    output_dtype: u32,
}

/// Remote eager runtime that proxies GPU ops to a gpu-service over socket.
pub struct RemoteEagerRuntime {
    tensors: HashMap<u64, RemoteTensor>,
    pending_ops: Vec<PendingOp>,
    /// Server-side tensor IDs that the server already knows about
    /// (sent in a previous eval and still alive).
    server_known: std::collections::HashSet<u64>,
    next_id: u64,
    /// Wire-protocol client connection.
    client: Option<applegpu_client::GpuClient>,
    /// Deferred free list (freed after flush, like EagerRuntime).
    pending_frees: Vec<u64>,
}

impl RemoteEagerRuntime {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            pending_ops: Vec::new(),
            server_known: std::collections::HashSet::new(),
            next_id: 1,
            client: None,
            pending_frees: Vec::new(),
        }
    }

    /// Connect to gpu-service. Called lazily on first op if not already connected.
    pub fn connect(&mut self) -> Result<(), String> {
        if self.client.is_some() {
            return Ok(());
        }
        match applegpu_client::GpuClient::connect_auto(512 * 1024 * 1024) {
            Ok(c) => {
                self.client = Some(c);
                Ok(())
            }
            Err(e) => Err(format!("Failed to connect to gpu-service: {}", e)),
        }
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ── Allocation ────────────────────────────────────────────────

    /// Allocate a tensor with a local proxy buffer.
    /// Returns (tensor_id, data_ptr). The data_ptr is stable.
    pub fn alloc(&mut self, shape: &[usize], dtype: DType) -> Result<(u64, *mut u8), String> {
        let id = self.next_id();
        let byte_size = shape.iter().product::<usize>() * dtype.size_bytes();
        // Allocate zeroed buffer, pinned so it won't move
        let buf: Pin<Box<[u8]>> = Pin::new(vec![0u8; byte_size.max(4)].into_boxed_slice());
        let ptr = buf.as_ptr() as *mut u8;
        let strides = contiguous_strides(shape);
        self.tensors.insert(id, RemoteTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
            strides,
            offset: 0,
            materialized: false,
            base_id: None,
        });
        Ok((id, ptr))
    }

    /// Free a tensor. If it's referenced by pending ops, defer the free.
    pub fn free(&mut self, id: u64) {
        // Check if any pending op references this tensor as input
        let in_use = self.pending_ops.iter().any(|op| op.inputs.contains(&id));
        if in_use {
            // Defer: will be freed after flush
            if !self.pending_frees.contains(&id) {
                self.pending_frees.push(id);
            }
            return;
        }
        self.tensors.remove(&id);
        self.server_known.remove(&id);
    }

    // ── Shape/dtype queries ───────────────────────────────────────

    pub fn shape(&self, id: u64) -> Result<&[usize], String> {
        self.tensors.get(&id)
            .map(|t| t.shape.as_slice())
            .ok_or_else(|| format!("tensor {} not found", id))
    }

    pub fn dtype(&self, id: u64) -> Result<DType, String> {
        self.tensors.get(&id)
            .map(|t| t.dtype)
            .ok_or_else(|| format!("tensor {} not found", id))
    }

    pub fn data_ptr(&self, id: u64) -> Option<*mut u8> {
        self.tensors.get(&id).map(|t| {
            if let Some(base_id) = t.base_id {
                // View: return base buffer + byte offset
                if let Some(base) = self.tensors.get(&base_id) {
                    unsafe { base.buf.as_ptr().add(t.offset) as *mut u8 }
                } else {
                    t.buf.as_ptr() as *mut u8
                }
            } else {
                t.buf.as_ptr() as *mut u8
            }
        })
    }

    // ── Register shape ────────────────────────────────────────────

    pub fn register_shape(&mut self, id: u64, shape: &[usize]) -> Result<(), String> {
        if let Some(t) = self.tensors.get_mut(&id) {
            t.strides = contiguous_strides(shape);
            t.shape = shape.to_vec();
            Ok(())
        } else {
            Err(format!("tensor {} not found", id))
        }
    }

    // ── Mark materialized ─────────────────────────────────────────

    /// Mark a tensor as materialized (its local buffer has valid data).
    /// Called after PyTorch copies data into the buffer (e.g., copy_ from CPU).
    pub fn mark_materialized(&mut self, id: u64) {
        if let Some(t) = self.tensors.get_mut(&id) {
            t.materialized = true;
        }
    }

    // ── Binary ops ────────────────────────────────────────────────

    pub fn binary_op(
        &mut self, op: wire::WireOpKind, a_id: u64, b_id: u64,
    ) -> Result<(u64, *mut u8), String> {
        let out_shape = self.infer_binary_shape(a_id, b_id)?;
        let out_dtype = self.tensors.get(&a_id)
            .ok_or_else(|| format!("tensor {} not found", a_id))?.dtype;
        let (id, ptr) = self.alloc(&out_shape, out_dtype)?;
        self.pending_ops.push(PendingOp {
            output_id: id,
            op,
            inputs: vec![a_id, b_id],
            output_shape: out_shape,
            output_dtype: out_dtype.to_wire(),
        });
        Ok((id, ptr))
    }

    fn infer_binary_shape(&self, a_id: u64, b_id: u64) -> Result<Vec<usize>, String> {
        let a = self.tensors.get(&a_id).ok_or("tensor a not found")?;
        let b = self.tensors.get(&b_id).ok_or("tensor b not found")?;
        broadcast_shapes(&a.shape, &b.shape)
    }

    // ── Unary ops ─────────────────────────────────────────────────

    pub fn unary_op(
        &mut self, op: wire::WireOpKind, input_id: u64,
    ) -> Result<(u64, *mut u8), String> {
        let t = self.tensors.get(&input_id)
            .ok_or_else(|| format!("tensor {} not found", input_id))?;
        let out_shape = t.shape.clone();
        let out_dtype = t.dtype;
        let (id, ptr) = self.alloc(&out_shape, out_dtype)?;
        self.pending_ops.push(PendingOp {
            output_id: id,
            op,
            inputs: vec![input_id],
            output_shape: out_shape,
            output_dtype: out_dtype.to_wire(),
        });
        Ok((id, ptr))
    }

    // ── Matmul ────────────────────────────────────────────────────

    pub fn matmul(&mut self, a_id: u64, b_id: u64) -> Result<(u64, *mut u8), String> {
        let a = self.tensors.get(&a_id).ok_or("tensor a not found")?;
        let b = self.tensors.get(&b_id).ok_or("tensor b not found")?;
        let out_shape = matmul_shape(&a.shape, &b.shape)?;
        let out_dtype = a.dtype;
        let (id, ptr) = self.alloc(&out_shape, out_dtype)?;
        self.pending_ops.push(PendingOp {
            output_id: id,
            op: wire::WireOpKind::Matmul,
            inputs: vec![a_id, b_id],
            output_shape: out_shape,
            output_dtype: out_dtype.to_wire(),
        });
        Ok((id, ptr))
    }

    // ── Scalar mul ────────────────────────────────────────────────

    pub fn scalar_mul(
        &mut self, input_id: u64, scale: f32,
    ) -> Result<(u64, *mut u8), String> {
        let t = self.tensors.get(&input_id)
            .ok_or_else(|| format!("tensor {} not found", input_id))?;
        let out_shape = t.shape.clone();
        let out_dtype = t.dtype;
        let (id, ptr) = self.alloc(&out_shape, out_dtype)?;
        self.pending_ops.push(PendingOp {
            output_id: id,
            op: wire::WireOpKind::ScalarMul(scale),
            inputs: vec![input_id],
            output_shape: out_shape,
            output_dtype: out_dtype.to_wire(),
        });
        Ok((id, ptr))
    }

    // ── Views ─────────────────────────────────────────────────────

    pub fn create_view(
        &mut self, base_id: u64, shape: &[usize], strides: &[usize], offset_elements: usize,
    ) -> Result<(u64, *mut u8), String> {
        // Extract info from base before mutable borrow
        let (actual_base_id, dtype, byte_offset, materialized) = {
            let base = self.tensors.get(&base_id)
                .ok_or_else(|| format!("tensor {} not found", base_id))?;
            let actual = base.base_id.unwrap_or(base_id);
            let bo = offset_elements * base.dtype.size_bytes();
            (actual, base.dtype, bo, base.materialized)
        };

        let id = self.next_id();
        let buf: Pin<Box<[u8]>> = Pin::new(vec![0u8; 0].into_boxed_slice());
        self.tensors.insert(id, RemoteTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
            strides: strides.to_vec(),
            offset: byte_offset,
            materialized,
            base_id: Some(actual_base_id),
        });

        let ptr = if let Some(base_t) = self.tensors.get(&actual_base_id) {
            unsafe { base_t.buf.as_ptr().add(byte_offset) as *mut u8 }
        } else {
            std::ptr::null_mut()
        };
        Ok((id, ptr))
    }

    // ── In-place ops ──────────────────────────────────────────────

    pub fn add_scaled_inplace(
        &mut self, self_id: u64, other_id: u64, alpha: f32,
    ) -> Result<(), String> {
        // Record as a pending op; the server will apply it in-place.
        // We create a new output that replaces self's buffer on the server.
        self.pending_ops.push(PendingOp {
            output_id: self_id, // in-place: output replaces input
            op: wire::WireOpKind::ScalarMul(alpha), // will need AddScaledInplace
            inputs: vec![self_id, other_id],
            output_shape: self.tensors.get(&self_id)
                .map(|t| t.shape.clone())
                .unwrap_or_default(),
            output_dtype: self.tensors.get(&self_id)
                .map(|t| t.dtype.to_wire())
                .unwrap_or(0),
        });
        Ok(())
    }

    // ── Flush and sync ────────────────────────────────────────────

    /// Flush all pending ops to gpu-service.
    /// Serializes ops as an EvalRequest, sends via socket, copies results back.
    pub fn flush_and_wait(&mut self) -> Result<(), String> {
        if self.pending_ops.is_empty() {
            // Release deferred frees
            let frees: Vec<u64> = self.pending_frees.drain(..).collect();
            for id in frees {
                self.free(id);
            }
            return Ok(());
        }

        self.connect()?;
        // Take client out to avoid borrow conflict with self.tensors
        let mut client = self.client.take()
            .ok_or("not connected to gpu-service")?;

        // Collect output IDs (tensors that will be computed server-side)
        let output_ids: std::collections::HashSet<u64> = self.pending_ops.iter()
            .map(|op| op.output_id)
            .collect();

        // Collect all input tensors that need uploading:
        // Referenced by ops, NOT an output of a pending op, NOT already on server.
        // We upload the local proxy buffer contents — PyTorch writes via copy_
        // directly to these buffers, so they always have valid data.
        let mut needed_inputs: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for op in &self.pending_ops {
            for &input_id in &op.inputs {
                if !self.server_known.contains(&input_id) && !output_ids.contains(&input_id) {
                    needed_inputs.insert(input_id);
                }
            }
        }

        let mut wire_tensors: Vec<WireTensorData> = Vec::new();
        for &id in &needed_inputs {
            if let Some(t) = self.tensors.get(&id) {
                if let Some(base_id) = t.base_id {
                    // View tensor: do a stride-aware copy into a contiguous buffer
                    if let Some(base_t) = self.tensors.get(&base_id) {
                        let data = stride_aware_copy(
                            &base_t.buf, t.offset, &t.shape, &t.strides, t.dtype.size_bytes(),
                        );
                        wire_tensors.push(WireTensorData {
                            id,
                            shape: t.shape.clone(),
                            dtype: t.dtype.to_wire(),
                            data,
                        });
                        self.server_known.insert(id);
                    }
                } else {
                    // Base tensor: upload directly
                    let data = &t.buf[..t.byte_size()];
                    wire_tensors.push(WireTensorData {
                        id,
                        shape: t.shape.clone(),
                        dtype: t.dtype.to_wire(),
                        data: data.to_vec(),
                    });
                    self.server_known.insert(id);
                }
            }
        }

        // Build wire op nodes
        let mut wire_nodes: Vec<WireOpNode> = Vec::new();
        for op in &self.pending_ops {
            wire_nodes.push(WireOpNode {
                id: op.output_id,
                op: op.op.clone(),
                inputs: op.inputs.clone(),
                out_shape: op.output_shape.clone(),
                out_dtype: op.output_dtype,
            });
        }

        // Collect all output IDs that need results copied back
        let all_output_ids: Vec<u64> = self.pending_ops.iter()
            .map(|op| op.output_id)
            .collect();

        // Evaluate: send graph with last output as target (forces full graph eval)
        let last_output_id = *all_output_ids.last().unwrap_or(&0);

        let req = EvalRequest {
            target_id: last_output_id,
            tensors: wire_tensors,
            nodes: wire_nodes,
        };

        let eval_result = client.eval(&req);

        // Read back results based on eval outcome
        match eval_result {
            Ok(EvalResponse::Ok { tensor_id, shape, data, .. }) => {
                self.copy_result_to_proxy(tensor_id, &shape, &data);
                self.server_known.insert(tensor_id);

                // Read back ALL other output tensors from the server
                for &oid in &all_output_ids {
                    if oid == last_output_id { continue; }
                    if !self.tensors.contains_key(&oid) { continue; }
                    match client.read_tensor(oid) {
                        Ok((shape, _dtype, data)) => {
                            self.copy_result_to_proxy(oid, &shape, &data);
                            self.server_known.insert(oid);
                        }
                        Err(_) => {} // intermediate may have been freed
                    }
                }
            }
            Ok(EvalResponse::Err(message)) => {
                self.client = Some(client);
                return Err(format!("gpu-service eval error: {}", message));
            }
            Err(e) => {
                self.client = Some(client);
                return Err(format!("gpu-service eval failed: {}", e));
            }
        }

        // Put client back
        self.client = Some(client);

        // Clear pending ops
        self.pending_ops.clear();

        // Release deferred frees
        let frees: Vec<u64> = self.pending_frees.drain(..).collect();
        for id in frees {
            self.free(id);
        }

        Ok(())
    }

    /// Copy server result data into a tensor's local proxy buffer.
    fn copy_result_to_proxy(&mut self, tensor_id: u64, shape: &[usize], data: &[u8]) {
        if let Some(t) = self.tensors.get_mut(&tensor_id) {
            let copy_len = data.len().min(t.buf.len());
            if copy_len > 0 {
                unsafe {
                    let dst = t.buf.as_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, copy_len);
                }
            }
            t.materialized = true;
            t.shape = shape.to_vec();
        }
    }

    // ── Deferred free ─────────────────────────────────────────────

    pub fn defer_free(&mut self, id: u64) {
        self.pending_frees.push(id);
    }

    // ── Find by data pointer ──────────────────────────────────────

    pub fn find_by_data_ptr(&self, ptr: *const u8) -> u64 {
        // Search base tensors first (offset=0)
        for (&id, t) in &self.tensors {
            if t.base_id.is_none() && t.buf.as_ptr() == ptr {
                return id;
            }
        }
        // Search views
        for (&id, t) in &self.tensors {
            if let Some(base_id) = t.base_id {
                if let Some(base) = self.tensors.get(&base_id) {
                    let view_ptr = unsafe { base.buf.as_ptr().add(t.offset) };
                    if view_ptr == ptr {
                        return id;
                    }
                }
            }
        }
        0
    }

    // These are used when eager_ffi.rs dispatches to RemoteEagerRuntime (Phase 2).
    #[allow(dead_code)]
    pub(crate) fn tensors_mut(&mut self) -> &mut HashMap<u64, RemoteTensor> {
        &mut self.tensors
    }

    #[allow(dead_code)]
    pub(crate) fn tensors_iter(&self) -> impl Iterator<Item = (&u64, &RemoteTensor)> + '_ {
        self.tensors.iter()
    }
}

// ── Shape helpers ─────────────────────────────────────────────────

/// Copy data from a strided view into a contiguous buffer.
/// Handles transposed and sliced views by iterating with strides.
fn stride_aware_copy(
    base_buf: &[u8], byte_offset: usize,
    shape: &[usize], strides: &[usize], elem_size: usize,
) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    let mut out = vec![0u8; numel * elem_size];
    if shape.is_empty() || numel == 0 { return out; }

    // N-D stride iteration
    let ndim = shape.len();
    let mut indices = vec![0usize; ndim];
    for flat_idx in 0..numel {
        // Compute source offset from strides
        let mut src_offset = byte_offset;
        for d in 0..ndim {
            src_offset += indices[d] * strides[d] * elem_size;
        }
        let dst_offset = flat_idx * elem_size;
        if src_offset + elem_size <= base_buf.len() && dst_offset + elem_size <= out.len() {
            out[dst_offset..dst_offset + elem_size]
                .copy_from_slice(&base_buf[src_offset..src_offset + elem_size]);
        }

        // Increment N-D indices (last dim fastest)
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] { break; }
            indices[d] = 0;
        }
    }
    out
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let max_ndim = a.len().max(b.len());
    let mut result = vec![0usize; max_ndim];
    for i in 0..max_ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da == db {
            result[max_ndim - 1 - i] = da;
        } else if da == 1 {
            result[max_ndim - 1 - i] = db;
        } else if db == 1 {
            result[max_ndim - 1 - i] = da;
        } else {
            return Err(format!("cannot broadcast shapes {:?} and {:?}", a, b));
        }
    }
    Ok(result)
}

fn matmul_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    match (a.len(), b.len()) {
        (2, 2) => {
            if a[1] != b[0] {
                return Err(format!("matmul shape mismatch: {:?} @ {:?}", a, b));
            }
            Ok(vec![a[0], b[1]])
        }
        (3, 3) => {
            // Batched matmul
            if a[0] != b[0] || a[2] != b[1] {
                return Err(format!("batched matmul shape mismatch: {:?} @ {:?}", a, b));
            }
            Ok(vec![a[0], a[1], b[2]])
        }
        _ => Err(format!("unsupported matmul shapes: {:?} @ {:?}", a, b)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_shape() {
        let mut rt = RemoteEagerRuntime::new();
        let (id, ptr) = rt.alloc(&[3, 4], DType::Float32).unwrap();
        assert_ne!(ptr, std::ptr::null_mut());
        assert_eq!(rt.shape(id).unwrap(), &[3, 4]);
        assert_eq!(rt.dtype(id).unwrap(), DType::Float32);
    }

    #[test]
    fn test_pointer_stability() {
        let mut rt = RemoteEagerRuntime::new();
        let (id1, ptr1) = rt.alloc(&[100], DType::Float32).unwrap();
        let (id2, ptr2) = rt.alloc(&[200], DType::Float32).unwrap();
        // Pointers should be different
        assert_ne!(ptr1, ptr2);
        // Pointers should remain stable after more allocations
        let (_, _) = rt.alloc(&[300], DType::Float32).unwrap();
        assert_eq!(rt.data_ptr(id1).unwrap(), ptr1);
        assert_eq!(rt.data_ptr(id2).unwrap(), ptr2);
    }

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&[3, 4], &[4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[1, 4], &[3, 1]).unwrap(), vec![3, 4]);
        assert!(broadcast_shapes(&[3, 4], &[5]).is_err());
    }

    #[test]
    fn test_matmul_shape() {
        assert_eq!(matmul_shape(&[3, 4], &[4, 5]).unwrap(), vec![3, 5]);
        assert_eq!(matmul_shape(&[2, 3, 4], &[2, 4, 5]).unwrap(), vec![2, 3, 5]);
        assert!(matmul_shape(&[3, 4], &[5, 6]).is_err());
    }

    #[test]
    fn test_binary_op_records_pending() {
        let mut rt = RemoteEagerRuntime::new();
        let (a, _) = rt.alloc(&[3, 4], DType::Float32).unwrap();
        let (b, _) = rt.alloc(&[3, 4], DType::Float32).unwrap();
        let (c, _) = rt.binary_op(wire::WireOpKind::Add, a, b).unwrap();
        assert_eq!(rt.pending_ops.len(), 1);
        assert_eq!(rt.shape(c).unwrap(), &[3, 4]);
    }

    #[test]
    fn test_find_by_data_ptr() {
        let mut rt = RemoteEagerRuntime::new();
        let (id, ptr) = rt.alloc(&[4], DType::Float32).unwrap();
        assert_eq!(rt.find_by_data_ptr(ptr), id);
        assert_eq!(rt.find_by_data_ptr(std::ptr::null()), 0);
    }
}
