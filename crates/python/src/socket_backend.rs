//! Linux SocketBackend — routes tensor ops over the wire protocol to gpu-service.
//!
//! Runs INSIDE containers. Mirrors the lazy evaluation model:
//! 1. `tensor_from_data()` stores data locally, assigns a local ID
//! 2. Binary/unary ops record pending nodes in a local graph
//! 3. `eval(id)` serializes the subgraph + input data, sends to gpu-service
//! 4. `read_bytes(id)` returns cached local data or fetches from gpu-service

#![cfg(target_os = "linux")]

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use applegpu_client::GpuClient;
use applegpu_wire::{
    EvalRequest, EvalResponse, WireDType, WireOpKind, WireOpNode, WireTensorData,
};

use crate::backend::{Backend, BackendResult};

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A tensor stored locally on the client side.
struct LocalTensor {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: WireDType,
    /// True if data bytes are valid (either created locally or fetched from server).
    materialized: bool,
}

/// A pending graph node — an op waiting to be sent to the gpu-service for evaluation.
struct PendingNode {
    id: u64,
    op: WireOpKind,
    inputs: Vec<u64>,
    out_shape: Vec<usize>,
    out_dtype: WireDType,
}

// ---------------------------------------------------------------------------
// SocketBackend
// ---------------------------------------------------------------------------

pub struct SocketBackend {
    /// GpuClient connection to the gpu-service (lazily initialized).
    client: Mutex<Option<GpuClient>>,
    /// Local tensor storage: id -> (bytes, shape, dtype, materialized).
    tensors: Mutex<HashMap<u64, LocalTensor>>,
    /// Pending graph nodes (ops recorded but not yet evaluated).
    graph: Mutex<HashMap<u64, PendingNode>>,
    /// Monotonically increasing ID counter.
    next_id: AtomicU64,
}

impl SocketBackend {
    pub fn new() -> Self {
        SocketBackend {
            client: Mutex::new(None),
            tensors: Mutex::new(HashMap::new()),
            graph: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    fn alloc_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Ensure the client is connected. Returns an error if not initialized.
    fn ensure_client<'a>(
        guard: &'a mut Option<GpuClient>,
    ) -> BackendResult<&'a mut GpuClient> {
        guard
            .as_mut()
            .ok_or_else(|| "SocketBackend not initialized — call init_backend() first".to_string())
    }

    /// Look up shape for a tensor (either materialized or pending).
    fn lookup_shape(
        tensors: &HashMap<u64, LocalTensor>,
        graph: &HashMap<u64, PendingNode>,
        id: u64,
    ) -> BackendResult<Vec<usize>> {
        if let Some(t) = tensors.get(&id) {
            return Ok(t.shape.clone());
        }
        if let Some(node) = graph.get(&id) {
            return Ok(node.out_shape.clone());
        }
        Err(format!("Tensor {} not found", id))
    }

    /// Look up dtype for a tensor (either materialized or pending).
    fn lookup_dtype(
        tensors: &HashMap<u64, LocalTensor>,
        graph: &HashMap<u64, PendingNode>,
        id: u64,
    ) -> BackendResult<WireDType> {
        if let Some(t) = tensors.get(&id) {
            return Ok(t.dtype);
        }
        if let Some(node) = graph.get(&id) {
            return Ok(node.out_dtype);
        }
        Err(format!("Tensor {} not found", id))
    }

    // -----------------------------------------------------------------------
    // Op helpers
    // -----------------------------------------------------------------------

    /// Record a unary op: same shape and dtype as input.
    fn record_unary(&self, a: u64, op: WireOpKind) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, a)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(id, PendingNode {
            id,
            op,
            inputs: vec![a],
            out_shape: shape,
            out_dtype: dtype,
        });
        Ok(id)
    }

    /// Record a binary op with broadcast shape inference.
    fn record_binary(&self, a: u64, b: u64, op: WireOpKind) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let sa = Self::lookup_shape(&tensors, &graph, a)?;
        let sb = Self::lookup_shape(&tensors, &graph, b)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let out_shape = applegpu_wire::infer_broadcast_shape(&sa, &sb)
            .map_err(|e| format!("Shape error in binary op: {}", e))?;

        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(id, PendingNode {
            id,
            op,
            inputs: vec![a, b],
            out_shape,
            out_dtype: dtype,
        });
        Ok(id)
    }

    /// Topological sort: collect all pending nodes required to evaluate `target_id`.
    /// Returns (nodes_in_topo_order, set_of_materialized_tensor_ids_needed).
    fn collect_subgraph(
        graph: &HashMap<u64, PendingNode>,
        tensors: &HashMap<u64, LocalTensor>,
        target_id: u64,
    ) -> BackendResult<(Vec<u64>, HashSet<u64>)> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut input_tensors = HashSet::new();

        // Iterative DFS (post-order)
        let mut visit_stack: Vec<(u64, bool)> = vec![(target_id, false)];
        while let Some((id, processed)) = visit_stack.pop() {
            if processed {
                if !visited.contains(&id) && graph.contains_key(&id) {
                    visited.insert(id);
                    order.push(id);
                }
                continue;
            }
            if visited.contains(&id) {
                continue;
            }
            if tensors.contains_key(&id) {
                input_tensors.insert(id);
                continue;
            }
            let node = graph.get(&id).ok_or_else(|| {
                format!("Tensor {} not found in graph or local storage", id)
            })?;
            visit_stack.push((id, true));
            for &inp in node.inputs.iter().rev() {
                if !visited.contains(&inp) {
                    visit_stack.push((inp, false));
                }
            }
        }

        Ok((order, input_tensors))
    }
}

// ---------------------------------------------------------------------------
// Backend trait implementation
// ---------------------------------------------------------------------------

impl Backend for SocketBackend {
    fn init(&self) -> BackendResult<HashMap<String, String>> {
        let requested_memory: u64 = std::env::var("APPLEGPU_MEMORY_MB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024)
            * 1024
            * 1024;

        let client = GpuClient::connect_auto(requested_memory)
            .map_err(|e| format!("Failed to connect to gpu-service: {}", e))?;

        let mut info = HashMap::new();
        info.insert("backend".to_string(), "socket".to_string());
        info.insert(
            "container_id".to_string(),
            client.container_id.to_string(),
        );
        info.insert(
            "granted_memory".to_string(),
            client.granted_memory.to_string(),
        );

        *self.client.lock().unwrap() = Some(client);
        Ok(info)
    }

    fn device_name(&self) -> BackendResult<String> {
        Ok("gpu-service (remote)".to_string())
    }

    fn tensor_from_data(
        &self,
        data: &[u8],
        shape: Vec<usize>,
        dtype: WireDType,
    ) -> BackendResult<u64> {
        let expected = shape.iter().product::<usize>() * dtype.size_bytes();
        if data.len() != expected {
            return Err(format!(
                "Data length {} does not match shape {:?} * dtype {} = {}",
                data.len(),
                shape,
                dtype.name(),
                expected
            ));
        }
        let id = self.alloc_id();
        self.tensors.lock().unwrap().insert(
            id,
            LocalTensor {
                data: data.to_vec(),
                shape,
                dtype,
                materialized: true,
            },
        );
        Ok(id)
    }

    fn insert_tensor_from_raw(
        &self,
        data: &[u8],
        shape: Vec<usize>,
        dtype: WireDType,
    ) -> BackendResult<u64> {
        self.tensor_from_data(data, shape, dtype)
    }

    fn destroy(&self, id: u64) -> BackendResult<()> {
        let mut tensors = self.tensors.lock().unwrap();
        let mut graph = self.graph.lock().unwrap();
        tensors.remove(&id);
        graph.remove(&id);
        Ok(())
    }

    fn try_destroy(&self, id: u64) {
        if let Ok(mut tensors) = self.tensors.try_lock() {
            tensors.remove(&id);
        }
        if let Ok(mut graph) = self.graph.try_lock() {
            graph.remove(&id);
        }
    }

    fn shape(&self, id: u64) -> BackendResult<Vec<usize>> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        Self::lookup_shape(&tensors, &graph, id)
    }

    fn dtype(&self, id: u64) -> BackendResult<WireDType> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        Self::lookup_dtype(&tensors, &graph, id)
    }

    fn read_bytes(&self, id: u64) -> BackendResult<Vec<u8>> {
        // Auto-eval if pending
        {
            let graph = self.graph.lock().unwrap();
            if graph.contains_key(&id) {
                drop(graph);
                self.eval(id)?;
            }
        }

        let tensors = self.tensors.lock().unwrap();
        if let Some(t) = tensors.get(&id) {
            if t.materialized {
                return Ok(t.data.clone());
            }
        }
        drop(tensors);

        // Try fetching from server
        let mut client_guard = self.client.lock().unwrap();
        let client = Self::ensure_client(&mut client_guard)?;
        let (shape, dtype_disc, data) = client
            .read_tensor(id)
            .map_err(|e| format!("Failed to read tensor {}: {}", id, e))?;

        let dtype = WireDType::from_discriminant(dtype_disc)
            .ok_or_else(|| format!("Unknown dtype discriminant {}", dtype_disc))?;

        // Cache locally
        self.tensors.lock().unwrap().insert(
            id,
            LocalTensor {
                data: data.clone(),
                shape,
                dtype,
                materialized: true,
            },
        );

        Ok(data)
    }

    fn eval(&self, id: u64) -> BackendResult<()> {
        // If already materialized, nothing to do
        {
            let tensors = self.tensors.lock().unwrap();
            if tensors.get(&id).map_or(false, |t| t.materialized) {
                return Ok(());
            }
        }

        // Collect the subgraph needed
        let (node_order, input_tensor_ids) = {
            let tensors = self.tensors.lock().unwrap();
            let graph = self.graph.lock().unwrap();
            if !graph.contains_key(&id) {
                // Not pending — might be a server-side tensor we haven't fetched yet
                return Ok(());
            }
            Self::collect_subgraph(&graph, &tensors, id)?
        };

        // Build wire tensors from materialized inputs
        let wire_tensors: Vec<WireTensorData> = {
            let tensors = self.tensors.lock().unwrap();
            input_tensor_ids
                .iter()
                .filter_map(|&tid| {
                    tensors.get(&tid).map(|t| WireTensorData {
                        id: tid,
                        shape: t.shape.clone(),
                        dtype: t.dtype.discriminant(),
                        data: t.data.clone(),
                    })
                })
                .collect()
        };

        // Build wire nodes in topological order
        let wire_nodes: Vec<WireOpNode> = {
            let graph = self.graph.lock().unwrap();
            node_order
                .iter()
                .filter_map(|&nid| {
                    graph.get(&nid).map(|n| WireOpNode {
                        id: n.id,
                        op: n.op.clone(),
                        inputs: n.inputs.clone(),
                        out_shape: n.out_shape.clone(),
                        out_dtype: n.out_dtype.discriminant(),
                    })
                })
                .collect()
        };

        let request = EvalRequest {
            target_id: id,
            tensors: wire_tensors,
            nodes: wire_nodes,
        };

        // Send to gpu-service
        let response = {
            let mut client_guard = self.client.lock().unwrap();
            let client = Self::ensure_client(&mut client_guard)?;
            client
                .eval(&request)
                .map_err(|e| format!("Eval RPC failed: {}", e))?
        };

        match response {
            EvalResponse::Ok {
                tensor_id,
                shape,
                data,
            } => {
                // Look up the expected dtype from the graph
                let dtype = {
                    let graph = self.graph.lock().unwrap();
                    graph
                        .get(&id)
                        .map(|n| n.out_dtype)
                        .unwrap_or(WireDType::Float32)
                };

                // Store result locally
                self.tensors.lock().unwrap().insert(
                    tensor_id,
                    LocalTensor {
                        data,
                        shape,
                        dtype,
                        materialized: true,
                    },
                );

                // Clean up evaluated nodes from graph
                {
                    let mut graph = self.graph.lock().unwrap();
                    for nid in &node_order {
                        graph.remove(nid);
                    }
                }

                Ok(())
            }
            EvalResponse::Err(msg) => Err(format!("gpu-service eval error: {}", msg)),
        }
    }

    fn is_materialized(&self, id: u64) -> bool {
        let tensors = self.tensors.lock().unwrap();
        tensors.get(&id).map_or(false, |t| t.materialized)
    }

    fn is_pending(&self, id: u64) -> bool {
        let graph = self.graph.lock().unwrap();
        graph.contains_key(&id)
    }

    // -----------------------------------------------------------------------
    // Binary ops
    // -----------------------------------------------------------------------

    fn add(&self, a: u64, b: u64) -> BackendResult<u64> {
        self.record_binary(a, b, WireOpKind::Add)
    }
    fn sub(&self, a: u64, b: u64) -> BackendResult<u64> {
        self.record_binary(a, b, WireOpKind::Sub)
    }
    fn mul(&self, a: u64, b: u64) -> BackendResult<u64> {
        self.record_binary(a, b, WireOpKind::Mul)
    }
    fn div(&self, a: u64, b: u64) -> BackendResult<u64> {
        self.record_binary(a, b, WireOpKind::Div)
    }

    fn matmul(&self, a: u64, b: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let sa = Self::lookup_shape(&tensors, &graph, a)?;
        let sb = Self::lookup_shape(&tensors, &graph, b)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let out_shape = applegpu_wire::infer_matmul_shape(&sa, &sb)
            .map_err(|e| format!("Matmul shape error: {}", e))?;

        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Matmul,
                inputs: vec![a, b],
                out_shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // Unary ops
    // -----------------------------------------------------------------------

    fn neg(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Neg)
    }
    fn relu(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Relu)
    }
    fn gelu(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Gelu)
    }
    fn exp(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Exp)
    }
    fn log(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Log)
    }
    fn sqrt(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Sqrt)
    }
    fn abs(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Abs)
    }
    fn sign(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Sign)
    }
    fn tanh(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Tanh)
    }

    // -----------------------------------------------------------------------
    // Parameterized ops
    // -----------------------------------------------------------------------

    fn scalar_mul(&self, a: u64, scale: f32) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::ScalarMul(scale))
    }

    fn pow(&self, a: u64, exponent: f32) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Pow { exponent })
    }

    fn clamp(&self, a: u64, min: f32, max: f32) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Clamp {
            min_val: min,
            max_val: max,
        })
    }

    // -----------------------------------------------------------------------
    // Reduction ops
    // -----------------------------------------------------------------------

    fn softmax(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Softmax)
    }
    fn softmax_causal(&self, a: u64) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::SoftmaxCausal)
    }

    fn argmax(&self, a: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let out_shape = applegpu_wire::infer_argmax_shape(&shape);
        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Argmax,
                inputs: vec![a],
                out_shape,
                out_dtype: WireDType::Int64,
            },
        );
        Ok(id)
    }

    fn sum(&self, a: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, a)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let out_shape = applegpu_wire::infer_reduce_shape(&shape);
        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Sum,
                inputs: vec![a],
                out_shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn mean(&self, a: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, a)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let out_shape = applegpu_wire::infer_reduce_shape(&shape);
        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Mean,
                inputs: vec![a],
                out_shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // Shape ops
    // -----------------------------------------------------------------------

    fn reshape(&self, a: u64, new_shape: Vec<usize>) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let dtype = Self::lookup_dtype(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let id = self.alloc_id();
        let out_shape = new_shape.clone();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Reshape {
                    new_shape,
                },
                inputs: vec![a],
                out_shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn transpose(&self, a: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, a)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let out_shape = applegpu_wire::infer_transpose_shape(&shape)
            .map_err(|e| format!("Transpose shape error: {}", e))?;
        let n = shape.len();
        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Transpose {
                    dim0: n - 2,
                    dim1: n - 1,
                },
                inputs: vec![a],
                out_shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn transpose_dims(&self, a: u64, dim0: usize, dim1: usize) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, a)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let out_shape = applegpu_wire::infer_transpose_dims_shape(&shape, dim0, dim1)
            .map_err(|e| format!("Transpose dims error: {}", e))?;
        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Transpose { dim0, dim1 },
                inputs: vec![a],
                out_shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn slice(&self, a: u64, dim: usize, start: usize, end: usize) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, a)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let out_shape = applegpu_wire::infer_slice_shape(&shape, dim, start, end)
            .map_err(|e| format!("Slice shape error: {}", e))?;
        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Slice { dim, start, end },
                inputs: vec![a],
                out_shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn concat(&self, a: u64, b: u64, dim: usize) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let sa = Self::lookup_shape(&tensors, &graph, a)?;
        let sb = Self::lookup_shape(&tensors, &graph, b)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, a)?;
        drop(tensors);
        drop(graph);

        let out_shape = applegpu_wire::infer_concat_shape(&sa, &sb, dim)
            .map_err(|e| format!("Concat shape error: {}", e))?;
        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Concat { dim },
                inputs: vec![a, b],
                out_shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // Conditional ops
    // -----------------------------------------------------------------------

    fn where_cond(&self, cond: u64, x: u64, y: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, x)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, x)?;
        drop(tensors);
        drop(graph);

        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Where,
                inputs: vec![cond, x, y],
                out_shape: shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn masked_fill(&self, input: u64, mask: u64, value: f32) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, input)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, input)?;
        drop(tensors);
        drop(graph);

        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::MaskedFill { value },
                inputs: vec![input, mask],
                out_shape: shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn triu(&self, a: u64, diagonal: i32) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Triu { diagonal })
    }
    fn tril(&self, a: u64, diagonal: i32) -> BackendResult<u64> {
        self.record_unary(a, WireOpKind::Tril { diagonal })
    }

    // -----------------------------------------------------------------------
    // Indexing
    // -----------------------------------------------------------------------

    // Bitwise ops
    fn bitwise_and(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("Bitwise ops not supported over socket".to_string()) }
    fn bitwise_or(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("Bitwise ops not supported over socket".to_string()) }
    fn bitwise_xor(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("Bitwise ops not supported over socket".to_string()) }
    fn bitwise_not(&self, _a: u64) -> BackendResult<u64> { Err("Bitwise ops not supported over socket".to_string()) }
    fn shl(&self, _a: u64, _shift: u32) -> BackendResult<u64> { Err("Shift ops not supported over socket".to_string()) }
    fn shr(&self, _a: u64, _shift: u32) -> BackendResult<u64> { Err("Shift ops not supported over socket".to_string()) }

    // Modulo
    fn mod_op(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("Mod op not supported over socket".to_string()) }

    // Element-wise min/max
    fn elem_min(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("ElemMin not supported over socket".to_string()) }
    fn elem_max(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("ElemMax not supported over socket".to_string()) }

    // Logical NOT
    fn logical_not(&self, _a: u64) -> BackendResult<u64> { Err("LogicalNot not supported over socket".to_string()) }

    // Comparison ops
    fn lt(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("Comparison ops not supported over socket".to_string()) }
    fn gt(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("Comparison ops not supported over socket".to_string()) }
    fn le(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("Comparison ops not supported over socket".to_string()) }
    fn ge(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("Comparison ops not supported over socket".to_string()) }
    fn eq_op(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("Comparison ops not supported over socket".to_string()) }
    fn ne_op(&self, _a: u64, _b: u64) -> BackendResult<u64> { Err("Comparison ops not supported over socket".to_string()) }

    fn cast(&self, _a: u64, _target_dtype: &str) -> BackendResult<u64> {
        Err("Cast not supported over socket backend".to_string())
    }

    fn quantize(&self, _a: u64, _target_dtype: &str, _scale: f32, _zero_point: i32) -> BackendResult<u64> {
        Err("Quantize not supported over socket backend".to_string())
    }

    fn dequantize(&self, _a: u64, _target_dtype: &str, _scale: f32, _zero_point: i32) -> BackendResult<u64> {
        Err("Dequantize not supported over socket backend".to_string())
    }

    fn gather(&self, input: u64, dim: usize, index: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, index)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, input)?;
        drop(tensors);
        drop(graph);

        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Gather { dim },
                inputs: vec![input, index],
                out_shape: shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn index_select(&self, input: u64, dim: usize, index: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let mut shape = Self::lookup_shape(&tensors, &graph, input)?;
        let idx_shape = Self::lookup_shape(&tensors, &graph, index)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, input)?;
        drop(tensors);
        drop(graph);

        if dim < shape.len() && !idx_shape.is_empty() {
            shape[dim] = idx_shape[0];
        }

        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::IndexSelect { dim },
                inputs: vec![input, index],
                out_shape: shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // Complex ops
    // -----------------------------------------------------------------------

    fn layer_norm(&self, input: u64, gamma: u64, beta: u64, eps: f32) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, input)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, input)?;
        drop(tensors);
        drop(graph);

        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::LayerNorm { eps },
                inputs: vec![input, gamma, beta],
                out_shape: shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn embedding(&self, weights: u64, indices: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let w_shape = Self::lookup_shape(&tensors, &graph, weights)?;
        let i_shape = Self::lookup_shape(&tensors, &graph, indices)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, weights)?;
        drop(tensors);
        drop(graph);

        // Output: indices_shape + [embed_dim]
        let embed_dim = w_shape.last().copied().unwrap_or(0);
        let mut out_shape = i_shape;
        out_shape.push(embed_dim);

        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Embedding,
                inputs: vec![weights, indices],
                out_shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn attention(&self, q: u64, k: u64, v: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, q)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, q)?;
        drop(tensors);
        drop(graph);

        // attention output has same shape as Q
        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::Softmax, // placeholder — server handles actual attention
                inputs: vec![q, k, v],
                out_shape: shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn attention_causal(&self, q: u64, k: u64, v: u64) -> BackendResult<u64> {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let shape = Self::lookup_shape(&tensors, &graph, q)?;
        let dtype = Self::lookup_dtype(&tensors, &graph, q)?;
        drop(tensors);
        drop(graph);

        let id = self.alloc_id();
        self.graph.lock().unwrap().insert(
            id,
            PendingNode {
                id,
                op: WireOpKind::SoftmaxCausal, // placeholder
                inputs: vec![q, k, v],
                out_shape: shape,
                out_dtype: dtype,
            },
        );
        Ok(id)
    }

    fn add_bias(&self, input: u64, bias: u64) -> BackendResult<u64> {
        self.record_binary(input, bias, WireOpKind::AddBias)
    }

    // -----------------------------------------------------------------------
    // CNN ops — not implemented over socket for v0.8.0
    // -----------------------------------------------------------------------

    fn conv1d(&self, _input: u64, _weight: u64, _stride: usize, _padding: usize) -> BackendResult<u64> {
        Err("conv1d not yet implemented over socket backend".to_string())
    }
    fn conv2d(&self, _input: u64, _weight: u64, _stride: (usize, usize), _padding: (usize, usize)) -> BackendResult<u64> {
        Err("conv2d not yet implemented over socket backend".to_string())
    }
    fn batch_norm(&self, _input: u64, _mean: u64, _var: u64, _weight: u64, _bias: u64, _eps: f32) -> BackendResult<u64> {
        Err("batch_norm not yet implemented over socket backend".to_string())
    }
    fn max_pool2d(&self, _input: u64, _kernel: (usize, usize), _stride: (usize, usize), _padding: (usize, usize)) -> BackendResult<u64> {
        Err("max_pool2d not yet implemented over socket backend".to_string())
    }
    fn avg_pool2d(&self, _input: u64, _kernel: (usize, usize), _stride: (usize, usize), _padding: (usize, usize)) -> BackendResult<u64> {
        Err("avg_pool2d not yet implemented over socket backend".to_string())
    }

    // -----------------------------------------------------------------------
    // Backward ops — not implemented over socket for v0.8.0
    // -----------------------------------------------------------------------

    fn softmax_backward(&self, _grad: u64, _output: u64) -> BackendResult<u64> {
        Err("softmax_backward not yet implemented over socket backend".to_string())
    }
    fn layer_norm_backward(&self, _grad: u64, _input: u64, _gamma: u64, _eps: f32) -> BackendResult<u64> {
        Err("layer_norm_backward not yet implemented over socket backend".to_string())
    }
    fn conv2d_backward_input(&self, _grad: u64, _weight: u64, _in_h: usize, _in_w: usize, _stride: (usize, usize), _padding: (usize, usize)) -> BackendResult<u64> {
        Err("conv2d_backward_input not yet implemented over socket backend".to_string())
    }
    fn embedding_backward(&self, _grad: u64, _indices: u64, _num_weights: usize) -> BackendResult<u64> {
        Err("embedding_backward not yet implemented over socket backend".to_string())
    }
    fn batch_norm_backward(&self, _grad: u64, _weight: u64, _var: u64, _eps: f32) -> BackendResult<u64> {
        Err("batch_norm_backward not yet implemented over socket backend".to_string())
    }

    // -----------------------------------------------------------------------
    // Resource management — local tracking only
    // -----------------------------------------------------------------------

    fn set_limits(&self, _max_tensor_size_mb: usize, _max_memory_mb: usize, _max_tensors: usize) {
        // Resource limits are enforced server-side; no-op on client
    }

    fn memory_usage(&self) -> usize {
        let tensors = self.tensors.lock().unwrap();
        tensors.values().map(|t| t.data.len()).sum()
    }

    fn tensor_count(&self) -> usize {
        let tensors = self.tensors.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        tensors.len() + graph.len()
    }

    fn pool_stats(&self) -> HashMap<String, usize> {
        HashMap::new() // No local pool
    }
    fn pool_drain(&self) {}
    fn set_pool_watermark(&self, _mb: usize) {}

    // -----------------------------------------------------------------------
    // Scheduler — not applicable on client side
    // -----------------------------------------------------------------------

    fn register_container(&self, _priority: &str, _max_memory_mb: usize, _max_tensors: usize, _max_pending: usize) -> BackendResult<u64> {
        Err("Scheduler not available on socket backend (client-side)".to_string())
    }
    fn deregister_container(&self, _container_id: u64) -> BackendResult<Vec<u64>> {
        Err("Scheduler not available on socket backend (client-side)".to_string())
    }
    fn pause_container(&self, _container_id: u64) -> BackendResult<()> {
        Err("Scheduler not available on socket backend (client-side)".to_string())
    }
    fn resume_container(&self, _container_id: u64) -> BackendResult<()> {
        Err("Scheduler not available on socket backend (client-side)".to_string())
    }
    fn submit_job(&self, _container_id: u64, _tensor_id: u64) -> BackendResult<u64> {
        Err("Scheduler not available on socket backend (client-side)".to_string())
    }
    fn run_next(&self) -> BackendResult<Option<u64>> {
        Err("Scheduler not available on socket backend (client-side)".to_string())
    }
    fn job_status(&self, _job_id: u64) -> BackendResult<String> {
        Err("Scheduler not available on socket backend (client-side)".to_string())
    }
    fn container_usage(&self, _container_id: u64) -> BackendResult<(usize, usize)> {
        Err("Scheduler not available on socket backend (client-side)".to_string())
    }
    fn global_usage(&self) -> (usize, usize) {
        (self.memory_usage(), self.tensor_count())
    }
    fn queue_depth(&self) -> usize {
        0
    }
}

// ===========================================================================
// Tests (run on any platform — tests the logic, not the actual socket)
// ===========================================================================

#[cfg(test)]
mod tests {
    // These tests verify the local graph-building and shape inference logic.
    // They don't require a gpu-service connection (no eval/read_bytes calls).
    //
    // On macOS the #[cfg(target_os = "linux")] gate means this module won't
    // compile. The tests for shape inference live in the wire crate instead.
    // On Linux CI, these tests exercise the full SocketBackend locally.
}
