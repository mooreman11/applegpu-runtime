# Phase 4b: Kernel Fusion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fusion optimization pass that detects chains of element-wise operations in the computation graph and replaces them with a single auto-generated Metal kernel, eliminating intermediate buffers and memory round-trips.

**Architecture:** A new `fusion` module scans the topo-sorted execution order for fusible element-wise chains. It generates MSL source code at runtime (e.g., `out[id] = relu(a[id] + b[id])` for `relu(add(a, b))`) and compiles it via the existing `KernelRegistry`. The fused chain is replaced in the graph by a single `FusedElementwise` node. A new `dispatchFused` Swift C ABI function handles variable-count input buffers for fused kernels.

**Tech Stack:** Rust (graph transformation, MSL code generation), Metal Shading Language (runtime-generated kernels)

**Fusion rules:**
- Only element-wise ops (unary + binary) are fusible. Matmul is never fused.
- A chain is fusible when each op's output is consumed by exactly one successor in the chain.
- Binary ops in a chain require one input to come from the predecessor; the other input is a "side input" passed as a buffer.
- The fused kernel reads all leaf inputs once, applies all ops inline, and writes one output.

---

## File Structure

### New Files
- `crates/core/src/fusion.rs` — Fusion pass: detect fusible chains, generate MSL, replace graph nodes
- `crates/core/tests/fusion_integration.rs` — Integration tests for fused execution

### Modified Files
- `crates/core/src/graph.rs` — Add `FusedElementwise` variant to `OpKind`, add `ref_count` helper
- `crates/core/src/lazy.rs` — Call fusion pass before execution in `eval()`
- `crates/core/src/lib.rs` — Add fusion module
- `crates/core/src/compute.rs` — Add `dispatch_fused` method for variable-buffer-count dispatch

---

## Chunk 1: Fusion Infrastructure

### Task 1: Add FusedElementwise to OpKind and ref counting to Graph

**Files:**
- Modify: `crates/core/src/graph.rs`

- [ ] **Step 1: Add FusedElementwise variant and ref counting**

Add to `OpKind` enum:

```rust
    /// A fused chain of element-wise ops. The kernel_source field contains
    /// runtime-generated MSL code. function_name is the generated kernel name.
    FusedElementwise {
        kernel_source: String,
        function_name: String,
    },
```

Change `kernel_name()` return type from `&'static str` to `&str` (lifetime tied to `&self`), and add the fused arm:

```rust
    pub fn kernel_name(&self) -> &str {
        // ... existing arms unchanged ...
            OpKind::FusedElementwise { ref function_name, .. } => function_name.as_str(),
    }
```

All call sites (`dispatch_unary`, `dispatch_binary` in `lazy.rs`) accept `&str` so this change is backward-compatible.

Update `is_unary()` to return `false` for fused (it has its own dispatch).

Update `is_matmul()` to return `false` for fused.

Add a new method:

```rust
    pub fn is_fused(&self) -> bool {
        matches!(self, OpKind::FusedElementwise { .. })
    }

    pub fn is_elementwise(&self) -> bool {
        matches!(self, OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div |
                       OpKind::Neg | OpKind::Relu | OpKind::Exp | OpKind::Log | OpKind::Sqrt)
    }
```

Add to `Graph`:

```rust
    /// Count how many nodes in the graph consume a given node's output.
    /// Returns 0 for leaf tensors not in the graph.
    pub fn ref_count(&self, id: u64) -> usize {
        self.nodes.values()
            .filter(|node| node.inputs.contains(&id))
            .count()
    }
```

- [ ] **Step 2: Run tests to verify nothing breaks**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All existing tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/graph.rs
git commit -m "feat: add FusedElementwise OpKind variant and ref_count to Graph"
```

---

### Task 2: Add fused dispatch to compute.rs

**Files:**
- Modify: `crates/core/src/compute.rs`

- [ ] **Step 1: Add dispatch_fused method to ComputePipeline**

Fused kernels can have 1-4+ input buffers, so we need a generalized dispatch that sets N buffers. We'll add a new C ABI function that accepts buffer pointers as an array.

Add to `swift/Sources/AppleGPUBridge/include/bridge.h` before `#endif`:

```c
// Execute a fused kernel with variable number of input buffers.
// input_buffers is an array of buffer_count buffer pointers.
// output is the output buffer. count is the element count.
int32_t gpu_bridge_compute_fused(
    GPUComputeHandle* compute,
    const GPUBufferHandle* const* input_buffers,
    uint32_t buffer_count,
    GPUBufferHandle* output,
    uint64_t element_count
);
```

Add to `swift/Sources/AppleGPUBridge/compute.swift`, in `GPUCompute` class:

```swift
    func dispatchFused(inputs: [MTLBuffer], output: MTLBuffer, count: Int) -> Bool {
        if count == 0 { return true }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipelineState)

        // Set input buffers at indices 0..n-1
        for (i, buf) in inputs.enumerated() {
            encoder.setBuffer(buf, offset: 0, index: i)
        }
        // Output at index n
        encoder.setBuffer(output, offset: 0, index: inputs.count)
        // Count at index n+1
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: inputs.count + 1)

        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, count)
        let threadGroups = (count + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return commandBuffer.status == .completed
    }
```

Add the C ABI export:

```swift
@_cdecl("gpu_bridge_compute_fused")
public func gpuBridgeComputeFused(
    _ computePtr: UnsafeMutableRawPointer?,
    _ inputBuffers: UnsafePointer<UnsafeRawPointer?>?,
    _ bufferCount: UInt32,
    _ outputPtr: UnsafeMutableRawPointer?,
    _ elementCount: UInt64
) -> Int32 {
    guard let computePtr = computePtr,
          let inputBuffers = inputBuffers,
          let outputPtr = outputPtr else { return -1 }

    let compute = Unmanaged<GPUCompute>.fromOpaque(computePtr).takeUnretainedValue()
    let bufOut = Unmanaged<GPUBuffer>.fromOpaque(outputPtr).takeUnretainedValue()

    var mtlBuffers: [MTLBuffer] = []
    for i in 0..<Int(bufferCount) {
        guard let bufPtr = inputBuffers[i] else { return -1 }
        let buf = Unmanaged<GPUBuffer>.fromOpaque(bufPtr).takeUnretainedValue()
        mtlBuffers.append(buf.buffer)
    }

    let success = compute.dispatchFused(
        inputs: mtlBuffers,
        output: bufOut.buffer,
        count: Int(elementCount)
    )
    return success ? 0 : -1
}
```

Add to Rust `ffi.rs`:

```rust
    pub fn gpu_bridge_compute_fused(
        compute: *mut GPUComputeHandle,
        input_buffers: *const *const GPUBufferHandle,
        buffer_count: u32,
        output: *mut GPUBufferHandle,
        element_count: u64,
    ) -> i32;
```

Add to `KernelRegistry`:

```rust
    /// Dispatch a fused kernel with variable input buffers.
    pub fn dispatch_fused(
        &self,
        device: &Device,
        kernel_source: &str,
        function_name: &str,
        input_buffers: &[&Buffer],
        buf_out: &Buffer,
        element_count: usize,
    ) -> Result<()> {
        let pipeline = self.get_or_create(device, kernel_source, function_name)?;

        let ptrs: Vec<*const ffi::GPUBufferHandle> = input_buffers
            .iter()
            .map(|b| b.raw_handle() as *const _)
            .collect();

        let result = unsafe {
            ffi::gpu_bridge_compute_fused(
                pipeline.handle,
                ptrs.as_ptr(),
                ptrs.len() as u32,
                buf_out.raw_handle(),
                element_count as u64,
            )
        };
        if result == 0 { Ok(()) } else { Err(GpuError::ComputeFailed("Fused kernel dispatch failed".to_string())) }
    }
```

Note: `pipeline.handle` is private. We need to expose it via a method on `ComputePipeline`:

```rust
    pub(crate) fn raw_handle(&self) -> *mut ffi::GPUComputeHandle {
        self.handle
    }
```

And update `dispatch_fused` on `KernelRegistry` to use `pipeline.raw_handle()`.

- [ ] **Step 2: Build and verify**

Run: `cd swift && swift build && cargo check -p applegpu-core 2>&1`
Expected: Both compile

- [ ] **Step 3: Commit**

```bash
git add swift/Sources/AppleGPUBridge/compute.swift swift/Sources/AppleGPUBridge/include/bridge.h crates/core/src/compute.rs crates/core/src/ffi.rs
git commit -m "feat: add fused dispatch with variable input buffer count"
```

---

### Task 3: Implement fusion pass with MSL code generation

**Files:**
- Create: `crates/core/src/fusion.rs`
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Create fusion.rs**

```rust
use crate::graph::{Graph, OpKind, OpNode};
use crate::tensor::{DType, Shape};

use std::sync::atomic::{AtomicU64, Ordering};

static FUSED_ID_COUNTER: AtomicU64 = AtomicU64::new(200_000);

/// MSL expression for a unary op applied to an expression string.
fn unary_msl(op: &OpKind, expr: &str) -> String {
    match op {
        OpKind::Neg => format!("(-{})", expr),
        OpKind::Relu => format!("max({}, 0.0f)", expr),
        OpKind::Exp => format!("exp({})", expr),
        OpKind::Log => format!("log({})", expr),
        OpKind::Sqrt => format!("sqrt({})", expr),
        _ => unreachable!("Not a unary op"),
    }
}

/// MSL expression for a binary op.
fn binary_msl(op: &OpKind, lhs: &str, rhs: &str) -> String {
    match op {
        OpKind::Add => format!("({} + {})", lhs, rhs),
        OpKind::Sub => format!("({} - {})", lhs, rhs),
        OpKind::Mul => format!("({} * {})", lhs, rhs),
        OpKind::Div => format!("({} / {})", lhs, rhs),
        _ => unreachable!("Not a binary elementwise op"),
    }
}

/// A fusible chain of element-wise ops.
struct FusionChain {
    /// Node IDs in the chain, in execution order.
    node_ids: Vec<u64>,
    /// IDs of leaf inputs that need buffer bindings (not intermediate results).
    leaf_inputs: Vec<u64>,
    /// Output shape.
    out_shape: Shape,
    /// Output dtype.
    out_dtype: DType,
}

/// Attempt to find fusible element-wise chains in the execution order.
/// Returns chains of length >= 2 (single ops aren't worth fusing).
fn find_fusible_chains(graph: &Graph, exec_order: &[u64]) -> Vec<FusionChain> {
    let mut chains: Vec<FusionChain> = Vec::new();
    let mut consumed: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for &node_id in exec_order {
        if consumed.contains(&node_id) {
            continue;
        }
        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };
        if !node.op.is_elementwise() {
            continue;
        }

        // Try to grow a chain starting from this node
        let mut chain_ids = vec![node_id];
        let mut current_id = node_id;

        // Walk forward: find successors that are also elementwise and only consume current
        loop {
            // Find a successor of current_id in exec_order
            let successor = exec_order.iter().find(|&&next_id| {
                if consumed.contains(&next_id) || chain_ids.contains(&next_id) {
                    return false;
                }
                if let Some(next_node) = graph.get_node(next_id) {
                    // Must be elementwise
                    if !next_node.op.is_elementwise() {
                        return false;
                    }
                    // Must consume current_id as an input
                    if !next_node.inputs.contains(&current_id) {
                        return false;
                    }
                    // current_id must have ref_count == 1 (only consumed by this successor)
                    // Otherwise the intermediate result is needed elsewhere
                    if graph.ref_count(current_id) != 1 {
                        return false;
                    }
                    true
                } else {
                    false
                }
            });

            match successor {
                Some(&next_id) => {
                    chain_ids.push(next_id);
                    current_id = next_id;
                }
                None => break,
            }
        }

        if chain_ids.len() < 2 {
            continue; // not worth fusing a single op
        }

        // Collect leaf inputs (inputs to chain nodes that aren't chain-internal)
        let mut leaf_inputs = Vec::new();
        for &cid in &chain_ids {
            let n = graph.get_node(cid).unwrap();
            for &input_id in &n.inputs {
                if !chain_ids.contains(&input_id) && !leaf_inputs.contains(&input_id) {
                    leaf_inputs.push(input_id);
                }
            }
        }

        let last_node = graph.get_node(*chain_ids.last().unwrap()).unwrap();

        for &cid in &chain_ids {
            consumed.insert(cid);
        }

        chains.push(FusionChain {
            node_ids: chain_ids,
            leaf_inputs,
            out_shape: last_node.out_shape.clone(),
            out_dtype: last_node.out_dtype,
        });
    }

    chains
}

/// Generate MSL kernel source for a fusion chain.
/// Returns (kernel_source, function_name, leaf_input_ids_in_buffer_order).
fn generate_fused_msl(graph: &Graph, chain: &FusionChain) -> (String, String, Vec<u64>) {
    let func_name = format!("fused_{}", FUSED_ID_COUNTER.fetch_add(1, Ordering::Relaxed));
    let leaf_inputs = &chain.leaf_inputs;

    // Map each leaf input to a buffer name
    // Map each chain node to an MSL expression
    let mut expr_map: std::collections::HashMap<u64, String> = std::collections::HashMap::new();

    // Leaf inputs are buffer parameters
    for (i, &leaf_id) in leaf_inputs.iter().enumerate() {
        expr_map.insert(leaf_id, format!("in{}[id]", i));
    }

    // Build expressions for each chain node in order
    for &node_id in &chain.node_ids {
        let node = graph.get_node(node_id).unwrap();
        let expr = if node.op.is_unary() {
            let input_expr = expr_map.get(&node.inputs[0]).unwrap().clone();
            unary_msl(&node.op, &input_expr)
        } else {
            // Binary op
            let lhs = expr_map.get(&node.inputs[0]).unwrap().clone();
            let rhs = expr_map.get(&node.inputs[1]).unwrap().clone();
            binary_msl(&node.op, &lhs, &rhs)
        };
        expr_map.insert(node_id, expr);
    }

    let last_id = *chain.node_ids.last().unwrap();
    let final_expr = expr_map.get(&last_id).unwrap();

    // Generate buffer parameters
    let mut params = Vec::new();
    for (i, _) in leaf_inputs.iter().enumerate() {
        params.push(format!("    device const float* in{} [[buffer({})]]", i, i));
    }
    let out_idx = leaf_inputs.len();
    params.push(format!("    device float* out [[buffer({})]]", out_idx));
    params.push(format!("    constant uint& count [[buffer({})]]", out_idx + 1));
    params.push("    uint id [[thread_position_in_grid]]".to_string());

    let kernel_source = format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {}(
{}
) {{
    if (id < count) {{
        out[id] = {};
    }}
}}"#,
        func_name,
        params.join(",\n"),
        final_expr
    );

    (kernel_source, func_name, leaf_inputs.clone())
}

/// Run the fusion optimization pass on the graph.
/// Modifies the graph in-place, replacing fusible chains with FusedElementwise nodes.
/// Returns the new execution order (with fused node IDs replacing chain node IDs).
pub fn optimize(graph: &mut Graph, exec_order: &[u64]) -> Vec<u64> {
    let chains = find_fusible_chains(graph, exec_order);

    if chains.is_empty() {
        return exec_order.to_vec();
    }

    let mut new_order = exec_order.to_vec();

    for chain in &chains {
        let (kernel_source, function_name, leaf_inputs) = generate_fused_msl(graph, chain);

        let fused_id = *chain.node_ids.last().unwrap(); // use last node's ID — this is what callers expect
        let fused_node = OpNode {
            id: fused_id,
            op: OpKind::FusedElementwise {
                kernel_source,
                function_name,
            },
            inputs: leaf_inputs,
            out_shape: chain.out_shape.clone(),
            out_dtype: chain.out_dtype,
        };

        // Remove all chain nodes from graph
        for &cid in &chain.node_ids {
            graph.remove_node(cid);
        }

        // Add the fused node
        graph.add_node(fused_node);

        // Update execution order: keep fused_id (last chain node) in place, remove all others
        let chain_set: std::collections::HashSet<u64> = chain.node_ids.iter().copied().collect();
        new_order.retain(|&id| {
            id == fused_id || !chain_set.contains(&id)
        });
    }

    new_order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_chain_add_relu() {
        // a(1) -> add(3) -> relu(4)
        //         b(2) /
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Relu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        });

        let chains = find_fusible_chains(&g, &[3, 4]);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].node_ids, vec![3, 4]);
        assert_eq!(chains[0].leaf_inputs, vec![1, 2]);
    }

    #[test]
    fn generate_msl_add_relu() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Relu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        });

        let chains = find_fusible_chains(&g, &[3, 4]);
        let (source, name, inputs) = generate_fused_msl(&g, &chains[0]);

        assert!(source.contains("max((in0[id] + in1[id]), 0.0f)"));
        assert!(source.contains(&name));
        assert_eq!(inputs, vec![1, 2]);
    }

    #[test]
    fn optimize_replaces_chain() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Relu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        });

        let new_order = optimize(&mut g, &[3, 4]);
        assert_eq!(new_order.len(), 1);
        let fused = g.get_node(new_order[0]).unwrap();
        assert!(fused.op.is_fused());
        assert_eq!(fused.inputs, vec![1, 2]);
    }

    #[test]
    fn no_fusion_for_single_op() {
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        });

        let new_order = optimize(&mut g, &[3]);
        assert_eq!(new_order, vec![3]);
        assert!(!g.get_node(3).unwrap().op.is_fused()); // not fused
    }

    #[test]
    fn no_fusion_across_matmul() {
        // add(3) -> matmul(4) -> relu(5) — matmul breaks the chain
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![2, 2]), out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Matmul, inputs: vec![3, 2],
            out_shape: Shape::new(vec![2, 2]), out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 5, op: OpKind::Relu, inputs: vec![4],
            out_shape: Shape::new(vec![2, 2]), out_dtype: DType::Float32,
        });

        let new_order = optimize(&mut g, &[3, 4, 5]);
        // No chain of length >= 2 (add is alone, matmul breaks it, relu is alone)
        assert_eq!(new_order.len(), 3);
    }

    #[test]
    fn no_fusion_when_intermediate_has_multiple_consumers() {
        // a(1) -> add(3) -> relu(4)
        //         b(2) /     \
        //                     mul(5) — add(3) has ref_count 2, can't fuse
        let mut g = Graph::new();
        g.add_node(OpNode {
            id: 3, op: OpKind::Add, inputs: vec![1, 2],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 4, op: OpKind::Relu, inputs: vec![3],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        });
        g.add_node(OpNode {
            id: 5, op: OpKind::Mul, inputs: vec![3, 1],
            out_shape: Shape::new(vec![4]), out_dtype: DType::Float32,
        });

        let chains = find_fusible_chains(&g, &[3, 4, 5]);
        assert!(chains.is_empty()); // add(3) has 2 consumers, can't fuse
    }
}
```

- [ ] **Step 2: Add fusion module to lib.rs**

Add `pub mod fusion;` to `crates/core/src/lib.rs`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p applegpu-core fusion 2>&1`
Expected: All 5 fusion tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/fusion.rs crates/core/src/lib.rs
git commit -m "feat: add fusion pass with MSL code generation and chain detection"
```

---

## Chunk 2: Wire Fusion into Lazy Evaluation

### Task 4: Integrate fusion into LazyRuntime eval and handle fused dispatch

**Files:**
- Modify: `crates/core/src/lazy.rs`

- [ ] **Step 1: Call fusion pass in eval() and handle FusedElementwise execution**

In `lazy.rs`, modify `eval()` to run the fusion pass before executing:

Replace the `eval` method:

```rust
    pub fn eval(&mut self, device: &Device, id: u64) -> Result<()> {
        if self.is_materialized(id) {
            return Ok(());
        }

        let mut order = self.graph.topo_sort(id)?;
        if order.is_empty() {
            return Err(GpuError::GraphError(format!("Tensor {} not found", id)));
        }

        // Run fusion optimization pass
        order = crate::fusion::optimize(&mut self.graph, &order);

        for node_id in order {
            if self.is_materialized(node_id) {
                continue;
            }

            let node = self.graph.remove_node(node_id).ok_or_else(|| {
                GpuError::GraphError(format!("Node {} not found in graph", node_id))
            })?;

            let result = self.execute_node(device, &node)?;
            self.tensors.insert(node_id, result);
        }

        Ok(())
    }
```

Add fused execution to `execute_node`:

```rust
    fn execute_node(&self, device: &Device, node: &OpNode) -> Result<Tensor> {
        if let OpKind::FusedElementwise { ref kernel_source, ref function_name } = node.op {
            // Collect input buffers
            let input_tensors: Vec<&Tensor> = node.inputs.iter()
                .map(|&id| self.get_tensor(id))
                .collect::<Result<Vec<_>>>()?;
            let input_buffers: Vec<&crate::buffer::Buffer> = input_tensors.iter()
                .map(|t| &t.buffer)
                .collect();
            let out = Tensor::empty_f32(device, node.out_shape.dims().to_vec())?;
            let numel = input_tensors[0].numel();
            REGISTRY.dispatch_fused(
                device,
                kernel_source,
                function_name,
                &input_buffers,
                &out.buffer,
                numel,
            )?;
            return Ok(out);
        }

        // ... existing dispatch code for unary/binary/matmul ...
```

- [ ] **Step 2: Run all tests to verify fusion works end-to-end**

Run: `cargo test -p applegpu-core 2>&1`
Expected: All tests pass (fusion is transparent — existing tests produce same results)

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/lazy.rs
git commit -m "feat: integrate fusion pass into LazyRuntime eval"
```

---

### Task 5: Integration tests and Python verification

**Files:**
- Create: `crates/core/tests/fusion_integration.rs`
- Create: `python/tests/test_fusion.py`

- [ ] **Step 1: Create Rust integration test**

```rust
use applegpu_core::device::Device;
use applegpu_core::lazy::LazyRuntime;
use applegpu_core::ops;
use applegpu_core::tensor::Tensor;

#[test]
fn fused_add_relu() {
    let device = match Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, -2.0, 3.0, -4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a);
    rt.insert_tensor(b);

    // add -> relu should be fused into a single kernel
    let sum_id = ops::add(&mut rt, a_id, b_id).unwrap();
    let relu_id = ops::relu(&mut rt, sum_id).unwrap();

    rt.eval(&device, relu_id).unwrap();
    // relu(add([1,-2,3,-4], [10,20,30,40])) = relu([11,18,33,36]) = [11,18,33,36]
    assert_eq!(rt.read_f32(relu_id).unwrap(), &[11.0, 18.0, 33.0, 36.0]);
}

#[test]
fn fused_chain_of_three() {
    let device = match Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![4], &[1.0, 4.0, 9.0, 16.0]).unwrap();
    let a_id = a.meta.id;
    rt.insert_tensor(a);

    // sqrt -> neg -> relu should be fused: relu(neg(sqrt(x)))
    let sqrt_id = ops::sqrt(&mut rt, a_id).unwrap();
    let neg_id = ops::neg(&mut rt, sqrt_id).unwrap();
    let relu_id = ops::relu(&mut rt, neg_id).unwrap();

    rt.eval(&device, relu_id).unwrap();
    // sqrt([1,4,9,16]) = [1,2,3,4]
    // neg([1,2,3,4]) = [-1,-2,-3,-4]
    // relu([-1,-2,-3,-4]) = [0,0,0,0]
    assert_eq!(rt.read_f32(relu_id).unwrap(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn unfused_matmul_not_affected() {
    let device = match Device::new() {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut rt = LazyRuntime::new();

    let a = Tensor::from_f32(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&device, vec![2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let a_id = a.meta.id;
    let b_id = b.meta.id;
    rt.insert_tensor(a);
    rt.insert_tensor(b);

    // matmul can't be fused but should still work
    let c_id = ops::matmul(&mut rt, a_id, b_id).unwrap();
    let relu_id = ops::relu(&mut rt, c_id).unwrap();

    rt.eval(&device, relu_id).unwrap();
    assert_eq!(rt.read_f32(relu_id).unwrap(), &[19.0, 22.0, 43.0, 50.0]);
}
```

- [ ] **Step 2: Create Python test**

```python
import applegpu_runtime as gpu


def test_fused_add_relu():
    gpu.init_backend()
    a = gpu.tensor([1.0, -2.0, 3.0, -4.0], shape=[4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    c = (a + b).relu()
    assert c.to_list() == [11.0, 18.0, 33.0, 36.0]


def test_fused_chain():
    gpu.init_backend()
    a = gpu.tensor([1.0, 4.0, 9.0, 16.0], shape=[4])
    c = a.sqrt().neg().relu()
    assert c.to_list() == [0.0, 0.0, 0.0, 0.0]


def test_fused_mul_add():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    b = gpu.tensor([10.0, 10.0, 10.0, 10.0], shape=[4])
    c = gpu.tensor([5.0, 5.0, 5.0, 5.0], shape=[4])
    # (a * b) + c should fuse: mul then add
    d = (a * b) + c
    assert d.to_list() == [15.0, 25.0, 35.0, 45.0]
```

- [ ] **Step 3: Build and run all tests**

Run: `make clean && make test 2>&1`
Expected: All tests pass across all layers

- [ ] **Step 4: Commit**

```bash
git add crates/core/tests/fusion_integration.rs python/tests/test_fusion.py
git commit -m "test: add fusion integration tests (Rust + Python)"
```

---

### Task 6: End-to-end verification and push

- [ ] **Step 1: Update backlog**

Mark kernel fusion as complete.

- [ ] **Step 2: Update README**

Add fusion note to status section.

- [ ] **Step 3: Push**

```bash
git push origin main
```
