# GPU Kernel Gaps Design — Issues #16–#22

**Date:** 2026-03-17
**Scope:** Seven GitHub issues addressing CPU fallbacks and missing GPU kernels, grouped into three PRs by complexity.

---

## PR 1 — Quick Wins (#20 blit copy, #22 vector norm)

### #20: GPU→GPU Blit Copy (eliminate CPU roundtrip in copy_)

**Problem:** `_op_copy_` in torch_backend.py converts GPU→CPU→GPU via `to_torch_cpu()` + `from_torch()`.

**Current state:** Non-blocking blit copy is fully implemented:
- Swift: `gpuBridgeBlitCopyNonBlocking` (compute.swift:2319-2365) uses `MTLBlitCommandEncoder.copy()`
- C header: `gpu_bridge_blit_copy_nb` declared in bridge.h:362-369
- Rust FFI: `gpu_bridge_blit_copy_nb` declared in ffi.rs:391-397

**Missing pieces:**
1. Rust dispatch wrapper for blit copy in compute.rs
2. Python handler update in torch_backend.py `_op_copy_` to use GPU→GPU path

**Design:**
- Add a standalone `blit_copy(device, src, dst, size_bytes)` function in compute.rs (NOT on `KernelRegistry` — blit copy uses `MTLBlitCommandEncoder`, not `MTLComputeCommandEncoder`). Calls `gpu_bridge_blit_copy_nb` + commit + wait.
- Update `_op_copy_` to detect when both src and dst are `ApplegpuTensor`, and use `gpu.blit_copy(src, dst)` instead of CPU roundtrip
- Fall back to current CPU path when src is a plain `torch.Tensor` (no GPU buffer to copy from)

**No new Metal kernel needed.** Pure plumbing.

### #22: GPU linalg_vector_norm (gradient clipping)

**Problem:** `linalg_vector_norm` falls back to CPU (torch_backend.py:1397-1407).

**Current state:** Sum, mean, var reductions are fully GPU-accelerated via template-generated MSL kernels in `kernel_templates.rs`, dispatched through the softmax dispatcher pattern.

**Design — composition approach (no new MSL kernel):**
- **L2 norm** (ord=2, most common): `gpu.mul(x, x) → sum(dims) → gpu.sqrt()`
- **L1 norm** (ord=1): `gpu.abs(x) → sum(dims)`
- **L∞ norm** (ord=inf): CPU fallback — no `amax` reduction kernel exists. File backlog issue.
- **Exotic norms** (other ord values): CPU fallback

Dimension handling reuses `_op_sum`'s arbitrary-dim reduction via transpose tricks (torch_backend.py:754-809). The composed path must handle `keepdim=True` by reshaping the output to restore reduced dimensions as size 1 (same pattern as `_op_sum`).

**L∞ backlog:** Add a TODO comment in the handler and file a GitHub issue for a future `amax` reduction kernel.

---

## PR 2 — New Kernel Variants (#16 pool indices, #18 exact GELU)

### #16: Forward max_pool2d — output indices on GPU

**Problem:** max_pool2d Metal kernel outputs values only. Indices computed on CPU via `torch.nn.functional.max_pool2d_with_indices` (torch_backend.py:1387-1391), causing GPU→CPU roundtrip every forward pass during training.

**Current kernel** (kernel_templates.rs:1156-1208):
- Grid: `(out_w, out_h * channels, batch)`
- Tracks `max_val` but not position
- Single output buffer (values)

**Design — new kernel variant `max_pool2d_with_indices`:**

MSL kernel buffer layout (with indices buffer passed as `input_buffers[1]`):
```metal
device const {t}* input      [[buffer(0)]],   // input_buffers[0] — input tensor
device int*       out_indices [[buffer(1)]],   // input_buffers[1] — writable "input" for indices
device {t}*       output      [[buffer(2)]],   // buf_out — values (placed at buffer(n) by dispatch_3d)
constant uint&    batch       [[buffer(3)]],   // uint_params start here
...
```
**Note on buffer ordering:** `dispatch_cnn_3d` places input buffers at indices 0..n-1 and the output buffer at index n. With 2 input buffers, output lands at buffer(2), and uint_params start at buffer(3). The indices buffer at buffer(1) is declared as `device int*` (writable) in MSL even though it's passed through the "input" buffer array.

- Track `uint max_idx` alongside `float max_val` in the nested kernel loop
- `max_idx = ih * in_w + iw` (flat spatial index within the channel plane)
- Write both: `output[...] = max_val; out_indices[...] = max_idx;`

**Important:** This creates a precedent where "input buffers" are actually outputs. The call site in lazy.rs MUST include a comment documenting that `input_buffers[1]` is a writable output. Long-term, a `dispatch_cnn_3d_multi_output` variant would be more correct.

Dispatch changes:
- New OpKind variant: `MaxPool2dWithIndices { kernel_size, stride, padding }`
- Returns **two tensors** (values + indices)

Rust layer:
- New op `max_pool2d_with_indices()` in ops.rs returning two tensor IDs
- Lazy execution allocates both output buffers, passes indices buffer as input_buffers[1]

Python layer:
- `_max_pool2d` handler calls `gpu.max_pool2d_with_indices()` instead of `gpu.max_pool2d()` + CPU indices
- Returns `(values_tensor, indices_tensor)` both as `ApplegpuTensor`

### #18: Support exact GELU mode (approximate='none')

**Problem:** Forward GELU kernel hardcodes tanh approximation (kernel_templates.rs:271-307). The `approximate` parameter is accepted but silently ignored (torch_backend.py:366). Backward is CPU-only and also only implements tanh mode.

**Correctness bug:** If a model passes `approximate="none"`, forward computes tanh-approximate GELU but backward uses tanh derivative — numerically consistent but mathematically wrong for the exact GELU the model requested.

**Design — four new kernel paths:**

1. **Exact GELU forward kernel** (`gelu_exact`):
   ```metal
   output = 0.5f * x * (1.0f + erf(x * M_SQRT1_2_F));
   ```
   Metal stdlib provides `erf()`.

2. **Exact GELU backward kernel** (`gelu_exact_backward`):
   ```metal
   float cdf = 0.5f * (1.0f + erf(x * M_SQRT1_2_F));
   // Standard normal PDF: 1/sqrt(2π) * exp(-x²/2)
   // M_2_SQRTPI_F = 2/sqrt(π), M_SQRT1_2_F = 1/sqrt(2)
   // So: M_2_SQRTPI_F * M_SQRT1_2_F * 0.5 = 1/sqrt(2π) ≈ 0.3989422804
   float pdf = exp(-0.5f * x * x) * 0.3989422804f;
   output = grad * (cdf + x * pdf);
   ```

3. **Tanh GELU backward kernel** (`gelu_tanh_backward`):
   ```metal
   float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
   inner = clamp(inner, -10.0f, 10.0f);
   float tanh_inner = tanh(inner);
   float dtanh = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
   output = grad * (0.5f * (1.0f + tanh_inner) + 0.5f * x * (1.0f - tanh_inner * tanh_inner) * dtanh);
   ```

4. **Existing tanh GELU forward** — unchanged

Rust layer:
- **Recommended: new separate OpKind variants** — `GeluExact`, `GeluTanhBackward`, `GeluExactBackward`
- Keep existing `OpKind::Gelu` unchanged (avoids breaking all `is_gelu()`, `is_elementwise()`, and `kernel_name()` match arms)
- Add `is_gelu_exact()` helper, update `is_elementwise()` to include new variants
- New kernel template functions in kernel_templates.rs

Python layer:
- `_op_gelu`: route based on `approximate` param — `"tanh"` → existing kernel, `"none"` → new exact kernel
- `_op_gelu_backward`: route based on `approximate` param — dispatch to appropriate backward kernel instead of CPU

Fusion system:
- Add exact GELU expression to fusion.rs (using `erf`)

---

## PR 3 — Heavy Kernels (#17 conv grad_weight, #19 grouped conv, #21 index/scatter)

### #17: Conv grad_weight/grad_bias on GPU

**Problem:** For conv1d, conv2d, layer_norm, batch_norm backward: grad_input runs on Metal but grad_weight/grad_bias fall back to CPU.

**Current state:**
- Conv2d grad_input: Metal kernel exists (kernel_templates.rs:1366-1428)
- Conv2d grad_weight: CPU via `torch.nn.grad.conv2d_weight()` (torch_backend.py:1610)
- Conv1d grad_input: CPU fallback (no Metal kernel)
- grad_bias: CPU sum (torch_backend.py:1616-1618)
- All kernels use single output buffer — no multi-output dispatch

**Design:**

**Conv2d grad_weight kernel** (`conv2d_backward_weight`):
- Thread per weight element: grid `(kw, kh * out_channels, in_channels)` — each thread maps to one unique `grad_weight[oc, ic, kh_i, kw_j]` element
- Each thread accumulates **locally** into a `float sum` register over `batch × out_h × out_w`: `sum += input[b,ic,ih,iw] * grad_output[b,oc,oh,ow]`
- **No atomics needed** — each thread writes to a unique weight element (unlike embedding_backward where multiple tokens scatter to the same embedding row)
- Single write per thread: `grad_weight[...] = sum`
- Dispatched as separate kernel from grad_input (single-output constraint)

**Conv1d backward_input kernel** (`conv1d_backward_input`):
- Same transposed-conv pattern as conv2d_backward_input but 1D
- Grid: `(in_length, in_channels, batch)`
- New OpKind: `Conv1dBackwardInput { stride, padding }`

**grad_bias:**
- Can be computed as `gpu.sum()` over batch and spatial dims with reshaping in Python
- No new kernel needed — compose from existing reduction ops

**layer_norm/batch_norm grad_weight/bias:**
- Small tensors (size = hidden_dim), CPU sum is acceptable
- Defer to future work unless profiling shows bottleneck

### #19: Grouped Convolution (groups > 1)

**Problem:** Conv forward/backward only support groups=1. `groups != 1` returns `NotImplemented` at Python layer (torch_backend.py:1341).

**Current state:** No `groups` parameter at any layer — Rust ops, OpKind variants, MSL kernels all assume full channel connectivity.

**Design — changes at 7 layers:**

1. **MSL kernels** (kernel_templates.rs):
   - Add `constant uint& groups` buffer parameter
   - Compute `in_channels_per_group = in_channels / groups` and `out_channels_per_group = out_channels / groups`
   - Map output channel to group: `uint group = oc / out_channels_per_group`
   - Adjust input channel iteration: `for (uint ic = group * in_channels_per_group; ic < (group + 1) * in_channels_per_group; ic++)`
   - Adjust weight indexing: `weight[oc * in_channels_per_group * kh * kw + (ic - group * in_channels_per_group) * kh * kw + ...]`
   - Apply to: conv1d, conv2d, conv2d_backward_input, conv1d_backward_input (new), conv2d_backward_weight (new)

2. **OpKind** (graph.rs): Add `groups: usize` to `Conv1d`, `Conv2d`, `Conv1dBackwardInput`, `Conv2dBackwardInput`, `Conv2dBackwardWeight`

3. **Rust ops** (ops.rs): Add `groups` param, validate `in_channels % groups == 0` and `out_channels % groups == 0`, check `weight[1] == in_channels / groups`

4. **Backend trait** (backend.rs): Add `groups` to conv signatures

5. **Metal backend** (metal_backend.rs): Pass `groups` through

6. **PyO3** (lib.rs): Add `groups=1` default parameter

7. **torch_backend.py**: Remove `groups != 1` rejection, pass `groups` to GPU layer

**Depthwise convolution** (groups == in_channels == out_channels) works automatically with this design.

### #21: GPU Index/Gather and Index_Put/Scatter

**Problem:** `index.Tensor` and `index_put_` fall back to CPU with GPU→CPU→GPU roundtrips.

**Current state:**
- GPU gather kernels exist but only for 2D tensors, dim 0/1, Int32 indices
- GPU index_select kernels exist with same constraints
- Atomic float add (CAS loop) exists in embedding_backward
- No prefix-sum/stream compaction anywhere in codebase

**Design — incremental improvements:**

**Phase 1 (this PR): Integer index improvements**

`index.Tensor` with integer indices:
- Detect when all indices are integer tensors (not boolean masks)
- **Phase 1 scope: single integer index on one dimension only.** Multi-index advanced indexing (multiple integer tensors with broadcasting) remains CPU fallback — the broadcasting/fancy indexing semantics are significantly more complex than single-dimension gather.
- For simple cases (single integer index on dim 0 or 1, 2D input): route to existing `gather`/`index_select` GPU kernels
- For N-D integer indexing on a single dim: extend gather kernel to support N-D tensors with stride-based addressing (similar to existing N-D elementwise kernels using `nd_index_helper`)

`index_put_` improvements:
- `accumulate=False`: new scatter kernel — each thread writes `output[index[i]] = value[i]`, no atomics needed (last write wins for duplicates, matching PyTorch semantics)
- `accumulate=True`: scatter-add kernel using `atomic_add_float` CAS loop (same as embedding_backward)
- Both limited to integer indices initially

**Phase 2 (deferred): Boolean mask indexing**
- Requires prefix-sum (parallel scan) for stream compaction
- Variable-length output makes this architecturally different
- Keep as CPU fallback with TODO comment

**New OpKind variants:**
- `ScatterWrite { dim: usize }` — for index_put_ without accumulate
- `ScatterAdd { dim: usize }` — for index_put_ with accumulate
- Extend existing `Gather`/`IndexSelect` to support N-D

---

## Testing Strategy

Each PR includes tests at all three layers:

**Rust tests** (crates/core/tests/):
- Unit tests for new OpKind variants and shape validation
- Integration tests for kernel dispatch correctness

**Swift tests** (swift/Tests/):
- Direct Metal kernel validation where applicable

**Python tests** (python/tests/):
- Correctness: GPU result vs CPU torch reference for each op
- Edge cases: empty tensors, single-element, large tensors
- Dtype coverage: Float32, Float16, BFloat16
- **Numerical tolerances**: `atol=1e-5` for Float32, `atol=1e-2` for Float16/BFloat16
- For grouped conv: groups=1 (regression), groups=2, depthwise (groups=channels)
- For GELU: both approximate modes; backward gradient checking via finite-difference at Float32 with relaxed tolerances (no Float64 on M4)
- For conv grad_weight: test with non-square dimensions (different in_channels, out_channels, kh, kw) to catch indexing bugs
- For index ops: various index patterns, accumulate flag
- **Performance sanity check**: verify GPU paths are not slower than CPU fallbacks they replace (simple wall-clock comparison)

---

## Backlog Items (not in scope)

- L∞ norm on GPU (needs `amax` reduction kernel) — file as separate issue
- Boolean mask indexing on GPU (needs prefix-sum) — deferred to Phase 2
- Multi-output dispatch infrastructure (`dispatch_cnn_3d_multi_output`) — future refactor
- Threadgroup shared memory reductions — no kernels use this yet, could improve grad_weight perf
