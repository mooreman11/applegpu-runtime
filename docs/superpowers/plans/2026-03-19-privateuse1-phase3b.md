# PrivateUse1 Phase 3b: Native addmm + MLP Performance

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CPU-fallback `addmm` with native GPU dispatch, eliminating the H2D/D2H copy bottleneck that makes MLP training 25x slower than CPU.

**Architecture:** Decompose `addmm(bias, input, weight.t(), beta=1, alpha=1)` in the C++ shim as: `contiguous(mat2)` → `matmul(input, mat2_c)` → `add(result, bias)`. The `contiguous()` call handles transposed weights via our existing `copy_` (which uses `from_blob` to read non-contiguous shared-memory buffers). All subsequent ops stay on GPU — no H2D/D2H copies per layer. Non-unit alpha/beta falls back to CPU (rare in practice).

**Tech Stack:** C++ (applegpu_backend.cpp), Python (tests + benchmark)

---

## File Structure

**Task 1 (Native addmm + contiguous mm):**
- Modify: `backend_cpp/applegpu_backend.cpp` — replace CPU addmm with native decomposition, add contiguous handling to mm
- Modify: `python/tests/test_cpp_backend.py` — addmm and nn.Linear tests

**Task 2 (Re-benchmark MLP):**
- Run: `benchmarks/bench_mlp_cpp.py` — compare against Phase 3a baseline (3.42ms/iter)

---

## Task 1: Native addmm + Contiguous mm

Replace the CPU-fallback `addmm` with a native GPU decomposition. Also make `mm` handle non-contiguous inputs by calling `contiguous()`.

**Files:**
- Modify: `backend_cpp/applegpu_backend.cpp:306-310` (applegpu_mm), `backend_cpp/applegpu_backend.cpp:324-332` (applegpu_addmm)
- Modify: `python/tests/test_cpp_backend.py`

- [ ] **Step 1: Write tests**

Add to `python/tests/test_cpp_backend.py`:

```python
def test_native_addmm():
    """addmm with transposed weight runs on GPU (not CPU fallback)."""
    _load()
    bias = torch.tensor([1.0, 2.0], device='applegpu')
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='applegpu')
    weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device='applegpu')
    # addmm(bias, input, weight.t()) = input @ weight.t() + bias
    result = torch.addmm(bias, input, weight.t()).cpu()
    expected = torch.tensor([[2.0, 4.0], [5.0, 7.0]])
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_linear_layer():
    """nn.Linear forward pass works end-to-end on GPU."""
    _load()
    torch.manual_seed(42)
    layer = torch.nn.Linear(4, 3).to('applegpu')
    x = torch.randn(2, 4).to('applegpu')
    y = layer(x)
    assert y.device.type == 'applegpu'
    assert y.shape == (2, 3)
    # Verify against CPU computation
    layer_cpu = torch.nn.Linear(4, 3)
    layer_cpu.load_state_dict({k: v.cpu() for k, v in layer.state_dict().items()})
    y_expected = layer_cpu(x.cpu())
    assert torch.allclose(y.cpu(), y_expected, atol=1e-5), f"Mismatch: {y.cpu()} vs {y_expected}"
```

- [ ] **Step 2: Verify tests pass (via current CPU fallback)**

Run: `uv run pytest python/tests/test_cpp_backend.py::test_native_addmm python/tests/test_cpp_backend.py::test_linear_layer -v`
Expected: PASS (via CPU fallback — slow but correct)

- [ ] **Step 3: Replace CPU addmm with native GPU decomposition**

In `backend_cpp/applegpu_backend.cpp`, replace the `applegpu_addmm` function (lines 324-332):

```cpp
// addmm: result = beta * bias + alpha * mm(mat1, mat2)
// Decomposed to native GPU ops for the common case (alpha=1, beta=1).
// Non-contiguous inputs (e.g., weight.t()) are handled by contiguous().
at::Tensor applegpu_addmm(const at::Tensor& self, const at::Tensor& mat1,
                           const at::Tensor& mat2, const at::Scalar& beta,
                           const at::Scalar& alpha) {
    double alpha_val = alpha.toDouble();
    double beta_val = beta.toDouble();

    // Non-unit alpha/beta: fall back to CPU (rare — SGD uses add_ with alpha, not addmm)
    if (alpha_val != 1.0 || beta_val != 1.0) {
        applegpu_ffi_synchronize();
        eval_applegpu_tensor_if_needed(mat1);
        eval_applegpu_tensor_if_needed(mat2);
        eval_applegpu_tensor_if_needed(self);
        return at::addmm(self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha)
            .to(c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    // Ensure inputs are contiguous (mat2 is often weight.t())
    auto mat1_c = mat1.is_contiguous() ? mat1 : mat1.contiguous();
    auto mat2_c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    auto self_c = self.is_contiguous() ? self : self.contiguous();

    // mm(mat1, mat2) → [M, N]
    uint64_t mm_out_id = 0;
    void* mm_ptr = applegpu_ffi_matmul_out(
        get_tensor_id(mat1_c), get_tensor_id(mat2_c), &mm_out_id);
    auto mm_result = wrap_ffi_output(
        mm_ptr, mm_out_id, query_output_shape(mm_out_id), mat1.scalar_type());

    // add(mm_result, bias) → [M, N] (bias broadcasts from [N])
    uint64_t add_out_id = 0;
    void* add_ptr = applegpu_ffi_add_out(
        get_tensor_id(mm_result), get_tensor_id(self_c), &add_out_id);
    return wrap_ffi_output(
        add_ptr, add_out_id, query_output_shape(add_out_id), mat1.scalar_type());
}
```

Also update `applegpu_mm` to handle non-contiguous inputs:

```cpp
at::Tensor applegpu_mm(const at::Tensor& self, const at::Tensor& mat2) {
    // Ensure contiguous — transposed views are common (weight.t())
    auto self_c = self.is_contiguous() ? self : self.contiguous();
    auto mat2_c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    uint64_t out_id = 0;
    void* ptr = applegpu_ffi_matmul_out(
        get_tensor_id(self_c), get_tensor_id(mat2_c), &out_id);
    return wrap_ffi_output(ptr, out_id, query_output_shape(out_id), self.scalar_type());
}
```

- [ ] **Step 4: Rebuild C++ extension**

Run: `cd backend_cpp && rm -rf build *.so && ARCHFLAGS="-arch arm64" uv run python setup.py build_ext --inplace`

- [ ] **Step 5: Run all Python tests**

Run: `uv run pytest python/tests/test_cpp_backend.py -v`
Expected: All 12 tests pass

- [ ] **Step 6: Commit**

```bash
git add backend_cpp/applegpu_backend.cpp python/tests/test_cpp_backend.py
git commit -m "feat: native GPU addmm — eliminate CPU fallback for nn.Linear

Decompose addmm(bias, input, weight.t()) to:
  contiguous(weight.t()) → matmul(input, weight_t) → add(result, bias)
All ops stay on GPU for alpha=1, beta=1 (common case).
Non-contiguous inputs handled via contiguous() + from_blob CPU alias.
Non-unit alpha/beta falls back to CPU (rare in practice)."
```

---

## Task 2: Re-benchmark MLP

Run the MLP benchmark to measure the speedup from native addmm.

- [ ] **Step 1: Run MLP benchmark (Phase 3a baseline config)**

Run: `uv run python benchmarks/bench_mlp_cpp.py --hidden 128 --layers 3 --iters 20`

Phase 3a baseline: CPU 0.13ms/iter, applegpu 3.42ms/iter (25x slower)

- [ ] **Step 2: Run with larger sizes (spec target: hidden>=256, batch>=16)**

Run: `uv run python benchmarks/bench_mlp_cpp.py --hidden 256 --layers 3 --batch 16 --iters 50`
Run: `uv run python benchmarks/bench_mlp_cpp.py --hidden 512 --layers 4 --batch 32 --iters 50`

- [ ] **Step 3: Commit benchmark results**

```bash
git commit --allow-empty -m "bench: Phase 3b MLP results — native addmm

[paste numbers]"
```

---

## End of Phase 3b

After these 2 tasks:

1. **Native addmm** — nn.Linear forward stays on GPU (no H2D/D2H per layer)
2. **Benchmark data** — quantified improvement vs Phase 3a (3.42ms/iter baseline)

The forward pass should be significantly faster. Backward pass ops (threshold_backward, mm_backward, etc.) still use CPU fallback, so full training improvement depends on forward-to-backward ratio.
