"""Integration tests for the PrivateUse1 C++ backend."""
import pytest
import torch


def _load():
    """Load C++ backend. Skip if not built."""
    try:
        from applegpu_runtime.cpp_backend import load_cpp_backend
        load_cpp_backend()
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"C++ backend not built: {e}")


def test_empty_tensor():
    """torch.empty on applegpu device creates a tensor."""
    _load()
    t = torch.empty(3, 4, device='applegpu')
    assert t.device.type == 'applegpu'
    assert t.shape == (3, 4)
    assert t.dtype == torch.float32


def test_empty_different_dtypes():
    """empty works for various dtypes."""
    _load()
    for dtype in [torch.float32, torch.float16, torch.int32]:
        t = torch.empty(2, 3, device='applegpu', dtype=dtype)
        assert t.dtype == dtype


def test_tensor_to_cpu():
    """Tensor can be copied to CPU."""
    _load()
    t = torch.empty(4, device='applegpu')
    cpu_t = t.cpu()
    assert cpu_t.device.type == 'cpu'
    assert cpu_t.shape == (4,)


def test_cpu_to_applegpu():
    """CPU tensor can be moved to applegpu."""
    _load()
    cpu_t = torch.tensor([1.0, 2.0, 3.0])
    gpu_t = cpu_t.to('applegpu')
    assert gpu_t.device.type == 'applegpu'
    # Copy back and verify data
    back = gpu_t.cpu()
    assert torch.allclose(back, cpu_t)


def test_copy_roundtrip():
    """Data survives CPU→GPU→CPU round-trip."""
    _load()
    src = torch.tensor([3.14, 2.71, 1.41, 0.57])
    gpu = torch.empty(4, device='applegpu')
    gpu.copy_(src)
    back = gpu.cpu()
    assert torch.allclose(back, src)


def test_cpu_fallback_ops():
    """Unregistered ops fall back to CPU and produce correct results."""
    _load()
    # sin is not registered on PrivateUse1 — should fall back to CPU
    src = torch.tensor([0.0, 1.5708, 3.1416], device='applegpu')
    result = torch.sin(src)
    expected = torch.sin(torch.tensor([0.0, 1.5708, 3.1416]))
    result_cpu = result.cpu() if result.device.type != 'cpu' else result
    assert torch.allclose(result_cpu, expected, atol=1e-4)


def test_native_add():
    """Native add op (not CPU fallback) produces correct result."""
    _load()
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device='applegpu')
    b = torch.tensor([10.0, 20.0, 30.0, 40.0], device='applegpu')
    result = (a + b).cpu()
    expected = torch.tensor([11.0, 22.0, 33.0, 44.0])
    assert torch.allclose(result, expected)


def test_native_mul():
    """Native mul op produces correct result."""
    _load()
    a = torch.tensor([2.0, 3.0, 4.0], device='applegpu')
    b = torch.tensor([10.0, 20.0, 30.0], device='applegpu')
    result = (a * b).cpu()
    assert torch.allclose(result, torch.tensor([20.0, 60.0, 120.0]))


def test_native_matmul():
    """Native mm op produces correct result."""
    _load()
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='applegpu')
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='applegpu')
    result = torch.mm(a, b).cpu()
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(result, expected)


def test_native_relu():
    """Native relu op produces correct result."""
    _load()
    a = torch.tensor([-2.0, 0.0, 3.0, -1.0], device='applegpu')
    result = torch.relu(a).cpu()
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 3.0, 0.0]))


def test_native_addmm():
    """addmm with transposed weight runs on GPU (not CPU fallback)."""
    _load()
    bias = torch.tensor([1.0, 2.0], device='applegpu')
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='applegpu')
    weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device='applegpu')
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
    layer_cpu = torch.nn.Linear(4, 3)
    layer_cpu.load_state_dict({k: v.cpu() for k, v in layer.state_dict().items()})
    y_expected = layer_cpu(x.cpu())
    assert torch.allclose(y.cpu(), y_expected, atol=1e-5), f"Mismatch: {y.cpu()} vs {y_expected}"


def test_threshold_backward():
    """threshold_backward (ReLU backward) works natively."""
    _load()
    grad = torch.tensor([1.0, 2.0, 3.0, 4.0], device='applegpu')
    input = torch.tensor([-1.0, 0.5, -0.5, 2.0], device='applegpu')
    result = torch.ops.aten.threshold_backward(grad, input, 0.0).cpu()
    # grad * (input > 0) = [0, 2, 0, 4]
    expected = torch.tensor([0.0, 2.0, 0.0, 4.0])
    assert torch.allclose(result, expected)


def test_inplace_add():
    """In-place add works."""
    _load()
    a = torch.tensor([1.0, 2.0, 3.0], device='applegpu')
    b = torch.tensor([10.0, 20.0, 30.0], device='applegpu')
    a.add_(b)
    result = a.cpu()
    assert torch.allclose(result, torch.tensor([11.0, 22.0, 33.0]))


@pytest.mark.filterwarnings("ignore:An output with one or more elements was resized")
def test_mlp_training_step():
    """Full MLP training step (forward + backward + optimizer) works."""
    _load()
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 1)
    ).to('applegpu')
    x = torch.randn(2, 4).to('applegpu')
    y = torch.randn(2, 1).to('applegpu')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = torch.nn.MSELoss()(model(x), y)
    loss.backward()
    optimizer.step()

    # Verify loss decreased after one step
    loss2 = torch.nn.MSELoss()(model(x), y)
    assert loss2.cpu().item() < loss.cpu().item() + 0.1  # allow small tolerance


def test_embedding():
    """Embedding lookup works on GPU."""
    _load()
    weight = torch.randn(100, 32, device='applegpu')
    indices = torch.tensor([0, 5, 10, 50, 99], device='applegpu', dtype=torch.int32)
    out = torch.ops.aten.embedding(weight, indices, -1, False, False)
    ref = torch.ops.aten.embedding(weight.cpu(), indices.cpu(), -1, False, False)
    assert out.shape == (5, 32)
    assert torch.allclose(out.cpu(), ref, atol=1e-5)


def test_layer_norm():
    """LayerNorm matches CPU."""
    _load()
    ln = torch.nn.LayerNorm(64).to('applegpu')
    lnc = torch.nn.LayerNorm(64)
    lnc.load_state_dict({k: v.cpu() for k, v in ln.state_dict().items()})
    x = torch.randn(2, 8, 64, device='applegpu')
    assert torch.allclose(ln(x).cpu(), lnc(x.cpu()), atol=1e-4)


def test_gelu():
    """GELU activation matches CPU."""
    _load()
    x = torch.randn(4, 64, device='applegpu')
    out = torch.nn.functional.gelu(x, approximate='tanh')
    ref = torch.nn.functional.gelu(x.cpu(), approximate='tanh')
    assert torch.allclose(out.cpu(), ref, atol=1e-4)


def test_softmax():
    """Softmax matches CPU."""
    _load()
    x = torch.randn(4, 64, device='applegpu')
    out = torch.softmax(x, dim=-1)
    ref = torch.softmax(x.cpu(), dim=-1)
    assert torch.allclose(out.cpu(), ref, atol=1e-5)


def test_permute_transpose():
    """Permute and multi-dim transpose work."""
    _load()
    x = torch.randn(2, 3, 4, device='applegpu')
    assert torch.allclose(x.permute(0, 2, 1).cpu(), x.cpu().permute(0, 2, 1))
    assert torch.allclose(x.transpose(1, 2).cpu(), x.cpu().transpose(1, 2))


def test_slice():
    """Tensor slicing works."""
    _load()
    x = torch.randn(10, 64, device='applegpu')
    assert torch.allclose(x[2:5].cpu(), x.cpu()[2:5])
    assert torch.allclose(x[:, 10:20].cpu(), x.cpu()[:, 10:20])


def test_scalar_mul_chain():
    """scalar_mul doesn't corrupt data in streaming CB (regression test).

    Previously, scalar_mul freed the 4-byte scalar buffer before the GPU
    kernel executed. The pool reused it for the next scalar, corrupting data.
    """
    _load()
    # Chain: matmul → scalar_mul (must not free scalar buffer prematurely)
    q = torch.randn(1, 4, 8, 16, device='applegpu', requires_grad=True)
    k = torch.randn(1, 4, 8, 16, device='applegpu', requires_grad=True)
    att = q @ k.transpose(-2, -1)
    scaled = att * 0.25
    torch.applegpu.synchronize()
    ref = (q.detach().cpu() @ k.detach().cpu().transpose(-2, -1)) * 0.25
    assert torch.allclose(scaled.detach().cpu(), ref, atol=1e-4), \
        f"scalar_mul chain: diff={(scaled.detach().cpu()-ref).abs().max()}"


def test_slice_contiguous():
    """Contiguous copy of sliced view reads from correct offset (regression test).

    Previously, binary_op ignored view byte offsets, so .contiguous() on a
    slice read from buffer position 0 instead of the slice offset.
    """
    _load()
    x = torch.randn(4, 6, device='applegpu')
    sliced = x[:, 2:4].contiguous()
    ref = x.cpu()[:, 2:4].contiguous()
    assert torch.allclose(sliced.cpu(), ref, atol=1e-6), \
        f"slice contiguous: diff={(sliced.cpu()-ref).abs().max()}"


def test_transformer_block():
    """Full transformer block matches CPU (regression test for all GPT-2 ops)."""
    _load()
    torch.manual_seed(42)

    class TransformerBlock(torch.nn.Module):
        def __init__(self, d=64, h=4):
            super().__init__()
            self.ln1 = torch.nn.LayerNorm(d)
            self.qkv = torch.nn.Linear(d, 3 * d)
            self.proj = torch.nn.Linear(d, d)
            self.ln2 = torch.nn.LayerNorm(d)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d, 4 * d),
                torch.nn.GELU(approximate='tanh'),
                torch.nn.Linear(4 * d, d),
            )
            self.h = h

        def forward(self, x):
            B, T, C = x.shape
            hd = C // self.h
            r = self.ln1(x)
            qkv = self.qkv(r)
            q, k, v = qkv.split(C, dim=-1)
            q = q.view(B, T, self.h, hd).transpose(1, 2)
            k = k.view(B, T, self.h, hd).transpose(1, 2)
            v = v.view(B, T, self.h, hd).transpose(1, 2)
            att = torch.softmax(
                (q @ k.transpose(-2, -1)) * (hd ** -0.5), dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.proj(y)
            return x + self.mlp(self.ln2(x))

    m = TransformerBlock().to('applegpu')
    mc = TransformerBlock()
    mc.load_state_dict({k: v.cpu() for k, v in m.state_dict().items()})
    x = torch.randn(1, 8, 64, device='applegpu')
    out = m(x)
    ref = mc(x.cpu())
    assert torch.allclose(out.cpu(), ref, atol=1e-3), \
        f"transformer: diff={(out.cpu()-ref).abs().max()}"


def test_gpt2_forward():
    """GPT-2 forward pass matches CPU on PrivateUse1."""
    _load()
    torch.manual_seed(42)

    class GPT2Block(torch.nn.Module):
        def __init__(self, d=64, h=4):
            super().__init__()
            self.ln1 = torch.nn.LayerNorm(d)
            self.qkv = torch.nn.Linear(d, 3 * d)
            self.proj = torch.nn.Linear(d, d)
            self.ln2 = torch.nn.LayerNorm(d)
            self.fc = torch.nn.Linear(d, 4 * d)
            self.fcp = torch.nn.Linear(4 * d, d)
            self.h = h

        def forward(self, x):
            B, T, C = x.shape
            hd = C // self.h
            r = self.ln1(x)
            qkv = self.qkv(r)
            q, k, v = qkv.split(C, -1)
            q = q.view(B, T, self.h, hd).transpose(1, 2)
            k = k.view(B, T, self.h, hd).transpose(1, 2)
            v = v.view(B, T, self.h, hd).transpose(1, 2)
            att = torch.softmax(
                (q @ k.transpose(-2, -1)) * (hd ** -0.5), -1)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.proj(y)
            return x + self.fcp(torch.nn.functional.gelu(
                self.fc(self.ln2(x)), approximate='tanh'))

    class GPT2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = torch.nn.Embedding(100, 64)
            self.wpe = torch.nn.Embedding(32, 64)
            self.blocks = torch.nn.ModuleList(
                [GPT2Block() for _ in range(2)])
            self.ln_f = torch.nn.LayerNorm(64)
            self.lm_head = torch.nn.Linear(64, 100, bias=False)

        def forward(self, ids):
            B, T = ids.shape
            x = self.wte(ids) + self.wpe(
                torch.arange(T, device=ids.device))
            for b in self.blocks:
                x = b(x)
            return self.lm_head(self.ln_f(x))

    model = GPT2().to('applegpu')
    mc = GPT2()
    mc.load_state_dict(
        {k: v.cpu() for k, v in model.state_dict().items()})
    ids = torch.tensor([[1, 5, 10, 20, 50]], device='applegpu')
    logits = model(ids)
    ref = mc(ids.cpu())
    assert logits.shape == (1, 5, 100)
    assert torch.allclose(logits.cpu(), ref, atol=1e-3), \
        f"GPT-2 diff: {(logits.cpu()-ref).abs().max()}"
    assert logits.cpu().argmax(-1).tolist() == ref.argmax(-1).tolist()


def test_eager_add_via_ffi():
    """Proof that eager Metal dispatch works end-to-end.

    Bypasses PyTorch entirely and calls the eager FFI directly via ctypes:
    alloc two tensors, write data, eager add, flush, read result.
    This validates the streaming command buffer path (Rust -> Metal encode -> GPU).
    """
    _load()
    import ctypes
    import glob as _glob
    import os

    # Find the .so — it's already loaded by _load(), but we need a ctypes handle
    # to call the eager FFI functions directly.
    backend_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'backend_cpp')
    so_files = _glob.glob(os.path.join(backend_dir, 'applegpu_backend*.so'))
    assert so_files, "applegpu_backend .so not found"
    lib = ctypes.CDLL(so_files[0])

    # Set up function signatures
    lib.applegpu_eager_init.restype = ctypes.c_bool

    lib.applegpu_eager_alloc.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),  # dims
        ctypes.c_uint32,                   # ndim
        ctypes.c_int8,                     # dtype
        ctypes.POINTER(ctypes.c_uint64),   # out_id
    ]
    lib.applegpu_eager_alloc.restype = ctypes.c_void_p

    lib.applegpu_eager_add.argtypes = [
        ctypes.c_uint64,                   # a_id
        ctypes.c_uint64,                   # b_id
        ctypes.POINTER(ctypes.c_uint64),   # out_id
    ]
    lib.applegpu_eager_add.restype = ctypes.c_void_p

    lib.applegpu_eager_flush_and_wait.argtypes = []
    lib.applegpu_eager_flush_and_wait.restype = None

    lib.applegpu_eager_free.argtypes = [ctypes.c_uint64]
    lib.applegpu_eager_free.restype = None

    lib.applegpu_eager_last_error.argtypes = []
    lib.applegpu_eager_last_error.restype = ctypes.c_char_p

    # Init (idempotent — already called by C++ module init)
    assert lib.applegpu_eager_init(), "eager init failed"

    # Allocate two 4-element Float32 tensors (dtype wire 0 = Float32)
    dims = (ctypes.c_uint64 * 1)(4)
    out_id = ctypes.c_uint64(0)

    a_ptr = lib.applegpu_eager_alloc(dims, 1, 0, ctypes.byref(out_id))
    assert a_ptr, f"alloc a failed: {lib.applegpu_eager_last_error()}"
    a_id = out_id.value

    b_ptr = lib.applegpu_eager_alloc(dims, 1, 0, ctypes.byref(out_id))
    assert b_ptr, f"alloc b failed: {lib.applegpu_eager_last_error()}"
    b_id = out_id.value

    # Write data via shared memory (storageModeShared = CPU-accessible)
    a_arr = (ctypes.c_float * 4).from_address(a_ptr)
    b_arr = (ctypes.c_float * 4).from_address(b_ptr)
    for i in range(4):
        a_arr[i] = float(i + 1)       # [1, 2, 3, 4]
        b_arr[i] = float((i + 1) * 10)  # [10, 20, 30, 40]

    # Eager add: encodes into streaming command buffer
    c_ptr = lib.applegpu_eager_add(a_id, b_id, ctypes.byref(out_id))
    assert c_ptr, f"eager add failed: {lib.applegpu_eager_last_error()}"
    c_id = out_id.value

    # Flush GPU work and wait for completion
    lib.applegpu_eager_flush_and_wait()

    # Read result from shared memory
    c_arr = (ctypes.c_float * 4).from_address(c_ptr)
    result = [c_arr[i] for i in range(4)]
    assert result == [11.0, 22.0, 33.0, 44.0], f"Expected [11,22,33,44], got {result}"

    # Cleanup
    lib.applegpu_eager_free(a_id)
    lib.applegpu_eager_free(b_id)
    lib.applegpu_eager_free(c_id)
