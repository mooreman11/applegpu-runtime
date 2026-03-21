"""Tests for the custom FX interpreter torch.compile backend."""
import subprocess
import sys
import pytest
import torch


def _load():
    """Load C++ backend. Skip if not built."""
    try:
        from applegpu_runtime.cpp_backend import load_cpp_backend
        load_cpp_backend()
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"C++ backend not built: {e}")


def _load_compile():
    """Load compile backend. Skip if not built."""
    _load()
    from applegpu_runtime.compile_backend import applegpu_compile_backend
    return applegpu_compile_backend


# ── Tensor ID extraction ──────────────────────────────────────────

def test_find_by_data_ptr():
    """applegpu_eager_find_by_data_ptr resolves a tensor's ID."""
    _load()
    from applegpu_runtime.compile_backend import _get_lib, _resolve_tensor_id
    lib = _get_lib()
    t = torch.randn(4, 3, device='applegpu')
    tid = _resolve_tensor_id(lib, t)
    assert tid != 0, "failed to resolve tensor_id"


def test_find_by_data_ptr_view():
    """Reverse lookup works for transposed (view) tensors."""
    _load()
    from applegpu_runtime.compile_backend import _get_lib, _resolve_tensor_id
    lib = _get_lib()
    t = torch.randn(4, 3, device='applegpu')
    tv = t.t()  # Transpose — shares storage
    tid = _resolve_tensor_id(lib, tv)
    assert tid != 0, "failed to resolve view tensor_id"


# ── FX interpreter: individual op dispatch ──────────────────────────

def test_compile_mm():
    """FX interpreter handles aten.mm correctly."""
    backend = _load_compile()
    x = torch.randn(4, 3, device='applegpu')
    w = torch.randn(3, 2, device='applegpu')
    expected = torch.mm(x, w)

    @torch.compile(backend=backend)
    def f(x, w):
        return torch.mm(x, w)

    result = f(x, w)
    assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)


def test_compile_add():
    """FX interpreter handles aten.add correctly."""
    backend = _load_compile()
    a = torch.randn(4, 3, device='applegpu')
    b = torch.randn(4, 3, device='applegpu')
    expected = a + b

    @torch.compile(backend=backend)
    def f(a, b):
        return a + b

    result = f(a, b)
    assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)


def test_compile_relu():
    """FX interpreter handles aten.relu correctly."""
    backend = _load_compile()
    x = torch.randn(4, 3, device='applegpu')
    expected = torch.relu(x)

    @torch.compile(backend=backend)
    def f(x):
        return torch.relu(x)

    result = f(x)
    assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)


# ── FX interpreter: compound ops ──────────────────────────────────

def test_compile_addmm():
    """FX interpreter handles aten.addmm (bias + mm)."""
    backend = _load_compile()
    bias = torch.randn(2, device='applegpu')
    x = torch.randn(4, 3, device='applegpu')
    w = torch.randn(3, 2, device='applegpu')
    expected = torch.addmm(bias, x, w)

    @torch.compile(backend=backend)
    def f(bias, x, w):
        return torch.addmm(bias, x, w)

    result = f(bias, x, w)
    assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-4)


def test_compile_linear_layer():
    """FX interpreter handles nn.Linear (addmm decomposition)."""
    backend = _load_compile()
    torch.manual_seed(42)
    layer = torch.nn.Linear(8, 4).to('applegpu')
    x = torch.randn(2, 8, device='applegpu')
    expected = layer(x)

    compiled_layer = torch.compile(layer, backend=backend)
    result = compiled_layer(x)
    assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-4)


# ── FX interpreter: backward pass ─────────────────────────────────

def _run_in_subprocess(code):
    """Run Python code in a fresh subprocess to avoid torch.compile state conflicts.

    Multiple torch.compile'd backward functions in the same process segfault
    because aot_autograd's compiled backward shares FX interpreter state.
    """
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed (rc={result.returncode}):\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}")


def test_compile_backward_simple():
    """FX interpreter handles backward pass through simple ops."""
    _load()  # ensure backend is built
    _run_in_subprocess("""
import torch
from applegpu_runtime.cpp_backend import load_cpp_backend
load_cpp_backend()
from applegpu_runtime.compile_backend import applegpu_compile_backend

x = torch.randn(4, 3, device='applegpu', requires_grad=True)
w = torch.randn(3, 2, device='applegpu', requires_grad=True)

# Reference (CPU) — detach+clone creates leaf tensors
x_ref = x.detach().cpu().clone().requires_grad_(True)
w_ref = w.detach().cpu().clone().requires_grad_(True)
y_ref = torch.mm(x_ref, w_ref)
y_ref.sum().backward()

# Compiled
@torch.compile(backend=applegpu_compile_backend)
def f(x, w):
    return torch.mm(x, w)

y = f(x, w)
y.sum().backward()
assert torch.allclose(x.grad.cpu(), x_ref.grad, atol=1e-4), \\
    f"x.grad mismatch: {x.grad.cpu()} vs {x_ref.grad}"
assert torch.allclose(w.grad.cpu(), w_ref.grad, atol=1e-4), \\
    f"w.grad mismatch: {w.grad.cpu()} vs {w_ref.grad}"
print("PASS")
""")


def test_compile_mlp_forward():
    """FX interpreter handles MLP forward pass."""
    backend = _load_compile()
    torch.manual_seed(42)

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(8, 16)
            self.fc2 = torch.nn.Linear(16, 4)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = MLP().to('applegpu')
    x = torch.randn(2, 8, device='applegpu')
    expected = model(x)

    compiled_model = torch.compile(model, backend=backend)
    result = compiled_model(x)
    assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-4)


def test_compile_mlp_training_step():
    """FX interpreter handles a full MLP training step (fwd + bwd + optim)."""
    _load()  # ensure backend is built
    _run_in_subprocess("""
import torch
from applegpu_runtime.cpp_backend import load_cpp_backend
load_cpp_backend()
from applegpu_runtime.compile_backend import applegpu_compile_backend

torch.manual_seed(42)

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Reference on CPU
ref_model = MLP()
x_cpu = torch.randn(2, 8)
target_cpu = torch.randn(2, 4)
ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
ref_loss = torch.nn.functional.mse_loss(ref_model(x_cpu), target_cpu)
ref_loss.backward()
ref_opt.step()
ref_params = [p.clone() for p in ref_model.parameters()]

# Compiled on GPU
torch.manual_seed(42)
model = MLP().to('applegpu')
x = x_cpu.to('applegpu')
target = target_cpu.to('applegpu')
compiled_model = torch.compile(model, backend=applegpu_compile_backend)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
out = compiled_model(x)
loss = torch.nn.functional.mse_loss(out, target)
loss.backward()
opt.step()

for i, (p, rp) in enumerate(zip(model.parameters(), ref_params)):
    diff = (p.cpu() - rp).abs().max().item()
    assert diff < 5e-3, f"param {i} mismatch: max diff = {diff}"
print("PASS")
""")
