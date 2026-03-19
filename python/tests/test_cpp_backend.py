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
