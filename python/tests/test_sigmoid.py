"""Tests for sigmoid op across all layers."""
import math
import applegpu_runtime as gpu


def test_sigmoid_basic():
    """Sigmoid forward on Metal — basic values."""
    gpu.init_backend()
    a = gpu.tensor([0.0, 1.0, -1.0], shape=[3])
    result = gpu.sigmoid(a).to_list()
    assert abs(result[0] - 0.5) < 1e-5
    assert abs(result[1] - 0.7311) < 1e-3
    assert abs(result[2] - 0.2689) < 1e-3


def test_sigmoid_extreme():
    """Sigmoid saturates correctly at extremes."""
    gpu.init_backend()
    a = gpu.tensor([10.0, -10.0, 50.0, -50.0], shape=[4])
    result = gpu.sigmoid(a).to_list()
    assert result[0] > 0.999
    assert result[1] < 0.001
    assert result[2] > 0.999
    assert result[3] < 0.001


def test_sigmoid_nd():
    """Sigmoid works on N-D tensors."""
    gpu.init_backend()
    a = gpu.tensor([0.0] * 24, shape=[2, 3, 4])
    result = gpu.sigmoid(a)
    assert result.shape == [2, 3, 4]
    vals = result.to_list()
    for v in vals:
        assert abs(v - 0.5) < 1e-5


def test_sigmoid_method():
    """GpuTensor.sigmoid() method works."""
    gpu.init_backend()
    a = gpu.tensor([0.0, 1.0], shape=[2])
    result = a.sigmoid().to_list()
    assert abs(result[0] - 0.5) < 1e-5


def test_sigmoid_torch_forward():
    """torch.sigmoid dispatches to Metal via aten."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    x = gpu.to_applegpu(torch.tensor([0.0, 1.0, -1.0]))
    y = torch.sigmoid(x)
    vals = y.to_torch_cpu().tolist()
    assert abs(vals[0] - 0.5) < 1e-5
    assert abs(vals[1] - 0.7311) < 1e-3
    assert abs(vals[2] - 0.2689) < 1e-3


def test_sigmoid_torch_backward():
    """sigmoid backward via autograd produces correct gradients."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    x = gpu.to_applegpu(torch.tensor([0.0, 1.0, -1.0], requires_grad=True))
    y = torch.sigmoid(x)
    y.sum().backward()
    grads = x.grad.to_torch_cpu().tolist()
    # grad = sigmoid(x) * (1 - sigmoid(x))
    for xi, gi in zip([0.0, 1.0, -1.0], grads):
        s = 1.0 / (1.0 + math.exp(-xi))
        expected = s * (1 - s)
        assert abs(gi - expected) < 1e-4, f"x={xi}: got {gi}, expected {expected}"
