"""Tests for autograd/training support."""
import pytest
import torch
import torch.nn as nn

import applegpu_runtime as gpu
from applegpu_runtime.torch_backend import ApplegpuTensor, set_eager_mode


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()
    gpu.enable_torch_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)
    # Eager mode required for training: lazy runtime may free intermediates
    # needed by backward pass
    set_eager_mode(True)
    yield
    set_eager_mode(False)


def test_simple_gradient():
    """Basic gradient computation: y = x^2, dy/dx = 2x."""
    x = ApplegpuTensor.from_torch(torch.tensor([2.0, 3.0], requires_grad=True))
    y = x * x  # y = x^2
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    expected = torch.tensor([4.0, 6.0])
    grad = x.grad.to_torch_cpu() if isinstance(x.grad, ApplegpuTensor) else x.grad
    assert torch.allclose(grad, expected, atol=1e-4)


def test_matmul_gradient():
    """Gradient flows through matmul."""
    a = ApplegpuTensor.from_torch(torch.randn(2, 3, requires_grad=True))
    b = ApplegpuTensor.from_torch(torch.randn(3, 4, requires_grad=True))
    c = a @ b
    loss = c.sum()
    loss.backward()
    assert a.grad is not None
    assert b.grad is not None


def test_relu_gradient():
    """Gradient flows through relu."""
    x = ApplegpuTensor.from_torch(torch.tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True))
    y = torch.relu(x)
    loss = y.sum()
    loss.backward()
    expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
    grad = x.grad.to_torch_cpu() if isinstance(x.grad, ApplegpuTensor) else x.grad
    assert torch.allclose(grad, expected, atol=1e-4)


def test_add_gradient():
    """Gradient flows through add."""
    a = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0], requires_grad=True))
    b = ApplegpuTensor.from_torch(torch.tensor([3.0, 4.0], requires_grad=True))
    c = a + b
    loss = c.sum()
    loss.backward()
    assert a.grad is not None
    assert b.grad is not None
    grad_a = a.grad.to_torch_cpu() if isinstance(a.grad, ApplegpuTensor) else a.grad
    grad_b = b.grad.to_torch_cpu() if isinstance(b.grad, ApplegpuTensor) else b.grad
    assert torch.allclose(grad_a, torch.tensor([1.0, 1.0]), atol=1e-4)
    assert torch.allclose(grad_b, torch.tensor([1.0, 1.0]), atol=1e-4)


def test_sub_gradient():
    """Gradient flows through sub."""
    a = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0], requires_grad=True))
    b = ApplegpuTensor.from_torch(torch.tensor([3.0, 4.0], requires_grad=True))
    c = a - b
    loss = c.sum()
    loss.backward()
    grad_a = a.grad.to_torch_cpu() if isinstance(a.grad, ApplegpuTensor) else a.grad
    grad_b = b.grad.to_torch_cpu() if isinstance(b.grad, ApplegpuTensor) else b.grad
    assert torch.allclose(grad_a, torch.tensor([1.0, 1.0]), atol=1e-4)
    assert torch.allclose(grad_b, torch.tensor([-1.0, -1.0]), atol=1e-4)


def test_mlp_training_step():
    """One training step on a simple MLP."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    model = gpu.to_applegpu(model)

    x = ApplegpuTensor.from_torch(torch.randn(16, 4))
    target = ApplegpuTensor.from_torch(torch.randn(16, 2))

    # Forward
    output = model(x)

    # Loss (MSE)
    diff = output - target
    loss = (diff * diff).sum()

    # Backward
    loss.backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"


def test_mlp_loss_decreases():
    """Training a simple MLP -- loss should decrease over steps."""
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Synthetic data
    x = ApplegpuTensor.from_torch(torch.randn(32, 4))
    target = ApplegpuTensor.from_torch(torch.randn(32, 1))

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        output = model(x)
        diff = output - target
        loss = (diff * diff).mean()

        loss_val = loss.to_torch_cpu().item() if isinstance(loss, ApplegpuTensor) else loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()

    print(f"Losses: {[f'{l:.4f}' for l in losses]}")
    # Loss should generally decrease
    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"
