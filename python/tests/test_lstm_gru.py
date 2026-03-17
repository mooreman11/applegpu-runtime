"""Validate LSTM and GRU work via PyTorch decomposition on applegpu Metal backend.

PyTorch auto-decomposes nn.LSTM/nn.GRU into matmul + sigmoid + tanh + element-wise
ops, all of which have GPU kernels. This test validates the decomposition path works
end-to-end for forward and backward passes.

See: https://github.com/mooreman11/applegpu-runtime/issues/10
"""
import pytest
import torch
import torch.nn as nn
import applegpu_runtime as gpu
from applegpu_runtime.torch_backend import ApplegpuTensor, set_eager_mode


@pytest.fixture(autouse=True)
def _backend():
    gpu.init_backend()
    gpu.enable_torch_backend()
    set_eager_mode(True)
    yield
    set_eager_mode(False)


def test_lstm_forward():
    """LSTM forward pass produces valid output shapes."""
    model = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
    model = gpu.to_applegpu(model)
    x = ApplegpuTensor.from_torch(torch.randn(4, 10, 32))
    output, (h_n, c_n) = model(x)
    assert output.shape == (4, 10, 64)
    assert h_n.shape == (2, 4, 64)
    assert c_n.shape == (2, 4, 64)
    out_cpu = output.to_torch_cpu()
    assert torch.isfinite(out_cpu).all()


def test_lstm_backward():
    """LSTM backward pass computes gradients."""
    model = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
    model = gpu.to_applegpu(model)
    x = ApplegpuTensor.from_torch(torch.randn(2, 5, 16, requires_grad=True))
    output, _ = model(x)
    loss = output.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        grad_cpu = param.grad.to_torch_cpu() if hasattr(param.grad, 'to_torch_cpu') else param.grad
        assert torch.isfinite(grad_cpu).all(), f"Non-finite gradient for {name}"


def test_gru_forward():
    """GRU forward pass produces valid output shapes.

    Note: batch_first=False avoids aten::transpose_ (in-place) which isn't
    yet supported on GPU. GRU input is (seq_len, batch, features).
    """
    model = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=False)
    model = gpu.to_applegpu(model)
    x = ApplegpuTensor.from_torch(torch.randn(10, 4, 32))  # (seq, batch, features)
    output, h_n = model(x)
    assert output.shape == (10, 4, 64)
    assert h_n.shape == (2, 4, 64)
    out_cpu = output.to_torch_cpu()
    assert torch.isfinite(out_cpu).all()


def test_gru_backward():
    """GRU backward pass computes gradients."""
    model = nn.GRU(input_size=16, hidden_size=32, num_layers=1, batch_first=False)
    model = gpu.to_applegpu(model)
    x = ApplegpuTensor.from_torch(torch.randn(5, 2, 16, requires_grad=True))
    output, _ = model(x)
    loss = output.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_lstm_training_loss_decreases():
    """LSTM training loop shows loss decrease over 10 steps."""
    torch.manual_seed(42)
    model = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = ApplegpuTensor.from_torch(torch.randn(4, 5, 8))
    target = ApplegpuTensor.from_torch(torch.randn(4, 5, 16))

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        output, _ = model(x)
        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.to_torch_cpu().item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
