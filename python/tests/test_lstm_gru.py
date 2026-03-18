"""Validate LSTM and GRU work via PyTorch decomposition on applegpu Metal backend.

PyTorch auto-decomposes nn.LSTM/nn.GRU into matmul + sigmoid + tanh + element-wise
ops, all of which have GPU kernels. This test validates the decomposition path works
end-to-end for forward and backward passes.

See: https://github.com/mooreman11/applegpu-runtime/issues/10
"""
import os
import tempfile

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


def test_gru_forward_batch_first():
    """GRU forward pass with batch_first=True produces correct shapes."""
    model = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
    model = gpu.to_applegpu(model)
    x = ApplegpuTensor.from_torch(torch.randn(4, 10, 32))
    output, h_n = model(x)
    assert output.shape == (4, 10, 64)
    assert h_n.shape == (2, 4, 64)
    out_cpu = output.to_torch_cpu()
    assert torch.isfinite(out_cpu).all()


def test_gru_backward_batch_first():
    """GRU backward pass with batch_first=True computes gradients."""
    model = nn.GRU(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
    model = gpu.to_applegpu(model)
    x = ApplegpuTensor.from_torch(torch.randn(2, 5, 16, requires_grad=True))
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


def test_gru_training_batch_first():
    """GRU training with batch_first=True and Adam optimizer."""
    torch.manual_seed(42)
    model = nn.GRU(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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


def test_save_load_checkpoint():
    """torch.save/load works with applegpu model state dicts."""
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    model = gpu.to_applegpu(model)

    # Train a few steps so weights diverge from init
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = ApplegpuTensor.from_torch(torch.randn(2, 8))
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    # Save directly — no manual .to_torch_cpu() needed
    path = tempfile.mktemp(suffix='.pt')
    try:
        torch.save(model.state_dict(), path)

        # Load into a fresh model
        model2 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        model2.load_state_dict(torch.load(path, weights_only=False))
        model2 = gpu.to_applegpu(model2)

        # Verify outputs match
        out1 = model(x).to_torch_cpu()
        out2 = model2(x).to_torch_cpu()
        assert (out1 - out2).abs().max().item() < 1e-5
    finally:
        os.unlink(path)


def test_save_load_lstm_checkpoint():
    """torch.save/load works with LSTM model checkpoints."""
    model = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
    model = gpu.to_applegpu(model)

    path = tempfile.mktemp(suffix='.pt')
    try:
        torch.save(model.state_dict(), path)
        model2 = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
        model2.load_state_dict(torch.load(path, weights_only=False))
        model2 = gpu.to_applegpu(model2)

        x = ApplegpuTensor.from_torch(torch.randn(2, 3, 8))
        out1, _ = model(x)
        out2, _ = model2(x)
        diff = (out1.to_torch_cpu() - out2.to_torch_cpu()).abs().max().item()
        assert diff < 1e-5
    finally:
        os.unlink(path)
