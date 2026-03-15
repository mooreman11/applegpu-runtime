"""Tests for max_pool2d real indices and backward pass."""
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
    set_eager_mode(True)
    yield
    set_eager_mode(False)


def test_max_pool2d_indices_shape():
    """max_pool2d_with_indices returns indices with correct shape."""
    x = ApplegpuTensor.from_torch(torch.randn(1, 1, 4, 4))
    pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    output, indices = pool(x)
    # Output shape: (1, 1, 2, 2)
    assert output.shape == (1, 1, 2, 2)
    # Indices must have the same shape as output
    idx_shape = indices.shape if isinstance(indices, ApplegpuTensor) else indices.shape
    assert idx_shape == (1, 1, 2, 2), f"Indices shape {idx_shape} != (1, 1, 2, 2)"


def test_max_pool2d_indices_correct_values():
    """max_pool2d indices point to correct positions in the input."""
    # Deterministic input where we know which positions are max
    x_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [9.0, 10.0, 11.0, 12.0],
                              [13.0, 14.0, 15.0, 16.0]]]])
    x = ApplegpuTensor.from_torch(x_data)
    pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    output, indices = pool(x)

    out_cpu = output.to_torch_cpu() if isinstance(output, ApplegpuTensor) else output
    idx_cpu = indices.to_torch_cpu() if isinstance(indices, ApplegpuTensor) else indices

    # Max of each 2x2 window: 6, 8, 14, 16
    expected_out = torch.tensor([[[[6.0, 8.0], [14.0, 16.0]]]])
    assert torch.allclose(out_cpu, expected_out, atol=1e-4)

    # CPU reference for indices
    _, ref_indices = torch.nn.functional.max_pool2d_with_indices(
        x_data, kernel_size=2, stride=2
    )
    assert torch.equal(idx_cpu.to(torch.int64), ref_indices), (
        f"Indices mismatch: got {idx_cpu}, expected {ref_indices}"
    )


def test_max_pool2d_indices_match_cpu():
    """GPU indices match CPU indices on random data."""
    torch.manual_seed(42)
    x_data = torch.randn(2, 3, 8, 8)
    x = ApplegpuTensor.from_torch(x_data)

    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
    output, indices = pool(x)

    out_cpu = output.to_torch_cpu() if isinstance(output, ApplegpuTensor) else output
    idx_cpu = indices.to_torch_cpu() if isinstance(indices, ApplegpuTensor) else indices

    ref_out, ref_idx = torch.nn.functional.max_pool2d_with_indices(
        x_data, kernel_size=3, stride=2, padding=1
    )
    assert torch.allclose(out_cpu, ref_out, atol=1e-4)
    assert torch.equal(idx_cpu.to(torch.int64), ref_idx)


def test_max_pool2d_backward_gradient_shape():
    """max_pool2d backward produces gradient with correct shape."""
    x = ApplegpuTensor.from_torch(torch.randn(1, 1, 4, 4, requires_grad=True))
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    output = pool(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    grad = x.grad.to_torch_cpu() if isinstance(x.grad, ApplegpuTensor) else x.grad
    assert grad.shape == (1, 1, 4, 4)


def test_max_pool2d_backward_gradient_values():
    """max_pool2d backward routes gradients to correct (max) positions."""
    x_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [9.0, 10.0, 11.0, 12.0],
                              [13.0, 14.0, 15.0, 16.0]]]], requires_grad=True)
    x = ApplegpuTensor.from_torch(x_data)
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    output = pool(x)
    loss = output.sum()
    loss.backward()

    grad = x.grad.to_torch_cpu() if isinstance(x.grad, ApplegpuTensor) else x.grad

    # CPU reference
    x_ref = x_data.clone().detach().requires_grad_(True)
    ref_out = nn.MaxPool2d(kernel_size=2, stride=2)(x_ref)
    ref_out.sum().backward()

    assert torch.allclose(grad, x_ref.grad, atol=1e-4), (
        f"Gradient mismatch:\ngot:    {grad}\nexpect: {x_ref.grad}"
    )


def test_max_pool2d_backward_random():
    """max_pool2d backward matches CPU on random data."""
    torch.manual_seed(123)
    x_data = torch.randn(2, 3, 8, 8, requires_grad=True)
    x = ApplegpuTensor.from_torch(x_data)

    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    output = pool(x)
    loss = output.sum()
    loss.backward()

    grad = x.grad.to_torch_cpu() if isinstance(x.grad, ApplegpuTensor) else x.grad

    x_ref = x_data.clone().detach().requires_grad_(True)
    ref_out = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x_ref)
    ref_out.sum().backward()

    assert torch.allclose(grad, x_ref.grad, atol=1e-4), "Backward gradient mismatch"


def test_resnet_block_with_maxpool_training():
    """A ResNet-like block with maxpool can do a training step."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )
    model.eval()  # BN in eval mode (inference path we support)
    model = gpu.to_applegpu(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = ApplegpuTensor.from_torch(torch.randn(2, 3, 8, 8))
    target = ApplegpuTensor.from_torch(torch.randn(2, 10))

    losses = []
    for step in range(3):
        optimizer.zero_grad()
        output = model(x)
        diff = output - target
        loss = (diff * diff).mean()
        loss_val = loss.to_torch_cpu().item() if isinstance(loss, ApplegpuTensor) else loss.item()
        losses.append(loss_val)
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"
