"""Tests for conv2d_backward_weight Metal GPU kernel."""
import applegpu_runtime as gpu
import torch


def test_conv2d_grad_weight():
    """Basic 3x3 kernel with stride=1, padding=0."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    w = torch.randn(4, 3, 3, 3, requires_grad=True)
    grad_out = torch.randn(2, 4, 6, 6)
    y = torch.nn.functional.conv2d(x, w)
    y.backward(grad_out)
    expected_gw = w.grad.clone()
    # GPU via native API
    x_gpu = gpu.from_torch(x.detach())
    go_gpu = gpu.from_torch(grad_out)
    gw_gpu = gpu.conv2d_backward_weight(go_gpu, x_gpu, 3, 3, 4, 3, 1, 1, 0, 0)
    result = gw_gpu.to_torch().float()
    assert torch.allclose(result, expected_gw, atol=1e-4), f"max diff: {(result - expected_gw).abs().max()}"


def test_conv2d_grad_weight_nonsquare():
    """Non-square kernel (3x4) with non-square input."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 2, 6, 8, requires_grad=True)
    w = torch.randn(5, 2, 3, 4, requires_grad=True)
    y = torch.nn.functional.conv2d(x, w)
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    expected_gw = w.grad.clone()
    x_gpu = gpu.from_torch(x.detach())
    go_gpu = gpu.from_torch(grad_out)
    gw_gpu = gpu.conv2d_backward_weight(go_gpu, x_gpu, 3, 4, 5, 2, 1, 1, 0, 0)
    result = gw_gpu.to_torch().float()
    assert torch.allclose(result, expected_gw, atol=1e-4), f"max diff: {(result - expected_gw).abs().max()}"


def test_conv2d_grad_weight_with_padding():
    """3x3 kernel with padding=1."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    w = torch.randn(4, 3, 3, 3, requires_grad=True)
    y = torch.nn.functional.conv2d(x, w, padding=1)
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    expected_gw = w.grad.clone()
    x_gpu = gpu.from_torch(x.detach())
    go_gpu = gpu.from_torch(grad_out)
    gw_gpu = gpu.conv2d_backward_weight(go_gpu, x_gpu, 3, 3, 4, 3, 1, 1, 1, 1)
    result = gw_gpu.to_torch().float()
    assert torch.allclose(result, expected_gw, atol=1e-4), f"max diff: {(result - expected_gw).abs().max()}"


def test_conv2d_grad_weight_with_stride():
    """3x3 kernel with stride=2."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 2, 10, 10, requires_grad=True)
    w = torch.randn(3, 2, 3, 3, requires_grad=True)
    y = torch.nn.functional.conv2d(x, w, stride=2)
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    expected_gw = w.grad.clone()
    x_gpu = gpu.from_torch(x.detach())
    go_gpu = gpu.from_torch(grad_out)
    gw_gpu = gpu.conv2d_backward_weight(go_gpu, x_gpu, 3, 3, 3, 2, 2, 2, 0, 0)
    result = gw_gpu.to_torch().float()
    assert torch.allclose(result, expected_gw, atol=1e-4), f"max diff: {(result - expected_gw).abs().max()}"


def test_conv2d_grad_weight_via_autograd():
    """End-to-end test via torch autograd with the torch backend enabled."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    w = torch.randn(4, 3, 3, 3, requires_grad=True)
    b = torch.randn(4, requires_grad=True)

    # Reference on CPU
    x_cpu = x.detach().clone().requires_grad_(True)
    w_cpu = w.detach().clone().requires_grad_(True)
    b_cpu = b.detach().clone().requires_grad_(True)
    y_cpu = torch.nn.functional.conv2d(x_cpu, w_cpu, b_cpu)
    loss_cpu = y_cpu.sum()
    loss_cpu.backward()

    # GPU via torch backend
    y_gpu = torch.nn.functional.conv2d(x, w, b)
    loss_gpu = y_gpu.sum()
    loss_gpu.backward()

    # Compare weight gradients
    gw_gpu = w.grad.detach()
    gw_cpu = w_cpu.grad.detach()
    if hasattr(gw_gpu, 'to_torch_cpu'):
        gw_gpu = gw_gpu.to_torch_cpu()
    assert torch.allclose(gw_gpu.float(), gw_cpu.float(), atol=1e-4), \
        f"weight grad max diff: {(gw_gpu.float() - gw_cpu.float()).abs().max()}"
