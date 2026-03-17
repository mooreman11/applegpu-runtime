"""Tests for exact GELU forward + backward kernels (#18)."""

import applegpu_runtime as gpu
import torch


def test_gelu_exact_forward():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.gelu(x_gpu, approximate="none")
    expected = torch.nn.functional.gelu(x, approximate="none")
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_gelu_tanh_forward_unchanged():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.gelu(x_gpu, approximate="tanh")
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_gelu_exact_backward():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4)
    grad = torch.randn(3, 4)
    x_gpu = gpu.to_applegpu(x)
    grad_gpu = gpu.to_applegpu(grad)
    result = torch.ops.aten.gelu_backward(grad_gpu, x_gpu, approximate="none")
    expected = torch.ops.aten.gelu_backward(grad, x, approximate="none")
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-4)


def test_gelu_tanh_backward_gpu():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4)
    grad = torch.randn(3, 4)
    x_gpu = gpu.to_applegpu(x)
    grad_gpu = gpu.to_applegpu(grad)
    result = torch.ops.aten.gelu_backward(grad_gpu, x_gpu, approximate="tanh")
    expected = torch.ops.aten.gelu_backward(grad, x, approximate="tanh")
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-4)
