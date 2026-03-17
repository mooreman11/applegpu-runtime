"""Tests for grouped convolution support (groups > 1)."""
import applegpu_runtime as gpu
import torch


def test_conv2d_groups2():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 4, 8, 8)
    w = torch.randn(4, 2, 3, 3)  # groups=2: 4 out, 4/2=2 ic per group
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv2d(x_gpu, gpu.to_applegpu(w), groups=2)
    expected = torch.nn.functional.conv2d(x, w, groups=2)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_depthwise_conv2d():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 3, 8, 8)
    w = torch.randn(3, 1, 3, 3)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv2d(x_gpu, gpu.to_applegpu(w), groups=3)
    expected = torch.nn.functional.conv2d(x, w, groups=3)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_conv1d_groups():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 4, 16)
    w = torch.randn(4, 2, 3)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv1d(x_gpu, gpu.to_applegpu(w), groups=2)
    expected = torch.nn.functional.conv1d(x, w, groups=2)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_conv2d_groups1_regression():
    """Ensure groups=1 (default) still works correctly."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv2d(x_gpu, gpu.to_applegpu(w))
    expected = torch.nn.functional.conv2d(x, w)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_conv2d_groups4_batched():
    """Test with batch > 1 and groups=4."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(2, 8, 6, 6)
    w = torch.randn(8, 2, 3, 3)  # groups=4: 8 out, 8/4=2 ic per group
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv2d(x_gpu, gpu.to_applegpu(w), groups=4)
    expected = torch.nn.functional.conv2d(x, w, groups=4)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_depthwise_conv1d():
    """Depthwise conv1d: groups == in_channels."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 4, 16)
    w = torch.randn(4, 1, 3)  # depthwise: groups=4
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv1d(x_gpu, gpu.to_applegpu(w), groups=4)
    expected = torch.nn.functional.conv1d(x, w, groups=4)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_conv2d_groups2_with_bias():
    """Grouped conv with bias."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 4, 8, 8)
    w = torch.randn(4, 2, 3, 3)
    b = torch.randn(4)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv2d(x_gpu, gpu.to_applegpu(w), bias=gpu.to_applegpu(b), groups=2)
    expected = torch.nn.functional.conv2d(x, w, bias=b, groups=2)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_conv2d_groups2_with_padding():
    """Grouped conv with padding."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 4, 8, 8)
    w = torch.randn(4, 2, 3, 3)
    x_gpu = gpu.to_applegpu(x)
    result = torch.nn.functional.conv2d(x_gpu, gpu.to_applegpu(w), padding=1, groups=2)
    expected = torch.nn.functional.conv2d(x, w, padding=1, groups=2)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)
