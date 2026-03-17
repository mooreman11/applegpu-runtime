# python/tests/test_vector_norm_gpu.py
import applegpu_runtime as gpu
import torch


def test_l2_norm_1d():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.tensor([3.0, 4.0])
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=2)
    assert abs(result.to_torch_cpu().item() - 5.0) < 1e-5


def test_l1_norm_1d():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.tensor([-3.0, 4.0])
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=1)
    assert abs(result.to_torch_cpu().item() - 7.0) < 1e-5


def test_l2_norm_2d_dim():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4)
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=2, dim=1)
    expected = torch.linalg.vector_norm(x, ord=2, dim=1)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_l2_norm_keepdim():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(3, 4)
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=2, dim=1, keepdim=True)
    expected = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
    assert result.to_torch_cpu().shape == expected.shape
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_linf_norm_cpu_fallback():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.tensor([-5.0, 3.0, 4.0])
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=float('inf'))
    assert abs(result.to_torch_cpu().item() - 5.0) < 1e-5


def test_l2_norm_single_element():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.tensor([3.0])
    x_gpu = gpu.to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=2)
    assert abs(result.to_torch_cpu().item() - 3.0) < 1e-5
