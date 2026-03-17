"""Tests for GPU→GPU blit copy."""
import applegpu_runtime as gpu
import torch


def test_blit_copy_gpu_to_gpu():
    """blit_copy transfers data between two GPU tensors without CPU roundtrip."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    src = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    dst = gpu.tensor([0.0, 0.0, 0.0, 0.0], shape=[2, 2])
    gpu.blit_copy(dst, src)
    result = dst.to_list()
    assert abs(result[0] - 1.0) < 1e-6
    assert abs(result[1] - 2.0) < 1e-6
    assert abs(result[2] - 3.0) < 1e-6
    assert abs(result[3] - 4.0) < 1e-6


def test_blit_copy_large():
    """blit_copy works for larger buffers."""
    gpu.init_backend()
    data = [float(i) for i in range(1024)]
    src = gpu.tensor(data, shape=[32, 32])
    dst = gpu.tensor([0.0] * 1024, shape=[32, 32])
    gpu.blit_copy(dst, src)
    result = dst.to_list()
    assert abs(result[0] - 0.0) < 1e-6
    assert abs(result[1023] - 1023.0) < 1e-6


def test_copy_op_gpu_tensor():
    """aten.copy_ uses blit for GPU→GPU when both are ApplegpuTensor."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    a_gpu = gpu.to_applegpu(a)
    b_gpu = gpu.to_applegpu(b)
    a_gpu.copy_(b_gpu)
    result = a_gpu.to_torch_cpu()
    assert torch.allclose(result, b, atol=1e-6)


def test_copy_op_cpu_to_gpu():
    """aten.copy_ still works for CPU→GPU (fallback path)."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    a = torch.randn(2, 3)
    a_gpu = gpu.to_applegpu(a)
    b_cpu = torch.randn(2, 3)
    a_gpu.copy_(b_cpu)
    result = a_gpu.to_torch_cpu()
    assert torch.allclose(result, b_cpu, atol=1e-6)
