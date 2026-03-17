import applegpu_runtime as gpu
import torch


def test_max_pool2d_indices_match_cpu():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(1, 1, 4, 4)
    x_gpu = gpu.to_applegpu(x)
    values_gpu, indices_gpu = torch.nn.functional.max_pool2d_with_indices(x_gpu, 2, stride=2)
    values_cpu, indices_cpu = torch.nn.functional.max_pool2d_with_indices(x, 2, stride=2)
    assert torch.allclose(values_gpu.to_torch_cpu(), values_cpu, atol=1e-5)
    assert torch.equal(indices_gpu.to_torch_cpu(), indices_cpu)


def test_max_pool2d_indices_with_padding():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(2, 3, 8, 8)
    x_gpu = gpu.to_applegpu(x)
    values_gpu, indices_gpu = torch.nn.functional.max_pool2d_with_indices(
        x_gpu, kernel_size=3, stride=2, padding=1
    )
    values_cpu, indices_cpu = torch.nn.functional.max_pool2d_with_indices(
        x, kernel_size=3, stride=2, padding=1
    )
    assert torch.allclose(values_gpu.to_torch_cpu(), values_cpu, atol=1e-5)
    assert torch.equal(indices_gpu.to_torch_cpu(), indices_cpu)
