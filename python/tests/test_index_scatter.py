import applegpu_runtime as gpu
import torch


def test_index_put_no_accumulate():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.zeros(5, 3)
    idx = torch.tensor([0, 2, 4], dtype=torch.int32)
    vals = torch.ones(3, 3)
    x_gpu = gpu.to_applegpu(x)
    torch.ops.aten.index_put_(x_gpu, [gpu.to_applegpu(idx)], gpu.to_applegpu(vals))
    expected = x.clone()
    expected.index_put_([idx.long()], vals)
    assert torch.allclose(x_gpu.to_torch_cpu(), expected, atol=1e-6)


def test_index_put_accumulate():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.zeros(3, 2)
    idx = torch.tensor([0, 0, 1], dtype=torch.int32)
    vals = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_gpu = gpu.to_applegpu(x)
    torch.ops.aten.index_put_(x_gpu, [gpu.to_applegpu(idx)], gpu.to_applegpu(vals), accumulate=True)
    expected = x.clone()
    expected.index_put_([idx.long()], vals, accumulate=True)
    assert torch.allclose(x_gpu.to_torch_cpu(), expected, atol=1e-5)


def test_index_tensor_integer():
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(5, 3)
    idx = torch.tensor([0, 2, 4], dtype=torch.int32)
    x_gpu = gpu.to_applegpu(x)
    result = torch.ops.aten.index.Tensor(x_gpu, [gpu.to_applegpu(idx)])
    expected = x[idx.long()]
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-6)


def test_index_tensor_boolean_fallback():
    """Boolean mask should still work via CPU fallback."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    x = torch.randn(5, 3)
    mask = torch.tensor([True, False, True, False, True])
    x_gpu = gpu.to_applegpu(x)
    result = torch.ops.aten.index.Tensor(x_gpu, [gpu.to_applegpu(mask)])
    expected = x[mask]
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-6)
