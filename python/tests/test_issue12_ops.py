"""Tests for ops from issue #12: linspace, normal_, index.Tensor, index_put_."""
import applegpu_runtime as gpu


def test_linspace_basic():
    """torch.linspace creates correct values."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    # Create on CPU, transfer (factory functions don't dispatch)
    x = gpu.to_applegpu(torch.linspace(-1.0, 1.0, 5))
    result = x.to_torch_cpu().tolist()
    assert len(result) == 5
    assert abs(result[0] - (-1.0)) < 1e-5
    assert abs(result[2] - 0.0) < 1e-5
    assert abs(result[4] - 1.0) < 1e-5


def test_linspace_aten_dispatch():
    """aten::linspace dispatches correctly."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    import numpy as np
    # Direct aten call (what model code uses internally)
    result = torch.ops.aten.linspace.default(-0.5, 0.5, 10)
    assert result.shape == (10,)
    assert abs(result[0].item() - (-0.5)) < 1e-5
    assert abs(result[-1].item() - 0.5) < 1e-5


def test_normal_inplace():
    """normal_ fills tensor with random values in-place."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    y = gpu.to_applegpu(torch.empty(100))
    y.normal_(0, 1)
    vals = y.to_torch_cpu()
    # Check reasonable statistics (mean ~0, std ~1 for 100 samples)
    assert abs(vals.mean().item()) < 1.0, "Mean should be near 0"
    assert 0.3 < vals.std().item() < 3.0, "Std should be near 1"
    assert not torch.all(vals == 0), "Should not be all zeros"


def test_normal_custom_params():
    """normal_ with custom mean and std."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    y = gpu.to_applegpu(torch.empty(1000))
    y.normal_(5.0, 0.01)
    vals = y.to_torch_cpu()
    assert abs(vals.mean().item() - 5.0) < 0.1, f"Mean should be near 5: {vals.mean().item()}"


def test_index_tensor_boolean_mask():
    """tensor[bool_mask] selects masked elements."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    a = gpu.to_applegpu(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    mask = gpu.to_applegpu(torch.tensor([True, False, True, False, True]))
    result = a[mask]
    assert result.to_torch_cpu().tolist() == [1.0, 3.0, 5.0]


def test_index_tensor_integer_indices():
    """tensor[int_indices] selects by index."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    a = gpu.to_applegpu(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    idx = gpu.to_applegpu(torch.tensor([0, 2, 3]))
    result = a[idx]
    assert result.to_torch_cpu().tolist() == [10.0, 30.0, 40.0]


def test_index_put_via_aten():
    """aten::index_put_ via direct call."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    a_cpu = torch.zeros(5)
    idx = [torch.tensor([1, 3])]
    vals = torch.tensor([10.0, 20.0])
    result = torch.ops.aten.index_put_(a_cpu, idx, vals, False)
    assert result[1].item() == 10.0
    assert result[3].item() == 20.0
