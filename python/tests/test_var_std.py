"""Tests for var and std ops."""
import math
import applegpu_runtime as gpu


def test_var_sample():
    """Variance with Bessel's correction (default)."""
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0], shape=[1, 5])
    result = gpu.var(a).to_list()
    assert abs(result[0] - 2.5) < 1e-5


def test_var_population():
    """Variance without Bessel's correction."""
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0], shape=[1, 5])
    result = gpu.var(a, 0).to_list()
    assert abs(result[0] - 2.0) < 1e-5


def test_var_multi_row():
    """Variance across multiple rows."""
    gpu.init_backend()
    # [[1,2,3], [4,4,4]]
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 4.0, 4.0], shape=[2, 3])
    result = gpu.var(a, 1).to_list()
    # Row 0: var([1,2,3], ddof=1) = 1.0
    # Row 1: var([4,4,4], ddof=1) = 0.0
    assert abs(result[0] - 1.0) < 1e-5
    assert abs(result[1] - 0.0) < 1e-5


def test_std_basic():
    """Standard deviation = sqrt(variance)."""
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0], shape=[1, 5])
    result = gpu.std_dev(a).to_list()
    assert abs(result[0] - math.sqrt(2.5)) < 1e-4


def test_var_method():
    """GpuTensor.var() method works."""
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[1, 4])
    result = a.var().to_list()
    # var([1,2,3,4], ddof=1) = 5/3 ≈ 1.6667
    assert abs(result[0] - 5.0 / 3.0) < 1e-4


def test_std_method():
    """GpuTensor.std() method works."""
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[1, 4])
    result = a.std().to_list()
    assert abs(result[0] - math.sqrt(5.0 / 3.0)) < 1e-4


def test_var_torch_dispatch():
    """torch.var dispatches to Metal."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    x = gpu.to_applegpu(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))
    v = torch.var(x)
    assert abs(v.to_torch_cpu().item() - 2.5) < 1e-4


def test_std_torch_dispatch():
    """torch.std dispatches to Metal."""
    gpu.init_backend()
    gpu.enable_torch_backend()
    import torch
    x = gpu.to_applegpu(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))
    s = torch.std(x)
    assert abs(s.to_torch_cpu().item() - math.sqrt(2.5)) < 1e-3
