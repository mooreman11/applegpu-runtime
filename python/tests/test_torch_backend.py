"""Tests for PyTorch custom device backend."""

import warnings

import pytest
import numpy as np

torch = pytest.importorskip("torch")

import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def setup():
    gpu.init_backend()
    gpu.enable_torch_backend()


def test_enable_backend():
    """Backend registration succeeds."""
    assert hasattr(torch.Tensor, "is_applegpu")


def test_tensor_to_applegpu():
    """CPU tensor can move to applegpu."""
    from applegpu_runtime.torch_backend import ApplegpuTensor
    t = torch.tensor([1.0, 2.0, 3.0])
    a = ApplegpuTensor.from_torch(t)
    assert isinstance(a, ApplegpuTensor)
    assert a.shape == (3,)


def test_tensor_roundtrip():
    """CPU -> applegpu -> CPU preserves data."""
    from applegpu_runtime.torch_backend import ApplegpuTensor
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    a = ApplegpuTensor.from_torch(t)
    result = a.to_torch_cpu()
    assert torch.allclose(result, t)


def test_tensor_2d_roundtrip():
    """2D tensor roundtrip."""
    from applegpu_runtime.torch_backend import ApplegpuTensor
    t = torch.randn(3, 4)
    a = ApplegpuTensor.from_torch(t)
    result = a.to_torch_cpu()
    assert result.shape == (3, 4)
    assert torch.allclose(result, t, atol=1e-6)


def test_detach():
    """Detach returns an ApplegpuTensor."""
    from applegpu_runtime.torch_backend import ApplegpuTensor
    t = torch.tensor([1.0, 2.0, 3.0])
    a = ApplegpuTensor.from_torch(t)
    d = a.detach()
    assert isinstance(d, ApplegpuTensor)


def test_clone():
    """Clone creates a copy."""
    from applegpu_runtime.torch_backend import ApplegpuTensor
    t = torch.tensor([1.0, 2.0, 3.0])
    a = ApplegpuTensor.from_torch(t)
    c = a.clone()
    assert isinstance(c, ApplegpuTensor)
    assert torch.allclose(c.to_torch_cpu(), t)


def test_cpu_fallback_warns():
    """Unsupported op warns and falls back to CPU."""
    from applegpu_runtime.torch_backend import ApplegpuTensor, _warned_ops
    # Clear warned ops so we get fresh warnings
    _warned_ops.clear()
    t = torch.tensor([1.0, 2.0, 3.0])
    a = ApplegpuTensor.from_torch(t)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # sin is not in our op set — should fall back to CPU
        result = torch.sin(a)
        # Should get a warning
        assert any("not supported" in str(warning.message) for warning in w)
    assert isinstance(result, ApplegpuTensor)
    expected = torch.sin(t)
    assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-5)


def test_repr():
    """ApplegpuTensor has a readable repr."""
    from applegpu_runtime.torch_backend import ApplegpuTensor
    t = torch.tensor([1.0, 2.0])
    a = ApplegpuTensor.from_torch(t)
    r = repr(a)
    assert "applegpu" in r
    assert "float32" in r
