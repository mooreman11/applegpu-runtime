"""Tests for GPU-composed vector norm."""
import numpy as np
import pytest
import torch
import applegpu_runtime as gpu
from applegpu_runtime.torch_backend import enable, to_applegpu


@pytest.fixture(scope="module", autouse=True)
def init():
    enable()


def test_l2_norm():
    x = torch.tensor([3.0, 4.0])
    x_gpu = to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu).to_torch_cpu().item()
    assert abs(result - 5.0) < 1e-4


def test_l1_norm():
    x = torch.tensor([-3.0, 4.0])
    x_gpu = to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=1.0).to_torch_cpu().item()
    assert abs(result - 7.0) < 1e-4


def test_l2_with_dim():
    np.random.seed(42)
    x = torch.randn(3, 4)
    x_gpu = to_applegpu(x)
    result_gpu = torch.linalg.vector_norm(x_gpu, dim=1).to_torch_cpu()
    result_cpu = torch.linalg.vector_norm(x, dim=1)
    np.testing.assert_allclose(result_gpu.numpy(), result_cpu.numpy(), atol=1e-4)


def test_linf_fallback():
    """L-infinity falls back to CPU -- should still work."""
    x = torch.tensor([1.0, -5.0, 3.0])
    x_gpu = to_applegpu(x)
    result = torch.linalg.vector_norm(x_gpu, ord=float('inf')).to_torch_cpu().item()
    assert abs(result - 5.0) < 1e-4
