"""Tests for GPU->GPU blit copy."""
import numpy as np
import pytest
import applegpu_runtime as gpu


@pytest.fixture(scope="module", autouse=True)
def init():
    gpu.init_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)


def _to_numpy(t):
    t.eval()
    return np.array(t.to_list()).reshape(t.shape)


def test_basic_blit_copy():
    src = gpu.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    dst = gpu.from_numpy(np.zeros(4, dtype=np.float32))
    gpu.blit_copy(src, dst)
    result = _to_numpy(dst)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])


def test_large_blit_copy():
    data = np.arange(1024, dtype=np.float32)
    src = gpu.from_numpy(data)
    dst = gpu.from_numpy(np.zeros(1024, dtype=np.float32))
    gpu.blit_copy(src, dst)
    result = _to_numpy(dst)
    np.testing.assert_allclose(result, data)


def test_blit_copy_3d():
    data = np.random.randn(2, 3, 4).astype(np.float32)
    src = gpu.from_numpy(data)
    dst = gpu.from_numpy(np.zeros_like(data))
    gpu.blit_copy(src, dst)
    result = _to_numpy(dst)
    np.testing.assert_allclose(result, data)
