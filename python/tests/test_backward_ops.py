"""Tests for Metal backward ops — verifies GPU path matches PyTorch CPU reference."""
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


class TestThresholdBackward:
    def test_basic(self):
        grad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        inp = np.array([-1.0, 0.5, -0.5, 2.0], dtype=np.float32)
        result = _to_numpy(gpu.threshold_backward(gpu.from_numpy(grad), gpu.from_numpy(inp), 0.0))
        expected = grad * (inp > 0).astype(np.float32)
        np.testing.assert_allclose(result, expected)

    def test_3d(self):
        np.random.seed(42)
        grad = np.random.randn(2, 3, 4).astype(np.float32)
        inp = np.random.randn(2, 3, 4).astype(np.float32)
        result = _to_numpy(gpu.threshold_backward(gpu.from_numpy(grad), gpu.from_numpy(inp), 0.0))
        expected = grad * (inp > 0).astype(np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_nonzero_threshold(self):
        grad = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        inp = np.array([-1.0, 0.5, 1.5, 2.0], dtype=np.float32)
        result = _to_numpy(gpu.threshold_backward(gpu.from_numpy(grad), gpu.from_numpy(inp), 1.0))
        expected = grad * (inp > 1.0).astype(np.float32)
        np.testing.assert_allclose(result, expected)


class TestTanhBackward:
    def test_basic(self):
        grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        output = np.array([0.5, -0.3, 0.9], dtype=np.float32)
        result = _to_numpy(gpu.tanh_backward(gpu.from_numpy(grad), gpu.from_numpy(output)))
        expected = grad * (1 - output ** 2)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_3d(self):
        np.random.seed(42)
        grad = np.random.randn(2, 3, 4).astype(np.float32)
        output = np.tanh(np.random.randn(2, 3, 4)).astype(np.float32)
        result = _to_numpy(gpu.tanh_backward(gpu.from_numpy(grad), gpu.from_numpy(output)))
        expected = grad * (1 - output ** 2)
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestSigmoidBackward:
    def test_basic(self):
        grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        output = np.array([0.5, 0.3, 0.8], dtype=np.float32)
        result = _to_numpy(gpu.sigmoid_backward(gpu.from_numpy(grad), gpu.from_numpy(output)))
        expected = grad * output * (1 - output)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_3d(self):
        np.random.seed(42)
        grad = np.random.randn(2, 3, 4).astype(np.float32)
        output = (1 / (1 + np.exp(-np.random.randn(2, 3, 4)))).astype(np.float32)
        result = _to_numpy(gpu.sigmoid_backward(gpu.from_numpy(grad), gpu.from_numpy(output)))
        expected = grad * output * (1 - output)
        np.testing.assert_allclose(result, expected, atol=1e-5)
