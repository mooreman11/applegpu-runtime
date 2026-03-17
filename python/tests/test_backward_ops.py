"""Tests for Metal backward ops — verifies GPU path matches PyTorch CPU reference."""
import numpy as np
import pytest
import torch
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


class TestGeluBackward:
    def test_matches_pytorch(self):
        np.random.seed(42)
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        grad_np = np.random.randn(2, 3, 4).astype(np.float32)
        # PyTorch reference (tanh approximation)
        x_t = torch.tensor(x_np, requires_grad=True)
        y = torch.nn.functional.gelu(x_t, approximate="tanh")
        y.backward(torch.tensor(grad_np))
        expected = x_t.grad.numpy()
        # Our GPU
        result = _to_numpy(gpu.gelu_backward(gpu.from_numpy(grad_np), gpu.from_numpy(x_np)))
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_basic_values(self):
        grad = np.ones(3, dtype=np.float32)
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        result = _to_numpy(gpu.gelu_backward(gpu.from_numpy(grad), gpu.from_numpy(x)))
        # gelu'(0) ~ 0.5, gelu'(1) ~ 1.083, gelu'(-1) ~ -0.083
        assert abs(result[0] - 0.5) < 0.01
        assert abs(result[1] - 1.083) < 0.02
        assert abs(result[2] - (-0.083)) < 0.02


class TestConv1dBackwardInput:
    def test_matches_pytorch(self):
        np.random.seed(42)
        x = torch.randn(1, 3, 16, requires_grad=True)
        w = torch.randn(8, 3, 3)
        y = torch.nn.functional.conv1d(x, w, stride=1, padding=1)
        grad = torch.randn_like(y)
        y.backward(grad)
        expected = x.grad.numpy()
        result = _to_numpy(gpu.conv1d_backward_input(
            gpu.from_numpy(grad.detach().numpy()), gpu.from_numpy(w.numpy()),
            in_channels=3, in_len=16, stride=1, padding=1))
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_stride2(self):
        np.random.seed(42)
        x = torch.randn(2, 4, 32, requires_grad=True)
        w = torch.randn(8, 4, 3)
        y = torch.nn.functional.conv1d(x, w, stride=2, padding=1)
        grad = torch.randn_like(y)
        y.backward(grad)
        expected = x.grad.numpy()
        result = _to_numpy(gpu.conv1d_backward_input(
            gpu.from_numpy(grad.detach().numpy()), gpu.from_numpy(w.numpy()),
            in_channels=4, in_len=32, stride=2, padding=1))
        np.testing.assert_allclose(result, expected, atol=1e-4)
