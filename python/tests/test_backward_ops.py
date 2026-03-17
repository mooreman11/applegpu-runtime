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


class TestMaxPool2dBackward:
    def test_matches_pytorch(self):
        """Non-overlapping pools (stride == kernel_size) — basic scatter."""
        x = torch.randn(1, 2, 4, 4, requires_grad=True)
        pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        y, indices = pool(x)
        grad = torch.randn_like(y)
        y.backward(grad)
        expected = x.grad.numpy()
        result = _to_numpy(gpu.max_pool2d_backward(
            gpu.from_numpy(grad.detach().numpy()),
            gpu.from_numpy(indices.numpy().astype(np.int32)),
            batch=1, channels=2, in_h=4, in_w=4))
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_overlapping_pools(self):
        """stride < kernel_size — multiple outputs can map to same input (needs atomics)."""
        x = torch.randn(1, 1, 6, 6, requires_grad=True)
        pool = torch.nn.MaxPool2d(3, stride=2, padding=0, return_indices=True)
        y, indices = pool(x)
        grad = torch.ones_like(y)
        y.backward(grad)
        expected = x.grad.numpy()
        result = _to_numpy(gpu.max_pool2d_backward(
            gpu.from_numpy(grad.detach().numpy()),
            gpu.from_numpy(indices.numpy().astype(np.int32)),
            batch=1, channels=1, in_h=6, in_w=6))
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_batched_multichannel(self):
        """Multi-batch, multi-channel with overlapping pools."""
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        pool = torch.nn.MaxPool2d(3, stride=2, padding=1, return_indices=True)
        y, indices = pool(x)
        grad = torch.randn_like(y)
        y.backward(grad)
        expected = x.grad.numpy()
        result = _to_numpy(gpu.max_pool2d_backward(
            gpu.from_numpy(grad.detach().numpy()),
            gpu.from_numpy(indices.numpy().astype(np.int32)),
            batch=2, channels=3, in_h=8, in_w=8))
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestTrainingIntegration:
    """Verify backward ops work in a real training loop."""

    def test_mlp_training_step(self):
        """MLP with ReLU + GELU + tanh + sigmoid — exercises all 4 element-wise backward ops."""
        from applegpu_runtime.torch_backend import enable, to_applegpu, set_eager_mode
        enable()
        set_eager_mode(True)

        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),        # threshold_backward
            torch.nn.Linear(16, 16),
            torch.nn.GELU(approximate="tanh"),  # gelu_backward
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),         # tanh_backward
            torch.nn.Linear(16, 4),
            torch.nn.Sigmoid(),      # sigmoid_backward
        )
        to_applegpu(model)

        x = torch.randn(2, 8)
        x_gpu = to_applegpu(x)
        target = torch.zeros(2, 4)
        target_gpu = to_applegpu(target)

        y = model(x_gpu)
        loss = ((y - target_gpu) ** 2).mean()
        loss_val = loss.to_torch_cpu().item()
        loss.backward()

        # Verify gradients exist and are non-zero
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            grad_cpu = param.grad.to_torch_cpu() if hasattr(param.grad, 'to_torch_cpu') else param.grad
            assert grad_cpu.abs().sum() > 0, f"Zero gradient for {name}"

        assert loss_val > 0, "Loss should be positive"
        set_eager_mode(False)

    def test_cnn_training_step(self):
        """CNN with max_pool2d — exercises max_pool2d_backward."""
        from applegpu_runtime.torch_backend import enable, to_applegpu, set_eager_mode
        enable()
        set_eager_mode(True)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),  # max_pool2d_backward
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 4 * 4, 2),
        )
        to_applegpu(model)

        x = torch.randn(2, 1, 8, 8)
        x_gpu = to_applegpu(x)
        target = torch.zeros(2, 2)
        target_gpu = to_applegpu(target)

        y = model(x_gpu)
        loss = ((y - target_gpu) ** 2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

        set_eager_mode(False)

    def test_conv1d_training_step(self):
        """Conv1d encoder — exercises conv1d_backward_input."""
        from applegpu_runtime.torch_backend import enable, to_applegpu, set_eager_mode
        enable()
        set_eager_mode(True)

        model = torch.nn.Sequential(
            torch.nn.Conv1d(8, 16, 3, padding=1),  # conv1d_backward_input
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 4, 3, padding=1),
            torch.nn.ReLU(),
        )
        to_applegpu(model)

        x = torch.randn(2, 8, 32)
        x_gpu = to_applegpu(x)

        y = model(x_gpu)
        loss = y.mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

        set_eager_mode(False)
