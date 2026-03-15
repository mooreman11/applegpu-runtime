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


# ============================================================
# Phase A tests (lifecycle)
# ============================================================

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
        # sin is not in our op set -- should fall back to CPU
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


# ============================================================
# Phase B tests (aten op dispatch)
# ============================================================

class TestElementwiseBinary:
    """Element-wise binary ops dispatched to GPU."""

    def test_add(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0, 3.0]))
        b = ApplegpuTensor.from_torch(torch.tensor([4.0, 5.0, 6.0]))
        c = a + b
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([5.0, 7.0, 9.0]))

    def test_sub(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([10.0, 20.0, 30.0]))
        b = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0, 3.0]))
        c = a - b
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([9.0, 18.0, 27.0]))

    def test_mul(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([2.0, 3.0, 4.0]))
        b = ApplegpuTensor.from_torch(torch.tensor([5.0, 6.0, 7.0]))
        c = a * b
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([10.0, 18.0, 28.0]))

    def test_div(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([10.0, 20.0, 30.0]))
        b = ApplegpuTensor.from_torch(torch.tensor([2.0, 4.0, 5.0]))
        c = a / b
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([5.0, 5.0, 6.0]))

    def test_add_with_alpha(self):
        """aten.add.Tensor with alpha kwarg: a + alpha * b."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0, 3.0]))
        b = ApplegpuTensor.from_torch(torch.tensor([1.0, 1.0, 1.0]))
        # torch.add(a, b, alpha=2) => a + 2*b
        c = torch.add(a, b, alpha=2)
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([3.0, 4.0, 5.0]))

    def test_broadcasting(self):
        """Binary op with broadcasting."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.ones(4, 3))
        b = ApplegpuTensor.from_torch(torch.tensor([10.0, 20.0, 30.0]))
        c = a + b
        assert c.shape == (4, 3)
        result = c.to_torch_cpu()
        assert torch.allclose(result[0], torch.tensor([11.0, 21.0, 31.0]))


class TestElementwiseUnary:
    """Element-wise unary ops dispatched to GPU."""

    def test_neg(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, -2.0, 3.0]))
        c = -a
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([-1.0, 2.0, -3.0]))

    def test_relu(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([-1.0, 2.0, -3.0, 4.0]))
        c = torch.relu(a)
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([0.0, 2.0, 0.0, 4.0]))

    def test_exp(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([0.0, 1.0, 2.0]))
        c = torch.exp(a)
        assert isinstance(c, ApplegpuTensor)
        expected = torch.exp(torch.tensor([0.0, 1.0, 2.0]))
        assert torch.allclose(c.to_torch_cpu(), expected, atol=1e-5)

    def test_log(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.718281828, 7.389056]))
        c = torch.log(a)
        assert isinstance(c, ApplegpuTensor)
        expected = torch.log(torch.tensor([1.0, 2.718281828, 7.389056]))
        assert torch.allclose(c.to_torch_cpu(), expected, atol=1e-4)

    def test_sqrt(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, 4.0, 9.0, 16.0]))
        c = torch.sqrt(a)
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_gelu(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
        c = torch.nn.functional.gelu(a)
        assert isinstance(c, ApplegpuTensor)
        expected = torch.nn.functional.gelu(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
        assert torch.allclose(c.to_torch_cpu(), expected, atol=2e-3)

    def test_abs(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([-1.0, 2.0, -3.0]))
        c = torch.abs(a)
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([1.0, 2.0, 3.0]))

    def test_sign(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([-5.0, 0.0, 3.0]))
        c = torch.sign(a)
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([-1.0, 0.0, 1.0]))

    def test_pow(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([2.0, 3.0, 4.0]))
        c = torch.pow(a, 2.0)
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([4.0, 9.0, 16.0]))

    def test_clamp(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, 5.0, 10.0]))
        c = torch.clamp(a, min=2.0, max=8.0)
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([2.0, 5.0, 8.0]))


class TestMatrixOps:
    """Matrix multiplication ops dispatched to GPU."""

    def test_matmul(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        b = ApplegpuTensor.from_torch(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        c = a @ b
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), expected)

    def test_mm(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        b = ApplegpuTensor.from_torch(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        c = torch.mm(a, b)
        assert isinstance(c, ApplegpuTensor)
        assert torch.allclose(c.to_torch_cpu(), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))


class TestShapeOps:
    """Shape manipulation ops dispatched to GPU."""

    def test_reshape(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.arange(6.0))
        b = a.reshape(2, 3)
        assert isinstance(b, ApplegpuTensor)
        assert b.shape == (2, 3)

    def test_view(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.arange(12.0))
        b = a.view(3, 4)
        assert isinstance(b, ApplegpuTensor)
        assert b.shape == (3, 4)

    def test_transpose(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        b = a.t()
        assert isinstance(b, ApplegpuTensor)
        assert b.shape == (3, 2)

    def test_transpose_dims(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.randn(2, 3, 4))
        b = a.transpose(0, 2)
        assert isinstance(b, ApplegpuTensor)
        assert b.shape == (4, 3, 2)

    def test_unsqueeze(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0, 3.0]))
        b = a.unsqueeze(0)
        assert isinstance(b, ApplegpuTensor)
        assert b.shape == (1, 3)

    def test_unsqueeze_neg_dim(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0, 3.0]))
        b = a.unsqueeze(-1)
        assert isinstance(b, ApplegpuTensor)
        assert b.shape == (3, 1)

    def test_squeeze(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.randn(1, 3))
        b = a.squeeze(0)
        assert isinstance(b, ApplegpuTensor)
        assert b.shape == (3,)


class TestReductions:
    """Reduction ops dispatched to GPU."""

    def test_softmax(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([[1.0, 2.0, 3.0]]))
        b = torch.softmax(a, dim=-1)
        result = b.to_torch_cpu()
        assert abs(result.sum().item() - 1.0) < 0.001

    def test_softmax_last_dim(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]))
        b = torch.softmax(a, dim=1)  # dim=1 is last dim for 2D
        result = b.to_torch_cpu()
        # Each row should sum to 1
        assert torch.allclose(result.sum(dim=1), torch.ones(2), atol=1e-3)


class TestTensorCreation:
    """Tensor creation ops dispatched to GPU."""

    def test_zeros(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0]))
        # Trigger zeros through an operation that creates them internally
        # Direct test: ensure zeros works via torch ops
        z = torch.zeros(3, 4)
        # This won't dispatch through our backend unless it's an ApplegpuTensor op
        # Instead test via the registry directly
        from applegpu_runtime.torch_backend import SUPPORTED_OPS
        result = SUPPORTED_OPS[torch.ops.aten.zeros.default]([2, 3])
        assert isinstance(result, ApplegpuTensor)
        assert result.shape == (2, 3)

    def test_ones(self):
        from applegpu_runtime.torch_backend import SUPPORTED_OPS, ApplegpuTensor
        result = SUPPORTED_OPS[torch.ops.aten.ones.default]([4])
        assert isinstance(result, ApplegpuTensor)
        assert result.shape == (4,)
        assert torch.allclose(result.to_torch_cpu(), torch.ones(4))


class TestCatSlice:
    """Concatenation and slicing ops."""

    def test_cat_2d(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        b = ApplegpuTensor.from_torch(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        c = torch.cat([a, b], dim=0)
        assert isinstance(c, ApplegpuTensor)
        assert c.shape == (4, 2)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        assert torch.allclose(c.to_torch_cpu(), expected)

    def test_select(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        # a[1] dispatches to aten.select.int
        b = a[1]
        assert isinstance(b, ApplegpuTensor)
        assert torch.allclose(b.to_torch_cpu(), torch.tensor([3.0, 4.0]))

    def test_slice_2d(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        # a[0:2] slices first 2 rows
        b = a[0:2]
        assert isinstance(b, ApplegpuTensor)
        assert b.shape == (2, 2)
        assert torch.allclose(b.to_torch_cpu(), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


class TestLayerOps:
    """Higher-level layer ops."""

    def test_layer_norm(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        x = ApplegpuTensor.from_torch(torch.ones(2, 4) * 5.0)
        gamma = torch.ones(4)
        beta = torch.zeros(4)
        result = torch.nn.functional.layer_norm(x, [4], gamma, beta)
        # Constant input -> normalized to 0
        assert isinstance(result, ApplegpuTensor)
        cpu_result = result.to_torch_cpu()
        # All same values -> after normalization, should be near 0
        assert torch.allclose(cpu_result, torch.zeros(2, 4), atol=1e-4)

    def test_layer_norm_nontrivial(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        x = ApplegpuTensor.from_torch(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        gamma = torch.ones(4)
        beta = torch.zeros(4)
        result = torch.nn.functional.layer_norm(x, [4], gamma, beta)
        assert isinstance(result, ApplegpuTensor)
        cpu_result = result.to_torch_cpu()
        # Mean should be ~0
        assert abs(cpu_result.mean().item()) < 0.01


class TestTensorCreationPhaseC:
    """Phase C: additional tensor creation ops."""

    def test_full(self):
        from applegpu_runtime.torch_backend import SUPPORTED_OPS, ApplegpuTensor
        result = SUPPORTED_OPS[torch.ops.aten.full.default]([2, 3], 7.0)
        assert isinstance(result, ApplegpuTensor)
        assert result.shape == (2, 3)
        assert torch.allclose(result.to_torch_cpu(), torch.full((2, 3), 7.0))

    def test_scalar_tensor(self):
        from applegpu_runtime.torch_backend import SUPPORTED_OPS, ApplegpuTensor
        result = SUPPORTED_OPS[torch.ops.aten.scalar_tensor.default](3.14)
        assert isinstance(result, ApplegpuTensor)
        assert abs(result.to_torch_cpu().item() - 3.14) < 1e-5

    def test_copy_applegpu_to_applegpu(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0, 3.0]))
        b = ApplegpuTensor.from_torch(torch.tensor([9.0, 8.0, 7.0]))
        from applegpu_runtime.torch_backend import SUPPORTED_OPS
        SUPPORTED_OPS[torch.ops.aten.copy_.default](a, b)
        assert torch.allclose(a.to_torch_cpu(), torch.tensor([9.0, 8.0, 7.0]))

    def test_copy_cpu_to_applegpu(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor, SUPPORTED_OPS
        a = ApplegpuTensor.from_torch(torch.tensor([0.0, 0.0]))
        src = torch.tensor([5.0, 6.0])
        SUPPORTED_OPS[torch.ops.aten.copy_.default](a, src)
        assert torch.allclose(a.to_torch_cpu(), torch.tensor([5.0, 6.0]))


# ============================================================
# Phase D tests (end-to-end)
# ============================================================

class TestEndToEndLinear:
    """End-to-end test: manual nn.Linear forward pass."""

    def test_nn_linear_manual(self):
        """Manual matmul + bias matching nn.Linear output."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        linear = torch.nn.Linear(4, 3, bias=True)
        x = torch.randn(2, 4)

        x_gpu = ApplegpuTensor.from_torch(x)
        w_gpu = ApplegpuTensor.from_torch(linear.weight.data)
        b_gpu = ApplegpuTensor.from_torch(linear.bias.data)

        # y = x @ w^T + b
        result = x_gpu @ w_gpu.t() + b_gpu
        expected = linear(x)
        assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-4)


class TestEndToEndModelParams:
    """End-to-end test: model parameter round-trip."""

    def test_model_params_roundtrip(self):
        """Model parameters can be moved to applegpu and back."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        model = torch.nn.Linear(4, 3)
        w_before = model.weight.data.clone()
        b_before = model.bias.data.clone()

        w_gpu = ApplegpuTensor.from_torch(model.weight.data)
        b_gpu = ApplegpuTensor.from_torch(model.bias.data)

        assert torch.allclose(w_gpu.to_torch_cpu(), w_before, atol=1e-6)
        assert torch.allclose(b_gpu.to_torch_cpu(), b_before, atol=1e-6)


class TestEndToEndMultiOpChain:
    """End-to-end test: multi-op chain dispatched to Metal."""

    def test_matmul_relu_softmax_chain(self):
        """Chain: matmul -> relu -> softmax."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.randn(3, 4))
        b = ApplegpuTensor.from_torch(torch.randn(4, 5))

        c = a @ b
        d = torch.relu(c)
        e = torch.softmax(d, dim=-1)

        result = e.to_torch_cpu()
        assert result.shape == (3, 5)
        # Each row should sum to ~1
        for i in range(3):
            assert abs(result[i].sum().item() - 1.0) < 0.01

    def test_add_mul_neg_chain(self):
        """Chain: add -> mul -> neg."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0, 3.0]))
        b = ApplegpuTensor.from_torch(torch.tensor([4.0, 5.0, 6.0]))

        c = a + b        # [5, 7, 9]
        d = c * a         # [5, 14, 27]
        e = -d            # [-5, -14, -27]

        expected = torch.tensor([-5.0, -14.0, -27.0])
        assert torch.allclose(e.to_torch_cpu(), expected, atol=1e-4)


class TestEndToEndTransformerBlock:
    """End-to-end test: simplified transformer self-attention."""

    def test_self_attention(self):
        """Q/K/V projection -> attention scores -> softmax -> output."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        seq_len, d_model = 4, 8
        x = ApplegpuTensor.from_torch(torch.randn(seq_len, d_model))
        w_q = ApplegpuTensor.from_torch(torch.randn(d_model, d_model))
        w_k = ApplegpuTensor.from_torch(torch.randn(d_model, d_model))
        w_v = ApplegpuTensor.from_torch(torch.randn(d_model, d_model))

        q = x @ w_q
        k = x @ w_k
        v = x @ w_v

        scores = q @ k.t()
        scores = torch.softmax(scores, dim=-1)
        out = scores @ v

        result = out.to_torch_cpu()
        assert result.shape == (seq_len, d_model)
        assert torch.all(torch.isfinite(result))

    def test_attention_scores_sum_to_one(self):
        """Softmax attention rows sum to 1."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        seq_len, d_model = 6, 4
        x = ApplegpuTensor.from_torch(torch.randn(seq_len, d_model))
        w_q = ApplegpuTensor.from_torch(torch.randn(d_model, d_model))
        w_k = ApplegpuTensor.from_torch(torch.randn(d_model, d_model))

        q = x @ w_q
        k = x @ w_k
        scores = torch.softmax(q @ k.t(), dim=-1)

        result = scores.to_torch_cpu()
        row_sums = result.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(seq_len), atol=0.01)


# ============================================================
# Phase E tests (to_applegpu + addmm)
# ============================================================

class TestToApplegpu:
    """Tests for to_applegpu() model/tensor migration."""

    def test_to_applegpu_tensor(self):
        """Single tensor migration."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        t = torch.tensor([1.0, 2.0, 3.0])
        a = gpu.to_applegpu(t)
        assert isinstance(a, ApplegpuTensor)
        assert torch.allclose(a.to_torch_cpu(), t)

    def test_to_applegpu_already_on_device(self):
        """Passing an ApplegpuTensor returns it unchanged."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        t = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0]))
        a = gpu.to_applegpu(t)
        assert a is t

    def test_to_applegpu_linear(self):
        """nn.Linear parameter migration."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        model = torch.nn.Linear(4, 3, bias=True)
        w_before = model.weight.data.clone()
        b_before = model.bias.data.clone()

        model = gpu.to_applegpu(model)

        assert isinstance(model.weight.data, ApplegpuTensor)
        assert isinstance(model.bias.data, ApplegpuTensor)
        assert torch.allclose(model.weight.data.to_torch_cpu(), w_before, atol=1e-6)
        assert torch.allclose(model.bias.data.to_torch_cpu(), b_before, atol=1e-6)

    def test_to_applegpu_linear_forward(self):
        """nn.Linear forward pass after migration produces correct result."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        model = torch.nn.Linear(4, 3, bias=True)
        x = torch.randn(2, 4)
        expected = model(x)  # CPU result

        model = gpu.to_applegpu(model)
        x_gpu = ApplegpuTensor.from_torch(x)

        result = model(x_gpu)
        assert isinstance(result, ApplegpuTensor)
        assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-3)

    def test_to_applegpu_sequential(self):
        """nn.Sequential with multiple layers."""
        from applegpu_runtime.torch_backend import ApplegpuTensor
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 3),
        )
        x = torch.randn(2, 4)
        expected = model(x)

        model = gpu.to_applegpu(model)
        x_gpu = ApplegpuTensor.from_torch(x)
        result = model(x_gpu)

        assert isinstance(result, ApplegpuTensor)
        assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-3)

    def test_to_applegpu_invalid_type(self):
        """Passing an unsupported type raises TypeError."""
        with pytest.raises(TypeError):
            gpu.to_applegpu("not a tensor")


class TestAddmm:
    """Tests for aten.addmm dispatch (used by nn.Linear)."""

    def test_addmm_basic(self):
        """Basic addmm: bias + input @ weight."""
        from applegpu_runtime.torch_backend import ApplegpuTensor, SUPPORTED_OPS
        bias = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0, 3.0]))
        input = ApplegpuTensor.from_torch(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        weight = ApplegpuTensor.from_torch(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        result = SUPPORTED_OPS[torch.ops.aten.addmm.default](bias, input, weight)
        expected = torch.tensor([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]])
        assert isinstance(result, ApplegpuTensor)
        assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-4)

    def test_addmm_with_alpha_beta(self):
        """addmm with non-default alpha and beta."""
        from applegpu_runtime.torch_backend import ApplegpuTensor, SUPPORTED_OPS
        bias = ApplegpuTensor.from_torch(torch.tensor([10.0, 20.0]))
        input = ApplegpuTensor.from_torch(torch.tensor([[1.0, 2.0]]))
        weight = ApplegpuTensor.from_torch(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        # beta * bias + alpha * (input @ weight)
        # 2 * [10, 20] + 3 * ([1,2] @ [[3,4],[5,6]]) = [20, 40] + 3 * [13, 16] = [59, 88]
        result = SUPPORTED_OPS[torch.ops.aten.addmm.default](bias, input, weight, beta=2, alpha=3)
        expected = torch.tensor([[59.0, 88.0]])
        assert isinstance(result, ApplegpuTensor)
        assert torch.allclose(result.to_torch_cpu(), expected, atol=1e-3)

    def test_addmm_registered(self):
        """addmm is in the dispatch table."""
        from applegpu_runtime.torch_backend import SUPPORTED_OPS
        assert torch.ops.aten.addmm.default in SUPPORTED_OPS


class TestCNNOps:
    """CNN ops: conv1d, conv2d, batch_norm, max_pool2d, avg_pool2d."""

    def test_conv1d_basic(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        input = ApplegpuTensor.from_torch(torch.randn(1, 3, 10))
        weight = ApplegpuTensor.from_torch(torch.randn(4, 3, 3))
        result = gpu.conv1d(input._gpu_tensor, weight._gpu_tensor, stride=1, padding=0)
        assert result.shape == [1, 4, 8]

    def test_conv2d_basic(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        input = ApplegpuTensor.from_torch(torch.randn(1, 3, 8, 8))
        weight = ApplegpuTensor.from_torch(torch.randn(16, 3, 3, 3))
        result = gpu.conv2d(input._gpu_tensor, weight._gpu_tensor, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
        assert result.shape == [1, 16, 8, 8]

    def test_max_pool2d_basic(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        input = ApplegpuTensor.from_torch(torch.randn(1, 3, 4, 4))
        result = gpu.max_pool2d(input._gpu_tensor, kh=2, kw=2)
        assert result.shape == [1, 3, 2, 2]

    def test_avg_pool2d_basic(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        input = ApplegpuTensor.from_torch(torch.randn(1, 3, 4, 4))
        result = gpu.avg_pool2d(input._gpu_tensor, kh=2, kw=2)
        assert result.shape == [1, 3, 2, 2]

    def test_batch_norm_basic(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        input = ApplegpuTensor.from_torch(torch.randn(1, 4, 2, 2))
        mean = ApplegpuTensor.from_torch(torch.zeros(4))
        var = ApplegpuTensor.from_torch(torch.ones(4))
        weight = ApplegpuTensor.from_torch(torch.ones(4))
        bias = ApplegpuTensor.from_torch(torch.zeros(4))
        result = gpu.batch_norm(input._gpu_tensor, mean._gpu_tensor, var._gpu_tensor, weight._gpu_tensor, bias._gpu_tensor, eps=1e-5)
        assert result.shape == [1, 4, 2, 2]

    def test_conv2d_with_padding(self):
        from applegpu_runtime.torch_backend import ApplegpuTensor
        input = ApplegpuTensor.from_torch(torch.randn(2, 1, 5, 5))
        weight = ApplegpuTensor.from_torch(torch.randn(8, 1, 3, 3))
        result = gpu.conv2d(input._gpu_tensor, weight._gpu_tensor, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
        assert result.shape == [2, 8, 3, 3]


class TestDispatchRegistry:
    """Test that the dispatch registry is wired correctly."""

    def test_supported_ops_populated(self):
        from applegpu_runtime.torch_backend import SUPPORTED_OPS
        # Check key ops are registered
        assert torch.ops.aten.add.Tensor in SUPPORTED_OPS
        assert torch.ops.aten.mul.Tensor in SUPPORTED_OPS
        assert torch.ops.aten.relu.default in SUPPORTED_OPS
        assert torch.ops.aten.mm.default in SUPPORTED_OPS
        assert torch.ops.aten.reshape.default in SUPPORTED_OPS
        assert torch.ops.aten._softmax.default in SUPPORTED_OPS
        # Phase C ops
        assert torch.ops.aten.full.default in SUPPORTED_OPS
        assert torch.ops.aten.scalar_tensor.default in SUPPORTED_OPS
        assert torch.ops.aten.copy_.default in SUPPORTED_OPS

    def test_unsupported_op_falls_back(self):
        """Ops not in the registry fall back to CPU."""
        from applegpu_runtime.torch_backend import ApplegpuTensor, _warned_ops
        _warned_ops.clear()
        a = ApplegpuTensor.from_torch(torch.tensor([1.0, 2.0, 3.0]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # cos is not registered
            result = torch.cos(a)
            assert any("not supported" in str(warning.message) for warning in w)
        assert isinstance(result, ApplegpuTensor)
