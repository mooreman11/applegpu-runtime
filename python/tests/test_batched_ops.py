"""Tests for batched N-D tensor operations.

Verifies that matmul, softmax, layer_norm, embedding, transpose,
attention, and attention_causal all work with 3D+ tensors.
"""

import numpy as np
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


# ── Batched matmul ────────────────────────────────────────────────────────────


def test_batched_matmul_3d():
    """[batch, M, K] @ [batch, K, N] -> [batch, M, N]"""
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    b_np = np.random.randn(2, 4, 5).astype(np.float32)
    a = gpu.from_numpy(a_np)
    b = gpu.from_numpy(b_np)
    c = a @ b
    result = c.to_numpy()
    assert result.shape == (2, 3, 5)
    expected = np.matmul(a_np, b_np)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_batched_matmul_4d():
    """[batch1, batch2, M, K] @ [batch1, batch2, K, N] -> [batch1, batch2, M, N]"""
    a_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b_np = np.random.randn(2, 3, 5, 6).astype(np.float32)
    a = gpu.from_numpy(a_np)
    b = gpu.from_numpy(b_np)
    result = (a @ b).to_numpy()
    assert result.shape == (2, 3, 4, 6)
    expected = np.matmul(a_np, b_np)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_batched_matmul_broadcast():
    """[batch, M, K] @ [K, N] -> [batch, M, N] via broadcast."""
    a_np = np.random.randn(3, 4, 5).astype(np.float32)
    b_np = np.random.randn(5, 6).astype(np.float32)
    a = gpu.from_numpy(a_np)
    b = gpu.from_numpy(b_np)
    result = (a @ b).to_numpy()
    assert result.shape == (3, 4, 6)
    expected = np.matmul(a_np, b_np)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


# ── Batched softmax ──────────────────────────────────────────────────────────


def test_batched_softmax_3d():
    """Softmax over last dim for [batch, rows, cols]."""
    x_np = np.random.randn(2, 3, 4).astype(np.float32)
    x = gpu.from_numpy(x_np)
    result = gpu.softmax(x).to_numpy()
    assert result.shape == (2, 3, 4)
    # Each row (last dim) should sum to 1
    for b in range(2):
        for r in range(3):
            assert abs(result[b, r].sum() - 1.0) < 0.001


def test_batched_softmax_4d():
    """Softmax over last dim for 4D tensor."""
    x_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x = gpu.from_numpy(x_np)
    result = gpu.softmax(x).to_numpy()
    assert result.shape == (2, 3, 4, 5)
    # Check a few row sums
    assert abs(result[0, 0, 0].sum() - 1.0) < 0.001
    assert abs(result[1, 2, 3].sum() - 1.0) < 0.001


def test_batched_softmax_values():
    """Verify softmax output matches numpy reference."""
    x_np = np.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                      [[10.0, 10.0, 10.0], [-1.0, 0.0, 1.0]]], dtype=np.float32)
    x = gpu.from_numpy(x_np)
    result = gpu.softmax(x).to_numpy()
    # Compute numpy reference: softmax(x, axis=-1)
    def np_softmax(arr):
        e = np.exp(arr - arr.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    expected = np_softmax(x_np)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


# ── Batched layer_norm ───────────────────────────────────────────────────────


def test_batched_layer_norm_3d():
    """Layer norm over last dim for [batch, seq, features]."""
    x = gpu.from_numpy(np.ones((2, 3, 4), dtype=np.float32) * 5.0)
    gamma = gpu.from_numpy(np.ones(4, dtype=np.float32))
    beta = gpu.from_numpy(np.zeros(4, dtype=np.float32))
    result = gpu.layer_norm(x, gamma, beta).to_numpy()
    # Constant input -> mean=5, var=0, normalized=0
    assert result.shape == (2, 3, 4)
    np.testing.assert_allclose(result, 0.0, atol=0.01)


def test_batched_layer_norm_3d_varied():
    """Layer norm with non-trivial inputs."""
    np.random.seed(42)
    x_np = np.random.randn(2, 4, 8).astype(np.float32)
    gamma_np = np.ones(8, dtype=np.float32) * 2.0
    beta_np = np.ones(8, dtype=np.float32) * 0.5

    x = gpu.from_numpy(x_np)
    gamma = gpu.from_numpy(gamma_np)
    beta = gpu.from_numpy(beta_np)
    result = gpu.layer_norm(x, gamma, beta).to_numpy()
    assert result.shape == (2, 4, 8)

    # Verify each [batch, row, :] is normalized then scaled/shifted
    for b in range(2):
        for r in range(4):
            row = x_np[b, r]
            mean = row.mean()
            var = row.var()
            normed = (row - mean) / np.sqrt(var + 1e-5)
            expected = normed * gamma_np + beta_np
            np.testing.assert_allclose(result[b, r], expected, rtol=1e-3, atol=1e-3)


def test_batched_layer_norm_4d():
    """Layer norm over last dim for 4D tensor."""
    x = gpu.from_numpy(np.random.randn(2, 3, 4, 8).astype(np.float32))
    gamma = gpu.from_numpy(np.ones(8, dtype=np.float32))
    beta = gpu.from_numpy(np.zeros(8, dtype=np.float32))
    result = gpu.layer_norm(x, gamma, beta).to_numpy()
    assert result.shape == (2, 3, 4, 8)
    # Check normalization: each [..., :] should have mean~0, std~1
    for b in range(2):
        for s in range(3):
            for r in range(4):
                row = result[b, s, r]
                assert abs(row.mean()) < 0.1
                assert abs(row.std() - 1.0) < 0.2


# ── Batched embedding ───────────────────────────────────────────────────────


def test_batched_embedding_2d_indices():
    """Embedding with 2D indices: [batch, seq] -> [batch, seq, dim]."""
    weights_np = np.arange(15, dtype=np.float32).reshape(5, 3)
    weights = gpu.from_numpy(weights_np)
    indices = gpu.from_numpy(np.array([[0, 2], [4, 1]], dtype=np.int32))
    result = gpu.embedding(weights, indices).to_numpy()
    assert result.shape == (2, 2, 3)
    np.testing.assert_allclose(result[0, 0], [0, 1, 2])
    np.testing.assert_allclose(result[0, 1], [6, 7, 8])
    np.testing.assert_allclose(result[1, 0], [12, 13, 14])
    np.testing.assert_allclose(result[1, 1], [3, 4, 5])


def test_batched_embedding_3d_indices():
    """Embedding with 3D indices: [a, b, c] -> [a, b, c, dim]."""
    weights_np = np.eye(4, dtype=np.float32)  # identity: each row is one-hot
    weights = gpu.from_numpy(weights_np)
    indices = gpu.from_numpy(np.array([[[0, 1], [2, 3]]], dtype=np.int32))
    result = gpu.embedding(weights, indices).to_numpy()
    assert result.shape == (1, 2, 2, 4)
    np.testing.assert_allclose(result[0, 0, 0], [1, 0, 0, 0])
    np.testing.assert_allclose(result[0, 0, 1], [0, 1, 0, 0])
    np.testing.assert_allclose(result[0, 1, 0], [0, 0, 1, 0])
    np.testing.assert_allclose(result[0, 1, 1], [0, 0, 0, 1])


# ── Batched transpose ───────────────────────────────────────────────────────


def test_batched_transpose_3d():
    """Transpose last 2 dims: [batch, rows, cols] -> [batch, cols, rows]."""
    x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = gpu.from_numpy(x_np)
    result = gpu.transpose(x).to_numpy()
    assert result.shape == (2, 4, 3)
    expected = np.transpose(x_np, (0, 2, 1))
    np.testing.assert_allclose(result, expected)


def test_batched_transpose_4d():
    """Transpose last 2 dims of a 4D tensor."""
    x_np = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
    x = gpu.from_numpy(x_np)
    result = gpu.transpose(x).to_numpy()
    assert result.shape == (2, 3, 5, 4)
    expected = np.transpose(x_np, (0, 1, 3, 2))
    np.testing.assert_allclose(result, expected)


# ── Batched attention ────────────────────────────────────────────────────────


def test_batched_attention_3d():
    """Attention with batch dim: [batch, seq, d_k]."""
    np.random.seed(123)
    q = gpu.from_numpy(np.random.randn(2, 4, 8).astype(np.float32))
    k = gpu.from_numpy(np.random.randn(2, 4, 8).astype(np.float32))
    v = gpu.from_numpy(np.random.randn(2, 4, 8).astype(np.float32))
    result = gpu.attention(q, k, v).to_numpy()
    assert result.shape == (2, 4, 8)
    assert np.all(np.isfinite(result))


def test_batched_attention_causal_3d():
    """Causal attention with batch dim: [batch, seq, d_k]."""
    np.random.seed(456)
    q = gpu.from_numpy(np.random.randn(2, 4, 8).astype(np.float32))
    k = gpu.from_numpy(np.random.randn(2, 4, 8).astype(np.float32))
    v = gpu.from_numpy(np.random.randn(2, 4, 8).astype(np.float32))
    result = gpu.attention_causal(q, k, v).to_numpy()
    assert result.shape == (2, 4, 8)
    assert np.all(np.isfinite(result))


def test_batched_attention_causal_masking():
    """Verify causal masking: first position should only attend to itself."""
    np.random.seed(789)
    # Use identity-like V so we can check which positions get attended to
    v_np = np.zeros((1, 4, 4), dtype=np.float32)
    v_np[0, 0] = [1, 0, 0, 0]
    v_np[0, 1] = [0, 1, 0, 0]
    v_np[0, 2] = [0, 0, 1, 0]
    v_np[0, 3] = [0, 0, 0, 1]

    # Large Q/K so attention weights are sharp
    q_np = np.zeros((1, 4, 4), dtype=np.float32)
    k_np = np.zeros((1, 4, 4), dtype=np.float32)
    for i in range(4):
        q_np[0, i, i] = 10.0
        k_np[0, i, i] = 10.0

    q = gpu.from_numpy(q_np)
    k = gpu.from_numpy(k_np)
    v = gpu.from_numpy(v_np)
    result = gpu.attention_causal(q, k, v).to_numpy()
    assert result.shape == (1, 4, 4)
    # Position 0 can only attend to position 0 -> output should be ~v[0]
    assert result[0, 0, 0] > 0.9


def test_batched_attention_4d():
    """Attention with [batch, heads, seq, d_k] shape."""
    np.random.seed(101)
    q = gpu.from_numpy(np.random.randn(2, 4, 6, 8).astype(np.float32))
    k = gpu.from_numpy(np.random.randn(2, 4, 6, 8).astype(np.float32))
    v = gpu.from_numpy(np.random.randn(2, 4, 6, 8).astype(np.float32))
    result = gpu.attention_causal(q, k, v).to_numpy()
    assert result.shape == (2, 4, 6, 8)
    assert np.all(np.isfinite(result))


# ── Batched softmax_causal ───────────────────────────────────────────────────


def test_batched_softmax_causal_3d():
    """Causal softmax with batch dim."""
    x = gpu.from_numpy(np.ones((2, 4, 4), dtype=np.float32))
    result = gpu.softmax_causal(x).to_numpy()
    assert result.shape == (2, 4, 4)
    # Row 0: only position 0 is unmasked -> should be 1.0
    assert abs(result[0, 0, 0] - 1.0) < 0.01
    # Row 1: positions 0,1 unmasked -> each ~0.5
    assert abs(result[0, 1, 0] - 0.5) < 0.01
    assert abs(result[0, 1, 1] - 0.5) < 0.01


# ── 2D backward compatibility ────────────────────────────────────────────────


def test_2d_matmul_still_works():
    """Standard 2D matmul still works correctly."""
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
    a = gpu.from_numpy(a_np)
    b = gpu.from_numpy(b_np)
    result = (a @ b).to_numpy()
    expected = a_np @ b_np
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_2d_softmax_still_works():
    """Standard 2D softmax still works correctly."""
    x = gpu.from_numpy(np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32))
    result = gpu.softmax(x).to_numpy()
    assert result.shape == (2, 3)
    assert abs(result[0].sum() - 1.0) < 0.001
    assert abs(result[1].sum() - 1.0) < 0.001
