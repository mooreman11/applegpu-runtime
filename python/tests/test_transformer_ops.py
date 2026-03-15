import numpy as np
import math
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


# --- GELU ---

def _ref_gelu(x):
    """Reference GELU implementation."""
    return x * 0.5 * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def test_gelu_eval():
    arr = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    t = gpu.from_numpy(arr)
    result = gpu.gelu(t).to_numpy()
    expected = np.array([_ref_gelu(x) for x in arr], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_gelu_method():
    t = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    result = t.gelu().to_list()
    assert len(result) == 3
    assert abs(result[0] - _ref_gelu(1.0)) < 0.01


def test_gelu_f16():
    arr = np.array([0.0, 1.0, -1.0], dtype=np.float16)
    t = gpu.from_numpy(arr)
    result = gpu.gelu(t).to_numpy()
    assert result.dtype == np.float16
    assert abs(float(result[1]) - _ref_gelu(1.0)) < 0.05


# --- LayerNorm ---

def test_layer_norm_eval():
    # Input: 2 rows of 4 elements
    x = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    gamma = np.ones(4, dtype=np.float32)
    beta = np.zeros(4, dtype=np.float32)

    t = gpu.from_numpy(x)
    g = gpu.from_numpy(gamma)
    b = gpu.from_numpy(beta)
    result = gpu.layer_norm(t, g, b).to_numpy()

    # Manual: for row [1,2,3,4], mean=2.5, var=1.25
    # (x - mean) / sqrt(var + 1e-5) with gamma=1, beta=0
    for row_idx in range(2):
        row = x[row_idx]
        mean = row.mean()
        var = row.var()  # numpy uses population variance by default for .var()
        expected = (row - mean) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(result[row_idx], expected, rtol=1e-4)


def test_layer_norm_with_affine():
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    gamma = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    beta = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    t = gpu.from_numpy(x)
    g = gpu.from_numpy(gamma)
    b = gpu.from_numpy(beta)
    result = gpu.layer_norm(t, g, b).to_numpy()

    # gamma=2, beta=1: result = 2 * normalized + 1
    mean = x[0].mean()
    var = x[0].var()
    normalized = (x[0] - mean) / np.sqrt(var + 1e-5)
    expected = 2.0 * normalized + 1.0
    np.testing.assert_allclose(result[0], expected, rtol=1e-4)


def test_layer_norm_method():
    x = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    g = gpu.tensor([1.0, 1.0], shape=[2])
    b = gpu.tensor([0.0, 0.0], shape=[2])
    result = x.layer_norm(g, b).to_list()
    assert len(result) == 4


def test_layer_norm_1d_works():
    """Layer norm now supports any dimensionality including 1D."""
    x = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    g = gpu.tensor([1.0, 1.0, 1.0], shape=[3])
    b = gpu.tensor([0.0, 0.0, 0.0], shape=[3])
    result = gpu.layer_norm(x, g, b).to_list()
    assert len(result) == 3


# --- Embedding ---

def test_embedding_eval():
    # Weights: 5 words x 3 dims
    weights = np.array([
        [0.1, 0.2, 0.3],
        [1.1, 1.2, 1.3],
        [2.1, 2.2, 2.3],
        [3.1, 3.2, 3.3],
        [4.1, 4.2, 4.3],
    ], dtype=np.float32)
    indices = np.array([0, 3, 1], dtype=np.int32)

    w = gpu.from_numpy(weights)
    idx = gpu.from_numpy(indices)
    result = gpu.embedding(w, idx).to_numpy()

    assert result.shape == (3, 3)
    np.testing.assert_allclose(result[0], weights[0], rtol=1e-5)
    np.testing.assert_allclose(result[1], weights[3], rtol=1e-5)
    np.testing.assert_allclose(result[2], weights[1], rtol=1e-5)


def test_embedding_f16():
    weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float16)
    indices = np.array([2, 0], dtype=np.int32)

    w = gpu.from_numpy(weights)
    idx = gpu.from_numpy(indices)
    result = gpu.embedding(w, idx).to_numpy()

    assert result.dtype == np.float16
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0], weights[2], rtol=1e-2)
    np.testing.assert_allclose(result[1], weights[0], rtol=1e-2)


def test_embedding_rejects_float_indices():
    weights = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    indices = gpu.tensor([0.0, 1.0], shape=[2])  # float, not int
    with pytest.raises(ValueError):
        gpu.embedding(weights, indices)
