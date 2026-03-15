import numpy as np
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


def test_concat_all_two_tensors_dim0():
    """concat_all with 2 tensors along dim=0 matches pairwise concat."""
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    result = gpu.concat_all([a, b], dim=0)
    assert result.shape == [4, 2]
    assert result.to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def test_concat_all_two_tensors_dim1():
    """concat_all with 2 tensors along dim=1 matches pairwise concat."""
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    result = gpu.concat_all([a, b], dim=1)
    assert result.shape == [2, 4]
    expected = gpu.concat(a, b, dim=1).to_list()
    assert result.to_list() == expected


def test_concat_all_three_tensors_dim0():
    """concat_all with 3 tensors along dim=0."""
    a = gpu.tensor([1.0, 2.0], shape=[1, 2])
    b = gpu.tensor([3.0, 4.0], shape=[1, 2])
    c = gpu.tensor([5.0, 6.0], shape=[1, 2])
    result = gpu.concat_all([a, b, c], dim=0)
    assert result.shape == [3, 2]
    assert result.to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_concat_all_three_tensors_dim1():
    """concat_all with 3 tensors along dim=1."""
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = gpu.tensor([9.0, 10.0, 11.0, 12.0], shape=[2, 2])
    result = gpu.concat_all([a, b, c], dim=1)
    assert result.shape == [2, 6]
    # Row 0: [1,2,5,6,9,10], Row 1: [3,4,7,8,11,12]
    assert result.to_list() == [1.0, 2.0, 5.0, 6.0, 9.0, 10.0,
                                 3.0, 4.0, 7.0, 8.0, 11.0, 12.0]


def test_concat_all_many_tensors():
    """concat_all with many tensors (simulates multi-head attention concat)."""
    n_heads = 12
    seq_len = 4
    d_head = 8
    heads = []
    for i in range(n_heads):
        data = [float(i * d_head + j) for _ in range(seq_len) for j in range(d_head)]
        heads.append(gpu.tensor(data, shape=[seq_len, d_head]))

    result = gpu.concat_all(heads, dim=1)
    assert result.shape == [seq_len, n_heads * d_head]

    # Verify by comparing with pairwise concat
    expected = heads[0]
    for h in heads[1:]:
        expected = gpu.concat(expected, h, dim=1)
    assert result.to_list() == expected.to_list()


def test_concat_all_single_tensor():
    """concat_all with 1 tensor returns it unchanged."""
    a = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    result = gpu.concat_all([a], dim=0)
    assert result.shape == [3]
    assert result.to_list() == [1.0, 2.0, 3.0]


def test_concat_all_empty_raises():
    """concat_all with empty list raises ValueError."""
    with pytest.raises(ValueError, match="at least 1 tensor"):
        gpu.concat_all([], dim=0)


def test_concat_all_default_dim():
    """concat_all defaults to dim=0."""
    a = gpu.tensor([1.0, 2.0], shape=[1, 2])
    b = gpu.tensor([3.0, 4.0], shape=[1, 2])
    result = gpu.concat_all([a, b])
    assert result.shape == [2, 2]
    assert result.to_list() == [1.0, 2.0, 3.0, 4.0]


def test_concat_all_numpy_roundtrip():
    """concat_all result matches numpy concatenation."""
    np.random.seed(42)
    arrays = [np.random.randn(3, 4).astype(np.float32) for _ in range(5)]
    tensors = [gpu.from_numpy(a) for a in arrays]

    result = gpu.concat_all(tensors, dim=0).to_numpy()
    expected = np.concatenate(arrays, axis=0)
    np.testing.assert_allclose(result, expected, atol=1e-5)

    result1 = gpu.concat_all(tensors, dim=1).to_numpy()
    expected1 = np.concatenate(arrays, axis=1)
    np.testing.assert_allclose(result1, expected1, atol=1e-5)
