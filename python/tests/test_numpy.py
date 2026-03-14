import numpy as np
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


def test_from_numpy_roundtrip():
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    t = gpu.from_numpy(arr)
    result = t.to_numpy()
    np.testing.assert_array_equal(result, arr)


def test_from_numpy_preserves_shape():
    arr = np.ones((3, 4), dtype=np.float32)
    t = gpu.from_numpy(arr)
    result = t.to_numpy()
    assert result.shape == (3, 4)


def test_from_numpy_rejects_non_float32():
    arr = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises((ValueError, TypeError)):
        gpu.from_numpy(arr)


def test_from_numpy_rejects_non_contiguous():
    # Use a stride-based view that is neither C- nor F-contiguous
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    assert not arr.flags['C_CONTIGUOUS']
    assert not arr.flags['F_CONTIGUOUS']
    with pytest.raises((ValueError, TypeError)):
        gpu.from_numpy(arr)


def test_to_numpy_auto_evals():
    a = gpu.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = gpu.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    c = a + b  # lazy
    result = c.to_numpy()
    np.testing.assert_array_equal(result, np.array([5.0, 7.0, 9.0]))


def test_from_numpy_copies_data():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = gpu.from_numpy(arr)
    arr[0] = 999.0  # modify original
    result = t.to_numpy()
    assert result[0] == 1.0  # tensor unaffected


def test_from_numpy_empty_array():
    """Empty arrays are rejected since Metal cannot allocate 0-byte buffers."""
    arr = np.array([], dtype=np.float32)
    with pytest.raises(ValueError):
        gpu.from_numpy(arr)
