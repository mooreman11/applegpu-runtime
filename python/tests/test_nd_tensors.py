import numpy as np
import pytest
import applegpu_runtime as gpu

@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()

def test_tensor_3d():
    t = gpu.tensor([float(i) for i in range(24)], shape=[2, 3, 4])
    assert t.shape == [2, 3, 4]

def test_3d_from_numpy():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = gpu.from_numpy(arr)
    result = t.to_numpy()
    assert result.shape == (2, 3, 4)
    np.testing.assert_array_equal(result, arr)

def test_broadcast_add():
    x = gpu.from_numpy(np.ones((4, 3), dtype=np.float32))
    bias = gpu.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    result = (x + bias).to_numpy()
    assert result.shape == (4, 3)
    np.testing.assert_allclose(result[0], [11.0, 21.0, 31.0])
    np.testing.assert_allclose(result[3], [11.0, 21.0, 31.0])

def test_broadcast_3d():
    a = gpu.from_numpy(np.ones((2, 1, 4), dtype=np.float32))
    b = gpu.from_numpy(np.ones((3, 4), dtype=np.float32) * 2)
    result = (a + b).to_numpy()
    assert result.shape == (2, 3, 4)
    np.testing.assert_allclose(result, 3.0)

def test_3d_relu():
    arr = np.array([[[-1, 2], [3, -4]], [[-5, 6], [7, -8]]], dtype=np.float32)
    t = gpu.from_numpy(arr)
    result = t.relu().to_numpy()
    expected = np.maximum(arr, 0)
    np.testing.assert_array_equal(result, expected)

def test_3d_gelu():
    arr = np.array([[[1.0, -1.0], [2.0, -2.0]]], dtype=np.float32)
    t = gpu.from_numpy(arr)
    result = t.gelu().to_numpy()
    assert result.shape == (1, 2, 2)
    assert np.all(np.isfinite(result))

def test_softmax_accepts_3d():
    t = gpu.tensor([1.0] * 8, shape=[2, 2, 2])
    result = gpu.softmax(t).to_numpy()
    assert result.shape == (2, 2, 2)
    # Each last-dim row should sum to 1
    assert abs(result[0, 0].sum() - 1.0) < 0.001

def test_matmul_accepts_3d():
    a = gpu.tensor([1.0] * 8, shape=[2, 2, 2])
    b = gpu.tensor([1.0] * 8, shape=[2, 2, 2])
    c = (a @ b).to_numpy()
    assert c.shape == (2, 2, 2)

def test_reshape_to_3d():
    t = gpu.tensor([float(i) for i in range(12)], shape=[12])
    t2 = gpu.reshape(t, [2, 3, 2])
    assert t2.shape == [2, 3, 2]
    assert len(t2.to_list()) == 12

def test_existing_2d_unchanged():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = a @ b
    result = c.to_list()
    assert abs(result[0] - 19.0) < 0.1
    assert abs(result[3] - 50.0) < 0.1

def test_4d_tensor():
    arr = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
    t = gpu.from_numpy(arr)
    result = t.to_numpy()
    assert result.shape == (2, 3, 4, 5)
    np.testing.assert_array_equal(result, arr)

def test_broadcast_scalar():
    x = gpu.from_numpy(np.ones((3, 4), dtype=np.float32))
    s = gpu.tensor([5.0], shape=[1])
    result = (x + s).to_numpy()
    np.testing.assert_allclose(result, 6.0)
