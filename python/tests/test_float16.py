import numpy as np
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


def test_tensor_f16_creation():
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4], dtype="float16")
    assert t.dtype == "float16"
    assert t.shape == [4]


def test_f16_to_list():
    t = gpu.tensor([1.0, 2.0, 3.0], shape=[3], dtype="float16")
    result = t.to_list()
    assert len(result) == 3
    assert abs(result[0] - 1.0) < 0.01


def test_f16_from_numpy_roundtrip():
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    t = gpu.from_numpy(arr)
    assert t.dtype == "float16"
    result = t.to_numpy()
    assert result.dtype == np.float16
    np.testing.assert_allclose(result, arr, rtol=1e-2)


def test_f16_ops():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4], dtype="float16")
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[4], dtype="float16")
    c = a + b
    result = c.to_list()
    assert abs(result[0] - 6.0) < 0.1
    assert abs(result[3] - 12.0) < 0.1


def test_f16_matmul():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2], dtype="float16")
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2], dtype="float16")
    c = a @ b
    result = c.to_list()
    assert abs(result[0] - 19.0) < 1.0


def test_mixed_dtype_error():
    a = gpu.tensor([1.0], shape=[1], dtype="float16")
    b = gpu.tensor([1.0], shape=[1])
    with pytest.raises(ValueError):
        c = a + b


def test_dtype_getter():
    t32 = gpu.tensor([1.0], shape=[1])
    t16 = gpu.tensor([1.0], shape=[1], dtype="float16")
    assert t32.dtype == "float32"
    assert t16.dtype == "float16"


def test_f16_from_torch_roundtrip():
    torch = pytest.importorskip("torch")
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    g = gpu.from_torch(t)
    assert g.dtype == "float16"
    result = g.to_torch()
    assert result.dtype == torch.float16
    assert torch.allclose(result, t, atol=0.01)
