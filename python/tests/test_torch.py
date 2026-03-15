import numpy as np
import pytest
import torch
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


def test_from_torch_roundtrip():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    g = gpu.from_torch(t)
    result = g.to_torch()
    assert torch.allclose(result, t)


def test_from_torch_preserves_shape():
    t = torch.ones(3, 4)
    g = gpu.from_torch(t)
    result = g.to_torch()
    assert result.shape == (3, 4)


def test_from_torch_float64_roundtrip():
    t = torch.tensor([1.0, 2.0], dtype=torch.float64)
    g = gpu.from_torch(t)
    assert g.dtype == "float64"
    result = g.to_torch()
    assert torch.allclose(result, t)


def test_from_torch_int32_roundtrip():
    t = torch.tensor([10, 20, 30], dtype=torch.int32)
    g = gpu.from_torch(t)
    assert g.dtype == "int32"
    result = g.to_torch()
    assert torch.equal(result, t)


def test_to_torch_auto_evals():
    a = gpu.from_torch(torch.tensor([1.0, 2.0, 3.0]))
    b = gpu.from_torch(torch.tensor([4.0, 5.0, 6.0]))
    c = a + b  # lazy
    result = c.to_torch()
    expected = torch.tensor([5.0, 7.0, 9.0])
    assert torch.allclose(result, expected)


def test_from_torch_copies_data():
    t = torch.tensor([1.0, 2.0, 3.0])
    g = gpu.from_torch(t)
    t[0] = 999.0  # modify original
    result = g.to_torch()
    assert result[0].item() == 1.0  # unaffected


def test_from_torch_non_contiguous():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).t()  # transposed = non-contiguous
    assert not t.is_contiguous()
    g = gpu.from_torch(t)
    result = g.to_torch()
    assert torch.allclose(result, t.contiguous())


def test_from_torch_requires_grad():
    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    g = gpu.from_torch(t)  # should not error (detach handles it)
    result = g.to_torch()
    assert not result.requires_grad
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))


def test_from_torch_scalar():
    t = torch.tensor(3.14)
    g = gpu.from_torch(t)
    result = g.to_torch()
    assert result.shape == ()  # 0-dim
    assert abs(result.item() - 3.14) < 0.001


def test_from_torch_empty():
    t = torch.zeros(0)
    with pytest.raises(ValueError, match="0 bytes"):
        gpu.from_torch(t)


def test_from_torch_speed():
    """from_torch should be fast (direct data_ptr, no numpy round-trip)."""
    import time
    t = torch.randn(1024, 1024)

    # Warm up
    gpu.from_torch(t)

    start = time.time()
    for _ in range(10):
        g = gpu.from_torch(t)
    elapsed = (time.time() - start) / 10

    print(f"from_torch: {elapsed*1000:.1f}ms for 1M elements")
    # Should be < 50ms per call (was 212ms with numpy path)
    assert elapsed < 0.1, f"from_torch too slow: {elapsed*1000:.0f}ms"


def test_from_torch_int64_roundtrip():
    t = torch.tensor([100, 200, 300], dtype=torch.int64)
    g = gpu.from_torch(t)
    assert g.dtype == "int64"
    result = g.to_torch()
    assert torch.equal(result, t)


def test_from_torch_float16_roundtrip():
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    g = gpu.from_torch(t)
    assert g.dtype == "float16"
    result = g.to_torch()
    assert torch.allclose(result.float(), t.float(), atol=0.01)


def test_from_torch_uint8_roundtrip():
    t = torch.tensor([0, 127, 255], dtype=torch.uint8)
    g = gpu.from_torch(t)
    assert g.dtype == "uint8"
    result = g.to_torch()
    assert torch.equal(result, t)


def test_from_torch_bool_roundtrip():
    t = torch.tensor([True, False, True], dtype=torch.bool)
    g = gpu.from_torch(t)
    assert g.dtype == "bool"
    result = g.to_torch()
    assert torch.equal(result, t)


def test_from_bytes_roundtrip():
    """from_bytes should create tensors from raw byte data."""
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    g = gpu.from_bytes(data.tobytes(), [4], "float32")
    result = gpu.to_list(g)
    assert result == [1.0, 2.0, 3.0, 4.0]


def test_from_bytes_int32():
    data = np.array([10, 20, 30], dtype=np.int32)
    g = gpu.from_bytes(data.tobytes(), [3], "int32")
    result = gpu.to_list(g)
    assert result == [10, 20, 30]


def test_from_bytes_2d():
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    g = gpu.from_bytes(data.tobytes(), [2, 2], "float32")
    assert gpu.shape(g) == [2, 2]
    result = gpu.to_list(g)
    assert result == [1.0, 2.0, 3.0, 4.0]


def test_from_numpy_speed():
    """from_numpy should be fast (direct ctypes data pointer, no tobytes)."""
    import time
    arr = np.random.randn(1024, 1024).astype(np.float32)

    # Warm up
    gpu.from_numpy(arr)

    start = time.time()
    for _ in range(10):
        g = gpu.from_numpy(arr)
    elapsed = (time.time() - start) / 10

    print(f"from_numpy: {elapsed*1000:.1f}ms for 1M elements")
    # Should be < 50ms per call
    assert elapsed < 0.1, f"from_numpy too slow: {elapsed*1000:.0f}ms"
