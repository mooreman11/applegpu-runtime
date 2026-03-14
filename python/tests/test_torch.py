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
