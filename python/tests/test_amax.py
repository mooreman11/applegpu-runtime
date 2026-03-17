"""Tests for amax (absolute max) reduction kernel."""
import numpy as np
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def _init():
    gpu.init_backend()


def test_amax_1d():
    a = gpu.from_numpy(np.array([-5.0, 3.0, -1.0, 4.0], dtype=np.float32))
    out = gpu.amax(a).to_list()
    assert abs(out[0] - 5.0) < 1e-5


def test_amax_2d():
    a = gpu.from_numpy(np.array([[1.0, -3.0], [2.0, -4.0]], dtype=np.float32))
    out = gpu.amax(a).to_list()
    assert abs(out[0] - 3.0) < 1e-5
    assert abs(out[1] - 4.0) < 1e-5


def test_amax_3d():
    data = np.random.randn(2, 3, 4).astype(np.float32)
    a = gpu.from_numpy(data)
    out = np.array(gpu.amax(a).to_list()).reshape(2, 3)
    expected = np.max(np.abs(data), axis=-1)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_amax_all_positive():
    a = gpu.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    out = gpu.amax(a).to_list()
    assert abs(out[0] - 3.0) < 1e-5


def test_amax_float16():
    data = np.array([-2.0, 1.0, -3.0, 0.5], dtype=np.float16)
    a = gpu.from_numpy(data)
    out = gpu.amax(a).to_list()
    assert abs(out[0] - 3.0) < 0.1


def test_amax_single_element():
    a = gpu.from_numpy(np.array([-7.0], dtype=np.float32))
    out = gpu.amax(a).to_list()
    assert abs(out[0] - 7.0) < 1e-5


def test_amax_zeros():
    a = gpu.from_numpy(np.zeros(8, dtype=np.float32))
    out = gpu.amax(a).to_list()
    assert abs(out[0]) < 1e-5
