import numpy as np
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


# --- tensor() creation for all dtypes ---

@pytest.mark.parametrize("dtype_str,np_dtype", [
    ("float16", np.float16),
    ("float32", np.float32),
    ("float64", np.float64),
    ("int8", np.int8),
    ("int16", np.int16),
    ("int32", np.int32),
    ("int64", np.int64),
    ("uint8", np.uint8),
    ("uint32", np.uint32),
    ("bool", np.bool_),
])
def test_tensor_creation_all_dtypes(dtype_str, np_dtype):
    if dtype_str == "bool":
        t = gpu.tensor([True, False, True], shape=[3], dtype=dtype_str)
    elif dtype_str.startswith(("int", "uint")):
        t = gpu.tensor([1, 2, 3], shape=[3], dtype=dtype_str)
    else:
        t = gpu.tensor([1.0, 2.0, 3.0], shape=[3], dtype=dtype_str)
    assert t.dtype == dtype_str
    assert t.shape == [3]


# --- from_numpy roundtrip for all dtypes ---

@pytest.mark.parametrize("np_dtype", [
    np.float16, np.float32, np.float64,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint32, np.bool_,
])
def test_from_numpy_roundtrip_all_dtypes(np_dtype):
    if np_dtype == np.bool_:
        arr = np.array([True, False, True, False], dtype=np_dtype)
    elif np.issubdtype(np_dtype, np.integer):
        arr = np.array([1, 2, 3, 4], dtype=np_dtype)
    else:
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np_dtype)
    t = gpu.from_numpy(arr)
    result = t.to_numpy()
    assert result.dtype == np_dtype
    np.testing.assert_array_equal(result, arr)


# --- to_list returns correct Python types ---

def test_to_list_float_types():
    for dtype in ["float16", "float32", "float64"]:
        t = gpu.tensor([1.0, 2.0, 3.0], shape=[3], dtype=dtype)
        result = t.to_list()
        assert len(result) == 3
        assert isinstance(result[0], float)
        assert abs(result[0] - 1.0) < 0.1


def test_to_list_int_types():
    for dtype in ["int8", "int16", "int32", "int64", "uint8", "uint32"]:
        t = gpu.tensor([1, 2, 3], shape=[3], dtype=dtype)
        result = t.to_list()
        assert len(result) == 3
        assert isinstance(result[0], int)
        assert result == [1, 2, 3]


def test_to_list_bool():
    t = gpu.tensor([True, False, True], shape=[3], dtype="bool")
    result = t.to_list()
    assert result == [True, False, True]
    assert isinstance(result[0], bool)


# --- Compute validation ---

def test_compute_rejects_float64():
    """Float64 is the only dtype not supported for compute."""
    a = gpu.tensor([1.0, 2.0, 3.0], shape=[3], dtype="float64")
    b = gpu.tensor([4.0, 5.0, 6.0], shape=[3], dtype="float64")
    with pytest.raises((ValueError, RuntimeError)):
        c = a + b


def test_int32_arithmetic_works():
    """Int32 arithmetic should work after multi-dtype template expansion."""
    a = gpu.tensor([1, 2, 3], shape=[3], dtype="int32")
    b = gpu.tensor([4, 5, 6], shape=[3], dtype="int32")
    c = a + b
    c.eval()
    assert c.to_list() == [5, 7, 9]


# --- dtype getter ---

def test_dtype_getter_all():
    for dtype in ["float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint32", "bool"]:
        if dtype == "bool":
            t = gpu.tensor([True], shape=[1], dtype=dtype)
        elif dtype.startswith(("int", "uint")):
            t = gpu.tensor([1], shape=[1], dtype=dtype)
        else:
            t = gpu.tensor([1.0], shape=[1], dtype=dtype)
        assert t.dtype == dtype


# --- Backward compat ---

def test_backward_compat_default_float32():
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    assert t.dtype == "float32"
    a = gpu.tensor([1.0, 2.0], shape=[2])
    b = gpu.tensor([3.0, 4.0], shape=[2])
    c = a + b
    assert c.to_list() == [4.0, 6.0]


# --- Cast op ---

def test_cast_f32_to_i32():
    t = gpu.tensor([1.5, 2.7, -3.1, 0.0], shape=[4])
    result = gpu.cast(t, "int32")
    result.eval()
    vals = result.to_list()
    assert vals == [1, 2, -3, 0], f"Got {vals}"


def test_cast_noop():
    t = gpu.tensor([1.0, 2.0], shape=[2])
    result = gpu.cast(t, "float32")
    assert result.shape == [2]


def test_cast_chain():
    t = gpu.tensor([1.9, -2.7, 3.0], shape=[3])
    i = gpu.cast(t, "int32")
    f = gpu.cast(i, "float32")
    f.eval()
    vals = f.to_list()
    assert vals == [1.0, -2.0, 3.0], f"Got {vals}"


def test_cast_rejects_float64():
    t = gpu.tensor([1.0, 2.0], shape=[2])
    with pytest.raises(Exception):
        gpu.cast(t, "float64")


# --- Float16 dispatch through templates ---

def test_f16_add():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4], dtype="float16")
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4], dtype="float16")
    c = a + b
    c.eval()
    assert c.to_list() == [11.0, 22.0, 33.0, 44.0]


def test_f16_relu():
    a = gpu.tensor([-1.0, 0.0, 1.0, -0.5], shape=[4], dtype="float16")
    b = gpu.relu(a)
    b.eval()
    assert b.to_list() == [0.0, 0.0, 1.0, 0.0]


def test_f16_matmul():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2], dtype="float16")
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2], dtype="float16")
    c = gpu.matmul(a, b)
    c.eval()
    assert c.to_list() == [19.0, 22.0, 43.0, 50.0]


def test_f16_softmax():
    a = gpu.tensor([1.0, 2.0, 3.0], shape=[1, 3], dtype="float16")
    b = gpu.softmax(a)
    b.eval()
    vals = b.to_list()
    assert abs(sum(vals) - 1.0) < 0.01


def test_mixed_dtype_errors():
    a = gpu.tensor([1.0, 2.0], shape=[2], dtype="float32")
    b = gpu.tensor([1.0, 2.0], shape=[2], dtype="float16")
    with pytest.raises(Exception):
        _ = a + b
