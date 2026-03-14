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

def test_compute_rejects_non_float():
    a = gpu.tensor([1, 2, 3], shape=[3], dtype="int32")
    b = gpu.tensor([4, 5, 6], shape=[3], dtype="int32")
    with pytest.raises(ValueError, match="[Cc]ompute|kernel"):
        c = a + b


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
