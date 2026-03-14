import applegpu_runtime as gpu


def test_dtype_size():
    assert gpu.dtype_size("float32") == 4
    assert gpu.dtype_size("float16") == 2
    assert gpu.dtype_size("int8") == 1


def test_dtype_size_invalid():
    try:
        gpu.dtype_size("invalid")
        assert False, "Should have raised"
    except ValueError:
        pass
