def test_import():
    import applegpu_runtime as gpu
    assert gpu.__version__


def test_version_string():
    import applegpu_runtime as gpu
    assert isinstance(gpu.version(), str)
    assert len(gpu.version()) > 0
