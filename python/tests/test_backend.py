import applegpu_runtime as gpu


def test_init_backend():
    runtime = gpu.init_backend()
    assert runtime is not None


def test_init_backend_returns_backend_name():
    runtime = gpu.init_backend()
    assert runtime["backend"] in ("mlx", "vm")


def test_device_name():
    gpu.init_backend()
    name = gpu.device_name()
    assert isinstance(name, str)
    assert "Apple" in name
