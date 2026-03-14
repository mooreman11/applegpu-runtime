import applegpu_runtime as gpu


def test_tensor_create():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    assert t is not None


def test_tensor_to_list():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    result = gpu.to_list(t)
    assert result == [1.0, 2.0, 3.0, 4.0]


def test_tensor_shape():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    assert gpu.shape(t) == [2, 3]
