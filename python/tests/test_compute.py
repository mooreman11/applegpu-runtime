import applegpu_runtime as gpu


def test_add_basic():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    c = gpu.add(a, b)
    result = gpu.to_list(c)
    assert result == [11.0, 22.0, 33.0, 44.0]


def test_add_2d():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = gpu.add(a, b)
    result = gpu.to_list(c)
    assert result == [6.0, 8.0, 10.0, 12.0]
    assert gpu.shape(c) == [2, 2]


def test_add_large():
    gpu.init_backend()
    n = 10000
    a = gpu.tensor([1.0] * n, shape=[n])
    b = gpu.tensor([2.0] * n, shape=[n])
    c = gpu.add(a, b)
    result = gpu.to_list(c)
    assert all(x == 3.0 for x in result)
    assert len(result) == n
