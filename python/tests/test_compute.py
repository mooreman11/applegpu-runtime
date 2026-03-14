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


def test_sub():
    gpu.init_backend()
    a = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    b = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    c = gpu.sub(a, b)
    assert gpu.to_list(c) == [9.0, 18.0, 27.0, 36.0]


def test_mul():
    gpu.init_backend()
    a = gpu.tensor([2.0, 3.0, 4.0, 5.0], shape=[4])
    b = gpu.tensor([10.0, 10.0, 10.0, 10.0], shape=[4])
    assert gpu.to_list(gpu.mul(a, b)) == [20.0, 30.0, 40.0, 50.0]


def test_div():
    gpu.init_backend()
    a = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    b = gpu.tensor([2.0, 4.0, 5.0, 8.0], shape=[4])
    assert gpu.to_list(gpu.div(a, b)) == [5.0, 5.0, 6.0, 5.0]


def test_neg():
    gpu.init_backend()
    a = gpu.tensor([1.0, -2.0, 3.0, -4.0], shape=[4])
    assert gpu.to_list(gpu.neg(a)) == [-1.0, 2.0, -3.0, 4.0]


def test_relu():
    gpu.init_backend()
    a = gpu.tensor([-1.0, 0.0, 3.0, -4.0], shape=[4])
    assert gpu.to_list(gpu.relu(a)) == [0.0, 0.0, 3.0, 0.0]


def test_exp():
    gpu.init_backend()
    import math
    a = gpu.tensor([0.0, 1.0], shape=[2])
    result = gpu.to_list(gpu.exp(a))
    assert abs(result[0] - 1.0) < 1e-6
    assert abs(result[1] - math.e) < 1e-5


def test_log():
    gpu.init_backend()
    import math
    a = gpu.tensor([1.0, math.e], shape=[2])
    result = gpu.to_list(gpu.log(a))
    assert abs(result[0] - 0.0) < 1e-6
    assert abs(result[1] - 1.0) < 1e-5


def test_sqrt():
    gpu.init_backend()
    a = gpu.tensor([4.0, 9.0, 16.0, 25.0], shape=[4])
    assert gpu.to_list(gpu.sqrt(a)) == [2.0, 3.0, 4.0, 5.0]


def test_matmul_2x2():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = gpu.matmul(a, b)
    assert gpu.to_list(c) == [19.0, 22.0, 43.0, 50.0]
    assert gpu.shape(c) == [2, 2]


def test_matmul_non_square():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = gpu.tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], shape=[3, 2])
    c = gpu.matmul(a, b)
    assert gpu.to_list(c) == [58.0, 64.0, 139.0, 154.0]
    assert gpu.shape(c) == [2, 2]
