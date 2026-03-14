import applegpu_runtime as gpu


def test_tensor_is_gpu_tensor():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    assert isinstance(t, gpu.GpuTensor)


def test_shape_property():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    assert t.shape == [2, 3]


def test_to_list_method():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    assert t.to_list() == [1.0, 2.0, 3.0]


def test_eval_method():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0], shape=[2])
    b = gpu.tensor([3.0, 4.0], shape=[2])
    c = a + b
    c.eval()
    assert c.to_list() == [4.0, 6.0]


def test_add_operator():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    b = gpu.tensor([10.0, 20.0, 30.0], shape=[3])
    c = a + b
    assert isinstance(c, gpu.GpuTensor)
    assert c.to_list() == [11.0, 22.0, 33.0]


def test_sub_operator():
    gpu.init_backend()
    a = gpu.tensor([10.0, 20.0], shape=[2])
    b = gpu.tensor([1.0, 2.0], shape=[2])
    assert (a - b).to_list() == [9.0, 18.0]


def test_mul_operator():
    gpu.init_backend()
    a = gpu.tensor([2.0, 3.0], shape=[2])
    b = gpu.tensor([4.0, 5.0], shape=[2])
    assert (a * b).to_list() == [8.0, 15.0]


def test_div_operator():
    gpu.init_backend()
    a = gpu.tensor([10.0, 20.0], shape=[2])
    b = gpu.tensor([2.0, 5.0], shape=[2])
    assert (a / b).to_list() == [5.0, 4.0]


def test_matmul_operator():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = a @ b
    assert c.to_list() == [19.0, 22.0, 43.0, 50.0]
    assert c.shape == [2, 2]


def test_neg_operator():
    gpu.init_backend()
    a = gpu.tensor([1.0, -2.0, 3.0], shape=[3])
    b = -a
    assert b.to_list() == [-1.0, 2.0, -3.0]


def test_chained_operators():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    # (a + b) * a = [11, 22, 33, 44] * [1, 2, 3, 4] = [11, 44, 99, 176]
    c = (a + b) * a
    assert c.to_list() == [11.0, 44.0, 99.0, 176.0]


def test_repr():
    gpu.init_backend()
    t = gpu.tensor([1.0, 2.0, 3.0], shape=[3])
    r = repr(t)
    assert "GpuTensor" in r
    assert "[3]" in r
    assert "materialized" in r


def test_repr_lazy():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0], shape=[2])
    b = gpu.tensor([3.0, 4.0], shape=[2])
    c = a + b
    r = repr(c)
    assert "lazy" in r


def test_unary_methods():
    gpu.init_backend()
    a = gpu.tensor([4.0, 9.0, 16.0], shape=[3])
    assert a.sqrt().to_list() == [2.0, 3.0, 4.0]

    b = gpu.tensor([-1.0, 2.0, -3.0], shape=[3])
    assert b.relu().to_list() == [0.0, 2.0, 0.0]
