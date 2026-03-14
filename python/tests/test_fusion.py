import applegpu_runtime as gpu


def test_fused_add_relu():
    gpu.init_backend()
    a = gpu.tensor([1.0, -2.0, 3.0, -4.0], shape=[4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    c = (a + b).relu()
    assert c.to_list() == [11.0, 18.0, 33.0, 36.0]


def test_fused_chain():
    gpu.init_backend()
    a = gpu.tensor([1.0, 4.0, 9.0, 16.0], shape=[4])
    c = a.sqrt().neg().relu()
    assert c.to_list() == [0.0, 0.0, 0.0, 0.0]


def test_fused_mul_add():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    b = gpu.tensor([10.0, 10.0, 10.0, 10.0], shape=[4])
    c = gpu.tensor([5.0, 5.0, 5.0, 5.0], shape=[4])
    # (a * b) + c should fuse: mul then add
    d = (a * b) + c
    assert d.to_list() == [15.0, 25.0, 35.0, 45.0]
