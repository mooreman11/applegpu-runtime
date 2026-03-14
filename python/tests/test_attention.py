import applegpu_runtime as gpu


def test_softmax_rows_sum_to_one():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 1.0, 1.0, 1.0], shape=[2, 3])
    s = a.softmax()
    result = s.to_list()
    row0_sum = sum(result[0:3])
    row1_sum = sum(result[3:6])
    assert abs(row0_sum - 1.0) < 1e-5
    assert abs(row1_sum - 1.0) < 1e-5


def test_softmax_uniform():
    gpu.init_backend()
    a = gpu.tensor([1.0, 1.0, 1.0, 1.0], shape=[1, 4])
    result = a.softmax().to_list()
    for v in result:
        assert abs(v - 0.25) < 1e-5


def test_transpose_2x3():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    t = a.transpose()
    assert t.shape == [3, 2]
    assert t.to_list() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]


def test_transpose_square():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    t = a.transpose()
    assert t.to_list() == [1.0, 3.0, 2.0, 4.0]


def test_attention_identity():
    gpu.init_backend()
    q = gpu.tensor([1.0, 0.0, 0.0, 1.0], shape=[2, 2])
    k = gpu.tensor([1.0, 0.0, 0.0, 1.0], shape=[2, 2])
    v = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    out = gpu.attention(q, k, v)
    assert out.shape == [2, 2]
    result = out.to_list()
    assert len(result) == 4
    for val in result:
        assert 0.0 <= val <= 10.0


def test_attention_shape():
    gpu.init_backend()
    q = gpu.tensor([1.0] * 32, shape=[4, 8])
    k = gpu.tensor([1.0] * 32, shape=[4, 8])
    v = gpu.tensor([1.0] * 64, shape=[4, 16])
    out = gpu.attention(q, k, v)
    assert out.shape == [4, 16]
    result = out.to_list()
    assert len(result) == 64
