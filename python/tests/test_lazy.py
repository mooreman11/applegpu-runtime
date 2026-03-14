import applegpu_runtime as gpu


def test_eval_materializes():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    c = gpu.add(a, b)
    # c is lazy — eval materializes it
    gpu.eval(c)
    assert gpu.to_list(c) == [11.0, 22.0, 33.0, 44.0]


def test_to_list_auto_evals():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], shape=[4])
    c = gpu.add(a, b)
    # to_list should auto-eval
    assert gpu.to_list(c) == [11.0, 22.0, 33.0, 44.0]


def test_lazy_chain():
    gpu.init_backend()
    a = gpu.tensor([1.0, -2.0, 3.0, -4.0], shape=[4])
    b = gpu.neg(a)
    c = gpu.relu(b)  # relu(neg(a)) = relu([-1, 2, -3, 4]) = [0, 2, 0, 4]
    assert gpu.to_list(c) == [0.0, 2.0, 0.0, 4.0]


def test_destroy_frees_memory():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0], shape=[2])
    gpu.destroy(a)
    try:
        gpu.to_list(a)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_shape_works_on_lazy():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    c = gpu.add(a, b)
    # shape should work even before eval
    assert gpu.shape(c) == [2, 3]


def test_lazy_matmul_chain():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = gpu.matmul(a, b)
    d = gpu.neg(c)
    assert gpu.to_list(d) == [-19.0, -22.0, -43.0, -50.0]
