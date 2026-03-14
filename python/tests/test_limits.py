import applegpu_runtime as gpu


def test_memory_usage_tracking():
    gpu.init_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)  # unlimited
    initial = gpu.memory_usage()
    t = gpu.tensor([1.0] * 1000, shape=[1000])
    after = gpu.memory_usage()
    assert after > initial
    assert after - initial == 1000 * 4  # 1000 floats * 4 bytes


def test_tensor_count_tracking():
    gpu.init_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)
    count_before = gpu.tensor_count()
    a = gpu.tensor([1.0, 2.0], shape=[2])
    b = gpu.tensor([3.0, 4.0], shape=[2])
    assert gpu.tensor_count() >= count_before + 2


def test_tensor_size_limit():
    gpu.init_backend()
    gpu.set_limits(max_tensor_size_mb=1, max_memory_mb=0, max_tensors=0)
    # 1MB = 262144 floats. Try to create ~1.14 MB tensor (> 1 MB limit)
    try:
        t = gpu.tensor([1.0] * 300000, shape=[300000])
        assert False, "Should have raised"
    except ValueError as e:
        assert "limit" in str(e).lower() or "quota" in str(e).lower()
    # Reset to unlimited
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)


def test_tensor_count_limit():
    gpu.init_backend()
    current = gpu.tensor_count()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=current + 3)
    created = []
    try:
        for i in range(10):
            created.append(gpu.tensor([float(i)], shape=[1]))
        assert False, "Should have raised"
    except ValueError as e:
        assert "limit" in str(e).lower() or "quota" in str(e).lower()
    # Reset
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)


def test_set_limits_unlimited():
    gpu.init_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)
    t = gpu.tensor([1.0] * 10000, shape=[10000])
    assert t.shape == [10000]
