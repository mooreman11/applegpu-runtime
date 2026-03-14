import applegpu_runtime as gpu


def test_pool_stats():
    gpu.init_backend()
    stats = gpu.pool_stats()
    assert "hits" in stats
    assert "misses" in stats
    assert "pooled_bytes" in stats
    assert "bucket_count" in stats


def test_pool_drain():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], [4])
    c = a + b
    c.to_list()
    gpu.destroy(c)
    gpu.pool_drain()
    stats = gpu.pool_stats()
    assert stats["pooled_bytes"] == 0


def test_set_pool_watermark():
    gpu.init_backend()
    gpu.set_pool_watermark(1)
    a = gpu.tensor([1.0] * 100, [100])
    b = gpu.tensor([2.0] * 100, [100])
    c = a + b
    result = c.to_list()
    assert len(result) == 100
    assert result[0] == 3.0


def test_pool_transparent_to_existing_api():
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], [4])
    c = a + b
    assert c.to_list() == [11.0, 22.0, 33.0, 44.0]
    d = (a * b).relu()
    result = d.to_list()
    assert len(result) == 4
    assert all(v >= 0 for v in result)
