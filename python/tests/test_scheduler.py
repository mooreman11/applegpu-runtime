import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


def test_backward_compatibility():
    """Existing API works unchanged with scheduler underneath."""
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = gpu.tensor([10.0, 20.0, 30.0, 40.0], [4])
    c = a + b
    result = c.to_list()
    assert result == [11.0, 22.0, 33.0, 44.0]


def test_register_deregister_container():
    cid = gpu.register_container(priority="normal", max_memory_mb=1, max_tensors=10, max_pending=5)
    assert cid > 0
    tensors = gpu.deregister_container(cid)
    assert tensors == []


def test_container_usage_query():
    cid = gpu.register_container(priority="normal", max_memory_mb=10, max_tensors=100, max_pending=50)
    bytes_used, count = gpu.container_usage(cid)
    assert bytes_used == 0
    assert count == 0
    gpu.deregister_container(cid)


def test_global_usage():
    before_bytes, before_count = gpu.global_usage()
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    after_bytes, after_count = gpu.global_usage()
    assert after_bytes > before_bytes
    assert after_count == before_count + 1


def test_queue_depth():
    assert gpu.queue_depth() >= 0
    a = gpu.tensor([1.0, 2.0], [2])
    b = gpu.tensor([3.0, 4.0], [2])
    c = a + b  # lazy
    job_id = gpu.submit_job(0, c)  # submit to default container
    assert gpu.queue_depth() >= 1
    result_id = gpu.run_next()
    assert result_id == job_id
    assert gpu.job_status(job_id) == "completed"


def test_submit_and_run_job():
    a = gpu.tensor([1.0, 2.0, 3.0], [3])
    b = gpu.tensor([4.0, 5.0, 6.0], [3])
    c = a + b
    job_id = gpu.submit_job(0, c)
    assert gpu.job_status(job_id) == "queued"
    gpu.run_next()
    assert gpu.job_status(job_id) == "completed"
    assert c.to_list() == [5.0, 7.0, 9.0]


def test_priority_scheduling():
    high_cid = gpu.register_container(priority="high", max_memory_mb=1, max_tensors=10, max_pending=10)
    low_cid = gpu.register_container(priority="low", max_memory_mb=1, max_tensors=10, max_pending=10)
    a = gpu.tensor([1.0, 2.0], [2])
    b = gpu.tensor([3.0, 4.0], [2])
    c_low = a + b
    c_high = a + b
    gpu.submit_job(low_cid, c_low)
    high_job = gpu.submit_job(high_cid, c_high)
    result_id = gpu.run_next()
    assert result_id == high_job
    # Clean up: run remaining job, then deregister
    gpu.run_next()
    gpu.deregister_container(high_cid)
    gpu.deregister_container(low_cid)


def test_admission_control():
    cid = gpu.register_container(priority="normal", max_memory_mb=1, max_tensors=10, max_pending=1)
    a = gpu.tensor([1.0], [1])
    b = gpu.tensor([2.0], [1])
    c1 = a + b
    c2 = a + b
    gpu.submit_job(cid, c1)
    with pytest.raises(ValueError):
        gpu.submit_job(cid, c2)
    # Clean up
    gpu.run_next()
    gpu.deregister_container(cid)
