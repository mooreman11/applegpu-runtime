"""Tests for concurrent Metal queue dispatch (Phase I concurrency)."""
import applegpu_runtime as gpu


def test_diamond_graph():
    """A -> B, A -> C, B+C -> D. B and C should execute on separate queues."""
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = a + a          # depends on a
    c = a * a          # depends on a, independent of b
    d = b + c          # depends on b and c
    d.eval()
    result = d.to_list()
    expected = [1+1+1*1, 2+2+2*2, 3+3+3*3, 4+4+4*4]
    assert result == expected, f"Got {result}, expected {expected}"


def test_wide_independent_ops():
    """4 independent operations that can all execute in parallel."""
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = a + a
    c = a * a
    d = a - a
    e = gpu.relu(a)
    # Force a join point
    result = b + c + d + e
    result.eval()
    vals = result.to_list()
    # b=2,4,6,8  c=1,4,9,16  d=0,0,0,0  e=1,2,3,4
    expected = [2+1+0+1, 4+4+0+2, 6+9+0+3, 8+16+0+4]
    assert vals == expected, f"Got {vals}, expected {expected}"


def test_linear_chain_fast_path():
    """Linear chain should take the single-CB fast path and produce correct results."""
    gpu.init_backend()
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], [4])
    b = a + a           # 2, 4, 6, 8
    c = b + a           # 3, 6, 9, 12
    c.eval()
    result = c.to_list()
    expected = [3.0, 6.0, 9.0, 12.0]
    assert result == expected, f"Got {result}, expected {expected}"
