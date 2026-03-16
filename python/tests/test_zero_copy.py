"""Tests for zero-copy tensor transfers (from_numpy_shared, from_torch_shared, aligned_numpy)."""

import numpy as np
import applegpu_runtime as gpu
import pytest
import sys

gpu.init_backend()

PAGE_SIZE = 16384  # Apple Silicon


def test_aligned_numpy_creates_page_aligned_array():
    arr = gpu.aligned_numpy(shape=(1024, 1024), dtype="float32")
    assert arr.shape == (1024, 1024)
    assert arr.dtype == np.float32
    assert arr.ctypes.data % PAGE_SIZE == 0
    nbytes = arr.size * arr.itemsize
    assert nbytes % PAGE_SIZE == 0


def test_aligned_numpy_float64():
    arr = gpu.aligned_numpy(shape=(2048,), dtype="float64")
    assert arr.dtype == np.float64
    assert arr.ctypes.data % PAGE_SIZE == 0


def test_aligned_numpy_int32():
    arr = gpu.aligned_numpy(shape=(4096,), dtype="int32")
    assert arr.dtype == np.int32
    assert arr.ctypes.data % PAGE_SIZE == 0


def test_aligned_numpy_writable():
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    arr[:] = 42.0
    assert arr[0] == 42.0
    assert arr[-1] == 42.0


def test_from_numpy_shared_basic():
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    arr[:] = np.arange(4096, dtype=np.float32)
    t = gpu.from_numpy_shared(arr)
    result = t.to_list()
    assert result[:5] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert result[-1] == 4095.0


def test_from_numpy_shared_rejects_misaligned():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="page-aligned"):
        gpu.from_numpy_shared(arr)


def test_from_numpy_shared_rejects_non_page_length():
    # aligned_numpy now rejects non-page-multiple sizes at creation time
    # 100 * 4 = 400 bytes, not a page multiple
    with pytest.raises(ValueError, match="page size"):
        gpu.aligned_numpy(shape=(100,), dtype="float32")


def test_from_numpy_shared_rejects_non_contiguous():
    arr = gpu.aligned_numpy(shape=(8192,), dtype="float32")
    # Slice with step creates a non-contiguous view
    non_contiguous = arr[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        gpu.from_numpy_shared(non_contiguous)


def test_shared_tensor_as_input():
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    arr[:] = 2.0
    t = gpu.from_numpy_shared(arr)
    result = t + t  # shared tensor used as input
    result.eval()
    vals = result.to_list()
    assert vals[0] == 4.0
    assert vals[-1] == 4.0


def test_shared_tensor_sees_mutations():
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    arr[:] = 1.0
    t = gpu.from_numpy_shared(arr)
    arr[:] = 99.0  # mutate source
    vals = t.to_list()
    assert vals[0] == 99.0  # shared memory -- sees mutation


def test_shared_tensor_refcount():
    # Assumption: Metal buffer deallocation is synchronous when triggered by gpu.destroy().
    # This means the deallocator callback (which calls Py_DecRef) fires immediately within
    # the gpu.destroy() call, so the refcount is decremented before the next assertion.
    # If Metal ever defers deallocation (e.g., pending command buffer references), this
    # test would need to account for async cleanup.
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    initial_refcount = sys.getrefcount(arr)
    t = gpu.from_numpy_shared(arr)
    # Refcount should have increased by 1 (Py_IncRef)
    assert sys.getrefcount(arr) == initial_refcount + 1
    gpu.destroy(t)
    # After destroy, refcount should return to original
    # (Metal deallocator fires Py_DecRef)
    assert sys.getrefcount(arr) == initial_refcount


def test_pool_not_affected_by_shared():
    stats_before = gpu.pool_stats()
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    t = gpu.from_numpy_shared(arr)
    gpu.destroy(t)
    stats_after = gpu.pool_stats()
    # Pool should not have gained a buffer from the shared tensor
    assert stats_after["pooled_bytes"] == stats_before["pooled_bytes"]


def test_from_numpy_shared_2d():
    arr = gpu.aligned_numpy(shape=(64, 64), dtype="float32")
    arr[:] = np.ones((64, 64), dtype=np.float32)
    t = gpu.from_numpy_shared(arr)
    s = gpu.shape(t)
    assert s == [64, 64]
    vals = t.to_list()
    assert vals[0] == 1.0


def test_aligned_numpy_large():
    # 1M elements = 4MB, which is 256 pages
    arr = gpu.aligned_numpy(shape=(1024, 1024), dtype="float32")
    assert arr.ctypes.data % PAGE_SIZE == 0
    nbytes = arr.size * arr.itemsize
    assert nbytes % PAGE_SIZE == 0
    assert nbytes == 4 * 1024 * 1024


def test_shared_tensor_works_as_op_input():
    """Verify that a shared (borrowed) tensor works correctly as an input to ops.

    Note on immutability enforcement: ImmutableBuffer / BufferKind::Borrowed is enforced
    at the Rust compute dispatch level -- borrowed buffers cannot be used as op output
    buffers. This cannot be triggered from the Python API since output buffers are always
    pool-allocated (BufferKind::Owned). The invariant is tested in Rust unit tests instead.
    """
    arr = gpu.aligned_numpy(shape=(4096,), dtype="float32")
    arr[:] = 3.0
    shared = gpu.from_numpy_shared(arr)
    # Use shared tensor as input to multiple ops
    result = shared + shared
    result.eval()
    vals = result.to_list()
    assert vals[0] == 6.0
    assert vals[-1] == 6.0
