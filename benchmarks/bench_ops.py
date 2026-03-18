"""
Benchmark individual GPU operation performance.

Measures throughput for core operations at various tensor sizes.

Usage:
    uv run python benchmarks/bench_ops.py
    uv run python benchmarks/bench_ops.py --size 4096
"""

import argparse
import os
import time

import numpy as np

# Suppress Apple Metal diagnostic noise
os.environ.setdefault("MTL_DEBUG_LAYER", "0")

import applegpu_runtime as gpu


def bench_op(name, op_fn, n_iters=100, warmup=10):
    """Benchmark a single operation. Returns median time in ms."""
    # Warmup
    for _ in range(warmup):
        op_fn()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        op_fn()
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        times.append(elapsed)

    times.sort()
    median = times[len(times) // 2]
    return median


def run_benchmarks(size):
    """Run benchmarks at a given matrix size."""
    gpu.init_backend()
    print(f"Device: {gpu.device_name()}")
    print(f"Matrix size: {size}x{size} (float32)")
    print()

    a_np = np.random.randn(size, size).astype(np.float32)
    b_np = np.random.randn(size, size).astype(np.float32)
    a = gpu.from_numpy(a_np)
    b = gpu.from_numpy(b_np)

    # 1D vector for reductions
    v_np = np.random.randn(size * size).astype(np.float32)
    v = gpu.from_numpy(v_np)

    results = []

    # Element-wise ops
    def do_add():
        r = gpu.add(a, b)
        gpu.eval(r)

    def do_mul():
        r = gpu.mul(a, b)
        gpu.eval(r)

    def do_sigmoid():
        r = gpu.sigmoid(a)
        gpu.eval(r)

    def do_gelu():
        r = gpu.gelu(a)
        gpu.eval(r)

    def do_fused_add_relu():
        r = gpu.relu(gpu.add(a, b))
        gpu.eval(r)

    def do_fused_add_sigmoid():
        r = gpu.sigmoid(gpu.add(a, b))
        gpu.eval(r)

    # Matmul
    def do_matmul():
        r = gpu.matmul(a, b)
        gpu.eval(r)

    # Reductions
    v_2d = gpu.from_numpy(np.random.randn(size, size).astype(np.float32))

    def do_softmax():
        r = gpu.softmax(v_2d)
        gpu.eval(r)

    def do_sum():
        r = gpu.sum(v_2d)
        gpu.eval(r)

    def do_amax():
        r = gpu.amax(v_2d)
        gpu.eval(r)

    def do_var():
        r = gpu.var(v_2d)
        gpu.eval(r)

    # Layer norm (requires gamma/beta scale/bias vectors)
    gamma = gpu.from_numpy(np.ones(size, dtype=np.float32))
    beta = gpu.from_numpy(np.zeros(size, dtype=np.float32))

    def do_layer_norm():
        r = gpu.layer_norm(v_2d, gamma, beta, 1e-5)
        gpu.eval(r)

    ops = [
        ("add", do_add),
        ("mul", do_mul),
        ("sigmoid", do_sigmoid),
        ("gelu", do_gelu),
        ("add+relu (fused)", do_fused_add_relu),
        ("add+sigmoid (fused)", do_fused_add_sigmoid),
        ("matmul", do_matmul),
        ("softmax", do_softmax),
        ("sum", do_sum),
        ("amax", do_amax),
        ("var", do_var),
        ("layer_norm", do_layer_norm),
    ]

    elements = size * size
    print(f"{'Operation':<25} {'Median (ms)':<14} {'GB/s':<10}")
    print("-" * 50)

    for name, fn in ops:
        median_ms = bench_op(name, fn)
        # Rough bandwidth: assume 2 reads + 1 write of float32 per element
        bytes_moved = elements * 4 * 3
        gbps = bytes_moved / (median_ms / 1000) / 1e9
        print(f"{name:<25} {median_ms:<14.3f} {gbps:<10.1f}")
        results.append((name, median_ms, gbps))

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU ops")
    parser.add_argument("--size", type=int, default=1024, help="Matrix dimension NxN")
    parser.add_argument("--sizes", nargs="+", type=int, help="Multiple sizes to sweep")
    args = parser.parse_args()

    sizes = args.sizes or [args.size]

    for size in sizes:
        print(f"\n{'=' * 50}")
        run_benchmarks(size)


if __name__ == "__main__":
    main()
