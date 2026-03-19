"""MLP benchmark: C++ PrivateUse1 backend vs CPU.

Runs forward + backward pass of a simple MLP and compares wall time.
Uses only ops registered natively: mm, add, relu.

Usage:
    uv run python benchmarks/bench_mlp_cpp.py
    uv run python benchmarks/bench_mlp_cpp.py --hidden 256 --layers 4 --iters 100
"""
import argparse
import sys
import time
import types

# Prevent applegpu_runtime's __init__.py from loading the PyO3 native extension.
# Both the PyO3 .so and the C++ backend .so link libAppleGPUBridge.a — loading
# both causes ObjC class conflicts. We only need the C++ backend here.
if 'applegpu_runtime' not in sys.modules:
    _stub = types.ModuleType('applegpu_runtime')
    _stub.__path__ = [__import__('os').path.join(
        __import__('os').path.dirname(__file__), '..', 'python', 'applegpu_runtime')]
    sys.modules['applegpu_runtime'] = _stub

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden, num_layers, out_features):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def bench(device_name, model, x, y, criterion, n_iters):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    start = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start

    return {
        "device": device_name,
        "total_ms": elapsed * 1000,
        "per_iter_ms": (elapsed / n_iters) * 1000,
        "final_loss": loss.cpu().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--input", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--output", type=int, default=1)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    print(f"MLP Benchmark: batch={args.batch}, input={args.input}, "
          f"hidden={args.hidden}, layers={args.layers}, iters={args.iters}")
    print("=" * 60)

    torch.manual_seed(42)
    x_cpu = torch.randn(args.batch, args.input)
    y_cpu = torch.randn(args.batch, args.output)
    criterion = nn.MSELoss()

    # CPU baseline
    model_cpu = SimpleMLP(args.input, args.hidden, args.layers, args.output)
    result_cpu = bench("cpu", model_cpu, x_cpu, y_cpu, criterion, args.iters)

    # C++ backend
    result_gpu = None
    try:
        from applegpu_runtime.cpp_backend import load_cpp_backend
        load_cpp_backend()

        model_gpu = SimpleMLP(args.input, args.hidden, args.layers, args.output)
        model_gpu = model_gpu.to("applegpu")
        x_gpu = x_cpu.to("applegpu")
        y_gpu = y_cpu.to("applegpu")

        result_gpu = bench("applegpu", model_gpu, x_gpu, y_gpu, criterion, args.iters)
    except Exception as e:
        print(f"C++ backend error: {e}")
        import traceback
        traceback.print_exc()

    # Results
    print(f"\n{'Device':<15} {'Total (ms)':>12} {'Per-iter (ms)':>14} {'Loss':>10}")
    print("-" * 53)
    print(f"{'CPU':<15} {result_cpu['total_ms']:>12.2f} {result_cpu['per_iter_ms']:>14.2f} {result_cpu['final_loss']:>10.4f}")
    if result_gpu:
        print(f"{'applegpu':<15} {result_gpu['total_ms']:>12.2f} {result_gpu['per_iter_ms']:>14.2f} {result_gpu['final_loss']:>10.4f}")
        speedup = result_cpu['per_iter_ms'] / result_gpu['per_iter_ms']
        print(f"\nSpeedup: {speedup:.2f}x {'(GPU faster)' if speedup > 1 else '(CPU faster)'}")


if __name__ == "__main__":
    main()
