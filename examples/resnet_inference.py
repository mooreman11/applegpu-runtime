"""ResNet-18 inference on Apple Silicon Metal GPU via PyTorch device backend.

Usage:
    python examples/resnet_inference.py
    python examples/resnet_inference.py --model resnet50
    python examples/resnet_inference.py --batch-size 8
"""

import argparse
import time
import torch
import torchvision.models as models
import applegpu_runtime as gpu
from applegpu_runtime.torch_backend import ApplegpuTensor


def main():
    parser = argparse.ArgumentParser(description="ResNet inference on Metal GPU")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    args = parser.parse_args()

    gpu.init_backend()
    gpu.enable_torch_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)

    # Load model
    print(f"Loading {args.model} (random weights)...")
    model_fn = getattr(models, args.model)
    model = model_fn(weights=None)
    model.eval()

    # Move to Metal GPU
    print("Moving to applegpu...")
    model = gpu.to_applegpu(model)

    # Create input
    x = ApplegpuTensor.from_torch(torch.randn(args.batch_size, 3, 224, 224))

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = model(x)

    # Benchmark
    print(f"Benchmarking ({args.iterations} iterations, batch_size={args.batch_size})...")
    start = time.time()
    for _ in range(args.iterations):
        with torch.no_grad():
            output = model(x)
    elapsed = time.time() - start

    result = output.to_torch_cpu()
    top5 = torch.topk(result[0], 5)

    print(f"\nResults:")
    print(f"  Output shape: {result.shape}")
    print(f"  Top-5 class indices: {top5.indices.tolist()}")
    print(f"  Top-5 scores: {[f'{v:.3f}' for v in top5.values.tolist()]}")
    print(f"\nPerformance:")
    print(f"  Total: {elapsed:.2f}s for {args.iterations} iterations")
    print(f"  Per inference: {elapsed/args.iterations*1000:.1f}ms")
    print(f"  Throughput: {args.iterations * args.batch_size / elapsed:.1f} images/sec")


if __name__ == "__main__":
    main()
