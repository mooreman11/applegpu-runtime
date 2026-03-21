#!/usr/bin/env python3
"""Benchmark: CPU vs MPS vs applegpu MLP training.

Run: uv run python python/tests/bench_comparison.py

NOTE: MPS must be tested BEFORE loading applegpu (PrivateUse1 registration
interferes with MPS). This script handles the ordering automatically.
"""
import time, torch

class MLP(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1 = torch.nn.Linear(h, h)
        self.fc2 = torch.nn.Linear(h, h)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def sync(device):
    if device == 'mps': torch.mps.synchronize()
    elif device == 'applegpu': torch.applegpu.synchronize()

def bench(device, h, batch=32, warmup=5, iters=50):
    torch.manual_seed(42)
    model = MLP(h).to(device)
    x = torch.randn(batch, h, device=device)
    target = torch.randn(batch, h, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(warmup):
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward(); opt.step(); opt.zero_grad()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward(); opt.step(); opt.zero_grad()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000

if __name__ == '__main__':
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    sizes = [64, 256, 1024, 4096]
    results = {}

    # Phase 1: CPU + MPS (before applegpu load)
    for device, label in [('cpu', 'CPU'), ('mps', 'MPS')]:
        if device == 'mps' and not has_mps: continue
        results[label] = {}
        for h in sizes:
            try: results[label][h] = bench(device, h)
            except: results[label][h] = None

    # Phase 2: applegpu
    try:
        from applegpu_runtime.cpp_backend import load_cpp_backend
        load_cpp_backend()
        results['applegpu'] = {}
        for h in sizes:
            try: results['applegpu'][h] = bench('applegpu', h)
            except: results['applegpu'][h] = None
    except: pass

    print(f"\n{'TRAINING (ms/step)':>20} " + " ".join(f"{'h='+str(h):>10}" for h in sizes))
    print("-" * 70)
    for label in ['CPU', 'MPS', 'applegpu']:
        if label not in results: continue
        row = f"{label:>20}"
        for h in sizes:
            v = results[label].get(h)
            row += f" {v:>9.3f}" if v else f" {'ERR':>9}"
        print(row)

    print(f"\n{'SPEEDUP vs CPU':>20} " + " ".join(f"{'h='+str(h):>10}" for h in sizes))
    print("-" * 70)
    for label in ['MPS', 'applegpu']:
        if label not in results: continue
        row = f"{label:>20}"
        for h in sizes:
            v = results[label].get(h)
            c = results['CPU'].get(h)
            row += f" {c/v:>9.2f}x" if v and c else f" {'N/A':>9}"
        print(row)
