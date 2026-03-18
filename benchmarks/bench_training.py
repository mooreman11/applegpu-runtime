"""
Benchmark applegpu vs CPU training performance.

Runs standard PyTorch models (LSTM, GRU, Transformer, CNN) on synthetic data
and compares wall-clock time, throughput, and loss convergence.

Usage:
    uv run python benchmarks/bench_training.py
    uv run python benchmarks/bench_training.py --models lstm gru --epochs 5
    uv run python benchmarks/bench_training.py --batch-size 128 --no-cpu
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definitions (self-contained, no external dependencies)
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])


class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def create_model(model_type, input_size, hidden_size, num_layers, output_size, seq_len):
    if model_type == "lstm":
        return LSTMModel(input_size, hidden_size, num_layers, output_size)
    elif model_type == "gru":
        return GRUModel(input_size, hidden_size, num_layers, output_size)
    elif model_type == "transformer":
        return TransformerModel(input_size, d_model=hidden_size, nhead=8,
                                num_layers=num_layers, output_size=output_size)
    elif model_type == "cnn":
        return CNNModel(input_size, hidden_size, output_size, seq_len)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def generate_data(n_samples, seq_len, n_features):
    """Deterministic synthetic time-series data."""
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, n_samples + seq_len)
    base = np.sin(t) + 0.5 * np.sin(2 * t) + np.random.randn(len(t)) * 0.1
    X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    for i in range(n_samples):
        w = base[i:i + seq_len]
        X[i, :, 0] = w
        for f in range(1, n_features):
            X[i, :, f] = w * (1 + 0.05 * np.random.randn(seq_len))
        y[i, 0] = base[i + seq_len]
    return torch.from_numpy(X), torch.from_numpy(y)


def bench_one(model_type, device, X, y, batch_size, epochs, hidden_size, num_layers, seq_len):
    """Run a single benchmark. Returns dict with results."""
    input_size = X.shape[2]
    model = create_model(model_type, input_size, hidden_size, num_layers, 1, seq_len)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if device == "applegpu":
        import applegpu_runtime as gpu
        model = gpu.to_applegpu(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Batch the data
    n = X.shape[0] - (X.shape[0] % batch_size)
    X_batched = X[:n].reshape(-1, batch_size, X.shape[1], X.shape[2])
    y_batched = y[:n].reshape(-1, batch_size, 1)

    def to_dev(t):
        if device == "applegpu":
            return gpu.to_applegpu(t.contiguous())
        return t

    # Warmup
    with torch.no_grad():
        _ = model(to_dev(X_batched[0]))

    epoch_times = []
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()

        for i in range(X_batched.shape[0]):
            bx = to_dev(X_batched[i])
            by = to_dev(y_batched[i])

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if device == "applegpu" and hasattr(loss, "to_torch_cpu"):
                epoch_loss += loss.to_torch_cpu().item()
            else:
                epoch_loss += loss.item()
            n_batches += 1

        elapsed = time.perf_counter() - t0
        epoch_times.append(elapsed)
        losses.append(epoch_loss / max(n_batches, 1))

    total_time = sum(epoch_times)
    total_samples = n * epochs
    sps = total_samples / total_time if total_time > 0 else 0

    return {
        "model_type": model_type,
        "device": device,
        "param_count": param_count,
        "epochs": epochs,
        "total_time_s": total_time,
        "avg_epoch_s": total_time / epochs,
        "samples_per_sec": sps,
        "final_loss": losses[-1],
        "losses": losses,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results):
    """Print comparison table."""
    by_model = {}
    for r in results:
        by_model.setdefault(r["model_type"], []).append(r)

    print("\n" + "=" * 78)
    print("BENCHMARK RESULTS")
    print("=" * 78)

    for model_type, runs in by_model.items():
        print(f"\n--- {model_type.upper()} ({runs[0]['param_count']:,} params) ---")
        print(f"{'Device':<12} {'Total (s)':<11} {'Epoch Avg':<11} "
              f"{'Samples/s':<13} {'Final Loss':<12}")
        print("-" * 60)

        cpu_time = None
        for r in runs:
            print(f"{r['device']:<12} {r['total_time_s']:<11.2f} "
                  f"{r['avg_epoch_s']:<11.3f} {r['samples_per_sec']:<13.0f} "
                  f"{r['final_loss']:<12.6f}")
            if r["device"] == "cpu":
                cpu_time = r["total_time_s"]

        if cpu_time:
            for r in runs:
                if r["device"] == "applegpu":
                    speedup = cpu_time / r["total_time_s"]
                    print(f"  -> applegpu speedup: {speedup:.2f}x")

    # Epoch loss convergence
    print("\n" + "=" * 78)
    print("LOSS CONVERGENCE")
    print("=" * 78)
    for model_type, runs in by_model.items():
        print(f"\n{model_type.upper()}:")
        header = f"{'Epoch':<8}"
        for r in runs:
            header += f"{r['device']:<14}"
        print(header)
        print("-" * (8 + 14 * len(runs)))
        for i in range(runs[0]["epochs"]):
            row = f"{i + 1:<8}"
            for r in runs:
                row += f"{r['losses'][i]:<14.6f}"
            print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark applegpu training")
    parser.add_argument("--models", nargs="+", default=["lstm", "gru", "transformer", "cnn"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--no-cpu", action="store_true", help="Skip CPU baseline")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmark")
    args = parser.parse_args()

    print(f"Config: {args.n_samples} samples, seq_len={args.seq_len}, "
          f"features={args.n_features}, batch={args.batch_size}, "
          f"hidden={args.hidden_size}, layers={args.num_layers}, epochs={args.epochs}")

    X, y = generate_data(args.n_samples, args.seq_len, args.n_features)

    devices = []
    if not args.no_cpu:
        devices.append("cpu")
    if not args.no_gpu:
        try:
            import applegpu_runtime as gpu
            gpu.init_backend()
            gpu.enable_torch_backend()
            from applegpu_runtime.torch_backend import set_eager_mode
            set_eager_mode(True)
            print(f"applegpu: {gpu.device_name()}")
            devices.append("applegpu")
        except Exception as e:
            print(f"applegpu not available: {e}")

    results = []
    for model_type in args.models:
        for device in devices:
            print(f"Benchmarking {model_type.upper()} on {device}... ", end="", flush=True)
            try:
                r = bench_one(model_type, device, X, y, args.batch_size, args.epochs,
                              args.hidden_size, args.num_layers, args.seq_len)
                results.append(r)
                print(f"{r['total_time_s']:.2f}s ({r['samples_per_sec']:.0f} samples/s)")
            except Exception as e:
                print(f"FAILED: {e}")

    if results:
        print_results(results)


if __name__ == "__main__":
    main()
