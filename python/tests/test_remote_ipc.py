"""Integration tests for the container IPC path.

These tests start gpu-service on a Unix socket and run the C++ PrivateUse1
backend in remote mode (APPLEGPU_SOCKET). Verifies that ops produce correct
results when dispatched over the wire protocol to gpu-service.
"""
import os
import signal
import subprocess
import sys
import time

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GPU_SERVICE = os.path.join(REPO_ROOT, "target", "debug", "gpu-service")
DYLIB_DIR = os.path.join(
    REPO_ROOT, "swift", ".build", "arm64-apple-macosx", "release"
)
SOCK_PATH = "/tmp/applegpu_test_remote.sock"
PID_PATH = "/tmp/applegpu_test_remote.pid"


def _run_remote_test(test_code: str, timeout: int = 30) -> str:
    """Start gpu-service and run test_code in a subprocess with APPLEGPU_SOCKET."""
    # Clean stale files
    for f in [SOCK_PATH, PID_PATH]:
        if os.path.exists(f):
            os.unlink(f)

    if not os.path.exists(GPU_SERVICE):
        pytest.skip("gpu-service not built (run: cargo build -p applegpu-service)")

    env = os.environ.copy()
    env["APPLEGPU_SOCKET"] = SOCK_PATH
    env["APPLEGPU_PID_FILE"] = PID_PATH
    env["DYLD_LIBRARY_PATH"] = DYLIB_DIR

    service = subprocess.Popen(
        [GPU_SERVICE], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    try:
        # Wait for socket
        for _ in range(50):
            if os.path.exists(SOCK_PATH):
                break
            time.sleep(0.1)
        else:
            pytest.fail("gpu-service failed to start")

        # Run test
        test_env = os.environ.copy()
        test_env["APPLEGPU_SOCKET"] = SOCK_PATH
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            env=test_env, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            pytest.fail(
                f"Remote test failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr[:1000]}"
            )
        return result.stdout
    finally:
        service.send_signal(signal.SIGTERM)
        try:
            service.wait(timeout=5)
        except Exception:
            service.kill()
        for f in [SOCK_PATH, PID_PATH]:
            if os.path.exists(f):
                os.unlink(f)


def test_remote_add():
    out = _run_remote_test("""
import torch, sys
sys.path.insert(0, 'python')
from applegpu_runtime.cpp_backend import load_cpp_backend
load_cpp_backend()
a = torch.tensor([1.0, 2.0, 3.0], device='applegpu')
b = torch.tensor([10.0, 20.0, 30.0], device='applegpu')
c = a + b
torch.applegpu.synchronize()
result = c.cpu().tolist()
assert result == [11.0, 22.0, 33.0], f"add failed: {result}"
print("PASS")
""")
    assert "PASS" in out


def test_remote_matmul():
    out = _run_remote_test("""
import torch, sys
sys.path.insert(0, 'python')
from applegpu_runtime.cpp_backend import load_cpp_backend
load_cpp_backend()
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='applegpu')
w = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='applegpu')
y = x @ w
torch.applegpu.synchronize()
ref = x.cpu() @ w.cpu()
assert torch.allclose(y.cpu(), ref, atol=1e-4), f"matmul diff: {(y.cpu()-ref).abs().max()}"
print("PASS")
""")
    assert "PASS" in out


def test_remote_linear():
    out = _run_remote_test("""
import torch, sys
sys.path.insert(0, 'python')
from applegpu_runtime.cpp_backend import load_cpp_backend
load_cpp_backend()
linear = torch.nn.Linear(4, 2).to('applegpu')
inp = torch.randn(1, 4).to('applegpu')
out = linear(inp)
torch.applegpu.synchronize()
# CPU reference
linear_cpu = torch.nn.Linear(4, 2)
linear_cpu.weight.data = linear.weight.data.cpu()
linear_cpu.bias.data = linear.bias.data.cpu()
ref = linear_cpu(inp.cpu())
diff = (out.cpu() - ref).abs().max().item()
assert diff < 1e-3, f"linear diff: {diff}"
print("PASS")
""")
    assert "PASS" in out


def test_remote_mlp():
    out = _run_remote_test("""
import torch, sys
sys.path.insert(0, 'python')
from applegpu_runtime.cpp_backend import load_cpp_backend
load_cpp_backend()
model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 2),
).to('applegpu')
x = torch.randn(1, 4).to('applegpu')
out = model(x)
torch.applegpu.synchronize()
# CPU ref
model_cpu = torch.nn.Sequential(
    torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
model_cpu[0].weight.data = model[0].weight.data.cpu()
model_cpu[0].bias.data = model[0].bias.data.cpu()
model_cpu[2].weight.data = model[2].weight.data.cpu()
model_cpu[2].bias.data = model[2].bias.data.cpu()
ref = model_cpu(x.cpu())
diff = (out.cpu() - ref).abs().max().item()
assert diff < 1e-3, f"mlp diff: {diff}"
print("PASS")
""")
    assert "PASS" in out
