"""PyTorch autograd.Function wrappers for applegpu_runtime ops.

These provide explicit backward implementations using our Metal GPU kernels.
For basic training, PyTorch's built-in autograd works with our __torch_dispatch__
since backward ops decompose into primitive aten ops we already handle.

These Function classes are available for advanced use cases where explicit
control over the backward pass is needed.
"""

import torch
import applegpu_runtime as gpu
from applegpu_runtime.torch_backend import ApplegpuTensor, _unwrap, _wrap


class ApplegpuMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        result_gpu = gpu.matmul(_unwrap(a), _unwrap(b))
        return _wrap(result_gpu)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # dA = grad_output @ B^T
        grad_a = _wrap(gpu.matmul(_unwrap(grad_output), gpu.transpose(_unwrap(b))))
        # dB = A^T @ grad_output
        grad_b = _wrap(gpu.matmul(gpu.transpose(_unwrap(a)), _unwrap(grad_output)))
        return grad_a, grad_b


class ApplegpuAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        result_gpu = gpu.add(_unwrap(a), _unwrap(b))
        return _wrap(result_gpu)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class ApplegpuMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        result_gpu = gpu.mul(_unwrap(a), _unwrap(b))
        return _wrap(result_gpu)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = _wrap(gpu.mul(_unwrap(grad_output), _unwrap(b)))
        grad_b = _wrap(gpu.mul(_unwrap(grad_output), _unwrap(a)))
        return grad_a, grad_b


class ApplegpuSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        result_gpu = gpu.sub(_unwrap(a), _unwrap(b))
        return _wrap(result_gpu)

    @staticmethod
    def backward(ctx, grad_output):
        grad_b = _wrap(gpu.neg(_unwrap(grad_output)))
        return grad_output, grad_b


class ApplegpuRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        result_gpu = gpu.relu(_unwrap(input))
        return _wrap(result_gpu)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        input_cpu = input.to_torch_cpu() if isinstance(input, ApplegpuTensor) else input
        grad_cpu = grad_output.to_torch_cpu() if isinstance(grad_output, ApplegpuTensor) else grad_output
        mask = (input_cpu > 0).float()
        grad_input = grad_cpu * mask
        return ApplegpuTensor.from_torch(grad_input)


class ApplegpuGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        result_gpu = gpu.gelu(_unwrap(input))
        return _wrap(result_gpu)

    @staticmethod
    def backward(ctx, grad_output):
        import math
        input, = ctx.saved_tensors
        x = input.to_torch_cpu() if isinstance(input, ApplegpuTensor) else input
        sqrt_2_pi = math.sqrt(2.0 / math.pi)
        a = sqrt_2_pi * (x + 0.044715 * x ** 3)
        a = a.clamp(-10, 10)
        tanh_a = torch.tanh(a)
        da = sqrt_2_pi * (1 + 3 * 0.044715 * x ** 2)
        gelu_grad = 0.5 * (1 + tanh_a) + 0.5 * x * (1 - tanh_a ** 2) * da

        grad_cpu = grad_output.to_torch_cpu() if isinstance(grad_output, ApplegpuTensor) else grad_output
        grad_input = grad_cpu * gelu_grad
        return ApplegpuTensor.from_torch(grad_input)


class ApplegpuTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        result_gpu = gpu.tanh(_unwrap(input))
        ctx.save_for_backward(_wrap(result_gpu))
        return _wrap(result_gpu)

    @staticmethod
    def backward(ctx, grad_output):
        tanh_output, = ctx.saved_tensors
        tanh_cpu = tanh_output.to_torch_cpu() if isinstance(tanh_output, ApplegpuTensor) else tanh_output
        grad_cpu = grad_output.to_torch_cpu() if isinstance(grad_output, ApplegpuTensor) else grad_output
        grad_input = grad_cpu * (1 - tanh_cpu ** 2)
        return ApplegpuTensor.from_torch(grad_input)
