"""Apple GPU Runtime - Unified API for GPU operations on Apple Silicon."""

from applegpu_runtime.applegpu_runtime import (
    GpuTensor,
    version,
    init_backend,
    device_name,
    dtype_size,
    from_numpy,
    from_numpy_shared,
    from_torch,
    from_torch_shared,
    from_bytes,
    aligned_numpy,
    tensor,
    eval,
    to_list,
    shape,
    destroy,
    set_limits,
    memory_usage,
    tensor_count,
    add,
    sub,
    mul,
    div,
    neg,
    relu,
    exp,
    log,
    sqrt,
    scalar_mul,
    reshape,
    softmax,
    log_softmax,
    transpose,
    transpose_dims,
    tanh,
    sin,
    cos,
    gelu,
    gelu_exact,
    sigmoid,
    var,
    amax,
    std_dev,
    layer_norm,
    embedding,
    gather,
    index_select,
    attention,
    matmul,
    slice,
    concat,
    add_bias,
    softmax_causal,
    argmax,
    cast,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    bitwise_not,
    shl,
    shr,
    mod_,
    elem_min,
    elem_max,
    logical_not,
    lt,
    gt,
    le,
    ge,
    eq_,
    ne_,
    sum,
    mean,
    attention_causal,
    abs,
    sign,
    pow,
    clamp,
    where_cond,
    masked_fill,
    triu,
    tril,
    register_container,
    deregister_container,
    pause_container,
    resume_container,
    submit_job,
    run_next,
    job_status,
    container_usage,
    global_usage,
    queue_depth,
    pool_stats,
    pool_drain,
    set_pool_watermark,
    conv1d,
    conv2d,
    batch_norm,
    max_pool2d,
    max_pool2d_with_indices,
    avg_pool2d,
    softmax_backward,
    layer_norm_backward,
    conv2d_backward_input,
    conv2d_backward_weight,
    conv1d_backward_input,
    embedding_backward,
    scatter_write,
    scatter_add,
    batch_norm_backward,
    threshold_backward,
    tanh_backward,
    sigmoid_backward,
    gelu_backward,
    gelu_tanh_backward,
    gelu_exact_backward,
    max_pool2d_backward,
    blit_copy,
)

__version__ = version()


def concat_all(tensors, dim=0):
    """Concatenate a list of tensors along a dimension.

    More efficient than repeated pairwise concat calls from Python,
    as it avoids N-1 Python-to-Rust round trips.

    Args:
        tensors: list of GpuTensor objects to concatenate
        dim: dimension along which to concatenate (default: 0)

    Returns:
        A new GpuTensor containing the concatenated result.
    """
    if len(tensors) == 0:
        raise ValueError("concat_all requires at least 1 tensor")
    if len(tensors) == 1:
        return tensors[0]
    result = tensors[0]
    for t in tensors[1:]:
        result = concat(result, t, dim=dim)
    return result


# High-level model API (lazy imports — transformers/torch only needed when called)
def load_model(model_name="gpt2"):
    """Load a pretrained model from HuggingFace."""
    from applegpu_runtime.models.gpt2 import load_gpt2_weights
    return load_gpt2_weights(model_name)

def run_model(model_name, prompt, max_tokens=50, temperature=1.0, top_k=50, top_p=0.9):
    """Run inference on a pretrained model.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2")
        prompt: input text string
        max_tokens: number of tokens to generate
        temperature: sampling temperature (0 = greedy, 1.0 = standard)
        top_k: only sample from top k tokens (default 50)
        top_p: nucleus sampling threshold (default 0.9)

    Returns:
        generated text string
    """
    from applegpu_runtime.models.gpt2 import load_gpt2_weights
    from applegpu_runtime.models.generate import tokenize, decode, generate

    model = load_gpt2_weights(model_name)
    input_ids = tokenize(model_name, prompt)
    output_ids = generate(model, input_ids, max_tokens=max_tokens,
                          temperature=temperature, top_k=top_k, top_p=top_p)
    return decode(model_name, output_ids)

def generate_text(model, prompt, max_tokens=50, temperature=1.0, top_k=50, top_p=0.9):
    """Generate text from a loaded model.

    Args:
        model: dict from load_model()
        prompt: input text or list of token IDs
        max_tokens: number of tokens to generate
        temperature: sampling temperature (0 = greedy, 1.0 = standard)
        top_k: only sample from top k tokens (default 50)
        top_p: nucleus sampling threshold (default 0.9)

    Returns:
        generated text string
    """
    from applegpu_runtime.models.generate import tokenize, decode, generate

    if isinstance(prompt, str):
        model_name = model.get("config", {}).get("_name_or_path", "gpt2")
        input_ids = tokenize("gpt2", prompt)
    else:
        input_ids = prompt

    output_ids = generate(model, input_ids, max_tokens=max_tokens,
                          temperature=temperature, top_k=top_k, top_p=top_p)
    return decode("gpt2", output_ids)


def enable_torch_backend():
    """Register applegpu as a PyTorch device backend. Requires torch >= 2.1."""
    from applegpu_runtime.torch_backend import enable
    enable()


def enable_training():
    """Enable training mode (eager evaluation + autograd support).

    Must be called before training loops. Disables lazy kernel fusion to ensure
    intermediate tensors survive for backward pass gradient computation.
    """
    from applegpu_runtime.torch_backend import set_eager_mode
    set_eager_mode(True)


def disable_training():
    """Disable training mode, re-enabling lazy evaluation with kernel fusion."""
    from applegpu_runtime.torch_backend import set_eager_mode
    set_eager_mode(False)


def to_applegpu(model_or_tensor):
    """Move a PyTorch model or tensor to applegpu Metal GPU.

    For nn.Module: converts all parameters and buffers in-place.
    For torch.Tensor: wraps as ApplegpuTensor.

    Args:
        model_or_tensor: an nn.Module or torch.Tensor

    Returns:
        The same object with data on applegpu.
    """
    from applegpu_runtime.torch_backend import to_applegpu
    return to_applegpu(model_or_tensor)


__all__ = [
    "GpuTensor",
    "version",
    "init_backend",
    "device_name",
    "dtype_size",
    "from_numpy",
    "from_numpy_shared",
    "from_torch",
    "from_torch_shared",
    "from_bytes",
    "aligned_numpy",
    "tensor",
    "eval",
    "to_list",
    "shape",
    "destroy",
    "set_limits",
    "memory_usage",
    "tensor_count",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "relu",
    "exp",
    "log",
    "sqrt",
    "scalar_mul",
    "reshape",
    "softmax",
    "transpose",
    "transpose_dims",
    "tanh",
    "gelu",
    "gelu_exact",
    "sigmoid",
    "var",
    "amax",
    "std_dev",
    "layer_norm",
    "embedding",
    "gather",
    "index_select",
    "attention",
    "matmul",
    "slice",
    "concat",
    "concat_all",
    "add_bias",
    "softmax_causal",
    "argmax",
    "sum",
    "mean",
    "attention_causal",
    "abs",
    "sign",
    "pow",
    "clamp",
    "where_cond",
    "masked_fill",
    "triu",
    "tril",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "shl",
    "shr",
    "mod_",
    "elem_min",
    "elem_max",
    "logical_not",
    "lt",
    "gt",
    "le",
    "ge",
    "eq_",
    "ne_",
    "register_container",
    "deregister_container",
    "pause_container",
    "resume_container",
    "submit_job",
    "run_next",
    "job_status",
    "container_usage",
    "global_usage",
    "queue_depth",
    "pool_stats",
    "pool_drain",
    "set_pool_watermark",
    "load_model",
    "run_model",
    "generate_text",
    "enable_torch_backend",
    "enable_training",
    "disable_training",
    "to_applegpu",
    "conv1d",
    "conv2d",
    "batch_norm",
    "max_pool2d",
    "max_pool2d_with_indices",
    "avg_pool2d",
    "softmax_backward",
    "layer_norm_backward",
    "conv2d_backward_input",
    "conv2d_backward_weight",
    "conv1d_backward_input",
    "embedding_backward",
    "scatter_write",
    "scatter_add",
    "batch_norm_backward",
    "threshold_backward",
    "tanh_backward",
    "sigmoid_backward",
    "gelu_backward",
    "gelu_tanh_backward",
    "gelu_exact_backward",
    "max_pool2d_backward",
    "blit_copy",
]
