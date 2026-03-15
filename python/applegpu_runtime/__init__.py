"""Apple GPU Runtime - Unified API for GPU operations on Apple Silicon."""

from applegpu_runtime.applegpu_runtime import (
    GpuTensor,
    version,
    init_backend,
    device_name,
    dtype_size,
    from_numpy,
    from_torch,
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
    transpose,
    gelu,
    layer_norm,
    embedding,
    attention,
    matmul,
    slice,
    concat,
    add_bias,
    softmax_causal,
    argmax,
    attention_causal,
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
)

__version__ = version()


# High-level model API (lazy imports — transformers/torch only needed when called)
def load_model(model_name="gpt2"):
    """Load a pretrained model from HuggingFace."""
    from applegpu_runtime.models.gpt2 import load_gpt2_weights
    return load_gpt2_weights(model_name)

def run_model(model_name, prompt, max_tokens=50):
    """Run inference on a pretrained model.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2")
        prompt: input text string
        max_tokens: number of tokens to generate

    Returns:
        generated text string
    """
    from applegpu_runtime.models.gpt2 import load_gpt2_weights
    from applegpu_runtime.models.generate import tokenize, decode, generate

    model = load_gpt2_weights(model_name)
    input_ids = tokenize(model_name, prompt)
    output_ids = generate(model, input_ids, max_tokens=max_tokens)
    return decode(model_name, output_ids)

def generate_text(model, prompt, max_tokens=50):
    """Generate text from a loaded model.

    Args:
        model: dict from load_model()
        prompt: input text or list of token IDs
        max_tokens: number of tokens to generate

    Returns:
        generated text string
    """
    from applegpu_runtime.models.generate import tokenize, decode, generate

    if isinstance(prompt, str):
        model_name = model.get("config", {}).get("_name_or_path", "gpt2")
        input_ids = tokenize("gpt2", prompt)
    else:
        input_ids = prompt

    output_ids = generate(model, input_ids, max_tokens=max_tokens)
    return decode("gpt2", output_ids)


__all__ = [
    "GpuTensor",
    "version",
    "init_backend",
    "device_name",
    "dtype_size",
    "from_numpy",
    "from_torch",
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
    "gelu",
    "layer_norm",
    "embedding",
    "attention",
    "matmul",
    "slice",
    "concat",
    "add_bias",
    "softmax_causal",
    "argmax",
    "attention_causal",
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
]
