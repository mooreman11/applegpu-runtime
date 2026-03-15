"""BERT inference on Apple Silicon Metal GPU via PyTorch device backend.

Usage:
    python examples/bert_inference.py
    python examples/bert_inference.py --model bert-base-uncased
    python examples/bert_inference.py --text "The capital of France is [MASK]."
"""

import argparse
import time
import torch
import applegpu_runtime as gpu
from applegpu_runtime.torch_backend import ApplegpuTensor


def main():
    parser = argparse.ArgumentParser(description="BERT inference on Metal GPU")
    parser.add_argument("--model", default=None, help="HuggingFace model name (default: tiny random BERT)")
    parser.add_argument("--text", default="Hello world, this is a test.", help="Input text")
    args = parser.parse_args()

    gpu.init_backend()
    gpu.enable_torch_backend()
    gpu.set_limits(max_tensor_size_mb=0, max_memory_mb=0, max_tensors=0)

    from transformers import BertModel, BertConfig, AutoTokenizer

    if args.model:
        print(f"Loading {args.model} from HuggingFace...")
        model = BertModel.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        print("Loading tiny BERT (random weights, no download)...")
        config = BertConfig(
            hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
            intermediate_size=512, vocab_size=30522, max_position_embeddings=512
        )
        model = BertModel(config)
        tokenizer = None

    model.eval()

    # Move to Metal GPU
    print("Moving to applegpu...")
    model = gpu.to_applegpu(model)

    # Tokenize
    if tokenizer:
        inputs = tokenizer(args.text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        print(f"Input: \"{args.text}\"")
        print(f"Tokens: {input_ids.shape[1]}")
    else:
        seq_len = 16
        input_ids = torch.randint(0, 30522, (1, seq_len))
        attention_mask = torch.ones(1, seq_len)
        print(f"Random input: {seq_len} tokens")

    # Move inputs to GPU
    input_ids_gpu = ApplegpuTensor.from_torch(input_ids)
    attention_mask_gpu = ApplegpuTensor.from_torch(attention_mask)

    # Run inference
    print("Running inference...")
    start = time.time()
    with torch.no_grad():
        output = model(input_ids_gpu, attention_mask=attention_mask_gpu)
    elapsed = time.time() - start

    hidden = output.last_hidden_state
    if isinstance(hidden, ApplegpuTensor):
        hidden = hidden.to_torch_cpu()

    print(f"\nResults:")
    print(f"  Hidden state shape: {hidden.shape}")
    print(f"  All finite: {torch.all(torch.isfinite(hidden)).item()}")
    print(f"  Mean activation: {hidden.mean().item():.4f}")
    print(f"  Std activation: {hidden.std().item():.4f}")
    print(f"  Inference time: {elapsed*1000:.1f}ms")


if __name__ == "__main__":
    main()
