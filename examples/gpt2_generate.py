"""GPT-2 text generation on Apple Silicon Metal GPU.

Usage:
    python examples/gpt2_generate.py
    python examples/gpt2_generate.py --prompt "Once upon a time" --max-tokens 100
    python examples/gpt2_generate.py --temperature 1.2 --top-k 100 --top-p 0.95
"""

import argparse
import time
import applegpu_runtime as gpu


def main():
    parser = argparse.ArgumentParser(description="GPT-2 text generation on Metal GPU")
    parser.add_argument("--prompt", default="The meaning of life is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    args = parser.parse_args()

    print(f"Loading GPT-2...")
    start = time.time()
    model = gpu.load_model("gpt2")
    print(f"Loaded in {time.time() - start:.1f}s")

    print(f"\nPrompt: {args.prompt}")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print(f"Generating {args.max_tokens} tokens...\n")

    start = time.time()
    output = gpu.generate_text(
        model, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    elapsed = time.time() - start

    print(output)
    print(f"\n--- {args.max_tokens} tokens in {elapsed:.1f}s ({elapsed/args.max_tokens:.2f}s/token) ---")


if __name__ == "__main__":
    main()
