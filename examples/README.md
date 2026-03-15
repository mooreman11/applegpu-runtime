# Examples

Standalone scripts demonstrating applegpu_runtime capabilities.

## Prerequisites

```bash
make setup
uv run maturin develop
```

## Scripts

### GPT-2 Text Generation
```bash
python examples/gpt2_generate.py
python examples/gpt2_generate.py --prompt "Once upon a time" --max-tokens 100 --temperature 0.8
```
Requires: `pip install transformers torch`

### ResNet-18 Image Classification
```bash
python examples/resnet_inference.py
python examples/resnet_inference.py --model resnet50 --batch-size 4
```
Requires: `pip install torch torchvision`

### BERT Encoder
```bash
python examples/bert_inference.py
python examples/bert_inference.py --model bert-base-uncased --text "Hello world"
```
Requires: `pip install transformers torch`

### PyTorch Device Backend
```bash
python examples/pytorch_device_backend.py
```
Demonstrates: MLP inference, broadcasting, matmul chains, multi-dtype support.
Requires: `pip install torch`
