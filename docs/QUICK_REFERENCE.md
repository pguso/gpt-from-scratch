# Quick Reference Guide

Quick access to common commands, hyperparameters, and code snippets for GPT From Scratch.

## Table of Contents

- [Common Commands](#common-commands)
- [Training Script Parameters](#training-script-parameters)
- [Hyperparameter Cheat Sheet](#hyperparameter-cheat-sheet)
- [Model Configuration Examples](#model-configuration-examples)
- [Code Snippets](#code-snippets)
- [Troubleshooting Quick Links](#troubleshooting-quick-links)

## Common Commands

### Installation

```bash
# Clone repository
git clone https://github.com/pguso/gpt-from-scratch.git
cd gpt-from-scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Quick Start (Untrained Model)

```bash
# Run untrained model (output will be gibberish - this is expected!)
python examples/quickstart.py
```

### Training

```bash
# Small model for quick experimentation
python examples/train_tiny_stories.py \
    --epochs 5 \
    --max-samples 5000 \
    --embedding-dimension 128 \
    --number-of-layers 2 \
    --number-of-heads 2

# Medium model for better results
python examples/train_tiny_stories.py \
    --epochs 20 \
    --max-samples 50000 \
    --embedding-dimension 512 \
    --number-of-layers 6 \
    --number-of-heads 8 \
    --batch-size 64

# Large model (requires more memory and time)
python examples/train_tiny_stories.py \
    --epochs 20 \
    --max-samples 100000 \
    --embedding-dimension 768 \
    --number-of-layers 12 \
    --number-of-heads 12 \
    --context-length 1024 \
    --batch-size 32
```

### Text Generation

```bash
# Generate with default settings
python examples/generate_text.py --checkpoint checkpoints/best_model.pt

# Custom prompt and parameters
python examples/generate_text.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Once upon a time" \
    --length 150 \
    --temperature 0.7 \
    --top-k 50

# More creative output (higher temperature)
python examples/generate_text.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "The little girl" \
    --temperature 1.2 \
    --top-k 100

# More deterministic output (lower temperature)
python examples/generate_text.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "The cat sat" \
    --temperature 0.3 \
    --top-k 10
```

### Attention Analysis (Optional)

```bash
# Analyze attention patterns (if analyze_attention.py exists)
python examples/analyze_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --text "The cat sat on the mat. It was fluffy."
```

## Training Script Parameters

Complete reference for `train_tiny_stories.py` command-line arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| **Data Arguments** |
| `--data-path` | str | None | Path to data file (None to download from Hugging Face) |
| `--max-samples` | int | 10000 | Maximum number of samples to use from dataset |
| **Training Arguments** |
| `--epochs` | int | 10 | Number of training epochs |
| `--batch-size` | int | 32 | Batch size (adjust for memory) |
| `--learning-rate` | float | 3e-4 | Learning rate for AdamW optimizer |
| **Model Arguments** |
| `--context-length` | int | 128 | Maximum sequence length |
| `--embedding-dimension` | int | 256 | Size of token embeddings |
| `--number-of-heads` | int | 4 | Number of attention heads (must divide embedding-dimension) |
| `--number-of-layers` | int | 4 | Number of transformer layers |
| **Other Arguments** |
| `--device` | str | "cuda" | Device to train on: "cuda", "cpu", or "mps" |
| `--save-dir` | str | "checkpoints" | Directory to save checkpoints |
| `--save-every` | int | 5 | Save checkpoint every N epochs |
| `--eval-every` | int | 1 | Run validation every N epochs |
| `--generate-every` | int | 2 | Generate sample text every N epochs |

**Example usage:**
```bash
python examples/train_tiny_stories.py \
    --epochs 20 \
    --max-samples 50000 \
    --embedding-dimension 512 \
    --number-of-layers 6 \
    --number-of-heads 8 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --context-length 256 \
    --device cuda \
    --save-dir checkpoints \
    --save-every 5 \
    --eval-every 1 \
    --generate-every 2
```

## Hyperparameter Cheat Sheet

### Model Architecture

| Parameter | Small | Medium | Large | GPT-2 Small | Description |
|-----------|-------|--------|-------|-------------|-------------|
| `embedding-dimension` | 128-256 | 512 | 768+ | 768 | Embedding dimension |
| `number-of-layers` | 2-4 | 6-8 | 12+ | 12 | Transformer blocks |
| `number-of-heads` | 2-4 | 8 | 12+ | 12 | Attention heads |
| `context-length` | 128-256 | 512 | 1024+ | 1024 | Max sequence length |
| `vocab-size` | 50257 | 50257 | 50257 | 50257 | Vocabulary size (GPT-2) |

**Note:** `number-of-heads` must evenly divide `embedding-dimension`.

### Training Hyperparameters

| Parameter | Typical Range | Default | Description |
|-----------|---------------|---------|-------------|
| `learning-rate` | 1e-5 to 1e-3 | 3e-4 | Learning rate for AdamW |
| `batch-size` | 16-128 | 32 | Batch size (adjust for memory) |
| `epochs` | 5-50 | 10 | Number of training epochs |
| `drop-rate` | 0.0-0.2 | 0.1 | Dropout rate |
| `weight-decay` | 0.01-0.1 | 0.1 | L2 regularization (AdamW) |

### Generation Parameters

| Parameter | Range | Typical | Description |
|-----------|-------|---------|-------------|
| `temperature` | 0.1-2.0 | 0.7-0.9 | Controls randomness (lower = more deterministic) |
| `top-k` | 1-100 | 50 | Sample from top k tokens |
| `top-p` | 0.1-1.0 | 0.9 | Nucleus sampling threshold |
| `max-length` | 50-500 | 150 | Maximum generation length |

### Device Selection

| Device | Command | When to Use |
|--------|---------|-------------|
| CUDA (NVIDIA GPU) | `--device cuda` | NVIDIA GPU available |
| MPS (Apple Silicon) | `--device mps` | Apple M1/M2/M3 Mac |
| CPU | `--device cpu` | No GPU or debugging |

## Model Configuration Examples

### Tiny Model (Fast, for Learning)

```python
from src.config import ModelConfig

config = ModelConfig(
    vocab_size=50257,
    context_length=128,
    embedding_dimension=128,
    number_of_heads=2,
    number_of_layers=2,
    dropout_rate=0.1,
    use_attention_bias=False
)
# ~1M parameters
```

### Small Model (Good Balance)

```python
config = ModelConfig(
    vocab_size=50257,
    context_length=256,
    embedding_dimension=256,
    number_of_heads=4,
    number_of_layers=4,
    dropout_rate=0.1,
    use_attention_bias=False
)
# ~4M parameters
```

### Medium Model (Better Quality)

```python
config = ModelConfig(
    vocab_size=50257,
    context_length=512,
    embedding_dimension=512,
    number_of_heads=8,
    number_of_layers=6,
    dropout_rate=0.1,
    use_attention_bias=False
)
# ~30M parameters
```

### GPT-2 Small Equivalent

```python
config = ModelConfig(
    vocab_size=50257,
    context_length=1024,
    embedding_dimension=768,
    number_of_heads=12,
    number_of_layers=12,
    dropout_rate=0.1,
    use_attention_bias=False
)
# ~117M parameters
```

## Code Snippets

### Create and Use Model

```python
from src.model.gpt import GPTModel
from src.config import ModelConfig
import torch

# Create config
config = ModelConfig(
    vocab_size=50257,
    context_length=128,
    embedding_dimension=256,
    number_of_heads=4,
    number_of_layers=4,
    dropout_rate=0.1
)

# Create model
model = GPTModel(config)

# Forward pass
input_ids = torch.randint(0, 50257, (1, 10))  # batch_size=1, seq_len=10
logits = model(input_ids)  # Shape: [1, 10, 50257]
```

### Load Trained Checkpoint

```python
import torch
from src.model.gpt import GPTModel
from src.config import ModelConfig

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')

# Recreate config
config = ModelConfig(**checkpoint['config'])

# Create and load model
model = GPTModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use model
with torch.no_grad():
    input_ids = torch.tensor([[15496, 995]])  # "Hello world"
    logits = model(input_ids)
```

### Generate Text Programmatically

```python
from src.generation.generate import generate_text
import tiktoken

# Load model (see above)
tokenizer = tiktoken.get_encoding("gpt2")

# Encode prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt)

# Generate
output_ids = generate_text(
    model,
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
)

# Decode
output_text = tokenizer.decode(output_ids)
print(output_text)
```

### Training Loop Snippet

```python
from src.training.trainer import GPTTrainer
import torch.optim as optim

# Setup
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95)
)

trainer = GPTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device
)

# Training
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch()
    val_loss = trainer.validate()
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
```

### Count Model Parameters

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_parameters(model)
print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Model size: {total * 4 / 1024 / 1024:.2f} MB (FP32)")
```

## Troubleshooting Quick Links

### Common Issues

**Q: Model outputs gibberish even after training**
- **Solution:** See [README Troubleshooting](../README.md#troubleshooting)
- Check validation loss - should decrease over time
- Try more epochs, larger model, or more data

**Q: Training is too slow**
- **Solution:** Use GPU (`--device cuda` or `--device mps`)
- Reduce model size or dataset size
- Reduce batch size if memory limited

**Q: Out of memory errors**
- **Solution:** Reduce `--batch-size`
- Reduce `--context-length`
- Reduce model size (`--embedding-dimension`, `--number-of-layers`)

**Q: Can't download TinyStories dataset**
- **Solution:** Install datasets: `pip install datasets`
- Or use sample data (script falls back automatically)
- Or provide your own data with `--data-path`

### Performance Tips

**For Faster Training:**
- Use GPU if available
- Reduce model size for experimentation
- Use smaller dataset (`--max-samples`)
- Reduce context length
- Use mixed precision (see [Advanced Topics](05-advanced-topics.md))

**For Better Quality:**
- Increase model size (embedding-dimension, layers)
- Use more training data
- Train for more epochs
- Use longer context length
- Tune hyperparameters

### Memory Management

| Issue | Solution |
|-------|----------|
| GPU OOM | Reduce batch-size, context-length, or model size |
| CPU OOM | Use smaller dataset or model |
| Slow loading | Use `num_workers=0` in DataLoader (already default) |

## Additional Resources

- **Full Documentation:** See [Documentation Index](README.md)
- **Model Implementation:** See [Model Implementation](01-model-usage-guide.md)
- **Training Details:** See [Training Implementation](02-training-implementation.md)
- **Using the Model:** See [Using the Model](03-using-the-model.md)
- **Pitfalls and Challenges:** See [Pitfalls and Challenges](04-pitfalls-and-challenges.md)
- **Main README:** See [README.md](../README.md) for installation and overview
