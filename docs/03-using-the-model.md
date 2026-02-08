# Using the Model

How to load a trained model and generate text.

## Overview

This document covers practical usage of trained models: loading checkpoints, generating text, and understanding generation parameters.

## Loading a Checkpoint

### Basic Loading

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
model.eval()  # Set to evaluation mode
```

**Key points:**
- `map_location='cpu'` loads to CPU (use `'cuda'` for GPU)
- `model.eval()` disables dropout and batch norm updates
- Config must match the saved model architecture

### Device Placement

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

**Why move to device?**
- GPU is much faster for inference
- All tensors must be on same device
- Model and input must match

### Checkpoint Information

```python
print(f"Epoch: {checkpoint['epoch']}")
print(f"Train loss: {checkpoint['train_loss']:.4f}")
print(f"Val loss: {checkpoint['val_loss']:.4f}")
```

**What's in a checkpoint:**
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state (for resuming training)
- `config`: Model configuration
- `epoch`: Training epoch
- `train_loss`, `val_loss`: Training metrics

## Text Generation

### Basic Generation

The `generate_text` function in `src/generation/generate.py`:

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
    maximum_new_tokens=100,
    temperature=0.8,
    top_k_tokens=50
)

# Decode
output_text = tokenizer.decode(output_ids)
print(output_text)
```

### Generation Process

The generation function works autoregressively:

```python
for _ in range(maximum_new_tokens):
    # 1. Get logits for next token
    logits = model(input_ids_conditioned)
    logits = logits[:, -1, :]  # Last position only
    
    # 2. Apply temperature
    logits = logits / temperature
    
    # 3. Apply top-k filtering (if specified)
    if top_k_tokens is not None:
        # Keep only top-k logits
        ...
    
    # 4. Convert to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # 5. Sample next token
    next_token = torch.multinomial(probs, num_samples=1)
    
    # 6. Append to sequence
    input_ids = torch.cat([input_ids, next_token], dim=1)
```

**Autoregressive generation:**
- Generate one token at a time
- Each new token depends on all previous tokens
- Context window limits how far back model can see

### Context Window Handling

```python
input_ids_conditioned = input_ids[:, -context_size:]
```

**Why truncate?**
- Model has fixed `context_length`
- Can't process sequences longer than this
- Keep only the most recent tokens

**Implications:**
- Long prompts get truncated
- Generated text can exceed context if you keep appending
- Need to manage context window manually

## Generation Parameters

### Temperature

Controls randomness in sampling:

```python
logits = logits / temperature
probs = torch.softmax(logits, dim=-1)
```

**Effect:**
- **Temperature < 1.0**: Sharper distribution (more deterministic)
- **Temperature = 1.0**: Standard sampling
- **Temperature > 1.0**: Flatter distribution (more random)

**Typical values:**
- **0.3-0.5**: Very deterministic, repetitive
- **0.7-0.9**: Balanced (default)
- **1.0-1.5**: More creative, less coherent
- **> 2.0**: Very random, often nonsensical

**Example:**
```python
# More deterministic
generate_text(model, input_ids, temperature=0.5, ...)

# More creative
generate_text(model, input_ids, temperature=1.2, ...)
```

### Top-k Sampling

Limits sampling to top k most likely tokens:

```python
if top_k_tokens is not None:
    top_k_values, _ = torch.topk(logits, min(top_k_tokens, logits.size(-1)), dim=-1)
    threshold = top_k_values[:, -1].unsqueeze(-1)
    logits = logits.masked_fill(logits < threshold, float('-inf'))
```

**Effect:**
- Removes low-probability tokens from consideration
- Prevents sampling very unlikely tokens
- Works well with temperature

**Typical values:**
- **10-20**: Very focused, may be repetitive
- **40-50**: Balanced (default)
- **100+**: More diverse, less focused

**Why use it?**
- Prevents model from sampling nonsensical tokens
- Improves coherence
- Common in modern language models

### Greedy Sampling

When temperature is 0:

```python
if temperature > 0:
    next_token = torch.multinomial(probs, num_samples=1)
else:
    next_token = torch.argmax(probs, dim=-1, keepdim=True)
```

**Effect:**
- Always picks most likely token
- Deterministic (same prompt → same output)
- Often repetitive

**When to use:**
- Need deterministic output
- Don't want randomness
- Usually not recommended (too repetitive)

## Using the Generation Script

The `examples/generate_text.py` script provides a command-line interface:

```bash
python examples/generate_text.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Once upon a time" \
    --length 150 \
    --temperature 0.7 \
    --top-k-tokens 50
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint
- `--prompt`: Starting text
- `--length`: Number of tokens to generate
- `--temperature`: Sampling temperature
- `--top-k-tokens`: Top-k sampling parameter
- `--device`: Device to use (cuda/cpu/mps)

## Practical Examples

### Basic Text Generation

```python
# Simple generation
prompt = "The cat sat on the"
input_ids = tokenizer.encode(prompt)
output_ids = generate_text(
    model, input_ids,
    maximum_new_tokens=50,
    temperature=0.8,
    top_k_tokens=50
)
print(tokenizer.decode(output_ids))
```

### More Creative Output

```python
# Higher temperature for more creativity
output_ids = generate_text(
    model, input_ids,
    maximum_new_tokens=100,
    temperature=1.2,      # More random
    top_k_tokens=100       # Wider selection
)
```

### More Deterministic Output

```python
# Lower temperature for more focused output
output_ids = generate_text(
    model, input_ids,
    maximum_new_tokens=50,
    temperature=0.5,      # Less random
    top_k_tokens=20       # Narrower selection
)
```

### Batch Generation

```python
# Generate from multiple prompts
prompts = ["Once upon a time", "The little girl", "In a far away land"]
for prompt in prompts:
    input_ids = tokenizer.encode(prompt)
    output_ids = generate_text(model, input_ids, ...)
    print(f"{prompt} → {tokenizer.decode(output_ids)}")
```

## Common Issues

### Repetitive Output

**Symptoms:**
- Model repeats same phrases
- Gets stuck in loops

**Solutions:**
- Increase temperature (more randomness)
- Increase top-k (wider selection)
- Check if model is undertrained

### Nonsensical Output

**Symptoms:**
- Random words
- No coherence

**Solutions:**
- Decrease temperature (less randomness)
- Decrease top-k (more focused)
- Model may be undertrained

### Context Window Exceeded

**Symptoms:**
- Error about sequence length
- Truncated output

**Solutions:**
- Shorten prompt
- Reduce `maximum_new_tokens`
- Model has fixed `context_length` limit

### Slow Generation

**Solutions:**
- Use GPU (`device='cuda'`)
- Reduce `maximum_new_tokens`
- Batch multiple generations together

## Generation Best Practices

### Start with Defaults

```python
# Good starting point
temperature=0.8
top_k_tokens=50
```

### Adjust Based on Use Case

- **Creative writing**: Higher temperature (1.0-1.2), higher top-k (100+)
- **Code generation**: Lower temperature (0.5-0.7), moderate top-k (40-50)
- **Factual text**: Lower temperature (0.5-0.7), lower top-k (20-30)

### Monitor Output Quality

- Check for repetition
- Verify coherence
- Ensure appropriate length
- Adjust parameters as needed

### Handle Context Limits

- Keep prompts short if generating long text
- Monitor total sequence length
- Truncate if needed

## Non-Determinism

**Important:** Generation is non-deterministic by default:

```python
# Same prompt, different outputs
output1 = generate_text(model, input_ids, ...)
output2 = generate_text(model, input_ids, ...)
# output1 != output2 (usually)
```

**Why?**
- Random sampling from probability distribution
- Temperature and top-k add randomness
- This is expected behavior

**For reproducibility:**
```python
torch.manual_seed(42)
output = generate_text(model, input_ids, ...)
```

**Note:** Even with fixed seed, results may vary between PyTorch versions or devices.

## Next Steps

- **Common pitfalls**: See [Pitfalls and Challenges](04-pitfalls-and-challenges.md)
- **Quick reference**: See [Quick Reference](QUICK_REFERENCE.md)
