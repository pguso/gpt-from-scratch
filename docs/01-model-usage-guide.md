# Model Usage Guide

How to use the GPT model - inputs, outputs, and basic usage.

## Overview

This document explains how to use the GPT model: what goes in, what comes out, and how to use it. For detailed explanations of how the model works internally, see the links at the end.

**Main file**: `src/model/gpt.py`

## Model Creation

### Basic Usage

```python
from src.model.gpt import GPTModel
from src.config import ModelConfig

# Create configuration
config = ModelConfig(
    vocab_size=50257,           # Vocabulary size (GPT-2 tokenizer)
    context_length=128,         # Maximum sequence length
    embedding_dimension=256, # Embedding dimension
    number_of_heads=4,          # Attention heads
    number_of_layers=4,         # Transformer layers
    dropout_rate=0.1,           # Dropout probability
    use_attention_bias=False  # No bias in attention projections (GPT-2 style)
)

# Create model
model = GPTModel(config)
```

**Location**: `src/config.py` (ModelConfig class), `src/model/gpt.py` (GPTModel class)

### Configuration Parameters

**File**: `src/config.py`, lines 9-36

| Parameter | Type | Description | Typical Values |
|-----------|------|-------------|----------------|
| `vocab_size` | int | Number of tokens in vocabulary | 50257 (GPT-2) |
| `context_length` | int | Maximum sequence length | 128, 256, 512, 1024 |
| `embedding_dimension` | int | Size of token embeddings | 128, 256, 512, 768 |
| `number_of_heads` | int | Attention heads (must divide embedding_dimension) | 2, 4, 8, 12 |
| `number_of_layers` | int | Number of transformer blocks | 2, 4, 6, 12 |
| `dropout_rate` | float | Dropout probability | 0.0-0.2 |
| `use_attention_bias` | bool | Use bias in attention layer projections | False (GPT-2 style) |

**Constraint**: `embedding_dimension` must be divisible by `number_of_heads`.

## Input and Output

### Forward Pass

**Location**: `src/model/gpt.py`, `GPTModel.forward()` method, lines 73-91

```python
# Input
input_ids = torch.tensor([[464, 2361, 373]])  # Shape: [batch_size, sequence_length]
# Example: [1, 3] = batch of 1, sequence of 3 tokens

# Forward pass
logits = model(input_ids)

# Output
# Shape: [batch_size, sequence_length, vocab_size]
# Example: [1, 3, 50257] = batch of 1, 3 positions, 50257 vocabulary scores
```

**What goes in:**
- `input_ids`: Tensor of token IDs
  - Shape: `[batch_size, sequence_length]`
  - Values: Integers from 0 to `vocab_size - 1`
  - Example: `[[15496, 995]]` might represent "Hello world"

**What comes out:**
- `logits`: Raw scores for next token prediction
  - Shape: `[batch_size, sequence_length, vocab_size]`
  - Values: Float scores (not probabilities)
  - Each position has scores for all vocabulary tokens

> Think of logits as the model saying: “Here is how much I like every possible next token.”

**Example:**
```python
input_ids = torch.tensor([[464, 2361, 373]])  # "The cat sat"
logits = model(input_ids)  # Shape: [1, 3, 50257]

# logits[0, 0, :] = scores for next token after position 0 ("The")
# logits[0, 1, :] = scores for next token after position 1 ("The cat")
# logits[0, 2, :] = scores for next token after position 2 ("The cat sat")
```

### Converting Logits to Probabilities

The model outputs logits (raw scores), not probabilities. To get probabilities:

```python
import torch.nn.functional as F

logits = model(input_ids)  # [batch, seq_len, vocab_size]

# Get probabilities for last position
last_logits = logits[:, -1, :]  # [batch, vocab_size]
probs = F.softmax(last_logits, dim=-1)  # [batch, vocab_size]

# Now probs[0, 1234] = probability of token 1234 being next
```

## Model States

### Training Mode

```python
model.train()  # Enable dropout, batch norm updates
```

**When to use:**
- During training
- When you want dropout active

### Evaluation Mode

```python
model.eval()  # Disable dropout, freeze batch norm
```

**When to use:**
- During inference/generation
- When evaluating on validation set
- When you want deterministic outputs

## Device Placement

The model and input must be on the same device:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model = model.to(device)

# Input must be on same device
input_ids = input_ids.to(device)
logits = model(input_ids)
```

## Common Usage Patterns

### 1. Create and Use Model

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

# Use model
input_ids = torch.randint(0, 50257, (1, 10))  # Random tokens for testing
model.eval()
with torch.no_grad():
    logits = model(input_ids)  # [1, 10, 50257]
```

### 2. Load from Checkpoint

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
```

### 3. Count Parameters

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
```

### 4. Get Model Configuration

```python
# Access config
config = model.config

# Convert to dict
config_dict = config.to_dict()

# Access individual values
embedding_dim = config.embedding_dimension
context_len = config.context_length
```

## Model Architecture Overview

The model consists of:

1. **Embeddings** (`src/model/gpt.py`, lines 59-62)
   - Token embeddings: Convert token IDs to vectors
   - Position embeddings: Encode token positions

2. **Transformer Blocks** (`src/model/gpt.py`, lines 65-67)
   - Multiple layers that process the sequence
   - Each block contains attention and feed-forward networks

3. **Output Layer** (`src/model/gpt.py`, lines 70-71)
   - Final normalization
   - Linear projection to vocabulary size

**File structure:**
- `src/model/gpt.py` - Main model class
- `src/model/attention.py` - Attention mechanism
- `src/model/blocks.py` - LayerNorm, FeedForward, GELU
- `src/config.py` - Configuration class

## Input Constraints

### Sequence Length

The input sequence length cannot exceed `context_length`:

```python
config = ModelConfig(context_length=128)
model = GPTModel(config)

# This works
input_ids = torch.randint(0, 50257, (1, 128))  # Exactly context_length
logits = model(input_ids)

# This will work but only first 128 tokens are processed
input_ids = torch.randint(0, 50257, (1, 200))  # Longer than context_length
logits = model(input_ids)  # Only processes first 128 tokens
```

### Token ID Range

Token IDs must be in valid range:

```python
# Valid: 0 to vocab_size - 1
input_ids = torch.tensor([[0, 100, 50256]])  # OK for vocab_size=50257

# Invalid: out of range
input_ids = torch.tensor([[50257]])  # Error! Must be < vocab_size
```

### Batch Processing

The model processes batches:

```python
# Single sequence
input_ids = torch.tensor([[464, 2361, 373]])  # [1, 3]

# Batch of sequences
input_ids = torch.tensor([
    [464, 2361, 373],      # Sequence 1
    [15496, 995, 0],       # Sequence 2
    [1234, 5678, 9012]     # Sequence 3
])  # [3, 3] = batch of 3, each length 3

logits = model(input_ids)  # [3, 3, 50257]
```

**Note**: Sequences in a batch can have different lengths (use padding), but the model processes them as-is.

## Output Interpretation

### Understanding Logits

Logits are raw scores before softmax:

```python
logits = model(input_ids)  # [1, 5, 50257]

# For position 2 (after 2 tokens):
position_2_logits = logits[0, 2, :]  # [50257] scores

# Higher score = more likely (but not a probability yet)
top_token = torch.argmax(position_2_logits)  # Most likely token ID
top_score = position_2_logits[top_token]  # Its score
```

### Converting to Next Token Prediction

For next token prediction, use the last position:

```python
logits = model(input_ids)  # [1, seq_len, vocab_size]

# Get logits for next token (after entire sequence)
next_token_logits = logits[:, -1, :]  # [1, vocab_size]

# Convert to probabilities
probs = F.softmax(next_token_logits, dim=-1)

# Sample or get most likely
most_likely = torch.argmax(probs, dim=-1)  # Greedy
sampled = torch.multinomial(probs, num_samples=1)  # Random sample
```

## Common Issues

### Shape Mismatch

**Error**: `RuntimeError: shape mismatch`

**Cause**: Input shape is wrong

**Solution**:
```python
# Wrong: 1D tensor
input_ids = torch.tensor([464, 2361, 373])  # Shape: [3]

# Correct: 2D tensor with batch dimension
input_ids = torch.tensor([[464, 2361, 373]])  # Shape: [1, 3]
```

### Device Mismatch

**Error**: `RuntimeError: Expected all tensors to be on the same device`

**Solution**:
```python
device = torch.device('cuda')
model = model.to(device)
input_ids = input_ids.to(device)
logits = model(input_ids)
```

### Out of Memory

**Cause**: Sequence too long or batch too large

**Solution**:
- Reduce `context_length` in config
- Use shorter sequences
- Reduce batch size
- Use smaller model (fewer layers, smaller embedding_dim)

## Quick Reference

### Model Creation
```python
config = ModelConfig(vocab_size=50257, context_length=128, ...)
model = GPTModel(config)
```

### Forward Pass
```python
logits = model(input_ids)  # [batch, seq_len, vocab_size]
```

### Evaluation Mode
```python
model.eval()
with torch.no_grad():
    logits = model(input_ids)
```

### Device Placement
```python
model = model.to(device)
input_ids = input_ids.to(device)
```

## Deep Dive: Understanding Each Component

If you want to understand how the model works internally, here are detailed explanations:

### 1. Model Architecture and Forward Pass
- **File**: `src/model/gpt.py`
- **Key classes**: `GPTModel`, `TransformerBlock`
- **What to learn**: How embeddings, transformer blocks, and output layer work together
- **See**: The model implementation code and comments

### 2. Attention Mechanism
- **File**: `src/model/attention.py`
- **Key class**: `MultiHeadAttention`
- **What to learn**: How queries, keys, values work, causal masking, multi-head attention
- **See**: Attention implementation with Q/K/V projections and attention computation

### 3. Transformer Block Components
- **File**: `src/model/blocks.py`
- **Key classes**: `LayerNorm`, `FeedForward`, `GELU`
- **What to learn**: Layer normalization, feed-forward networks, activation functions
- **See**: Individual component implementations

### 4. Configuration System
- **File**: `src/config.py`
- **Key class**: `ModelConfig`
- **What to learn**: How configuration is structured and validated
- **See**: Configuration dataclass and validation logic

### 5. Training Integration
- **File**: `src/training/trainer.py`
- **What to learn**: How the model is used during training
- **See**: [Training Implementation](02-training-implementation.md)

### 6. Text Generation
- **File**: `src/generation/generate.py`
- **What to learn**: How logits are converted to generated text
- **See**: [Using the Model](03-using-the-model.md)

### 7. Attention Visualization
- **File**: `examples/analyze_attention.py`
- **What to learn**: How to visualize what the model is "looking at"
- **See**: [Understanding Attention Analysis](05-understanding-attention-analysis.md)

## Next Steps

- **Training**: See [Training Implementation](02-training-implementation.md) to learn how to train the model
- **Usage**: See [Using the Model](03-using-the-model.md) to learn how to generate text
- **Challenges**: See [Pitfalls and Challenges](04-pitfalls-and-challenges.md) for common issues
- **Quick Reference**: See [Quick Reference](QUICK_REFERENCE.md) for code snippets
