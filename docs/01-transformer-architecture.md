# Transformer Architecture

Detailed explanation of the transformer architecture used in GPT.

## Introduction

The transformer architecture, introduced in the 2017 paper "Attention Is All You Need," revolutionized natural language processing. GPT (Generative Pre-trained Transformer) is built on this foundation, adapted for autoregressive language modeling.

## High-Level Structure

A GPT model follows this structure:

```
Input Tokens
    ↓
Token Embeddings + Position Embeddings
    ↓
Transformer Block 1
    ↓
Transformer Block 2
    ↓
...
    ↓
Transformer Block N
    ↓
Layer Normalization
    ↓
Language Model Head (Linear Layer)
    ↓
Output Logits (Vocabulary Scores)
```

## Core Components

### 1. Embedding Layer

The embedding layer converts discrete tokens into continuous vector representations.

#### Token Embeddings

Each token in the vocabulary is mapped to a dense vector of dimension `embedding_dimension` (typically 768 for base models, 256-512 for smaller models).

```python
token_embedding = nn.Embedding(vocab_size, embedding_dimension)
```

- **Vocabulary Size**: Usually 50,257 tokens (GPT-2) or 50,304 (GPT-3)
- **Embedding Dimension**: Determines the model's capacity (typically 256-4096)

#### Position Embeddings

Since transformers process all tokens in parallel (unlike RNNs), they need explicit position information.

```python
position_embedding = nn.Embedding(context_length, embedding_dimension)
```

- **Context Length**: Maximum sequence length (128-4096 tokens)
- **Learnable**: Unlike the original transformer paper's sinusoidal embeddings, GPT uses learnable position embeddings

The final embedding is the sum of token and position embeddings:

```python
x = token_embeds + position_embeds
```

### 2. Transformer Block

The transformer block is the heart of GPT. Each block contains:

```
Input
  ↓
Layer Norm → Multi-Head Attention → Dropout → Add (Residual)
  ↓
Layer Norm → Feed-Forward Network → Dropout → Add (Residual)
  ↓
Output
```

#### Residual Connections

Residual (skip) connections allow gradients to flow directly through the network, enabling training of very deep models. The pattern is:

```python
x = x + sublayer(layer_norm(x))
```

This is called **pre-norm** (normalization before the sublayer), which is more stable than post-norm.

#### Layer Normalization

Normalizes activations across the embedding dimension:

```python
normalized = (x - mean(x)) / sqrt(var(x) + eps)
output = scale * normalized + shift
```

This stabilizes training and allows for larger learning rates.

### 3. Multi-Head Attention

The attention mechanism allows each token to attend to all previous tokens. We'll cover this in detail in the next section, but at a high level:

- **Query, Key, Value**: Each token generates Q, K, V vectors
- **Attention Scores**: Computes similarity between queries and keys
- **Weighted Sum**: Combines values based on attention scores
- **Multiple Heads**: Parallel attention mechanisms capture different relationships

### 4. Feed-Forward Network

A simple two-layer MLP applied to each token independently:

```python
FFN(x) = Linear(4 * embedding_dimension, embedding_dimension)(
    GELU(Linear(embedding_dimension, 4 * embedding_dimension)(x))
)
```

- **Expansion Factor**: Typically 4x (embedding_dimension → 4*embedding_dimension → embedding_dimension)
- **Activation**: GELU (Gaussian Error Linear Unit) is used instead of ReLU
- **Position-wise**: Applied independently to each token position

### 5. Output Layer

The final layers convert hidden states to vocabulary predictions:

```python
x = final_layer_norm(x)
logits = linear_head(embedding_dimension, vocab_size)(x)
```

- **Final Normalization**: One more layer norm for stability
- **Language Model Head**: Linear projection to vocabulary size
- **No Bias**: Typically, the LM head has no bias term (ties weights with token embeddings)

## Architecture Variants

### GPT vs. Original Transformer

The original transformer had:
- **Encoder-Decoder**: Two stacks (encoder for input, decoder for output)
- **Cross-Attention**: Decoder attends to encoder outputs
- **Bidirectional**: Encoder can see full input

GPT uses:
- **Decoder-Only**: Single stack of transformer blocks
- **Causal Masking**: Can only attend to previous tokens
- **Autoregressive**: Generates one token at a time

### GPT Configuration

In this repository, you configure GPT with:

```python
config = GPTConfig(
    vocab_size=50257,              # Vocabulary size
    context_length=1024,            # Maximum sequence length
    embedding_dimension=768,        # Embedding dimension
    number_of_heads=12,             # Number of attention heads
    number_of_layers=12,            # Number of transformer blocks
    dropout_rate=0.1,               # Dropout rate
    query_key_value_bias=False      # Whether to use bias in QKV projections
)
```

## Parameter Count

The number of parameters in a GPT model:

1. **Embeddings**: `vocab_size * embedding_dimension + context_length * embedding_dimension`
2. **Each Transformer Block**:
   - Attention: `4 * embedding_dimension^2` (Q, K, V, output projections)
   - Feed-Forward: `2 * embedding_dimension * 4 * embedding_dimension = 8 * embedding_dimension^2`
   - Layer Norms: `2 * embedding_dimension * 2` (scale and shift per norm)
   - Total per block: `~12 * embedding_dimension^2`
3. **Output**: `embedding_dimension * vocab_size` (LM head)

**Total**: `number_of_layers * 12 * embedding_dimension^2 + vocab_size * embedding_dimension + ...`

For a small model (embedding_dimension=256, number_of_layers=4, vocab_size=50257):
- ~4.2M parameters

For GPT-2 Small (embedding_dimension=768, number_of_layers=12):
- ~117M parameters

## Forward Pass Flow

Here's what happens during a forward pass:

1. **Input**: `[batch_size, seq_len]` token indices
2. **Embed**: `[batch_size, seq_len, embedding_dimension]` dense vectors
3. **For each transformer block**:
   - Normalize → Attention → Add residual
   - Normalize → FFN → Add residual
4. **Final norm**: Normalize all hidden states
5. **Project**: `[batch_size, seq_len, vocab_size]` logits
6. **Output**: Probabilities for next token at each position

## Key Design Choices

### Why Pre-Norm?

Pre-norm (normalization before sublayer) is more stable than post-norm:
- Better gradient flow
- Allows deeper networks
- More common in modern architectures

### Why Residual Connections?

- Enable training of very deep networks (12-96 layers)
- Allow information to skip layers
- Help with vanishing gradient problem

### Why Causal Masking?

GPT is autoregressive—it must only see previous tokens:
- Prevents "cheating" during training
- Matches inference behavior
- Enables next-token prediction

### Why Multiple Heads?

Different attention heads learn different patterns:
- Some focus on syntax
- Others on semantics
- Some on long-range dependencies
- Parallel processing is efficient

## Computational Complexity

- **Time Complexity**: O(n²) where n is sequence length (due to attention)
- **Space Complexity**: O(n²) for attention matrices
- **Parallelization**: All tokens processed simultaneously (unlike RNNs)

This quadratic complexity is why context length is limited. Techniques like sparse attention help scale to longer sequences.

## Implementation Details

In this codebase:

- **Modular Design**: Each component is a separate module
- **Config-Driven**: Architecture defined by `GPTConfig`
- **PyTorch Native**: Uses standard PyTorch layers
- **Educational**: Code is clear and well-commented

The `GPTModel` class in `src/model/gpt.py` assembles all components:

```python
class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        # Embeddings
        # Transformer blocks (number_of_layers)
        # Final norm and LM head
```

## Summary

The transformer architecture consists of:
- **Embeddings**: Token + position
- **Transformer Blocks**: Attention + FFN with residuals
- **Output Layer**: Final norm + vocabulary projection

Key innovations:
- Parallel processing (vs. sequential RNNs)
- Attention mechanism (vs. fixed recurrence)
- Residual connections (enables deep networks)
- Layer normalization (stabilizes training)

---

Previous: [Overview](00-overview.md) | Next: [Attention Mechanism](02-attention-mechanism.md)
