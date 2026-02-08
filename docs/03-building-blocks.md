# Building Blocks

Understanding each component of the transformer.

## Introduction

This document breaks down each building block of GPT, explaining how they work and why they're designed the way they are. We'll cover embeddings, normalization, activation functions, and the complete transformer block.

## 1. Token Embeddings

### Purpose

Convert discrete token indices into continuous vector representations that capture semantic meaning.

### Implementation

```python
token_embedding = nn.Embedding(vocab_size, embedding_dimension)
```

- **Input**: Token indices `[0, 1, 2, ..., vocab_size-1]`
- **Output**: Dense vectors of shape `[vocab_size, embedding_dimension]`
- **Learnable**: Each token gets a learned vector representation

### How It Works

The embedding layer is essentially a lookup table:
- Token ID 15496 → Vector `[0.23, -0.45, 0.67, ...]`
- Similar tokens (learned during training) have similar vectors
- The embedding space captures semantic relationships

### Key Properties

- **High Dimensional**: Typically 256-4096 dimensions
- **Learned**: Initialized randomly, learned from data
- **Shared**: Same embeddings used for input and output (weight tying)

## 2. Position Embeddings

### Purpose

Since transformers process tokens in parallel (unlike RNNs), they need explicit position information.

### Implementation

```python
position_embedding = nn.Embedding(context_length, embedding_dimension)
```

- **Input**: Position indices `[0, 1, 2, ..., context_length-1]`
- **Output**: Position vectors of shape `[context_length, embedding_dimension]`
- **Learnable**: GPT uses learnable embeddings (unlike sinusoidal in original transformer)

### Why Learnable?

- **Flexibility**: Model learns optimal position encodings
- **Simplicity**: Easier to implement than sinusoidal
- **Performance**: Works well in practice

### Combining Embeddings

```python
x = token_embeds + position_embeds
```

The final embedding is the sum of token and position embeddings. This works because:
- Both are in the same space (same dimension)
- Addition is simple and effective
- The model learns to use both signals

## 3. Layer Normalization

### Purpose

Normalize activations to stabilize training and enable larger learning rates.

### Implementation

```python
class LayerNorm(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / sqrt(var + eps)
        return scale * normalized + shift
```

### Key Features

- **Normalize Across Features**: Computes mean/variance over the embedding dimension
- **Learnable Scale and Shift**: `scale` and `shift` are learnable parameters
- **Epsilon**: Small constant (1e-5) prevents division by zero

### Why Layer Norm?

- **Stability**: Prevents activations from growing too large
- **Faster Training**: Allows larger learning rates
- **Better Gradients**: Normalized inputs lead to better gradient flow

### Pre-Norm vs. Post-Norm

GPT uses **pre-norm** (normalization before sublayer):
```python
x = x + attention(layer_norm(x))  # Pre-norm
```

vs. post-norm:
```python
x = layer_norm(x + attention(x))  # Post-norm
```

Pre-norm is more stable for deep networks.

## 4. GELU Activation

### Purpose

Non-linear activation function used in the feed-forward network.

### Implementation

```python
def gelu(x):
    return 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

### Why GELU?

- **Smooth**: Unlike ReLU, it's smooth everywhere
- **Probabilistic Interpretation**: Can be seen as expected value of ReLU with Gaussian noise
- **Better Performance**: Often outperforms ReLU in transformers

### Comparison

- **ReLU**: `max(0, x)` - simple but not smooth
- **GELU**: Smooth approximation, better for language models
- **Swish**: Similar to GELU, `x * sigmoid(x)`

## 5. Feed-Forward Network

### Purpose

Processes each token's representation independently through a two-layer MLP.

### Implementation

```python
class FeedForward(nn.Module):
    def __init__(self, embedding_dimension):
        self.net = nn.Sequential(
            nn.Linear(embedding_dimension, 4 * embedding_dimension),  # Expand
            GELU(),                                                  # Activate
            nn.Linear(4 * embedding_dimension, embedding_dimension)  # Contract
        )
```

### Architecture

1. **Expansion**: `embedding_dimension → 4 * embedding_dimension`
   - Increases capacity
   - Allows complex transformations

2. **Activation**: GELU
   - Introduces non-linearity
   - Enables learning complex patterns

3. **Contraction**: `4 * embedding_dimension → embedding_dimension`
   - Returns to original dimension
   - Combines expanded features

### Why This Design?

- **Position-wise**: Applied independently to each token
- **Expansion Factor**: 4x is a good balance (not too large, not too small)
- **Bottleneck**: Forces the model to compress information

## 6. Multi-Head Attention

We covered this in detail in the previous section. Key points:

- **Query, Key, Value**: Three learned projections
- **Multiple Heads**: Parallel attention mechanisms
- **Causal Masking**: Only see previous tokens
- **Scaled Dot-Product**: QK^T / √d_k then softmax

See [Attention Mechanism](02-attention-mechanism.md) for details.

## 7. Transformer Block

### Complete Structure

The transformer block combines all components:

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Attention sublayer
        residual = x
        x = self.norm1(x)           # Pre-norm
        x = self.attention(x)        # Multi-head attention
        x = self.dropout(x)
        x = x + residual            # Residual connection
        
        # Feed-forward sublayer
        residual = x
        x = self.norm2(x)           # Pre-norm
        x = self.feed_forward(x)     # FFN
        x = self.dropout(x)
        x = x + residual            # Residual connection
        
        return x
```

### Two Sublayers

1. **Attention Sublayer**:
   - Normalize → Attend → Dropout → Add residual
   - Allows tokens to interact

2. **Feed-Forward Sublayer**:
   - Normalize → Transform → Dropout → Add residual
   - Processes each token independently

### Residual Connections

Why they matter:
- **Gradient Flow**: Allow gradients to flow directly
- **Identity Mapping**: Model can "skip" layers if needed
- **Deep Networks**: Enable training of very deep models (12-96 layers)

### Dropout

Applied after attention and FFN:
- **Regularization**: Prevents overfitting
- **Typical Rate**: 0.1 (10% of activations set to zero)
- **Training Only**: Disabled during inference

## 8. Complete GPT Model

### Assembly

```python
class GPTModel(nn.Module):
    def __init__(self, config):
        # 1. Embeddings
        self.token_embedding = nn.Embedding(...)
        self.position_embedding = nn.Embedding(...)
        
        # 2. Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(number_of_layers)]
        )
        
        # 3. Output
        self.final_norm = LayerNorm(embedding_dimension)
        self.lm_head = nn.Linear(embedding_dimension, vocab_size)
```

### Forward Pass

```python
def forward(self, input_ids):
    # 1. Embed
    token_embeds = self.token_embedding(input_ids)
    position_embeds = self.position_embedding(positions)
    x = token_embeds + position_embeds
    x = self.embedding_dropout(x)
    
    # 2. Transform
    x = self.transformer_blocks(x)
    
    # 3. Output
    x = self.final_norm(x)
    logits = self.lm_head(x)
    return logits
```

## Component Interactions

### Information Flow

1. **Embeddings** → Convert tokens to vectors
2. **Transformer Blocks** → Process and transform
3. **Final Norm** → Stabilize before output
4. **LM Head** → Predict next token

### Why This Order?

- **Embeddings First**: Need continuous representations
- **Blocks in Middle**: Core processing happens here
- **Norm Before Output**: Ensures stable predictions
- **Linear Head**: Simple projection to vocabulary

## Design Patterns

### Pre-Norm Architecture

All sublayers use pre-norm:
```python
x = x + sublayer(layer_norm(x))
```

Benefits:
- Better gradient flow
- More stable training
- Works better for deep networks

### Residual Everywhere

Every sublayer has a residual connection:
- Attention: `x = x + attention(norm(x))`
- FFN: `x = x + ffn(norm(x))`

### Dropout for Regularization

Applied after:
- Embeddings
- Attention output
- FFN output

But NOT after:
- Layer norm (would break normalization)
- Residual addition (would break identity)

## Hyperparameter Impact

### Embedding Dimension (`embedding_dimension`)

- **Small (256)**: Faster, less capacity
- **Medium (768)**: Good balance (GPT-2 base)
- **Large (4096)**: More capacity, slower

### Number of Layers (`number_of_layers`)

- **Few (4-6)**: Shallow, fast
- **Medium (12)**: Standard (GPT-2)
- **Many (24-96)**: Deep, powerful, slow

### Number of Heads (`number_of_heads`)

- **Few (4-8)**: Less parallel attention
- **Standard (12)**: Good diversity
- **Many (16-32)**: More specialized heads

### Context Length (`context_length`)

- **Short (128-256)**: Fast, limited context
- **Medium (1024)**: Standard (GPT-2)
- **Long (2048-4096)**: More context, slower

## Common Issues and Solutions

### Problem: Training Instability

**Solutions**:
- Use pre-norm (not post-norm)
- Lower learning rate
- Gradient clipping
- Proper initialization

### Problem: Overfitting

**Solutions**:
- Increase dropout rate
- More training data
- Data augmentation
- Regularization

### Problem: Slow Training

**Solutions**:
- Reduce model size
- Shorter context length
- Mixed precision training
- Better hardware

## Summary

Key building blocks:

1. **Embeddings**: Token + position → continuous vectors
2. **Layer Norm**: Normalize for stability
3. **Attention**: Token interactions
4. **FFN**: Per-token processing
5. **Residuals**: Enable deep networks
6. **Dropout**: Regularization

Design principles:
- **Pre-norm**: Normalize before sublayer
- **Residuals**: Add input to output
- **Modularity**: Each component is independent
- **Scalability**: Easy to scale up or down

---

Previous: [Attention Mechanism](02-attention-mechanism.md) | Next: [Training Pipeline](04-training-pipeline.md)
