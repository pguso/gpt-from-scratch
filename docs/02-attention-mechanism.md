# Attention Mechanism

Deep dive into how attention works in transformers.

## Introduction

The attention mechanism is the core innovation that makes transformers powerful. It allows the model to focus on relevant parts of the input when processing each token. In GPT, we use **self-attention** (tokens attend to other tokens in the same sequence) with **causal masking** (can only see previous tokens).

## The Intuition

Think of attention like reading a sentence:
- When you see "it" in "The cat sat on the mat. It was fluffy."
- You need to look back at "cat" to understand what "it" refers to
- Attention does this automatically—it learns which previous tokens are relevant

## Scaled Dot-Product Attention

The fundamental attention operation:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I represent?"
- **V (Value)**: "What information do I contain?"

### Step-by-Step

1. **Compute Attention Scores**: `Q @ K^T`
   - Measures similarity between queries and keys
   - Shape: `[batch, num_heads, seq_len, seq_len]`

2. **Scale**: Divide by `√d_k` (head dimension)
   - Prevents softmax from saturating
   - Keeps gradients healthy

3. **Apply Mask**: Set future positions to `-inf`
   - Ensures causal (autoregressive) behavior
   - Only previous tokens are visible

4. **Softmax**: Convert scores to probabilities
   - Each row sums to 1
   - Represents attention weights

5. **Weighted Sum**: Multiply attention weights by values
   - Combines information from relevant tokens
   - Output shape: `[batch, num_heads, seq_len, head_dim]`

## Multi-Head Attention

Instead of one attention mechanism, we use multiple "heads" in parallel:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Why Multiple Heads?

Different heads learn different patterns:
- **Head 1**: Might focus on syntactic relationships (subject-verb)
- **Head 2**: Might capture semantic similarity
- **Head 3**: Might track long-range dependencies
- **Head 4**: Might identify entity relationships

### Implementation Details

1. **Split Dimensions**: `embedding_dimension` is divided into `number_of_heads` heads
   - Each head has dimension `head_dim = embedding_dimension / number_of_heads`
   - Example: 768 dims, 12 heads → 64 dims per head

2. **Parallel Computation**: All heads computed simultaneously
   - Efficient on GPUs
   - No sequential dependency

3. **Concatenation**: Heads are concatenated back together
   - Shape: `[batch, seq_len, number_of_heads * head_dim]`

4. **Output Projection**: Final linear layer
   - Combines information from all heads
   - Output shape: `[batch, seq_len, embedding_dimension]`

## Causal Masking

GPT is autoregressive—it must only see previous tokens:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# Upper triangle is True (masked), lower triangle is False (visible)
```

### Example

For sequence `["The", "cat", "sat"]`:

```
      The   cat   sat
The    ✓     ✗     ✗
cat    ✓     ✓     ✗
sat    ✓     ✓     ✓
```

- "The" can only attend to itself
- "cat" can attend to "The" and itself
- "sat" can attend to all previous tokens

### Why This Matters

- **Training**: Model learns to predict next token from context
- **Inference**: Matches how generation works (only previous tokens available)
- **Causality**: Prevents information leakage from future

## Query, Key, Value Explained

### The Analogy

Think of attention like a library search:

- **Query (Q)**: Your search question ("books about transformers")
- **Key (K)**: Book titles/index (what each book is about)
- **Value (V)**: Book content (the actual information)

You:
1. Compare your query to all keys (compute QK^T)
2. Find relevant books (high attention scores)
3. Read those books (weighted sum of values)

### In GPT

- **Query**: "What information do I need from other tokens?"
- **Key**: "What information do I provide?"
- **Value**: "What is my actual content?"

Each token generates Q, K, V through learned linear projections.

## Attention Visualization

Here's what attention looks like for the sentence "The cat sat on the mat":

```
Token    Attention Weights
The      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cat      [0.3, 0.7, 0.0, 0.0, 0.0, 0.0]
sat      [0.1, 0.2, 0.7, 0.0, 0.0, 0.0]
on       [0.0, 0.1, 0.2, 0.7, 0.0, 0.0]
the      [0.0, 0.0, 0.0, 0.1, 0.9, 0.0]
mat      [0.0, 0.0, 0.0, 0.0, 0.1, 0.9]
```

Notice:
- Each row sums to 1.0 (probability distribution)
- Future tokens have 0.0 weight (causal masking)
- Strong self-attention (diagonal is often high)
- Some tokens attend to related words (e.g., "the" → "mat")

## Mathematical Formulation

### Single Head

Given input `X` of shape `[batch, seq_len, embedding_dimension]`:

1. **Project to Q, K, V**:
   ```
   Q = X W_Q  # [batch, seq_len, head_dim]
   K = X W_K  # [batch, seq_len, head_dim]
   V = X W_V  # [batch, seq_len, head_dim]
   ```

2. **Compute Attention Scores**:
   ```
   scores = Q K^T / √d_k  # [batch, seq_len, seq_len]
   ```

3. **Apply Mask and Softmax**:
   ```
   scores = scores.masked_fill(mask, -inf)
   attn_weights = softmax(scores, dim=-1)
   ```

4. **Weighted Sum**:
   ```
   output = attn_weights @ V  # [batch, seq_len, head_dim]
   ```

### Multi-Head

1. Split `embedding_dimension` into `number_of_heads` heads
2. Compute attention for each head independently
3. Concatenate: `[head_1, head_2, ..., head_h]`
4. Project: `output = concat @ W_O`

## Implementation in This Codebase

The `MultiHeadAttention` class in `src/model/attention.py` implements:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, ...):
        # Q, K, V projections
        self.W_query = nn.Linear(d_in, d_out, bias=query_key_value_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=query_key_value_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=query_key_value_bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Causal mask
        self.register_buffer("mask", ...)
    
    def forward(self, x):
        # 1. Project to Q, K, V
        # 2. Split into heads
        # 3. Compute scaled dot-product attention
        # 4. Combine heads
        # 5. Output projection
```

## Key Hyperparameters

- **`number_of_heads`**: Number of parallel attention heads (typically 8-16)
- **`head_dim`**: Dimension per head (`embedding_dimension / number_of_heads`)
- **`query_key_value_bias`**: Whether to use bias in QKV projections (GPT-2 uses False)
- **`dropout`**: Dropout rate on attention weights (typically 0.1)

## Computational Complexity

- **Time**: O(n²) where n is sequence length
  - Attention matrix is `[seq_len, seq_len]`
  - Each token attends to all previous tokens

- **Space**: O(n²) for storing attention matrix
  - This limits context length
  - Techniques like sparse attention help

- **Parallelization**: All tokens processed simultaneously
  - Much faster than RNNs (O(n) sequential steps)

## Common Patterns Learned

Attention heads learn various linguistic patterns:

1. **Syntactic**: Subject-verb, noun-adjective relationships
2. **Semantic**: Related concepts, synonyms
3. **Coreference**: Pronouns referring to entities
4. **Long-Range**: Dependencies across many tokens
5. **Positional**: Relative positions, distances

## Attention vs. RNNs

| Aspect | RNN/LSTM | Attention |
|--------|----------|-----------|
| Processing | Sequential | Parallel |
| Context | Limited (forgets) | Full (all tokens) |
| Dependencies | Hard to learn long-range | Easy long-range |
| Speed | Slow (sequential) | Fast (parallel) |
| Memory | O(n) | O(n²) |

## Tips for Understanding

1. **Visualize**: Use attention visualization tools (see notebooks and `examples/analyze_attention.py`)
2. **Experiment**: Try different numbers of heads
3. **Inspect**: Look at attention weights for specific examples
4. **Compare**: Contrast with and without masking

### Analyzing Attention in Your Model

This repository includes `examples/analyze_attention.py` for visualizing attention patterns:

```bash
# Analyze attention in a trained model
python examples/analyze_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --text "The cat sat on the mat" \
    --layer 2 \
    --head 0
```

This helps you understand:
- Which tokens attend to which other tokens
- How different heads learn different patterns
- How attention evolves across layers
- Whether the model is learning meaningful relationships

See [Advanced Topics](05-advanced-topics.md) for more details on attention analysis.

## Summary

Attention is the mechanism that allows transformers to:
- Process all tokens in parallel
- Capture long-range dependencies
- Learn complex linguistic patterns
- Scale to very large models

Key concepts:
- **Scaled dot-product**: QK^T / √d_k then softmax
- **Multi-head**: Parallel attention mechanisms
- **Causal masking**: Only see previous tokens
- **Query-Key-Value**: The three components of attention

---

Previous: [Transformer Architecture](01-transformer-architecture.md) | Next: [Building Blocks](03-building-blocks.md)
