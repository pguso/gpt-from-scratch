# Understanding Attention Analysis

A beginner-friendly guide to `analyze_attention.py` - how it works and what it shows.

> After training, attention patterns shift from diffuse self-focus to structured linguistic relationships such as subject–verb alignment, modifier attachment, and cross-sentence reference.

## What Does This Script Do?

The `analyze_attention.py` script lets you "see" what your GPT model is paying attention to when it processes text. It's like putting on X-ray glasses to see which words the model looks at when understanding each word.

**File location**: `examples/analyze_attention.py`

**What it produces:**
- Heatmaps showing attention patterns
- Text analysis showing which tokens attend to which
- Visualizations of how attention changes across layers

## The Big Picture

When the model processes "The cat sat on the mat", it needs to understand relationships:
- "cat" relates to "The" (the article)
- "sat" relates to "cat" (what the cat did)
- "mat" relates to "on" (what it sat on)
- "It" (if present) relates to "cat" (what "it" refers to)

Attention weights show these relationships as numbers: how much each token "pays attention" to other tokens.

## How It Works: PyTorch Hooks

The script uses **PyTorch hooks** to intercept the model's computation and extract attention weights.

### What Are Hooks?

Think of hooks as "listeners" that get called when something happens in the model. In this case, we hook into the attention layers to capture the attention weights as they're computed.

**Location**: `examples/analyze_attention.py`, lines 27-66

```python
class AttentionHook:
    """Hook to capture attention weights from MultiHeadAttention module."""
    
    def __init__(self):
        self.attention_weights = []  # Store captured weights
    
    def __call__(self, module, input, output):
        """This gets called during the forward pass"""
        # We intercept here and compute attention weights
        ...
```

**How hooks work:**
1. Register a hook on an attention module
2. When the model runs, the hook function is called
3. We compute attention weights inside the hook
4. Store them for later analysis

### Step-by-Step: How Attention Weights Are Extracted

**Location**: `examples/analyze_attention.py`, function `extract_attention_weights()` at lines 69-116

```python
def extract_attention_weights(model, input_ids, device='cpu'):
    # Step 1: Set model to evaluation mode
    model.eval()
    
    # Step 2: Create hooks for each transformer block
    hooks = []
    attention_hooks = []
    
    for i, block in enumerate(model.transformer_blocks):
        hook = AttentionHook()
        attention_hooks.append(hook)
        
        # Register the hook on the attention module
        handle = block.attention.register_forward_hook(hook)
        hooks.append(handle)
    
    # Step 3: Run the model (this triggers the hooks)
    with torch.no_grad():
        _ = model(input_ids)
    
    # Step 4: Extract the captured weights
    all_weights = []
    for hook in attention_hooks:
        all_weights.append(hook.attention_weights[0])
    
    # Step 5: Clean up - remove hooks
    for handle in hooks:
        handle.remove()
    
    return all_weights
```

**What happens:**
1. **Register hooks**: Attach a hook to each attention layer
2. **Run model**: Forward pass triggers hooks automatically
3. **Capture weights**: Each hook computes and stores attention weights
4. **Extract**: Collect all weights from all layers
5. **Clean up**: Remove hooks (important for memory)

## Understanding the AttentionHook Class

**Location**: `examples/analyze_attention.py`, lines 27-66

The hook manually recomputes attention weights because the attention module doesn't store them by default.

```python
def __call__(self, module, input, output):
    """Called during forward pass"""
    x = input[0]  # Input to attention layer
    
    # Step 1: Compute Q, K, V (same as attention module does)
    queries = module.W_query(x)
    keys = module.W_key(x)
    values = module.W_value(x)
    
    # Step 2: Split into heads
    queries = queries.view(batch_size, num_tokens, num_heads, head_dim)
    queries = queries.transpose(1, 2)  # [batch, heads, tokens, head_dim]
    
    # Step 3: Compute attention scores
    attention_scores = queries @ keys.transpose(-2, -1)
    
    # Step 4: Apply causal mask
    mask = module.mask[:num_tokens, :num_tokens]
    attention_scores = attention_scores.masked_fill(mask, float('-inf'))
    
    # Step 5: Convert to probabilities (attention weights)
    scaling_factor = module.head_dimension ** 0.5
    attention_weights = torch.softmax(attention_scores / scaling_factor, dim=-1)
    
    # Step 6: Store for later
    self.attention_weights.append(attention_weights[0].cpu().numpy())
```

**Why recompute?**
- The attention module computes weights internally but doesn't expose them
- We need to replicate the computation to capture them
- This gives us the exact weights used during the forward pass

## Understanding the Output

### Attention Weight Matrix

Each layer produces a matrix of attention weights:

```
Shape: [number_of_heads, sequence_length, sequence_length]

Example for 4 heads, 5 tokens:
Head 0: [5, 5] matrix
Head 1: [5, 5] matrix
Head 2: [5, 5] matrix
Head 3: [5, 5] matrix
```

**What the matrix means:**
- Rows = query tokens (the token "asking")
- Columns = key tokens (the token "being asked about")
- Values = attention weight (0.0 to 1.0, higher = more attention)

**Example matrix:**
```
        Key positions (attended to)
        0     1     2     3     4
Query 0 [0.9, 0.1, 0.0, 0.0, 0.0]  ← Token 0 mostly attends to itself
      1 [0.3, 0.6, 0.1, 0.0, 0.0]  ← Token 1 attends to 0 and 1
      2 [0.2, 0.3, 0.4, 0.1, 0.0]  ← Token 2 attends to multiple
      3 [0.1, 0.2, 0.3, 0.3, 0.1]  ← Token 3 spreads attention
      4 [0.1, 0.1, 0.2, 0.3, 0.3]  ← Token 4 attends to recent tokens
```

**Reading the matrix:**
- Row 2, Column 0 = 0.2 means token 2 pays 20% attention to token 0
- Diagonal values = self-attention (token attending to itself)
- Lower triangle = attention to previous tokens (causal masking)

## Visualization Functions

### Single Head Heatmap

**Location**: `examples/analyze_attention.py`, function `visualize_attention_head()` at lines 119-157

```python
def visualize_attention_head(attention_weights, tokens, layer_idx, head_idx, ax=None):
    # attention_weights: [sequence_length, sequence_length] matrix
    # tokens: List of token strings like ["The", "cat", "sat", ...]
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', vmin=0, vmax=1)
    
    # Label axes with tokens
    ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticklabels(tokens)
    
    ax.set_xlabel('Key (Attended To)')
    ax.set_ylabel('Query (Attending From)')
```

**What you see:**
- Blue heatmap: Darker blue = higher attention
- X-axis: Which tokens are being attended to
- Y-axis: Which tokens are doing the attending
- Each cell shows attention weight (0.0 = white, 1.0 = dark blue)

**Example interpretation:**
- Dark blue at (row=2, col=1) means token 2 ("sat") strongly attends to token 1 ("cat")
- This makes sense: "sat" needs to know what did the sitting

### All Heads Grid

**Location**: `examples/analyze_attention.py`, function `visualize_all_heads()` at lines 160-204

Shows all attention heads for one layer in a grid:

```python
def visualize_all_heads(attention_weights, tokens, layer_idx):
    # attention_weights: [number_of_heads, seq_len, seq_len]
    
    # Create grid: 4 columns, enough rows for all heads
    cols = 4
    rows = (number_of_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    # Plot each head
    for head_idx in range(number_of_heads):
        visualize_attention_head(attention_weights[head_idx], ...)
```

**Why view all heads?**
- Different heads learn different patterns
- Head 0 might focus on syntax, Head 1 on semantics
- Comparing heads shows diversity of learned patterns

### Attention Pattern Analysis

**Location**: `examples/analyze_attention.py`, function `analyze_attention_patterns()` at lines 207-240

Prints text analysis of attention patterns:

```python
def analyze_attention_patterns(attention_weights, tokens, top_k_tokens=5):
    for layer_idx, layer_weights in enumerate(attention_weights):
        # Average across all heads
        average_weights = layer_weights.mean(axis=0)
        
        for token_idx, token in enumerate(tokens):
            # Get attention from this token
            attention_from_token = average_weights[token_idx, :token_idx+1]
            
            # Find top-k most attended tokens
            top_indices = np.argsort(attention_from_token)[-top_k_tokens:][::-1]
            
            print(f"  '{token}' attends most to:")
            for idx, weight in zip(top_indices, top_weights):
                print(f"    '{tokens[idx]}' (weight: {weight:.3f})")
```

**Example output:**
```
Layer 0:
  'cat' attends most to:
    1. 'The' (weight: 0.450)
    2. 'cat' (weight: 0.350)  # self-attention
    3. 'sat' (weight: 0.200)

  'It' attends most to:
    1. 'cat' (weight: 0.600)  # coreference!
    2. 'mat' (weight: 0.250)
    3. 'It' (weight: 0.150)
```

**What this tells you:**
- Which tokens the model considers related
- How strong the relationships are (weights)
- Whether the model learned correct patterns (e.g., "It" → "cat")

## How to Use the Script

### Basic Usage

```bash
# Analyze a trained model
python examples/analyze_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --text "The cat sat on the mat. It was fluffy."
```

**What happens:**
1. Loads the model from checkpoint
2. Tokenizes the input text
3. Extracts attention weights from all layers
4. Creates visualizations
5. Saves plots to `attention_plots/` directory

### Command-Line Arguments

**Location**: `examples/analyze_attention.py`, `main()` function starting at line 288

```bash
# Analyze specific layer and head
python examples/analyze_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --text "The cat sat" \
    --layer 2 \
    --head 0 \
    --show-plots  # Display plots interactively

# Analyze all layers, limit heads shown
python examples/analyze_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --text "Once upon a time" \
    --max-heads 4  # Show only first 4 heads per layer

# Analyze untrained model (random weights)
python examples/analyze_attention.py \
    --text "The cat sat" \
    --embedding-dimension 256 \
    --number-of-heads 4 \
    --number-of-layers 4
```

**Key arguments:**
- `--checkpoint`: Path to trained model (optional, uses random weights if not provided)
- `--text`: Input text to analyze
- `--layer`: Specific layer to visualize (None = all layers)
- `--head`: Specific head to visualize (None = all heads)
- `--show-plots`: Display plots in window (otherwise just saves files)
- `--output-dir`: Where to save plots (default: `attention_plots/`)

## Understanding the Output Files

The script creates several files:

### 1. Single Head Heatmaps

**File**: `attention_plots/layer_X_head_Y.png`

Shows attention for one head in one layer:
- Heatmap with tokens on axes
- Darker = more attention
- Diagonal = self-attention
- Lower triangle = attention to previous tokens

### 2. All Heads Grid

**File**: `attention_plots/layer_X_all_heads.png`

Shows all heads for one layer in a grid:
- Compare patterns across heads
- See diversity of learned patterns
- Identify which heads focus on what

### 3. Attention Flow

**File**: `attention_plots/attention_flow.png`

Shows how attention changes across layers:
- X-axis: Layer number
- Y-axis: Average attention weight
- Lines show self-attention vs. attention to previous tokens

**What to look for:**
- Early layers: Often more self-attention
- Later layers: More attention to other tokens
- Trained models: Clear patterns
- Untrained models: Random patterns

## Common Patterns to Look For

### Good Patterns (Trained Model)

1. **Coreference resolution**: "It" attends strongly to its referent ("cat")
2. **Subject-verb relationships**: Verbs attend to their subjects
3. **Modifier relationships**: Adjectives attend to nouns they modify
4. **Long-range dependencies**: Tokens attending to distant but relevant tokens

### Problem Patterns

1. **Uniform attention**: All weights similar (model not learning)
2. **Only self-attention**: Tokens only attend to themselves (not using context)
3. **Random patterns**: No clear structure (untrained or undertrained)
4. **Attention to wrong tokens**: Model learned incorrect patterns

## Step-by-Step: What Happens When You Run It

1. **Load model** (`main()`, lines 354-410)
   - Load checkpoint if provided
   - Create model with correct config
   - Load weights

2. **Tokenize input** (lines 341-351)
   - Convert text to token IDs
   - Decode back to tokens for labels
   - Check context length

3. **Extract attention** (line 417)
   - Call `extract_attention_weights()`
   - Hooks capture weights during forward pass
   - Returns list of weight matrices

4. **Analyze patterns** (line 426)
   - Call `analyze_attention_patterns()`
   - Prints text analysis to console

5. **Create visualizations** (lines 428-465)
   - Generate heatmaps for each layer/head
   - Save to files
   - Optionally display

6. **Plot attention flow** (line 468)
   - Shows how attention changes across layers
   - Saves summary plot

## Troubleshooting

### "No attention weights captured"

**Problem**: Hooks didn't capture weights

**Solution**: Check that:
- Model has transformer blocks
- Hooks were registered correctly
- Forward pass actually ran

### "Out of memory"

**Problem**: Too many layers/heads or long sequences

**Solution**:
- Use shorter text (`--text` with fewer tokens)
- Analyze fewer layers at once
- Use `--max-heads` to limit visualization

### "Plots are all random"

**Problem**: Model is untrained or undertrained

**Solution**: This is expected! Untrained models have random attention patterns. Train the model first, then analyze.

### "Can't see patterns"

**Problem**: Model might need more training

**Solution**:
- Train for more epochs
- Check if validation loss is decreasing
- Try analyzing a well-trained model first

## Key Takeaways

1. **Hooks intercept computation**: They let us capture intermediate values during forward pass
2. **Attention weights are matrices**: Each shows how tokens relate to each other
3. **Different heads learn different patterns**: View all heads to see diversity
4. **Layers show progression**: Early layers = local patterns, later layers = complex relationships
5. **Visualization helps debugging**: See if model learned correct patterns

## Code Reference

| Function | Location | Purpose |
|----------|----------|---------|
| `AttentionHook` | Lines 27-66 | Captures attention weights via hook |
| `extract_attention_weights()` | Lines 69-116 | Main extraction function |
| `visualize_attention_head()` | Lines 119-157 | Single head heatmap |
| `visualize_all_heads()` | Lines 160-204 | Grid of all heads |
| `analyze_attention_patterns()` | Lines 207-240 | Text analysis |
| `plot_attention_flow()` | Lines 243-285 | Cross-layer analysis |
| `main()` | Lines 288-481 | Command-line interface |

## Next Steps

- **Try it yourself**: Run the script on a trained model
- **Experiment**: Try different texts and see how attention changes
- **Compare**: Analyze untrained vs. trained models
- **Debug**: Use attention patterns to understand model behavior

For more on using the model, see [Using the Model](03-using-the-model.md).
