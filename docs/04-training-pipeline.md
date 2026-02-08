# Training Pipeline

Understanding how to train GPT models from scratch.

## Introduction

Training a GPT model involves teaching it to predict the next token in a sequence by learning patterns from large amounts of text data. This document covers the fundamental concepts, components, and workflow of training GPT models. For a detailed walkthrough of the specific training script, see [Training Walkthrough](04-training-walkthrough.md).

## What You'll Learn

By the end of this document, you'll understand:
- The training objective: next-token prediction
- Data preparation and preprocessing
- The training loop structure
- Loss functions and metrics
- Validation and monitoring
- Checkpointing and model saving

## The Training Objective

GPT models are trained using **next-token prediction** (also called language modeling):

1. **Input**: A sequence of tokens `[t1, t2, t3, ..., tn]`
2. **Target**: The same sequence shifted by one: `[t2, t3, t4, ..., tn+1]`
3. **Goal**: Predict the next token at each position

### Example

Given the text: "The cat sat on the mat"

- **Input sequence**: `["The", "cat", "sat", "on", "the"]`
- **Target sequence**: `["cat", "sat", "on", "the", "mat"]`

At each position, the model learns to predict what comes next based on all previous tokens.

### Why This Works

- **Self-supervised**: No manual labeling needed-the text itself provides supervision
- **Scalable**: Can use massive amounts of unlabeled text data
- **General**: Learns language patterns, facts, reasoning, and style
- **Autoregressive**: Matches how generation works (predicting one token at a time)

## Data Preparation

### 1. Text Collection

Gather large amounts of text data:
- Books, articles, websites
- Code repositories (for code models)
- Domain-specific texts (for specialized models)

**Key Considerations:**
- **Quality matters**: Better data → better model
- **Diversity**: Various topics, styles, and domains
- **Size**: More data generally improves performance
- **Format**: Plain text, one document per line or separated by special tokens

### 2. Tokenization

Convert text into tokens (subword units):

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, world!"
token_ids = tokenizer.encode(text)  # [15496, 11, 995, 0]
```

**Why Tokenization?**
- **Vocabulary size**: Reduces vocabulary from millions of words to ~50k tokens
- **Out-of-vocabulary**: Handles new words by breaking them into subwords
- **Efficiency**: Smaller vocabulary = faster training and smaller models

**Common Tokenizers:**
- **BPE (Byte Pair Encoding)**: Used by GPT-2, GPT-3
- **SentencePiece**: Used by T5, LLaMA
- **WordPiece**: Used by BERT

### 3. Sequence Creation

Create training sequences from tokenized text:

```python
# Full text: [t1, t2, t3, ..., t1000]
# Context length: 128

# Sequence 1: [t1, t2, ..., t128] → target: [t2, t3, ..., t129]
# Sequence 2: [t65, t66, ..., t192] → target: [t66, t67, ..., t193]
# (with 50% overlap)
```

**Key Parameters:**
- **Context Length**: Maximum sequence length (e.g., 128, 256, 1024)
- **Stride**: Overlap between sequences (typically 50% of context length)
- **Overlap Benefits**: More training examples, better learning

### 4. Train/Validation Split

Split data into training and validation sets:

```python
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```

**Typical Split**: 90% train, 10% validation

**Why Validation?**
- **Monitor overfitting**: Training loss can decrease while validation loss increases
- **Model selection**: Choose best model based on validation performance
- **Early stopping**: Stop training when validation loss stops improving

## The Training Loop

### Basic Structure

```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits = model(input_ids)
            val_loss = criterion(logits, targets)
```

### Key Components

#### 1. Forward Pass

```python
logits = model(input_ids)  # Shape: [batch_size, seq_len, vocab_size]
```

- Model processes input tokens
- Outputs logits (raw scores) for each token position
- Each position has scores for all vocabulary tokens

#### 2. Loss Computation

```python
# Reshape for loss computation
logits = logits.view(-1, logits.size(-1))  # [batch*seq_len, vocab_size]
targets = targets.view(-1)  # [batch*seq_len]

# Cross-entropy loss
loss = F.cross_entropy(logits, targets)
```

**Cross-Entropy Loss:**
- Measures how well predicted probabilities match actual tokens
- Lower loss = better predictions
- Standard for classification and language modeling

#### 3. Backward Pass

```python
loss.backward()  # Compute gradients
optimizer.step()  # Update parameters
optimizer.zero_grad()  # Clear gradients for next iteration
```

**Gradient Flow:**
- Backpropagation computes gradients for all parameters
- Optimizer updates parameters to reduce loss
- Learning rate controls step size

#### 4. Optimization

**AdamW Optimizer** (standard for transformers):

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,  # Learning rate
    weight_decay=0.1,  # L2 regularization
    betas=(0.9, 0.95)  # Momentum parameters
)
```

**Why AdamW?**
- **Adaptive**: Different learning rates for different parameters
- **Momentum**: Helps escape local minima
- **Weight Decay**: Prevents overfitting
- **Proven**: Works well for transformers

## Loss Functions and Metrics

### Cross-Entropy Loss

The standard loss for language modeling:

```python
loss = -log(P(target_token | context))
```

**Interpretation:**
- Penalizes confident wrong predictions more
- Encourages high probability for correct tokens
- Smooth gradient flow

### Perplexity

More interpretable than raw loss:

```python
perplexity = torch.exp(loss)
```

**Interpretation:**
- **Perplexity = 10**: Model is as "surprised" as if choosing from 10 equally likely tokens
- **Lower is better**: Perplexity of 5 is better than 10
- **Realistic values**: 
  - Untrained model: ~50,000 (vocab size)
  - Well-trained small model: ~10-50
  - GPT-3 level: ~2-5

### Other Metrics

- **Token Accuracy**: Percentage of correctly predicted tokens
- **BLEU Score**: For translation tasks
- **ROUGE Score**: For summarization tasks

## Validation and Monitoring

### Why Validate?

- **Overfitting Detection**: Training loss decreases, validation loss increases
- **Model Selection**: Choose best checkpoint based on validation performance
- **Hyperparameter Tuning**: Compare different configurations

### Validation Loop

```python
model.eval()
total_val_loss = 0
with torch.no_grad():  # No gradients needed
    for batch in val_loader:
        logits = model(input_ids)
        loss = criterion(logits, targets)
        total_val_loss += loss.item()

avg_val_loss = total_val_loss / len(val_loader)
val_perplexity = torch.exp(torch.tensor(avg_val_loss))
```

**Key Points:**
- **`model.eval()`**: Disables dropout and batch norm updates
- **`torch.no_grad()`**: Saves memory, faster computation
- **No parameter updates**: Only evaluate, don't train

### Monitoring Training

Track these metrics:

1. **Training Loss**: Should decrease over time
2. **Validation Loss**: Should decrease (watch for overfitting)
3. **Perplexity**: More interpretable than loss
4. **Sample Generations**: Qualitative feedback
5. **Learning Rate**: If using scheduling

**Red Flags:**
- Validation loss increases while training loss decreases (overfitting)
- Loss not decreasing (learning rate too low, or other issues)
- Loss exploding (learning rate too high, gradient clipping needed)

## Checkpointing

### What to Save

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config.to_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
}
torch.save(checkpoint, 'checkpoint.pt')
```

**Why Each Component?**
- **Model weights**: To load and use the model
- **Optimizer state**: To resume training smoothly
- **Config**: To recreate model architecture
- **Losses**: To track training history

### Loading Checkpoints

```python
checkpoint = torch.load('checkpoint.pt')

# Recreate model
config = GPTConfig(**checkpoint['config'])
model = GPTModel(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Resume training
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Best Model Tracking

Save the model with lowest validation loss:

```python
best_val_loss = float('inf')

if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(checkpoint, 'best_model.pt')
```

## Training Workflow Summary

1. **Prepare Data**
   - Collect and clean text
   - Tokenize into sequences
   - Split into train/val sets

2. **Initialize Model**
   - Create model with desired architecture
   - Initialize optimizer
   - Set up device (CPU/GPU)

3. **Training Loop**
   - For each epoch:
     - Train on training set
     - Validate on validation set
     - Monitor metrics
     - Save checkpoints
     - Generate sample text (optional)

4. **Model Selection**
   - Choose best model based on validation loss
   - Load and use for generation

## Key Hyperparameters

| Hyperparameter | Typical Range | Impact |
|----------------|---------------|--------|
| **Learning Rate** | 1e-5 to 1e-3 | Controls step size (3e-4 common) |
| **Batch Size** | 16-128 | Memory vs. speed tradeoff |
| **Context Length** | 128-4096 | Longer = more context, more memory |
| **Epochs** | 5-50 | More = better, but diminishing returns |
| **Weight Decay** | 0.01-0.1 | Regularization strength |
| **Dropout** | 0.0-0.2 | Prevents overfitting |

## Common Challenges

### Overfitting

**Symptoms**: Training loss decreases, validation loss increases

**Solutions**:
- Increase dropout rate
- Add more training data
- Reduce model size
- Increase weight decay
- Early stopping

### Slow Training

**Solutions**:
- Use GPU instead of CPU
- Increase batch size (if memory allows)
- Reduce model size
- Reduce context length
- Use mixed precision training

### Out of Memory

**Solutions**:
- Reduce batch size
- Reduce context length
- Reduce model size
- Use gradient accumulation
- Use gradient checkpointing

### Loss Not Decreasing

**Solutions**:
- Check learning rate (might be too low)
- Verify data is being loaded correctly
- Check model architecture
- Ensure gradients are flowing
- Try different initialization

## Best Practices

1. **Start Small**: Begin with small models and datasets for faster iteration
2. **Monitor Closely**: Watch training and validation metrics
3. **Save Regularly**: Checkpoint frequently to avoid losing progress
4. **Validate Early**: Run validation from the start
5. **Use Validation Loss**: Choose best model based on validation, not training
6. **Generate Samples**: Qualitative feedback is valuable
7. **Reproducibility**: Set random seeds for reproducibility
8. **Documentation**: Keep track of hyperparameters and results

## Next Steps

Now that you understand the training pipeline:

1. **See Implementation**: Read [Training Walkthrough](04-training-walkthrough.md) for detailed script walkthrough
2. **Learn Optimizations**: See [Advanced Topics](05-advanced-topics.md) for training optimizations
3. **Try Training**: Run `python examples/train_tiny_stories.py`
4. **Experiment**: Modify hyperparameters and observe effects

---

Previous: [Building Blocks](03-building-blocks.md) | Next: [Training Walkthrough](04-training-walkthrough.md)
