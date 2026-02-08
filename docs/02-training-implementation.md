# Training Implementation

How training works in practice - the complete pipeline from data to trained model.

## Overview

This document walks through the training implementation, focusing on what actually happens during training and the practical details you need to know.

**Main files:**
- **Training script**: `examples/train_tiny_stories.py` - Complete training script you can run
- **Trainer class**: `src/training/trainer.py` - Core training logic
- **Dataset class**: `src/data/dataset.py` - Data preparation
- **Model**: `src/model/gpt.py` - The GPT model itself

## Training Pipeline

The training process:

1. **Data Loading** - Load and tokenize text
2. **Dataset Creation** - Create training sequences
3. **Model Initialization** - Create model with config
4. **Training Loop** - Forward pass, loss, backward pass, update
5. **Validation** - Evaluate on held-out data
6. **Checkpointing** - Save model state

## Data Preparation

### Tokenization

The code uses `tiktoken` with GPT-2 encoding. This happens in `examples/train_tiny_stories.py`:

**Location**: `examples/train_tiny_stories.py`, around line 232-234

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
```

**Key points:**
- Vocabulary size: 50,257 tokens
- BPE (Byte Pair Encoding) - handles out-of-vocabulary words
- Special token `<|endoftext|>` marks document boundaries

**Why GPT-2 tokenizer?**
- Widely used and well-tested
- Good balance of vocabulary size and coverage
- Compatible with many existing models

### Dataset Creation

The `GPTDataset` class creates training sequences from text.

**Location**: `src/data/dataset.py`, lines 10-40

```python
class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer, maximum_length: int, stride: int):
        # Tokenize entire text once
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        # Create sliding window sequences
        self.input_ids = []
        self.target_ids = []
        
        for i in range(0, len(token_ids) - maximum_length, stride):
            input_chunk = token_ids[i:i + maximum_length]
            target_chunk = token_ids[i + 1:i + maximum_length + 1]  # Shifted by 1
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
```

**Sliding window approach:**
- Creates overlapping sequences
- `stride` controls overlap (typically `context_length // 2`)
- More training examples from same text

**Example:**
```
Text: [t1, t2, t3, t4, t5, t6, t7, t8, ...]
context_length = 4, stride = 2

Sequence 1: input=[t1,t2,t3,t4], target=[t2,t3,t4,t5]
Sequence 2: input=[t3,t4,t5,t6], target=[t4,t5,t6,t7]
Sequence 3: input=[t5,t6,t7,t8], target=[t6,t7,t8,t9]
```

**Why overlap?**
- More training examples
- Better learning (each token appears in multiple contexts)
- Especially important for small datasets

**Usage in training script**: `examples/train_tiny_stories.py`, around line 252-257

```python
full_dataset = GPTDataset(
    text=text,
    tokenizer=tokenizer,
    maximum_length=context_length,
    stride=max(1, context_length // 2)  # 50% overlap, at least 1
)
```

### Train/Validation Split

**Location**: `examples/train_tiny_stories.py`, around line 278-292

```python
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)
```

**Standard split:** 90% train, 10% validation

**Why validation set?**
- Monitor overfitting (training loss can decrease while validation loss increases)
- Choose best model checkpoint
- Early stopping decisions

## Training Loop

The `GPTTrainer` class handles the actual training logic.

**Location**: `src/training/trainer.py`, lines 10-66

### Trainer Initialization

**Location**: `src/training/trainer.py`, lines 13-19

```python
class GPTTrainer:
    def __init__(self, model, train_loader, val_loader=None, optimizer=None, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=3e-4)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
```

**Usage in training script**: `examples/train_tiny_stories.py`, around line 337-343

```python
trainer = GPTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device
)
```

### Forward Pass

**Location**: `src/training/trainer.py`, line 32

```python
logits = self.model(input_ids)  # [batch_size, seq_len, vocab_size]
```

**What happens:**
1. Model processes input tokens
2. Outputs logits (raw scores) for each position
3. Each position has scores for all vocabulary tokens

### Loss Computation

**Location**: `src/training/trainer.py`, lines 33-36

```python
# Reshape for cross-entropy
logits = logits.view(-1, logits.size(-1))  # [batch*seq_len, vocab_size]
targets = target_ids.view(-1)               # [batch*seq_len]

loss = self.criterion(logits, targets)
```

**Why reshape?**
- Cross-entropy expects 2D logits and 1D targets
- Flatten batch and sequence dimensions
- Each position is a separate prediction

**Cross-entropy loss:**
- Measures how well predicted probabilities match actual tokens
- Lower = better predictions
- Standard for language modeling

### Backward Pass

**Location**: `src/training/trainer.py`, lines 38-39

```python
loss.backward()        # Compute gradients
self.optimizer.step() # Update parameters
```

**Note**: `zero_grad()` is called at line 30, before the forward pass.

**Gradient flow:**
- `backward()` computes gradients via backpropagation
- `step()` updates parameters using optimizer (AdamW)
- `zero_grad()` clears gradients (must call before next forward pass)

### Complete Training Epoch

**Location**: `src/training/trainer.py`, lines 21-43

```python
def train_epoch(self):
    """Train for one epoch."""
    self.model.train()  # Set to training mode
    total_loss = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(tqdm(self.train_loader)):
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        self.optimizer.zero_grad()  # Clear gradients
        
        logits = self.model(input_ids)  # Forward pass
        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        loss.backward()        # Backward pass
        self.optimizer.step()  # Update parameters
        
        total_loss += loss.item()
    
    return total_loss / len(self.train_loader)  # Average loss
```

### Optimizer Setup

**Location**: `examples/train_tiny_stories.py`, around line 329-334

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,      # Default: 3e-4
    weight_decay=0.1,      # L2 regularization
    betas=(0.9, 0.95)      # Momentum parameters
)
```

**AdamW parameters:**
- **Learning rate (3e-4)**: Common starting point for transformers
- **Weight decay (0.1)**: Prevents overfitting
- **Betas (0.9, 0.95)**: Momentum for gradient and squared gradient

**Why AdamW?**
- Adaptive learning rates (different for each parameter)
- Works well for transformers
- Weight decay decoupled from gradient updates (better than L2 in Adam)

## Training Script Walkthrough

The main training script is `examples/train_tiny_stories.py`. This is the file you run to train a model.

### Model Creation

**Location**: `examples/train_tiny_stories.py`, around line 316-324

```python
config = ModelConfig(
    vocab_size=vocab_size,           # From tokenizer (50257)
    context_length=context_length,   # Max sequence length
    embedding_dimension=embedding_dimension,  # Default: 256
    number_of_heads=number_of_heads,         # Default: 4
    number_of_layers=number_of_layers,       # Default: 4
    dropout_rate=0.1,
    use_attention_bias=False
)

model = GPTModel(config)
```

**Parameter counting**: `examples/train_tiny_stories.py`, function `create_model()` around line 152-177

```python
def create_model(config):
    """Create and initialize GPT model."""
    model = GPTModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    return model
```

### Main Training Loop

**Location**: `examples/train_tiny_stories.py`, around line 355-424

```python
for epoch in range(1, epochs + 1):
    # Training
    train_loss = trainer.train_epoch()  # Calls GPTTrainer.train_epoch()
    
    # Validation
    if epoch % eval_every == 0:
        val_loss = trainer.validate()  # Calls GPTTrainer.validate()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.to_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Generate samples
    if epoch % generate_every == 0:
        # Generate sample text (see examples/train_tiny_stories.py, lines 389-410)
        ...
    
    # Save checkpoint every epoch
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({...}, checkpoint_path)
```

### Validation

**Location**: `src/training/trainer.py`, lines 45-66

```python
def validate(self):
    """Validate the model."""
    if self.val_loader is None:
        return None
    
    self.model.eval()  # Set to evaluation mode
    total_loss = 0
    
    with torch.no_grad():  # No gradients needed
        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits = self.model(input_ids)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            total_loss += loss.item()
    
    return total_loss / len(self.val_loader)
```

**Key differences from training:**
- `model.eval()` - Disables dropout
- `torch.no_grad()` - No gradient computation (saves memory)
- No `optimizer.step()` - Don't update parameters

### Checkpoint Saving

**Location**: `examples/train_tiny_stories.py`, around line 377-386 (best model) and 415-423 (regular checkpoints)

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config.to_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
}
torch.save(checkpoint, checkpoint_path)
```

**What to save:**
- **Model weights**: To load and use the model
- **Optimizer state**: To resume training smoothly
- **Config**: To recreate model architecture
- **Losses**: To track training history

**Why save optimizer state?**
- AdamW maintains per-parameter momentum
- Resuming without it loses this information
- Training becomes less effective

### Loading Checkpoints

**Location**: `examples/generate_text.py`, around line 58-75 (example of loading)

```python
checkpoint = torch.load('checkpoint.pt', map_location='cpu')

# Recreate config
config = ModelConfig(**checkpoint['config'])

# Create and load model
model = GPTModel(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Resume training (if needed)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Understanding Training Output

When you run `train_tiny_stories.py`, you'll see output like this for each epoch:

```
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2437/2437 [29:25<00:00,  1.38it/s]
Train Loss: 2.3067 | Perplexity: 10.04
Val Loss: 2.1191 | Perplexity: 8.32
✓ Saved best model to checkpoints/best_model.pt

Generating sample text...
  Prompt: 'Once upon a time'
  Output: Once upon a time, there was a little boy named Timmy. Timmy loved to play outside in his backyard. He had a big yard with lots of trees,

  Prompt: 'The little girl'
  Output: The little girl said yes. She said goodbye to the swing and ran off to play. She was so happy and grateful. The little girl smiled back at the swing

  Prompt: 'In a far away land'
  Output: In a far away land, but he couldn't find anything.

Then he heard a voice. It was the voice. "What's going there?" Tom asked.

✓ Saved checkpoint to checkpoints/checkpoint_epoch_2.pt
```

### Progress Bar

```
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2437/2437 [29:25<00:00,  1.38it/s]
```

**What it shows:**
- **Progress bar**: Visual indicator of training progress (filled bar = complete)
- **2437/2437**: Current batch / Total batches in this epoch
- **[29:25<00:00]**: Time elapsed / Estimated time remaining
  - `29:25` = 29 minutes and 25 seconds elapsed
  - `<00:00` = Less than a second remaining (almost done)
- **1.38it/s**: Processing speed (1.38 batches per second)

**What to expect:**
- Speed depends on your hardware (GPU is much faster than CPU)
- Larger models or batches = slower processing
- First epoch may be slower (data loading, initialization)

### Training Metrics

```
Train Loss: 2.3067 | Perplexity: 10.04
Val Loss: 2.1191 | Perplexity: 8.32
```

**Train Loss (2.3067):**
- Average loss on training data for this epoch
- Lower is better
- Typical range: 2-5 for well-trained small models
- Should decrease over epochs (model is learning)

**Train Perplexity (10.04):**
- More intuitive than raw loss
- "Model is as surprised as choosing from 10 equally likely tokens"
- Lower is better
- Typical range: 10-50 for small models, 2-5 for large models

**Val Loss (2.1191):**
- Average loss on validation (held-out) data
- Lower is better
- Should be similar to or slightly higher than train loss
- **Red flag**: If val loss increases while train loss decreases = overfitting

**Val Perplexity (8.32):**
- Validation perplexity
- Lower than train perplexity is good (model generalizes well)
- If much higher than train perplexity = overfitting

**What to look for:**
- **Good**: Both losses decreasing, val loss close to train loss
- **Warning**: Val loss increasing while train loss decreases (overfitting)
- **Problem**: Losses not decreasing (learning rate too low, or other issues)

### Best Model Saving

```
✓ Saved best model to checkpoints/best_model.pt
```

**What this means:**
- Current validation loss is the lowest seen so far
- Model weights are saved to `checkpoints/best_model.pt`
- This is the model you should use for generation (best performance)

**When it appears:**
- Only when validation loss improves
- May not appear every epoch (only when model gets better)

### Sample Text Generation

```
Generating sample text...
  Prompt: 'Once upon a time'
  Output: Once upon a time, there was a little boy named Timmy. Timmy loved to play outside in his backyard. He had a big yard with lots of trees,
```

**What this shows:**
- Model generates text from fixed prompts during training
- Appears every `--generate-every` epochs (default: 2)
- Provides qualitative feedback (more intuitive than numbers)

**What to look for:**
- **Good**: Coherent sentences, proper grammar, relevant to prompt
- **Early training**: Repetitive, nonsensical, or incomplete sentences (normal)
- **Improving**: Text quality should improve over epochs

**Example progression:**
- **Epoch 1**: "Once upon a time the the the cat cat cat..." (repetitive)
- **Epoch 5**: "Once upon a time there was a cat. The cat was happy." (basic)
- **Epoch 10**: "Once upon a time, there was a little boy named Timmy. Timmy loved to play..." (coherent)

### Checkpoint Saving

```
✓ Saved checkpoint to checkpoints/checkpoint_epoch_2.pt
```

**What this means:**
- Model state saved at end of epoch
- Includes model weights, optimizer state, config, losses
- Can resume training from this point if needed

**File naming:**
- `checkpoint_epoch_N.pt`: Saved every epoch (or every `--save-every` epochs)
- `best_model.pt`: Best model based on validation loss (updated when val loss improves)

## Metrics and Monitoring

### Loss

**Location**: `src/training/trainer.py`, line 33-36 (computation), line 41 (accumulation)

```python
loss = self.criterion(logits, targets)  # Cross-entropy loss
total_loss += loss.item()  # Accumulate
```

**Interpretation:**
- Lower is better
- Typical values: 2-5 for well-trained small models
- Untrained model: ~10-11 (near random)

### Perplexity

**Location**: `examples/train_tiny_stories.py`, around line 361-362

```python
train_perplexity = torch.exp(torch.tensor(train_loss)).item()
```

**Interpretation:**
- More intuitive than raw loss
- "How many equally likely tokens does the model think it's choosing from?"
- Lower is better
- Typical values: 10-50 for small models, 2-5 for large models

**Example:**
- Perplexity = 10 → Model is as "surprised" as choosing from 10 equally likely tokens
- Perplexity = 50,000 → Model is completely random (vocab size)

### Sample Generation During Training

**Location**: `examples/train_tiny_stories.py`, around line 389-410

```python
if epoch % generate_every == 0:
    print("\nGenerating sample text...")
    model.eval()
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a far away land",
    ]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        output_ids = generate_text(
            model,
            input_ids,
            maximum_new_tokens=30,
            temperature=0.8,
            top_k_tokens=50
        )
        output_text = tokenizer.decode(output_ids)
        print(f"  Prompt: '{prompt}'")
        print(f"  Output: {output_text}")
    
    model.train()  # Back to training mode
```

**Why generate during training?**
- Qualitative feedback (more intuitive than loss)
- See model improve over time
- Catch issues early (repetition, nonsense, etc.)

## Common Training Issues

### Loss Not Decreasing

**Possible causes:**
- Learning rate too low
- Data not loading correctly
- Model architecture issue
- Gradients not flowing

**Debugging code** (add to training script):

```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}")
    else:
        print(f"{name}: grad_norm={param.grad.norm()}")
```

### Loss Exploding

**Possible causes:**
- Learning rate too high
- Gradient clipping needed
- Numerical instability

**Solution** (add to `src/training/trainer.py` after `loss.backward()`):

```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Overfitting

**Symptoms:**
- Training loss decreases
- Validation loss increases or plateaus

**Solutions:**
- Increase dropout (in `ModelConfig`)
- Add more training data
- Reduce model size
- Increase weight decay (in optimizer)
- Early stopping

### Out of Memory

**Solutions:**
- Reduce batch size (in `DataLoader`)
- Reduce context length (in `ModelConfig`)
- Reduce model size (embedding_dim, num_layers)
- Use gradient accumulation (simulate larger batch)

**Gradient accumulation** (modify `src/training/trainer.py`):

```python
# Gradient accumulation
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Training Best Practices

### Start Small

- Small model (128-256 embedding dim, 2-4 layers)
- Small dataset (5,000-10,000 samples)
- Few epochs (5-10)
- Verify pipeline works before scaling up

### Monitor Closely

- Watch training and validation loss
- Generate samples regularly
- Check for overfitting
- Save checkpoints frequently

### Reproducibility

**Location**: Add at the start of `examples/train_tiny_stories.py` `main()` function

```python
# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)
```

### Device Selection

**Location**: `examples/train_tiny_stories.py`, around line 216-222

```python
# Set device
if device == "cuda" and not torch.cuda.is_available():
    print("CUDA not available, using CPU")
    device = "cpu"
elif device == "mps" and not torch.backends.mps.is_available():
    print("MPS not available, using CPU")
    device = "cpu"
device = torch.device(device)
```

## Adaptive Context Length

The training script includes adaptive context length handling.

**Location**: `examples/train_tiny_stories.py`, around line 240-250

```python
# Check text length and adjust context_length if needed
token_ids_preview = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
text_length = len(token_ids_preview)
print(f"Text length: {text_length:,} tokens")

# If text is shorter than context_length, reduce context_length
if text_length < context_length:
    print(f"Warning: Text length ({text_length}) is shorter than context_length ({context_length})")
    print(f"Reducing context_length to {text_length // 2} to create training sequences")
    context_length = max(32, text_length // 2)  # At least 32, or half of text length
    print(f"Using context_length: {context_length}")
```

**Why this matters:**
- Prevents errors with small sample datasets
- Ensures at least some sequences can be created
- Minimum of 32 tokens for reasonable sequences

## File Reference Summary

Here's where to find each piece of code:

| Component | File | Lines |
|-----------|------|-------|
| **Main training script** | `examples/train_tiny_stories.py` | Entire file |
| **Trainer class** | `src/training/trainer.py` | 10-66 |
| **Dataset class** | `src/data/dataset.py` | 10-40 |
| **Model class** | `src/model/gpt.py` | 51-91 |
| **Config class** | `src/config.py` | 9-36 |
| **Generation function** | `src/generation/generate.py` | 8-69 |
| **Example: loading checkpoint** | `examples/generate_text.py` | 58-75 |

## Next Steps

- **Using the model**: See [Using the Model](03-using-the-model.md) to see how to generate text
- **Common issues**: See [Pitfalls and Challenges](04-pitfalls-and-challenges.md) for common mistakes
- **Quick reference**: See [Quick Reference](QUICK_REFERENCE.md) for commands and snippets
