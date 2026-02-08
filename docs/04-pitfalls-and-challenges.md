# Pitfalls and Challenges

Common mistakes and the challenging aspects of working with probabilistic language models.

## Overview

This document covers common pitfalls when implementing and using GPT models, and the unique challenges that arise from their probabilistic nature.

## Common Pitfalls

### 1. Forgetting to Set Model to Eval Mode

**Mistake:**
```python
model = load_model(...)
# Forgot model.eval()
logits = model(input_ids)  # Dropout still active!
```

**Problem:**
- Dropout layers remain active during inference
- Randomly zeros out activations
- Inconsistent outputs

**Solution:**
```python
model.eval()  # Always call this before inference
with torch.no_grad():
    logits = model(input_ids)
```

### 2. Not Handling Context Window Limits

**Mistake:**
```python
# Prompt is longer than context_length
long_prompt = "..." * 1000  # 2000 tokens
input_ids = tokenizer.encode(long_prompt)
output = generate_text(model, input_ids, ...)  # Error!
```

**Problem:**
- Model has fixed `context_length` (e.g., 128, 512, 1024)
- Can't process sequences longer than this
- Will error or truncate unexpectedly

**Solution:**
```python
# Truncate if needed
max_context = model.config.context_length
if len(input_ids) > max_context:
    input_ids = input_ids[-max_context:]  # Keep last N tokens
```

### 3. Mismatched Vocabulary Size

**Mistake:**
```python
# Using wrong tokenizer
tokenizer = tiktoken.get_encoding("gpt2")  # vocab_size = 50257
config = ModelConfig(vocab_size=10000)  # Wrong!
model = GPTModel(config)
```

**Problem:**
- Model expects different vocabulary size
- Token IDs may be out of range
- Runtime errors or incorrect outputs

**Solution:**
```python
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab  # 50257
config = ModelConfig(vocab_size=vocab_size)
```

### 4. Not Saving/Loading Config

**Mistake:**
```python
# Saving
torch.save({'model_state_dict': model.state_dict()}, 'checkpoint.pt')

# Loading (later)
checkpoint = torch.load('checkpoint.pt')
# How do I recreate the model? Config is missing!
```

**Problem:**
- Can't recreate model without config
- Need to remember architecture details
- Easy to make mistakes

**Solution:**
```python
# Always save config
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.to_dict(),
    ...
}, 'checkpoint.pt')

# Always load config
config = ModelConfig(**checkpoint['config'])
model = GPTModel(config)
```

### 5. Device Mismatch

**Mistake:**
```python
model = model.to('cuda')
input_ids = torch.tensor([1, 2, 3])  # On CPU!
logits = model(input_ids)  # Error: device mismatch
```

**Problem:**
- Model and input must be on same device
- Common when loading from CPU, using on GPU

**Solution:**
```python
device = torch.device('cuda')
model = model.to(device)
input_ids = input_ids.to(device)
```

### 6. Not Zeroing Gradients

**Mistake:**
```python
for batch in train_loader:
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()
    # Forgot optimizer.zero_grad()
    # Gradients accumulate!
```

**Problem:**
- Gradients accumulate across batches
- Training becomes unstable
- Model doesn't learn properly

**Solution:**
```python
for batch in train_loader:
    optimizer.zero_grad()  # Always clear first
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()
```

### 7. Wrong Loss Reshaping

**Mistake:**
```python
logits = model(input_ids)  # [B, T, V]
targets = target_ids  # [B, T]
loss = F.cross_entropy(logits, targets)  # Error: shape mismatch
```

**Problem:**
- Cross-entropy expects `[N, C]` logits and `[N]` targets
- Need to flatten batch and sequence dimensions

**Solution:**
```python
logits = logits.view(-1, logits.size(-1))  # [B*T, V]
targets = targets.view(-1)  # [B*T]
loss = F.cross_entropy(logits, targets)
```

### 8. Using Training Mode for Inference

**Mistake:**
```python
model.train()  # Still in training mode
with torch.no_grad():
    logits = model(input_ids)  # Dropout active!
```

**Problem:**
- Dropout and batch norm behave differently
- Inconsistent results
- Lower quality outputs

**Solution:**
```python
model.eval()  # Set to eval mode
with torch.no_grad():
    logits = model(input_ids)
```

### 9. Not Handling Empty Datasets

**Mistake:**
```python
dataset = GPTDataset(text, tokenizer, max_length=128, stride=64)
# Text is only 50 tokens - dataset is empty!
train_loader = DataLoader(dataset, ...)  # Empty loader
for batch in train_loader:  # Never executes
    ...
```

**Problem:**
- Training loop does nothing
- No error, but model doesn't train
- Hard to debug

**Solution:**
```python
if len(dataset) == 0:
    raise ValueError("Dataset is empty! Text too short.")
```

### 10. Incorrect Position Embedding Indexing

**Mistake:**
```python
# Using batch indices instead of sequence positions
pos_ids = input_ids  # Wrong! These are token IDs, not positions
pos_emb = position_embedding(pos_ids)
```

**Problem:**
- Position embeddings should use `0, 1, 2, ..., seq_len-1`
- Not the token IDs themselves
- Model gets confused about positions

**Solution:**
```python
seq_len = input_ids.size(1)
pos_ids = torch.arange(seq_len, device=input_ids.device)
pos_emb = position_embedding(pos_ids)
```

## Probabilistic Challenges

### 1. Non-Deterministic Outputs

**Challenge:**
```python
# Same prompt, different outputs
output1 = generate_text(model, prompt, temperature=0.8)
output2 = generate_text(model, prompt, temperature=0.8)
# output1 != output2
```

**Why this happens:**
- Model samples from probability distribution
- Random sampling → different results each time
- This is **expected behavior**, not a bug

**When it's a problem:**
- Need reproducible outputs
- Testing/debugging
- Production systems requiring consistency

**Solutions:**
```python
# Set random seed
torch.manual_seed(42)
output = generate_text(model, prompt, ...)

# Use greedy sampling (temperature=0)
output = generate_text(model, prompt, temperature=0.0)
```

**Trade-offs:**
- Reproducible = less creative, more repetitive
- Random = more creative, less consistent

### 2. Evaluating Probabilistic Models

**Challenge:**
How do you measure quality of a probabilistic model?

**Problems:**
- No single "correct" output
- Multiple valid continuations
- Subjective quality (fluency, coherence, relevance)

**Common metrics:**
- **Perplexity**: Statistical measure, but doesn't capture quality
- **BLEU/ROUGE**: For specific tasks, but limited
- **Human evaluation**: Best but expensive

**Example:**
```python
# Model generates: "The cat sat on the mat"
# Reference: "The cat sat on the rug"
# Both are valid, but BLEU penalizes difference
```

**Best practice:**
- Use multiple metrics
- Combine quantitative (perplexity) with qualitative (human eval)
- Consider use case (creative vs. factual)

### 3. Temperature Sensitivity

**Challenge:**
Small temperature changes can dramatically affect output:

```python
output1 = generate_text(model, prompt, temperature=0.7)  # Coherent
output2 = generate_text(model, prompt, temperature=0.8)  # Slightly more random
output3 = generate_text(model, prompt, temperature=1.5)  # Nonsensical
```

**Why:**
- Temperature scales logits before softmax
- Small changes in logits → large changes in probabilities
- Exponential nature of softmax amplifies differences

**Implications:**
- Need to tune temperature carefully
- No universal "best" temperature
- Depends on model, task, and desired output style

**Best practice:**
- Start with temperature=0.8
- Adjust based on output quality
- Document what works for your use case

### 4. Mode Collapse

**Challenge:**
Model gets stuck generating same patterns:

```python
# Model keeps generating:
"The cat sat on the mat. The cat sat on the mat. The cat sat on..."
```

**Why:**
- Model learns to be "safe" by repeating
- High probability for certain sequences
- Sampling keeps picking same tokens

**Solutions:**
- Increase temperature (more randomness)
- Increase top-k (wider selection)
- Check training data (may have repetitive patterns)
- Adjust training (may be undertrained or overfitted)

### 5. Context Window Limitations

**Challenge:**
Model can only "see" limited context:

```python
# Long conversation
prompt = "User: Hello\nAssistant: Hi there!\nUser: How are you?\n..."
# Model only sees last N tokens (context_length)
# Earlier context is lost
```

**Implications:**
- Can't maintain long conversations
- Earlier information forgotten
- Need to manage context manually

**Workarounds:**
- Summarize earlier context
- Keep only recent tokens
- Use external memory systems (not in this codebase)

### 6. Sampling vs. Greedy Trade-off

**Challenge:**
Choosing between deterministic and random:

**Greedy (temperature=0):**
- ✅ Deterministic
- ✅ Fast
- ❌ Repetitive
- ❌ Misses creative solutions

**Sampling (temperature>0):**
- ✅ More creative
- ✅ Less repetitive
- ❌ Non-deterministic
- ❌ Can be less coherent

**Best practice:**
- Use sampling for creative tasks
- Use greedy for factual/technical tasks
- Tune temperature for your use case

### 7. Probability Distribution Interpretation

**Challenge:**
Understanding what logits/probabilities mean:

```python
logits = model(input_ids)  # Raw scores
probs = torch.softmax(logits, dim=-1)  # Probabilities

# What does prob[0, 0, 1234] = 0.05 mean?
# "5% chance of token 1234 at position 0"
# But model is probabilistic - it's an estimate, not certainty
```

**Implications:**
- Probabilities are model's beliefs, not ground truth
- High probability ≠ correct
- Low probability ≠ wrong
- Model can be confident and wrong

**Example:**
```python
# Model assigns 90% probability to wrong token
# Still samples it 90% of the time
# This is expected behavior for probabilistic models
```

### 8. Evaluation During Training

**Challenge:**
Loss decreases but output quality doesn't improve:

```python
# Training progress
Epoch 1: loss=4.5, output="asdfgh jklqwerty..."  # Nonsense
Epoch 5: loss=3.2, output="The cat sat on the mat"  # Better!
Epoch 10: loss=2.8, output="The cat sat on the mat. The cat sat..."  # Repetitive
```

**Why:**
- Loss measures statistical fit, not quality
- Model can minimize loss by being repetitive
- Need qualitative evaluation (read the outputs!)

**Best practice:**
- Monitor both loss and sample outputs
- Generate samples during training
- Don't rely solely on loss

### 9. Handling Special Tokens

**Challenge:**
Special tokens behave differently:

```python
tokenizer.encode("Hello<|endoftext|>world")
# How does model handle <|endoftext|>?
# Should generation stop when it sees this?
```

**Issues:**
- Model may generate special tokens
- Need to handle them in generation
- Stop conditions need special token awareness

**Solution:**
```python
# In generation loop
if next_token == endoftext_token_id:
    break  # Stop generation
```

### 10. Batch Generation Inconsistencies

**Challenge:**
Generating multiple sequences in parallel:

```python
# Batch of prompts
prompts = ["Once upon", "The cat", "In a land"]
# Generate all at once?
# Or one at a time?
```

**Issues:**
- Different prompts → different lengths
- Padding needed for batching
- Sampling is independent per sequence
- Hard to manage context windows

**Best practice:**
- Generate one at a time for simplicity
- Or handle padding/attention masks carefully
- Consider using `generate_text` separately for each prompt

## Debugging Strategies

### 1. Check Model State

```python
# Is model in eval mode?
print(model.training)  # Should be False for inference

# Are gradients enabled?
print(torch.is_grad_enabled())  # Should be False for inference
```

### 2. Verify Inputs

```python
# Check input shape and values
print(f"Input shape: {input_ids.shape}")
print(f"Input range: [{input_ids.min()}, {input_ids.max()}]")
print(f"Vocab size: {model.config.vocab_size}")
# Input values should be in [0, vocab_size)
```

### 3. Monitor Outputs

```python
# Check logits
logits = model(input_ids)
print(f"Logits shape: {logits.shape}")
print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

# Check probabilities
probs = torch.softmax(logits, dim=-1)
print(f"Max prob: {probs.max():.4f}")
print(f"Top-5 tokens: {torch.topk(probs, 5).indices}")
```

### 4. Reproducibility Checks

```python
# Set seeds
torch.manual_seed(42)
# Generate
output1 = generate_text(model, prompt, ...)

# Reset seed and generate again
torch.manual_seed(42)
output2 = generate_text(model, prompt, ...)

# Should be identical (if temperature > 0 and same PyTorch version)
assert output1 == output2
```

### 5. Context Window Verification

```python
# Check if input fits in context
seq_len = len(input_ids)
context_len = model.config.context_length
if seq_len > context_len:
    print(f"Warning: Sequence length {seq_len} > context {context_len}")
    print("Will be truncated")
```

## Best Practices Summary

1. **Always use `model.eval()` for inference**
2. **Save and load config with checkpoints**
3. **Handle context window limits**
4. **Verify vocabulary size matches tokenizer**
5. **Check device placement (CPU/GPU)**
6. **Monitor both loss and sample outputs**
7. **Understand probabilistic nature of outputs**
8. **Tune temperature and top-k for your use case**
9. **Generate samples during training for qualitative feedback**
10. **Set random seeds for reproducibility when needed**

## Next Steps

- **Quick reference**: See [Quick Reference](QUICK_REFERENCE.md)
- **Model implementation**: See [Model Implementation](01-model-usage-guide.md)
- **Training details**: See [Training Implementation](02-training-implementation.md)
