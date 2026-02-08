# Advanced Topics

Optimizations, extensions, and advanced techniques.

## Introduction

This document covers advanced topics for GPT models: optimizations for training and inference, extensions to the architecture, fine-tuning techniques, and cutting-edge research directions.

## 1. Training Optimizations

### Mixed Precision Training

Use FP16 (half precision) to speed up training and reduce memory:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- **2x faster**: On modern GPUs (V100, A100)
- **Half memory**: Can use larger batch sizes
- **Minimal accuracy loss**: Automatic loss scaling

### Gradient Accumulation

Simulate larger batch sizes when memory is limited:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Use cases**:
- Limited GPU memory
- Want larger effective batch size
- Distributed training

### Learning Rate Scheduling

#### Warmup

Gradually increase learning rate at start:

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
```

#### Cosine Annealing

Smoothly decrease learning rate:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```

#### OneCycleLR

Increase then decrease (PyTorch):

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=3e-4,
    total_steps=num_epochs * steps_per_epoch
)
```

### Compile Model (PyTorch 2.0+)

Speed up inference and training:

```python
model = torch.compile(model)
```

**Benefits**:
- **Faster**: 20-30% speedup
- **Easy**: Just wrap model
- **Compatible**: Works with most code

## 2. Inference Optimizations

### KV Caching

Cache key-value pairs to avoid recomputation:

```python
class GPTModelWithCache(nn.Module):
    def forward(self, input_ids, past_key_values=None):
        if past_key_values is None:
            # First call: compute all
            past_key_values = []
            for block in self.transformer_blocks:
                # Compute and cache KV
                past_key_values.append(kv_cache)
        else:
            # Subsequent calls: use cache
            # Only compute for new tokens
            pass
```

**Benefits**:
- **Much faster**: Only compute for new tokens
- **Lower latency**: Critical for real-time applications

### Quantization

Reduce model size and speed up inference:

#### Post-Training Quantization

```python
import torch.quantization as quantization

model_fp32 = GPTModel(config)
model_int8 = quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
```

**Benefits**:
- **4x smaller**: INT8 vs FP32
- **Faster inference**: On supported hardware
- **Easy**: No retraining needed

#### Quantization-Aware Training

Train with quantization in mind for better accuracy.

### Pruning

Remove unimportant weights:

```python
import torch.nn.utils.prune as prune

# Prune 20% of weights
prune.l1_unstructured(module, name="weight", amount=0.2)
```

**Benefits**:
- **Smaller model**: Fewer parameters
- **Faster inference**: Less computation
- **Maintains accuracy**: If done carefully

## 3. Architecture Extensions

### Rotary Position Embeddings (RoPE)

Alternative to learned position embeddings:

- **Rotary**: Rotate query/key vectors based on position
- **Relative**: Better generalization to longer sequences
- **Used in**: LLaMA, PaLM

### Flash Attention

Memory-efficient attention implementation:

- **Block-wise**: Process attention in blocks
- **Memory efficient**: O(n) instead of O(n²)
- **Faster**: Optimized CUDA kernels

### Sparse Attention

Only attend to subset of tokens:

- **Local**: Attend to nearby tokens
- **Strided**: Attend to every k-th token
- **Random**: Random subset
- **Longformer, BigBird**: Examples

### Mixture of Experts (MoE)

Multiple expert networks, route tokens to experts:

- **Scale**: More parameters without more computation
- **Routing**: Learn which expert to use
- **Used in**: Switch Transformer, GShard

## 4. Fine-Tuning Techniques

### Full Fine-Tuning

Update all parameters:

```python
# Standard training loop
for param in model.parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=1e-5)
```

**Use when**:
- Have enough data
- Want maximum performance
- Can afford compute

### Parameter-Efficient Fine-Tuning

#### LoRA (Low-Rank Adaptation)

Add low-rank matrices instead of updating all weights:

```python
# Instead of updating W (d x d)
# Update A (d x r) and B (r x d) where r << d
# W' = W + BA
```

**Benefits**:
- **Fewer parameters**: Only train A and B
- **Faster**: Less computation
- **Modular**: Easy to switch tasks

#### Adapters

Add small modules between layers:

```python
class Adapter(nn.Module):
    def __init__(self, dim):
        self.down = nn.Linear(dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, dim)
    
    def forward(self, x):
        return x + self.up(self.down(x))
```

#### Prompt Tuning

Learn soft prompts (continuous embeddings):

- **No model changes**: Just learn prompt embeddings
- **Very efficient**: Only few parameters
- **Task-specific**: Different prompts for different tasks

### Instruction Tuning

Fine-tune on instruction-following data:

```python
# Format: "Instruction: ...\nResponse: ..."
instruction_data = [
    "Instruction: Translate to French\nResponse: Bonjour",
    ...
]
```

**Benefits**:
- **Better following**: Follows instructions
- **Few-shot**: Can do new tasks
- **ChatGPT-like**: More conversational

## 5. Attention Analysis and Visualization

Understanding what your model is "looking at" is crucial for debugging and improving performance. This repository includes a comprehensive attention analysis tool.

### Using the Attention Analysis Script

The `examples/analyze_attention.py` script extracts and visualizes attention patterns from your trained model:

```bash
# Analyze attention in a trained model
python examples/analyze_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --text "The cat sat on the mat. It was fluffy."

# Visualize specific layer and head
python examples/analyze_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --layer 2 \
    --head 0 \
    --show-plots

# Analyze all layers
python examples/analyze_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --text "Once upon a time there was a little girl"
```

### How It Works

The script uses PyTorch hooks to capture attention weights during the forward pass:

```python
class AttentionHook:
    """Hook to capture attention weights from MultiHeadAttention module."""
    
    def __call__(self, module, input, output):
        # Manually compute attention weights
        queries = module.W_query(input[0])
        keys = module.W_key(input[0])
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(-2, -1)
        
        # Apply mask and softmax
        attn_weights = torch.softmax(attn_scores / scaling_factor, dim=-1)
        
        # Store for visualization
        self.attention_weights.append(attn_weights)
```

### Visualizations

**1. Single Head Heatmap**:
- Shows attention weights for one head in one layer
- Rows = query tokens (attending from)
- Columns = key tokens (attended to)
- Intensity = attention weight (0-1)

**2. All Heads Grid**:
- Shows all heads in a layer side-by-side
- Reveals different attention patterns per head
- Some heads focus on syntax, others on semantics

**3. Attention Flow**:
- Tracks attention patterns across layers
- Shows how self-attention changes
- Reveals information flow through the network

### What to Look For

**Good Patterns**:
- Strong attention to relevant previous tokens
- Pronouns attending to their referents ("it" → "cat")
- Subject-verb relationships
- Long-range dependencies

**Problem Patterns**:
- Uniform attention (model not learning)
- Only self-attention (not using context)
- Random patterns (untrained model)

### Example Analysis

```python
# The script automatically analyzes patterns:
Layer 0:
  'cat' attends most to:
    1. 'The' (weight: 0.45)
    2. 'cat' (weight: 0.35)  # self-attention
    3. 'sat' (weight: 0.20)

  'It' attends most to:
    1. 'cat' (weight: 0.60)  # coreference!
    2. 'mat' (weight: 0.25)
    3. 'It' (weight: 0.15)
```

### Integration with Training

You can analyze attention at different training stages:

```bash
# Analyze untrained model (random patterns)
python examples/analyze_attention.py --text "The cat sat"

# Analyze after 5 epochs
python examples/analyze_attention.py \
    --checkpoint checkpoints/checkpoint_epoch_5.pt \
    --text "The cat sat"

# Analyze best model (learned patterns)
python examples/analyze_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --text "The cat sat"
```

This helps you understand:
- What the model learns at different stages
- Which layers capture which patterns
- How attention evolves during training

## 6. Advanced Generation Techniques

### Using the Generation Script

This repository includes a complete text generation script (`examples/generate_text.py`) that loads trained checkpoints and generates text:

```bash
# Generate with default settings
python examples/generate_text.py --checkpoint checkpoints/best_model.pt

# Customize generation parameters
python examples/generate_text.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Once upon a time" \
    --length 150 \
    --temperature 0.7 \
    --top-k 50
```

**Features**:
- Automatically loads model configuration from checkpoint
- Supports temperature and top-k sampling
- Shows training information (epochs, validation loss)
- Handles device placement automatically

**Checkpoint Loading**:
```python
# The script automatically:
# 1. Loads checkpoint
checkpoint = torch.load(checkpoint_path)

# 2. Recreates config
config = GPTConfig(**checkpoint['config'])

# 3. Creates and loads model
model = GPTModel(config)
model.load_state_dict(checkpoint['model_state_dict'])

# 4. Generates text with proper sampling
output = generate_text(model, input_ids, temperature=0.8, top_k=50)
```

### Top-k Sampling

Only sample from top k tokens:

```python
def top_k_sampling(logits, k=50):
    top_k_values, top_k_indices = torch.topk(logits, k)
    # Set others to -inf
    logits[logits < top_k_values[-1]] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)
```

### Top-p (Nucleus) Sampling

Sample from tokens whose cumulative probability reaches p:

```python
def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability > p
    sorted_indices_to_remove = cumprobs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    return torch.multinomial(torch.softmax(logits, dim=-1), 1)
```

### Temperature Scaling

Control randomness:

```python
def temperature_sampling(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, 1)
```

- **Low temperature (< 1.0)**: More deterministic, less creative
- **High temperature (> 1.0)**: More random, more creative
- **Temperature = 1.0**: Standard sampling

### Beam Search

Keep multiple candidate sequences:

```python
def beam_search(model, prompt, beam_width=5, max_length=100):
    beams = [(prompt, 0.0)]  # (sequence, score)
    
    for _ in range(max_length):
        new_beams = []
        for seq, score in beams:
            logits = model(seq)
            top_k = torch.topk(logits[-1], beam_width)
            for token, log_prob in zip(top_k.indices, top_k.values):
                new_beams.append((seq + [token], score + log_prob))
        
        beams = sorted(new_beams, key=lambda x: x[1])[-beam_width:]
    
    return beams[0][0]
```

**Use when**:
- Need high-quality output
- Can afford slower generation
- Not for creative tasks

## 7. Evaluation Metrics

### Perplexity

Measure of model's uncertainty:

```python
perplexity = torch.exp(loss)
```

- **Lower is better**: More confident predictions
- **Interpretable**: Average branching factor

### BLEU Score

For translation tasks:

```python
from nltk.translate.bleu_score import sentence_bleu

bleu = sentence_bleu(reference, candidate)
```

### ROUGE Score

For summarization:

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score(reference, candidate)
```

### Human Evaluation

Best but expensive:
- **Fluency**: Does it read naturally?
- **Coherence**: Does it make sense?
- **Relevance**: Does it answer the question?

## 8. Distributed Training

### Data Parallelism

Replicate model on multiple GPUs:

```python
model = nn.DataParallel(model)
```

### Model Parallelism

Split model across GPUs:

```python
# Layer 0-5 on GPU 0, Layer 6-11 on GPU 1
```

### Pipeline Parallelism

Process different batches on different GPUs:

- **GPipe**: Pipeline parallel training
- **Used in**: Very large models

### DeepSpeed

Microsoft's library for distributed training:

- **ZeRO**: Zero redundancy optimizer
- **Gradient checkpointing**: Trade compute for memory
- **Mixed precision**: Automatic FP16

## 9. Research Directions

### Scaling Laws

How performance scales with:
- **Model size**: More parameters → better (up to a point)
- **Data size**: More data → better
- **Compute**: More compute → better

**Key insight**: Predictable scaling relationships.

### Emergent Ababilities

Capabilities that appear at scale:
- **In-context learning**: Learn from examples in prompt
- **Chain-of-thought**: Step-by-step reasoning
- **Code generation**: Write code from description

### Alignment

Making models helpful, harmless, honest:
- **RLHF**: Reinforcement Learning from Human Feedback
- **Constitutional AI**: Self-improvement with principles
- **Red teaming**: Finding failure modes

### Multimodal Models

Beyond text:
- **Vision + Language**: GPT-4V, LLaVA
- **Audio**: Whisper, AudioLM
- **Code**: Codex, GitHub Copilot

## 10. Practical Considerations

### Memory Management

- **Gradient checkpointing**: Recompute activations
- **Offloading**: Move to CPU when not needed
- **Mixed precision**: Use FP16

### Debugging

- **Gradient checking**: Verify gradients
- **Activation statistics**: Monitor activations
- **Loss curves**: Watch for anomalies

### Production Deployment

- **Model serving**: Use frameworks like vLLM, TensorRT
- **Batching**: Process multiple requests together
- **Caching**: Cache common prompts
- **Monitoring**: Track latency, throughput, errors

## 11. Resources

### Papers

- "Attention Is All You Need" - Original transformer
- "Language Models are Unsupervised Multitask Learners" - GPT-2
- "Training language models to follow instructions" - InstructGPT
- "Scaling Laws for Neural Language Models" - Scaling laws

### Libraries

- **Transformers**: Hugging Face library
- **vLLM**: Fast inference
- **DeepSpeed**: Distributed training
- **Flash Attention**: Efficient attention

### Communities

- **Hugging Face**: Models, datasets, discussions
- **r/MachineLearning**: Research discussions
- **Papers with Code**: Implementations

## Summary

Advanced topics covered:

1. **Training**: Mixed precision, gradient accumulation, scheduling
2. **Inference**: KV caching, quantization, pruning
3. **Architecture**: RoPE, Flash Attention, MoE
4. **Fine-tuning**: LoRA, adapters, prompt tuning
5. **Attention Analysis**: Visualization, pattern analysis, debugging
6. **Generation**: Top-k, top-p, temperature, beam search
7. **Evaluation**: Perplexity, BLEU, ROUGE
8. **Distributed**: Data/model/pipeline parallelism
9. **Research**: Scaling, emergence, alignment

Key takeaways:
- **Optimize for your use case**: Training vs. inference
- **Start simple**: Add complexity as needed
- **Measure everything**: Metrics guide decisions
- **Stay updated**: Field moves fast

---

Previous: [Training Walkthrough](04-training-walkthrough.md)
