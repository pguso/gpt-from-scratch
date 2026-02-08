# Loading Pretrained Weights

This guide explains how to load pretrained model weights. There are two main scenarios:
1. **Loading your own checkpoints** - Models you've trained with this codebase
2. **Loading OpenAI GPT-2 weights** - Pretrained weights from OpenAI via Hugging Face

Both approaches are covered in this guide.

## Overview

When you train a model, PyTorch saves checkpoints containing:
- **Model weights**: All learned parameters (weights and biases)
- **Model configuration**: Architecture parameters (layer sizes, number of heads, etc.)
- **Training state**: Optimizer state, epoch number, loss values
- **Metadata**: Training metrics, timestamps, etc.

Loading pretrained weights involves:
1. Loading the checkpoint file from disk
2. Extracting the model configuration
3. Creating a model instance with matching architecture
4. Loading the weights into the model
5. Setting the model to the appropriate mode (eval/train)

## Checkpoint Structure

A checkpoint is a Python dictionary saved using `torch.save()`. Here's what it typically contains:

```python
checkpoint = {
    'model_state_dict': {...},      # Model weights and biases
    'optimizer_state_dict': {...},  # Optimizer state (for resuming training)
    'config': {...},                # Model configuration dictionary
    'epoch': 10,                    # Training epoch
    'train_loss': 2.345,            # Training loss
    'val_loss': 2.456,              # Validation loss
}
```

### Model State Dict

The `model_state_dict` contains all trainable parameters:

```python
{
    'token_embedding.weight': tensor([...]),      # Token embedding weights
    'position_embedding.weight': tensor([...]),   # Position embedding weights
    'transformer_blocks.0.attention.qkv.weight': tensor([...]),  # Attention weights
    'transformer_blocks.0.norm1.weight': tensor([...]),          # Layer norm weights
    # ... and so on for all layers
}
```

### Config Dictionary

The `config` dictionary contains architecture parameters:

```python
{
    'vocab_size': 50257,
    'context_length': 128,
    'embedding_dimension': 256,
    'number_of_heads': 4,
    'number_of_layers': 4,
    'dropout_rate': 0.1,
    'use_attention_bias': False
}
```

## Step-by-Step Loading Process

### Step 1: Load the Checkpoint File

```python
import torch

checkpoint_path = 'checkpoints/best_model.pt'
device = 'cpu'  # or 'cuda', 'mps'

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
```

**Key parameters:**
- `map_location`: Specifies where to load tensors (CPU, CUDA, etc.)
- `weights_only=True`: Security feature that only loads tensors (prevents code execution)

**Why `map_location`?**
- If checkpoint was saved on GPU but you're loading on CPU, tensors need to be moved
- If checkpoint was saved on CPU but you want GPU, tensors need to be moved
- `map_location` handles this automatically

### Step 2: Extract Model Configuration

```python
from src.config import ModelConfig

if 'config' in checkpoint:
    checkpoint_config = checkpoint['config']
    
    # Handle backward compatibility if needed
    if 'use_attention_bias' not in checkpoint_config:
        if 'query_key_value_bias' in checkpoint_config:
            checkpoint_config['use_attention_bias'] = checkpoint_config.pop('query_key_value_bias')
    
    config = ModelConfig(**checkpoint_config)
else:
    # Fallback to defaults (not recommended)
    config = ModelConfig()
```

**Why recreate config?**
- Model architecture must match saved weights exactly
- Wrong architecture → shape mismatches → loading fails
- Config ensures model is created with correct dimensions

### Step 3: Create Model Instance

```python
from src.model.gpt import GPTModel

model = GPTModel(config)
```

**Important:** The model architecture must match the checkpoint exactly:
- Same number of layers
- Same embedding dimensions
- Same number of attention heads
- Same vocabulary size
- Same context length

### Step 4: Load the Weights

```python
model.load_state_dict(checkpoint['model_state_dict'])
```

**What happens:**
- PyTorch matches parameter names from checkpoint to model
- Copies weights into model's parameters
- Raises error if shapes don't match

**Common errors:**
- `RuntimeError: Error(s) in loading state_dict`: Shape mismatch
- `KeyError`: Missing or extra keys (architecture mismatch)

### Step 5: Set Model Mode and Device

```python
model = model.to(device)  # Move to GPU/CPU
model.eval()  # Set to evaluation mode
```

**Why `model.eval()`?**
- Disables dropout (uses all neurons)
- Freezes batch normalization statistics
- Required for consistent inference

**When to use `model.train()`?**
- When continuing training
- When fine-tuning
- When you want dropout active

## Complete Example

Here's a complete example combining all steps:

```python
import torch
from src.model.gpt import GPTModel
from src.config import ModelConfig

def load_pretrained_model(checkpoint_path, device='cpu'):
    """Load a pretrained model from checkpoint."""
    
    # Step 1: Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Step 2: Extract config
    if 'config' in checkpoint:
        config = ModelConfig(**checkpoint['config'])
    else:
        raise ValueError("Config not found in checkpoint")
    
    # Step 3: Create model
    model = GPTModel(config)
    
    # Step 4: Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Step 5: Set mode and device
    model = model.to(device)
    model.eval()
    
    return model, checkpoint

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, checkpoint = load_pretrained_model('checkpoints/best_model.pt', device)

# Check checkpoint info
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation loss: {checkpoint['val_loss']:.4f}")

# Use model for inference
with torch.no_grad():
    input_ids = torch.tensor([[15496, 995]])  # "Hello world"
    logits = model(input_ids)
```

## Using the Example Script

The project includes a complete example script:

```bash
python examples/load_pretrained_weights.py --checkpoint checkpoints/best_model.pt
```

**With text generation:**
```bash
python examples/load_pretrained_weights.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Once upon a time" \
    --length 100 \
    --device cuda
```

## Loading for Different Purposes

### For Inference Only

```python
model, checkpoint = load_pretrained_model('checkpoints/best_model.pt', device='cpu')
model.eval()  # Evaluation mode

# Generate text
with torch.no_grad():
    logits = model(input_ids)
```

**Key points:**
- Use `model.eval()` to disable dropout
- Use `torch.no_grad()` to disable gradient computation (saves memory)
- Don't need optimizer state

### For Continuing Training

```python
import torch.optim as optim

# Load model
checkpoint = torch.load('checkpoints/checkpoint_epoch_10.pt', map_location=device)
config = ModelConfig(**checkpoint['config'])
model = GPTModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.train()  # Training mode (enables dropout)

# Load optimizer state
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Resume from epoch
start_epoch = checkpoint['epoch'] + 1
```

**Key points:**
- Use `model.train()` to enable dropout
- Load optimizer state to resume training exactly
- Resume from next epoch

### For Fine-Tuning

```python
# Load pretrained model
model, checkpoint = load_pretrained_model('checkpoints/best_model.pt', device)

# Modify for fine-tuning
model.train()  # Enable dropout
# Optionally freeze some layers
for param in model.token_embedding.parameters():
    param.requires_grad = False

# Create new optimizer (don't load old optimizer state)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)  # Lower learning rate
```

**Key points:**
- Start with pretrained weights
- Use lower learning rate
- Optionally freeze some layers
- Don't load optimizer state (start fresh)

## Common Issues and Solutions

### Issue: Shape Mismatch Error

**Error:**
```
RuntimeError: Error(s) in loading state_dict for GPTModel:
    size mismatch for token_embedding.weight: copying a param with shape [50257, 256] 
    from checkpoint, the shape in current model is [50257, 512].
```

**Cause:** Model architecture doesn't match checkpoint.

**Solution:**
- Use config from checkpoint (don't create new config)
- Ensure all architecture parameters match

```python
# Wrong: Creating config manually
config = ModelConfig(embedding_dimension=512)  # Doesn't match checkpoint!

# Right: Using checkpoint config
config = ModelConfig(**checkpoint['config'])
```

### Issue: Missing Keys

**Error:**
```
KeyError: 'transformer_blocks.5.attention.qkv.weight'
```

**Cause:** Model has different number of layers than checkpoint.

**Solution:**
- Check `number_of_layers` in config
- Ensure model architecture matches exactly

### Issue: Extra Keys

**Warning:**
```
Unexpected key(s) in state_dict: "extra_layer.weight"
```

**Cause:** Checkpoint has parameters not in current model.

**Solution:**
- Usually safe to ignore if using `strict=False`
- Or ensure model architecture matches checkpoint

```python
# Ignore extra keys
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### Issue: Device Mismatch

**Error:**
```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) 
should be the same
```

**Cause:** Model and input tensors on different devices.

**Solution:**
- Move model to same device as inputs
- Or use `map_location` when loading

```python
# Load to correct device
checkpoint = torch.load(path, map_location='cuda')
model = model.to('cuda')
input_ids = input_ids.to('cuda')
```

### Issue: Model in Wrong Mode

**Symptom:** Inconsistent results between runs.

**Cause:** Forgot to call `model.eval()`.

**Solution:**
```python
model.eval()  # Always call before inference
```

## Best Practices

### 1. Always Use Checkpoint Config

```python
# ✅ Good: Use config from checkpoint
config = ModelConfig(**checkpoint['config'])

# ❌ Bad: Create config manually (may not match)
config = ModelConfig(embedding_dimension=256)
```

### 2. Verify Checkpoint Before Loading

```python
# Check if checkpoint exists and is valid
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# Check required keys
required_keys = ['model_state_dict', 'config']
for key in required_keys:
    if key not in checkpoint:
        raise ValueError(f"Missing key in checkpoint: {key}")
```

### 3. Handle Device Placement Correctly

```python
# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load to device
checkpoint = torch.load(path, map_location=device)
model = model.to(device)
```

### 4. Set Appropriate Mode

```python
# For inference
model.eval()
with torch.no_grad():
    output = model(input_ids)

# For training
model.train()
output = model(input_ids)
```

### 5. Check Model After Loading

```python
# Verify model loaded correctly
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
test_input = torch.randint(0, config.vocab_size, (1, 10))
with torch.no_grad():
    test_output = model(test_input)
print(f"Output shape: {test_output.shape}")  # Should match expected shape
```

## Checkpoint Information

After loading, you can access checkpoint metadata:

```python
checkpoint = torch.load('checkpoints/best_model.pt')

# Training information
epoch = checkpoint.get('epoch', 'Unknown')
train_loss = checkpoint.get('train_loss', None)
val_loss = checkpoint.get('val_loss', None)

# Model information
config = checkpoint.get('config', {})
print(f"Model trained for {epoch} epochs")
print(f"Validation loss: {val_loss:.4f}")
print(f"Architecture: {config['number_of_layers']} layers, {config['embedding_dimension']} dims")
```

## Loading OpenAI GPT-2 Pretrained Weights

This codebase also supports loading OpenAI's pretrained GPT-2 weights from Hugging Face. This allows you to use models that were trained on massive datasets without training from scratch.

### Overview

OpenAI released several GPT-2 model sizes:
- **GPT-2 Small**: 117M parameters (12 layers, 768 dims, 12 heads)
- **GPT-2 Medium**: 345M parameters (24 layers, 1024 dims, 16 heads)
- **GPT-2 Large**: 762M parameters (36 layers, 1280 dims, 20 heads)
- **GPT-2 XL**: 1.5B parameters (48 layers, 1600 dims, 25 heads)

These weights are available through Hugging Face's `transformers` library.

### Installation

First, install the `transformers` library:

```bash
pip install transformers
```

### Using the Example Script

The easiest way to load OpenAI weights is using the provided example script:

```bash
# Load GPT-2 small
python examples/load_openai_weights.py --model-size small

# Load GPT-2 medium and generate text
python examples/load_openai_weights.py --model-size medium --prompt "The future of AI"

# Load on GPU
python examples/load_openai_weights.py --model-size small --device cuda
```

**What the script does:**
1. Downloads GPT-2 weights from Hugging Face (first run only)
2. Converts weight names from Hugging Face format to this codebase's format
3. Splits combined QKV projections into separate Q, K, V matrices
4. Maps layer normalization parameters correctly
5. Loads the converted weights into the model

### Weight Conversion Process

Hugging Face and this codebase use different parameter naming conventions. The conversion handles:

**Embeddings:**
- `transformer.wte.weight` → `token_embedding.weight`
- `transformer.wpe.weight` → `position_embedding.weight`

**Transformer Blocks:**
- `transformer.h.{i}.ln_1.weight/bias` → `transformer_blocks.{i}.norm1.scale/shift`
- `transformer.h.{i}.attn.c_attn.weight` → Split into `W_query`, `W_key`, `W_value`
- `transformer.h.{i}.attn.c_proj.weight` → `transformer_blocks.{i}.attention.out_proj.weight`
- `transformer.h.{i}.ln_2.weight/bias` → `transformer_blocks.{i}.norm2.scale/shift`
- `transformer.h.{i}.mlp.c_fc.weight` → `transformer_blocks.{i}.feed_forward.net.0.weight`
- `transformer.h.{i}.mlp.c_proj.weight` → `transformer_blocks.{i}.feed_forward.net.2.weight`

**Output Layer:**
- `transformer.ln_f.weight/bias` → `final_norm.scale/shift`
- `lm_head.weight` → `lm_head.weight` (same)

**Key Conversion: QKV Projection**

Hugging Face stores Query, Key, and Value projections as a single combined matrix:
```python
# Hugging Face: [3 * embedding_dim, embedding_dim]
c_attn.weight  # Combined QKV
```

This codebase uses separate matrices:
```python
# This codebase: Three separate [embedding_dim, embedding_dim] matrices
W_query.weight
W_key.weight
W_value.weight
```

The conversion splits the combined matrix into three parts.

### Programmatic Loading

You can also load OpenAI weights programmatically:

```python
from transformers import GPT2LMHeadModel
from src.model.gpt import GPTModel
from src.config import ModelConfig

# Load Hugging Face model
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')  # or 'gpt2-medium', etc.
hf_state_dict = hf_model.state_dict()

# Create config matching GPT-2 small
config = ModelConfig(
    vocab_size=50257,
    context_length=1024,
    embedding_dimension=768,
    number_of_heads=12,
    number_of_layers=12,
    dropout_rate=0.1,
    use_attention_bias=False
)

# Create model
model = GPTModel(config)

# Convert and load weights (see load_openai_weights.py for conversion function)
custom_state_dict = convert_hf_to_custom(hf_state_dict, config)
model.load_state_dict(custom_state_dict, strict=False)
model.eval()
```

### Model Configurations

The example script includes pre-configured settings for all GPT-2 sizes:

```python
GPT2_CONFIGS = {
    'small': {
        'vocab_size': 50257,
        'context_length': 1024,
        'embedding_dimension': 768,
        'number_of_heads': 12,
        'number_of_layers': 12,
        'dropout_rate': 0.1,
        'use_attention_bias': False
    },
    # ... medium, large, xl
}
```

### First-Time Download

On first run, the script downloads model weights from Hugging Face:
- **GPT-2 Small**: ~500MB
- **GPT-2 Medium**: ~1.5GB
- **GPT-2 Large**: ~3GB
- **GPT-2 XL**: ~6GB

Weights are cached in `~/.cache/huggingface/` for future use.

### Using Loaded OpenAI Models

Once loaded, OpenAI models work exactly like models trained with this codebase:

```python
from src.generation.generate import generate_text
import tiktoken

# Load model (see above)
model, config = load_openai_gpt2('small', device='cpu')

# Generate text
tokenizer = tiktoken.get_encoding("gpt2")
input_ids = tokenizer.encode("Hello, world")
output_ids = generate_text(
    model,
    input_ids,
    maximum_new_tokens=100,
    temperature=0.8,
    top_k_tokens=50
)
output_text = tokenizer.decode(output_ids)
print(output_text)
```

### Differences from Hugging Face

This codebase's architecture matches GPT-2, but there are some implementation differences:

1. **Layer Normalization**: This codebase uses custom `LayerNorm` with `scale` and `shift` parameters, while Hugging Face uses `weight` and `bias`
2. **Attention Bias**: GPT-2 doesn't use bias in attention projections (this codebase matches this)
3. **QKV Projection**: This codebase uses separate Q, K, V matrices instead of combined

The conversion function handles all these differences automatically.

### Fine-Tuning OpenAI Models

You can fine-tune OpenAI models loaded this way:

```python
# Load OpenAI weights
model, config = load_openai_gpt2('small', device='cuda')

# Set to training mode
model.train()

# Create optimizer (use lower learning rate for fine-tuning)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Fine-tune on your data...
```

### Troubleshooting

**Issue: `transformers` library not found**
```bash
pip install transformers
```

**Issue: Out of memory when loading large models**
- Use smaller model size (`--model-size small`)
- Load on CPU instead of GPU
- Use model quantization (advanced)

**Issue: Shape mismatches during conversion**
- Ensure you're using the correct model size configuration
- Check that `embedding_dimension % number_of_heads == 0`

## Summary

Loading pretrained weights involves:
1. ✅ Load checkpoint with `torch.load()` (for your own checkpoints)
2. ✅ Or download from Hugging Face and convert (for OpenAI weights)
3. ✅ Extract and use config from checkpoint or use GPT-2 configs
4. ✅ Create model with matching architecture
5. ✅ Load weights with `load_state_dict()`
6. ✅ Set model mode (`eval()` for inference, `train()` for training)
7. ✅ Move to correct device

**Key takeaways:**
- Always use the config from the checkpoint to ensure architecture matches
- For OpenAI weights, use the provided conversion function
- OpenAI models are ready for inference immediately after loading

## Related Documentation

- [Using the Model](03-using-the-model.md) - How to use loaded models for text generation
- [Training Implementation](02-training-implementation.md) - How checkpoints are created during training
- [Quick Reference](QUICK_REFERENCE.md) - Code snippets for loading checkpoints
