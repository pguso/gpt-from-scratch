# Training Walkthrough: train_tiny_stories.py Explained

This document provides a comprehensive walkthrough of the `train_tiny_stories.py` script, which demonstrates how to train a GPT model from scratch on the TinyStories dataset.

> **Note:** This document focuses on the specific implementation details of the training script. For general training concepts, see [Training Pipeline](04-training-pipeline.md). For optimization techniques and advanced training topics, see [Advanced Topics](05-advanced-topics.md).

## Overview

The script implements a complete training pipeline for a GPT model, including data loading, model initialization, training loop, validation, and text generation. TinyStories is a dataset of simple, short stories written at a basic reading level, making it ideal for training small language models.

## Code Structure

### 1. Imports and Dependencies

```python
import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import tiktoken

from src.model.gpt import GPTModel
from src.config import GPTConfig
from src.data.dataset import GPTDataset
from src.training.trainer import GPTTrainer
from src.generation.generate import generate_text
```

**Key Components:**
- **tiktoken**: OpenAI's tokenizer library, using the GPT-2 tokenizer (BPE encoding)
- **GPTModel**: The transformer architecture implementation
- **GPTDataset**: Handles text preprocessing and sequence creation
- **GPTTrainer**: Encapsulates training and validation logic
- **generate_text**: Function for autoregressive text generation

**Insight**: Using `tiktoken` ensures compatibility with GPT-2's tokenization scheme, which is widely used and well-optimized. The modular design separates concerns (model, data, training, generation) for better maintainability.

---

### 2. Sample Data Generation (`generate_sample_stories()`)

```python
def generate_sample_stories():
    """Generate simple sample stories for testing."""
    stories = [...]
    return "<|endoftext|>".join(stories)
```

**Purpose**: Provides fallback data when the TinyStories dataset cannot be downloaded.

**Key Design Decisions:**
- Uses `<|endoftext|>` as a separator token, which is the standard GPT-2 end-of-text marker
- Stories are simple and follow a consistent structure (subject-verb-object patterns)
- Acts as a safety net for testing without internet access

**Insight**: This demonstrates defensive programming—the script gracefully degrades when external dependencies fail, allowing development and testing to continue.

---

### 3. Dataset Download (`download_tinystories_data()`)

```python
def download_tinystories_data(data_dir="data", max_samples=None):
    """
    Download TinyStories dataset from Hugging Face.
    
    Args:
        data_dir: Directory to save the data
        max_samples: Maximum number of samples to download (None for all)
    
    Returns:
        Combined text string from all stories
    """
```

**Function Flow:**
1. **Import Check**: Attempts to import `datasets` library
2. **Fallback Mechanism**: If import fails, falls back to sample data
3. **Dataset Loading**: Uses Hugging Face's `load_dataset` API
4. **Text Combination**: Joins all stories with `<|endoftext|>` separators
5. **Persistence**: Saves combined text to disk for future use

**Key Implementation Details:**

```python
# Limit samples if specified
if max_samples:
    dataset = dataset.select(range(min(max_samples, len(dataset))))

# Combine all stories into a single text
texts = [item["text"] for item in dataset]
combined_text = "<|endoftext|>".join(texts)
```

**Insights:**
- **Memory Efficiency**: The `max_samples` parameter allows limiting dataset size for faster iteration during development
- **Text Concatenation**: Joining with `<|endoftext|>` teaches the model document boundaries, which is crucial for generation quality
- **Error Handling**: Multiple fallback layers ensure the script doesn't crash on network issues

**Why This Matters**: In production, you'd want more robust error handling, but for educational purposes, this demonstrates graceful degradation.

---

### 4. Data Loading (`load_tinystories_data()`)

```python
def load_tinystories_data(data_path="data/tinystories.txt"):
    """
    Load TinyStories data from a file.
    
    Args:
        data_path: Path to the data file
    
    Returns:
        Text string
    """
```

**Purpose**: Loads pre-downloaded data from disk, or triggers download if file doesn't exist.

**Design Pattern**: This follows the "lazy loading" pattern—only downloads when necessary, but prefers cached data for speed.

**Insight**: Separating download and load functions allows for flexible data management. You can pre-download data once and reuse it across multiple training runs.

---

### 5. Model Creation (`create_model()`)

```python
def create_model(config):
    """Create and initialize GPT model."""
    model = GPTModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    return model
```

**Key Features:**
- **Parameter Counting**: Calculates total and trainable parameters
- **Memory Estimation**: Estimates model size in MB (assuming FP32, 4 bytes per parameter)

**Insights:**
- **Parameter Counting Formula**: `numel()` returns the number of elements in a tensor. Summing across all parameters gives total model size.
- **Memory Calculation**: `total_params * 4 / 1024 / 1024` converts parameter count to MB (4 bytes per FP32 parameter)
- **Why This Matters**: Understanding model size helps with:
  - Memory planning (GPU/CPU RAM requirements)
  - Training time estimation
  - Model selection (larger models need more data and compute)

**Example**: A model with 10M parameters ≈ 40MB in FP32, but only 20MB in FP16 (half precision).

---

### 6. Main Training Function (`train()`)

This is the core of the script. Let's break it down section by section.

#### 6.1 Device Setup

```python
# Set device
if device == "cuda" and not torch.cuda.is_available():
    print("CUDA not available, using CPU")
    device = "cpu"
device = torch.device(device)
```

**Insight**: Automatic CPU fallback ensures the script works on machines without GPU support, making it more accessible.

#### 6.2 Data Loading and Tokenization

```python
# Load data
if data_path and os.path.exists(data_path):
    text = load_tinystories_data(data_path)
else:
    text = download_tinystories_data(max_samples=max_samples)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab
```

**Key Points:**
- **Tokenizer Choice**: GPT-2 tokenizer uses Byte Pair Encoding (BPE), which handles out-of-vocabulary words gracefully
- **Vocabulary Size**: GPT-2 has 50,257 tokens (includes special tokens like `<|endoftext|>`)

**Insight**: BPE is subword tokenization—it breaks words into smaller pieces, allowing the model to handle rare words and reduce vocabulary size while maintaining coverage.

#### 6.3 Dataset Creation with Adaptive Context Length

```python
# Check text length and adjust context_length if needed
token_ids_preview = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
text_length = len(token_ids_preview)
print(f"Text length: {text_length:,} tokens")

# If text is shorter than context_length, reduce context_length
if text_length < context_length:
    print(f"Warning: Text length ({text_length}) is shorter than context_length ({context_length})")
    print(f"Reducing context_length to {text_length // 2} to create training sequences")
    context_length = max(32, text_length // 2)
```

**Why This Matters:**
- **Adaptive Design**: Prevents errors when using small sample datasets
- **Sequence Creation**: Need at least `context_length` tokens to create one training sequence
- **Minimum Threshold**: `max(32, text_length // 2)` ensures we can still create sequences

**Insight**: This is a form of "defensive programming"—the script adapts to available data rather than failing. In production, you'd want to validate data size upfront.

#### 6.4 Dataset and DataLoader Creation

```python
full_dataset = GPTDataset(
    text=text,
    tokenizer=tokenizer,
    max_length=context_length,
    stride=max(1, context_length // 2)  # 50% overlap, at least 1
)

# Split into train/val
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False
)
```

**Key Concepts:**

1. **Stride Parameter**: `stride=max(1, context_length // 2)` creates overlapping sequences
   - With 50% overlap, each token appears in multiple training sequences
   - This increases effective data size and helps the model learn better
   - Example: With `context_length=128` and `stride=64`, sequences are: [0:128], [64:192], [128:256], ...

2. **Train/Val Split**: 90/10 split is standard for language modeling
   - Validation set monitors overfitting
   - Random split with fixed seed ensures reproducibility

3. **DataLoader Parameters**:
   - `shuffle=True`: Randomizes training order each epoch
   - `num_workers=0`: Avoids multiprocessing issues (can increase for faster loading)
   - `pin_memory=True`: Speeds up GPU transfer (only useful with CUDA)

**Insight**: The stride parameter is crucial for small datasets. Without overlap, you'd have far fewer training examples. With 50% overlap, you roughly double the number of sequences.

#### 6.5 Model Configuration

```python
config = GPTConfig(
    vocab_size=vocab_size,
    context_length=context_length,  # May have been adjusted above
    embedding_dimension=256,
    number_of_heads=4,
    number_of_layers=4,
    dropout_rate=0.1,
    query_key_value_bias=False
)
```

**Configuration Breakdown:**
- **vocab_size**: Must match tokenizer vocabulary (50,257 for GPT-2)
- **context_length**: Maximum sequence length the model can process
- **embedding_dimension**: Embedding dimension (256 is small, good for experimentation)
- **number_of_heads**: Number of attention heads (4 is standard for small models)
- **number_of_layers**: Transformer blocks (4 layers is shallow but fast to train)
- **dropout_rate**: Dropout probability (0.1 is conservative, helps prevent overfitting)
- **query_key_value_bias**: Whether to use bias in Q/K/V projections (GPT-2 uses False)

**Insight**: These hyperparameters create a "toy" model suitable for learning. Real GPT models use:
- Much larger `embedding_dimension` (768+ for GPT-2 small, 12,288 for GPT-3)
- More layers (12+ for GPT-2 small, 96 for GPT-3)
- Longer context (1024+ tokens)

#### 6.6 Optimizer Setup

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.1,
    betas=(0.9, 0.95)
)
```

**AdamW Explained:**
- **AdamW**: Improved version of Adam with decoupled weight decay
- **Learning Rate**: `3e-4` is a common starting point for transformers
- **Weight Decay (0.1)**: L2 regularization to prevent overfitting
- **Betas (0.9, 0.95)**: Momentum parameters
  - First beta (0.9): Controls gradient momentum
  - Second beta (0.95): Controls squared gradient momentum (used in Adam's variance estimate)

**Insight**: AdamW is the optimizer of choice for transformers. The weight decay is applied separately from gradient updates, which works better than traditional L2 regularization with Adam.

#### 6.7 Training Loop

```python
for epoch in range(1, epochs + 1):
    # Train
    train_loss = trainer.train_epoch()
    train_perplexity = torch.exp(torch.tensor(train_loss)).item()
    
    # Validate
    if epoch % eval_every == 0:
        val_loss = trainer.validate()
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # ... save checkpoint ...
    
    # Generate sample text
    if epoch % generate_every == 0:
        # ... generation code ...
    
    # Save checkpoint
    if epoch % save_every == 0:
        # ... save checkpoint ...
```

**Key Components:**

1. **Loss and Perplexity**:
   - Loss is typically cross-entropy (negative log-likelihood)
   - Perplexity = exp(loss) is more interpretable
   - Lower perplexity = better model (perplexity of 10 means model is as "surprised" as if it had to choose from 10 equally likely tokens)

2. **Best Model Tracking**: Saves model with lowest validation loss
   - Prevents overfitting (training loss can decrease while validation loss increases)
   - Validation loss is the true measure of generalization

3. **Text Generation During Training**:
   ```python
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
           max_new_tokens=30,
           temperature=0.8,
           top_k=50
       )
   ```
   
   **Generation Parameters**:
   - **temperature (0.8)**: Controls randomness (lower = more deterministic)
   - **top_k (50)**: Samples from top 50 most likely tokens (nucleus sampling alternative)
   - **max_new_tokens (30)**: Limits generation length

   **Insight**: Generating text during training provides qualitative feedback. You can see the model improve over time, which is more intuitive than just watching loss decrease.

4. **Checkpoint Saving**:
   ```python
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'config': config.to_dict(),
       'train_loss': train_loss,
       'val_loss': val_loss,
   }, checkpoint_path)
   ```
   
   **Why Save Everything**:
   - **model_state_dict**: Model weights
   - **optimizer_state_dict**: Optimizer state (allows resuming training)
   - **config**: Model configuration (needed to reconstruct model architecture)
   - **losses**: Training history

   **Insight**: Saving optimizer state is crucial for resuming training. Without it, you'd lose momentum and adaptive learning rate information, making resumed training less effective.

---

### 7. Command-Line Interface (`main()`)

```python
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train GPT on TinyStories dataset")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, default=None, ...)
    parser.add_argument("--max-samples", type=int, default=10000, ...)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, ...)
    parser.add_argument("--batch-size", type=int, default=32, ...)
    parser.add_argument("--learning-rate", type=float, default=3e-4, ...)
    
    # Model arguments
    parser.add_argument("--context-length", type=int, default=128, ...)
    parser.add_argument("--embedding-dimension", type=int, default=256, ...)
    parser.add_argument("--number-of-heads", type=int, default=4, ...)
    parser.add_argument("--number-of-layers", type=int, default=4, ...)
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda", ...)
    parser.add_argument("--save-dir", type=str, default="checkpoints", ...)
    parser.add_argument("--save-every", type=int, default=5, ...)
    parser.add_argument("--eval-every", type=int, default=1, ...)
    parser.add_argument("--generate-every", type=int, default=2, ...)
```

**Design Philosophy**: All hyperparameters are configurable via command line, making experimentation easy without code changes.

**Example Usage**:
```bash
python examples/train_tiny_stories.py \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --context-length 256 \
    --embedding-dimension 512 \
    --number-of-layers 6
```

### 7.1 Complete Argument Reference

The following table lists all available command-line arguments, their types, default values, and descriptions:

| Argument | Type | Default | Options/Constraints | Description |
|----------|------|---------|---------------------|-------------|
| `--data-path` | `str` | `None` | Any file path | Path to data file. If `None`, downloads TinyStories dataset from Hugging Face. |
| `--max-samples` | `int` | `10000` | Positive integer | Maximum number of samples to use from the dataset. Limits dataset size for faster iteration. |
| `--epochs` | `int` | `10` | Positive integer | Number of training epochs. Each epoch processes the entire training dataset once. |
| `--batch-size` | `int` | `32` | Positive integer | Number of sequences processed in each training step. Larger values use more memory but train faster. |
| `--learning-rate` | `float` | `3e-4` | Positive float | Learning rate for the AdamW optimizer. Common values: `1e-4` to `1e-3`. |
| `--context-length` | `int` | `128` | Positive integer | Maximum sequence length the model can process. Longer contexts require more memory. |
| `--embedding-dimension` | `int` | `256` | Positive integer | Dimension of token embeddings and hidden states. Larger values increase model capacity and memory usage. |
| `--number-of-heads` | `int` | `4` | Positive integer | Number of attention heads in multi-head attention. Must divide `embedding-dimension` evenly. |
| `--number-of-layers` | `int` | `4` | Positive integer | Number of transformer layers (blocks). More layers increase model depth and capacity. |
| `--device` | `str` | `"cuda"` | `"cuda"`, `"cpu"`, `"mps"` | Device to train on. `"cuda"` for NVIDIA GPU, `"mps"` for Apple Silicon, `"cpu"` for CPU. Automatically falls back to CPU if requested device is unavailable. |
| `--save-dir` | `str` | `"checkpoints"` | Any directory path | Directory to save model checkpoints. Created if it doesn't exist. |
| `--save-every` | `int` | `5` | Positive integer | Save checkpoint every N epochs. Set to `1` to save after every epoch. |
| `--eval-every` | `int` | `1` | Positive integer | Run validation evaluation every N epochs. Set to `1` to validate after every epoch. |
| `--generate-every` | `int` | `2` | Positive integer | Generate sample text every N epochs. Set to `0` to disable generation during training. |

**Notes:**
- All integer arguments must be positive (> 0)
- `--number-of-heads` must evenly divide `--embedding-dimension` (e.g., if `embedding-dimension=256`, valid values for `number-of-heads` are 1, 2, 4, 8, 16, 32, 64, 128, 256)
- If `--data-path` is provided but the file doesn't exist, the script will attempt to download the dataset
- The script automatically adjusts `--context-length` if the dataset is too short to create sequences of the specified length

---

## Key Insights and Best Practices

### 1. **Defensive Programming**
The script includes multiple fallback mechanisms:
- Dataset download failures → sample data
- CUDA unavailable → CPU fallback
- Text too short → adaptive context length
- Empty dataset → helpful error messages

### 2. **Memory Efficiency**
- `max_samples` parameter limits dataset size
- Stride parameter creates more sequences without storing more text
- Checkpoint saving allows training in stages

### 3. **Reproducibility**
- Fixed random seed for train/val split (`manual_seed(42)`)
- Deterministic data loading (no multiprocessing randomness)
- Saves configuration with checkpoints

### 4. **Monitoring and Debugging**
- Parameter counting helps understand model complexity
- Perplexity is more interpretable than raw loss
- Text generation provides qualitative feedback
- Best model tracking prevents overfitting

### 4.1 Qualitative Model Evaluation: What Generated Stories Reveal

Beyond validation loss and perplexity, inspecting generated text at different training stages provides deep insight into *what* the model has actually learned. Comparing outputs from two models trained on different amounts of data reveals clear qualitative differences that metrics alone cannot capture.

#### Observed Progression in Generated Text

When comparing an early model (trained on less data) with a later model (trained longer and on more text), several improvements become apparent:

- **Sentence Structure**  
  Early outputs tend to merge multiple clauses into a single sentence, often losing grammatical clarity. With more training data, sentences become shorter, cleaner, and more deliberate.

- **Causal Reasoning**  
  Later generations show clearer cause → effect chains (e.g. an action leading to a consequence, followed by a reaction). This indicates the model has learned *event sequencing*, not just local token patterns.

- **Dialogue Handling**  
  Dialogue emerges naturally only after sufficient exposure to conversational text. Improved outputs correctly place quotation marks, attribution (“she said”), and emotionally appropriate responses.

- **Story Closure**  
  Better-trained models end stories more cleanly, often resolving the situation instead of drifting or cutting off mid-thought. This reflects improved long-range coherence within the context window.

#### “Good” Mistakes Are a Sign of Learning

As models improve, their errors change in nature:

- Early models make **syntactic or coherence errors**
- More mature small models make **near-miss semantic errors**

For example, choosing the wrong helper (“mechanic” instead of a more appropriate role) shows that the model understands the *problem → helper* pattern, but lacks fine-grained real-world grounding. These are *reasonable mistakes* and indicate meaningful internal structure.

#### Why This Matters During Training

This reinforces why text generation during training is essential:

- Loss and perplexity measure **statistical fit**
- Generated text reveals **conceptual understanding**
- Small improvements in data quality often yield larger gains than more epochs

In practice, once a model reaches this stage, **targeted data curation** (e.g. clearer cause–effect stories, everyday situations, role grounding) tends to outperform simply training longer.

**Key takeaway:**  
A small language model that produces coherent stories with minor semantic mistakes has crossed an important threshold — it is no longer guessing text blindly, but modeling intent at the sentence and story level.

### 5. **Modularity**
- Separates data loading, model creation, training, and generation
- Each component can be tested independently
- Easy to swap implementations (e.g., different tokenizers)

### 6. **Training Best Practices**
- Train/validation split for monitoring overfitting
- Learning rate scheduling (could be added)
- Gradient clipping (could be added for stability)
- Early stopping (could be added based on validation loss)

---

## Potential Improvements

1. **Learning Rate Scheduling**: Add cosine annealing or warmup
2. **Gradient Clipping**: Prevent exploding gradients
3. **Early Stopping**: Stop training when validation loss plateaus
4. **Mixed Precision Training**: Use FP16 to speed up training
5. **Distributed Training**: Support multi-GPU training
6. **TensorBoard Logging**: Visualize training metrics
7. **Hyperparameter Tuning**: Add support for grid search or random search

---

## Conclusion

The `train_tiny_stories.py` script demonstrates a complete training pipeline with:
- Robust error handling
- Flexible configuration
- Comprehensive monitoring
- Best practices for transformer training

While it's designed for educational purposes (small model, simple dataset), the patterns and practices shown here scale to larger models and datasets. Understanding this code provides a solid foundation for training language models of any size.

---

Previous: [Training Pipeline](04-training-pipeline.md) | Next: [Advanced Topics](05-advanced-topics.md)
