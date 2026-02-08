# GPT From Scratch: A Top-Down Learning Journey

A comprehensive, educational implementation of GPT (Generative Pre-trained Transformer) designed for developers who want to deeply understand how Large Language Models work.

## What Makes This Different?

This isn't just another GPT implementation. It's a simple, practical introduction that:

- Top-down approach: Start with concepts, then dive into code
- Easy to get started: Minimal setup, clear examples, straightforward code
- Focused on essentials: Learn GPT fundamentals without unnecessary complexity

## Who Is This For?

- Software Developers learning deep learning
- Students studying transformer architectures
- Researchers needing a clean reference implementation
- Anyone curious about how ChatGPT-like models work

Prerequisites: Basic Python, some PyTorch knowledge helpful (but not required)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/pguso/gpt-from-scratch.git
cd gpt-from-scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Step 1: Run Your First Model (Untrained - 2 minutes)

This creates an **untrained model** with random weights. The output will be gibberish, but it demonstrates the model structure:

```bash
python examples/quickstart.py
```

**What you'll see:**
- Model architecture and parameter count
- Text generation from a prompt
- **Note:** The output will be random/gibberish since the model hasn't been trained yet

This is expected! An untrained model is like a newborn - it can make sounds but doesn't understand language yet.

### Step 2: Train Your Model (30 minutes - several hours)

To get meaningful output, you need to train the model on text data. We'll use the TinyStories dataset:

```
pip install datasets
```

```bash
# Train a small model (quick, for learning)
python examples/train_tiny_stories.py --epochs 10 --max-samples 10000

# Train a larger model (better results, takes longer)
python examples/train_tiny_stories.py --epochs 20 --max-samples 50000 --embedding-dimension 512 --number-of-layers 6
```

**What happens during training:**
- Downloads TinyStories dataset (or uses sample data)
- Trains the model to predict next tokens
- Saves checkpoints to `checkpoints/` directory
- Shows training progress and sample generations

**Training time depends on:**
- Model size (embedding-dimension, number-of-layers, number-of-heads)
- Dataset size (--max-samples)
- Number of epochs
- Your hardware (CPU is slow, GPU is much faster)

**Expected outputs:**
- `checkpoints/best_model.pt` - Best model based on validation loss
- `checkpoints/checkpoint_epoch_N.pt` - Periodic checkpoints

**All available parameters:** See the [Training Script Parameters table](docs/QUICK_REFERENCE.md#training-script-parameters) in the Quick Reference for a complete list of command-line arguments you can pass to `train_tiny_stories.py`.

### Step 3: Generate Text with Trained Model

Once you have a trained checkpoint, generate text:

```bash
# Generate with default settings
python examples/generate_text.py --checkpoint checkpoints/best_model.pt

# Customize generation
python examples/generate_text.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "The little girl" \
    --length 150 \
    --temperature 0.7 \
    --top-k 50
```

**You should now see:**
- Coherent text that follows the prompt
- Story-like continuations (if trained on TinyStories)
- Much better quality than the untrained model!

### Quick Comparison

| Stage | Command | Output Quality |
|-------|---------|----------------|
| **Untrained** | `python examples/quickstart.py` | Random gibberish |
| **Training** | `python examples/train_tiny_stories.py` | Creates checkpoint |
| **Trained** | `python examples/generate_text.py --checkpoint checkpoints/best_model.pt` | Coherent text |

### Training Tips

**For faster experimentation:**
```bash
# Small model, small dataset, few epochs
python examples/train_tiny_stories.py \
    --epochs 5 \
    --max-samples 5000 \
    --embedding-dimension 128 \
    --number-of-layers 2 \
    --number-of-heads 2
```

**For better results:**
```bash
# Larger model, more data, more epochs
python examples/train_tiny_stories.py \
    --epochs 20 \
    --max-samples 50000 \
    --embedding-dimension 512 \
    --number-of-layers 6 \
    --number-of-heads 8 \
    --batch-size 64
```

**Monitor training:**
- Watch the loss decrease over epochs
- Check sample generations during training
- Lower validation loss = better model

### Other Examples

The `examples/` directory contains additional scripts:

- **`quickstart.py`** - Create and test an untrained model (gibberish output)
- **`train_tiny_stories.py`** - Train a model on TinyStories dataset
- **`generate_text.py`** - Generate text with a trained model checkpoint
- **`analyze_attention.py`** - Visualize attention patterns in the model

**Example: Analyze attention patterns**
```bash
# Analyze attention in a trained model
python examples/analyze_attention.py \
    --checkpoint checkpoints/best_model.pt \
    --text "The cat sat on the mat" \
    --layer 2 \
    --show-plots
```

### Troubleshooting

**Q: The model outputs gibberish even after training**
- Training might not be complete - try more epochs
- Model might be too small - increase `--embedding-dimension` and `--number-of-layers`
- Need more data - increase `--max-samples`
- Check validation loss - it should decrease over time

**Q: Training is too slow**
- Use GPU if available: `--device cuda` or `--device mps` on apple silicon
- Reduce model size: smaller `--embedding-dimension`, `--number-of-layers`
- Reduce dataset: smaller `--max-samples`
- Reduce batch size: smaller `--batch-size`

**Q: Out of memory errors**
- Reduce `--batch-size`
- Reduce `--context-length`
- Reduce model size (`--embedding-dimension`, `--number-of-layers`)

**Q: Can't download TinyStories dataset**
- Install datasets library: `pip install datasets`
- Or use sample data (script will fall back automatically)
- Or provide your own data file with `--data-path`

## Documentation

**Start here:** [Documentation Index](docs/README.md) - Complete guide to all documentation

**Quick access:**
- [Quick Reference](docs/QUICK_REFERENCE.md) - Common commands, hyperparameters, and code snippets

**Documentation structure:**
- [Model Usage Guide](docs/01-model-usage-guide.md) - How to use the model (inputs, outputs, configuration)
- [Training Implementation](docs/02-training-implementation.md) - How training works
- [Using the Model](docs/03-using-the-model.md) - How to use the trained model
- [Pitfalls and Challenges](docs/04-pitfalls-and-challenges.md) - Common mistakes and probabilistic challenges

## Repository Structure

```
gpt-from-scratch/
├── docs/              # Comprehensive documentation
├── src/              # Clean source code
├── examples/         # Practical usage examples
└── data/             # Sample datasets
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Inspired by:
- "Attention Is All You Need" - Original Transformer paper
- "Language Models are Unsupervised Multitask Learners" - GPT-2 paper
- nanoGPT by Andrej Karpathy
- The Illustrated Transformer by Jay Alammar

---

Made with love for the ML community
