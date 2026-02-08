# Documentation Index

Welcome to the GPT From Scratch documentation! This guide will help you navigate the learning materials and find what you need.

## Learning Path

This documentation follows a **top-down learning approach**: start with high-level concepts, then dive into implementation details. Follow this order for the best learning experience:

### Phase 1: Understanding the Concepts
1. **[Overview](00-overview.md)** - What is GPT and why transformers matter
   - High-level introduction to GPT
   - Training and generation process
   - How transformers changed NLP

2. **[Transformer Architecture](01-transformer-architecture.md)** - Detailed architecture breakdown
   - Complete model structure
   - Core components explained
   - Design choices and rationale

3. **[Attention Mechanism](02-attention-mechanism.md)** - Deep dive into how attention works
   - Scaled dot-product attention
   - Multi-head attention
   - Causal masking
   - Query, Key, Value explained

### Phase 2: Building Components
4. **[Building Blocks](03-building-blocks.md)** - Layer-by-layer construction guide
   - Token and position embeddings
   - Layer normalization
   - Feed-forward networks
   - Complete transformer block assembly

### Phase 3: Training and Usage
5. **[Training Pipeline](04-training-pipeline.md)** - General training concepts
   - Next-token prediction objective
   - Data preparation and preprocessing
   - Training loop structure
   - Loss functions and metrics
   - Validation and monitoring
   - Checkpointing

6. **[Training Walkthrough](04-training-walkthrough.md)** - Complete training script walkthrough
   - Detailed explanation of `train_tiny_stories.py`
   - Data loading and tokenization
   - Model configuration and initialization
   - Training loop implementation
   - Command-line arguments reference

### Phase 4: Advanced Topics
7. **[Advanced Topics](05-advanced-topics.md)** - Optimizations, fine-tuning, and extensions
   - Training optimizations (mixed precision, gradient accumulation)
   - Inference optimizations (KV caching, quantization)
   - Architecture extensions (RoPE, Flash Attention)
   - Fine-tuning techniques (LoRA, adapters)
   - Attention analysis and visualization

## Quick Navigation

### By Topic

**Getting Started:**
- [Overview](00-overview.md) - Start here if you're new to GPT
- [Quick Reference](QUICK_REFERENCE.md) - Common commands and cheat sheets

**Architecture & Design:**
- [Transformer Architecture](01-transformer-architecture.md) - Overall structure
- [Attention Mechanism](02-attention-mechanism.md) - Core innovation
- [Building Blocks](03-building-blocks.md) - Component details

**Practical Usage:**
- [Training Pipeline](04-training-pipeline.md) - Training concepts and fundamentals
- [Training Walkthrough](04-training-walkthrough.md) - Script implementation walkthrough
- [Quick Reference](QUICK_REFERENCE.md) - Commands and examples

**Going Deeper:**
- [Advanced Topics](05-advanced-topics.md) - Optimizations and extensions

### By Experience Level

**Beginner:**
1. Read [Overview](00-overview.md)
2. Skim [Transformer Architecture](01-transformer-architecture.md) for the big picture
3. Use [Quick Reference](QUICK_REFERENCE.md) to run examples
4. Come back to detailed docs as needed

**Intermediate:**
1. Review [Overview](00-overview.md) and [Transformer Architecture](01-transformer-architecture.md)
2. Deep dive into [Attention Mechanism](02-attention-mechanism.md)
3. Study [Building Blocks](03-building-blocks.md)
4. Follow [Training Pipeline](04-training-pipeline.md) walkthrough

**Advanced:**
1. Use docs as reference for specific topics
2. Focus on [Advanced Topics](05-advanced-topics.md)
3. Explore codebase with docs as guide

## Document Summaries

### [00-overview.md](00-overview.md)
**What you'll learn:**
- What GPT is and why it matters
- High-level architecture overview
- Training and generation process
- How transformers changed NLP

**Best for:** First-time readers, getting the big picture

---

### [01-transformer-architecture.md](01-transformer-architecture.md)
**What you'll learn:**
- Complete GPT model structure
- Embedding layers (token + position)
- Transformer block components
- Forward pass flow
- Parameter counting
- Design choices (pre-norm, residuals, causal masking)

**Best for:** Understanding how all pieces fit together

---

### [02-attention-mechanism.md](02-attention-mechanism.md)
**What you'll learn:**
- Scaled dot-product attention formula
- Multi-head attention implementation
- Causal masking for autoregressive models
- Query, Key, Value analogy and math
- Attention visualization techniques
- Computational complexity

**Best for:** Deep understanding of the core innovation

---

### [03-building-blocks.md](03-building-blocks.md)
**What you'll learn:**
- Token and position embeddings
- Layer normalization (pre-norm vs post-norm)
- GELU activation function
- Feed-forward network design
- Complete transformer block assembly
- Component interactions
- Hyperparameter impact

**Best for:** Understanding each component in detail

---

### [04-training-pipeline.md](04-training-pipeline.md)
**What you'll learn:**
- Next-token prediction training objective
- Data preparation and preprocessing
- Training loop structure and components
- Loss functions (cross-entropy) and metrics (perplexity)
- Validation and monitoring strategies
- Checkpointing and model saving
- Common challenges and best practices

**Best for:** Understanding the fundamentals of training GPT models

---

### [04-training-walkthrough.md](04-training-walkthrough.md)
**What you'll learn:**
- Complete walkthrough of `train_tiny_stories.py` script
- Data loading and tokenization implementation
- Dataset creation with adaptive context length
- Model configuration and initialization
- Training loop implementation details
- Validation and checkpointing code
- Text generation during training
- Complete command-line arguments reference

**Best for:** Learning the specific implementation details of the training script

**Note:** This document focuses on the specific training script implementation. For general training concepts, see [Training Pipeline](04-training-pipeline.md).

---

### [05-advanced-topics.md](05-advanced-topics.md)
**What you'll learn:**
- Training optimizations (mixed precision, gradient accumulation, LR scheduling)
- Inference optimizations (KV caching, quantization, pruning)
- Architecture extensions (RoPE, Flash Attention, MoE)
- Fine-tuning techniques (LoRA, adapters, prompt tuning)
- Attention analysis and visualization
- Advanced generation techniques (top-k, top-p, beam search)
- Evaluation metrics
- Distributed training
- Research directions

**Best for:** Optimizing performance and extending the model

---

### [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
**What you'll find:**
- Common commands for training and generation
- Hyperparameter cheat sheet
- Troubleshooting quick links
- Code snippets for common tasks
- Model configuration examples

**Best for:** Quick lookups while working

## Additional Resources

- **Main README**: See the root [README.md](../README.md) for installation, quick start, and project overview
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to contribute
- **Examples**: Check the `examples/` directory for runnable scripts
- **Notebooks**: Interactive Jupyter notebooks in `notebooks/` directory
- **Tests**: Educational tests in `tests/` directory

## Reading Tips

1. **Don't skip the overview** - It provides essential context
2. **Follow the numbered order** - Each doc builds on previous ones
3. **Use Quick Reference** - Keep it open while coding
4. **Experiment as you read** - Run examples from the docs
5. **Revisit docs** - Understanding deepens with practice

## Feedback

Found an error or have suggestions? Please open an issue or submit a pull request!

---

**Next Steps:** Start with [Overview](00-overview.md) or jump to [Quick Reference](QUICK_REFERENCE.md) if you're ready to code.
