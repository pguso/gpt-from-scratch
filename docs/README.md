# Documentation Index

Welcome to the GPT From Scratch documentation! This guide focuses on **practical usage** - how to use the model, how it's trained, and the challenges you'll face when working with probabilistic language models.

## Documentation Structure

This documentation is organized for developers who want to understand the code and use the model effectively:

### 1. [Model Usage Guide](01-model-usage-guide.md)
**How to use the GPT model**
- Model creation and configuration
- Input and output specifications
- Forward pass usage
- Common usage patterns
- Constraints and limitations
- Links to deep dive resources

### 2. [Training Implementation](02-training-implementation.md)
**How training works in practice**
- Training pipeline walkthrough
- Data preparation and tokenization
- Training loop details
- Loss computation and metrics
- Checkpointing and resuming
- Common training issues

### 3. [Using the Model](03-using-the-model.md)
**How to use the trained model**
- Loading checkpoints
- Text generation basics
- Sampling strategies (temperature, top-k)
- Generation parameters and their effects
- Practical examples

### 4. [Pitfalls and Challenges](04-pitfalls-and-challenges.md)
**Common mistakes and probabilistic challenges**
- Common pitfalls and how to avoid them
- Challenges of probabilistic generation
- Non-determinism and reproducibility
- Evaluation difficulties
- Debugging strategies

### 5. [Understanding Attention Analysis](05-understanding-attention-analysis.md)
**How attention visualization works**
- What `analyze_attention.py` does
- How PyTorch hooks work
- Understanding attention weight matrices
- Reading visualizations
- Using the script

### 6. [Quick Reference](QUICK_REFERENCE.md)
**Quick access to commands and code snippets**
- Common commands
- Hyperparameter cheat sheet
- Code snippets
- Troubleshooting quick links

## Quick Start

1. **Want to use the model?** → Start with [Model Usage Guide](01-model-usage-guide.md)
2. **Want to train a model?** → Read [Training Implementation](02-training-implementation.md)
3. **Want to generate text?** → Jump to [Using the Model](03-using-the-model.md)
4. **Running into issues?** → Check [Pitfalls and Challenges](04-pitfalls-and-challenges.md)
5. **Want to visualize attention?** → See [Understanding Attention Analysis](05-understanding-attention-analysis.md)

## What This Documentation Covers

✅ **Model usage** - How to create and use the model  
✅ **Training pipeline** - What happens during training  
✅ **Practical usage** - How to use the model  
✅ **Common pitfalls** - Mistakes to avoid  
✅ **Probabilistic challenges** - Working with non-deterministic models  

## What This Documentation Doesn't Cover

❌ **Transformer theory** - How attention mechanisms work conceptually  
❌ **Architecture explanations** - Why transformers were designed this way  
❌ **Mathematical derivations** - The theory behind the algorithms  

For those topics, there are excellent resources available:
- "Attention Is All You Need" (original transformer paper)
- "The Illustrated Transformer" by Jay Alammar
- "Language Models are Unsupervised Multitask Learners" (GPT-2 paper)

## Additional Resources

- **Main README**: See the root [README.md](../README.md) for installation and quick start
- **Examples**: Check the `examples/` directory for runnable scripts
- **Tests**: Educational tests in `tests/` directory
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to contribute

---

**Next Steps:** Choose a topic above or jump to [Quick Reference](QUICK_REFERENCE.md) if you're ready to code.
