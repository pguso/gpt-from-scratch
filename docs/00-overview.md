# Overview: Understanding GPT from 30,000 Feet

## What You'll Learn

By the end of this document, you'll understand:
- What GPT is and why it matters
- The high-level architecture
- How transformers changed everything
- The training and generation process

## What is GPT?

GPT (Generative Pre-trained Transformer) is a type of neural network that:

1. **Generative**: Creates text, one token at a time, predicting the next word based on context
2. **Pre-trained**: Learns from massive amounts of text data in an unsupervised way
3. **Transformer**: Uses the revolutionary transformer architecture introduced in "Attention Is All You Need"

### The Three Pillars

**Generative**: Unlike classification models that predict categories, GPT generates sequences. Given a prompt like "The weather today is", it predicts the next token, then the next, building text autoregressively.

**Pre-trained**: GPT learns language patterns from vast text corpora (books, websites, articles). This pre-training gives it a deep understanding of grammar, facts, reasoning, and style that can be fine-tuned for specific tasks.

**Transformer**: The transformer architecture, introduced in 2017, replaced recurrent neural networks (RNNs) and long short-term memory (LSTM) networks. It uses attention mechanisms to process entire sequences in parallel, making training much faster and more effective.

## Why GPT Matters

GPT models have revolutionized natural language processing:

- **Language Understanding**: They capture semantic meaning, context, and nuance
- **Few-Shot Learning**: Can perform new tasks with minimal examples
- **Versatility**: Same architecture for translation, summarization, question-answering, and more
- **Scalability**: Performance improves predictably with more data and parameters

## High-Level Architecture

A GPT model consists of several key components:

```
Input Text → Tokenization → Embeddings → Transformer Blocks → Output Logits → Generated Text
```

### 1. Tokenization
Text is converted into tokens (subword units). For example, "Hello world" might become `[15496, 995]` using the GPT-2 tokenizer.

### 2. Embeddings
Each token is converted into a dense vector representation:
- **Token Embeddings**: Learn the meaning of each token
- **Position Embeddings**: Encode the position of each token in the sequence

### 3. Transformer Blocks
The core of GPT. Each block contains:
- **Multi-Head Attention**: Allows the model to focus on relevant parts of the input
- **Feed-Forward Network**: Processes the attended information
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Helps gradients flow during training

### 4. Output Layer
Converts the final hidden states into logits (raw scores) over the vocabulary, which are then converted to probabilities for the next token.

## The Training Process

GPT is trained using **next-token prediction**:

1. **Input**: A sequence of tokens `[t1, t2, t3, ..., tn]`
2. **Target**: The same sequence shifted by one: `[t2, t3, t4, ..., tn+1]`
3. **Objective**: Predict the next token at each position

For example, given "The cat sat on the", the model learns to predict "mat" (or whatever comes next in the training data).

### Key Training Details

- **Causal Masking**: The model can only attend to previous tokens (not future ones), ensuring it learns to predict sequentially
- **Self-Supervised**: No manual labeling needed - the text itself provides the supervision
- **Massive Scale**: GPT-3 was trained on hundreds of gigabytes of text

## The Generation Process

Once trained, GPT generates text autoregressively:

1. Start with a prompt (e.g., "Once upon a time")
2. Feed it through the model to get logits for the next token
3. Sample from the probability distribution (with temperature for creativity)
4. Append the sampled token to the sequence
5. Repeat until desired length or stop token

### Sampling Strategies

- **Greedy**: Always pick the most likely token (deterministic but repetitive)
- **Temperature**: Scale logits before softmax to control randomness
- **Top-k**: Only sample from the k most likely tokens
- **Top-p (nucleus)**: Sample from tokens whose cumulative probability reaches p

## How Transformers Changed Everything

Before transformers, sequence models used RNNs or LSTMs:

**Problems with RNNs**:
- Sequential processing (slow)
- Vanishing gradients (hard to train)
- Limited context window

**Transformer Advantages**:
- **Parallel Processing**: All tokens processed simultaneously
- **Long-Range Dependencies**: Attention can connect distant tokens directly
- **Scalability**: Easy to scale with more layers and parameters
- **Efficiency**: Faster training and inference

The attention mechanism is the key innovation-it allows the model to "look" at all previous tokens simultaneously and decide which ones are most relevant.

## Model Sizes and Capabilities

GPT models come in various sizes:

- **Small** (117M params): Good for learning and experimentation
- **Medium** (345M params): Better language understanding
- **Large** (1.5B+ params): Strong performance on many tasks
- **XL** (175B+ params): GPT-3 level, few-shot learning capabilities

In this repository, you can configure models from tiny (for learning) to larger sizes (for better performance).

## Example Scripts

This repository includes complete, working examples:

### `quickstart.py` - Untrained Model Demo
```bash
python examples/quickstart.py
```
- Creates an untrained model with random weights
- Demonstrates model structure and parameter count
- Generates text (will be gibberish - this is expected!)
- **Purpose**: See the model architecture in action

### `train_tiny_stories.py` - Complete Training Script
```bash
python examples/train_tiny_stories.py --epochs 10
```
- Downloads/loads TinyStories dataset
- Trains model from scratch
- Saves checkpoints automatically
- Generates sample text during training
- **Purpose**: Learn the complete training pipeline

**Key Features**:
- Automatic dataset handling (downloads or uses sample data)
- Smart context_length adjustment for short texts
- Comprehensive error handling and helpful messages
- Progress tracking with loss and perplexity
- Automatic checkpointing

### `generate_text.py` - Text Generation
```bash
python examples/generate_text.py --checkpoint checkpoints/best_model.pt
```
- Loads trained model checkpoints
- Generates coherent text from prompts
- Supports temperature and top-k sampling
- **Purpose**: Use your trained model to generate text

### `analyze_attention.py` - Attention Visualization
```bash
python examples/analyze_attention.py --checkpoint checkpoints/best_model.pt
```
- Extracts attention weights from all layers
- Creates heatmap visualizations
- Analyzes attention patterns
- **Purpose**: Understand what your model is "looking at"

## What's Next?

Now that you understand the big picture:

1. **Dive into Architecture**: Learn how transformer blocks are constructed
2. **Understand Attention**: See how the attention mechanism works in detail
3. **Build Components**: Implement each building block yourself
4. **Train a Model**: Learn the training pipeline (use `train_tiny_stories.py`)
5. **Explore Advanced Topics**: Optimizations, fine-tuning, and extensions

## Key Takeaways

- GPT is a generative language model that predicts the next token
- It uses the transformer architecture with attention mechanisms
- Training is self-supervised via next-token prediction
- Generation is autoregressive, one token at a time
- Transformers enable parallel processing and better long-range dependencies

---

Next: [Transformer Architecture](01-transformer-architecture.md)
