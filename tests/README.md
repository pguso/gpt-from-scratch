# Test Suite Documentation

This directory contains comprehensive tests for the GPT-from-scratch implementation. The tests are designed to be both functional (ensuring correctness) and educational (helping you understand how each component works).

## Table of Contents

- [Overview](#overview)
- [Running Tests](#running-tests)
- [Test Files](#test-files)
- [Test Coverage](#test-coverage)
- [Writing New Tests](#writing-new-tests)
- [Troubleshooting](#troubleshooting)

## Overview

The test suite uses [pytest](https://docs.pytest.org/) as the testing framework. All tests are organized by component, making it easy to understand what each part of the codebase does and verify its correctness.

### Test Structure

```
tests/
├── __init__.py          # Test package initialization
├── test_data.py         # Data processing and tokenization tests
├── test_blocks.py       # Transformer building blocks tests
├── test_attention.py    # Multi-head attention mechanism tests
├── test_model.py        # Complete GPT model tests
└── test_generation.py   # Text generation tests
```

## Running Tests

### Prerequisites

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

The test suite requires:
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting (optional but recommended)
- `torch>=2.0.0` - PyTorch for model components

### Basic Usage

**Run all tests:**
```bash
pytest tests/
```

**Run a specific test file:**
```bash
pytest tests/test_model.py
```

**Run a specific test class:**
```bash
pytest tests/test_model.py::TestGPTModel
```

**Run a specific test function:**
```bash
pytest tests/test_model.py::TestGPTModel::test_forward_pass
```

**Run tests with verbose output:**
```bash
pytest tests/ -v
```

**Run tests with detailed output (shows print statements):**
```bash
pytest tests/ -v -s
```

### Advanced Usage

**Run tests with coverage report:**
```bash
pytest tests/ --cov=src --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html` that you can open in your browser.

**Run tests in parallel (faster):**
```bash
pip install pytest-xdist
pytest tests/ -n auto
```

**Run only tests that match a pattern:**
```bash
pytest tests/ -k "attention"
```

**Stop on first failure:**
```bash
pytest tests/ -x
```

**Show local variables on failure:**
```bash
pytest tests/ -l
```

## Test Files

### `test_data.py` - Data Processing Tests

Tests for data loading, tokenization, and dataset creation.

**Test Classes:**
- `TestDataProcessing` - Tests data processing functionality

**What it tests:**
- ✅ Tokenization encoding and decoding
- ✅ Dataset creation with proper length
- ✅ Input and target sequence generation
- ✅ Dataset indexing and batching

**Example test:**
```python
def test_tokenization(self):
    """Test basic tokenization."""
    tokenizer = get_tokenizer()
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text
```

**Run these tests:**
```bash
pytest tests/test_data.py -v
```

### `test_blocks.py` - Transformer Building Blocks Tests

Tests for individual transformer components: LayerNorm, GELU activation, and FeedForward networks.

**Test Classes:**
- `TestBlocks` - Tests transformer building blocks

**What it tests:**
- ✅ Layer normalization output shape and behavior
- ✅ GELU activation function
- ✅ FeedForward network forward pass
- ✅ Shape preservation through all blocks

**Example test:**
```python
def test_layer_norm(self):
    """Test layer normalization."""
    norm = LayerNorm(64)
    x = torch.randn(2, 10, 64)
    output = norm(x)
    assert output.shape == x.shape
```

**Run these tests:**
```bash
pytest tests/test_blocks.py -v
```

### `test_attention.py` - Attention Mechanism Tests

Tests for the multi-head attention mechanism, which is the core of the transformer architecture.

**Test Classes:**
- `TestAttention` - Tests attention mechanism

**What it tests:**
- ✅ Multi-head attention forward pass
- ✅ Output shape matches input shape
- ✅ Causal masking is properly applied
- ✅ Mask shape matches context length

**Example test:**
```python
def test_attention_forward(self):
    """Test attention forward pass."""
    attention = MultiHeadAttention(
        input_dimension=64, output_dimension=64, context_length=128,
        dropout=0.1, number_of_heads=4
    )
    x = torch.randn(2, 10, 64)
    output = attention(x)
    assert output.shape == x.shape
```

**Run these tests:**
```bash
pytest tests/test_attention.py -v
```

### `test_model.py` - Complete GPT Model Tests

Tests for the complete GPT model, including model creation and forward pass.

**Test Classes:**
- `TestGPTModel` - Tests complete GPT model

**What it tests:**
- ✅ Model can be instantiated with configuration
- ✅ Forward pass produces correct output shapes
- ✅ Logits shape matches (batch_size, sequence_length, vocab_size)
- ✅ Model handles different input sizes

**Example test:**
```python
def test_forward_pass(self):
    """Test forward pass."""
    config = GPTConfig(
        vocab_size=1000,
        embedding_dimension=128,
        number_of_heads=4,
        number_of_layers=2,
        context_length=64
    )
    model = GPTModel(config)
    input_ids = torch.randint(0, 1000, (2, 10))
    logits = model(input_ids)
    assert logits.shape == (2, 10, 1000)
```

**Run these tests:**
```bash
pytest tests/test_model.py -v
```

### `test_generation.py` - Text Generation Tests

Tests for text generation functionality, including token generation and sequence extension.

**Test Classes:**
- `TestGeneration` - Tests text generation

**What it tests:**
- ✅ Text generation function works
- ✅ Generated sequence has correct length
- ✅ Generation respects maximum token limits
- ✅ Model can generate from input prompts

**Example test:**
```python
def test_generation(self):
    """Test that generation works."""
    config = GPTConfig(
        vocab_size=1000,
        embedding_dimension=128,
        number_of_heads=4,
        number_of_layers=2,
        context_length=64
    )
    model = GPTModel(config)
    model.eval()
    input_ids = [1, 2, 3]
    output = generate_text(model, input_ids, maximum_new_tokens=5)
    assert len(output) == len(input_ids) + 5
```

**Run these tests:**
```bash
pytest tests/test_generation.py -v
```

## Test Coverage

The test suite covers the following components:

| Component | Coverage | Test File |
|-----------|----------|-----------|
| Data Processing | ✅ Tokenization, Dataset | `test_data.py` |
| Building Blocks | ✅ LayerNorm, GELU, FeedForward | `test_blocks.py` |
| Attention | ✅ Multi-head attention, Causal masking | `test_attention.py` |
| GPT Model | ✅ Model creation, Forward pass | `test_model.py` |
| Generation | ✅ Text generation | `test_generation.py` |

### Coverage Report

To generate a detailed coverage report:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

This shows which lines are covered and which are missing.

For an HTML report:
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## Writing New Tests

### Test Structure

Follow this structure when adding new tests:

```python
"""
Tests for [component name].
"""

import pytest
import torch
from src.module.component import Component


class TestComponent:
    """Test [component] functionality."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        component = Component(...)
        # Your test code here
        assert condition
```

### Best Practices

1. **Use descriptive test names**: Test names should clearly describe what they test
   ```python
   # Good
   def test_attention_forward_pass_with_causal_masking(self):
   
   # Bad
   def test_attention(self):
   ```

2. **One assertion per test concept**: Each test should verify one specific behavior
   ```python
   # Good - tests one thing
   def test_output_shape(self):
       assert output.shape == expected_shape
   
   # Also good - but tests related things
   def test_forward_pass(self):
       assert output.shape == expected_shape
       assert not torch.isnan(output).any()
   ```

3. **Use fixtures for common setup**: If you find yourself repeating setup code, use pytest fixtures
   ```python
   @pytest.fixture
   def sample_model():
       config = GPTConfig(...)
       return GPTModel(config)
   
   def test_something(sample_model):
       # Use sample_model here
   ```

4. **Test edge cases**: Don't just test the happy path
   ```python
   def test_empty_input(self):
       # Test with empty input
   
   def test_large_input(self):
       # Test with very large input
   ```

5. **Use meaningful assertions**: Include messages in assertions when helpful
   ```python
   assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
   ```

### Example: Adding a New Test

Let's say you want to add a test for dropout behavior:

```python
# In test_attention.py

def test_dropout_during_training(self):
    """Test that dropout is applied during training."""
    attention = MultiHeadAttention(
        input_dimension=64, output_dimension=64, context_length=128,
        dropout=0.5, number_of_heads=4
    )
    attention.train()  # Enable dropout
    
    x = torch.randn(2, 10, 64)
    output1 = attention(x)
    output2 = attention(x)
    
    # With dropout, outputs should differ (with high probability)
    assert not torch.allclose(output1, output2, atol=1e-6)
```

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'src'`**

**Solution:** Make sure you've installed the package in development mode:
```bash
pip install -e .
```

**Issue: Tests pass locally but fail in CI**

**Solution:** Check for:
- Random number generation without fixed seeds
- Device-specific code (CPU vs GPU)
- File system dependencies
- Missing dependencies in CI environment

**Issue: Tests are slow**

**Solution:**
- Use smaller model configurations in tests
- Run tests in parallel: `pytest tests/ -n auto`
- Use `pytest-xdist` for parallel execution
- Mark slow tests and skip them: `pytest tests/ -m "not slow"`

**Issue: CUDA/GPU errors**

**Solution:** Tests should work on CPU by default. If you need GPU tests:
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_functionality():
    # GPU test code
```

### Debugging Failed Tests

**Run with verbose output:**
```bash
pytest tests/test_model.py::TestGPTModel::test_forward_pass -v -s
```

**Run with Python debugger:**
```bash
pytest tests/test_model.py --pdb
```

**Show local variables on failure:**
```bash
pytest tests/test_model.py -l
```

**Print statements during test:**
```bash
pytest tests/test_model.py -s
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Example GitHub Actions configuration:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install -e .
      - run: pytest tests/ --cov=src --cov-report=xml
```

## Contributing

When contributing new features:

1. **Write tests first** (TDD approach) or alongside your code
2. **Ensure all tests pass** before submitting PR
3. **Maintain or improve coverage** - aim for >80% coverage
4. **Follow existing test patterns** for consistency
5. **Document complex test logic** with comments

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [PyTorch Testing Best Practices](https://pytorch.org/docs/stable/testing.html)
- [Python Testing Guide](https://realpython.com/python-testing/)

---

**Note:** These tests are designed to be both functional and educational. Reading through the tests can help you understand how each component of the GPT model works!
