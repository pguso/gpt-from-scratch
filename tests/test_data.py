"""
Tests for data processing.
"""

import pytest
import torch
from src.data.dataset import GPTDataset
from src.data.tokenizer import get_tokenizer


class TestDataProcessing:
    """Test data processing functionality."""
    
    def test_tokenization(self):
        """Test basic tokenization."""
        tokenizer = get_tokenizer()
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        tokenizer = get_tokenizer()
        text = "The quick brown fox jumps over the lazy dog."
        dataset = GPTDataset(text, tokenizer, maximum_length=8, stride=4)
        assert len(dataset) > 0
        
        input_ids, target_ids = dataset[0]
        assert len(input_ids) == 8
        assert len(target_ids) == 8
