"""
Tests for text generation.
"""

import pytest
import torch
from src.config import GPTConfig
from src.model.gpt import GPTModel
from src.generation.generate import generate_text


class TestGeneration:
    """Test text generation."""
    
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
