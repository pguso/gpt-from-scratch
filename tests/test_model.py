"""
Tests for complete GPT model.
"""

import pytest
import torch
from src.config import GPTConfig
from src.model.gpt import GPTModel


class TestGPTModel:
    """Test complete GPT model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        config = GPTConfig(
            vocab_size=1000,
            embedding_dimension=128,
            number_of_heads=4,
            number_of_layers=2,
            context_length=64
        )
        
        model = GPTModel(config)
        assert model is not None
    
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
