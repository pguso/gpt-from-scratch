"""
Tests for attention mechanism.
"""

import pytest
import torch
from src.model.attention import MultiHeadAttention


class TestAttention:
    """Test attention mechanism."""
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        attention = MultiHeadAttention(
            input_dimension=64, output_dimension=64, context_length=128,
            dropout=0.1, number_of_heads=4
        )
        
        x = torch.randn(2, 10, 64)
        output = attention(x)
        
        assert output.shape == x.shape
    
    def test_causal_masking(self):
        """Test that causal masking is applied."""
        attention = MultiHeadAttention(
            input_dimension=64, output_dimension=64, context_length=128,
            dropout=0.0, number_of_heads=1
        )
        
        # Check mask exists
        assert hasattr(attention, 'mask')
        assert attention.mask.shape[0] == 128
