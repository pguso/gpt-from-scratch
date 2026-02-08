"""
Tests for transformer blocks.
"""

import pytest
import torch
from src.model.blocks import LayerNorm, GELU, FeedForward


class TestBlocks:
    """Test transformer building blocks."""
    
    def test_layer_norm(self):
        """Test layer normalization."""
        norm = LayerNorm(64)
        x = torch.randn(2, 10, 64)
        output = norm(x)
        
        assert output.shape == x.shape
    
    def test_gelu(self):
        """Test GELU activation."""
        gelu = GELU()
        x = torch.randn(2, 10, 64)
        output = gelu(x)
        
        assert output.shape == x.shape
    
    def test_feedforward(self):
        """Test feed-forward network."""
        ff = FeedForward(64)
        x = torch.randn(2, 10, 64)
        output = ff(x)
        
        assert output.shape == x.shape
