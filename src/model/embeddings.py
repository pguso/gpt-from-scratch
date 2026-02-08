"""
Embedding layers for GPT.
"""

import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, embedding_dimension: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
    
    def forward(self, x):
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """Positional embedding layer."""
    
    def __init__(self, context_length: int, embedding_dimension: int):
        super().__init__()
        self.embedding = nn.Embedding(context_length, embedding_dimension)
    
    def forward(self, positions):
        return self.embedding(positions)
