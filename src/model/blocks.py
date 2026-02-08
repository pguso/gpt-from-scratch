"""
Transformer block components.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.shift = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized + self.shift


class GELU(nn.Module):
    """GELU activation function."""
    
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
                (x + 0.044715 * x.pow(3))
            )
        )


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, embedding_dimension: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dimension, 4 * embedding_dimension),
            GELU(),
            nn.Linear(4 * embedding_dimension, embedding_dimension)
        )
    
    def forward(self, x):
        return self.net(x)
