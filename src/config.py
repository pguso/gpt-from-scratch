"""
Configuration classes for GPT model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """Configuration for GPT model architecture."""
    
    vocab_size: int = 50257
    context_length: int = 1024
    embedding_dimension: int = 768
    number_of_heads: int = 12
    number_of_layers: int = 12
    dropout_rate: float = 0.1
    query_key_value_bias: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.embedding_dimension % self.number_of_heads == 0, \
            f"embedding_dimension ({self.embedding_dimension}) must be divisible by number_of_heads ({self.number_of_heads})"
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dimension": self.embedding_dimension,
            "number_of_heads": self.number_of_heads,
            "number_of_layers": self.number_of_layers,
            "dropout_rate": self.dropout_rate,
            "query_key_value_bias": self.query_key_value_bias
        }
