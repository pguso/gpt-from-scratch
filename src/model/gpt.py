"""
Complete GPT model implementation.
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .blocks import LayerNorm, FeedForward
from ..config import GPTConfig


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        config_dict = config.to_dict()
        
        self.attention = MultiHeadAttention(
            input_dimension=config_dict["embedding_dimension"],
            output_dimension=config_dict["embedding_dimension"],
            context_length=config_dict["context_length"],
            number_of_heads=config_dict["number_of_heads"],
            dropout=config_dict["dropout_rate"],
            query_key_value_bias=config_dict["query_key_value_bias"]
        )
        
        self.feed_forward = FeedForward(config_dict["embedding_dimension"])
        self.norm1 = LayerNorm(config_dict["embedding_dimension"])
        self.norm2 = LayerNorm(config_dict["embedding_dimension"])
        self.dropout = nn.Dropout(config_dict["dropout_rate"])
    
    def forward(self, x):
        # Attention block with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward block with residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class GPTModel(nn.Module):
    """Complete GPT model."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        config_dict = config.to_dict()
        
        # Embeddings
        self.token_embedding = nn.Embedding(config_dict["vocab_size"], config_dict["embedding_dimension"])
        self.position_embedding = nn.Embedding(config_dict["context_length"], config_dict["embedding_dimension"])
        self.embedding_dropout = nn.Dropout(config_dict["dropout_rate"])
        
        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config_dict["number_of_layers"])]
        )
        
        # Output layers
        self.final_norm = LayerNorm(config_dict["embedding_dimension"])
        self.lm_head = nn.Linear(config_dict["embedding_dimension"], config_dict["vocab_size"], bias=False)
    
    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape
        
        # Embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_embeddings = self.position_embedding(position_ids)
        
        x = token_embeddings + position_embeddings
        x = self.embedding_dropout(x)
        
        # Transformer blocks
        x = self.transformer_blocks(x)
        
        # Output
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
