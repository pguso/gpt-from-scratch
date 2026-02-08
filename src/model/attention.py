"""
Multi-head attention implementation.
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.
    """
    
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        context_length: int,
        dropout: float,
        number_of_heads: int,
        use_attention_bias: bool = False
    ):
        super().__init__()
        assert output_dimension % number_of_heads == 0, "output_dimension must be divisible by number_of_heads"
        
        self.output_dimension = output_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = output_dimension // number_of_heads
        
        # Query, Key, Value projections
        self.W_query = nn.Linear(input_dimension, output_dimension, bias=use_attention_bias)
        self.W_key = nn.Linear(input_dimension, output_dimension, bias=use_attention_bias)
        self.W_value = nn.Linear(input_dimension, output_dimension, bias=use_attention_bias)
        
        # Output projection
        self.out_proj = nn.Linear(output_dimension, output_dimension)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        
        # Linear projections
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Reshape for multi-head attention
        queries = self._split_heads(queries, batch_size, sequence_length)
        keys = self._split_heads(keys, batch_size, sequence_length)
        values = self._split_heads(values, batch_size, sequence_length)
        
        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(
            queries, keys, values, sequence_length
        )
        
        # Combine heads
        attention_output = self._combine_heads(attention_output, batch_size, sequence_length)
        
        return self.out_proj(attention_output)
    
    def _split_heads(self, tensor, batch_size, sequence_length):
        tensor = tensor.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension)
        return tensor.transpose(1, 2)
    
    def _combine_heads(self, tensor, batch_size, sequence_length):
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, sequence_length, self.output_dimension)
    
    def _scaled_dot_product_attention(self, queries, keys, values, sequence_length):
        attention_scores = queries @ keys.transpose(-2, -1)
        mask = self.mask[:sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        scaling_factor = self.head_dimension ** 0.5
        attention_weights = torch.softmax(attention_scores / scaling_factor, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ values
