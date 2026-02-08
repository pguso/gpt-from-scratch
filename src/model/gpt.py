"""
Complete GPT model implementation.

This module contains the main GPT model architecture, which consists of:
- Token and position embeddings
- Multiple transformer blocks (each with attention and feed-forward layers)
- Output layer that converts hidden states to vocabulary predictions
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .blocks import LayerNorm, FeedForward
from ..config import ModelConfig


class TransformerBlock(nn.Module):
    """
    Single transformer block.
    
    Each block processes the input sequence through:
    1. Self-attention: Allows each token to "look at" other tokens
    2. Feed-forward: Processes the attended information
    
    Uses pre-norm architecture (normalize before attention/FFN) which is more
    stable for deep networks than post-norm.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        config_dict = config.to_dict()
        
        # Self-attention layer: learns which tokens to pay attention to
        # Input and output dimensions are the same (embedding_dimension)
        self.attention = MultiHeadAttention(
            input_dimension=config_dict["embedding_dimension"],
            output_dimension=config_dict["embedding_dimension"],
            context_length=config_dict["context_length"],
            number_of_heads=config_dict["number_of_heads"],
            dropout=config_dict["dropout_rate"],
            use_attention_bias=config_dict["use_attention_bias"]
        )
        
        # Feed-forward network: processes information with expansion/contraction
        # Expands to 4x embedding_dimension, then contracts back
        self.feed_forward = FeedForward(config_dict["embedding_dimension"])
        
        # Layer normalization: stabilizes activations and helps with training
        # Applied before attention and feed-forward (pre-norm architecture)
        self.norm1 = LayerNorm(config_dict["embedding_dimension"])
        self.norm2 = LayerNorm(config_dict["embedding_dimension"])
        
        # Dropout: randomly zeros some activations during training to prevent overfitting
        self.dropout = nn.Dropout(config_dict["dropout_rate"])
    
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, embedding_dimension]
        
        Returns:
            Output tensor of same shape [batch_size, sequence_length, embedding_dimension]
        """
        # Attention block with residual connection
        # Residual connection helps gradients flow through many layers
        residual = x  # Save original input
        x = self.norm1(x)              # Normalize first (pre-norm)
        x = self.attention(x)          # Apply self-attention
        x = self.dropout(x)            # Apply dropout during training
        x = x + residual               # Add residual (skip connection)
        
        # Feed-forward block with residual connection
        residual = x  # Save input to feed-forward block
        x = self.norm2(x)              # Normalize first (pre-norm)
        x = self.feed_forward(x)       # Apply feed-forward network
        x = self.dropout(x)            # Apply dropout during training
        x = x + residual               # Add residual (skip connection)
        
        return x


class GPTModel(nn.Module):
    """
    Complete GPT model.
    
    Architecture:
    1. Embeddings: Convert token IDs to dense vectors + add position information
    2. Transformer blocks: Process sequence through multiple layers
    3. Output layer: Convert hidden states to vocabulary predictions (logits)
    
    Input: Token IDs [batch_size, sequence_length]
    Output: Logits [batch_size, sequence_length, vocab_size]
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        config_dict = config.to_dict()
        
        # Token embeddings: Convert token IDs (integers) to dense vectors
        # Each token ID maps to a learned vector of size embedding_dimension
        # Shape: [vocab_size, embedding_dimension]
        self.token_embedding = nn.Embedding(
            config_dict["vocab_size"], 
            config_dict["embedding_dimension"]
        )
        
        # Position embeddings: Encode position of each token in the sequence
        # Each position (0, 1, 2, ...) gets a learned vector
        # Shape: [context_length, embedding_dimension]
        # Note: Only context_length positions are supported (model can't see beyond this)
        self.position_embedding = nn.Embedding(
            config_dict["context_length"], 
            config_dict["embedding_dimension"]
        )
        
        # Dropout on embeddings: Helps prevent overfitting
        self.embedding_dropout = nn.Dropout(config_dict["dropout_rate"])
        
        # Transformer blocks: Stack of identical blocks that process the sequence
        # Each block refines the understanding of the sequence
        # More blocks = deeper model = can learn more complex patterns
        # Using nn.Sequential for cleaner code (all blocks share same config)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config_dict["number_of_layers"])]
        )
        
        # Final layer normalization: Normalize before output layer
        # Helps stabilize the final hidden states
        self.final_norm = LayerNorm(config_dict["embedding_dimension"])
        
        # Language model head: Converts hidden states to vocabulary predictions
        # Projects from embedding_dimension to vocab_size (one score per token)
        # No bias term (standard in GPT models) - reduces parameters slightly
        # Outputs "logits" (raw scores), not probabilities
        self.lm_head = nn.Linear(
            config_dict["embedding_dimension"], 
            config_dict["vocab_size"], 
            bias=False
        )
    
    def forward(self, input_ids):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs tensor of shape [batch_size, sequence_length]
                      Values must be in range [0, vocab_size)
                      Sequence length cannot exceed context_length
        
        Returns:
            logits: Raw scores for next token prediction
                    Shape: [batch_size, sequence_length, vocab_size]
                    Each position has scores for all vocabulary tokens
                    Higher score = more likely next token (but not a probability yet)
        """
        batch_size, sequence_length = input_ids.shape
        
        # ===== EMBEDDINGS =====
        # Convert token IDs to dense vectors
        # Shape: [batch_size, sequence_length] -> [batch_size, sequence_length, embedding_dimension]
        token_embeddings = self.token_embedding(input_ids)
        
        # Create position IDs: [0, 1, 2, ..., sequence_length-1]
        # Must be on same device as input_ids (CPU or GPU)
        position_ids = torch.arange(sequence_length, device=input_ids.device)
        
        # Get position embeddings for each position
        # Shape: [sequence_length] -> [sequence_length, embedding_dimension]
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine token and position embeddings by addition
        # Broadcasting: [batch, seq, dim] + [seq, dim] -> [batch, seq, dim]
        # Each token now has both its meaning (token embedding) and position (position embedding)
        x = token_embeddings + position_embeddings
        
        # Apply dropout to embeddings (only active during training)
        x = self.embedding_dropout(x)
        
        # ===== TRANSFORMER BLOCKS =====
        # Process sequence through all transformer blocks
        # Each block refines the understanding:
        #   - Early blocks: Learn simple patterns (syntax, local relationships)
        #   - Later blocks: Learn complex patterns (semantics, long-range dependencies)
        # Shape remains: [batch_size, sequence_length, embedding_dimension]
        x = self.transformer_blocks(x)
        
        # ===== OUTPUT LAYER =====
        # Normalize final hidden states (stabilizes output)
        x = self.final_norm(x)
        
        # Project to vocabulary size to get logits (raw scores)
        # Shape: [batch_size, sequence_length, embedding_dimension] 
        #     -> [batch_size, sequence_length, vocab_size]
        # Each position now has a score for every possible next token
        logits = self.lm_head(x)
        
        return logits
