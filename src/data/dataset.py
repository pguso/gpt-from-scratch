"""
Dataset implementation for GPT training.
"""

import torch
from torch.utils.data import Dataset
from typing import List


class GPTDataset(Dataset):
    """
    Dataset for GPT training using sliding window approach.
    """
    
    def __init__(self, text: str, tokenizer, maximum_length: int, stride: int):
        """
        Args:
            text: Raw text to create dataset from
            tokenizer: Tokenizer for encoding text
            maximum_length: Maximum sequence length
            stride: Step size for sliding window
        """
        # Tokenize entire text once
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        # Create sliding window sequences
        self.input_ids = []
        self.target_ids = []
        
        for i in range(0, len(token_ids) - maximum_length, stride):
            input_chunk = token_ids[i:i + maximum_length]
            target_chunk = token_ids[i + 1:i + maximum_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
