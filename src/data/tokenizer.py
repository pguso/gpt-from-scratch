"""
Tokenizer utilities for GPT.
"""

import tiktoken


def get_tokenizer(encoding_name: str = "gpt2"):
    """
    Get a tokenizer.
    
    Args:
        encoding_name: Name of the encoding to use
        
    Returns:
        Tokenizer instance
    """
    return tiktoken.get_encoding(encoding_name)
