"""
Quickstart example for GPT From Scratch.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.gpt import GPTModel
from src.config import GPTConfig
from src.generation.generate import generate_text
import tiktoken


def main():
    # Create a small model
    config = GPTConfig(
        vocab_size=50257,
        embedding_dimension=256,
        number_of_heads=4,
        number_of_layers=4,
        context_length=128
    )
    
    print("Creating GPT model...")
    model = GPTModel(config)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Generate text (untrained model = gibberish)
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello, I am"
    
    print(f"\nPrompt: {prompt}")
    print("Generating text...\n")
    
    input_ids = tokenizer.encode(prompt)
    output_ids = generate_text(model, input_ids, maximum_new_tokens=20)
    output_text = tokenizer.decode(output_ids)
    
    print(f"Generated: {output_text}")


if __name__ == "__main__":
    main()
