"""
Example: Generate text with a trained model.

This script loads a trained model checkpoint and generates text from a prompt.
"""

import argparse
import torch
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.gpt import GPTModel
from src.config import ModelConfig
from src.generation.generate import generate_text
import tiktoken


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint file")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt to start generation")
    parser.add_argument("--length", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top-k-tokens", type=int, default=50,
                        help="Top-k sampling (None to disable)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cuda", "cpu", "mps"],
                        help="Device to run on")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        print("\nTo train a model, run:")
        print("  python examples/train_tiny_stories.py")
        return
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    # Recreate config from checkpoint
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        
        # Handle backward compatibility: old checkpoints may have 'query_key_value_bias'
        # instead of 'use_attention_bias'
        if 'use_attention_bias' not in checkpoint_config and 'query_key_value_bias' in checkpoint_config:
            checkpoint_config['use_attention_bias'] = checkpoint_config.pop('query_key_value_bias')
            print("Note: Converted old parameter name 'query_key_value_bias' to 'use_attention_bias'")
        
        config = ModelConfig(**checkpoint_config)
    else:
        # Fallback to default config if not in checkpoint
        print("Warning: Config not found in checkpoint, using defaults")
        config = ModelConfig()
    
    # Create model
    print("Creating model...")
    model = GPTModel(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Tokenize prompt
    print(f"\nPrompt: '{args.prompt}'")
    input_ids = tokenizer.encode(args.prompt)
    
    # Generate
    print(f"Generating {args.length} tokens (temperature={args.temperature}, top_k_tokens={args.top_k_tokens})...")
    print("-" * 60)
    
    output_ids = generate_text(
        model,
        input_ids,
        maximum_new_tokens=args.length,
        temperature=args.temperature,
        top_k_tokens=args.top_k_tokens if args.top_k_tokens > 0 else None
    )
    
    output_text = tokenizer.decode(output_ids)
    
    print(output_text)
    print("-" * 60)


if __name__ == "__main__":
    main()
