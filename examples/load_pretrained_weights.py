"""
Example: Load pretrained weights from a checkpoint.

This script demonstrates how to load a trained model checkpoint and use it
for inference. It shows the complete process of:
1. Loading the checkpoint file
2. Extracting the model configuration
3. Creating the model with the correct architecture
4. Loading the pretrained weights
5. Using the model for text generation

Usage:
    python examples/load_pretrained_weights.py --checkpoint checkpoints/best_model.pt
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


def load_pretrained_model(checkpoint_path, device='cpu'):
    """
    Load a pretrained model from a checkpoint file.
    
    This function demonstrates the complete process of loading pretrained weights:
    1. Load the checkpoint dictionary from disk
    2. Extract the model configuration
    3. Create a model instance with the correct architecture
    4. Load the pretrained weights into the model
    5. Set the model to evaluation mode
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt file)
        device: Device to load the model on ('cpu', 'cuda', or 'mps')
    
    Returns:
        model: Loaded GPT model ready for inference
        checkpoint: Dictionary containing checkpoint information
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Step 1: Load the checkpoint file
    # map_location ensures the checkpoint is loaded to the specified device
    # weights_only=True provides security by only loading tensors
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    print("✓ Checkpoint loaded successfully")
    
    # Step 2: Extract model configuration
    # The checkpoint contains a 'config' key with the model architecture parameters
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        
        # Handle backward compatibility: old checkpoints may have 'query_key_value_bias'
        # instead of 'use_attention_bias'
        if 'use_attention_bias' not in checkpoint_config and 'query_key_value_bias' in checkpoint_config:
            checkpoint_config['use_attention_bias'] = checkpoint_config.pop('query_key_value_bias')
            print("  Note: Converted old parameter name 'query_key_value_bias' to 'use_attention_bias'")
        
        # Create ModelConfig from the checkpoint's config dictionary
        config = ModelConfig(**checkpoint_config)
        print(f"✓ Model configuration loaded:")
        print(f"  - Vocabulary size: {config.vocab_size}")
        print(f"  - Context length: {config.context_length}")
        print(f"  - Embedding dimension: {config.embedding_dimension}")
        print(f"  - Number of heads: {config.number_of_heads}")
        print(f"  - Number of layers: {config.number_of_layers}")
        print(f"  - Dropout rate: {config.dropout_rate}")
    else:
        # Fallback to default config if not in checkpoint
        print("  Warning: Config not found in checkpoint, using defaults")
        config = ModelConfig()
    
    # Step 3: Create the model with the correct architecture
    # The model architecture must match the saved weights exactly
    print("\nCreating model with loaded configuration...")
    model = GPTModel(config)
    print("✓ Model created")
    
    # Step 4: Load the pretrained weights
    # model_state_dict contains all the learned parameters (weights and biases)
    print("\nLoading pretrained weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Weights loaded successfully")
    
    # Step 5: Move model to the specified device and set to evaluation mode
    # .to(device) moves all model parameters to the specified device
    # .eval() disables dropout and batch normalization updates (important for inference)
    model = model.to(device)
    model.eval()
    print(f"✓ Model moved to {device} and set to evaluation mode")
    
    # Display checkpoint metadata if available
    print("\nCheckpoint information:")
    if 'epoch' in checkpoint:
        print(f"  - Trained for {checkpoint['epoch']} epochs")
    if 'train_loss' in checkpoint and checkpoint['train_loss'] is not None:
        print(f"  - Training loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
        print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, checkpoint


def main():
    """Main function demonstrating pretrained weight loading."""
    parser = argparse.ArgumentParser(
        description="Load pretrained weights from a checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and use a checkpoint
  python examples/load_pretrained_weights.py --checkpoint checkpoints/best_model.pt
  
  # Load on GPU
  python examples/load_pretrained_weights.py --checkpoint checkpoints/best_model.pt --device cuda
  
  # Load and generate text
  python examples/load_pretrained_weights.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file (.pt file)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu", "mps"],
        help="Device to load the model on"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional: Generate text from this prompt after loading"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=50,
        help="Number of tokens to generate (if --prompt is provided)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation"
    )
    parser.add_argument(
        "--top-k-tokens",
        type=int,
        default=50,
        help="Top-k sampling parameter for generation"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        print("\nTo train a model and create a checkpoint, run:")
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
    
    print("=" * 70)
    print("Loading Pretrained Weights Example")
    print("=" * 70)
    print()
    
    # Load the pretrained model
    model, checkpoint = load_pretrained_model(args.checkpoint, device)
    
    print("\n" + "=" * 70)
    print("Model loaded successfully! Ready for inference.")
    print("=" * 70)
    
    # Optional: Generate text if prompt is provided
    if args.prompt:
        print(f"\nGenerating text from prompt: '{args.prompt}'")
        print("-" * 70)
        
        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # Tokenize prompt
        input_ids = tokenizer.encode(args.prompt)
        
        # Generate text
        output_ids = generate_text(
            model,
            input_ids,
            maximum_new_tokens=args.length,
            temperature=args.temperature,
            top_k_tokens=args.top_k_tokens if args.top_k_tokens > 0 else None
        )
        
        # Decode and print
        output_text = tokenizer.decode(output_ids)
        print(output_text)
        print("-" * 70)
    
    print("\nYou can now use the loaded model for inference:")
    print("  - model(input_ids)  # Forward pass")
    print("  - generate_text(model, input_ids, ...)  # Text generation")
    print("\nNote: The model is in evaluation mode (model.eval())")
    print("      This disables dropout and batch normalization updates.")


if __name__ == "__main__":
    main()
