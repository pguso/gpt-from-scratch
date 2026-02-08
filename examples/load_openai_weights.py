"""
Example: Load OpenAI GPT-2 pretrained weights from Hugging Face.

This script demonstrates how to load OpenAI's pretrained GPT-2 weights from
Hugging Face and convert them to work with this codebase's model architecture.

The script handles:
1. Downloading GPT-2 weights from Hugging Face
2. Converting weight names from Hugging Face format to this codebase's format
3. Splitting combined QKV projections into separate Q, K, V matrices
4. Mapping layer normalization parameters correctly
5. Loading the converted weights into the model

Usage:
    # Load GPT-2 small (117M parameters)
    python examples/load_openai_weights.py --model-size small
    
    # Load GPT-2 medium (345M parameters)
    python examples/load_openai_weights.py --model-size medium
    
    # Load and generate text
    python examples/load_openai_weights.py --model-size small --prompt "Hello, world"
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

# Try to import transformers, provide helpful error if not available
try:
    from transformers import GPT2LMHeadModel
except ImportError:
    print("Error: transformers library not found.")
    print("Install it with: pip install transformers")
    sys.exit(1)


# GPT-2 model configurations
GPT2_CONFIGS = {
    'small': {
        'vocab_size': 50257,
        'context_length': 1024,
        'embedding_dimension': 768,
        'number_of_heads': 12,
        'number_of_layers': 12,
        'dropout_rate': 0.1,
        'use_attention_bias': False
    },
    'medium': {
        'vocab_size': 50257,
        'context_length': 1024,
        'embedding_dimension': 1024,
        'number_of_heads': 16,
        'number_of_layers': 24,
        'dropout_rate': 0.1,
        'use_attention_bias': False
    },
    'large': {
        'vocab_size': 50257,
        'context_length': 1024,
        'embedding_dimension': 1280,
        'number_of_heads': 20,
        'number_of_layers': 36,
        'dropout_rate': 0.1,
        'use_attention_bias': False
    },
    'xl': {
        'vocab_size': 50257,
        'context_length': 1024,
        'embedding_dimension': 1600,
        'number_of_heads': 25,
        'number_of_layers': 48,
        'dropout_rate': 0.1,
        'use_attention_bias': False
    }
}


def convert_hf_to_custom(hf_state_dict, config):
    """
    Convert Hugging Face GPT-2 weights to this codebase's format.
    
    IMPORTANT: Hugging Face GPT-2 uses Conv1D layers which store weights as
    [in_features, out_features], while PyTorch Linear uses [out_features, in_features].
    This means we need to transpose ALL weight matrices.

    Args:
        hf_state_dict: Hugging Face model state dictionary
        config: ModelConfig object

    Returns:
        custom_state_dict: Converted state dictionary for this codebase
    """
    custom_state_dict = {}
    embedding_dim = config.embedding_dimension
    num_heads = config.number_of_heads
    num_layers = config.number_of_layers

    print("Converting weights from Hugging Face format...")

    # 1. Token embeddings (no transpose needed - it's an Embedding layer)
    if 'transformer.wte.weight' in hf_state_dict:
        custom_state_dict['token_embedding.weight'] = hf_state_dict['transformer.wte.weight']
        print("  ✓ Token embeddings")

    # 2. Position embeddings (no transpose needed - it's an Embedding layer)
    if 'transformer.wpe.weight' in hf_state_dict:
        custom_state_dict['position_embedding.weight'] = hf_state_dict['transformer.wpe.weight']
        print("  ✓ Position embeddings")

    # 3. Transformer blocks
    for layer_idx in range(num_layers):
        prefix_hf = f'transformer.h.{layer_idx}'
        prefix_custom = f'transformer_blocks.{layer_idx}'

        # Layer norm 1 (before attention) - no transpose needed
        if f'{prefix_hf}.ln_1.weight' in hf_state_dict:
            custom_state_dict[f'{prefix_custom}.norm1.scale'] = hf_state_dict[f'{prefix_hf}.ln_1.weight']
            custom_state_dict[f'{prefix_custom}.norm1.shift'] = hf_state_dict[f'{prefix_hf}.ln_1.bias']

        # Attention: QKV projection
        # HF stores as Conv1D: [in_features, out_features] = [1024, 3072]
        # We need to:
        # 1. Transpose to [3072, 1024] to match nn.Linear format
        # 2. Split along dim 0 (first dimension) to get Q, K, V each [1024, 1024]
        if f'{prefix_hf}.attn.c_attn.weight' in hf_state_dict:
            qkv_weight = hf_state_dict[f'{prefix_hf}.attn.c_attn.weight']  # [1024, 3072]

            if layer_idx == 0:
                print(f"    Layer 0 - c_attn.weight original shape: {qkv_weight.shape}")

            # Transpose from Conv1D format [in, out] to Linear format [out, in]
            qkv_weight = qkv_weight.t()  # [3072, 1024]

            if layer_idx == 0:
                print(f"    Layer 0 - After transpose: {qkv_weight.shape}")

            # Split into Q, K, V along the first dimension
            q_weight, k_weight, v_weight = qkv_weight.split(embedding_dim, dim=0)

            if layer_idx == 0:
                print(f"    Layer 0 - Q: {q_weight.shape}, K: {k_weight.shape}, V: {v_weight.shape}")

            custom_state_dict[f'{prefix_custom}.attention.W_query.weight'] = q_weight
            custom_state_dict[f'{prefix_custom}.attention.W_key.weight'] = k_weight
            custom_state_dict[f'{prefix_custom}.attention.W_value.weight'] = v_weight

            # Handle bias if present
            if f'{prefix_hf}.attn.c_attn.bias' in hf_state_dict:
                qkv_bias = hf_state_dict[f'{prefix_hf}.attn.c_attn.bias']  # [3072]
                q_bias, k_bias, v_bias = qkv_bias.split(embedding_dim)
                custom_state_dict[f'{prefix_custom}.attention.W_query.bias'] = q_bias
                custom_state_dict[f'{prefix_custom}.attention.W_key.bias'] = k_bias
                custom_state_dict[f'{prefix_custom}.attention.W_value.bias'] = v_bias

        # Attention output projection
        # HF Conv1D: [in, out] = [1024, 1024]
        # Need to transpose to [out, in] = [1024, 1024] for nn.Linear
        if f'{prefix_hf}.attn.c_proj.weight' in hf_state_dict:
            custom_state_dict[f'{prefix_custom}.attention.out_proj.weight'] = \
                hf_state_dict[f'{prefix_hf}.attn.c_proj.weight'].t()
            if f'{prefix_hf}.attn.c_proj.bias' in hf_state_dict:
                custom_state_dict[f'{prefix_custom}.attention.out_proj.bias'] = \
                    hf_state_dict[f'{prefix_hf}.attn.c_proj.bias']

        # Layer norm 2 (before feed-forward) - no transpose needed
        if f'{prefix_hf}.ln_2.weight' in hf_state_dict:
            custom_state_dict[f'{prefix_custom}.norm2.scale'] = hf_state_dict[f'{prefix_hf}.ln_2.weight']
            custom_state_dict[f'{prefix_custom}.norm2.shift'] = hf_state_dict[f'{prefix_hf}.ln_2.bias']

        # Feed-forward network
        # First linear: HF Conv1D [in, out] = [1024, 4096]
        # Transpose to [out, in] = [4096, 1024]
        if f'{prefix_hf}.mlp.c_fc.weight' in hf_state_dict:
            custom_state_dict[f'{prefix_custom}.feed_forward.net.0.weight'] = \
                hf_state_dict[f'{prefix_hf}.mlp.c_fc.weight'].t()
            if f'{prefix_hf}.mlp.c_fc.bias' in hf_state_dict:
                custom_state_dict[f'{prefix_custom}.feed_forward.net.0.bias'] = \
                    hf_state_dict[f'{prefix_hf}.mlp.c_fc.bias']

        # Second linear: HF Conv1D [in, out] = [4096, 1024]
        # Transpose to [out, in] = [1024, 4096]
        if f'{prefix_hf}.mlp.c_proj.weight' in hf_state_dict:
            custom_state_dict[f'{prefix_custom}.feed_forward.net.2.weight'] = \
                hf_state_dict[f'{prefix_hf}.mlp.c_proj.weight'].t()
            if f'{prefix_hf}.mlp.c_proj.bias' in hf_state_dict:
                custom_state_dict[f'{prefix_custom}.feed_forward.net.2.bias'] = \
                    hf_state_dict[f'{prefix_hf}.mlp.c_proj.bias']

    # 4. Final layer norm - no transpose needed
    if 'transformer.ln_f.weight' in hf_state_dict:
        custom_state_dict['final_norm.scale'] = hf_state_dict['transformer.ln_f.weight']
        custom_state_dict['final_norm.shift'] = hf_state_dict['transformer.ln_f.bias']
        print("  ✓ Final layer norm")

    # 5. Language model head
    # This one is tricky: in HF GPT-2, lm_head often shares weights with token embeddings
    # and is stored as an Embedding, not Conv1D, so no transpose needed
    if 'lm_head.weight' in hf_state_dict:
        custom_state_dict['lm_head.weight'] = hf_state_dict['lm_head.weight']
        print("  ✓ Language model head")
    else:
        # If lm_head.weight doesn't exist, it's tied to token embeddings
        custom_state_dict['lm_head.weight'] = hf_state_dict['transformer.wte.weight']
        print("  ✓ Language model head (tied to token embeddings)")

    print(f"  ✓ Converted {len(custom_state_dict)} parameter tensors")
    print(f"  ✓ All Conv1D weights transposed to match nn.Linear format")

    return custom_state_dict


def load_openai_gpt2(model_size='small', device='cpu'):
    """
    Load OpenAI GPT-2 pretrained weights from Hugging Face.
    
    Args:
        model_size: One of 'small', 'medium', 'large', 'xl'
        device: Device to load model on
    
    Returns:
        model: Loaded GPT model with OpenAI weights
        config: ModelConfig used
    """
    if model_size not in GPT2_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(GPT2_CONFIGS.keys())}")
    
    print(f"Loading GPT-2 {model_size} from Hugging Face...")
    print("This will download the model weights on first run (~500MB for small).")
    
    # Map model size to Hugging Face model name
    hf_model_name = {
        'small': 'gpt2',
        'medium': 'gpt2-medium',
        'large': 'gpt2-large',
        'xl': 'gpt2-xl'
    }[model_size]
    
    # Load Hugging Face model
    print(f"\nDownloading {hf_model_name} from Hugging Face...")
    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name)
    hf_model.eval()
    
    # Get Hugging Face state dict
    hf_state_dict = hf_model.state_dict()
    print(f"✓ Loaded Hugging Face model ({len(hf_state_dict)} parameters)")
    
    # Create config for this codebase
    config_dict = GPT2_CONFIGS[model_size]
    config = ModelConfig(**config_dict)
    
    print(f"\nModel configuration:")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Context length: {config.context_length}")
    print(f"  - Embedding dimension: {config.embedding_dimension}")
    print(f"  - Number of heads: {config.number_of_heads}")
    print(f"  - Number of layers: {config.number_of_layers}")
    
    # Convert weights
    print("\nConverting weights to this codebase's format...")
    custom_state_dict = convert_hf_to_custom(hf_state_dict, config)
    
    # Create model
    print("\nCreating model...")
    model = GPTModel(config)
    
    # Load converted weights
    print("Loading converted weights...")
    missing_keys, unexpected_keys = model.load_state_dict(custom_state_dict, strict=False)
    
    if missing_keys:
        print(f"  Warning: {len(missing_keys)} missing keys (this is normal for some parameters)")
    if unexpected_keys:
        print(f"  Warning: {len(unexpected_keys)} unexpected keys")
    
    # Move to device and set to eval
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    return model, config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Load OpenAI GPT-2 pretrained weights from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load GPT-2 small
  python examples/load_openai_weights.py --model-size small
  
  # Load GPT-2 medium and generate text
  python examples/load_openai_weights.py --model-size medium --prompt "The future of AI"
  
  # Load on GPU
  python examples/load_openai_weights.py --model-size small --device cuda
        """
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xl"],
        help="GPT-2 model size to load"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu", "mps"],
        help="Device to load model on"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional: Generate text from this prompt"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=50,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--top-k-tokens",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    
    print("=" * 70)
    print("Loading OpenAI GPT-2 Pretrained Weights")
    print("=" * 70)
    print()
    
    # Load model
    model, config = load_openai_gpt2(args.model_size, device)
    
    print("\n" + "=" * 70)
    print("Model loaded successfully! Ready for inference.")
    print("=" * 70)
    
    # Generate text if prompt provided
    if args.prompt:
        print(f"\nGenerating text from prompt: '{args.prompt}'")
        print("-" * 70)
        
        tokenizer = tiktoken.get_encoding("gpt2")
        input_ids = tokenizer.encode(args.prompt)
        
        output_ids = generate_text(
            model,
            input_ids,
            maximum_new_tokens=args.length,
            temperature=args.temperature,
            top_k_tokens=args.top_k_tokens if args.top_k_tokens > 0 else None
        )
        
        output_text = tokenizer.decode(output_ids)
        print(output_text)
        print("-" * 70)
    else:
        print("\nTip: Add --prompt 'your text here' to generate text with the loaded model")


if __name__ == "__main__":
    main()
