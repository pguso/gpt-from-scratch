"""
Example: Analyze attention patterns in GPT model.

This script visualizes and analyzes attention patterns across different layers
and heads of a GPT model. It shows which tokens attend to which other tokens,
helping understand what the model is "looking at" when processing text.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.gpt import GPTModel
from src.config import GPTConfig


class AttentionHook:
    """Hook to capture attention weights from MultiHeadAttention module."""
    
    def __init__(self):
        self.attention_weights = []
        self.layer_idx = 0
    
    def __call__(self, module, input, output):
        """Capture attention weights during forward pass."""
        # The attention module computes weights internally
        # We need to manually compute them here
        x = input[0]  # Input tensor
        
        # Get Q, K, V
        queries = module.W_query(x)
        keys = module.W_key(x)
        values = module.W_value(x)
        
        batch_size, num_tokens, _ = x.shape
        
        # Split into heads
        queries = queries.view(batch_size, num_tokens, module.number_of_heads, module.head_dimension)
        queries = queries.transpose(1, 2)  # [batch, heads, tokens, head_dimension]
        
        keys = keys.view(batch_size, num_tokens, module.number_of_heads, module.head_dimension)
        keys = keys.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = queries @ keys.transpose(-2, -1)  # [batch, heads, tokens, tokens]
        
        # Apply mask
        mask = module.mask[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply scaling and softmax
        scaling_factor = module.head_dimension ** 0.5
        attention_weights = torch.softmax(attention_scores / scaling_factor, dim=-1)
        
        # Store weights (remove batch dimension, take first sample)
        self.attention_weights.append(attention_weights[0].detach().cpu().numpy())


def extract_attention_weights(model, input_ids, device='cpu'):
    """
    Extract attention weights from all layers of the model.
    
    Args:
        model: GPT model
        input_ids: Input token indices [batch_size, sequence_length]
        device: Device to run on
    
    Returns:
        List of attention weight matrices, one per layer
        Each matrix has shape [number_of_heads, sequence_length, sequence_length]
    """
    model.eval()
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    # Create hooks for all attention layers
    hooks = []
    attention_hooks = []
    
    # Register hooks on all transformer blocks
    for i, block in enumerate(model.transformer_blocks):
        hook = AttentionHook()
        hook.layer_idx = i
        attention_hooks.append(hook)
        
        # Register forward hook
        handle = block.attention.register_forward_hook(hook)
        hooks.append(handle)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)
    
    # Extract attention weights
    all_weights = []
    for hook in attention_hooks:
        if len(hook.attention_weights) > 0:
            all_weights.append(hook.attention_weights[0])
        else:
            raise RuntimeError(f"No attention weights captured for layer {hook.layer_idx}")
    
    # Remove hooks
    for handle in hooks:
        handle.remove()
    
    return all_weights


def visualize_attention_head(attention_weights, tokens, layer_idx, head_idx, ax=None, title=None):
    """
    Visualize attention weights for a single head.
    
    Args:
        attention_weights: Attention weight matrix [sequence_length, sequence_length]
        tokens: List of token strings
        layer_idx: Layer index
        head_idx: Head index
        ax: Matplotlib axis (if None, creates new figure)
        title: Optional title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    # Title
    if title is None:
        title = f'Layer {layer_idx}, Head {head_idx}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Key (Attended To)', fontsize=10)
    ax.set_ylabel('Query (Attending From)', fontsize=10)
    
    # Add grid
    ax.grid(False)
    
    return ax


def visualize_all_heads(attention_weights, tokens, layer_idx, max_heads=None):
    """
    Visualize all heads for a layer in a grid.
    
    Args:
        attention_weights: Attention weights [number_of_heads, sequence_length, sequence_length]
        tokens: List of token strings
        layer_idx: Layer index
        max_heads: Maximum number of heads to visualize (None for all)
    """
    number_of_heads = attention_weights.shape[0]
    if max_heads:
        number_of_heads = min(number_of_heads, max_heads)
    
    # Calculate grid size
    cols = 4
    rows = (number_of_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for head_idx in range(number_of_heads):
        row = head_idx // cols
        col = head_idx % cols
        ax = axes[row, col]
        
        visualize_attention_head(
            attention_weights[head_idx],
            tokens,
            layer_idx,
            head_idx,
            ax=ax
        )
    
    # Hide unused subplots
    for head_idx in range(number_of_heads, rows * cols):
        row = head_idx // cols
        col = head_idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'All Attention Heads - Layer {layer_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def analyze_attention_patterns(attention_weights, tokens, top_k_tokens=5):
    """
    Analyze attention patterns and print interesting findings.
    
    Args:
        attention_weights: List of attention weight matrices (one per layer)
        tokens: List of token strings
        top_k_tokens: Number of top attention targets to show
    """
    print("\n" + "="*60)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*60)
    
    for layer_idx, layer_weights in enumerate(attention_weights):
        # Average across all heads
        average_weights = layer_weights.mean(axis=0)  # [sequence_length, sequence_length]
        
        print(f"\nLayer {layer_idx}:")
        print("-" * 60)
        
        # Find tokens with highest attention to other tokens
        for token_idx, token in enumerate(tokens):
            # Get attention from this token to all others
            attention_from_token = average_weights[token_idx, :token_idx+1]  # Only previous tokens (causal)
            
            if len(attention_from_token) > 1:  # Not just self-attention
                # Get top-k attended tokens
                top_indices = np.argsort(attention_from_token)[-top_k_tokens:][::-1]
                top_weights = attention_from_token[top_indices]
                
                print(f"  '{token}' attends most to:")
                for i, (idx, weight) in enumerate(zip(top_indices, top_weights)):
                    if idx < len(tokens):
                        print(f"    {i+1}. '{tokens[idx]}' (weight: {weight:.3f})")


def plot_attention_flow(attention_weights, tokens, layer_indices=None):
    """
    Plot how attention flows through layers.
    
    Args:
        attention_weights: List of attention weight matrices
        tokens: List of token strings
        layer_indices: Which layers to plot (None for all)
    """
    if layer_indices is None:
        layer_indices = range(len(attention_weights))
    
    # Average across heads for each layer
    average_weights = [w.mean(axis=0) for w in attention_weights]
    
    # Plot average self-attention across layers
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = list(layer_indices)
    # Average self-attention (diagonal) across all tokens for each layer
    self_attention = [np.mean(np.diag(average_weights[l])) for l in layers]
    
    ax.plot(layers, self_attention, marker='o', linewidth=2, markersize=8, label='Average Self-Attention')
    
    # Also plot average attention to previous tokens
    previous_attention = []
    for l in layers:
        # Get attention to previous tokens (lower triangle, excluding diagonal)
        weights = average_weights[l]
        mask = np.tril(np.ones_like(weights), k=-1).astype(bool)
        previous_attention.append(np.mean(weights[mask]) if np.any(mask) else 0.0)
    
    ax.plot(layers, previous_attention, marker='s', linewidth=2, markersize=8, label='Average Attention to Previous Tokens')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Attention Patterns Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    
    plt.tight_layout()
    return fig


def main():
    """Main function to analyze attention patterns."""
    parser = argparse.ArgumentParser(description="Analyze attention patterns in GPT model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (None for untrained model)")
    parser.add_argument("--embedding-dimension", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--number-of-heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--number-of-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--context-length", type=int, default=128,
                        help="Context length")
    
    # Input arguments
    parser.add_argument("--text", type=str, default="The cat sat on the mat. It was fluffy.",
                        help="Text to analyze")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cuda", "cpu", "mps"],
                        help="Device to run on")
    
    # Visualization arguments
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to visualize (None for all)")
    parser.add_argument("--head", type=int, default=None,
                        help="Specific head to visualize (None for all)")
    parser.add_argument("--max-heads", type=int, default=None,
                        help="Maximum number of heads to visualize per layer")
    parser.add_argument("--output-dir", type=str, default="attention_plots",
                        help="Directory to save plots")
    parser.add_argument("--show-plots", action="store_true",
                        help="Show plots interactively")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    
    # Tokenize input
    print(f"\nInput text: '{args.text}'")
    token_ids = tokenizer.encode(args.text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    if len(tokens) > args.context_length:
        print(f"Warning: Input length ({len(tokens)}) exceeds context length ({args.context_length})")
        tokens = tokens[:args.context_length]
        token_ids = token_ids[:args.context_length]
    
    # Load checkpoint if provided to get config
    checkpoint = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        
        # Try to load config from checkpoint
        if 'config' in checkpoint:
            print("Found config in checkpoint, using checkpoint config...")
            checkpoint_config = checkpoint['config']
            checkpoint_vocab_size = checkpoint_config.get('vocab_size', vocab_size)
            if checkpoint_vocab_size != vocab_size:
                print(f"  Warning: Checkpoint vocab_size ({checkpoint_vocab_size}) differs from tokenizer vocab_size ({vocab_size})")
                print(f"  Using checkpoint vocab_size: {checkpoint_vocab_size}")
            config = GPTConfig(
                vocab_size=checkpoint_vocab_size,
                context_length=checkpoint_config.get('context_length', args.context_length),
                embedding_dimension=checkpoint_config.get('embedding_dimension', args.embedding_dimension),
                number_of_heads=checkpoint_config.get('number_of_heads', args.number_of_heads),
                number_of_layers=checkpoint_config.get('number_of_layers', args.number_of_layers),
                dropout_rate=0.0,  # Disable dropout for analysis
                query_key_value_bias=checkpoint_config.get('query_key_value_bias', False)
            )
            print(f"  Config from checkpoint: d_model={config.embedding_dimension}, "
                  f"n_layers={config.number_of_layers}, n_heads={config.number_of_heads}")
        else:
            print("No config found in checkpoint, using command-line arguments...")
            config = GPTConfig(
                vocab_size=vocab_size,
                context_length=args.context_length,
                embedding_dimension=args.embedding_dimension,
                number_of_heads=args.number_of_heads,
                number_of_layers=args.number_of_layers,
                dropout_rate=0.0  # Disable dropout for analysis
            )
    else:
        # Create model with command-line arguments
        print("\nCreating model...")
        config = GPTConfig(
            vocab_size=vocab_size,
            context_length=args.context_length,
            embedding_dimension=args.embedding_dimension,
            number_of_heads=args.number_of_heads,
            number_of_layers=args.number_of_layers,
            dropout_rate=0.0  # Disable dropout for analysis
        )
    
    # Create model
    print("\nCreating model...")
    model = GPTModel(config)
    
    # Load model weights from checkpoint if provided
    if checkpoint is not None:
        print("Loading model weights from checkpoint...")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")
    else:
        print("Using untrained model (random weights)")
    
    # Prepare input
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    
    # Extract attention weights
    print("\nExtracting attention weights...")
    all_attention_weights = extract_attention_weights(model, input_ids, device)
    
    print(f"Extracted attention from {len(all_attention_weights)} layers")
    print(f"Shape per layer: {all_attention_weights[0].shape}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze patterns
    analyze_attention_patterns(all_attention_weights, tokens, top_k_tokens=3)
    
    # Visualize
    layers_to_plot = [args.layer] if args.layer is not None else range(len(all_attention_weights))
    
    for layer_idx in layers_to_plot:
        if layer_idx >= len(all_attention_weights):
            continue
        
        attention_weights = all_attention_weights[layer_idx]  # [number_of_heads, sequence_length, sequence_length]
        
        if args.head is not None:
            # Visualize single head
            fig, ax = plt.subplots(figsize=(12, 10))
            visualize_attention_head(
                attention_weights[args.head],
                tokens,
                layer_idx,
                args.head,
                ax=ax
            )
            filename = f"layer_{layer_idx}_head_{args.head}.png"
            filepath = os.path.join(args.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
            if args.show_plots:
                plt.show()
            else:
                plt.close()
        else:
            # Visualize all heads
            fig = visualize_all_heads(attention_weights, tokens, layer_idx, max_heads=args.max_heads)
            filename = f"layer_{layer_idx}_all_heads.png"
            filepath = os.path.join(args.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
            if args.show_plots:
                plt.show()
            else:
                plt.close()
    
    # Plot attention flow
    fig = plot_attention_flow(all_attention_weights, tokens)
    filepath = os.path.join(args.output_dir, "attention_flow.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    if args.show_plots:
        plt.show()
    else:
        plt.close()
    
    print(f"\nAll plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
