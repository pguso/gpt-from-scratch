"""
Example: Train GPT on TinyStories dataset.

TinyStories is a dataset of short stories written at a simple reading level,
perfect for training small language models. This script demonstrates how to
train a GPT model from scratch on this dataset.
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import tiktoken

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.gpt import GPTModel
from src.config import ModelConfig
from src.data.dataset import GPTDataset
from src.training.trainer import GPTTrainer
from src.generation.generate import generate_text


def generate_sample_stories():
    """Generate simple sample stories for testing."""
    stories = [
        "Once upon a time, there was a little girl named Emma. She loved to play in the garden. Every morning, Emma would wake up early and run outside. She would pick flowers and watch the butterflies. The garden was her favorite place in the whole world. She spent hours there every day, playing and exploring.",
        
        "The cat sat on the mat. It was a sunny day. The cat was very happy. It purred softly and stretched its paws. The sun felt warm on its fur. The cat closed its eyes and took a nap. When it woke up, it saw a bird outside the window. The bird was singing a beautiful song.",
        
        "Tom had a red ball. He threw it high in the sky. The ball went up and up. Then it came back down. Tom caught it with both hands. He was very happy. He threw it again and again. His dog watched and wanted to play too. Tom threw the ball for his dog. The dog ran and fetched it.",
        
        "Mary found a flower. It was very pretty. The flower was pink and smelled sweet. Mary picked it carefully. She brought it home and put it in a vase. The flower made her room look beautiful. Every day, Mary would water the flower. It stayed fresh for many days.",
        
        "The dog ran fast. It was happy. The dog loved to run in the park. It ran after a ball. It ran after other dogs. It ran just for fun. The dog was very fast. No one could catch it. When it was tired, the dog would lie down. It would pant and rest. Then it would run again.",
        
        "Sam had a toy car. He drove it around. The car was red and shiny. Sam pushed it on the floor. It went zooming across the room. Sam made car sounds with his mouth. He pretended to be a race car driver. The car went under the table. Sam crawled to get it. He played with the car all afternoon.",
        
        "Lucy saw a bird. It was flying high. The bird had colorful feathers. It sang a beautiful song. Lucy watched it fly from tree to tree. The bird was free and happy. Lucy wished she could fly too. She watched the bird until it flew away. Then she went inside to tell her mom about the bird.",
        
        "The boy ate an apple. It was sweet. The apple was red and crunchy. The boy took a big bite. Juice ran down his chin. He laughed and wiped it away. The apple was delicious. He ate the whole thing. Then he asked for another one. His mom said he could have one more.",
        
        "Anna read a book. It was interesting. The book had pictures of animals. Anna learned about lions and tigers. She learned about elephants and monkeys. The book told stories about each animal. Anna read for a long time. She didn't want to stop. Books were her favorite thing. She had many books at home.",
        
        "The sun was bright. The day was warm. Children played outside. They ran and laughed. They played games and had fun. The warm sun made everyone happy. People sat outside and talked. Birds sang in the trees. It was a perfect day. Everyone enjoyed the beautiful weather.",
        
        "Once there was a rabbit. The rabbit was white and fluffy. It lived in a garden. The rabbit loved to eat carrots. It would hop around all day. The rabbit had many friends. There were birds and squirrels. They all played together. The rabbit was very happy in the garden.",
        
        "A little boy named Jack loved to draw. He drew pictures every day. He drew his family and friends. He drew animals and trees. His drawings were colorful and fun. Jack's mom put his drawings on the wall. The wall was covered with his art. Jack was very proud of his work.",
        
        "The old tree stood tall. It had been there for many years. Birds built nests in its branches. Children played under its shade. The tree saw many seasons pass. In spring, it had green leaves. In fall, the leaves turned colors. In winter, the branches were bare. The tree was strong and wise.",
        
        "Sara loved to dance. She danced every day. She danced in her room. She danced in the garden. She danced when she was happy. She danced when she was sad. Dancing made her feel free. Sara dreamed of being a dancer. She practiced every chance she got. Her family loved to watch her dance.",
        
        "The library was quiet. People read books there. There were many books on the shelves. Books about history and science. Books about stories and adventures. The library was a special place. It was full of knowledge. People came to learn and explore. The library was open every day.",
    ]
    return "<|endoftext|>".join(stories)


def download_tinystories_data(data_dir="data", max_samples=None):
    """
    Download TinyStories dataset from Hugging Face.
    
    Args:
        data_dir: Directory to save the data
        max_samples: Maximum number of samples to download (None for all)
    
    Returns:
        Combined text string from all stories
    """
    # Try to import datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: 'datasets' library not found.")
        print("To download TinyStories, install it with: pip install datasets")
        print("Falling back to sample data...")
        # Fallback to sample data
        sample_path = os.path.join(data_dir, "sample_text.txt")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            # Generate some simple sample stories
            return generate_sample_stories()
    
    print("Downloading TinyStories dataset from Hugging Face...")
    
    # Load the TinyStories dataset
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="train")
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
        print("This might be due to network issues or the dataset being unavailable.")
        print("Falling back to sample data...")
        # Fallback to sample data
        sample_path = os.path.join(data_dir, "sample_text.txt")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return generate_sample_stories()
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Combine all stories into a single text
    # Add end-of-text token between stories
    texts = [item["text"] for item in dataset]
    combined_text = "<|endoftext|>".join(texts)
    
    # Save to file for future use
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "tinystories.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined_text)
    
    print(f"Downloaded {len(texts)} stories. Total characters: {len(combined_text):,}")
    print(f"Saved to {output_path}")
    
    return combined_text


def load_tinystories_data(data_path="data/tinystories.txt"):
    """
    Load TinyStories data from a file.
    
    Args:
        data_path: Path to the data file
    
    Returns:
        Text string
    """
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Downloading dataset...")
        return download_tinystories_data()
    
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Loaded {len(text):,} characters")
    return text


def create_model(config):
    """Create and initialize GPT model."""
    model = GPTModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    # Count buffers (like causal masks) - these are not trainable
    total_buffers = sum(b.numel() for b in model.buffers())
    
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    if non_trainable_params > 0:
        print(f"  Non-trainable parameters: {non_trainable_params:,}")
    if total_buffers > 0:
        print(f"  Buffers (non-trainable): {total_buffers:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Verify counting
    if total_params != trainable_params + non_trainable_params:
        print(f"  Warning: Parameter count mismatch!")
    
    return model


def train(
    data_path=None,
    max_samples=10000,
    epochs=10,
    batch_size=32,
    learning_rate=3e-4,
    context_length=128,
    embedding_dimension=256,
    number_of_heads=4,
    number_of_layers=4,
    device="cuda",
    save_dir="checkpoints",
    save_every=5,
    eval_every=1,
    generate_every=2,
):
    """
    Main training function.
    
    Args:
        data_path: Path to data file (None to download)
        max_samples: Max samples to use from dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        context_length: Context length
        embedding_dimension: Embedding dimension
        number_of_heads: Number of attention heads
        number_of_layers: Number of transformer layers
        device: Device to train on
        save_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        eval_every: Evaluate every N epochs
        generate_every: Generate sample text every N epochs
    """
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = "cpu"
    device = torch.device(device)
    print(f"\nUsing device: {device}")
    
    # Load data
    if data_path and os.path.exists(data_path):
        text = load_tinystories_data(data_path)
    else:
        text = download_tinystories_data(max_samples=max_samples)
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset
    print("\nCreating dataset...")
    
    # Check text length and adjust context_length if needed
    token_ids_preview = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    text_length = len(token_ids_preview)
    print(f"Text length: {text_length:,} tokens")
    
    # If text is shorter than context_length, reduce context_length
    if text_length < context_length:
        print(f"Warning: Text length ({text_length}) is shorter than context_length ({context_length})")
        print(f"Reducing context_length to {text_length // 2} to create training sequences")
        context_length = max(32, text_length // 2)  # At least 32, or half of text length
        print(f"Using context_length: {context_length}")
    
    full_dataset = GPTDataset(
        text=text,
        tokenizer=tokenizer,
        maximum_length=context_length,
        stride=max(1, context_length // 2)  # 50% overlap, at least 1
    )
    
    print(f"Total sequences: {len(full_dataset):,}")
    
    # Check if dataset is empty
    if len(full_dataset) == 0:
        print("\n" + "="*60)
        print("ERROR: Dataset is empty!")
        print("="*60)
        print(f"\nThe text is too short ({text_length} tokens) to create sequences")
        print(f"with context_length={context_length}.")
        print("\nSolutions:")
        print("1. Install datasets library to download TinyStories:")
        print("   pip install datasets")
        print("2. Use a smaller context_length:")
        print(f"   python examples/train_tiny_stories.py --context-length {min(64, text_length // 2)}")
        print("3. Provide your own data file with --data-path")
        print("4. Use the sample_text.txt file in the data/ directory")
        return
    
    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Ensure we have at least 1 sample in each split
    if train_size == 0:
        train_size = 1
        val_size = len(full_dataset) - 1
    if val_size == 0:
        val_size = 1
        train_size = len(full_dataset) - 1
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences: {len(val_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid issues on some systems
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Create model (use adjusted context_length)
    print("\nCreating model...")
    config = ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,  # May have been adjusted above
        embedding_dimension=embedding_dimension,
        number_of_heads=number_of_heads,
        number_of_layers=number_of_layers,
        dropout_rate=0.1,
        use_attention_bias=False
    )
    
    model = create_model(config)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    # Create trainer
    trainer = GPTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device
    )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss = trainer.train_epoch()
        train_perplexity = torch.exp(torch.tensor(train_loss)).item()
        print(f"Train Loss: {train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
        
        # Initialize val_loss to None (will be set if validation runs)
        val_loss = None
        
        # Validate
        if epoch % eval_every == 0:
            val_loss = trainer.validate()
            if val_loss is not None:
                val_perplexity = torch.exp(torch.tensor(val_loss)).item()
                print(f"Val Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(save_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config.to_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f"✓ Saved best model to {checkpoint_path}")
        
        # Generate sample text
        if epoch % generate_every == 0:
            print("\nGenerating sample text...")
            model.eval()
            prompts = [
                "Once upon a time",
                "The little girl",
                "In a far away land",
            ]
            
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt)
                output_ids = generate_text(
                    model,
                    input_ids,
                    maximum_new_tokens=30,
                    temperature=0.8,
                    top_k_tokens=50
                )
                output_text = tokenizer.decode(output_ids)
                print(f"  Prompt: '{prompt}'")
                print(f"  Output: {output_text}")
                print()
            
            model.train()
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.to_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"✓ Saved checkpoint to {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(save_dir, 'best_model.pt')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train GPT on TinyStories dataset")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to data file (None to download)")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Maximum number of samples to use")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    
    # Model arguments
    parser.add_argument("--context-length", type=int, default=128,
                        help="Context length")
    parser.add_argument("--embedding-dimension", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--number-of-heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--number-of-layers", type=int, default=4,
                        help="Number of transformer layers")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"],
                        help="Device to train on")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--generate-every", type=int, default=2,
                        help="Generate sample text every N epochs")
    
    args = parser.parse_args()
    
    # Run training
    train(
        data_path=args.data_path,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        context_length=args.context_length,
        embedding_dimension=args.embedding_dimension,
        number_of_heads=args.number_of_heads,
        number_of_layers=args.number_of_layers,
        device=args.device,
        save_dir=args.save_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        generate_every=args.generate_every,
    )


if __name__ == "__main__":
    main()
