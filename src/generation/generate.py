"""
Text generation utilities.
"""

import torch
import torch.nn.functional as F


def generate_text(
    model,
    input_ids,
    maximum_new_tokens,
    context_size=None,
    temperature=1.0,
    top_k_tokens=None,
    top_p=None,
    repetition_penalty=1.0
):
    """
    Generate text autoregressively with improved sampling.

    Args:
        model: GPT model
        input_ids: Starting token indices (list or tensor)
        maximum_new_tokens: Number of tokens to generate
        context_size: Maximum context length
        temperature: Sampling temperature (higher = more random)
        top_k_tokens: Top-k sampling parameter (None = disabled)
        top_p: Nucleus sampling parameter (None = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)

    Returns:
        Generated token indices as a list
    """
    model.eval()

    if context_size is None:
        context_size = model.config.context_length

    # Get device from model
    device = next(model.parameters()).device

    # Convert input to tensor if needed
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    else:
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

    generated_tokens = []

    for _ in range(maximum_new_tokens):
        # Crop context if needed
        input_ids_conditioned = input_ids[:, -context_size:]

        # Get predictions
        with torch.no_grad():
            logits = model(input_ids_conditioned)

        # Get logits for last position
        logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                # If token has appeared, reduce its probability
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k_tokens is not None and top_k_tokens > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k_tokens, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Apply nucleus (top-p) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # Convert to probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Sample next token
        if temperature == 0:
            # Greedy sampling
            next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
        else:
            # Sample from distribution
            next_token = torch.multinomial(probabilities, num_samples=1)

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_tokens.append(next_token.item())

    return input_ids.squeeze(0).tolist()