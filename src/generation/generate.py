"""
Text generation utilities.
"""

import torch


def generate_text(
    model,
    input_ids,
    maximum_new_tokens,
    context_size=None,
    temperature=1.0,
    top_k_tokens=None
):
    """
    Generate text autoregressively.
    
    Args:
        model: GPT model
        input_ids: Starting token indices
        maximum_new_tokens: Number of tokens to generate
        context_size: Maximum context length
        temperature: Sampling temperature
        top_k_tokens: Top-k sampling parameter
        
    Returns:
        Generated token indices
    """
    model.eval()
    
    if context_size is None:
        context_size = model.config.context_length
    
    # Get device from model
    device = next(model.parameters()).device
    
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    else:
        input_ids = input_ids.to(device)
    
    for _ in range(maximum_new_tokens):
        input_ids_conditioned = input_ids[:, -context_size:]
        
        with torch.no_grad():
            logits = model(input_ids_conditioned)
        
        logits = logits[:, -1, :] / temperature
        
        if top_k_tokens is not None:
            # Get top-k values for each sample in the batch
            top_k_values, _ = torch.topk(logits, min(top_k_tokens, logits.size(-1)), dim=-1)
            # Create mask: keep only top-k logits
            threshold = top_k_values[:, -1].unsqueeze(-1)  # k-th largest value for each sample
            logits = logits.masked_fill(logits < threshold, float('-inf'))
        
        probabilities = torch.softmax(logits, dim=-1)
        
        # Sample from the distribution (not greedy)
        if temperature > 0:
            next_token = torch.multinomial(probabilities, num_samples=1)
        else:
            # Greedy sampling when temperature is 0
            next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids.squeeze(0).tolist()
