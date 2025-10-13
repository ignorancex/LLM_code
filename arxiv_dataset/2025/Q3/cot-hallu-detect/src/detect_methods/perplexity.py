import numpy as np
import torch
from torch import Tensor

def calcu_perplexity(_probs: Tensor) -> float:
    """
    Calculate perplexity from probabilities.

    Args:
        probs (np.ndarray): Array of shape (batch_size, seq_len), dtype=np.float64

    Returns:
        float: Perplexity value
    """
    # Ensure dtype is float64
    probs = _probs.to("cpu").numpy()
    probs = probs.astype(np.float64)
    log_probs = np.log(np.clip(probs, a_min=1e-9, a_max=None))
    mean_log_probs = np.mean(log_probs, axis=1)

    perplexity = np.exp(-mean_log_probs).mean()

    return perplexity.item()
