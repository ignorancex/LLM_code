import logging

import torch

logger = logging.getLogger(__name__)


def get_device():
    """
    Utility function to get the currently available PyTorch device.

    Returns
    -------
    torch.device
        Available torch device (either cuda or cpu).
    """
    try:
        # Try to get CUDA device count to check if CUDA is properly initialized
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    except Exception as e:
        logger.warning(f"CUDA error: {e}. Falling back to CPU.")
        return torch.device("cpu")

def gradient2d(tensor):
    """
    Compute gradients (dy, dx) of a 2D tensor using centered differences.
    """
    dx = (torch.roll(tensor, shifts=-1, dims=-1) - torch.roll(tensor, shifts=1, dims=-1)) / 2.0
    dy = (torch.roll(tensor, shifts=-1, dims=-2) - torch.roll(tensor, shifts=1, dims=-2)) / 2.0

    # fix edges with forward/backward differences
    dx[..., 0]  = tensor[..., 1]  - tensor[..., 0]
    dx[..., -1] = tensor[..., -1] - tensor[..., -2]
    dy[..., 0, :]  = tensor[..., 1, :]  - tensor[..., 0, :]
    dy[..., -1, :] = tensor[..., -1, :] - tensor[..., -2, :]

    return dy, dx