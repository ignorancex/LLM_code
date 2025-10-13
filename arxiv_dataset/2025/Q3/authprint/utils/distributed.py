"""
Utilities for distributed training setup.
"""
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist


def setup_distributed(backend: str = "nccl", init_method: str = "env://") -> Tuple[int, int, int, torch.device]:
    """
    Setup distributed training environment.
    
    Args:
        backend (str): PyTorch distributed backend (nccl, gloo, etc.).
        init_method (str): URL specifying how to initialize the process group.
        
    Returns:
        tuple: (local_rank, rank, world_size, device) - local rank, global rank, 
               world size, and torch device.
    """
    if not dist.is_available():
        logging.warning("Distributed training not available, falling back to single-GPU mode")
        return 0, 0, 1, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not dist.is_initialized():
        # Initialize the process group
        try:
            dist.init_process_group(backend=backend, init_method=init_method)
        except Exception as e:
            logging.error(f"Failed to initialize distributed process group: {str(e)}")
            raise
    
    # Get local and global rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set the device for this process
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    
    logging.info(f"Initialized distributed: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
    
    return local_rank, rank, world_size, device


def cleanup_distributed():
    """
    Clean up the distributed environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("Destroyed distributed process group") 