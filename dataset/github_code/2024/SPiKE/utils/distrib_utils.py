"""
Module for distributed utilities.
"""

from __future__ import print_function
import builtins as __builtin__
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process.
    """
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.

    Returns:
        bool: True if distributed training is available and initialized, False otherwise.
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    """
    Get the number of processes in the current process group.

    Returns:
        int: The number of processes in the current process group.
    """
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    """
    Get the rank of the current process.

    Returns:
        int: The rank of the current process.
    """
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    """
    Check if the current process is the main process.

    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return get_rank() == 0
