"""
src/utils/distributed.py
Utilities for distributed training (DDP).
"""
import os
import torch.distributed as dist

def is_dist_avail_and_initialized() -> bool:
    """Checking if DDP is currently available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size() -> int:
    """Get total number of processes."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    """Get global rank of current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process() -> bool:
    """
    Check if the current process is the main process (Rank 0).
    Useful for logging, saving checkpoints, etc.
    """
    return get_rank() == 0

def rank_zero_only(func):
    """
    Decorator to ensure a function only runs on the main process.
    Usage:
        @rank_zero_only
        def print_status(msg):
            print(msg)
    """
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        return None
    return wrapper
