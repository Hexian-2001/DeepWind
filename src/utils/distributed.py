"""
src/utils/distributed.py
Utilities for distributed training (DDP).
"""
import os
import functools
from typing import Tuple

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
        return int(os.environ.get("WORLD_SIZE", 1))
    return dist.get_world_size()


def get_rank() -> int:
    """Get global rank of current process."""
    if not is_dist_avail_and_initialized():
        return int(os.environ.get("RANK", 0))
    return dist.get_rank()


def get_rank_world():
    if dist.is_available() and dist.is_initialized():  
        return dist.get_rank(), dist.get_world_size()
    return 0, 1  


def get_rank_world() -> Tuple[int, int]:
    """
    Return (rank, world_size) in a way that is safe to call at any point,
    including inside DataLoader worker subprocesses where dist is not
    initialized.

    Priority:
        1. dist already initialized  → use dist API         (main process)
        2. launcher env vars         → use RANK/WORLD_SIZE  (worker subprocess)
        3. fallback                  → (0, 1)               (single GPU / CPU)

    This is the correct function to use inside IterableDataset.__iter__()
    for DDP-aware file sharding, because DataLoader workers are forked
    subprocesses in which dist.is_initialized() is always False.
    """
    if is_dist_avail_and_initialized():
        return dist.get_rank(), dist.get_world_size()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        return None
    return wrapper