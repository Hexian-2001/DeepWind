"""
src/utils/distributed.py
Utilities for distributed training and inference (DDP).
"""
import os
import functools
import logging
from typing import Tuple, Dict

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ── Availability Checks ────────────────────────────────────────────────────────

def is_dist_avail_and_initialized() -> bool:
    """Check if distributed processing is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Get total number of processes in the current process group."""
    if not is_dist_avail_and_initialized():
        return int(os.environ.get("WORLD_SIZE", 1))
    return dist.get_world_size()


def get_rank() -> int:
    """Get global rank of the current process."""
    if not is_dist_avail_and_initialized():
        return int(os.environ.get("RANK", 0))
    return dist.get_rank()


def get_rank_world() -> Tuple[int, int]:
    """
    Return (rank, world_size) safely at any point, including inside
    DataLoader worker subprocesses where dist is not initialized.

    Priority:
        1. dist already initialized  -> use dist API          (main process)
        2. launcher env vars         -> use RANK/WORLD_SIZE   (worker subprocess)
        3. fallback                  -> (0, 1)                (single GPU / CPU)
    """
    if is_dist_avail_and_initialized():
        return dist.get_rank(), dist.get_world_size()
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def is_main_process() -> bool:
    """Return True if the current process is rank 0."""
    return get_rank() == 0


# ── Lifecycle Management ───────────────────────────────────────────────────────

def setup_ddp() -> Tuple[int, int, int]:
    """
    Initialize the DDP process group via NCCL backend.
    Reads RANK, LOCAL_RANK, and WORLD_SIZE from environment variables
    set by the launcher (e.g. torchrun).

    Returns:
        (rank, local_rank, world_size). Falls back to (0, 0, 1) for single-GPU runs.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)  # set device before init_process_group

        dist.init_process_group(
            backend   = "nccl",
            device_id = torch.device(f"cuda:{local_rank}"),  # explicit GPU binding
        )

        logger.info(f"DDP initialized: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp() -> None:
    """Destroy the DDP process group. Safe to call in non-DDP environments."""
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """
    Synchronize all processes. No-op if DDP is not initialized,
    so it is safe to call unconditionally.
    """
    if is_dist_avail_and_initialized():
        dist.barrier()


# ── Data Communication ─────────────────────────────────────────────────────────

def gather_and_merge(local_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    All-gather a local tensor across all ranks and concatenate along dim 0.
    Handles uneven splits caused by DistributedSampler with drop_last=False.

    Each rank broadcasts its local size first, then pads to the max size
    before all_gather, so shape mismatches across nodes are handled correctly.
    """
    world_size = get_world_size()
    if world_size == 1:
        return local_tensor

    tensor_gpu  = local_tensor.to(device)
    local_size  = torch.tensor([tensor_gpu.shape[0]], dtype=torch.long, device=device)

    # Step 1: share each rank's local size so every rank knows the max
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device)
                 for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes  = [s.item() for s in all_sizes]
    max_size   = max(all_sizes)

    # Step 2: pad local tensor to max_size along dim 0
    pad_rows = max_size - tensor_gpu.shape[0]
    if pad_rows > 0:
        pad_shape  = (pad_rows,) + tensor_gpu.shape[1:]
        padding    = torch.zeros(pad_shape, dtype=tensor_gpu.dtype, device=device)
        tensor_gpu = torch.cat([tensor_gpu, padding], dim=0)

    # Step 3: all_gather padded tensors (now uniform shape across ranks)
    gathered = [torch.zeros_like(tensor_gpu) for _ in range(world_size)]
    dist.all_gather(gathered, tensor_gpu)

    # Step 4: trim each rank's contribution back to its true size
    trimmed = [gathered[r][:all_sizes[r]] for r in range(world_size)]
    return torch.cat(trimmed, dim=0).cpu()


def to_device(batch: Dict, device: torch.device) -> Dict:
    """
    Move all Tensor values in a batch dictionary to the specified device.
    Non-tensor values are passed through unchanged.

    Args:
        batch:  Dictionary of batch data from a DataLoader.
        device: Target device.

    Returns:
        New dictionary with all tensors on the target device.
    """
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ── Decorators ─────────────────────────────────────────────────────────────────

def rank_zero_only(func):
    """
    Decorator that restricts execution to rank 0.
    All other ranks receive None as the return value.

    Usage:
        @rank_zero_only
        def log_summary(metrics):
            print(metrics)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        return None
    return wrapper