"""
Debug DataLoader — clean global-batch view.

Gathers all samples across 16 GPUs to rank 0, prints one summary per
global batch (1024 samples = 16 ranks × 2 accum × 32 bs).

Only rank 0 prints. 5 steps.
"""
        
import argparse
import json
import logging
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, default_collate

from src.data.datasets import DeepWindTrainDataset, _pad_and_mask
from src.utils.distributed import get_rank_world, is_main_process

logger = logging.getLogger(__name__)


def _setup_logging(rank):
    if rank == 0:
        logging.basicConfig(
            format="[%(asctime)s] %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
            force=True,
        )
    else:
        logging.basicConfig(level=logging.WARNING, force=True)


# ---------------------------------------------------------------------------
# Instrumented dataset
# ---------------------------------------------------------------------------

class InstrumentedTrainDataset(DeepWindTrainDataset):

    def _generate_from_file(self, path, file_rng):
        values = np.load(path, mmap_mode="r")
        if values.ndim != 2:
            return
        C, T = values.shape
        if T < self.seq_len:
            return
        n_all = T - self.seq_len + 1
        if n_all <= 0:
            return

        num_win = self._num_windows(T)
        starts = file_rng.integers(0, n_all, size=num_win)
        meta = self.meta.get(path.name)
        tag = meta["dataset_tag"]

        coverage = (num_win * self.seq_len) / T
        if coverage > self.coverage_thresh:
            buf = values.astype(np.float32)
            for s in starts:
                sample = _pad_and_mask(
                    buf[:, s : s + self.seq_len],
                    meta, self.max_vars, self.seq_len, self.pad_val_id,
                )
                sample["__file__"] = path.name
                sample["__start__"] = int(s)
                sample["__tag__"] = tag
                yield sample
        else:
            for s in starts:
                w = values[:, s : s + self.seq_len].astype(np.float32)
                sample = _pad_and_mask(
                    w, meta, self.max_vars, self.seq_len, self.pad_val_id,
                )
                sample["__file__"] = path.name
                sample["__start__"] = int(s)
                sample["__tag__"] = tag
                yield sample


def debug_collate(batch):
    files = [b.pop("__file__") for b in batch]
    starts = [b.pop("__start__") for b in batch]
    tags = [b.pop("__tag__") for b in batch]
    tensors = default_collate(batch)
    tensors["__file__"] = files
    tensors["__start__"] = starts
    tensors["__tag__"] = tags
    return tensors


# ---------------------------------------------------------------------------
# Gather metadata strings across ranks via dist
# ---------------------------------------------------------------------------

def gather_metadata_to_rank0(files, starts, tags, elapsed_ms, rank, world_size):
    """Serialize per-rank metadata to JSON, pad, all_gather, decode on rank 0."""
    local_data = json.dumps({
        "files": files,
        "starts": starts,
        "tags": tags,
        "elapsed_ms": elapsed_ms,
    }).encode("utf-8")

    local_len = torch.tensor([len(local_data)], dtype=torch.long, device="cuda")
    all_lens = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)]
    dist.all_gather(all_lens, local_len)
    max_len = max(l.item() for l in all_lens)

    padded = local_data + b"\x00" * (max_len - len(local_data))
    local_tensor = torch.frombuffer(bytearray(padded), dtype=torch.uint8).cuda()
    gathered = [torch.zeros(max_len, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)

    if rank != 0:
        return None

    all_data = []
    for r in range(world_size):
        raw = gathered[r].cpu().numpy().tobytes()
        raw = raw[: all_lens[r].item()]
        all_data.append(json.loads(raw.decode("utf-8")))
    return all_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_root", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    _setup_logging(rank)

    global_batch = args.batch_size * args.grad_accum * world_size

    if rank == 0:
        logger.info(f"world_size={world_size}  bs={args.batch_size}  "
                     f"accum={args.grad_accum}  global_batch={global_batch}  "
                     f"workers={args.num_workers}  steps={args.num_steps}")

    ds = InstrumentedTrainDataset(
        npy_root=args.npy_root,
        metadata_path=args.metadata_path,
        seq_len=args.seq_len,
        max_vars=6,
        pad_val_id=10,
        dataset_weights={"windtoolkit": 0.7, "scada": 0.3},
        sample_ratio=0.005,
        min_windows_per_file=4,
        max_windows_per_file=128,
        full_read_coverage_threshold=0.3,
        repeat=True,
        seed=42,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=debug_collate,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
    )

    dist.barrier()
    it = iter(loader)

    # Track cumulative duplicates across steps
    cumulative_seen = set()
    cumulative_dups = 0
    all_step_times = []

    for step_i in range(args.num_steps):
        # Each rank collects grad_accum micro-batches
        local_files = []
        local_starts = []
        local_tags = []

        torch.cuda.synchronize(local_rank)
        t0 = time.perf_counter()

        for _ in range(args.grad_accum):
            batch = next(it)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(local_rank, non_blocking=True)
            local_files.extend(batch["__file__"])
            local_starts.extend(batch["__start__"])
            local_tags.extend(batch["__tag__"])

        torch.cuda.synchronize(local_rank)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Gather to rank 0
        all_rank_data = gather_metadata_to_rank0(
            local_files, local_starts, local_tags, elapsed_ms, rank, world_size
        )

        if rank == 0:
            # Merge all ranks
            g_files, g_starts, g_tags = [], [], []
            rank_times = []
            for rd in all_rank_data:
                g_files.extend(rd["files"])
                g_starts.extend(rd["starts"])
                g_tags.extend(rd["tags"])
                rank_times.append(rd["elapsed_ms"])

            n = len(g_files)

            # Tag distribution
            tag_counter = Counter(g_tags)
            tag_str = "  ".join(
                f"{t}: {c} ({100*c/n:.1f}%)" for t, c in sorted(tag_counter.items())
            )

            # Duplicates within this batch
            pairs = [(f, s) for f, s in zip(g_files, g_starts)]
            pair_counter = Counter(pairs)
            batch_dups = sum(c - 1 for c in pair_counter.values() if c > 1)

            # Cumulative duplicates (across steps)
            step_cross_dups = 0
            for p in pairs:
                if p in cumulative_seen:
                    step_cross_dups += 1
                cumulative_seen.add(p)
            cumulative_dups += step_cross_dups

            # Diversity: unique files
            unique_files = len(set(g_files))
            diversity = unique_files / n

            # File distribution (top 10)
            file_counter = Counter(g_files)
            top_files = file_counter.most_common(10)

            # Timing
            mean_time = np.mean(rank_times)
            max_time = np.max(rank_times)
            min_time = np.min(rank_times)
            all_step_times.append(max_time)  # bottleneck = slowest rank

            # ── Print ──
            logger.info("")
            logger.info(f"{'='*80}")
            logger.info(f"  STEP {step_i}  |  {n} samples  |  {mean_time:.1f}ms mean  "
                         f"{max_time:.1f}ms max  {min_time:.1f}ms min")
            logger.info(f"{'='*80}")
            logger.info(f"  Tags:       {tag_str}")
            logger.info(f"  Diversity:  {unique_files} unique files / {n} samples = {diversity:.4f}")
            logger.info(f"  Duplicates: {batch_dups} within batch, "
                         f"{step_cross_dups} cross-step (cumulative: {cumulative_dups})")
            logger.info(f"  Top 10 files by sample count:")
            # File distribution per tag (top 10 each)
            for tag_name in sorted(tag_counter.keys()):
                tag_files = [f for f, t in zip(g_files, g_tags) if t == tag_name]
                fc = Counter(tag_files)
                top = fc.most_common(10)
                logger.info(f"  Top 10 [{tag_name}] ({len(fc)} unique files):")
                for fname, cnt in top:
                    logger.info(f"    {fname:<45s} ×{cnt}")

            # Per-rank timing
            logger.info(f"  Per-rank time (ms): "
                         + "  ".join(f"r{i}={t:.1f}" for i, t in enumerate(rank_times)))

        dist.barrier()

    # ── Final summary ──
    if rank == 0:
        logger.info("")
        logger.info(f"{'='*80}")
        logger.info(f"  FINAL SUMMARY  ({args.num_steps} steps × {global_batch} samples)")
        logger.info(f"{'='*80}")
        logger.info(f"  Total samples:      {args.num_steps * global_batch}")
        logger.info(f"  Cumulative dups:    {cumulative_dups}")
        logger.info(f"  Step times (max across ranks):")
        for i, t in enumerate(all_step_times):
            logger.info(f"    step {i}: {t:.1f} ms")
        mean_all = np.mean(all_step_times)
        logger.info(f"  Mean: {mean_all:.1f} ms  |  Max: {np.max(all_step_times):.1f} ms")
        if mean_all > 200:
            logger.info("  ⚠️  SLOW >200ms: GPU will likely starve.")
        elif mean_all > 50:
            logger.info("  ⚡ MODERATE 50-200ms: OK if fwd+bwd >200ms.")
        else:
            logger.info("  ✅ FAST <50ms: data loading not a bottleneck.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()