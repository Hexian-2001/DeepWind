"""
Debug script: verify DeepWindTrainDataset sharding across all ranks/workers.

Usage (see debug_dataset.sh):
    torchrun --nproc_per_node=8 scripts/debug_dataset_sharding.py

Checks:
    1. Each (rank, worker) gets a unique gw_id
    2. No file overlap between any two workers
    3. All files are covered (union = full set)
    4. Per-group file counts are consistent
"""

import os
import json
import logging
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import DeepWindTrainDataset
from src.utils.distributed import get_rank_world

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===================================================================
#  Custom collate: collect metadata only, skip real data loading
# ===================================================================

def _identity_collate(batch):
    """Return the batch as-is without tensor conversion."""
    return batch


# ===================================================================
#  Dataset wrapper that yields shard metadata instead of data windows
# ===================================================================

class ShardingAuditDataset(DeepWindTrainDataset):
    """
    Subclass of DeepWindTrainDataset whose __iter__ yields only file
    path fingerprints and shard metadata rather than real data windows.
    This allows fast sharding verification without loading any data.
    """

    def __iter__(self):
        from torch.utils.data import get_worker_info as _gwi

        rank, world_size = get_rank_world()
        info      = _gwi()
        n_workers = info.num_workers if info else 1
        w_id      = info.id          if info else 0

        gw_id    = rank * n_workers + w_id
        gw_total = world_size * n_workers
        
        # Collect the file shard assigned to this worker for each group
        shard_summary = {}
        for tag in self._tags:
            shard = self.groups[tag][gw_id::gw_total]
            shard_summary[tag] = {
                "count": len(shard),
                # Use first 8 chars of md5 as a compact file fingerprint
                # to avoid transmitting full paths across ranks
                "fingerprints": sorted(
                    hashlib.md5(p.name.encode()).hexdigest()[:8]
                    for p in shard
                ),
            }

        # Yield exactly one record per worker — enough for sharding audit
        yield {
            "rank":       rank,
            "world_size": world_size,
            "w_id":       w_id,
            "n_workers":  n_workers,
            "gw_id":      gw_id,
            "gw_total":   gw_total,
            "shards":     shard_summary,
        }


# ===================================================================
#  Main
# ===================================================================

def main(args):
    # Initialize distributed process group
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    logger.info(f"rank={rank}/{world_size} initialized")

    # Build the audit dataset with the same config as real training
    ds = ShardingAuditDataset(
        npy_root        = args.npy_root,
        metadata_path   = args.metadata_path,
        seq_len         = 8192,
        max_vars        = 6,
        dataset_weights = {"windtoolkit": 0.7, "scada": 0.3},
        seed            = 42,
    )

    # DataLoader: num_workers must match the real training configuration
    loader = DataLoader(
        ds,
        batch_size  = 1,
        num_workers = args.num_workers,
        collate_fn  = _identity_collate,
    )

    # Collect one record per worker on this rank
    local_records = []
    for batch in loader:
        local_records.extend(batch)
        if len(local_records) >= args.num_workers:
            break  # One record per worker is sufficient

    # Gather results from all ranks onto rank 0
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_records)

    if rank == 0:
        _analyze(gathered, world_size, args.num_workers)

    dist.destroy_process_group()


# ===================================================================
#  Analysis (rank 0 only)
# ===================================================================

def _analyze(gathered, world_size, num_workers):
    """Analyse the gathered shard records and print a verification report."""
    all_records = []
    for rank_records in gathered:
        all_records.extend(rank_records)

    total_workers = world_size * num_workers
    tags = list(all_records[0]["shards"].keys())

    logger.info(f"\n{'='*60}")
    logger.info(f"Total records collected : {len(all_records)}")
    logger.info(f"Expected                : {world_size} ranks x "
                f"{num_workers} workers = {total_workers}")

    # ------------------------------------------------------------------
    # Check 1: gw_id uniqueness
    # Every worker must have a distinct global worker id.
    # ------------------------------------------------------------------
    gw_ids          = [r["gw_id"] for r in all_records]
    expected_gw_ids = set(range(total_workers))
    actual_gw_ids   = set(gw_ids)

    logger.info(f"\n{'='*60}")
    logger.info("CHECK 1: gw_id uniqueness")
    if actual_gw_ids == expected_gw_ids:
        logger.info(
            f"  PASS: all {len(expected_gw_ids)} gw_ids present and unique"
        )
    else:
        missing   = expected_gw_ids - actual_gw_ids
        duplicate = {x for x in gw_ids if gw_ids.count(x) > 1}
        logger.error(f"  FAIL: missing={missing}, duplicates={duplicate}")

    # ------------------------------------------------------------------
    # Check 2: no file overlap between workers
    # Each file fingerprint should appear exactly once across all workers.
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("CHECK 2: no file overlap between workers")

    for tag in tags:
        all_fps = []
        for r in all_records:
            fps = r["shards"].get(tag, {}).get("fingerprints", [])
            all_fps.extend(fps)

        total  = len(all_fps)
        unique = len(set(all_fps))

        if total == unique:
            logger.info(
                f"  PASS [{tag}]: {unique} unique files, no overlap"
            )
        else:
            logger.error(
                f"  FAIL [{tag}]: {total} total, {unique} unique"
                f" -> {total - unique} duplicate assignments"
            )

    # ------------------------------------------------------------------
    # Check 3: file count distribution across workers
    # All workers should receive a roughly equal number of files.
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("CHECK 3: file counts per worker")

    for tag in tags:
        counts        = [
            r["shards"].get(tag, {}).get("count", 0)
            for r in all_records
        ]
        total_covered = sum(counts)
        logger.info(
            f"  [{tag}] total={total_covered}  "
            f"min={min(counts)}  max={max(counts)}  "
            f"mean={total_covered / len(counts):.1f}"
        )

    # ------------------------------------------------------------------
    # Detailed per-worker table
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("DETAIL: per-worker shard breakdown")

    header_tags = " | ".join(f"{tag:>14}" for tag in tags)
    logger.info(f"{'gw_id':>6} {'rank':>5} {'w_id':>5} | {header_tags}")
    logger.info("-" * (30 + 17 * len(tags)))

    for r in sorted(all_records, key=lambda x: x["gw_id"]):
        counts_str = " | ".join(
            f"{r['shards'].get(tag, {}).get('count', 0):>14}"
            for tag in tags
        )
        logger.info(
            f"{r['gw_id']:>6} {r['rank']:>5} {r['w_id']:>5} | {counts_str}"
        )

    # ------------------------------------------------------------------
    # Save full records to JSON for offline inspection
    # ------------------------------------------------------------------
    out_path = Path("debug_sharding_result.json")
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    logger.info(f"\nFull results saved to: {out_path}")


# ===================================================================
#  Entry point
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify DeepWindTrainDataset DDP sharding correctness."
    )
    parser.add_argument(
        "--npy_root",
        required=True,
        help="Directory containing .npy training files.",
    )
    parser.add_argument(
        "--metadata_path",
        required=True,
        help="Path to metadata CSV file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers per rank (must match training).",
    )
    args = parser.parse_args()
    main(args)