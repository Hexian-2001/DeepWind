"""
evaluate.py
Entry point for distributed evaluation of DeepWindModel.

Responsibilities (this file only):
    - Parse config via Hydra.
    - Initialise DDP.
    - Build dataset / dataloader for each (dataset, horizon) combination.
    - Orchestrate evaluator and reporter.
    - Tear down DDP.

All business logic lives in src/evaluation/ and src/inference/.
"""
import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from src.data.datasets import DeepWindTestDataset
from src.evaluation import DistributedEvaluator, EvaluationReporter, get_capacity, get_pred_len
from src.inference.pipeline import InferencePipeline
from src.utils.distributed import (
    barrier,
    cleanup_ddp,
    is_main_process,
    setup_ddp,
)
from src.utils.provenance import write_run_info

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_dataloader(
    npy_path:   str,
    pred_len:   int,
    cfg:        DictConfig,
    rank:       int,
    world_size: int,
) -> tuple[DeepWindTestDataset, DataLoader]:
    """
    Construct a DeepWindTestDataset and a DistributedSampler-backed DataLoader.

    Args:
        npy_path:   Path to the .npy test file.
        pred_len:   Number of prediction steps.
        cfg:        Hydra config.
        rank:       Current process rank.
        world_size: Total number of processes.

    Returns:
        (dataset, dataloader) tuple.
    """
    stride = cfg.data.get("stride", None) or pred_len

    # Per-dataset context override (e.g. gefc12/gefc14 use 1024 instead of the
    # model's 8192 so DeepWind is not disadvantaged on the short series).
    dataset_name   = os.path.basename(npy_path).replace(".npy", "")
    context_length = int(
        cfg.data.get("context_length_override", {}).get(
            dataset_name, cfg.data.context_length
        )
    )

    dataset = DeepWindTestDataset(
        npy_path          = npy_path,
        metadata_path     = cfg.data.metadata_path,
        context_length    = context_length,
        prediction_length = pred_len,
        stride            = stride,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas = world_size,
        rank         = rank,
        shuffle      = False,
        drop_last    = False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size  = cfg.inference.batch_size,
        num_workers = cfg.inference.num_workers,
        sampler     = sampler,
        pin_memory  = True,
    )

    return dataset, dataloader


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:

    # ── DDP setup ──────────────────────────────────────────────────────────────
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # ── Seed ───────────────────────────────────────────────────────────────────
    import random
    import numpy as np

    seed = cfg.seed + rank   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logging.basicConfig(
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        level   = logging.INFO if is_main_process() else logging.WARNING,
    )

    if is_main_process():
        logger.info(
            f"Evaluation started — "
            f"world_size={world_size}, "
            f"context_len={cfg.data.context_length}, "
            f"horizons={cfg.inference.horizon_h}h"
        )

    # ── Provenance manifest (rank-0 only; written before evaluation runs) ─────
    if is_main_process():
        write_run_info(cfg.output.output_dir, cfg)

    # ── Build pipeline (one per rank, on its own device) ──────────────────────
    pipeline = InferencePipeline(
        checkpoint_path = cfg.inference.checkpoint_path,
        cfg             = cfg,
        device          = device,
    )

    # ── Build evaluator and reporter ───────────────────────────────────────────
    evaluator = DistributedEvaluator(pipeline, cfg, rank, world_size)

    if is_main_process():
        reporter = EvaluationReporter(
            output_dir = cfg.output.output_dir,
            cfg        = cfg,
            quantiles  = pipeline.forecaster._inference_quantiles,
        )

    # ── Evaluation loop ────────────────────────────────────────────────────────
    test_files = cfg.data.get("test_npy_paths", [])
    horizon_hs = cfg.inference.get("horizon_h", [])

    if not test_files:
        logger.warning("cfg.data.test_npy_paths is empty — nothing to evaluate.")

    for npy_path in test_files:
            
        dataset_name = os.path.basename(npy_path).replace(".npy", "")
        
        for horizon_h in horizon_hs:

            if is_main_process():
                logger.info(f"── Dataset: {dataset_name}  |  Horizon: {horizon_h}h ──")

            # -- Resolve dataset metadata --------------------------------------
            try:
                pred_len     = get_pred_len(dataset_name, horizon_h)
                capacity_val = get_capacity(dataset_name)
                if is_main_process():
                    logger.info(f"capacity_val = {capacity_val}") 
            except KeyError as e:
                if is_main_process():
                    logger.warning(f"Skipping {dataset_name} @ {horizon_h}h — {e}")
                barrier()
                continue
        
            # -- Build dataloader ----------------------------------------------
            dataset, dataloader = _build_dataloader(
                npy_path, pred_len, cfg, rank, world_size
            )

            # -- Run inference -------------------------------------------------
            results = evaluator.run(dataloader, len(dataset), pred_len)

            # -- Report results (rank 0 only) ----------------------------------
            if is_main_process():
                reporter.report(
                    results      = results,
                    dataset_name = dataset_name,
                    horizon_h    = horizon_h,
                    pred_len     = pred_len,
                    capacity_val = capacity_val,
                )

            barrier()

    # ── Teardown ───────────────────────────────────────────────────────────────
    cleanup_ddp()
    if is_main_process():
        logger.info("Evaluation complete.")

    
if __name__ == "__main__":
    main()