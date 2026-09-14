"""
src/evaluation/evaluator.py
Distributed inference loop for DeepWindModel evaluation.

Responsibilities:
    - Run batched inference across all DDP ranks via InferencePipeline.
    - Gather per-rank results into a single array on rank 0.
    - Trim padding introduced by DistributedSampler.
    - Return raw numpy arrays; all disk I/O is delegated to reporter.py.
"""
import logging
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.inference.pipeline import InferencePipeline
from src.utils.distributed import gather_and_merge, get_rank, get_world_size, is_main_process

logger = logging.getLogger(__name__)


class DistributedEvaluator:
    """
    Runs the inference loop across all DDP ranks and collects results on rank 0.

    Design principles:
        - Owns nothing except the pipeline reference and DDP metadata.
        - Does not write to disk; returns plain numpy dicts.
        - Safe to call in single-GPU environments (world_size=1).

    Args:
        pipeline:   Initialised InferencePipeline. Each rank owns one instance
                    pointing to its own device.
        cfg:        Hydra DictConfig. Used to read inference flags
                    (mqd_infer, use_kv_cache).
        rank:       Global rank of the current process.
        world_size: Total number of processes.
    """

    def __init__(
        self,
        pipeline:   InferencePipeline,
        cfg:        DictConfig,
        rank:       int = 0,
        world_size: int = 1,
    ) -> None:
        self.pipeline   = pipeline
        self.cfg        = cfg
        self.rank       = rank
        self.world_size = world_size
        self.device     = pipeline.device

    # ── Public API ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(self, dataloader, dataset_len, pred_len):
        local        = self._run_local_inference(dataloader, pred_len)
        num_quantiles = len(self.cfg.model.quantiles)
        gathered     = self._gather(local, pred_len, num_quantiles)

        if not is_main_process():
            return {}

        # Check emptiness after gather, on rank 0 only
        if gathered["targets"].shape[0] == 0:
            logger.warning("No results gathered — skipping.")
            return {}

        return self._trim(gathered, dataset_len)

    # ── Private: local inference ───────────────────────────────────────────────

    def _run_local_inference(
        self,
        dataloader: torch.utils.data.DataLoader,
        pred_len:   int,
    ) -> Dict[str, list]:
        """
        Iterate over the local shard of data and collect raw predictions.

        Returns a dict of Python lists (one entry per batch) to be
        concatenated after the loop.
        """
        mqd_infer = self.cfg.inference.get("mqd_infer", False)

        local = {
            "point_preds":    [],
            "quantile_preds": [],
            "targets":        [],
            "history":        [],
            "indices":        [], 
        }

        iterator = (
            tqdm(dataloader, desc=f"Inference [rank {self.rank}]", leave=False)
            if is_main_process()
            else dataloader
        )

        for batch in iterator:
            forecast = self.pipeline.predict(batch, pred_len, mqd_infer=mqd_infer)

            # Retain only the target variate (index 0) for evaluation.
            # Shape after slice: point (B, pred_len) | quantile (B, pred_len, Q)
            local["point_preds"].append(forecast.point_preds[:, 0, :].cpu())
            if forecast.quantile_preds is None:
                raise ValueError(
                    "quantile_preds is None — model must return quantile predictions for evaluation."
                )
            local["quantile_preds"].append(forecast.quantile_preds[:, 0, :, :].cpu())
            local["targets"].append(batch["target"][:, 0, :].cpu())
            local["history"].append(batch["context"][:, 0, :].cpu())
            local["indices"].append(batch["index"].cpu()) 

        return local

    # ── Private: gather & trim ─────────────────────────────────────────────────

    def _gather(self, local, pred_len, num_quantiles):
        def safe_cat(lst, empty_shape, dtype=torch.float32):
            # Return a correctly-shaped zero-size tensor if the list is empty,
            # so all_gather still gets a valid (0-row) tensor to work with.
            # The trailing dims must match non-empty ranks or the padded
            # all_gather in `gather_and_merge` raises a shape mismatch.
            if len(lst) == 0:
                return torch.zeros(empty_shape, dtype=dtype)
            return torch.cat(lst, dim=0)

        return {
            "point_preds":    gather_and_merge(safe_cat(local["point_preds"], (0, pred_len)), self.device),
            "quantile_preds": gather_and_merge(safe_cat(local["quantile_preds"], (0, pred_len, num_quantiles)), self.device),
            "targets":        gather_and_merge(safe_cat(local["targets"], (0, pred_len)), self.device),
            "history":        gather_and_merge(safe_cat(local["history"], (0, pred_len)), self.device),
            "indices":        gather_and_merge(safe_cat(local["indices"], (0,), dtype=torch.long), self.device),
        }

    @staticmethod
    def _trim(gathered, dataset_len):
        gathered_len = next(iter(gathered.values())).shape[0]
        if gathered_len != dataset_len:
            logger.info(
                f"Trimming gathered results: {gathered_len} → {dataset_len} samples "
                f"(removed {gathered_len - dataset_len} padding rows)."
            )

        trimmed = {k: v[:dataset_len] for k, v in gathered.items()}

        if "indices" in trimmed:
            indices = trimmed["indices"]

            # Guard: detect duplicate indices caused by DistributedSampler padding
            unique, counts = torch.unique(indices, return_counts=True)
            if (counts > 1).any():
                logger.warning(
                    f"Duplicate indices detected after trim "
                    f"({(counts > 1).sum().item()} values appear more than once). "
                    f"This is caused by DistributedSampler padding. "
                    f"Only the first occurrence of each index is kept."
                )
                # Keep only first occurrence of each index
                seen    = set()
                keep    = []
                for i, idx in enumerate(indices.tolist()):
                    if idx not in seen:
                        seen.add(idx)
                        keep.append(i)
                keep    = torch.tensor(keep, dtype=torch.long)
                trimmed = {k: v[keep] for k, v in trimmed.items()}

            order   = torch.argsort(trimmed.pop("indices"))
            trimmed = {k: v[order] for k, v in trimmed.items()}
        else:
            logger.warning(
                "_trim: 'indices' key not found — "
                "output order may not match original dataset order."
            )

        return {k: v.numpy() for k, v in trimmed.items()}