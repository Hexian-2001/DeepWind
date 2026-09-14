"""
src/inference/pipeline.py
Self-contained inference pipeline for DeepWindModel.

Design decisions:
    - Device-agnostic: auto-selects CUDA if available, falls back to CPU.
    - Batch device management is handled internally; callers pass raw CPU batches.
    - adapter_path resolution: explicit argument takes priority over cfg value,
      so the same pipeline class works for both scripted and interactive use.
    - DDP-unaware by design: instantiate once per process in multi-GPU settings.
"""
import logging
import os
from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from peft import PeftModel

from src.inference.generator import DeepWindForecaster, Forecast
from src.models.deepwind import DeepWindModel
from src.utils.distributed import to_device, is_main_process

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Thin wrapper that owns model loading and exposes a single predict() call.

    Responsibilities:
        - Load the base DeepWindModel from a checkpoint.
        - Optionally merge a LoRA adapter and discard the adapter scaffolding.
        - Construct a DeepWindForecaster.
        - Accept raw CPU batches, move them to the correct device, and return
          a Forecast dataclass.

    This class is intentionally DDP-unaware. In multi-GPU evaluation, each
    rank instantiates its own pipeline on its own device.

    Args:
        checkpoint_path: Path to the pretrained DeepWindModel directory.
        cfg:             Hydra DictConfig. Must contain cfg.data.context_length
                         and optionally cfg.inference.adapter_path.
        adapter_path:    Optional explicit path to a LoRA adapter directory.
                         Overrides cfg.inference.adapter_path when provided.
        device:          Target device. Defaults to cuda:0 if available, else cpu.
    """

    def __init__(
        self,
        checkpoint_path: str,
        cfg:             DictConfig,
        adapter_path:    Optional[str]          = None,
        device:          Optional[torch.device] = None,
    ) -> None:
        self.cfg    = cfg
        self.device = device or self._default_device()

        self.model      = self._load_model(checkpoint_path, adapter_path)
        self.forecaster = DeepWindForecaster(
            model               = self.model,
            cfg                 = cfg,
            inference_quantiles = cfg.inference.get("inference_quantiles", None),
        )

    # ── Setup ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _default_device() -> torch.device:
        """Select CUDA if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_adapter_path(self, explicit: Optional[str]) -> Optional[str]:
        """
        Determine the effective adapter path.

        Priority:
            1. Explicit argument passed to __init__.
            2. cfg.inference.adapter_path (if set and non-empty).
            3. None — base model only.
        """
        if explicit:
            return explicit
        cfg_path = self.cfg.inference.get("adapter_path", None)
        return cfg_path if cfg_path else None

    def _load_model(self, checkpoint_path: str, adapter_path: Optional[str]) -> DeepWindModel:
    
        # Rank 0 loads first to avoid Lustre I/O contention,
        # other ranks wait at the barrier before loading.
        import torch.distributed as dist
    
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                dist.barrier()   # non-rank-0 waits here
    
        if is_main_process():
            logger.info(f"Loading base model from: {checkpoint_path}")
    
        model: DeepWindModel = DeepWindModel.from_pretrained(checkpoint_path)
        model.to(self.device)
        model.eval()
    
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                dist.barrier()   # rank 0 signals it's done, others now proceed
    
        resolved = self._resolve_adapter_path(adapter_path)
        if resolved:
            model = self._merge_adapter(model, resolved)
    
        return model

    def _merge_adapter(self, model: DeepWindModel, adapter_path: str) -> DeepWindModel:
        """
        Load a LoRA adapter and merge its weights into the base model.

        Args:
            model:        Base DeepWindModel (already on device).
            adapter_path: Path to the saved PeftModel adapter directory.

        Returns:
            Merged DeepWindModel with adapter weights folded in.

        Raises:
            FileNotFoundError: If adapter_path does not exist.
        """
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter path does not exist: {adapter_path}\n"
                "Set cfg.inference.adapter_path to a valid directory, "
                "or pass adapter_path=None to use the base model only."
            )
        
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        model.eval()
        logger.info("Adapter merged and unloaded successfully.")
        return model

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        batch:      Dict[str, torch.Tensor],
        pred_len:   int,
        mqd_infer:  bool = False,
    ) -> Forecast:
        """
        Run inference on a single batch.

        Handles device transfer internally; the caller does not need to move
        tensors before passing them in.

        Args:
            batch:      Dictionary from a DataLoader. Expected keys:
                            context      (B, V, T)
                            site_coords  (B, 2)
                            variate_ids  (B, V)
                            channel_mask (B, V)   optional
                            has_coords   (B, 1)   optional
            pred_len:   Number of future time steps to forecast.
            mqd_infer:  If True, use MQD decoding; otherwise greedy.

        Returns:
            Forecast:
                point_preds    (B, V, pred_len)    — median forecast
                quantile_preds (B, V, pred_len, Q) — full quantile predictions
        """
        batch = to_device(batch, self.device)
        
        # ── AMP: only active on CUDA, no-op on CPU ────────────────────────────
        amp_enabled = (
            self.cfg.inference.get("use_amp", False)
            and self.device.type == "cuda"
        )

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            return self.forecaster.forecast(
                context=batch["context"],
                site_coords=batch.get("site_coords"),
                variate_ids=batch.get("variate_ids"),
                channel_mask=batch.get("channel_mask"),
                has_coords=batch.get("has_coords"),
                prediction_length=pred_len,
                mqd_infer=mqd_infer,
            )