"""
infer.py
User-facing single-site forecasting CLI for DeepWindModel.

Loads a foundation checkpoint (optionally merged with a LoRA adapter) and
produces a probabilistic forecast for one wind-power series from a ``.npy``
file.  The context window is the final ``context_length`` steps of the series;
outputs are written as NPZ (authoritative, full precision) and optionally JSON.

The same ``MetadataStore`` / ``_pad_and_mask`` utilities used by training and
evaluation build the input batch, so there is no train/serve preprocessing
skew.

Examples
--------
    python infer.py \
        inference.checkpoint_path=/path/to/deepwind_small \
        data.npy_path=/path/to/site.npy \
        inference.prediction_length=96 \
        output.output_path=./site_forecast.npz

    # With a fine-tuned LoRA adapter:
    python infer.py \
        inference.checkpoint_path=/path/to/deepwind_small \
        inference.adapter_path=/path/to/best_adapter_site \
        data.npy_path=/path/to/site.npy
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# _pad_and_mask is imported so the served input is byte-for-byte identical to
# the preprocessing used during pretraining/evaluation (no train/serve skew).
from src.data.datasets import MetadataStore, _pad_and_mask
from src.inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_context(cfg: DictConfig) -> dict[str, torch.Tensor]:
    """Build a single-sample batch from the final ``context_length`` steps."""
    npy_path = cfg.data.npy_path
    context_length = int(cfg.data.context_length)

    data = np.load(npy_path, mmap_mode="r")
    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2-D (channels, time) .npy file, got shape {data.shape} "
            f"for {npy_path}"
        )
    C, T = data.shape
    if T < context_length:
        raise ValueError(
            f"Series length {T} < context_length {context_length} ({npy_path})"
        )

    meta = MetadataStore(cfg.data.metadata_path).get(Path(npy_path).name)
    window = np.array(data[:, -context_length:], dtype=np.float32)

    sample = _pad_and_mask(
        window, meta, cfg.data.max_vars, context_length, cfg.data.pad_val_id
    )

    # Add a batch dimension of 1 (the pipeline expects (B, ...) tensors).
    return {k: torch.from_numpy(v).unsqueeze(0) for k, v in sample.items()}


def _save_outputs(
    forecast,
    quantile_levels: np.ndarray,
    context: torch.Tensor,
    channel_mask: torch.Tensor,
    cfg: DictConfig,
) -> Path:
    """Write point + quantile forecasts to NPZ (and JSON if enabled)."""
    output_path = Path(cfg.output.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    point = forecast.point_preds[0].cpu().numpy()            # (V, H)
    quant = forecast.quantile_preds[0].cpu().numpy()         # (V, H, Q)
    ctx = context[0].cpu().numpy()                           # (V, context_length)
    mask = channel_mask[0].cpu().numpy()                     # (V,)

    if cfg.output.save_npz:
        npz_path = output_path.with_suffix(".npz")
        np.savez(
            npz_path,
            point_preds=point,
            quantile_preds=quant,
            quantiles=np.asarray(quantile_levels, dtype=np.float64),
            context=ctx,
            channel_mask=mask,
            npy_path=str(cfg.data.npy_path),
            checkpoint_path=str(cfg.inference.checkpoint_path),
        )
        logger.info("NPZ forecast → %s", npz_path)

    if cfg.output.save_json:
        json_path = output_path.with_suffix(".json")
        payload = {
            "meta": {
                "npy_path": str(cfg.data.npy_path),
                "checkpoint_path": str(cfg.inference.checkpoint_path),
                "adapter_path": str(cfg.inference.adapter_path),
                "context_length": int(cfg.data.context_length),
                "prediction_length": int(cfg.inference.prediction_length),
                "quantiles": np.asarray(quantile_levels, dtype=np.float64)
                .round(6)
                .tolist(),
            },
            "point_preds": point.round(6).tolist(),
            "quantile_preds": quant.round(6).tolist(),
            "channel_mask": mask.astype(int).tolist(),
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("JSON forecast → %s", json_path)

    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", device)

    pipeline = InferencePipeline(
        checkpoint_path=cfg.inference.checkpoint_path,
        cfg=cfg,
        adapter_path=cfg.inference.adapter_path,
        device=device,
    )

    batch = _load_context(cfg)
    logger.info(
        "Forecasting %d steps for %s",
        cfg.inference.prediction_length,
        cfg.data.npy_path,
    )

    forecast = pipeline.predict(
        batch,
        pred_len=cfg.inference.prediction_length,
        mqd_infer=cfg.inference.mqd_infer,
    )

    # The Q axis of quantile_preds is labelled by the forecaster's resolved
    # inference quantiles (== training quantiles for a quantile head).
    quantile_levels = pipeline.forecaster._inference_quantiles

    _save_outputs(
        forecast, quantile_levels, batch["context"], batch["channel_mask"], cfg
    )
    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
