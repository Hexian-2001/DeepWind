#!/usr/bin/env python3
"""Load a released checkpoint and run one real-data forward pass."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.data.datasets import DeepWindEvalDataset
from src.models.deepwind import DeepWindModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = DeepWindModel.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device)
    model.eval()

    dataset = DeepWindEvalDataset(
        npy_root=args.data_root / "eval",
        metadata_path=args.data_root / "eval_metadata.csv",
        seq_len=model.config.context_length,
        max_vars=6,
        pad_val_id=model.config.num_known_variates,
        max_samples_per_file=1,
    )
    sample = dataset[0]
    def tensor(key: str) -> torch.Tensor:
        return torch.from_numpy(sample[key]).unsqueeze(0).to(args.device)

    with torch.inference_mode(), torch.autocast(
        device_type=torch.device(args.device).type,
        dtype=torch.bfloat16,
        enabled=args.device.startswith("cuda"),
    ):
        output = model(
            context=tensor("context"),
            site_coords=tensor("site_coords"),
            variate_ids=tensor("variate_ids"),
            channel_mask=tensor("channel_mask"),
            has_coords=tensor("has_coords"),
        )

    if output.loss is None or not math.isfinite(float(output.loss)):
        raise RuntimeError(f"Non-finite checkpoint smoke loss: {output.loss}")
    expected = (
        1,
        6,
        model.config.context_length,
        len(model.config.quantiles),
    )
    if tuple(output.pred_params.shape) != expected:
        raise RuntimeError(f"Prediction shape {tuple(output.pred_params.shape)} != {expected}")

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "loss": float(output.loss),
                "prediction_shape": list(output.pred_params.shape),
                "dtype": str(output.pred_params.dtype),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
