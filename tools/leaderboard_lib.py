# tools/leaderboard_lib.py
"""Shared helpers for the DeepWind leaderboard / model registry.

The leaderboard is a git-committed JSONL file (``results/leaderboard.jsonl``),
one JSON object per evaluated model. This module provides the metric constants,
config hashing, and read/upsert helpers shared by ``register_eval.py``,
``compare_models.py`` and ``export_report.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Headline metrics (mirrors tools/aggregate_eval.py). The first group is
# "lower is better"; the second group is "higher is better".
HEADLINE = [
    "nCRPS", "nMAE", "Accuracy", "Qualified_Rate",
    "MAE_Coverage", "R2", "mean_wQuantileLoss",
]
LOWER_IS_BETTER = {"nCRPS", "nMAE", "MAE_Coverage", "mean_wQuantileLoss"}
HIGHER_IS_BETTER = {"Accuracy", "Qualified_Rate", "R2"}

# Training fields that only describe *where/how* the run is logged/saved and do
# not affect the trained model — excluded from both storage and the config hash.
TRAINING_IDENTITY_KEYS = {
    "output_dir", "run_name", "report_to", "overwrite_output_dir",
}
# Reporting / eval knobs that do not change the model — excluded from the hash
# only (kept in storage for completeness).
TRAINING_REPORTING_KEYS = {
    "logging_strategy", "logging_steps", "logging_first_step",
    "disable_tqdm", "log_on_each_node",
    "save_strategy", "save_steps", "save_total_limit",
    "load_best_model_at_end", "ddp_find_unused_parameters",
    "eval_strategy", "eval_steps", "per_device_eval_batch_size",
    "prediction_loss_only",
}


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and v == v and abs(v) != float("inf")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_hash(cfg: dict[str, Any]) -> str:
    """Deterministic hash of the *design* (model + training + data), excluding
    run identity, reporting knobs and the seed, so that a seed sweep (same
    design, different seed) yields the same hash and groups as one family."""
    training = {
        k: v for k, v in cfg.get("training", {}).items()
        if k not in (TRAINING_IDENTITY_KEYS | TRAINING_REPORTING_KEYS)
    }
    data = {k: v for k, v in cfg.get("data", {}).items() if k != "seed"}
    design = {
        "model": cfg.get("model", {}),
        "training": training,
        "data": data,
        "data_eval": cfg.get("data_eval", {}),
    }
    return hashlib.sha256(canonical_json(design).encode("utf-8")).hexdigest()


def load_leaderboard(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def upsert_row(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    rows = [r for r in load_leaderboard(p) if r.get("model_id") != row.get("model_id")]
    rows.append(row)
    rows.sort(key=lambda r: (str(r.get("variant", "")), str(r.get("model_id", ""))))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def infer_variant(s: str) -> str | None:
    for v in ("small", "base", "large"):
        if v in s.lower():
            return v
    return None
