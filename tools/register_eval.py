#!/usr/bin/env python
# tools/register_eval.py
"""Register one evaluation run into the DeepWind leaderboard.

Reads the provenance + metrics produced by ``evaluate.py`` and the resolved
training config from the checkpoint's ``run_info.json``, then upserts a single
row into ``results/leaderboard.jsonl``.

Usage:
    python tools/register_eval.py <eval_output_dir> [--tags a,b] [--variant small] [--model-id X]

Example:
    python tools/register_eval.py \\
        /scratch/pawsey0115/hwang4/projects/deepwind/runs/results/deepwind/eval-small-paper \\
        --variant small --tags paper-spec,baseline,seed42
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaderboard_lib import (  # noqa: E402
    HEADLINE,
    TRAINING_IDENTITY_KEYS,
    _finite,
    config_hash,
    infer_variant,
    upsert_row,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEADERBOARD = REPO_ROOT / "results" / "leaderboard.jsonl"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _err(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 2


def _load_metrics(eval_dir: Path) -> dict:
    """Return {mean, per_dataset, per_horizon_nCRPS, n_cells} from eval output."""
    agg_path = eval_dir / "_aggregate.json"
    agg = _read_json(agg_path) if agg_path.exists() else {}

    per: dict[str, dict[str, list]] = {}
    for jf in sorted(eval_dir.glob("*/H_*.json")):
        d = _read_json(jf)
        ds = d.get("dataset")
        m = d.get("metrics", {})
        if ds is None:
            continue
        per.setdefault(ds, {k: [] for k in HEADLINE})
        for k in HEADLINE:
            v = m.get(k)
            if _finite(v):
                per[ds][k].append(v)
    per_dataset = {
        ds: {k: float(statistics.mean(vs)) for k, vs in vals.items() if vs}
        for ds, vals in per.items()
    }

    return {
        "mean": agg.get("mean", {}),
        "per_dataset": per_dataset,
        "per_horizon_nCRPS": agg.get("per_horizon_nCRPS", {}),
        "n_cells": agg.get("n_cells", len(list(eval_dir.glob("*/H_*.json")))),
    }


def _resolve_checkpoint(eval_dir: Path) -> Path:
    ri = _read_json(eval_dir / "run_info.json")
    cp = ri.get("config", {}).get("inference", {}).get("checkpoint_path")
    if not cp:
        raise SystemExit(_err(f"no checkpoint_path in {eval_dir}/run_info.json"))
    return Path(cp)


def _resolve_training_run_info(checkpoint_path: Path) -> dict | None:
    """Return the training run's ``run_info.json`` (checkpoints/run_info.json),
    or None if unavailable (older checkpoints without provenance)."""
    run_info = checkpoint_path.parent / "run_info.json"
    if run_info.exists():
        return _read_json(run_info)
    return None


def _load_outcome(checkpoint_path: Path) -> dict:
    outcome: dict = {}
    ts = checkpoint_path / "trainer_state.json"
    if ts.exists():
        t = _read_json(ts)
        outcome["global_step"] = t.get("global_step")
        losses = [
            x.get("loss") for x in t.get("log_history", [])
            if "loss" in x and _finite(x.get("loss"))
        ]
        outcome["final_loss"] = float(losses[-1]) if losses else None
    return outcome


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("eval_dir", help="evaluation output dir (contains run_info.json + H_*.json)")
    ap.add_argument("--tags", default="", help="comma-separated tags")
    ap.add_argument("--variant", default=None, help="small|base|large (auto-inferred if omitted)")
    ap.add_argument("--model-id", default=None, help="override model_id (default: eval run_name)")
    ap.add_argument("--leaderboard", default=str(DEFAULT_LEADERBOARD),
                    help="path to leaderboard.jsonl (default: repo results/leaderboard.jsonl)")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).expanduser()
    if not (eval_dir / "run_info.json").exists():
        return _err(f"{eval_dir} is not an eval output dir (missing run_info.json)")

    eval_ri = _read_json(eval_dir / "run_info.json")
    eval_git_commit = eval_ri.get("git", {}).get("commit")
    eval_name = eval_ri.get("config", {}).get("run_name") or eval_dir.name

    checkpoint_path = _resolve_checkpoint(eval_dir)
    model_id = args.model_id or eval_name
    variant = args.variant or infer_variant(model_id) or infer_variant(str(checkpoint_path))

    architecture: dict = {}
    training: dict = {}
    seed = None
    data_seed = None
    training_git_commit = None
    train_ri = _resolve_training_run_info(checkpoint_path)
    if train_ri is not None:
        cfg = train_ri.get("config", {})
        architecture = cfg.get("model", {})
        training = {
            k: v for k, v in cfg.get("training", {}).items()
            if k not in TRAINING_IDENTITY_KEYS
        }
        seed = cfg.get("seed")
        data_seed = (cfg.get("data") or {}).get("seed")
        training_git_commit = train_ri.get("git", {}).get("commit")
        chash = config_hash(cfg)
    else:
        # Fallback: no checkpoints/run_info.json — hash what we have.
        chash = config_hash({"model": {}, "training": {}, "data": {}, "data_eval": {}})

    row = {
        "model_id": model_id,
        "variant": variant,
        "checkpoint": str(checkpoint_path),
        "eval_name": eval_name,
        "eval_dir": str(eval_dir),
        "git_commit": eval_git_commit,
        "training_git_commit": training_git_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": chash,
        "architecture": architecture,
        "training": training,
        "seed": seed,
        "data_seed": data_seed,
        "metrics": _load_metrics(eval_dir),
        "training_outcome": _load_outcome(checkpoint_path),
        "tags": [t for t in args.tags.split(",") if t],
    }

    upsert_row(args.leaderboard, row)
    summary = {
        k: row[k] for k in ("model_id", "variant", "config_hash", "seed",
                            "training_outcome") if k in row
    }
    summary["mean_metrics"] = row["metrics"]["mean"]
    print(json.dumps(summary, indent=2))
    print(f"\nregistered -> {args.leaderboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
