#!/usr/bin/env python
# tools/register_baselines.py
"""Register a *baseline* model (TSFM / small model / statistical) into the leaderboard.

Baselines live outside the DeepWind eval pipeline and use a simpler result layout:

    <result_root>/<dataset>/H_<h>.json     # {"dataset", "horizon_hours", "metrics": {...}}

The metric keys are normalised to the DeepWind HEADLINE names (``ncrps`` -> ``nCRPS``,
``nmae`` -> ``nMAE``, etc.), macro-averaged over the selected horizons, and upserted
as one row in ``results/leaderboard.jsonl`` so they appear in ``compare_models.py``.

Usage:
    python tools/register_baselines.py \\
        /scratch/.../deepwind_experiments/baselines/results/chronos-2 \\
        --model-id baseline-chronos-2 --tags baseline,tsfm,zero-shot

    python tools/register_baselines.py .../results/patchtst \\
        --model-id baseline-patchtst --tags baseline,deep-learning,trained
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaderboard_lib import HEADLINE, _finite, upsert_row  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEADERBOARD = REPO_ROOT / "results" / "leaderboard.jsonl"

# Lowercase/normalised -> HEADLINE name (for baseline result JSONs).
_KEY_ALIASES = {
    "ncrps": "nCRPS", "crps": "nCRPS", "normalized_crps": "nCRPS",
    "nmae": "nMAE", "mae": "nMAE", "normalized_mae": "nMAE",
    "accuracy": "Accuracy", "acc": "Accuracy",
    "qualified_rate": "Qualified_Rate", "qualifiedrate": "Qualified_Rate",
    "qr": "Qualified_Rate",
    "mae_coverage": "MAE_Coverage", "coverage": "MAE_Coverage",
    "r2": "R2", "r_squared": "R2",
    "mean_wquantileloss": "mean_wQuantileLoss", "wquantileloss": "mean_wQuantileLoss",
    "mean_wql": "mean_wQuantileLoss",
}

STANDARD_HORIZONS = [1, 2, 4, 6, 8, 12]


def _normalise_key(k: str) -> str | None:
    return _KEY_ALIASES.get(k.lower().replace(" ", "_").replace("-", "_"))


def _read_metric_files(root: Path, horizons: list[int]) -> list[dict]:
    rows = []
    for jf in sorted(root.glob("*/H_*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        ds = d.get("dataset")
        h = d.get("horizon_hours")
        m = d.get("metrics", {})
        if ds is None or h is None or h not in horizons or not isinstance(m, dict):
            continue
        norm = {}
        for k, v in m.items():
            hk = _normalise_key(k)
            if hk and _finite(v):
                norm[hk] = float(v)
        if norm:
            rows.append({"dataset": ds, "horizon": h, **norm})
    return rows


def _build_metrics(rows: list[dict]) -> dict:
    per: dict[str, dict[str, list]] = {}
    for r in rows:
        ds = r["dataset"]
        per.setdefault(ds, {k: [] for k in HEADLINE})
        for k in HEADLINE:
            if _finite(r.get(k)):
                per[ds][k].append(r[k])
    per_dataset = {
        ds: {k: float(statistics.mean(vs)) for k, vs in vals.items() if vs}
        for ds, vals in per.items()
    }
    mean = {}
    for k in HEADLINE:
        vals = [r[k] for r in rows if _finite(r.get(k))]
        mean[k] = float(statistics.mean(vals)) if vals else None

    horizons = sorted({r["horizon"] for r in rows})
    per_h = {}
    for h in horizons:
        vals = [r["nCRPS"] for r in rows if r["horizon"] == h and _finite(r.get("nCRPS"))]
        per_h[f"nCRPS_H{h}"] = float(statistics.mean(vals)) if vals else None

    return {
        "mean": mean,
        "per_dataset": per_dataset,
        "per_horizon_nCRPS": per_h,
        "n_cells": len(rows),
        "n_datasets": len(per_dataset),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("result_root", help="baseline results dir (<root>/<dataset>/H_<h>.json)")
    ap.add_argument("--model-id", required=True, help="leaderboard model_id")
    ap.add_argument("--variant", default="baseline", help="leaderboard variant (default: baseline)")
    ap.add_argument("--tags", default="baseline", help="comma-separated tags")
    ap.add_argument("--horizons", default="1,2,4,6,8,12",
                    help="comma-separated horizons to aggregate (default: DeepWind 6)")
    ap.add_argument("--source", default=None, help="free-text provenance note")
    ap.add_argument("--leaderboard", default=str(DEFAULT_LEADERBOARD))
    args = ap.parse_args()

    root = Path(args.result_root).expanduser()
    if not root.exists():
        print(f"result root not found: {root}", file=sys.stderr)
        return 2

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    rows = _read_metric_files(root, horizons)
    if not rows:
        print(f"no metric files with horizons {horizons} under {root}", file=sys.stderr)
        return 2

    metrics = _build_metrics(rows)
    n_metrics = sum(1 for k in HEADLINE if _finite(metrics["mean"].get(k)))
    print(f"{args.model_id}: {metrics['n_cells']} cells "
          f"({metrics['n_datasets']} datasets x {len(set(r['horizon'] for r in rows))} horizons), "
          f"{n_metrics}/7 headline metrics")

    row = {
        "model_id": args.model_id,
        "variant": args.variant,
        "checkpoint": None,
        "eval_name": args.model_id,
        "eval_dir": str(root),
        "git_commit": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": "baseline",
        "architecture": {},
        "training": {},
        "seed": None,
        "data_seed": None,
        "metrics": metrics,
        "training_outcome": {},
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()],
        "source": args.source,
        "horizons": horizons,
    }
    upsert_row(args.leaderboard, row)
    print(f"registered -> {args.leaderboard} (mean nCRPS={_f(metrics['mean'].get('nCRPS'))})")
    return 0


def _f(v):
    return f"{v:.4f}" if _finite(v) else "—"


if __name__ == "__main__":
    raise SystemExit(main())
