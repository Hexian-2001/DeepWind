# tools/aggregate_eval.py
"""Aggregate per-(dataset, horizon) eval metric JSONs into a summary table.

Reads every ``<output_dir>/<dataset>/H_<h>.json`` produced by ``evaluate.py``
and computes the macro-average (and per-horizon average) of the headline
metrics, so two checkpoints can be compared on a single number each.

Usage:
    python tools/aggregate_eval.py <eval_output_dir>

Writes ``<eval_output_dir>/_aggregate.json`` and prints the same to stdout.
"""
import json
import sys
from pathlib import Path

import numpy as np

HEADLINE = ["nCRPS", "nMAE", "Accuracy", "Qualified_Rate", "MAE_Coverage", "R2", "mean_wQuantileLoss"]


def _finite(v):
    return isinstance(v, (int, float)) and v == v and abs(v) != float("inf")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])

    rows = []
    for jf in sorted(root.glob("*/H_*.json")):
        d = json.loads(jf.read_text())
        m = d.get("metrics", {})
        rows.append({
            "dataset": d.get("dataset"),
            "horizon": d.get("horizon_hours"),
            **{k: m.get(k, float("nan")) for k in HEADLINE},
        })

    if not rows:
        print(f"no metric files found under {root}")
        return 1

    def mean_over(key, pred=None):
        vals = [r[key] for r in rows if pred is None or pred(r)]
        vals = [v for v in vals if _finite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    horizons = sorted({r["horizon"] for r in rows})
    per_h = {}
    for h in horizons:
        per_h[f"nCRPS_H{h}"] = mean_over("nCRPS", lambda r: r["horizon"] == h)

    agg = {k: mean_over(k) for k in HEADLINE}

    out = {
        "n_datasets": len({r["dataset"] for r in rows}),
        "n_horizons": len(horizons),
        "n_cells": len(rows),
        "mean": agg,
        "per_horizon_nCRPS": per_h,
    }

    print(json.dumps(out, indent=2))
    (root / "_aggregate.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\naggregate written → {root / '_aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
