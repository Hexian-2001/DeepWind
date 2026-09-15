#!/usr/bin/env python
# tools/compare_models.py
"""Compare registered DeepWind models from the leaderboard.

Usage:
    python tools/compare_models.py --summary                    # quick ranking by nCRPS
    python tools/compare_models.py --summary --variant small,base,large
    python tools/compare_models.py --summary --csv ranking.csv  # ranking as CSV
    python tools/compare_models.py --all
    python tools/compare_models.py --variant small,base,large
    python tools/compare_models.py deepwind-small-paper-seed42 deepwind-base-paper-seed42
    python tools/compare_models.py --family <config_hash> --group-family
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaderboard_lib import (  # noqa: E402
    HEADLINE,
    LOWER_IS_BETTER,
    _finite,
    load_leaderboard,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEADERBOARD = REPO_ROOT / "results" / "leaderboard.jsonl"


def _fmt(v):
    if v is None or not _finite(v):
        return "—"
    return f"{v:.4f}"


def _table(header, body):
    ncol = len(header)
    widths = [len(str(header[c])) for c in range(ncol)]
    for row in body:
        for c in range(ncol):
            widths[c] = max(widths[c], len(str(row[c])))

    def line(cells):
        return "  ".join(str(cells[c]).rjust(widths[c]) for c in range(ncol))

    out = [line(header), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in body]
    return "\n".join(out)


def _err(msg):
    print(msg, file=sys.stderr)
    return 2


def _pick(rows, args):
    if args.model_ids:
        ids = set(args.model_ids)
        picked = [r for r in rows if r.get("model_id") in ids]
    elif args.variant:
        vs = {v.strip() for v in args.variant.split(",")}
        picked = [r for r in rows if r.get("variant") in vs]
    elif args.family:
        picked = [r for r in rows if r.get("config_hash") == args.family]
    else:
        picked = rows
    seen = {}
    for r in picked:
        seen[r.get("model_id")] = r
    return list(seen.values())


def _horizon_key(h):
    m = re.search(r"(\d+)$", h)
    return int(m.group(1)) if m else 0


def _best_idx(k, vals):
    idxs = [i for i, v in enumerate(vals) if _finite(v)]
    if not idxs:
        return None
    if k in LOWER_IS_BETTER:
        return min(idxs, key=lambda i: vals[i])
    return max(idxs, key=lambda i: vals[i])


def _mark(vals, k, i, n_models):
    s = _fmt(vals[i])
    if n_models > 1:
        best = _best_idx(k, vals)
        if best is not None and i == best:
            s += "  <--"
    return s


def _group_by_family(rows):
    fams = defaultdict(list)
    for r in rows:
        fams[r.get("config_hash")].append(r)
    return list(fams.values())


def _summary_view(rows, csv_path=None):
    """Compact one-line-per-model ranking sorted by nCRPS (primary metric)."""
    def _ncrps(r):
        v = r.get("metrics", {}).get("mean", {}).get("nCRPS")
        return v if _finite(v) else float("inf")

    rows = sorted(rows, key=_ncrps)
    raw_rows = []
    for i, r in enumerate(rows, 1):
        m = r.get("metrics", {}).get("mean", {})
        o = r.get("training_outcome", {})
        raw_rows.append([
            i, r.get("model_id"), r.get("variant") or "",
            m.get("nCRPS"), m.get("nMAE"), m.get("Accuracy"), m.get("R2"),
            o.get("global_step"), o.get("final_loss"),
        ])

    if csv_path:
        import csv as _csv
        with open(csv_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["rank", "model_id", "variant", "nCRPS", "nMAE",
                        "Accuracy", "R2", "global_step", "final_loss"])
            for row in raw_rows:
                w.writerow(["" if v is None else v for v in row])
        print(f"wrote {csv_path}")
        return 0

    header = ["#", "model_id", "variant", "nCRPS", "nMAE", "Accuracy", "R2", "step", "loss"]
    body = []
    for row in raw_rows:
        body.append([
            row[0], row[1], row[2],
            _fmt(row[3]), _fmt(row[4]), _fmt(row[5]), _fmt(row[6]),
            row[7] if row[7] is not None else "—",
            _fmt(row[8]),
        ])
    print("=== Quick ranking (sorted by nCRPS, lower better) ===")
    print(_table(header, body))
    print("\nnCRPS / nMAE / loss: lower better.  Accuracy / R2: higher better.  #1 = best nCRPS.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_ids", nargs="*", help="explicit model_id(s) to compare")
    ap.add_argument("--all", action="store_true",
                    help="compare every registered model (default when no selection given)")
    ap.add_argument("--variant", default=None, help="comma-separated variants (small,base,large)")
    ap.add_argument("--family", default=None, help="config_hash to compare seeds of")
    ap.add_argument("--group-family", action="store_true",
                    help="aggregate seeds sharing a config_hash into mean±std")
    ap.add_argument("--summary", action="store_true",
                    help="compact one-line-per-model ranking sorted by nCRPS (lower better)")
    ap.add_argument("--csv", default=None, metavar="PATH",
                    help="with --summary: write the ranking as CSV to PATH")
    ap.add_argument("--leaderboard", default=str(DEFAULT_LEADERBOARD))
    args = ap.parse_args()

    rows = load_leaderboard(args.leaderboard)
    if not rows:
        return _err(f"no rows in {args.leaderboard}; run register_eval.py first")

    picked = _pick(rows, args)
    if not picked:
        return _err("no models matched the selection")

    if args.summary:
        return _summary_view(picked, args.csv)

    # --- Seed-family aggregation (mean ± std) ---
    if args.group_family:
        fams = _group_by_family(picked)
        labels, means, stds = [], [], []
        for fam in fams:
            seeds = [r.get("seed") for r in fam]
            label = f"{fam[0].get('variant') or fam[0].get('model_id')} " \
                    f"(seeds {','.join(str(s) for s in seeds)})"
            m, s = {}, {}
            for k in HEADLINE:
                vals = [r.get("metrics", {}).get("mean", {}).get(k) for r in fam]
                vals = [v for v in vals if _finite(v)]
                m[k] = float(statistics.mean(vals)) if vals else None
                s[k] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
            labels.append(label)
            means.append(m)
            stds.append(s)
        header = ["metric"] + labels
        body = []
        for k in HEADLINE:
            cells = []
            for i in range(len(labels)):
                m, sd = means[i].get(k), stds[i].get(k)
                cells.append(f"{_fmt(m)} ± {_fmt(sd)}" if sd else _fmt(m))
            body.append([k] + cells)
        print(_table(header, body))
        return 0

    labels = [r.get("model_id") for r in picked]
    n = len(labels)
    means = [r.get("metrics", {}).get("mean", {}) for r in picked]

    print("=== Macro metrics (mean over 8 datasets x 6 horizons) ===")
    print(_table(["metric"] + labels, [
        [k] + [_mark([m.get(k) for m in means], k, i, n) for i in range(n)]
        for k in HEADLINE
    ]))
    print("\n<-- = best (lower better for nCRPS/nMAE/MAE_Coverage/mean_wQuantileLoss; "
          "higher better for Accuracy/Qualified_Rate/R2)")

    per_ds = [r.get("metrics", {}).get("per_dataset", {}) for r in picked]
    ds_order = sorted({ds for p in per_ds for ds in p})
    for metric in ("nCRPS", "nMAE"):
        body = []
        for ds in ds_order:
            vals = [p.get(ds, {}).get(metric) for p in per_ds]
            body.append([ds] + [_mark(vals, metric, i, n) for i in range(n)])
        print(f"\n=== Per-dataset {metric} (mean over 6 horizons) ===")
        print(_table(["dataset"] + labels, body))

    per_h = [r.get("metrics", {}).get("per_horizon_nCRPS", {}) for r in picked]
    h_order = sorted({h for p in per_h for h in p}, key=_horizon_key)
    if h_order:
        body = []
        for h in h_order:
            vals = [p.get(h) for p in per_h]
            body.append([h] + [_mark(vals, "nCRPS", i, n) for i in range(n)])
        print(f"\n=== Per-horizon nCRPS (mean over 8 datasets) ===")
        print(_table(["horizon"] + labels, body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
