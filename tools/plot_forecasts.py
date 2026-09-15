#!/usr/bin/env python
# tools/plot_forecasts.py
"""Generate forecast visualisations (prediction intervals) from eval raw results.

Reads the per-(dataset, horizon) raw .npz files produced by ``evaluate.py``
(``<eval_dir>/raw_results/<dataset>/raw_H<h>.npz``) and renders one figure per
(dataset, horizon) with ``--num-plots`` stacked forecast windows: history,
ground truth, point forecast, and 50% / 90% prediction intervals.

The exact sample windows are pinned in a shared manifest (default
``results/forecast_samples.json``) so every model plots the *same* samples —
enabling side-by-side comparison across models / seeds. Re-run with the same
``--horizons`` and the manifest is reused verbatim; new horizon/dataset entries
are appended deterministically.

Output: ``reports/forecasts/<dataset>/H<h>__<model_id>.png`` (git-ignored).

Usage:
    python tools/plot_forecasts.py --model-id deepwind-small-paper-seed42
    python tools/plot_forecasts.py --model-id ... --horizons H1,H6 --num-plots 3
    python tools/plot_forecasts.py --model-id ... --datasets 30651,penmanshiel_15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))               # tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))        # repo root (src/)
from leaderboard_lib import load_leaderboard  # noqa: E402
from src.utils.vis import visualize_forecasts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEADERBOARD = REPO_ROOT / "results" / "leaderboard.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "results" / "forecast_samples.json"
DEFAULT_OUT = REPO_ROOT / "reports" / "forecasts"

ALL_DATASETS = ["30651", "43458", "75354", "76016", "csg_wind_5",
                "gefc12_wind_7", "gefc14_wind_10", "penmanshiel_15"]


def _parse_horizon(s: str) -> str:
    return s.strip().upper().lstrip("H")


def _resolve_eval_dir(args, leaderboard) -> Path:
    if args.eval_dir:
        return Path(args.eval_dir).expanduser()
    for r in leaderboard:
        if r.get("model_id") == args.model_id:
            return Path(r["eval_dir"])
    raise SystemExit(f"model_id {args.model_id!r} not in leaderboard; pass --eval-dir")


def _load_manifest(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"num_plots": None, "plot_hist_len": None, "seed": None, "samples": {}}


def _save_manifest(path, m):
    Path(path).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-id", default=None,
                    help="model_id in the leaderboard (used to locate eval_dir)")
    ap.add_argument("--eval-dir", default=None,
                    help="explicit eval output dir (overrides --model-id lookup)")
    ap.add_argument("--datasets", default=None,
                    help=f"comma-separated datasets (default: all {len(ALL_DATASETS)})")
    ap.add_argument("--horizons", default="H1,H6",
                    help="comma-separated horizons (default: H1,H6)")
    ap.add_argument("--num-plots", type=int, default=3,
                    help="samples per (dataset, horizon) (default 3)")
    ap.add_argument("--plot-hist-len", type=int, default=144,
                    help="history steps shown (default 144)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the initial deterministic sample selection")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--leaderboard", default=str(DEFAULT_LEADERBOARD))
    args = ap.parse_args()

    if not args.model_id and not args.eval_dir:
        ap.error("one of --model-id or --eval-dir is required")

    leaderboard = load_leaderboard(args.leaderboard)
    eval_dir = _resolve_eval_dir(args, leaderboard)
    raw_root = eval_dir / "raw_results"
    if not raw_root.exists():
        print(f"no raw_results/ under {eval_dir} — was save_raw_results enabled?",
              file=sys.stderr)
        return 2

    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else ALL_DATASETS
    horizons = [_parse_horizon(h) for h in args.horizons.split(",")]

    manifest = _load_manifest(args.manifest)
    manifest["num_plots"] = args.num_plots
    manifest["plot_hist_len"] = args.plot_hist_len
    manifest["seed"] = args.seed
    samples = manifest.setdefault("samples", {})
    dirty = False

    model_id = args.model_id or eval_dir.name
    out = Path(args.out)
    written = []

    for ds in datasets:
        d = samples.setdefault(ds, {})
        for h in horizons:
            npz_path = raw_root / ds / f"raw_H{h}.npz"
            if not npz_path.exists():
                print(f"skip {ds}/H{h}: {npz_path} not found", file=sys.stderr)
                continue
            z = np.load(npz_path, allow_pickle=True)
            n = int(z["targets"].shape[0])
            n_plot = min(args.num_plots, n)

            if h in d and len(d[h]) == n_plot:
                idx = [int(i) for i in d[h] if 0 <= int(i) < n]
            else:
                idx = sorted(int(i) for i in
                             np.random.RandomState(args.seed).choice(n, size=n_plot, replace=False))
                d[h] = idx
                dirty = True

            cap = float(z["capacity"])
            ylabel = "Power (normalised)" if cap == 1.0 else "Power (MW)"
            results = {
                "targets": z["targets"],
                "point_preds": z["point_preds"],
                "quantile_preds": z["quantile_preds"],
                "history": z["history"],
            }
            p = visualize_forecasts(
                results=results,
                quantiles_list=z["quantiles"].tolist(),
                save_dir=str(out / ds),
                dataset_name=f"{ds} · H{h}",
                num_plots=n_plot,
                plot_hist_len=args.plot_hist_len,
                seed=args.seed,
                indices=idx,
                fname=f"H{h}__{model_id}",
                ylabel=ylabel,
            )
            written.append(p)
            del z

    if dirty:
        _save_manifest(args.manifest, manifest)
        print(f"manifest -> {args.manifest}")

    print(f"wrote {len(written)} forecast figure(s):")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
