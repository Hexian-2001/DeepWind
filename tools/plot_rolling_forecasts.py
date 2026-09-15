#!/usr/bin/env python
# tools/plot_rolling_forecasts.py
"""Stitch per-window eval predictions into a contiguous rolling forecast and plot.

Reads ``<eval_dir>/raw_results/<dataset>/raw_H<h>.npz`` (already index-sorted,
non-overlapping windows with ``stride == pred_len``) and concatenates consecutive
per-window forecasts into one long series — a *rolling-origin backtest* — then
renders a fixed-length segment (``--steps``, default 128) per (dataset, horizon):
ground truth, median point forecast, and 50% / 90% prediction intervals.

Why this is correct: ``DeepWindTestDataset`` slides its context by exactly
``prediction_length`` each window (stride defaults to ``pred_len``), and the
evaluator re-sorts all samples by their original index. Consecutive windows are
therefore contiguous in time, so ``targets.reshape(-1)`` reproduces one unbroken
slice of the test series. Each window is still an independent forecast
conditioned on its own *true* 8192-step history (``mqd_infer=True``), so the
stitched figure is a backtest view, not a single autonomous multi-step rollout.

The segment start is pinned in ``results/forecast_samples.json``
(``rolling.starts``, auto-created, deterministic ``seed=42``) so every model
plots the *same* contiguous block — directly comparable across models / seeds.

Output: ``reports/forecasts_rolling/<dataset>/H<h>__<model_id>__T<steps>.png``.

Usage:
    python tools/plot_rolling_forecasts.py --model-id deepwind-small-paper-seed42
    python tools/plot_rolling_forecasts.py --model-id ... --steps 256 --horizons H1,H6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))               # tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))        # repo root (src/)
from leaderboard_lib import load_leaderboard  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEADERBOARD = REPO_ROOT / "results" / "leaderboard.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "results" / "forecast_samples.json"
DEFAULT_OUT = REPO_ROOT / "reports" / "forecasts_rolling"

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
    return {"seed": None, "samples": {}, "rolling": {}}


def _save_manifest(path, m):
    Path(path).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _q_idx(quantiles, q):
    return int(np.argmin(np.abs(np.asarray(quantiles) - q)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-id", default=None, help="model_id in the leaderboard")
    ap.add_argument("--eval-dir", default=None, help="explicit eval output dir")
    ap.add_argument("--datasets", default=None,
                    help=f"comma-separated datasets (default: all {len(ALL_DATASETS)})")
    ap.add_argument("--horizons", default="H1,H6",
                    help="comma-separated horizons (default: H1,H6)")
    ap.add_argument("--steps", type=int, default=128,
                    help="fixed number of steps per figure (default 128)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the deterministic start selection")
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
        print(f"no raw_results/ under {eval_dir}", file=sys.stderr)
        return 2

    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else ALL_DATASETS
    horizons = [_parse_horizon(h) for h in args.horizons.split(",")]

    manifest = _load_manifest(args.manifest)
    manifest["seed"] = args.seed
    rolling = manifest.setdefault("rolling", {})
    rolling["steps"] = args.steps
    starts = rolling.setdefault("starts", {})
    dirty = False

    model_id = args.model_id or eval_dir.name
    out = Path(args.out)
    rng = np.random.RandomState(args.seed)   # seed once: starts vary across datasets
    written = []

    for ds in datasets:
        d = starts.setdefault(ds, {})
        for h in horizons:
            npz_path = raw_root / ds / f"raw_H{h}.npz"
            if not npz_path.exists():
                print(f"skip {ds}/H{h}: {npz_path} not found", file=sys.stderr)
                continue

            z = np.load(npz_path, allow_pickle=True)
            P = int(z["targets"].shape[1])          # steps per window
            N = int(z["targets"].shape[0])          # number of windows
            n_win = max(1, min(int(np.ceil(args.steps / P)), N))

            # Pinned start window (manifest) or deterministic fresh draw.
            if h in d and 0 <= int(d[h]) <= N - n_win:
                start = int(d[h])
            else:
                start = int(rng.randint(0, N - n_win + 1))
                d[h] = start
                dirty = True

            sl = slice(start, start + n_win)
            Q = int(z["quantile_preds"].shape[-1])
            targets_full = z["targets"][sl].reshape(-1)[:args.steps]
            point_full = z["point_preds"][sl].reshape(-1)[:args.steps]
            qp_full = z["quantile_preds"][sl].reshape(-1, Q)[:args.steps]
            quantiles = z["quantiles"].tolist()

            i05, i95 = _q_idx(quantiles, 0.05), _q_idx(quantiles, 0.95)
            i25, i75 = _q_idx(quantiles, 0.25), _q_idx(quantiles, 0.75)

            cap = float(z["capacity"])
            ylabel = "Power (normalised)" if cap == 1.0 else "Power (MW)"
            # Physical span: P steps == h hours, so T steps ≈ T/P * h hours.
            span_h = len(targets_full) / P * float(h)

            T = len(targets_full)
            t = np.arange(T)

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.fill_between(t, qp_full[:, i05], qp_full[:, i95], color="dodgerblue",
                            alpha=0.20, label="90% PI", zorder=1)
            ax.fill_between(t, qp_full[:, i25], qp_full[:, i75], color="dodgerblue",
                            alpha=0.40, label="50% PI", zorder=2)
            ax.plot(t, point_full, color="firebrick", lw=1.1, ls="--", alpha=0.9,
                    label="Point forecast (median)", zorder=3)
            ax.plot(t, targets_full, color="green", lw=1.3, alpha=0.9,
                    label="Ground truth", zorder=4)

            for b in range(0, T, P):                 # mark window boundaries
                ax.axvline(b, color="gray", lw=0.4, ls=":", alpha=0.4)

            ax.set_title(
                f"{ds} · H{h}h · rolling-origin backtest  |  "
                f"{T} steps ≈ {span_h:.1f} h  |  window {start}"
            )
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()

            out_dir = out / ds
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f"H{h}__{model_id}__T{T}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            written.append(str(save_path))
            del z

    if dirty:
        _save_manifest(args.manifest, manifest)
        print(f"manifest -> {args.manifest}")

    print(f"wrote {len(written)} rolling forecast figure(s):")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
