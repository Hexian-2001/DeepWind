#!/usr/bin/env python
# tools/plot_results.py
"""Generate publication-quality figures from the DeepWind leaderboard.

Reads ``results/leaderboard.jsonl`` and writes a standard figure set to
``--out`` (default ``reports/figures/``):

    ranking_ncrps.png       headline: mean nCRPS per model, sorted (lower better)
    macro_metrics.png       one panel per HEADLINE metric
    per_dataset_ncrps.png   nCRPS grouped by benchmark dataset
    per_horizon_ncrps.png   nCRPS vs forecast horizon, one line per model
    seed_sweep_<v>.png      per-horizon lines per seed (only when >1 seed shares
                            a config_hash)

Figures are regenerable views over the leaderboard; the leaderboard JSONL (not
the PNGs) is the committed source of truth.

Usage:
    python tools/plot_results.py                            # default out dir
    python tools/plot_results.py --variant small,base,large
    python tools/plot_results.py --formats png,svg --dpi 200
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaderboard_lib import (  # noqa: E402
    HEADLINE,
    LOWER_IS_BETTER,
    _finite,
    load_leaderboard,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEADERBOARD = REPO_ROOT / "results" / "leaderboard.jsonl"
DEFAULT_OUT = REPO_ROOT / "reports" / "figures"

VARIANT_COLOR = {"small": "#1f77b4", "base": "#ff7f0e", "large": "#2ca02c"}
_FALLBACK = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

_LOWER_NOTE = ("nCRPS / nMAE / MAE_Coverage / mean_wQuantileLoss: lower better  ·  "
               "Accuracy / Qualified_Rate / R2: higher better")


def _variant_color(variant, i):
    if variant in VARIANT_COLOR:
        return VARIANT_COLOR[variant]
    return _FALLBACK[i % len(_FALLBACK)]


def _variant_legend_handles(rows):
    seen = {}
    for i, r in enumerate(rows):
        v = r.get("variant")
        if v and v not in seen:
            seen[v] = _variant_color(v, i)
    return [mpatches.Patch(color=c, label=v) for v, c in seen.items()]


def _set_style():
    plt.rcParams.update({
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "axes.axisbelow": True,
    })


def _save(fig, stem, formats):
    written = []
    for fmt in formats:
        p = Path(f"{stem}.{fmt}")
        fig.savefig(p)
        written.append(str(p))
    plt.close(fig)
    return written


def _horizon_key(h):
    m = re.search(r"(\d+)$", h)
    return int(m.group(1)) if m else 0


def _mean(r, key):
    return r.get("metrics", {}).get("mean", {}).get(key)


def _label_bars(ax, ys, vals, fmt="{:.3f}"):
    """Annotate each bar with its value, offset just past the bar end."""
    finite = [v for v in vals if _finite(v)]
    xmax = max(finite) if finite else 0.0
    pad = xmax * 0.02 or 0.004
    for y, v in zip(ys, vals):
        if _finite(v):
            ax.text(v + pad, y, fmt.format(v), va="center", fontsize=8)


# --- Figure builders ---------------------------------------------------------

def _plot_ranking(rows, out, formats):
    order = sorted(rows, key=lambda r: _mean(r, "nCRPS") or float("inf"))
    labels = [r.get("model_id") for r in order]
    vals = [_mean(r, "nCRPS") for r in order]
    colors = [_variant_color(r.get("variant"), i) for i, r in enumerate(order)]

    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.45 * len(order) + 1.2)))
    ys = list(range(len(order)))[::-1]
    ax.barh(ys, [v if _finite(v) else 0 for v in vals], color=colors,
            height=0.6, edgecolor="white", linewidth=0.8)
    _label_bars(ax, ys, vals, fmt="{:.4f}")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean nCRPS (lower is better)")
    ax.set_title("Model ranking by mean nCRPS")
    handles = _variant_legend_handles(order)
    if handles:
        ax.legend(handles=handles, loc="lower right")
    return _save(fig, out / "ranking_ncrps", formats)


def _plot_macro(rows, out, formats):
    order = sorted(rows, key=lambda r: r.get("model_id") or "")
    labels = [r.get("model_id") for r in order]
    colors = [_variant_color(r.get("variant"), i) for i, r in enumerate(order)]

    ncols, n = 4, len(HEADLINE)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
    axes = list(axes.flat)

    for i, k in enumerate(HEADLINE):
        ax = axes[i]
        vals = [_mean(r, k) for r in order]
        finite_idx = [j for j, v in enumerate(vals) if _finite(v)]
        ys = list(range(len(order)))[::-1]
        ax.barh(ys, [v if _finite(v) else 0 for v in vals], color=colors,
                height=0.6, edgecolor="white", linewidth=0.6)
        if finite_idx:
            best = (min if k in LOWER_IS_BETTER else max)(finite_idx, key=lambda j: vals[j])
            ax.barh(ys[best], vals[best], color=colors[best], height=0.6,
                    edgecolor="black", linewidth=1.4)
        _label_bars(ax, ys, vals)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_title(f"{k} {'↓' if k in LOWER_IS_BETTER else '↑'}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Macro metrics (mean over 8 datasets x 6 horizons)", fontsize=12)
    handles = _variant_legend_handles(order)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles))
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return _save(fig, out / "macro_metrics", formats)


def _plot_per_dataset(rows, out, formats):
    order = sorted(rows, key=lambda r: r.get("model_id") or "")
    ds_order = sorted({ds for r in rows
                       for ds in (r.get("metrics", {}).get("per_dataset", {}) or {})})
    if not ds_order:
        return []

    n = len(order)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(ds_order)), 4))
    xs = list(range(len(ds_order)))
    for i, r in enumerate(order):
        vals = [r.get("metrics", {}).get("per_dataset", {}).get(ds, {}).get("nCRPS")
                for ds in ds_order]
        offset = (i - (n - 1) / 2) * width
        ax.bar([x + offset for x in xs], [v if _finite(v) else 0 for v in vals],
               width=width, color=_variant_color(r.get("variant"), i),
               label=r.get("model_id"), edgecolor="white", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(ds_order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("nCRPS (lower is better)")
    ax.set_title("Per-dataset nCRPS (mean over horizons)")
    if n > 1:
        ax.legend()
    return _save(fig, out / "per_dataset_ncrps", formats)


def _plot_per_horizon(rows, out, formats):
    order = sorted(rows, key=lambda r: r.get("model_id") or "")
    h_order = sorted({h for r in rows
                      for h in (r.get("metrics", {}).get("per_horizon_nCRPS", {}) or {})},
                     key=_horizon_key)
    if not h_order:
        return []

    xs = [_horizon_key(h) for h in h_order]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(order):
        vals = [r.get("metrics", {}).get("per_horizon_nCRPS", {}).get(h)
                for h in h_order]
        ax.plot(xs, [v if _finite(v) else float("nan") for v in vals], marker="o",
                linewidth=1.5, color=_variant_color(r.get("variant"), i),
                label=r.get("model_id"))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"H{x}" for x in xs])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("nCRPS (lower is better)")
    ax.set_title("Per-horizon nCRPS (mean over datasets)")
    if len(order) > 1:
        ax.legend()
    return _save(fig, out / "per_horizon_ncrps", formats)


def _plot_seed_sweep(rows, out, formats):
    fams = defaultdict(list)
    for r in rows:
        fams[r.get("config_hash")].append(r)

    written = []
    for chash, group in fams.items():
        if len(group) < 2:
            continue
        h_order = sorted({h for r in group
                          for h in (r.get("metrics", {}).get("per_horizon_nCRPS", {}) or {})},
                         key=_horizon_key)
        if not h_order:
            continue
        xs = [_horizon_key(h) for h in h_order]
        label = group[0].get("variant") or "family"
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in sorted(group, key=lambda r: r.get("seed") or 0):
            vals = [r.get("metrics", {}).get("per_horizon_nCRPS", {}).get(h)
                    for h in h_order]
            ax.plot(xs, [v if _finite(v) else float("nan") for v in vals],
                    marker="o", linewidth=1.3, label=f"seed {r.get('seed')}")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"H{x}" for x in xs])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("nCRPS (lower is better)")
        ax.set_title(f"Seed sweep — {label} (config_hash {chash[:8]})")
        ax.legend()
        written += _save(fig, out / f"seed_sweep_{label}", formats)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--leaderboard", default=str(DEFAULT_LEADERBOARD))
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output dir (default: reports/figures)")
    ap.add_argument("--variant", default=None, help="comma-separated variants (small,base,large)")
    ap.add_argument("--all", action="store_true", help="plot every registered model (default)")
    ap.add_argument("--formats", default="png", help="comma-separated output formats (png,svg,pdf)")
    ap.add_argument("--dpi", type=int, default=150, help="savefig DPI for raster formats")
    args = ap.parse_args()

    rows = load_leaderboard(args.leaderboard)
    if not rows:
        print(f"no rows in {args.leaderboard}; run register_eval.py first", file=sys.stderr)
        return 2
    if args.variant:
        vs = {v.strip() for v in args.variant.split(",")}
        rows = [r for r in rows if r.get("variant") in vs]
    if not rows:
        print("no models matched the selection", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    plt.rcParams["savefig.dpi"] = args.dpi
    _set_style()

    written = []
    written += _plot_ranking(rows, out, formats)
    written += _plot_macro(rows, out, formats)
    written += _plot_per_dataset(rows, out, formats)
    written += _plot_per_horizon(rows, out, formats)
    written += _plot_seed_sweep(rows, out, formats)

    if written:
        print(f"wrote {len(written)} figure(s) to {out}:")
        for p in written:
            print(f"  {p}")
    else:
        print("nothing to plot (no plottable metrics in selection)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
