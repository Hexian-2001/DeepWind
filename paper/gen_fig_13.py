"""
paper/gen_fig13.py

Batch Figure 13 generator for csg_wind_5.

Workflow:
    1. Run one full pass over the test set to compute global routing stats.
    2. Scan every sample; generate a figure for each sample that contains
       all three regimes above the minimum fraction thresholds.
    3. Save a compact thumbnail grid for quick visual browsing.

Output layout (--output_dir):
    individual/fig13_csg_wind_5_000.pdf
    ...
    thumbnail_grid.pdf

Usage:
    python ./paper/gen_fig13.py \
        [--min_boundary_span 5] \
        [--max_figures 30]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper.moe_analysis import (
    FILENAME_MAP,
    REGIME_NAMES, REGIME_COLORS, REGIME_SHADE,
    NUM_EXPERTS, PATCH_SIZE, TARGET_MAX_WINDOWS,
    get_regime_labels,
    extract_last_moe_outputs,
    _get_contiguous_spans,
    analyze_dataset,
)

logger = logging.getLogger(__name__)

DATASET_NAME = "csg_wind_5"
CAPACITY     = 35.0


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helper
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_dominant_experts_global(
    ax:            plt.Axes,
    activation:    np.ndarray,   # (E, N_tokens) bool
    regime_labels: np.ndarray,   # (N_tokens,)
    global_stats:  List[Dict],
    min_span_len:  int,
) -> None:
    """
    Annotate sufficiently long regime spans with the globally dominant expert
    and its selection frequency (fraction of span tokens where it was in top-2).
    Selection frequency matches the expert_usage metric in the paper table.
    """
    dom_map = {s["regime_name"]: s["dominant_expert"] for s in global_stats}

    for (start_t, end_t, r) in _get_contiguous_spans(regime_labels):
        if end_t - start_t < min_span_len:
            continue
        regime_name = REGIME_NAMES[r]
        dom = dom_map.get(regime_name)
        if dom is None:
            continue
        frac  = float(activation[dom, start_t:end_t].mean())
        mid_t = (start_t + end_t) // 2
        ax.annotate(
            f"E{dom}\n{frac:.0%}",
            xy=(mid_t, 0.9),
            xycoords=("data", "axes fraction"),
            fontsize=7.5, ha="center",
            color=REGIME_COLORS[r], fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white",
                ec=REGIME_COLORS[r], alpha=0.75, linewidth=0.8,
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_figure13(
    power_raw:         np.ndarray,   # (T,) MW
    gate_probs:        np.ndarray,   # (N_tokens, E)
    regime_labels:     np.ndarray,   # (N_tokens,)
    capacity:          float,
    global_stats:      List[Dict],
    dataset_name:      str,
    save_path:         str,
    min_boundary_span: int = 5,
) -> None:
    """
    Generate one upgraded Figure 13.

    Alignment:
        Both panels created with sharex=True so they share one x-axis.
        constrained_layout=True replaces tight_layout and correctly
        respects the shared-axis constraint, giving pixel-perfect column
        alignment between the power curve and the activation bars.

    Boundary lines:
        A dashed vertical line is drawn at every regime transition.
        The same axvline call is made on BOTH axes so the line appears
        continuous across both panels (no gap).  Only transitions where
        both adjacent spans are >= min_boundary_span are drawn to keep
        the figure clean.

    Bottom panel:
        Binary activation map — each cell filled if that expert was in
        the top-2 for that token.  Regime-colour if globally dominant,
        grey otherwise.  No colourbar needed.
    """
    N_tokens, E = gate_probs.shape
    P           = PATCH_SIZE
    spans       = _get_contiguous_spans(regime_labels)

    # ── Pre-compute activation and dominance ──────────────────────────────────
    top2       = np.argsort(gate_probs, axis=-1)[:, -2:]   # (N_tokens, 2)
    dom_map    = {s["regime_name"]: s["dominant_expert"] for s in global_stats}
    activation = np.zeros((E, N_tokens), dtype=bool)
    is_dom     = np.zeros((E, N_tokens), dtype=bool)

    for t in range(N_tokens):
        r       = int(regime_labels[t])
        dom_exp = dom_map.get(REGIME_NAMES[r], -1)
        for exp in top2[t]:
            activation[exp, t] = True
            is_dom[exp, t]     = (exp == dom_exp)
    
    # ── Legend entries (built before figure so expert IDs are known) ──────────
    legend_elements = [
        Patch(facecolor=REGIME_COLORS[0], alpha=0.9,
              label=f"Idle dominant (E{dom_map.get('Idle', '?')})"),
        Patch(facecolor=REGIME_COLORS[1], alpha=0.9,
              label=f"Ramping dominant (E{dom_map.get('Ramping', '?')})"),
        Patch(facecolor=REGIME_COLORS[2], alpha=0.9,
              label=f"Saturation dominant (E{dom_map.get('Saturation', '?')})"),
        Patch(facecolor="#BBBBBB", alpha=0.6, label="Other activated"),
    ]

    # ── Figure: sharex + constrained_layout for strict alignment ─────────────
    fig, (ax_p, ax_h) = plt.subplots(
        2, 1,
        figsize=(13, 5.5),
        gridspec_kw={"height_ratios": [1.7, 1]},
        sharex=True,
        constrained_layout=True,   # replaces tight_layout; respects sharex
    )

    # ── Top panel: power curve ────────────────────────────────────────────────
    legend_added = set()
    for (s, e, r) in spans:
        lbl = REGIME_NAMES[r] if r not in legend_added else None
        ax_p.axvspan(s, e, alpha=0.22, color=REGIME_SHADE[r],
                     label=lbl, zorder=1)
        legend_added.add(r)

    # x-axis in token units so both panels are on the same scale
    t_token = np.arange(len(power_raw)) / P   # shape (T,), range [0, N_tokens)
    ax_p.plot(t_token, power_raw / capacity,
              color="#2C7BB6", lw=1.0, zorder=3, label="Power")
    ax_p.set_ylabel("Normalised Power (P/Cap)", fontsize=10)
    ax_p.set_title(
        f"Power Curve with Regime Segmentation and MoE Activation",
        fontsize=11,
    )
    ax_p.set_xlim(0, N_tokens)
    ax_p.set_ylim(-0.05, 1.20)
    ax_p.grid(alpha=0.3, linewidth=0.4)

    _annotate_dominant_experts_global(
        ax_p, activation, regime_labels,
        global_stats=global_stats,
        min_span_len=min_boundary_span,
    )
    ax_p.legend(
        fontsize   = 8,
        loc        = "lower right",
        bbox_to_anchor = (1.0, 1.01),  # 紧贴ax_p顶边外侧
        ncol       = 4,
        framealpha = 0.85,
        borderaxespad = 0,
    )

    # ── Bottom panel: binary activation map ───────────────────────────────────
    bar_height = 0.80

    for e_idx in range(E):
        for t in range(N_tokens):
            if not activation[e_idx, t]:
                continue
            r     = int(regime_labels[t])
            color = REGIME_COLORS[r] if is_dom[e_idx, t] else "#BBBBBB"
            alpha = 0.90            if is_dom[e_idx, t] else 0.40
            ax_h.bar(
                t, bar_height,
                bottom    = e_idx + (1.0 - bar_height) / 2.0,
                width     = 1.0,
                align     = 'edge',   # bar spans [t, t+1] not [t-0.5, t+0.5]
                color     = color,
                alpha     = alpha,
                linewidth = 0,
            )

    ax_h.set_ylim(0, E)
    ax_h.set_yticks(np.arange(E) + 0.5)
    ax_h.set_yticklabels([f"E{i}" for i in range(E)], fontsize=9)
    ax_h.set_xlabel("Token Index", fontsize=10)
    ax_h.set_ylabel("Expert", fontsize=10)
    ax_h.invert_yaxis()   # E0 at top

    fig.legend(
        handles=legend_elements, fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        framealpha=0.85,
    )

    # ── Cross-panel dashed boundary lines ────────────────────────────────────
    # Draw at every regime transition, no span-length filter.
    for (s, e, r) in spans:
        if s == 0:
            continue
        x_pos = float(s)
        ax_p.axvline(x_pos, color="#444444", lw=0.8, ls="--",
                     alpha=0.55, zorder=4)
        ax_h.axvline(x_pos, color="#444444", lw=0.8, ls="--",
                     alpha=0.55, zorder=4)

    # No tight_layout call — constrained_layout handles spacing automatically
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail grid
# ─────────────────────────────────────────────────────────────────────────────

def save_thumbnail_grid(
    samples:      List[Dict],
    global_stats: List[Dict],
    capacity:     float,
    dataset_name: str,
    save_path:    str,
    ncols:        int = 4,
) -> None:
    """Compact grid of power-curve thumbnails for quick visual browsing."""
    n     = len(samples)
    nrows = (n + ncols - 1) // ncols
    P     = PATCH_SIZE

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 2.5 * nrows),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, sample in enumerate(samples):
        row, col  = divmod(idx, ncols)
        ax        = axes[row, col]
        power_raw = sample["power_raw"]
        rl        = sample["regime_labels"]
        N_tokens  = sample["gate_probs"].shape[0]

        legend_added = set()
        for (s, e, r) in _get_contiguous_spans(rl):
            lbl = REGIME_NAMES[r] if r not in legend_added else None
            ax.axvspan(s, e, alpha=0.25, color=REGIME_SHADE[r],
                       label=lbl, zorder=1)
            legend_added.add(r)

        t_token = np.arange(len(power_raw)) / P
        ax.plot(t_token, power_raw / capacity, color="#2C7BB6", lw=0.8, zorder=3)
        ax.set_xlim(0, N_tokens - 1)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"#{sample['global_idx']:03d}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=7)

        f0 = (rl == 0).mean()
        f1 = (rl == 1).mean()
        f2 = (rl == 2).mean()
        ax.text(0.02, 0.97, f"I:{f0:.0%} R:{f1:.0%} S:{f2:.0%}",
                transform=ax.transAxes, fontsize=6.5, va="top", color="#333333")

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"{dataset_name} — thumbnail grid ({n} samples)",
                 fontsize=11, y=1.01)
    fig.tight_layout(pad=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[saved] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main(args) -> None:
    from omegaconf import OmegaConf
    from src.data.datasets import DeepWindTestDataset
    from src.inference.pipeline import InferencePipeline

    out_dir = Path(args.output_dir)
    ind_dir = out_dir / "individual"
    ind_dir.mkdir(parents=True, exist_ok=True)

    cfg      = OmegaConf.load(args.config_path)
    pipeline = InferencePipeline(checkpoint_path=args.checkpoint_path, cfg=cfg)
    device   = pipeline.device

    filename = FILENAME_MAP.get(DATASET_NAME, f"{DATASET_NAME}.npy")
    npy_path = Path(args.npy_root) / filename

    data     = np.load(npy_path, mmap_mode="r")
    T        = data.shape[-1]
    test_len = int(T * 0.2)
    ctx_len  = args.context_length
    pred_len = args.prediction_length

    if test_len < ctx_len + pred_len:
        ctx_len = max(512, test_len // 2)
        logger.info("Short dataset, falling back to ctx_len=%d", ctx_len)

    available = max(1, test_len - ctx_len - pred_len + 1)
    stride    = max(pred_len, available // TARGET_MAX_WINDOWS)

    test_ds = DeepWindTestDataset(
        npy_path          = npy_path,
        metadata_path     = args.metadata_path,
        context_length    = ctx_len,
        prediction_length = pred_len,
        split             = "test",
        stride            = stride,
    )

    def make_loader():
        return DataLoader(test_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    # ── Pass 1: global stats ──────────────────────────────────────────────────
    logger.info("Pass 1: computing global routing statistics...")
    global_stats, _, _ = analyze_dataset(
        pipeline=pipeline, dataloader=make_loader(),
        capacity=CAPACITY, dataset_name=DATASET_NAME,
    )
    logger.info("Global stats:")
    for s in global_stats:
        logger.info("  %-12s dom=E%d(%.1f%%)  H=%.3f bits",
                    s["regime_name"], s["dominant_expert"],
                    s["dominant_frac"] * 100, s["routing_entropy"])

    # ── Pass 2: collect qualifying samples ────────────────────────────────────
    logger.info("Pass 2: scanning for qualifying sequences...")
    qualifying: List[Dict] = []
    global_idx = 0

    for batch in make_loader():
        context = batch["context"]
        B       = context.shape[0]
        regime_labels_batch = get_regime_labels(context, CAPACITY)

        batch_gpu = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
        output = pipeline.model(
            context              = batch_gpu["context"],
            site_coords          = batch_gpu.get("site_coords"),
            variate_ids          = batch_gpu.get("variate_ids"),
            channel_mask         = batch_gpu.get("channel_mask"),
            has_coords           = batch_gpu.get("has_coords"),
            output_router_logits = True,
        )

        gate_probs_batch, _ = extract_last_moe_outputs(output.all_ffn_details)
        gp_power = gate_probs_batch[:, 0, :, :].cpu().numpy()

        for i in range(B):
            rl = regime_labels_batch[i]
            f0 = (rl == 0).float().mean().item()
            f1 = (rl == 1).float().mean().item()
            f2 = (rl == 2).float().mean().item()

            if (f0 >= args.min_idle_frac
                    and f2 >= args.min_sat_frac
                    and f1 >= args.min_ramp_frac):
                qualifying.append({
                    "global_idx":    global_idx,
                    "power_raw":     context[i, 0, :].numpy(),
                    "gate_probs":    gp_power[i],
                    "regime_labels": rl.cpu().numpy(),
                    "fracs":         (f0, f1, f2),
                })
                logger.info(
                    "  Qualifying #%03d (global=%d): "
                    "idle=%.1f%% ramp=%.1f%% sat=%.1f%%",
                    len(qualifying) - 1, global_idx,
                    f0*100, f1*100, f2*100,
                )
                if len(qualifying) >= args.max_figures:
                    logger.info("Reached max_figures=%d, stopping.", args.max_figures)
                    break
            global_idx += 1

        if len(qualifying) >= args.max_figures:
            break

    logger.info("Found %d qualifying sequences.", len(qualifying))
    if not qualifying:
        logger.warning("No qualifying sequences found. Try lowering thresholds.")
        return

    # ── Generate figures ──────────────────────────────────────────────────────
    logger.info("Generating individual figures...")
    for fig_idx, sample in enumerate(qualifying):
        save_path = str(ind_dir / f"fig13_{DATASET_NAME}_{fig_idx:03d}.pdf")
        plot_single_figure13(
            power_raw         = sample["power_raw"],
            gate_probs        = sample["gate_probs"],
            regime_labels     = sample["regime_labels"],
            capacity          = CAPACITY,
            global_stats      = global_stats,
            dataset_name      = f"{DATASET_NAME} [#{sample['global_idx']:03d}]",
            save_path         = save_path,
            min_boundary_span = args.min_boundary_span,
        )
        logger.info("  [saved] %s", save_path)

    save_thumbnail_grid(
        samples=qualifying, global_stats=global_stats,
        capacity=CAPACITY, dataset_name=DATASET_NAME,
        save_path=str(out_dir / "thumbnail_grid.pdf"), ncols=4,
    )
    logger.info("\nDone. %d figures saved to %s", len(qualifying), ind_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path",   default="/scratch/pawsey0115/hwang4/results/DeepWind-Research/checkpoints/deepwind/deepwind_large_v5")
    parser.add_argument("--config_path",       default="/scratch/pawsey0115/hwang4/project_codes/deepwind_research/configs/eval.yaml")
    parser.add_argument("--npy_root",          default="/scratch/pawsey0115/hwang4/deepwindData/test")
    parser.add_argument("--metadata_path",     default="/scratch/pawsey0115/hwang4/deepwindData/train_metadata.csv")
    parser.add_argument("--output_dir",        default="./paper/fig13_outputs")
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--context_length",    type=int,   default=1024)
    parser.add_argument("--prediction_length", type=int,   default=16)
    parser.add_argument("--min_boundary_span", type=int,   default=2,
                        help="Min token span for boundary line and annotation")
    parser.add_argument("--max_figures",       type=int,   default=30)
    parser.add_argument("--min_idle_frac",     type=float, default=0.05)
    parser.add_argument("--min_sat_frac",      type=float, default=0.01)
    parser.add_argument("--min_ramp_frac",     type=float, default=0.10)

    args = parser.parse_args()
    main(args)