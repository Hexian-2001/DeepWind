"""
src/analysis/moe_analysis.py

Quantitative analysis of MoE routing vs. physical wind operating regimes.

Pipeline:
    1. Replicate internal Patch operation externally to assign regime labels
       (idle / ramping / saturation) to each token.
    2. Run inference and extract gate_probs + selected_experts from the
       last MoE layer.
    3. Accumulate per-regime routing statistics:
       Dominant Expert, Routing Entropy H, JSD².
    4. Generate paper table and upgraded Figure 13 for each dataset.

Usage:
    python -m src.analysis.moe_analysis \
        --checkpoint_path /path/to/checkpoint \
        --config_path     /path/to/config.yaml \
        --npy_root        /path/to/test_npy \
        --metadata_path   /path/to/metadata.csv \
        --output_dir      ./moe_analysis_outputs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IDLE_THRESH = 0.05    # P/capacity <= 5%  -> idle
SAT_THRESH  = 0.9    # P/capacity >= 90% -> saturation (pitch control onset)

REGIME_NAMES  = ["Idle", "Ramping", "Saturation"]
REGIME_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
REGIME_SHADE  = ["#AEC6E8", "#FFE5B4", "#B5EAD7"]

# Four analysis datasets (consistent with Figure 12)
TARGET_DATASETS = {
    "csg_wind_5":      35.0,
    "gefc12_wind_7":    1.0,
    "penmanshiel_15": 2080.0,
    "wtk_76016":        4.0,
}

# Filename mapping for datasets whose on-disk name differs from the dict key
FILENAME_MAP = {
    "wtk_76016":      "76016.npy",
}
# Cap total windows per dataset to keep inference time manageable
TARGET_MAX_WINDOWS = 500

NUM_EXPERTS  = 8
TOP_K        = 2
PATCH_SIZE   = 16
PATCH_STRIDE = 16


# ─────────────────────────────────────────────────────────────────────────────
# 1. External Patch (replicates internal Patch(16,16) logic)
# ─────────────────────────────────────────────────────────────────────────────

def external_patch(
    context: torch.Tensor,        # (B, V, T)
    patch_size: int   = PATCH_SIZE,
    patch_stride: int = PATCH_STRIDE,
) -> torch.Tensor:                 # (B, V, N_tokens, patch_size)
    """
    Replicates Patch(16, 16).forward(): left-padding + unfold.

    For T=8192 and patch_size=16: 8192 % 16 = 0, no padding triggered.
    The padding branch is kept for robustness with other context lengths.
    """
    T = context.shape[-1]
    remainder = T % patch_size
    if remainder != 0:
        pad_len = patch_size - remainder
        padding = torch.zeros(
            *context.shape[:-1], pad_len,
            dtype=context.dtype,
            device=context.device,
        )
        context = torch.cat([padding, context], dim=-1)  # left padding

    return context.unfold(dimension=-1, size=patch_size, step=patch_stride)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Regime labels
# ─────────────────────────────────────────────────────────────────────────────

def get_regime_labels(
    context: torch.Tensor,   # (B, V, T), raw MW values, V=0 is power channel
    capacity: float,
) -> torch.Tensor:           # (B, N_tokens), long, 0=idle 1=ramping 2=saturation
    """
    Assigns a physical regime label to each token (patch).

    Steps:
        1. External patch -> (B, V, N_tokens, patch_size)
        2. Take V=0 (power channel) -> (B, N_tokens, patch_size)
        3. Mean over the 16 timesteps within each patch -> (B, N_tokens) in MW
        4. Divide by capacity -> normalised power in [0, 1]
        5. Threshold classification
    """
    patched = external_patch(context)         # (B, V, N_tokens, P)
    mean_pw = patched[:, 0, :, :].mean(-1)   # (B, N_tokens), MW
    normed  = (mean_pw / capacity).clamp(min=0.0)  # guard against curtailment negatives

    labels  = torch.ones_like(normed, dtype=torch.long)   # default 1 = ramping
    labels[normed <= IDLE_THRESH] = 0
    labels[normed >= SAT_THRESH]  = 2

    return labels                              # (B, N_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Extract last MoE layer outputs
# ─────────────────────────────────────────────────────────────────────────────

def extract_last_moe_outputs(
    all_ffn_details: Tuple,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract routing info from the last MoE layer in all_ffn_details.

    With num_layers=18 and all layers being MoE, all_ffn_details[-1] is the
    target layer (TIME attention + MoE FFN). The reverse traversal is kept
    as a safety check against config changes.

    Returns:
        gate_probs:       (B, V, N_tokens, E)
        selected_experts: (B, V, N_tokens, K)
    """
    for ffn_out in reversed(all_ffn_details):
        if ffn_out is not None and ffn_out.gate_probs is not None:
            return ffn_out.gate_probs, ffn_out.selected_experts
    raise RuntimeError(
        "No valid MoE layer found in all_ffn_details. "
        "Ensure use_moe=True and output_router_logits=True."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistics accumulator
# ─────────────────────────────────────────────────────────────────────────────

class RegimeAccumulator:
    """
    Accumulates per-regime routing statistics across batches.

    Stores only slot counts and prob sums (not full gate_probs tensors)
    to keep memory overhead low.

    Attributes:
        slot_counts:  (3, E) total routing slots assigned to each expert per regime
        token_counts: (3,)   total tokens per regime
        prob_sum:     (3, E) cumulative gate_prob per expert per regime
    """

    def __init__(self, num_experts: int = NUM_EXPERTS):
        self.E            = num_experts
        self.slot_counts  = np.zeros((3, num_experts), dtype=np.float64)
        self.token_counts = np.zeros(3,                dtype=np.int64)
        self.prob_sum     = np.zeros((3, num_experts), dtype=np.float64)

    def update(
        self,
        gate_probs:       torch.Tensor,   # (B, N_tokens, E), V=0 already selected
        selected_experts: torch.Tensor,   # (B, N_tokens, K), V=0 already selected
        regime_labels:    torch.Tensor,   # (B, N_tokens)
    ) -> None:
        gp  = gate_probs.detach().cpu().float()
        se  = selected_experts.detach().cpu().long()
        rl  = regime_labels.detach().cpu().long()

        B, N, E = gp.shape
        K       = se.shape[-1]

        gp_flat = gp.reshape(-1, E)   # (B*N, E)
        se_flat = se.reshape(-1, K)   # (B*N, K)
        rl_flat = rl.reshape(-1)      # (B*N,)

        for r in range(3):
            mask  = rl_flat == r
            n_tok = int(mask.sum())
            if n_tok == 0:
                continue

            self.token_counts[r] += n_tok
            self.prob_sum[r]     += gp_flat[mask].sum(0).numpy()

            flat_idx = se_flat[mask].reshape(-1).numpy()
            self.slot_counts[r] += np.bincount(
                flat_idx, minlength=self.E
            ).astype(np.float64)

    def compute(self) -> List[Dict]:
        """
        Compute statistics for all three regimes.

        Returns:
            List of dicts, one per regime, each containing:
                regime_name, n_tokens,
                expert_usage (E,),
                expert_prob_mean (E,),
                routing_entropy (float),
                dominant_expert (int),
                dominant_frac (float)
        """
        results = []
        for r, name in enumerate(REGIME_NAMES):
            n_tok = int(self.token_counts[r])
            if n_tok == 0:
                logger.warning(f"Regime {name} has 0 tokens, skipping.")
                continue

            usage     = self.slot_counts[r] / (n_tok * TOP_K)  # (E,)
            prob_mean = self.prob_sum[r]     / n_tok            # (E,)
            h         = float(entropy(usage + 1e-12, base=2))
            dominant  = int(np.argmax(usage))

            results.append({
                "regime_name":      name,
                "n_tokens":         n_tok,
                "expert_usage":     usage,
                "expert_prob_mean": prob_mean,
                "routing_entropy":  h,
                "dominant_expert":  dominant,
                "dominant_frac":    float(usage[dominant]),
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. JSD²
# ─────────────────────────────────────────────────────────────────────────────

def jsd_squared(p: np.ndarray, q: np.ndarray) -> float:
    """JSD²(p, q) in [0, 1]. 0 = identical distributions, 1 = no overlap."""
    return float(jensenshannon(p + 1e-12, q + 1e-12) ** 2)


def compute_jsd_matrix(stats: List[Dict]) -> np.ndarray:
    """Returns the (n_regimes, n_regimes) pairwise JSD² matrix."""
    n   = len(stats)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = jsd_squared(
                stats[i]["expert_usage"],
                stats[j]["expert_usage"],
            )
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_dataset(
    pipeline,
    dataloader:     DataLoader,
    capacity:       float,
    dataset_name:   str,
    seq_for_figure: Optional[int] = None,
) -> Tuple[List[Dict], np.ndarray, Optional[Dict]]:
    """
    Run the full analysis for one dataset's test split.

    Args:
        pipeline:        Initialised InferencePipeline.
        dataloader:      DataLoader wrapping DeepWindTestDataset(split="test").
        capacity:        Installed capacity in MW for this dataset.
        dataset_name:    Used in log messages.
        seq_for_figure:  Global sample index to save for Figure 13 visualisation.
                         If None, fig_data is not collected.

    Returns:
        stats:    List of per-regime stat dicts.
        jsd_mat:  (3, 3) JSD² matrix.
        fig_data: Dict with raw data for plotting, or None.
    """
    accumulator = RegimeAccumulator(NUM_EXPERTS)
    fig_data    = None
    device      = pipeline.device

    for batch_idx, batch in enumerate(dataloader):
        context = batch["context"]                          # (B, V, T), CPU, raw MW

        # Compute regime labels on CPU (no GPU needed)
        regime_labels = get_regime_labels(context, capacity)  # (B, N_tokens)

        # Model forward pass with router logits enabled
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

        gate_probs, selected_experts = extract_last_moe_outputs(
            output.all_ffn_details
        )

        # Select V=0 (power channel)
        gp_power = gate_probs[:, 0, :, :]        # (B, N_tokens, E)
        se_power = selected_experts[:, 0, :, :]  # (B, N_tokens, K)

        # Accumulate statistics
        accumulator.update(gp_power, se_power, regime_labels)

        # Optionally collect data for Figure 13
        if seq_for_figure is not None and fig_data is None:
            B         = context.shape[0]
            start_idx = batch_idx * B
            if start_idx <= seq_for_figure < start_idx + B:
                i = seq_for_figure - start_idx
                fig_data = {
                    "power_raw":     context[i, 0, :].cpu().numpy(),   # (T,) MW
                    "gate_probs":    gp_power[i].cpu().numpy(),        # (N_tokens, E)
                    "regime_labels": regime_labels[i].cpu().numpy(),   # (N_tokens,)
                    "capacity":      capacity,
                }

        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  [{dataset_name}] processed {batch_idx+1} batches")

    stats   = accumulator.compute()
    jsd_mat = compute_jsd_matrix(stats)
    return stats, jsd_mat, fig_data


# ─────────────────────────────────────────────────────────────────────────────
# 7. Automatic Figure 13 sequence selection
# ─────────────────────────────────────────────────────────────────────────────

def find_representative_sequence(
    dataloader:    DataLoader,
    capacity:      float,
    min_idle_frac: float = 0.05,
    min_sat_frac:  float = 0.01,
    min_ramp_frac: float = 0.10,
) -> int:
    """
    Find the first sample in the test set that contains all three regimes
    at or above the specified minimum fractions.

    Returns immediately upon finding a match to avoid full dataset traversal.
    Falls back to index 0 if no suitable sequence is found.
    """
    global_idx = 0
    for batch in dataloader:
        context = batch["context"]
        labels  = get_regime_labels(context, capacity)  # (B, N_tokens)
        B       = context.shape[0]
        for i in range(B):
            l  = labels[i]
            f0 = (l == 0).float().mean().item()
            f1 = (l == 1).float().mean().item()
            f2 = (l == 2).float().mean().item()
            if f0 >= min_idle_frac and f2 >= min_sat_frac and f1 >= min_ramp_frac:
                logger.info(
                    "Found representative sequence at index=%d: "
                    "idle=%.1f%%, ramp=%.1f%%, sat=%.1f%%",
                    global_idx, f0*100, f1*100, f2*100,
                )
                return global_idx
            global_idx += 1

    logger.warning("No suitable sequence found, falling back to index=0.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _get_contiguous_spans(
    labels: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Return list of contiguous regime spans: (start, end, regime_id)."""
    spans = []
    if len(labels) == 0:
        return spans
    start, cur = 0, int(labels[0])
    for t in range(1, len(labels)):
        if int(labels[t]) != cur:
            spans.append((start, t, cur))
            start, cur = t, int(labels[t])
    spans.append((start, len(labels), cur))
    return spans


def _annotate_dominant_experts(
    ax:            plt.Axes,
    gate_probs:    np.ndarray,   # (N_tokens, E)
    regime_labels: np.ndarray,   # (N_tokens,)
    min_span_len:  int = 20,
) -> None:
    """Annotate each sufficiently long regime span with its dominant expert."""
    for (start_t, end_t, r) in _get_contiguous_spans(regime_labels):
        if end_t - start_t < min_span_len:
            continue
        seg   = gate_probs[start_t:end_t]
        usage = seg.mean(0)
        dom   = int(np.argmax(usage))
        frac  = float(usage[dom])
        mid_t = (start_t + end_t) // 2
        ax.annotate(
            f"E{dom}\n{frac:.0%}",
            xy=(mid_t, 0.03),
            xycoords=("data", "axes fraction"),
            fontsize=7.5, ha="center",
            color=REGIME_COLORS[r], fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white",
                ec=REGIME_COLORS[r], alpha=0.75, linewidth=0.8,
            ),
        )


def plot_upgraded_figure13(
    fig_data:     Dict,
    dataset_name: str,
    save_path:    str = "fig13_upgraded.pdf",
) -> None:
    """
    Upgraded Figure 13.

    Top panel:  power curve + regime background shading + dominant expert annotations
    Bottom panel: MoE gate probability heatmap + white dashed regime boundaries
    """
    power_raw     = fig_data["power_raw"]
    gate_probs    = fig_data["gate_probs"]
    regime_labels = fig_data["regime_labels"]
    capacity      = fig_data["capacity"]

    N_tokens, E = gate_probs.shape
    P           = PATCH_SIZE

    fig, (ax_p, ax_h) = plt.subplots(
        2, 1, figsize=(13, 5.5),
        gridspec_kw={"height_ratios": [1.7, 1]},
    )

    # Top panel: power curve
    legend_added = set()
    for (s, e, r) in _get_contiguous_spans(regime_labels):
        lbl = REGIME_NAMES[r] if r not in legend_added else None
        ax_p.axvspan(s, e, alpha=0.22, color=REGIME_SHADE[r],
                     label=lbl, zorder=1)
        legend_added.add(r)

    # Align time-step axis to token index for consistency with heatmap
    t_token = np.arange(len(power_raw)) / P
    ax_p.plot(t_token, power_raw / capacity,
              color="#2C7BB6", lw=1.0, zorder=3, label="Power")
    ax_p.set_ylabel("Normalised Power (P/Cap)", fontsize=10)
    ax_p.set_title(
        f"Power Curve with Regime Segmentation and MoE Activation  [{dataset_name}]",
        fontsize=11,
    )
    ax_p.set_xlim(0, N_tokens - 1)
    ax_p.set_ylim(-0.05, 1.20)
    ax_p.grid(alpha=0.3, linewidth=0.4)

    _annotate_dominant_experts(ax_p, gate_probs, regime_labels)

    ax_p.legend(fontsize=8, loc="upper right", ncol=4)

    # Bottom panel: gate probability heatmap
    im = ax_h.imshow(
        gate_probs.T,
        aspect="auto", cmap="viridis",
        origin="upper", vmin=0,
        interpolation="nearest",
    )
    ax_h.set_yticks(range(E))
    ax_h.set_yticklabels([f"E{i}" for i in range(E)], fontsize=9)
    ax_h.set_xlabel("Token Index", fontsize=10)
    ax_h.set_ylabel("Expert", fontsize=10)

    # White dashed lines at regime boundaries
    prev = int(regime_labels[0])
    for t in range(1, N_tokens):
        cur = int(regime_labels[t])
        if cur != prev:
            ax_h.axvline(t - 0.5, color="white", lw=1.0, ls="--", alpha=0.8)
        prev = cur

    plt.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02, label="Gate Prob.")

    fig.tight_layout(pad=1.0)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"[saved] {save_path}")
    plt.close(fig)


def plot_regime_routing_summary(
    all_stats: Dict[str, List[Dict]],
    all_jsd:   Dict[str, np.ndarray],
    save_path: str = "fig_regime_routing_summary.pdf",
) -> None:
    """
    Overview figure: 4 datasets x 3 regimes.
    Each row: left = usage bar chart with H annotations, right = JSD² matrix.
    """
    datasets = list(all_stats.keys())
    n_ds     = len(datasets)
    E        = NUM_EXPERTS
    x        = np.arange(E)
    width    = 0.26

    fig, axes = plt.subplots(
        n_ds, 2,
        figsize=(12, 3.0 * n_ds),
        gridspec_kw={"width_ratios": [2.8, 1]},
    )
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    for row, ds_name in enumerate(datasets):
        stats   = all_stats[ds_name]
        jsd_mat = all_jsd[ds_name]
        ax_bar  = axes[row, 0]
        ax_jsd  = axes[row, 1]

        # Bar chart
        for k, (s, col) in enumerate(zip(stats, REGIME_COLORS)):
            ax_bar.bar(
                x + k * width, s["expert_usage"] * 100,
                width=width, label=s["regime_name"],
                color=col, alpha=0.85, edgecolor="white", linewidth=0.4,
            )
        ax_bar.set_xticks(x + width)
        ax_bar.set_xticklabels([f"E{i}" for i in x], fontsize=9)
        ax_bar.set_ylabel("Token Fraction (%)", fontsize=9)
        ax_bar.set_title(
            ds_name, fontsize=10, fontweight="bold"
        )
        ax_bar.legend(fontsize=8, loc="upper left", framealpha=0.7)
        ax_bar.set_ylim(0, ax_bar.get_ylim()[1] * 1.32)
        ax_bar.grid(axis="y", alpha=0.3, linewidth=0.5)

        # Annotate entropy H
        for k, (s, col) in enumerate(zip(stats, REGIME_COLORS)):
            ax_bar.annotate(
                f"H={s['routing_entropy']:.2f} bits",
                xy=(0.99, 0.97 - k * 0.10),
                xycoords="axes fraction",
                ha="right", va="top",
                fontsize=8, color=col, fontweight="bold",
            )

        # JSD² matrix
        n_r = len(stats)
        im  = ax_jsd.imshow(
            jsd_mat[:n_r, :n_r],
            cmap="Blues", vmin=0, vmax=1,
        )
        names = [s["regime_name"] for s in stats]
        ax_jsd.set_xticks(range(n_r))
        ax_jsd.set_xticklabels(names, fontsize=7, rotation=20)
        ax_jsd.set_yticks(range(n_r))
        ax_jsd.set_yticklabels(names, fontsize=7)
        ax_jsd.set_title("JSD²", fontsize=9)
        for i in range(n_r):
            for j in range(n_r):
                v = jsd_mat[i, j]
                ax_jsd.text(j, i, f"{v:.3f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if v > 0.5 else "black")
        plt.colorbar(im, ax=ax_jsd, fraction=0.07, pad=0.04)

    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"[saved] {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 9. LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def print_latex_table(
    all_stats: Dict[str, List[Dict]],
    all_jsd:   Dict[str, np.ndarray],
) -> None:
    lines = [
        "",
        r"% ── Table: Per-Regime MoE Routing Statistics ──────────────────────",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Quantitative analysis of MoE expert routing across physical",
        r"wind operating regimes on four WindBench datasets.",
        r"$H$ (bits): routing entropy; higher $=$ more uniform utilisation.",
        r"JSD$^2$: Jensen-Shannon divergence squared between adjacent regime",
        r"routing distributions ($0$\,=\,identical, $1$\,=\,maximally different).",
        r"All results are from the final MoE layer, evaluated on test splits.}",
        r"\label{tab:moe_regime}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrcccr}",
        r"\toprule",
        r"Dataset & Regime & Tokens & Dom.\ Expert & Dom.\ \%"
        r" & $H$ (bits) & JSD$^2$ (prev.) \\",
        r"\midrule",
    ]
    for ds_name, stats in all_stats.items():
        jsd    = all_jsd[ds_name]
        ds_tex = r"\texttt{" + ds_name.replace("_", r"\_") + r"}"
        n_r    = len(stats)
        lines.append(r"\multirow{" + str(n_r) + r"}{*}{" + ds_tex + r"}")
        for k, s in enumerate(stats):
            jsd_prev = f"{jsd[k, k-1]:.3f}" if k > 0 else "---"
            lines.append(
                f" & {s['regime_name']}"
                f" & {s['n_tokens']:,}"
                f" & E{s['dominant_expert']}"
                f" & {s['dominant_frac']*100:.1f}\\%"
                f" & {s['routing_entropy']:.2f}"
                f" & {jsd_prev}"
                r" \\"
            )
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 10. Save JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_results_json(
    all_stats: Dict[str, List[Dict]],
    all_jsd:   Dict[str, np.ndarray],
    save_path: str,
) -> None:
    output = {}
    for ds, stats in all_stats.items():
        output[ds] = {
            "stats": [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in s.items()}
                for s in stats
            ],
            "jsd_matrix": all_jsd[ds].tolist(),
        }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"[saved] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(
    pipeline,
    test_dataset_factory,
    output_dir:  str = "./moe_analysis_outputs",
    batch_size:  int = 8,
    num_workers: int = 2,
) -> None:
    """
    Run the full analysis pipeline for all TARGET_DATASETS.

    Args:
        pipeline:              Initialised InferencePipeline.
        test_dataset_factory:  callable(ds_name: str) -> DeepWindTestDataset.
        output_dir:            Root directory for all outputs.
        batch_size:            DataLoader batch size.
        num_workers:           DataLoader worker count.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: Dict[str, List[Dict]] = {}
    all_jsd:   Dict[str, np.ndarray] = {}

    for ds_name, capacity in TARGET_DATASETS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing: {ds_name}  (capacity={capacity} MW)")
        logger.info(f"{'='*60}")

        test_ds = test_dataset_factory(ds_name)

        def make_loader(ds):
            return DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )

        seq_idx = find_representative_sequence(
            make_loader(test_ds), capacity,
            min_idle_frac = 0.05,
            min_sat_frac  = 0.01,
            min_ramp_frac = 0.10,
        )

        stats, jsd_mat, fig_data = analyze_dataset(
            pipeline       = pipeline,
            dataloader     = make_loader(test_ds),
            capacity       = capacity,
            dataset_name   = ds_name,
            seq_for_figure = seq_idx,
        )

        all_stats[ds_name] = stats
        all_jsd[ds_name]   = jsd_mat

        logger.info(f"\n[{ds_name}] Results:")
        for s in stats:
            logger.info(
                f"  {s['regime_name']:12s} | tokens={s['n_tokens']:>6,} | "
                f"dom=E{s['dominant_expert']}({s['dominant_frac']:.1%}) | "
                f"H={s['routing_entropy']:.3f} bits"
            )
        logger.info(f"  JSD²:\n{np.round(jsd_mat, 3)}")

        if fig_data is not None:
            plot_upgraded_figure13(
                fig_data     = fig_data,
                dataset_name = ds_name,
                save_path    = str(out_dir / f"fig13_{ds_name}.pdf"),
            )
        else:
            logger.warning(f"[{ds_name}] No suitable visualisation sequence found, skipping Figure 13.")

    # Summary figure + JSON + LaTeX table
    plot_regime_routing_summary(
        all_stats, all_jsd,
        save_path=str(out_dir / "fig_regime_routing_summary.pdf"),
    )
    save_results_json(
        all_stats, all_jsd,
        save_path=str(out_dir / "regime_routing_stats.json"),
    )
    print_latex_table(all_stats, all_jsd)
    logger.info(f"\nAnalysis complete. All outputs saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from omegaconf import OmegaConf

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path",   default="/scratch/pawsey0115/hwang4/results/DeepWind-Research/checkpoints/deepwind/deepwind_large_v5")
    parser.add_argument("--config_path",       default="/scratch/pawsey0115/hwang4/project_codes/deepwind_research/configs/eval.yaml")
    parser.add_argument("--npy_root",          default="/scratch/pawsey0115/hwang4/deepwindData/test")
    parser.add_argument("--metadata_path",     default="/scratch/pawsey0115/hwang4/deepwindData/train_metadata.csv")
    parser.add_argument("--output_dir",        default="./paper/moe_analysis_outputs")
    parser.add_argument("--batch_size",        type=int, default=4)
    parser.add_argument("--context_length",    type=int, default=8192)
    parser.add_argument("--prediction_length", type=int, default=16)
    args = parser.parse_args()

    from src.data.datasets import DeepWindTestDataset
    from src.inference.pipeline import InferencePipeline

    cfg      = OmegaConf.load(args.config_path)
    pipeline = InferencePipeline(checkpoint_path=args.checkpoint_path, cfg=cfg)

    def test_dataset_factory(ds_name: str):
        # Resolve on-disk filename (some datasets have no ds_name prefix)
        filename = FILENAME_MAP.get(ds_name, f"{ds_name}.npy")
        npy_path = Path(args.npy_root) / filename

        # Probe data shape to compute adaptive stride
        data     = np.load(npy_path, mmap_mode="r")
        T        = data.shape[-1]
        test_len = int(T * 0.2)   # mirrors DeepWindTestDataset split ratios

        ctx_len  = args.context_length
        pred_len = args.prediction_length

        # Adaptive context_length fallback for datasets shorter than one window
        if test_len < ctx_len + pred_len:
            ctx_len = max(512, test_len // 2)
            logger.info(
                "[%s] Short dataset (test_len=%d), falling back to ctx_len=%d",
                ds_name, test_len, ctx_len,
            )

        # Adaptive stride: cap total windows at TARGET_MAX_WINDOWS
        available = max(1, test_len - ctx_len - pred_len + 1)
        stride    = max(pred_len, available // TARGET_MAX_WINDOWS)

        if stride > pred_len:
            logger.info(
                "[%s] Large dataset (test_len=%d, available=%d), "
                "using stride=%d (~%d windows)",
                ds_name, test_len, available, stride, available // stride,
            )

        return DeepWindTestDataset(
            npy_path          = npy_path,
            metadata_path     = args.metadata_path,
            context_length    = ctx_len,
            prediction_length = pred_len,
            split             = "test",
            stride            = stride,
        )

    run_full_analysis(
        pipeline             = pipeline,
        test_dataset_factory = test_dataset_factory,
        output_dir           = args.output_dir,
        batch_size           = args.batch_size,
    )