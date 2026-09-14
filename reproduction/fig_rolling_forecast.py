import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "deepwind_large_v5"
RESULTS_DIR = Path(
    os.environ.get(
        "DEEPWIND_RESULTS_ROOT",
        "/scratch/pawsey0115/hwang4/results/DeepWind-Research",
    )
) / "results/deepwind/deepwind_large_v5/raw_results"

DISPLAY_NAMES = {
   # "csg_wind_5":   "CSG-Wind-5",
   # "penmanshiel_15":    "Penmanshiel-13",
   # "gefc12_wind_7":     "GEFC12-7",
   # "gefc14_wind_10":    "GEFC14-10",
    "75354":        "WTK-75354",
    "30651":        "WTK-30651",
    "76016":        "WTK-76016",
    "43458":        "WTK-43458",
}


QUANTILE_LEVELS = np.array([
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95, 0.99,
])

# Plot only these horizons; set to None to plot all found
TARGET_HORIZONS = [1, 2, 4, 6, 8, 12]

# Split long series into segments of this length; None = no split
ROLLING_PLOT_SEGMENT = 2160

# Maximum segments to save per dataset/horizon; None = no limit
ROLLING_PLOT_MAX_SEGS = 20

OUTPUT_DIR = Path("./reproduction/figures")
# ─────────────────────────────────────────────────────────────────────────────


def load_npz(npz_path: Path) -> dict:
    """
    Load a raw result .npz file.
    Expected arrays: targets (N, T), point_preds (N, T), quantile_preds (N, T, Q).
    """
    data = np.load(npz_path)
    return {
        "targets":        data["targets"],
        "point_preds":    data["point_preds"],
        "quantile_preds": data["quantile_preds"],
    }


def make_segments(total_len: int, seg_len: int | None) -> list[tuple[int, int]]:
    """Split [0, total_len) into fixed-length segments."""
    if seg_len is None:
        return [(0, total_len)]
    return [
        (start, min(start + seg_len, total_len))
        for start in range(0, total_len, seg_len)
    ]


def q_index(q: float) -> int:
    """Return the index of the closest quantile level."""
    return int(np.argmin(np.abs(QUANTILE_LEVELS - q)))


def plot_rolling(
    results:      dict,
    dataset_name: str,
    horizon_h:    int,
    output_dir:   Path,
) -> None:

    color_gt   = '#27ae60'
    color_pred = '#e74c3c'  # default orange

    targets     = results["targets"]
    point_preds = results["point_preds"]
    q_preds     = results["quantile_preds"]

    targets_full     = targets.reshape(-1)
    point_preds_full = point_preds.reshape(-1)

    q10_full = q_preds[:, :, q_index(0.05)].reshape(-1)
    q25_full = q_preds[:, :, q_index(0.25)].reshape(-1)
    q75_full = q_preds[:, :, q_index(0.75)].reshape(-1)
    q90_full = q_preds[:, :, q_index(0.95)].reshape(-1)

    T = len(targets_full)
    t = np.arange(T)

    segments = make_segments(T, ROLLING_PLOT_SEGMENT)
    if ROLLING_PLOT_MAX_SEGS is not None:
        segments = segments[:ROLLING_PLOT_MAX_SEGS]

    save_dir = output_dir / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    n_segs = len(segments)

    for seg_idx, (start, end) in enumerate(segments):
        sl = slice(start, end)
        x_plot = np.arange(end - start)

        fig, ax = plt.subplots(figsize=(16, 4))

        ax.fill_between(
            x_plot,
            q10_full[sl],
            q90_full[sl],
            alpha=0.2,
            color=color_pred,    # "steelblue"
            linewidth=0,        
            edgecolor="none",   
            label="90% PI" 
        )

        ax.fill_between(
            x_plot,
            q25_full[sl],
            q75_full[sl],
            alpha=0.3,
            color=color_pred,   # "steelblue"
            linewidth=0,
            edgecolor="none",
            label="50% PI"
        )

        # ── lines ──
        ax.plot(x_plot, targets_full[sl],
                color=color_gt, lw=1.0, label="Ground truth")  #"black"
        ax.plot(x_plot, point_preds_full[sl],
                color=color_pred, lw=1.0, label="Forecast (median)", linestyle="--")    # "steelblue" 1.5 width

        # ── title ──
        seg_str = f"  (segment {seg_idx + 1}/{n_segs})" if n_segs > 1 else ""

        ax.set_title(
            f"Rolling forecast results (horizon = {horizon_h}h)",
            fontsize=11
        )

        ax.set_xlabel("Time step", fontsize=10)
        ax.set_ylabel("Power", fontsize=10)

        # ── ticks ──
        ax.tick_params(direction="in", top=False, right=False, labelsize=9)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        # ── grid (optional very light) ──
        ax.grid(True, which="major", linestyle="--",
                linewidth=0.5, alpha=0.3)

        # ── spines (clean style) ──
        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["right"].set_visible(True)

        # ── legend ──
        ax.legend(loc="upper right", fontsize=8)

        fig.tight_layout()

        fname = (
            f"{MODEL_NAME}_rolling_H{horizon_h}_seg{seg_idx + 1:03d}.pdf"
            if n_segs > 1
            else f"rolling_H{horizon_h}.pdf"
        )

        fig.savefig(save_dir / fname, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved → {save_dir / fname}")

# ── Main ──────────────────────────────────────────────────────────────────────
for dataset_dir in sorted(RESULTS_DIR.iterdir()):
    if not dataset_dir.is_dir():
        continue

    # Skip datasets not in DISPLAY_NAMES
    if dataset_dir.name not in DISPLAY_NAMES:
        continue

    display_name = DISPLAY_NAMES[dataset_dir.name]

    for npz_file in sorted(dataset_dir.glob("raw_H*.npz")):
        # Parse horizon from filename: raw_H1.npz → 1
        try:
            horizon_h = int(npz_file.stem.replace("raw_H", ""))
        except ValueError:
            print(f"  Skipping unrecognised filename: {npz_file.name}")
            continue

        if TARGET_HORIZONS is not None and horizon_h not in TARGET_HORIZONS:
            continue

        print(f"[{display_name}] H={horizon_h}h  ← {npz_file.name}")
        results = load_npz(npz_file)
        plot_rolling(results, display_name, horizon_h, OUTPUT_DIR)