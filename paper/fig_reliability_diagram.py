import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import math

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(
    os.environ.get(
        "DEEPWIND_RESULTS_ROOT",
        "/scratch/pawsey0115/hwang4/results/DeepWind-Research",
    ),
    "results/deepwind/deepwind_large_v5",
)

# Display name mapping: folder name → paper label
DISPLAY_NAMES = {
    "csg_wind_5":     "CSG-Wind-5",
    "penmanshiel_15": "Penmanshiel-13",
    "gefc12_wind_7":  "GEFC12-7",
    "gefc14_wind_10": "GEFC14-10",
    "75354":        "WTK-75354",
    "30651":        "WTK-30651",
    "76016":        "WTK-76016",
    "43458":        "WTK-43458",
}

DISPLAY_ORDER = [
    "csg_wind_5",
    "gefc12_wind_7",
    "gefc14_wind_10",
    "75354",
    "30651",
    "43458",
    "76016",
    "penmanshiel_15",
]

QUANTILE_LEVELS = [
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95, 0.99,
]

SAVE_PATH = "./paper/figures/reliability_diagram.pdf"
# ─────────────────────────────────────────────────────────────────────────────

results_dir = Path(RESULTS_DIR)
for subdir in sorted(results_dir.iterdir()):
    if subdir.is_dir() and any(subdir.glob("H_*.json")):
        print(repr(subdir.name))

def discover_datasets(results_dir: Path) -> list[tuple[Path, str]]:
    # Collect all valid dataset dirs
    available = {}
    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue
        if not any(subdir.glob("H_*.json")):
            continue
        available[subdir.name] = subdir

    # Sort by DISPLAY_ORDER; unlisted datasets appended at the end
    ordered_names = [n for n in DISPLAY_ORDER if n in available]
    remaining     = [n for n in sorted(available) if n not in DISPLAY_ORDER]

    return [
        (available[name], DISPLAY_NAMES.get(name, name))
        for name in ordered_names + remaining
    ]


def load_coverage(dataset_dir: Path) -> np.ndarray | None:
    rows = []
    for json_file in sorted(dataset_dir.glob("H_*.json")):
        with open(json_file) as f:
            data = json.load(f)
        metrics = data.get("metrics", data)
        try:
            row = [metrics[f"Coverage[{q:.2f}]"] for q in QUANTILE_LEVELS]
        except KeyError as e:
            print(f"  Warning: missing key {e} in {json_file.name}, skipping.")
            continue
        rows.append(row)
    if not rows:
        return None
    coverage = np.mean(rows, axis=0)
    return coverage


def calibration_mae(coverage: np.ndarray) -> float:
    return float(np.mean(np.abs(coverage - np.array(QUANTILE_LEVELS))))


def make_grid(n: int) -> tuple[int, int]:
    n_cols = min(n, 4)
    n_rows = math.ceil(n / n_cols)
    return n_rows, n_cols


# ── Discover datasets ─────────────────────────────────────────────────────────
results_dir = Path(RESULTS_DIR)
datasets = discover_datasets(results_dir)

if not datasets:
    raise FileNotFoundError(f"No dataset subdirectories with H_*.json found in {RESULTS_DIR}")

print(f"Found {len(datasets)} dataset(s): {[d for _, d in datasets]}")

# ── Build figure ──────────────────────────────────────────────────────────────
n_rows, n_cols = make_grid(len(datasets))
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3.2 * n_cols, 3.2 * n_rows),
    sharex=True, sharey=True,
    squeeze=False,
)
axes_flat = axes.flatten()

nominal = np.array(QUANTILE_LEVELS)

for i, (dataset_dir, display_name) in enumerate(datasets):
    ax = axes_flat[i]
    coverage = load_coverage(dataset_dir)

    if coverage is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title(display_name, fontsize=9, fontweight="bold")
        continue

    mae = calibration_mae(coverage)

    # Shaded areas
    ax.fill_between(nominal, nominal, coverage,
                    where=(coverage >= nominal),
                    alpha=0.15, color="#3266ad", interpolate=True)
    ax.fill_between(nominal, nominal, coverage,
                    where=(coverage <= nominal),
                    alpha=0.15, color="#d85a30", interpolate=True)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2,
            color="gray", alpha=0.6)

    # Reliability curve
    ax.plot(nominal, coverage, linewidth=2, color="#3266ad",
            marker="o", markersize=3)

    ax.set_title(display_name, fontsize=9, fontweight="bold", pad=4)
    ax.text(0.05, 0.93, f"MAE = {mae:.3f}",
            transform=ax.transAxes, fontsize=8, color="#333", va="top")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.tick_params(labelsize=7, direction="in", top=False, right=False)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=False, right=False, length=2)

    ax.grid(True, linewidth=0.4, alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Hide unused subplots
for j in range(len(datasets), len(axes_flat)):
    axes_flat[j].set_visible(False)

# Axis labels
for ax in axes[-1]:
    if ax.get_visible():
        ax.set_xlabel("Nominal quantile level", fontsize=9)
for row in axes:
    row[0].set_ylabel("Empirical coverage", fontsize=9)

# Legend
legend_elements = [
    plt.Line2D([0], [0], linestyle="--", color="gray", linewidth=1.2,
               label="Perfect calibration"),
    plt.Line2D([0], [0], linestyle="-", color="#3266ad", linewidth=2,
               label="DeepWind-Large"),
    mpatches.Patch(facecolor="#3266ad", alpha=0.2, label="Over-coverage"),
    mpatches.Patch(facecolor="#d85a30", alpha=0.2, label="Under-coverage"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Reliability diagrams of DeepWind-Large across WindBench datasets\n"
    "(averaged over 1,2,4,6,8,12 forecasting hours)",
    fontsize=10, y=1.01,
)

plt.tight_layout(h_pad=1.0, w_pad=0.8)

plt.savefig(SAVE_PATH, bbox_inches="tight", dpi=150)
print(f"Saved → {SAVE_PATH}")