import json
import os

import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_DIR = Path(
    os.environ.get(
        "DEEPWIND_RESULTS_ROOT",
        "/scratch/pawsey0115/hwang4/results/DeepWind-Research",
    )
) / "results/deepwind/deepwind_large_v5"

DISPLAY_NAMES = {
    "75354":      "WTK-75354",
    "30651":      "WTK-30651",
    "76016":      "WTK-76016",
    "43458":      "WTK-43458",
    "csg_wind_5":     "CSG-Wind-5",
    "penmanshiel_15": "Penmanshiel-13",
    "gefc12_wind_7":  "GEFC12-7",
    "gefc14_wind_10": "GEFC14-10",
}

QUANTILE_LEVELS = [
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95, 0.99,
]
Q_IDX = {q: i for i, q in enumerate(QUANTILE_LEVELS)}

TARGET_HORIZONS = [1, 6, 12]

# PI_PAIRS: label → (q_lower, q_upper)
# Matches the shaded intervals shown in Figure 7
PI_PAIRS = {
    "50\\% PI": (0.25, 0.75),
    "90\\% PI": (0.05, 0.95),
}

# Nominal coverage for each PI (for the reference row)
PI_NOMINALS = {
    "50\\% PI": "50.0\\%",
    "90\\% PI": "90.0\\%",
}
# ─────────────────────────────────────────────────────────────────────────────


def load_horizon(dataset_dir: Path, horizon_hours: int) -> dict | None:
    """Load metrics dict for a specific horizon. Returns None if file missing."""
    json_file = dataset_dir / f"H_{horizon_hours}.json"
    if not json_file.exists():
        return None
    with open(json_file) as f:
        data = json.load(f)
    return data.get("metrics", data)


def compute_stats(metrics: dict) -> dict:
    """
    Compute per-horizon summary statistics:
      - nCRPS
      - empirical coverage for each PI in PI_PAIRS
      - calibration MAE (mean |coverage(q) - q| over all quantile levels)
    """
    cov = np.array([metrics[f"Coverage[{q:.2f}]"] for q in QUANTILE_LEVELS])
    stats = {
        "nCRPS":   metrics["nCRPS"],
        "cal_mae": float(np.mean(np.abs(cov - np.array(QUANTILE_LEVELS)))),
    }
    for label, (q_lo, q_hi) in PI_PAIRS.items():
        stats[label] = float(cov[Q_IDX[q_hi]] - cov[Q_IDX[q_lo]])
    return stats


# ── Collect results ───────────────────────────────────────────────────────────
results = {}
for subdir in sorted(RESULTS_DIR.iterdir()):
    if not subdir.is_dir() or not any(subdir.glob("H_*.json")):
        continue
    display = DISPLAY_NAMES.get(subdir.name, subdir.name)
    results[display] = {}
    for h in TARGET_HORIZONS:
        m = load_horizon(subdir, h)
        if m is not None:
            results[display][h] = compute_stats(m)


# ── Helper formatters ─────────────────────────────────────────────────────────
def fmt_ncrps(stats: dict | None) -> str:
    return f"{stats['nCRPS']:.4f}" if stats else "---"

def fmt_pi(stats: dict | None, label: str) -> str:
    return f"{stats[label]*100:.1f}\\%" if stats else "---"

def fmt_mae(stats: dict | None) -> str:
    return f"{stats['cal_mae']:.3f}" if stats else "---"


# ── Build column structure from PI_PAIRS ──────────────────────────────────────
# Metric groups: nCRPS | PI_1 | PI_2 | ... | Cal. MAE
# Each group has len(TARGET_HORIZONS) columns
n_pi     = len(PI_PAIRS)
n_groups = 2 + n_pi          # nCRPS + n_pi groups + Cal. MAE
n_cols   = len(TARGET_HORIZONS)

# Column spec: 1 text col + n_groups * n_cols numeric cols
col_spec = "l" + "r" * (n_groups * n_cols)

# cmidrule ranges
def cmidrule(group_idx: int) -> str:
    """group_idx is 0-based; col 1 is the Dataset label column."""
    start = 2 + group_idx * n_cols
    end   = start + n_cols - 1
    return f"\\cmidrule(lr){{{start}-{end}}}"

h_subheader = " & ".join(f"{h}h" for h in TARGET_HORIZONS)

# ── Print LaTeX ───────────────────────────────────────────────────────────────
pi_labels_tex = " and ".join(
    f"{k.replace(chr(92), '')} (nominal: {PI_NOMINALS[k].replace(chr(92), '')})"
    for k in PI_PAIRS
)

print(r"\begin{table}[h]")
print(r"\centering")
print(
    r"\caption{Probabilistic calibration of DeepWind-Large at 1h, 6h, and 12h horizons. "
    + pi_labels_tex.replace("%", r"\%")
    + r" denote empirical prediction interval coverage rates. "
    r"Cal.\ MAE is the mean absolute deviation of the reliability curve from the diagonal.}"
)
print(r"\label{tab:calibration_by_horizon}")
print(r"\resizebox{\textwidth}{!}{%")
print(f"\\begin{{tabular}}{{{col_spec}}}")
print(r"\toprule")

# Top header row
pi_multicols = "".join(
    f" & \\multicolumn{{{n_cols}}}{{c}}{{{label}}}"
    for label in PI_PAIRS
)
print(
    f"& \\multicolumn{{{n_cols}}}{{c}}{{nCRPS $\\downarrow$}}"
    + pi_multicols
    + f" & \\multicolumn{{{n_cols}}}{{c}}{{Cal.\\ MAE $\\downarrow$}} \\\\"
)

# cmidrules
cmidrules = " ".join(cmidrule(i) for i in range(n_groups))
print(cmidrules)

# Sub-header: horizon labels repeated per group
subheader_cols = " & ".join([h_subheader] * n_groups)
print(f"Dataset & {subheader_cols} \\\\")
print(r"\midrule")

# Data rows
for display, horizon_data in results.items():
    h_stats = [horizon_data.get(h) for h in TARGET_HORIZONS]

    ncrps_cols = " & ".join(fmt_ncrps(s) for s in h_stats)
    pi_cols    = " & ".join(
        " & ".join(fmt_pi(s, label) for s in h_stats)
        for label in PI_PAIRS
    )
    mae_cols   = " & ".join(fmt_mae(s) for s in h_stats)

    print(f"{display} & {ncrps_cols} & {pi_cols} & {mae_cols} \\\\")

# Nominal reference row
print(r"\midrule")
nominal_ncrps = " & ".join(["---"] * n_cols)
nominal_pi = " & ".join(
    " & ".join([f"\\textit{{{PI_NOMINALS[label]}}}"] * n_cols)
    for label in PI_PAIRS
)
nominal_mae = " & ".join(["---"] * n_cols)
print(f"\\textit{{Nominal}} & {nominal_ncrps} & {nominal_pi} & {nominal_mae} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}%")
print(r"}")
print(r"\end{table}")