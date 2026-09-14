# tools/collect_results.py
"""
Scan all H_*.json under a results root and aggregate into a single CSV/Excel.

Usage:
    python tools/collect_results.py \
        --results_root /scratch/pawsey0115/hwang4/results \
        --output       /scratch/pawsey0115/hwang4/results/summary/compare_all.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd


# Metrics shown in the summary table (order matters for readability)
SUMMARY_METRICS = [
    "Accuracy",
    "nMAE",
    "nCRPS",
    "Qualified_Rate",
    "MAE_Coverage",
    "mean_wQuantileLoss",
    "R2",
]


def collect(results_root: Path) -> pd.DataFrame:
    rows = []
    for json_path in sorted(results_root.rglob("H_*.json")):
        with open(json_path) as f:
            payload = json.load(f)

        metrics = payload.get("metrics", {})

        # model name = parent of dataset dir = run_name dir
        # e.g. results/deepwind/deepwind_patch_16/csg_wind_5/H_12.json
        #       → group=deepwind, run=deepwind_patch_16
        parts = json_path.relative_to(results_root).parts
        group   = parts[0] if len(parts) >= 4 else "unknown"
        run     = parts[1] if len(parts) >= 4 else "unknown"

        row = {
            "group":    group,
            "run":      run,
            "dataset":  payload.get("dataset", json_path.parent.name),
            "horizon_h": payload.get("horizon_hours"),
            "pred_len": payload.get("pred_len_steps"),
        }
        for m in SUMMARY_METRICS:
            row[m] = metrics.get(m, float("nan"))

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output",       type=Path, required=True)
    args = parser.parse_args()

    df = collect(args.results_root)

    # Sort for readability
    df = df.sort_values(["dataset", "horizon_h", "group", "run"])

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(args.output, index=False, float_format="%.4f")

    # Save Excel with conditional formatting if openpyxl available
    try:
        xlsx_path = args.output.with_suffix(".xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for dataset, grp in df.groupby("dataset"):
                grp.to_excel(writer, sheet_name=dataset[:31], index=False)
                # Highlight best Accuracy per horizon
                ws = writer.sheets[dataset[:31]]
                ws.freeze_panes = "A2"
        print(f"Excel saved → {xlsx_path}")
    except ImportError:
        pass

    print(f"CSV saved → {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()