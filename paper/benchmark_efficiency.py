"""
paper/benchmark_efficiency.py

Measures inference time, throughput, and peak GPU memory for all three
DeepWind model variants (Small, Base, Large) across multiple forecasting
horizons and batch sizes.

Outputs:
    efficiency_results.json   — raw numbers for all configurations
    efficiency_table.tex      — ready-to-paste LaTeX table
    efficiency_table.csv      — for quick inspection in spreadsheet

Usage:
    python ./paper/benchmark_efficiency.py \
        [--small_ckpt  /path/to/small]  \
        [--base_ckpt   /path/to/base]   \
        [--large_ckpt  /path/to/large]  \
        [--config_path /path/to/eval.yaml] \
        [--output_dir  ./paper/efficiency_outputs]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Forecasting horizons to benchmark (hours)
HORIZONS_H = [1, 2, 4, 6, 8, 12]

# Batch sizes to benchmark
BATCH_SIZES = [1, 8, 16]

# Context length used during inference (must match training config)
CONTEXT_LENGTH = 8192

# Number of variates (must match training config)
NUM_VARIATES = 6

# Warmup iterations before timing (to avoid CUDA lazy init overhead)
N_WARMUP = 10

# Timed iterations (mean and std computed over these)
N_ITERS = 50

# Dataset resolution in minutes (used to convert hours -> prediction steps)
# Using csg_wind_5 (15 min) as the representative dataset
RESOLUTION_MIN = 15

MODEL_CONFIGS = {
    "Small": {"params_m": 33},
    "Base":  {"params_m": 890},
    "Large": {"params_m": 1300},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def horizon_to_steps(horizon_h: int, resolution_min: int = RESOLUTION_MIN) -> int:
    """Convert forecast horizon in hours to number of prediction steps."""
    return max(1, round(horizon_h * 60 / resolution_min))


def make_dummy_batch(
    batch_size:     int,
    num_variates:   int,
    context_length: int,
    device:         torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Create a random dummy batch that matches the shape expected by
    InferencePipeline.predict().  Values are random floats in [0, 35]
    to loosely resemble raw MW power values.
    """
    B, V, T = batch_size, num_variates, context_length
    return {
        "context":      torch.rand(B, V, T,  device=device) * 35.0,
        "site_coords":  torch.rand(B, 2,     device=device),
        "variate_ids":  torch.zeros(B, V,    device=device, dtype=torch.long),
        "channel_mask": torch.ones(B, V,     device=device),
        "has_coords":   torch.ones(B, 1,     device=device),
        # target and index are not used during inference
    }


def measure_peak_vram_gb() -> float:
    """Return peak allocated VRAM in GB since last reset."""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark function
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one_config(
    pipeline,
    batch_size:   int,
    pred_len:     int,
    horizon_h:    int,
    n_warmup:     int = N_WARMUP,
    n_iters:      int = N_ITERS,
) -> Dict:
    """
    Benchmark a single (batch_size, horizon) configuration.

    Returns a dict with:
        mean_ms, std_ms       — latency statistics in milliseconds
        throughput_sps        — samples per second
        peak_vram_gb          — peak GPU memory in GB
        batch_size, horizon_h, pred_len
    """
    device = pipeline.device
    batch  = make_dummy_batch(batch_size, NUM_VARIATES, CONTEXT_LENGTH, device)

    use_cuda = (device.type == "cuda")

    # ── Warmup ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(n_warmup):
            pipeline.predict(batch, pred_len=pred_len)
    if use_cuda:
        torch.cuda.synchronize()

    # ── Timed iterations ──────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms: List[float] = []

    with torch.no_grad():
        for _ in range(n_iters):
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                start_event.record()
                pipeline.predict(batch, pred_len=pred_len)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                pipeline.predict(batch, pred_len=pred_len)
                elapsed_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(elapsed_ms))
    std_ms  = float(np.std(elapsed_ms))
    # throughput: how many samples processed per second
    throughput = batch_size / (mean_ms / 1000.0)
    peak_vram  = measure_peak_vram_gb() if use_cuda else 0.0

    return {
        "batch_size":    batch_size,
        "horizon_h":     horizon_h,
        "pred_len":      pred_len,
        "mean_ms":       round(mean_ms,  2),
        "std_ms":        round(std_ms,   2),
        "throughput_sps": round(throughput, 1),
        "peak_vram_gb":  round(peak_vram, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-model benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(
    checkpoint_path: str,
    model_name:      str,
    cfg,
    horizons_h:      List[int],
    batch_sizes:     List[int],
) -> List[Dict]:
    """Load one model variant and run all (batch_size, horizon) combinations."""
    from src.inference.pipeline import InferencePipeline

    logger.info("Loading %s from %s", model_name, checkpoint_path)
    pipeline = InferencePipeline(checkpoint_path=checkpoint_path, cfg=cfg)
    pipeline.model.eval()

    device     = pipeline.device
    results    = []
    n_configs  = len(horizons_h) * len(batch_sizes)
    done       = 0

    for horizon_h in horizons_h:
        pred_len = horizon_to_steps(horizon_h)
        for batch_size in batch_sizes:
            done += 1
            logger.info(
                "  [%s] (%d/%d) H=%dh  pred_len=%d  batch=%d",
                model_name, done, n_configs, horizon_h, pred_len, batch_size,
            )
            result = benchmark_one_config(
                pipeline  = pipeline,
                batch_size = batch_size,
                pred_len   = pred_len,
                horizon_h  = horizon_h,
            )
            result["model"] = model_name
            result["params_m"] = MODEL_CONFIGS[model_name]["params_m"]
            results.append(result)
            
            logger.info(
                "    latency=%.1f±%.1f ms  throughput=%.1f sps  "
                "peak_vram=%.2f GB",
                result["mean_ms"], result["std_ms"],
                result["throughput_sps"], result["peak_vram_gb"],
            )

    # Free GPU memory before loading next model
    del pipeline
    torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output: JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_json(results: List[Dict], save_path: str) -> None:
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("[saved] %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Output: CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results: List[Dict], save_path: str) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("[saved] %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Output: LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def save_latex_table(
    results:   List[Dict],
    save_path: str,
    batch_size_main: int = 1,   # which batch size to feature in the main table
) -> None:
    """
    Generate a LaTeX table showing latency and VRAM for the three model
    variants across all horizons, at the specified batch size.

    Table layout:
        Rows:    one per (model, horizon) combination
        Columns: Model | Params | H (h) | Latency (ms) | VRAM (GB)
    """
    # Filter to the chosen batch size
    subset = [r for r in results if r["batch_size"] == batch_size_main]
    if not subset:
        logger.warning("No results for batch_size=%d", batch_size_main)
        return

    # Get GPU info for caption
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    lines = [
        "",
        r"% ── Table: Inference Efficiency ─────────────────────────────────────",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Inference latency and peak GPU memory of the three DeepWind",
        f"variants across forecasting horizons. Results are measured on a single",
        f"\\texttt{{{gpu_name}}} with batch size {batch_size_main} and context",
        r"length 8{,}192. Latency is reported as mean$\,\pm\,$std over 50 runs",
        r"after 10 warm-up iterations.}",
        r"\label{tab:efficiency}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Params & Horizon (h) & Latency (ms) & Peak VRAM (GB) \\",
        r"\midrule",
    ]

    model_order  = ["Small", "Base", "Large"]
    horizon_order = sorted(set(r["horizon_h"] for r in subset))

    for m_idx, model in enumerate(model_order):
        m_rows = [r for r in subset if r["model"] == model]
        params = MODEL_CONFIGS[model]["params_m"]
        params_str = f"{params}\\,M" if params < 1000 else f"{params/1000:.1f}\\,B"
        n_rows = len(horizon_order)

        if m_rows:
            lines.append(
                r"\multirow{" + str(n_rows) + r"}{*}{\textbf{DW-" + model + r"}}"
                r" & \multirow{" + str(n_rows) + r"}{*}{" + params_str + r"}"
            )
            for h in horizon_order:
                row = next((r for r in m_rows if r["horizon_h"] == h), None)
                if row is None:
                    lines.append(f" & & {h} & --- & --- \\\\")
                else:
                    lines.append(
                        f" & & {h}"
                        f" & ${row['mean_ms']:.1f} \\pm {row['std_ms']:.1f}$"
                        f" & {row['peak_vram_gb']:.2f}"
                        r" \\"
                    )
            if m_idx < len(model_order) - 1:
                lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("[saved] %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Output: throughput table (batch_size comparison)
# ─────────────────────────────────────────────────────────────────────────────

def save_throughput_table(
    results:  List[Dict],
    save_path: str,
    horizon_h: int = 1,
) -> None:
    """
    Secondary table showing throughput (samples/sec) across batch sizes
    for a fixed horizon, to characterise batch inference efficiency.
    """
    subset = [r for r in results if r["horizon_h"] == horizon_h]
    if not subset:
        return

    batch_sizes  = sorted(set(r["batch_size"] for r in subset))
    model_order  = ["Small", "Base", "Large"]

    lines = [
        "",
        r"% ── Table: Throughput vs Batch Size ─────────────────────────────────",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Inference throughput (samples/sec) at the 1-hour forecasting",
        r"horizon across batch sizes and model variants. Higher is better.}",
        r"\label{tab:throughput}",
        r"\begin{tabular}{l" + "r" * len(batch_sizes) + "}",
        r"\toprule",
        r"Model & " + " & ".join(f"BS={b}" for b in batch_sizes) + r" \\",
        r"\midrule",
    ]

    for model in model_order:
        row_vals = []
        for bs in batch_sizes:
            match = next(
                (r for r in subset
                 if r["model"] == model and r["batch_size"] == bs),
                None,
            )
            row_vals.append(
                f"{match['throughput_sps']:.1f}" if match else "---"
            )
        lines.append(
            f"DW-{model} & " + " & ".join(row_vals) + r" \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("[saved] %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args) -> None:
    from omegaconf import OmegaConf

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config_path)

    # Map model name -> checkpoint path (skip if not provided)
    checkpoints = {
        "Small": args.small_ckpt,
        "Base":  args.base_ckpt,
        "Large": args.large_ckpt,
    }

    all_results: List[Dict] = []

    for model_name, ckpt_path in checkpoints.items():
        if not ckpt_path:
            logger.warning("No checkpoint provided for %s, skipping.", model_name)
            continue
        if not Path(ckpt_path).exists():
            logger.warning("Checkpoint not found for %s: %s", model_name, ckpt_path)
            continue

        results = benchmark_model(
            checkpoint_path = ckpt_path,
            model_name      = model_name,
            cfg             = cfg,
            horizons_h      = args.horizons,
            batch_sizes     = args.batch_sizes,
        )
        all_results.extend(results)

    if not all_results:
        logger.error("No results collected. Check checkpoint paths.")
        return

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_json(all_results, str(out_dir / "efficiency_results.json"))
    save_csv(all_results,  str(out_dir / "efficiency_results.csv"))

    save_latex_table(
        all_results,
        save_path       = str(out_dir / "efficiency_table.tex"),
        batch_size_main = 1,
    )
    save_throughput_table(
        all_results,
        save_path = str(out_dir / "throughput_table.tex"),
        horizon_h = 1,
    )

    # ── Print summary to console ──────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("Summary (batch_size=1)")
    logger.info("%-10s  %-6s  %-8s  %-18s  %-10s",
                "Model", "H (h)", "pred_len", "Latency (ms)", "VRAM (GB)")
    logger.info("-" * 60)
    for r in sorted(all_results, key=lambda x: (x["model"], x["horizon_h"])):
        if r["batch_size"] != 1:
            continue
        logger.info(
            "%-10s  %-6d  %-8d  %7.1f ± %5.1f    %-10.2f",
            r["model"], r["horizon_h"], r["pred_len"],
            r["mean_ms"], r["std_ms"], r["peak_vram_gb"],
        )
    logger.info("=" * 60)
    logger.info("All outputs saved to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)s %(message)s",
        datefmt = "%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    # Checkpoint paths
    parser.add_argument("--small_ckpt",  default="/scratch/pawsey0115/hwang4/results/DeepWind-Research/checkpoints/deepwind/deepwind_small_scada_1_wtk_9", help="Path to DeepWind-Small checkpoint")
    parser.add_argument("--base_ckpt",   default="/scratch/pawsey0115/hwang4/deepwind_experiments/checkpoints/deepwind_base_v1", help="Path to DeepWind-Base checkpoint")
    parser.add_argument("--large_ckpt",  default="/scratch/pawsey0115/hwang4/results/DeepWind-Research/checkpoints/deepwind/deepwind_large_v5", help="Path to DeepWind-Large checkpoint")
    parser.add_argument("--config_path", default="/scratch/pawsey0115/hwang4/project_codes/deepwind_research/configs/eval.yaml")
    parser.add_argument("--output_dir",  default="./paper/efficiency_outputs")
    # Benchmark settings
    parser.add_argument("--horizons",    type=int, nargs="+",
                        default=HORIZONS_H,
                        help="Forecasting horizons in hours")
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                        default=BATCH_SIZES,
                        help="Batch sizes to benchmark")
    parser.add_argument("--n_warmup",    type=int, default=N_WARMUP)
    parser.add_argument("--n_iters",     type=int, default=N_ITERS)

    args = parser.parse_args()
    main(args)