"""
src/evaluation/reporter.py
Metric computation and result persistence for DeepWindModel evaluation.

Responsibilities:
    - Compute forecasting metrics from raw inference results.
    - Save metrics as JSON and raw arrays as compressed .npz.
    - Print a formatted summary to stdout / logger.
    - Expose a single report() entry point for the evaluate.py main loop.

This module is rank-0-only. All public methods assume they are called
exclusively on the main process; callers must guard with is_main_process().
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from omegaconf import DictConfig

from src.utils.metrics import ForecastingEvaluator
from src.evaluation.registry import get_resolution

logger = logging.getLogger(__name__)


# ── JSON serialisation helper ──────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars and arrays to native Python types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Reporter ───────────────────────────────────────────────────────────────────

class EvaluationReporter:
    """
    Handles all output resulting from one evaluation run.

    Separates metric computation from inference so that evaluator.py remains
    focused on data movement and this class owns all result representation.

    Args:
        output_dir: Root directory for all saved outputs.
                    Sub-directories are created per dataset automatically.
        cfg:        Hydra DictConfig. Used to read:
                        cfg.inference.seasonality  — season length for MASE
                        cfg.output.save_raw_results — whether to write .npz files
                        cfg.output.save_plots       — whether to call visualiser
                        cfg.output.num_plots        — number of forecast plots
                        cfg.output.plot_hist_len    — history steps shown in plots
                        cfg.seed                    — random seed for plot sampling
        quantiles:  Quantile levels registered in the model (e.g. [0.05, …, 0.95]).
                    Used to initialise ForecastingEvaluator.
    """

    def __init__(
        self,
        output_dir: str,
        cfg:        DictConfig,
        quantiles:  np.ndarray,
    ) -> None:
        self.output_dir = output_dir
        self.cfg        = cfg
        self.quantiles  = np.asarray(quantiles)

    # ── Public entry point ─────────────────────────────────────────────────────

    def report(
        self,
        results:      Dict[str, np.ndarray],
        dataset_name: str,
        horizon_h:    int,
        pred_len:     int,
        capacity_val: float,
    ) -> Dict:
        """
        Full reporting pipeline for one (dataset, horizon) combination.

        Sequentially:
            1. Compute metrics.
            2. Save metrics to JSON.
            3. Optionally save raw arrays to .npz.
            4. Optionally generate forecast plots.
            5. Print a summary to logger.

        Args:
            results:      Output of DistributedEvaluator.run() on rank 0.
                          Keys: point_preds, quantile_preds, targets, history.
            dataset_name: Short identifier used for directory naming and logging.
            horizon_h:    Forecast horizon in hours (used for file naming).
            pred_len:     Forecast horizon in time steps.
            capacity_val: Installed capacity (MW) for normalised metric computation.
        
        Returns:
            Computed metrics dict (also persisted to disk).
        """
        if not results:
            logger.warning(f"[{dataset_name}] Empty results passed to reporter — skipping.")
            return {}

        results["point_preds"]    = np.clip(results["point_preds"],    0, None)
        results["quantile_preds"] = np.clip(results["quantile_preds"], 0, None)

        capacity_array = self._make_capacity_array(results["targets"], capacity_val)

        metrics = self._compute_metrics(results, capacity_array, dataset_name)
        self._save_metrics(metrics, dataset_name, horizon_h, pred_len)

        if self.cfg.output.get("save_rolling_plots", False):
            self._save_rolling_plots(results, dataset_name, horizon_h, pred_len)    

        if self.cfg.output.get("save_raw_results", False):
            self._save_raw_results(results, dataset_name, horizon_h, capacity_val)

        if self.cfg.output.get("save_plots", False):
            self._save_plots(results, dataset_name, horizon_h)

        self._print_summary(metrics, dataset_name, horizon_h)

        return metrics

    def _get_evaluator(self, dataset_name: str) -> ForecastingEvaluator:
            """
            Build a ForecastingEvaluator with the correct seasonality for this dataset.
            Falls back to cfg.inference.seasonality_fallback if dataset not in registry.
            """
            try:
                resolution   = get_resolution(dataset_name)   # minutes per step
                season_length = (24 * 60) // resolution        # steps per day
            except KeyError:
                season_length = self.cfg.inference.get("seasonality_fallback", 96)
                logger.warning(
                    f"[{dataset_name}] Not found in registry — "
                    f"using fallback seasonality={season_length}"
                )
    
            logger.info(f"[{dataset_name}] seasonality={season_length} steps/day")
    
            return ForecastingEvaluator(
                season_length = season_length,
                quantiles     = self.quantiles,
            )

    # ── Metric computation ─────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        results:      Dict[str, np.ndarray],
        capacity:     np.ndarray,
        dataset_name: str,
    ) -> Dict:
        """
        Compute all standard forecasting metrics via ForecastingEvaluator.

        Delegates to evaluator.compute_all() which returns capacity-normalised
        deterministic metrics, nCRPS, coverage calibration, and per-quantile
        hit rates in a single call.

        Args:
            results:      Dict with point_preds, quantile_preds, targets, history.
            capacity:     Per-sample capacity array. Shape: (N,).
            dataset_name: Used only for log messages.

        Returns:
            Flat dict of scalar metric values.
        """
        logger.info(f"[{dataset_name}] Computing metrics...")
        evaluator = self._get_evaluator(dataset_name)
        return evaluator.compute_all(
            y_true      = results["targets"],        # (N, pred_len)
            y_pred      = results["point_preds"],    # (N, pred_len)
            y_quantiles = results["quantile_preds"], # (N, pred_len, Q)
            capacity    = capacity,                  # (N,)
        )

    # ── Persistence ────────────────────────────────────────────────────────────
    
    def _save_metrics(
        self,
        metrics:      Dict,
        dataset_name: str,
        horizon_h:    int,
        pred_len:     int,
    ) -> str:
        """
        Serialise metrics to a JSON file.

        Output path: <output_dir>/<dataset_name>/H_<horizon_h>.json

        Returns:
            Absolute path of the written file.
        """
        save_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"H_{horizon_h}.json")

        payload = {
            "dataset":        dataset_name,
            "horizon_hours":  horizon_h,
            "pred_len_steps": pred_len,
            "metrics":        metrics,
            "timestamp":      datetime.now().isoformat(),
        }

        with open(save_path, "w") as f:
            json.dump(payload, f, indent=4, cls=_NumpyEncoder)

        logger.info(f"[{dataset_name}] Metrics saved → {save_path}")
        return save_path

    def _save_raw_results(
        self,
        results:      Dict[str, np.ndarray],
        dataset_name: str,
        horizon_h:    int,
        capacity_val: float,
    ) -> str:
        """
        Save raw inference arrays as a compressed .npz archive.

        Useful for post-hoc analysis and custom plotting without re-running
        the full inference pipeline.

        Output path: <output_dir>/raw_results/<dataset_name>/raw_H<horizon_h>.npz

        Returns:
            Absolute path of the written file.
        """
        save_dir = os.path.join(self.output_dir, "raw_results", dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"raw_H{horizon_h}.npz")

        np.savez_compressed(
            save_path,
            point_preds    = results["point_preds"],      # (N, pred_len)
            quantile_preds = results["quantile_preds"],   # (N, pred_len, Q)
            targets        = results["targets"],           # (N, pred_len)
            history        = results["history"],           # (N, context_len)
            capacity       = np.float32(capacity_val),
            dataset_name   = dataset_name,
            horizon_h      = horizon_h,
            quantiles      = self.quantiles,
        )

        logger.info(f"[{dataset_name}] Raw results saved → {save_path}")
        return save_path

    def _save_plots(
        self,
        results:      Dict[str, np.ndarray],
        dataset_name: str,
        horizon_h:    int,
    ) -> None:
        """
        Generate and save forecast visualisation plots.

        Delegates entirely to src.utils.vis.visualize_forecasts.
        Controlled by cfg.output.{num_plots, plot_hist_len, save_plots, seed}.
        """
        from src.utils.vis import visualize_forecasts

        plot_dir = os.path.join(self.output_dir, "plots")
        logger.info(f"[{dataset_name}] Generating {self.cfg.output.num_plots} forecast plots...")

        visualize_forecasts(
            results         = results,
            quantiles_list  = self.quantiles,
            save_dir        = plot_dir,
            dataset_name    = dataset_name,
            num_plots       = self.cfg.output.num_plots,
            plot_hist_len   = self.cfg.output.plot_hist_len,
            seed            = self.cfg.seed,
        )
    
    def _save_rolling_plots(
        self,
        results:      Dict[str, np.ndarray],
        dataset_name: str,
        horizon_h:    int,
        pred_len:     int,
    ) -> None:
        """
        Concatenate rolling forecasts and plot against ground truth.

        If cfg.output.rolling_plot_segment is set, the full sequence is split
        into fixed-length segments and saved as separate figures.

        Requires results to be index-sorted (guaranteed by _trim after the
        'indices' fix in evaluator.py).

        Args:
            results:      Trimmed, sorted inference results from DistributedEvaluator.
            dataset_name: Used for file naming and plot title.
            horizon_h:    Forecast horizon in hours (for title/filename).
            pred_len:     Number of steps per window.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        targets     = results["targets"]       # (N, pred_len)
        point_preds = results["point_preds"]   # (N, pred_len)
        q_preds     = results["quantile_preds"] # (N, pred_len, Q)

        # ── Concatenate into full time series ────────────────────────────────────
        targets_full     = targets.reshape(-1)       # (N * pred_len,)
        point_preds_full = point_preds.reshape(-1)
        
        # Pick a few representative quantiles for shading
        q_indices = {
            "q10": np.argmin(np.abs(self.quantiles - 0.10)),
            "q90": np.argmin(np.abs(self.quantiles - 0.90)),
            "q25": np.argmin(np.abs(self.quantiles - 0.25)),
            "q75": np.argmin(np.abs(self.quantiles - 0.75)),
        }
        q10_full = q_preds[:, :, q_indices["q10"]].reshape(-1)
        q90_full = q_preds[:, :, q_indices["q90"]].reshape(-1)
        q25_full = q_preds[:, :, q_indices["q25"]].reshape(-1)
        q75_full = q_preds[:, :, q_indices["q75"]].reshape(-1)

        T = len(targets_full)
        t = np.arange(T)

        # ── Segment config ───────────────────────────────────────────────────────
        seg_len  = self.cfg.output.get("rolling_plot_segment", None)
        segments = self._make_segments(T, seg_len)
        
        # Cap number of segments to save
        max_segs = self.cfg.output.get("rolling_plot_max_segs", None)
        if max_segs is not None and len(segments) > max_segs:
            logger.info(
                f"[{dataset_name}] {len(segments)} segments total — "
                f"saving first {max_segs} only."
            )
            segments = segments[:max_segs]

        # ── Output dir ───────────────────────────────────────────────────────────
        plot_dir = os.path.join(
            self.output_dir, "rolling_plots", dataset_name
        )
        os.makedirs(plot_dir, exist_ok=True)

        # ── Plot each segment ────────────────────────────────────────────────────
        for seg_idx, (start, end) in enumerate(segments):
            sl = slice(start, end)

            fig, ax = plt.subplots(figsize=(16, 4))
            
            ax.fill_between(
                t[sl], q10_full[sl], q90_full[sl],
                alpha=0.15, color="steelblue", label="90% PI"
            )
            ax.fill_between(
                t[sl], q25_full[sl], q75_full[sl],
                alpha=0.25, color="steelblue", label="50% PI"
            )
            ax.plot(t[sl], targets_full[sl],
                    color="black",     lw=1.2, label="Ground truth")
            ax.plot(t[sl], point_preds_full[sl],
                    color="steelblue", lw=1.0, alpha=0.85, label="Forecast (median)")

            # Mark horizon boundaries — step adapts to rolling_plot_segment
            seg_range = end - start
            if seg_range <= 480:
                n_lines = 10
            elif seg_range <= 1440:
                n_lines = 8
            elif seg_range <= 4320:
                n_lines = 6
            else:
                n_lines = 4

            step = max(pred_len, int(seg_range / n_lines))
            boundaries = np.arange(start, end, step)
            for b in boundaries:
                ax.axvline(b, color="gray", lw=0.4, ls="--", alpha=0.5)

            n_segs  = len(segments)
            seg_str = f"  (segment {seg_idx + 1}/{n_segs})" if n_segs > 1 else ""
            ax.set_title(
                f"{dataset_name}  |  H={horizon_h}h  |  "
                f"steps [{start}–{end}]{seg_str}"
            )
            ax.set_xlabel("Time step")
            ax.set_ylabel("Power (normalised)")
            ax.legend(loc="upper right", fontsize=8)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=12))
            ax.grid(True, alpha=0.3)

            fig.tight_layout()

            fname = (
                f"rolling_H{horizon_h}_seg{seg_idx + 1:03d}.png"
                if n_segs > 1
                else f"rolling_H{horizon_h}.png"
            )
            save_path = os.path.join(plot_dir, fname)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"[{dataset_name}] Rolling plot saved → {save_path}")

    @staticmethod
    def _make_segments(
        total_len: int,
        seg_len:   Optional[int],
    ) -> list[tuple[int, int]]:
        """
        Split [0, total_len) into fixed-length segments.

        Args:
            total_len: Total number of time steps.
            seg_len:   Segment length. None = single segment (no split).

        Returns:
            List of (start, end) tuples.
        """
        if seg_len is None or seg_len >= total_len:
            return [(0, total_len)]

        segments = []
        start = 0
        while start < total_len:
            end = min(start + seg_len, total_len)
            segments.append((start, end))
            start = end
        return segments

        # ── Summary printing ───────────────────────────────────────────────────────

    def _print_summary(
        self,
        metrics:      Dict,
        dataset_name: str,
        horizon_h:    int,
    ) -> None:
        """Log a formatted metric summary to the console."""
        header = f"[{dataset_name}]  Horizon: {horizon_h}h"
        divider = "─" * 50
        lines = [
            "",
            divider,
            header,
            divider,
        ]
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<30s}: {v:.4f}")
            else:
                lines.append(f"  {k:<30s}: {v}")
        lines.append(divider)

        logger.info("\n".join(lines))

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_capacity_array(targets: np.ndarray, capacity_val: float) -> np.ndarray:
        """
        Build a per-sample capacity array broadcastable with targets.

        Args:
            targets:      Ground truth array. Shape: (N, pred_len).
            capacity_val: Scalar installed capacity (MW).

        Returns:
            np.ndarray of shape (N,) filled with capacity_val.
        """
        N = targets.shape[0]
        return np.full(N, capacity_val, dtype=np.float32)