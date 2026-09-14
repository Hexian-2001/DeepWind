"""
src/utils/metrics.py
Forecasting evaluation metrics for wind power probabilistic forecasts.

All metrics normalise by installed capacity (MW) rather than by the sum of
actuals, which is the convention used in grid-code accuracy assessment.

Capacity-normalised metrics (lower is better unless noted):
    nRMSE          — normalised root mean square error
    nMAE           — normalised mean absolute error
    Accuracy       — grid accuracy: 1 - nRMSE  (higher is better)
    Qualified_Rate — fraction of steps within a capacity threshold

Probabilistic metrics:
    nCRPS          — normalised continuous ranked probability score
    Coverage       — empirical quantile hit rates  (calibration check)
    MAE_Coverage   — mean absolute deviation of Coverage from nominal levels

Optional / auxiliary:
    MASE           — mean absolute scaled error vs seasonal naive
"""
import logging
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class ForecastingEvaluator:
    """
    Compute deterministic and probabilistic forecasting metrics.

    Args:
        season_length: Seasonal period used for the naive baseline (MASE).
        quantiles:     Array of quantile levels in (0, 1). Defaults to a
                       standard 21-quantile grid from 0.01 to 0.99.
    """

    _DEFAULT_QUANTILES = np.array([
        0.01, 0.05, 0.10, 0.15, 0.20,
        0.25, 0.30, 0.35, 0.40, 0.45,
        0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 0.99,
    ])

    def __init__(
        self,
        season_length: int = 24,
        quantiles: Optional[np.ndarray] = None,
    ) -> None:
        self.season_length = season_length
        self.quantiles     = (
            np.asarray(quantiles, dtype=np.float64)
            if quantiles is not None
            else self._DEFAULT_QUANTILES.copy()
        )
        # Pre-compute z-scores for Gaussian probabilistic naive baseline
        self._z_scores = norm.ppf(self.quantiles)

    # ── Capacity helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _prepare_capacity(capacity: np.ndarray) -> np.ndarray:
        """
        Normalise capacity to shape (N, 1) for broadcasting against (N, T).

        Accepts shapes: scalar, (N,), or (N, 1).

        Raises:
            ValueError: If capacity is None, contains non-positive values,
                        or has an incompatible shape.
        """
        if capacity is None:
            raise ValueError("capacity must not be None.")

        cap = np.asarray(capacity, dtype=np.float64)

        if cap.ndim == 0:
            cap = cap.reshape(1, 1)
        elif cap.ndim == 1:
            cap = cap[:, np.newaxis]          # (N,) → (N, 1)
        elif cap.ndim == 2 and cap.shape[1] == 1:
            pass                              # already (N, 1)
        else:
            raise ValueError(
                f"capacity must be scalar, shape (N,) or (N, 1); got {cap.shape}."
            )

        if np.any(cap <= 0):
            raise ValueError("All capacity values must be strictly positive.")

        return cap

    # ── Deterministic metrics ──────────────────────────────────────────────────

    def calc_nrmse(
        self,
        y_true:    np.ndarray,
        y_pred:    np.ndarray,
        capacity:  np.ndarray,
    ) -> float:
        """
        Normalised Root Mean Square Error.

        Formula:  sqrt( mean( ((y_true - y_pred) / C)^2 ) )
        Range:    [0, ∞), lower is better.

        Args:
            y_true:   Ground truth.          Shape: (N, T)
            y_pred:   Point forecast.        Shape: (N, T)
            capacity: Installed capacity.    Shape: (N,) or (N, 1)
        """
        cap    = self._prepare_capacity(capacity)     # (N, 1)
        sq_err = ((y_true - y_pred) / cap) ** 2       # (N, T)
        return float(np.sqrt(np.nanmean(sq_err)))

    def calc_nmae(
        self,
        y_true:    np.ndarray,
        y_pred:    np.ndarray,
        capacity:  np.ndarray,
    ) -> float:
        """
        Normalised Mean Absolute Error.

        Formula:  mean( |y_true - y_pred| / C )
        Range:    [0, ∞), lower is better.
        """
        cap     = self._prepare_capacity(capacity)
        abs_err = np.abs(y_true - y_pred) / cap       # (N, T)
        return float(np.nanmean(abs_err))

    def calc_grid_accuracy(
        self,
        y_true:   np.ndarray,
        y_pred:   np.ndarray,
        capacity: np.ndarray,
    ) -> float:
        """
        Grid Accuracy: 1 - nRMSE.

        Range: (-∞, 1.0], higher is better (ideal = 1.0).
        """
        return 1.0 - self.calc_nrmse(y_true, y_pred, capacity)

    def calc_qualified_rate(
        self,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        capacity:   np.ndarray,
        threshold:  float = 0.25,
    ) -> float:
        """
        Qualified Rate: fraction of time steps where the absolute error is
        within a capacity-relative tolerance band.

        Formula:  mean( |y_true - y_pred| <= C * threshold )
        Range:    [0.0, 1.0], higher is better.

        Args:
            threshold: Tolerance as a fraction of capacity.
                       0.25 for day-ahead, 0.15 for ultra-short-term.
        """
        cap         = self._prepare_capacity(capacity)
        abs_err     = np.abs(y_true - y_pred)          # (N, T)
        allowed     = cap * threshold                   # (N, 1)
        is_qual     = abs_err <= allowed                # (N, T) bool
        return float(np.nanmean(is_qual.astype(np.float64)))

    # ── Probabilistic metrics ──────────────────────────────────────────────────

    def calc_ncrps(
        self,
        y_true:       np.ndarray,
        y_quantiles:  np.ndarray,
        capacity:     np.ndarray,
    ) -> float:
        """
        Normalised Continuous Ranked Probability Score (nCRPS).

        Approximated via the average pinball (quantile) loss across all
        registered quantile levels, following the relationship:
            CRPS ≈ (2 / Q) * sum_q pinball_q

        Normalised by capacity to allow cross-site comparison:
            nCRPS = mean_{n,t} [ (2/Q) * sum_q pinball_q(y, f_q) / C_n ]

        Implemented via fully vectorised operations — no Python quantile loop.

        Args:
            y_true:      Ground truth.              Shape: (N, T)
            y_quantiles: Quantile predictions.      Shape: (N, T, Q)
            capacity:    Installed capacity.        Shape: (N,) or (N, 1)

        Range: [0, ∞), lower is better.
        """
        cap = self._prepare_capacity(capacity)         # (N, 1)

        # Squeeze y_true to 2D if needed
        if y_true.ndim == 3:
            y_true = y_true[..., 0]

        # Expand y_true for broadcasting against quantile axis: (N, T, 1)
        y_exp = y_true[..., np.newaxis]

        # errors[n, t, q] = y_true[n,t] - y_quantile[n,t,q]
        errors = y_exp - y_quantiles                   # (N, T, Q)

        # Quantile levels broadcast: (1, 1, Q)
        q = self.quantiles.reshape(1, 1, -1)

        # Pinball loss: max(q*e, (q-1)*e)   shape: (N, T, Q)
        pinball = np.maximum(q * errors, (q - 1) * errors)

        # Sum over quantiles → (N, T), scale by 2/Q
        crps_per_step = (2.0 / len(self.quantiles)) * pinball.sum(axis=-1)

        # Normalise by capacity: (N, T) / (N, 1) → (N, T)
        ncrps_per_step = crps_per_step / cap

        return float(np.nanmean(ncrps_per_step))

    def calc_coverage(
        self,
        y_true:      np.ndarray,
        y_quantiles: np.ndarray,
    ) -> np.ndarray:
        """
        Empirical coverage for each quantile level.

        Coverage(q) = mean_{n,t} [ 1(y_{n,t} <= f_{q,n,t}) ]

        A well-calibrated model has Coverage(q) ≈ q for all q.

        Args:
            y_true:      Ground truth.           Shape: (N, T)
            y_quantiles: Quantile predictions.   Shape: (N, T, Q)

        Returns:
            np.ndarray of shape (Q,) — empirical hit rate per quantile.
        """
        # (N, T, 1) <= (N, T, Q) → (N, T, Q) bool
        hits = y_true[..., np.newaxis] <= y_quantiles
        return np.nanmean(hits.astype(np.float64), axis=(0, 1))   # (Q,)

    def calc_mae_coverage(
        self,
        y_true:      np.ndarray,
        y_quantiles: np.ndarray,
    ) -> float:
        """
        Mean Absolute Error of empirical Coverage vs nominal quantile levels.

        Formula:  (1/Q) * sum_q |Coverage(q) - q|

        Range: [0.0, 1.0], lower is better (0 = perfect calibration).
        """
        coverage = self.calc_coverage(y_true, y_quantiles)        # (Q,)
        return float(np.mean(np.abs(coverage - self.quantiles)))

    # ── Optional: MASE ────────────────────────────────────────────────────────

    def calc_mase(
        self,
        y_true:  np.ndarray,
        y_pred:  np.ndarray,
        history: np.ndarray,
    ) -> float:
        """
        Mean Absolute Scaled Error relative to a seasonal naive baseline.

        Range: [0, ∞), values < 1 indicate the model outperforms naive.

        Note: this metric is not capacity-normalised and is optional.
              It is not included in reporter.py by default.

        Args:
            y_true:  Ground truth.     Shape: (N, T)
            y_pred:  Point forecast.   Shape: (N, T)
            history: Past context.     Shape: (N, context_len)
        """
        forecast_mae = np.mean(np.abs(y_true - y_pred), axis=1)   # (N,)

        if history.shape[1] > self.season_length:
            seasonal_diff = np.abs(
                history[:, self.season_length:] - history[:, :-self.season_length]
            )
            scale = np.mean(seasonal_diff, axis=1)                 # (N,)
        else:
            scale = np.mean(np.abs(np.diff(history, axis=1)), axis=1)

        valid = scale > 1e-8
        if valid.sum() == 0:
            logger.warning("MASE: all scale values are near zero — returning NaN.")
            return float("nan")

        return float(np.mean(forecast_mae[valid] / scale[valid]))

    # ── R^2 ─────────────────────────────────────────────────────────

    def calc_r2(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Coefficient of Determination (R²).

        Formula:  1 - SS_res / SS_tot
        Range:    (-∞, 1.0], higher is better (ideal = 1.0).

        Note: computed globally across all (N, T) steps, not per-series.
        """
        ss_res = np.nansum((y_true - y_pred) ** 2)
        ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
        if ss_tot < 1e-10:
            logger.warning("R²: ss_tot near zero — returning NaN.")
            return float("nan")
        return float(1.0 - ss_res / ss_tot)


    # ── wQuantileLoss[q], mean_wQuantileLoss ───────────────────────

    def calc_wquantile_losses(
        self,
        y_true:      np.ndarray,
        y_quantiles: np.ndarray,
    ) -> Dict[str, float]:
        """
        Weighted Quantile Loss (GluonTS convention), normalised by sum of actuals.

        wQuantileLoss[q] = Σ_{n,t} 2·pinball_q(y,f_q) / Σ_{n,t} |y|

        Returns dict with keys:
            wQuantileLoss[q]    for each q
            mean_wQuantileLoss  mean across all q
        """
        if y_true.ndim == 3:
            y_true = y_true[..., 0]

        y_exp   = y_true[..., np.newaxis]                     # (N, T, 1)
        errors  = y_exp - y_quantiles                         # (N, T, Q)
        q       = self.quantiles.reshape(1, 1, -1)            # (1, 1, Q)
        pinball = np.maximum(q * errors, (q - 1) * errors)   # (N, T, Q)

        # Sum over (N, T) for each quantile → (Q,)
        ql_sum      = 2.0 * np.nansum(pinball, axis=(0, 1))
        abs_y_sum   = np.nansum(np.abs(y_true))

        if abs_y_sum < 1e-10:
            logger.warning("wQuantileLoss: abs_target_sum near zero — returning NaN.")
            wql = np.full(len(self.quantiles), float("nan"))
        else:
            wql = ql_sum / abs_y_sum                          # (Q,)

        result: Dict[str, float] = {}
        for q_val, w in zip(self.quantiles, wql):
            result[f"wQuantileLoss[{q_val:.2f}]"] = float(w)
        result["mean_wQuantileLoss"] = float(np.nanmean(wql))
        return result
        
    # ── Auxiliary ─────────────────────────────────────────────────────────────

    def get_seasonal_naive_forecast(
        self,
        history:  np.ndarray,
        pred_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Seasonal naive point and probabilistic forecast (Gaussian assumption).

        Args:
            history:  Past observations.  Shape: (N, context_len)
            pred_len: Number of steps to forecast.

        Returns:
            naive_mean:      Shape: (N, pred_len)
            naive_quantiles: Shape: (N, pred_len, Q)
        """
        N, H = history.shape
        naive_mean = np.zeros((N, pred_len), dtype=np.float64)

        for i in range(pred_len):
            idx = -self.season_length + (i % self.season_length)
            naive_mean[:, i] = history[:, idx]

        if H > self.season_length:
            seasonal_diff = history[:, self.season_length:] - history[:, :-self.season_length]
            sigma = np.std(seasonal_diff, axis=1, keepdims=True) + 1e-8
        else:
            sigma = np.std(history, axis=1, keepdims=True) + 1e-8

        # (N, pred_len, Q)
        naive_quantiles = (
            naive_mean[:, :, np.newaxis]
            + self._z_scores.reshape(1, 1, -1) * sigma[:, :, np.newaxis]
        )

        return naive_mean, naive_quantiles

    def compute_all(
        self,
        y_true:       np.ndarray,
        y_pred:       np.ndarray,
        y_quantiles:  np.ndarray,
        capacity:     np.ndarray,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "nCRPS":          self.calc_ncrps(y_true, y_quantiles, capacity),
            "nMAE":           self.calc_nmae(y_true, y_pred, capacity),
            "Accuracy":       self.calc_grid_accuracy(y_true, y_pred, capacity),
            "Qualified_Rate": self.calc_qualified_rate(y_true, y_pred, capacity),
            "MAE_Coverage":   self.calc_mae_coverage(y_true, y_quantiles),
            "R2":             self.calc_r2(y_true, y_pred),           # 新增
        }

        # Per-quantile coverage
        coverage = self.calc_coverage(y_true, y_quantiles)
        for q_val, cov in zip(self.quantiles, coverage):
            metrics[f"Coverage[{q_val:.2f}]"] = float(cov)

        # GluonTS-compatible weighted quantile losses       
        metrics.update(self.calc_wquantile_losses(y_true, y_quantiles))

        return metrics