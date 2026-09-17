"""Unified zero-shot evaluation protocol shared by DeepWind and every baseline.

Goal
----
Make the *target* (test) windows byte-identical across all compared models so
the leaderboard is a fair, apples-to-apples comparison:

  * The full series for a site is (C, T); the evaluation "test" slice is
    ``series[t_val : T]`` with ``t_val = int(0.8 * T)``.
  * Targets are the non-overlapping length-``pred_len`` windows tiling
    ``[t_val, T)``:  ``target_k = series[s : s + pred_len]`` for
    ``s in [t_val, t_val + pred_len, ..., < T - pred_len]``.
  * Every model predicts those exact targets. The only per-model difference is
    the context length: ``context_k = series[s - ctx : s]`` for the model's
    ``ctx`` (DeepWind 8192, small/TSFM baselines 1024). Context may extend
    before ``t_val`` into val/train -- that is the history a forecaster
    legitimately holds at prediction time.

This replaces the earlier start-aligned windowing where a longer context
shifted the target timestamps, making the model scores incomparable.
"""
from __future__ import annotations

import numpy as np


def test_target_starts(
    T: int,
    pred_len: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> list[int]:
    """Absolute start indices of the canonical test targets (row-0 time axis).

    Returns start indices ``s`` (into the full series) such that the targets
    ``series[s : s + pred_len]`` tile ``[t_val, T)`` non-overlapping with the
    trailing (possibly partial) window dropped. Identical across all models.
    """
    t_val = int(T * (train_ratio + val_ratio))
    if T - t_val < pred_len:
        return []
    return list(range(t_val, T - pred_len + 1, pred_len))


def make_windows(
    series: np.ndarray,
    target_starts: list[int],
    ctx: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` using ``ctx`` steps of history before each target.

    Args:
        series:        1-D array (the single wind-power series, row 0).
        target_starts: start indices from :func:`test_target_starts`.
        ctx:           context length (model-specific).
        pred_len:      prediction length.

    Returns:
        X: (N, ctx) float32 contexts; y: (N, pred_len) float32 targets.
    """
    if not target_starts:
        return (
            np.zeros((0, ctx), dtype=np.float32),
            np.zeros((0, pred_len), dtype=np.float32),
        )
    X = np.stack([series[s - ctx:s] for s in target_starts]).astype(np.float32)
    y = np.stack([series[s:s + pred_len] for s in target_starts]).astype(np.float32)
    return X, y
