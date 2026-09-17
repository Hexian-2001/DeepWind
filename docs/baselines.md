# Baselines & Model Comparison

## Contents

- [1. Model taxonomy](#1-model-taxonomy)
- [2. Data protocol (identical for every model)](#2-data-protocol-identical-for-every-model)
- [3. Metrics](#3-metrics)
- [4. Small-model hyperparameters](#4-small-model-hyperparameters)
- [5. Results](#5-results)
- [6. Takeaway](#6-takeaway)
- [7. Reproduce](#7-reproduce)
- [8. Register into the leaderboard](#8-register-into-the-leaderboard)


This document records the comparison models evaluated against DeepWind on the
8 WindBench test datasets × 6 horizons, how they were trained/evaluated, and
where their results live (the git-committed `results/leaderboard.jsonl`).

The harness itself is **decoupled** from DeepWind and lives at
`/scratch/pawsey0115/hwang4/baselines/` (see `small_models.py`). It only
imports `src/utils/metrics.py::ForecastingEvaluator` to produce the *exact*
same metrics as DeepWind's own evaluation.

## 1. Model taxonomy

| Family | Models | Protocol |
|---|---|---|
| Time-series foundation models (TSFM) | chronos-2, timesfm-2.5, moirai-2 (R-small), tirex | zero-shot inference |
| Statistical | ARIMA | trained on the 70% train split |
| Gradient-boosted | LightGBM | trained on the 70% train split |
| Deep learning | DLinear, N-BEATS, PatchTST, DeepAR | trained on the 70% train split |

Time-LLM was skipped (requires Llama weights; out of scope for this pass).

> **TiRex is not re-inferred in this pass.** The PyPI package `tirex` is an
> unrelated scikit-learn estimator, not NX-AI's TiRex TSFM, and the original
> TiRex inference source is no longer present. Its legacy partial results
> (horizons `3/5/7/9/10/11` only — not the standard 6 — with some datasets
> missing horizons) are therefore stale and **excluded** from the comparison
> table below.

## 2. Data protocol (identical for every model)

* Each `.npy` is `(V, T)`; the single wind-power series is row 0.
* `train = series[0 : 0.7T]`, `val = series[0.7T : 0.8T]`, `test = series[0.8T : T]`.
* Test windows slide over the test slice with `context = 1024` and
  `stride = pred_len`. TSFMs, statistical and deep-learning baselines all use
  this identical windowing, so their test targets are byte-identical.
* `pred_len = horizon_hours × 60 / resolution` (resolution per dataset in
  `DATASET_RES_CONFIG`).

## 3. Metrics

Capacity-normalised. The 7 headline metrics are `nCRPS, nMAE, Accuracy,
Qualified_Rate, MAE_Coverage, R2, mean_wQuantileLoss`. `Qualified_Rate` uses the
same horizon-aware threshold as DeepWind: `0.15` for `horizon ≤ 4 h`, else
`0.25`. The small models emit a 21-quantile grid `[0.01 … 0.99]`; the TSFMs emit
their native 9-quantile grid `[0.1 … 0.9]`.

## 4. Small-model hyperparameters

| Model | Specification |
|---|---|
| ARIMA | `ARIMA(2,1,1)`, refit per test window on the last 384 steps; Gaussian residual std |
| LightGBM | 300 trees, `num_leaves=31`, `lr=0.05`, features = last 96 lags + seasonal lags (`day±1`, `2/3/7×day`); direct multi-step (one tree per step); residual std |
| DLinear | decomposition kernel 25, two linear heads (mean + log-std), Gaussian NLL |
| N-BEATS | 2 stacks × 1 block, hidden 256, backcast/forecast residual, Gaussian NLL |
| PatchTST | patch 32 / stride 16, `d_model=96`, 8 heads, 2 layers, Gaussian NLL |
| DeepAR | LSTM (hidden 128, 2 layers) over last 256 steps → linear Gaussian head |

Common training: Adam `lr=1e-3`, batch 64, ≤ 25 epochs, early stop patience 6,
≤ 8000 training windows (stride 1), z-score normalisation (fit on train only).
Quantiles are produced from the Gaussian predictive distribution
(`mean + Φ⁻¹(q)·σ`) for every model, so `MAE_Coverage` / `nCRPS` are directly
comparable across models.

## 5. Results

Macro-average (8 datasets × 6 horizons) — see `results/leaderboard.jsonl` for
the authoritative rows and `tools/compare_models.py --summary` for ranking.
Ordered by nCRPS (lower is better).

| model_id | variant | nCRPS ↓ | nMAE ↓ | Accuracy ↑ | Qualified_Rate ↑ | MAE_Coverage ↓ | R2 ↑ | mean_wQuantileLoss ↓ |
|---|---|---|---|---|---|---|---|---|
| baseline-deepar | baseline (deep learning) | 0.0937 | 0.1381 | 0.8060 | 0.7626 | 0.0469 | 0.6433 | 0.2533 |
| baseline-lightgbm | baseline (gradient-boosted) | 0.0996 | 0.1393 | 0.8028 | 0.7564 | 0.0502 | 0.6319 | 0.2710 |
| baseline-patchtst | baseline (deep learning) | 0.1019 | 0.1474 | 0.7978 | 0.7436 | 0.0371 | 0.6139 | 0.2757 |
| baseline-chronos-2 | baseline (TSFM) | 0.1027 | 0.1329 | 0.7961 | 0.7728 | 0.0379 | 0.6026 | 0.2766 |
| baseline-moirai-2 | baseline (TSFM) | 0.1043 | 0.1331 | 0.7929 | 0.7748 | 0.0242 | 0.5882 | 0.2814 |
| baseline-timesfm-2.5 | baseline (TSFM) | 0.1056 | 0.1351 | 0.7945 | 0.7683 | 0.0439 | 0.5944 | 0.2844 |
| baseline-dlinear | baseline (deep learning) | 0.1087 | 0.1528 | 0.7911 | 0.7325 | 0.0344 | 0.5897 | 0.2961 |
| baseline-nbeats | baseline (deep learning) | 0.1109 | 0.1639 | 0.7827 | 0.6928 | 0.0474 | 0.5585 | 0.3006 |
| deepwind-small-paper-seed42 | small | 0.1121 | 0.1482 | 0.7814 | 0.7327 | 0.1085 | 0.5268 | 0.3120 |
| baseline-arima | baseline (statistical) | 0.1198 | 0.1384 | 0.7825 | 0.7619 | 0.0966 | 0.5407 | 0.3210 |

> The TSFMs were **re-inferred** on the *same* context and windowing as the
> small models (context = 1024, test = `series[0.8T:]`, stride = `pred_len`) and
> scored with the full 7-metric, horizon-aware protocol. The only remaining
> difference is the quantile grid: the TSFMs natively emit 9 quantiles
> `[0.1 … 0.9]`, whereas the small models emit 21 `[0.01 … 0.99]`, so the TSFMs'
> `nCRPS` / `MAE_Coverage` / `mean_wQuantileLoss` are a slightly coarser CRPS
> approximation. Widening the context 512 → 1024 moved `nCRPS`/`nMAE`/`Accuracy`
> by < 1e-4; the visible change versus the previously-registered (2026-01-30)
> numbers is `Qualified_Rate` — now horizon-aware (0.15 for ≤ 4 h / 0.25
> otherwise) instead of a fixed 0.25 — and the three previously-missing metrics,
> which are now populated.

## 6. Takeaway

DeepAR is the strongest trained small model (nCRPS 0.0937, R2 0.6433),
beating all three TSFMs and DeepWind-Small on this protocol; LightGBM is the
strongest non-deep baseline (0.0996). The TSFMs now sit on the same context and
threshold as the small models; the only remaining mismatch is 9 vs 21 quantiles
(a slightly coarser CRPS for the TSFMs), so the ordering across the two groups
is directly comparable with that one caveat.

## 7. Reproduce

```bash
# install deps (into the deepwind venv)
/scratch/pawsey0115/hwang4/conda_envs/deepwind/bin/pip install lightgbm statsmodels scikit-learn

# run one cell
/scratch/pawsey0115/hwang4/conda_envs/deepwind/bin/python \
  /scratch/pawsey0115/hwang4/baselines/small_models.py \
  --model dlinear --dataset 30651 --horizon 1 --device cuda

# full sweep (see baselines/run_cpu.sbatch, run_small.sbatch, submit_all.sh)

# TSFMs (zero-shot) — same protocol, 9 quantiles
/scratch/pawsey0115/hwang4/conda_envs/deepwind/bin/python \
  /scratch/pawsey0115/hwang4/baselines/tsfm_models.py \
  --model chronos-2 --datasets all --horizons 1,2,4,6,8,12 --device cuda
# (or via baselines/run_tsfm.sbatch, which also sets HF_HOME/HOME for cached weights)
```

## 8. Register into the leaderboard

```bash
python tools/register_baselines.py \
  /scratch/pawsey0115/hwang4/deepwind_experiments/baselines/results/<MODEL_DIR> \
  --model-id baseline-<name> --tags baseline,<family>,trained
```

---

**Related docs:** [Reproducibility record](reproducibility.md)
