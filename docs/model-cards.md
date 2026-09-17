# Model cards

<details>
<summary>Contents</summary>

- [1. DeepWind variants](#1-deepwind-variants)
  - [1.1. deepwind-small](#11-deepwind-small)
  - [1.2. deepwind-base](#12-deepwind-base)
  - [1.3. deepwind-large](#13-deepwind-large)
- [2. Baselines](#2-baselines)
- [3. Leaderboard snapshot](#3-leaderboard-snapshot)

</details>

Quick reference for every model on the leaderboard: architecture, parameter
count, model hyperparameters, and training hyperparameters. Authoritative
sources are the YAML configs for DeepWind (`configs/model/*.yaml`,
`configs/training/*.yaml`) and `docs/baselines.md` + `baselines/small_models.py`
for the baselines. Use `tools/model_card.py <model_id>` for the same info on the
CLI.

## 1. DeepWind variants

### 1.1. deepwind-small

| Field | Value |
|---|---|
| Parameters (trainable) | 33.21 M |
| d_model / d_ff | 384 / 1024 |
| Layers / heads | 6 / 6 |
| Dropout | 0.05 |
| Context length | 8192 |
| Patch size / stride | 16 / 16 |
| MoE | 4 experts, top-2 per token |
| Variate attention | every 2 layers |
| Positional | RoPE + xPos |
| Normalisation / activation | RMSNorm / SiLU |
| Prediction head | 21-quantile |

Training — `lr` 1e-4 (cosine, warmup 3% = 3000 steps), 100k steps, global batch
256 (8 x 1 x 32), Adam(0.9, 0.95), weight decay 0.01, grad clip 1.0, BF16.
Config: `configs/model/deepwind_small.yaml` + `configs/training/deepwind_small.yaml`.

### 1.2. deepwind-base

| Field | Value |
|---|---|
| Parameters (trainable) | 888.24 M |
| d_model / d_ff | 1024 / 2816 |
| Layers / heads | 12 / 16 |
| Dropout | 0.05 |
| Context length | 8192 |
| Patch size / stride | 16 / 16 |
| MoE | 8 experts, top-2 per token |
| Variate attention | every 2 layers |
| Positional | RoPE + xPos |
| Normalisation / activation | RMSNorm / SiLU |
| Prediction head | 21-quantile |

Training — `lr` 1e-4 (cosine, warmup 3% = 3000 steps), 100k steps, global batch
256 (4 x 2 x 32), Adam(0.9, 0.95), weight decay 0.01, grad clip 1.0, BF16.
Config: `configs/model/deepwind_base.yaml` + `configs/training/deepwind_base.yaml`.

### 1.3. deepwind-large

| Field | Value |
|---|---|
| Parameters (trainable) | ~1.3 B (estimate — not yet measured) |
| d_model / d_ff | 1024 / 2816 |
| Layers / heads | 18 / 16 |
| Dropout | 0.1 |
| Context length | 8192 |
| Patch size / stride | 16 / 16 |
| MoE | 8 experts, top-2 per token |
| Variate attention | every 2 layers |
| Positional | RoPE + xPos |
| Normalisation / activation | RMSNorm / SiLU |
| Prediction head | 21-quantile |

Training — `lr` 1e-4 (cosine, warmup 3% = 3000 steps), 100k steps, global batch
256 (4 x 2 x 32), Adam(0.9, 0.95), weight decay 0.01, grad clip 1.0, BF16.
Config: `configs/model/deepwind_large.yaml` + `configs/training/deepwind_large.yaml`.

## 2. Baselines

Baselines are trained on the 70% train split with identical windowing
(context 1024, stride = pred_len). Common training: Adam `lr=1e-3`, batch 64,
<= 25 epochs, early-stop patience 6, <= 8000 windows, z-score normalisation.
All emit a Gaussian predictive distribution (mean + Phi^-1(q) * sigma).

| model_id | family | architecture / spec |
|---|---|---|
| baseline-arima | statistical | ARIMA(2,1,1), refit per window on last 384 steps, Gaussian residual std |
| baseline-lightgbm | gradient-boosted | 300 trees, num_leaves=31, lr=0.05, last 96 lags + seasonal lags, direct multi-step |
| baseline-dlinear | deep learning | decomposition kernel 25, two linear heads (mean + log-std), Gaussian NLL |
| baseline-nbeats | deep learning | 2 stacks x 1 block, hidden 256, backcast/forecast residual, Gaussian NLL |
| baseline-patchtst | deep learning | patch 32 / stride 16, d_model=96, 8 heads, 2 layers, Gaussian NLL |
| baseline-deepar | deep learning | LSTM (hidden 128, 2 layers) over last 256 steps -> linear Gaussian head |
| baseline-chronos-2 | TSFM (zero-shot) | native 9-quantile grid [0.1 .. 0.9] |
| baseline-timesfm-2.5 | TSFM (zero-shot) | native 9-quantile grid [0.1 .. 0.9] |
| baseline-moirai-2 | TSFM (zero-shot) | Moirai-2 (R-small), native 9-quantile grid [0.1 .. 0.9] |

TiRex is excluded (the PyPI `tirex` package is unrelated; only stale partial
results remain).

## 3. Leaderboard snapshot

Macro-average (8 datasets x 6 horizons), ordered by nCRPS (lower is better).
Authoritative rows live in `results/leaderboard.jsonl`.

| model_id | variant | nCRPS | nMAE | Accuracy | Qualified_Rate | MAE_Coverage | R2 | mean_wQuantileLoss |
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

---

**Related docs:** [Architecture audit](architecture-audit.md) · [Asset inventory](asset-inventory.md) · [Baselines & model comparison](baselines.md) · [Data](data.md) · [Pawsey workflow](pawsey.md) · [Refactor roadmap](refactor-roadmap.md) · [Reproducibility record](reproducibility.md)

