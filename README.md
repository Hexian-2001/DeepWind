<div align="center">

<img src="assets/deepwind-logo.png" alt="DeepWind logo" width="150">

# DeepWind

### A foundation model for zero-shot wind power forecasting

**Give it a wind site's history — get a calibrated, probabilistic forecast back. No site-specific training.**

[![🤗 small](https://img.shields.io/badge/%F0%9F%A4%97%20Model-DeepWind1.0--33M-FFD21E)](https://huggingface.co/Hexian-2001/DeepWind1.0-33M)
[![🤗 base](https://img.shields.io/badge/%F0%9F%A4%97%20Model-DeepWind1.0--890M-FFD21E)](https://huggingface.co/Hexian-2001/DeepWind1.0-890M)
[![🤗 large](https://img.shields.io/badge/%F0%9F%A4%97%20Model-DeepWind1.0--1.3B-FFD21E)](https://huggingface.co/Hexian-2001/DeepWind1.0-1.3B)
[![license](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

</div>

---

## What is DeepWind?

DeepWind is a decoder-only Transformer pre-trained on **~562 billion wind
observations** drawn from 20 sources (WIND Toolkit + 19 real-world SCADA
collections). It forecasts wind power **zero-shot**: point it at a site it has
*never* seen, feed it a window of history — power, plus whatever weather
covariates you have — and it returns a **21-quantile predictive distribution**
over the next few hours to days.

This repository is the reference implementation for:

> H. Wang et al., "DeepWind: A foundation model for zero-shot wind power
> forecasting," *Energy*, vol. 360, 141793, 2026.
> https://doi.org/10.1016/j.energy.2026.141793

The recipe, in one sentence: **time-aware patching + decoupled time/variate
attention + sparse mixture-of-experts + autoregressive multi-quantile decoding.**

- 🧩 **Time-aware patching** — the series is split into 16-step patches *before*
  it reaches the Transformer, so attention runs over ~512 tokens instead of
  8192 raw steps.
- 🪞 **Decoupled time & variate attention** — one attention axis looks *along*
  time, a second looks *across* variables. Time layers use RoPE, variate layers
  use xPOS.
- 🎛️ **Sparse mixture-of-experts** — every token is routed to the top-2 of a
  handful of experts, so capacity scales without a proportional compute bill.
- 📊 **Direct multi-quantile head** — the model emits 21 quantiles
  (0.01 … 0.99) straight from the final token, giving you intervals and not
  just a single number.

## 🚀 Model zoo

| Model | Params | Zero-shot nCRPS ↓ | Checkpoint |
|---|---|---|---|
| **DeepWind1.0-33M** · small | 33.21 M | 0.0998 | [🤗 `Hexian-2001/DeepWind1.0-33M`](https://huggingface.co/Hexian-2001/DeepWind1.0-33M) |
| **DeepWind1.0-890M** · base | 888.24 M | 0.0961 | [🤗 `Hexian-2001/DeepWind1.0-890M`](https://huggingface.co/Hexian-2001/DeepWind1.0-890M) |
| **DeepWind1.0-1.3B** · large | ~1.33 B | 0.0671 † | [🤗 `Hexian-2001/DeepWind1.0-1.3B`](https://huggingface.co/Hexian-2001/DeepWind1.0-1.3B) |

> † `large` is the recovered paper checkpoint; its number is from the paper's
> original evaluation protocol. `small` and `base` are scored under the unified
> protocol described in [Results](#results).

All three sizes share **exactly the same interface** — swap the checkpoint
string and nothing else in your code changes. Pick by compute budget: `small`
runs happily on a laptop GPU, `base` is the recommended default, `large` is for
when you can afford it.

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="DeepWind overall architecture" width="720">
</p>

1. **Normalise** — each variable is instance-normalised (arcsinh of a z-score)
   *online*, computed from the context window itself, so you never need to
   pre-scale your data.
2. **Patch** — each normalised variable is cut into 16-step patches, and 9
   per-patch statistics (mean, std, min, max, range, net change, …) are appended
   to every patch as a cheap, robust feature bank.
3. **Embed** — three embeddings are summed into every token:
   - a *time* embedding (relative position),
   - a *variate* embedding (which physical variable this channel is),
   - a *spatial* embedding (latitude/longitude → 3-D Cartesian → MLP).
4. **Encode** — a stack of Transformer blocks alternating time-attention and
   variate-attention, with a top-2 Mixture-of-Experts feed-forward in each.
5. **Decode** — the forecaster expands-and-collapses the context to produce a
   21-quantile forecast for every channel, step by step.

### Variants at a glance

| Setting | Small | Base | Large |
|---|---|---|---|
| hidden size (`d_model`) | 384 | 1024 | 1024 |
| layers | 6 | 12 | 18 |
| attention heads | 6 | 16 | 16 |
| FFN width (`d_ff`, SwiGLU) | 1024 | 2816 | 2816 |
| experts / top-k | 4 / 2 | 8 / 2 | 8 / 2 |
| dropout | 0.05 | 0.05 | 0.10 |
| patch size / stride | 16 / 16 | 16 / 16 | 16 / 16 |
| context window | 8192 | 8192 | 8192 |
| output quantiles | 21 | 21 | 21 |
| **trainable params** | **33.21 M** | **888.24 M** | **1.33 B** |

Shared by all variants: RMSNorm, arcsinh normalisation, RoPE + xPOS positional
encoding, coordinate embedding, and a 21-quantile prediction head.

## ⚡ Quick start

### 1. Install

Python 3.10–3.12. Install PyTorch for your platform first (CUDA or ROCm), then:

```bash
pip install git+https://github.com/Hexian-2001/DeepWind.git
```

Or, for a development checkout:

```bash
git clone https://github.com/Hexian-2001/DeepWind.git && cd DeepWind
pip install -e ".[dev]"
```

### 2. Make some (fake) data

DeepWind reads a **2-D `.npy` array of shape `(channels, time)`** plus a small
CSV that says where the site is and which variable each row is. You don't have a
wind farm lying around? Generate one — it takes three lines of NumPy:

```python
import numpy as np
import pandas as pd

T = 8640                                   # 30 days at 5-min resolution
t = np.arange(T) / 288.0                   # fraction of a day

# channel 0 = power (MW). The rest are weather covariates (see the table below).
wind_speed = np.clip(8 + 4*np.sin(2*np.pi*t) + 1.5*np.cos(2*np.pi*t/3.5)
                     + 0.5*np.random.randn(T), 0, 25)
power = 100.0 * np.clip((wind_speed - 3) / 10.0, 0, 1)      # toy power curve
power = np.clip(power + 2*np.random.randn(T), 0, 100.0)

wind_dir    = (180 + 60*np.sin(2*np.pi*t/2) + 5*np.random.randn(T)) % 360
temperature = 15 + 8*np.sin(2*np.pi*t - 2) + 1.5*np.random.randn(T)
pressure    = 1013 + 5*np.sin(2*np.pi*t/3) + 0.5*np.random.randn(T)
density     = 1.225 * (273.15 / (273.15 + temperature)) * (pressure / 1013)

data = np.stack([power, wind_speed, wind_dir, temperature, pressure, density])
np.save("my_site.npy", data.astype(np.float32))             # (6, T)

pd.DataFrame([{
    "filename":     "my_site.npy",
    "longitude":    -97.5,
    "latitude":     35.2,
    "variate_ids":  "0,1,2,3,4,5",   # which global variable each row is
    "dataset":      "my_site",
}]).to_csv("my_metadata.csv", index=False)
```

### 3. Forecast

```bash
python infer.py \
  model=deepwind_base \
  inference.checkpoint_path=Hexian-2001/DeepWind1.0-890M \
  data.npy_path=my_site.npy \
  data.metadata_path=my_metadata.csv \
  output.output_path=my_forecast.npz
```

That writes `my_forecast.npz` with:

| key | shape | meaning |
|---|---|---|
| `point_preds` | `(6, 96)` | median forecast per channel |
| `quantile_preds` | `(6, 96, 21)` | all 21 quantiles per channel & step |
| `quantiles` | `(21,)` | the probability levels |
| `context` | `(6, 8192)` | the context window that was used |
| `channel_mask` | `(6,)` | 1 = real channel, 0 = padded |

Channel `0` is power — `quantile_preds[0]` is your power forecast. Read it back:

```python
out = np.load("my_forecast.npz")
p10, p50, p90 = out["quantile_preds"][0, :, [2, 10, 18]]   # power, 10/50/90th
```

> ⚠️ First call downloads the checkpoint from the Hub (~3.5 GB for base). Want a
> quick CPU sanity check? Add `data.context_length=512` to shrink the context.

## 🔮 Zero-shot inference on your own data

The whole game is getting your data into DeepWind's input shape. Here's the
contract.

### Input: one `.npy` per site

- A 2-D array of **`float32`**, shape **`(C, T)`** — `C` channels, `T` time
  steps, row-major by time.
- **Row 0 must be power.** It's the channel the model is scored on; everything
  else is optional context.
- Rows 1–5 are weather covariates, mapped to a fixed global ontology:

| channel | variate id | variable |
|---|---|---|
| 0 | `0` | **power** (normalised to capacity, or raw — see below) |
| 1 | `1` | hub-height wind speed |
| 2 | `2` | wind direction |
| 3 | `3` | temperature |
| 4 | `4` | air pressure |
| 5 | `5` | air density |

- You may have **fewer** than 6 channels. A SCADA turbine that only logs power +
  wind speed is a `(2, T)` array — the loader pads the rest to 6 and masks them
  out. A site with power only is a `(1, T)` array.
- The model reads the **last `context_length` (= 8192) steps** as context and
  predicts the next `prediction_length` (= 96) steps.

### Input: one `metadata.csv`

Columns, in order of importance:

| column | required? | meaning |
|---|---|---|
| `filename` | ✅ | the `.npy` filename, exactly as on disk |
| `variate_ids` | 👍 recommended | comma-separated global ids, e.g. `"0,1,2"`, one per channel (in row order) |
| `longitude`, `latitude` | optional | site coordinates in degrees |
| `dataset` | optional | free-form tag (used for balanced pre-training sampling) |

**With coordinates** — DeepWind converts lat/lon to a 3-D Cartesian vector and
adds a learned spatial embedding, so the model "knows" where the site is.

**Without coordinates** — just leave `longitude`/`latitude` empty. DeepWind
falls back to a shared learned embedding for "unknown site". Zero-shot still
works; it's just one less signal.

**Missing `variate_ids`** — if you don't tell it which variable each row is, the
loader fills every slot with the padding id, and the model treats all channels
as unlabelled. Fine in a pinch, but **providing `variate_ids` is the single
biggest accuracy lever** for zero-shot — it's how the model knows "row 1 is wind
speed, not temperature."

> 💡 **Normalisation is free.** DeepWind applies arcsinh + instance normalisation
> *inside* the model, computed from the context window. Feed raw values; don't
> pre-scale.

## 🎯 Fine-tuning

When you *do* have target-site data, a few epochs of LoRA adapt DeepWind to it.
The recipe keeps the base frozen except for a low-rank update on the
**variate-attention QKV/output projections** (`r=16`, `α=32`), while the MoE
**router** and the **prediction head** fine-tune in full — parameter-efficient
and cheap.

```bash
python finetune.py \
  model=deepwind_base \
  finetune.pretrained_path=Hexian-2001/DeepWind1.0-890M \
  data.npy_path=/path/to/site.npy \
  data.metadata_path=/path/to/metadata.csv \
  run_name=my-site
```

| knob | default | what it does |
|---|---|---|
| `finetune.lora_r` / `finetune.lora_alpha` | 16 / 32 | LoRA rank & scale |
| `training.num_train_epochs` | 30 | epochs over the site's windows |
| `training.per_device_train_batch_size` | 64 | batch size |
| `data.stride` | 16 | step between context windows |
| `data.finetune_rate` | 1.0 | keep a fraction of windows (few-shot) |

The result is a LoRA adapter directory. Use it at inference by adding
`inference.adapter_path=/path/to/adapter` — the pipeline merges it into the
base weights and unloads the scaffolding, so the serving path is identical.

## 🏋️ Pretraining

To train your own DeepWind from scratch, point the data config at a corpus of
`.npy` files and a metadata CSV:

```text
$DEEPWIND_DATA_ROOT/
├── train/               # one (channels, time) .npy per series
├── eval/                # held-out series
├── train_metadata.csv
└── eval_metadata.csv
```

```bash
export DEEPWIND_DATA_ROOT=/path/to/deepwind-data
export DEEPWIND_RUNS_ROOT=/path/to/deepwind-runs

python train.py \
  model=deepwind_base \
  training=deepwind_base \
  run_name=my-run
```

Model variants live in `configs/model/` (`deepwind_small`, `deepwind_base`,
`deepwind_large`); training schedules in `configs/training/`. The paper
protocol — global batch 256, 100,000 steps, peak LR `1e-4`, 3% warmup, cosine
decay, BF16 — is encoded in those files and needs no flags.

Corpus design matters more than the model: DeepWind was trained with **0.7
synthetic (WIND Toolkit) / 0.3 real (SCADA)** sampling weights, and each site's
channels are labelled with `variate_ids` so the variate embedding can transfer
across sources. See `configs/data/train.yaml` for the knobs (balanced sampling,
adaptive windowing, and the Phase-1 sim-to-real augmentation).

On Pawsey Setonix, the job templates under `scripts/setonix/` are
account-portable — they resolve run/data/venv/project roots from `$MYSCRATCH`
and `$MYSOFTWARE` (overridable with the `DEEPWIND_*` variables). Set your Slurm
account and create `slurm-logs/` before a direct `sbatch`.

## 📊 Results

Zero-shot evaluation on **WindBench** (8 datasets × 6 horizons, 1–12 h),
macro-averaged. Lower is better for ↓, higher for ↑.

**DeepWind1.0-890M (base)** — the recommended default:

| Metric | Value |
|---|---|
| nCRPS ↓ | 0.0961 |
| nMAE ↓ | 0.1211 |
| MAE_Coverage | 0.1229 |
| Accuracy ↑ | 0.8097 |
| Qualified_Rate ↑ | 0.8001 |
| R² ↑ | 0.6430 |
| mean_wQuantileLoss ↓ | 0.2667 |

**DeepWind1.0-33M (small)** — 0.0998 nCRPS (see its [model card](https://huggingface.co/Hexian-2001/DeepWind1.0-33M) for the full table). **DeepWind1.0-1.3B (large)** — 0.0671 nCRPS under the paper's original protocol.

## 📁 Repository layout

```text
configs/        Hydra model, data, training, and inference configurations
src/            Model, datasets, training, inference, and evaluation
scripts/        Local and Pawsey/Setonix launch templates
test/           Unit and integration tests
model_cards/    Per-checkpoint model cards (also published on the Hub)
```

Large datasets, checkpoints, logs, and experiment outputs are deliberately not
stored in Git.

## 📜 License & citation

Code is Apache-2.0; model weights are Apache-2.0. Dataset licenses are handled
separately — the proprietary Shanxi Wind source is not redistributed. If this
helps your work, cite:

```bibtex
@article{wang2026deepwind,
  title     = {DeepWind: A foundation model for zero-shot wind power forecasting},
  author    = {Wang, Hexian and Zhou, Tongming and Jia, Chengzhen and Liu, Yushan and Wang, Lingmei},
  journal   = {Energy},
  volume    = {360},
  pages     = {141793},
  year      = {2026},
  doi       = {10.1016/j.energy.2026.141793}
}
```
