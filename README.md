<div align="center">

<img src="assets/deepwind-logo.png" alt="DeepWind logo" width="150">

# DeepWind: A Foundation Model for Zero-Shot Wind Power Forecasting

**Domain foundation model for probabilistic wind power forecasting**

[![HF model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-DeepWind1.0--890M-FFD21E)](https://huggingface.co/Hexian-2001/DeepWind1.0-890M)
[![license](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

</div>

DeepWind is a domain foundation model for zero-shot probabilistic wind-power
forecasting. It uses time-aware patching, decoupled time/variate attention,
sparse mixture-of-experts layers, and autoregressive multi-quantile decoding.

This repository is the reference implementation for:

> H. Wang et al., "DeepWind: A foundation model for zero-shot wind power
> forecasting," *Energy*, vol. 360, 141793, 2026.
> https://doi.org/10.1016/j.energy.2026.141793

## Models

| Model | Params | Checkpoint |
|---|---|---|
| DeepWind1.0-890M (base) | 888.24 M | [🤗 `Hexian-2001/DeepWind1.0-890M`](https://huggingface.co/Hexian-2001/DeepWind1.0-890M) |

## Quick start

Install the code and load the released base model:

```bash
pip install git+https://github.com/Hexian-2001/DeepWind.git
```

```python
from src.models.deepwind import DeepWindModel

model = DeepWindModel.from_pretrained("Hexian-2001/DeepWind1.0-890M")
```

## Installation

Python 3.10-3.12 is supported. Install PyTorch for your CUDA or ROCm platform
first, then install DeepWind:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Repository layout

```text
configs/        Hydra model, data, and training configurations
src/            DeepWind model, datasets, training, inference, and evaluation
scripts/        Launch scripts (including Pawsey/Setonix job templates)
test/           Unit and integration tests
model_cards/    Per-checkpoint model cards (also published on the Hub)
```

Large datasets, checkpoints, logs, and experiment outputs are deliberately not
stored in Git.

## Data preparation

Set portable roots so the code finds your data without editing source files:

```bash
export DEEPWIND_DATA_ROOT=/path/to/deepwind-data
export DEEPWIND_RUNS_ROOT=/path/to/deepwind-runs
```

The expected array layout and metadata schema are defined in
`src/data/datasets.py`.

## Inference

Produce a probabilistic forecast on your own data:

```bash
python infer.py \
  model=deepwind_base \
  inference.checkpoint_path=Hexian-2001/DeepWind1.0-890M \
  data.npy_path=/path/to/site.npy \
  data.metadata_path=/path/to/metadata.csv \
  output.output_path=./forecast.npz
```

See `configs/infer.yaml` for all options.

## Fine-tuning

Fine-tune the base model on your own data with LoRA:

```bash
python finetune.py \
  model=deepwind_base \
  finetune.pretrained_path=Hexian-2001/DeepWind1.0-890M \
  run_name=my-finetune
```

See `configs/finetune.yaml` for the LoRA and data options.

## Pretraining

Pretrain from scratch on your own corpus:

```bash
python train.py \
  model=deepwind_base \
  training=deepwind_base \
  run_name=my-run
```

Model variants live under `configs/model/` (`deepwind_small`, `deepwind_base`,
`deepwind_large`); training schedules live under `configs/training/`.

On Pawsey Setonix, use the centre-provided PyTorch ROCm container and the job
templates under `scripts/setonix/`. The templates are account-portable: they
resolve run/data/venv/project roots from `$MYSCRATCH` and `$MYSOFTWARE`
(override with the `DEEPWIND_*` variables above), and you must set your own
Slurm account (`--account=YOUR_PROJECT-gpu`) and create `slurm-logs/` in your
submit directory before a direct `sbatch`.

## Evaluation

Evaluate a checkpoint on your test data:

```bash
python evaluate.py \
  inference.checkpoint_path=Hexian-2001/DeepWind1.0-890M \
  run_name=my-evaluation
```

## License and citation

Code is released under the Apache License 2.0. Model weights are released under
Apache-2.0; dataset licenses must be handled separately. Please cite the paper
using `CITATION.cff`.
