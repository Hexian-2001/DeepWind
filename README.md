# DeepWind

DeepWind is a domain foundation model for zero-shot probabilistic wind-power
forecasting. It uses time-aware patching, decoupled time/variate attention,
sparse mixture-of-experts layers, and autoregressive multi-quantile decoding.

This repository is the reproducibility implementation for:

> H. Wang et al., "DeepWind: A foundation model for zero-shot wind power
> forecasting," *Energy*, vol. 360, 141793, 2026.
> https://doi.org/10.1016/j.energy.2026.141793

## Release status

The `release/open-source-v1` branch is being prepared from the exact training
and evaluation code used for the paper. Model logic is frozen while packaging,
tests, manifests, portable paths, and documentation are added. Public model
weights will contain inference weights and their exact saved `config.json`, not
optimizer states or private infrastructure paths.

## Repository layout

```text
configs/        Hydra model, data, training, and reproduction configurations
src/            DeepWind model, datasets, training, inference, and evaluation
scripts/        Local and Pawsey/Setonix launch scripts
test/           Unit, integration, and data-pipeline checks
reproduction/   Scripts used to reproduce paper analyses and figures
tools/          Result collection and operational utilities
docs/           Data, reproducibility, release, and architecture notes
```

Large datasets, checkpoints, logs, and experiment outputs are deliberately not
stored in Git.

## Installation

Python 3.10-3.12 is supported. Install PyTorch for your CUDA or ROCm platform
first, then install DeepWind:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Pawsey Setonix, use the centre-provided PyTorch ROCm container and the job
templates under `scripts/setonix/`; see [docs/pawsey.md](docs/pawsey.md).

## Data configuration

Set portable roots instead of editing Python code:

```bash
export DEEPWIND_DATA_ROOT=/path/to/deepwind-data
export DEEPWIND_RUNS_ROOT=/path/to/deepwind-runs
```

The expected arrays, metadata schema, preprocessing rules, split isolation,
and redistribution restrictions are documented in [docs/data.md](docs/data.md).
The proprietary Shanxi Wind dataset cannot be redistributed.

## Training and evaluation

The paper checkpoint was trained for 100,000 steps with a global batch size of
256. Its recovered resolved configuration is preserved under
`configs/reproduction/paper_large_actual.yaml`.

```bash
python train.py \
  model=deepwind_large \
  training=deepwind_large \
  run_name=my-run \
  model.pred_head_type=quantile

python evaluate.py \
  inference.checkpoint_path=/path/to/checkpoint \
  run_name=my-evaluation
```

Run a Setonix smoke test before a full job:

```bash
sbatch scripts/setonix/smoke.sbatch
```

## Reproducibility notes

The recovered final checkpoint configuration differs from two statements in
the published method description: it records `use_rotary_emb=false` and
`aux_loss_weight=0.01`. The paper describes RoPE/xPOS and reports 0.02. Both
the published specification and the actual checkpoint configuration are kept
explicitly; see [docs/reproducibility.md](docs/reproducibility.md).

## License and citation

Code is released under the Apache License 2.0. Dataset licenses and model-weight
terms must be handled separately. Please cite the paper using `CITATION.cff`.
