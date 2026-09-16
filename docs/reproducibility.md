# Reproducibility record

## Project map — where everything lives

One-page overview of the DeepWind codebase and its on-disk layout on Setonix.
All paths are absolute and current as of the 2026-09 retrain campaign.

### Repo, environment & data

| Item | Path |
|---|---|
| Code repo (branch `release/open-source-v1`) | `/software/projects/pawsey0115/hwang4/research_projects/DeepWind` |
| Dedicated venv (Python 3.11.16) | `/scratch/pawsey0115/hwang4/conda_envs/deepwind` |
| venv interpreter | `/scratch/pawsey0115/hwang4/conda_envs/deepwind/bin/python` |
| Data root | `/scratch/pawsey0115/hwang4/deepwindData` |
| Runs root (`DEEPWIND_RUNS_ROOT`) | `/scratch/pawsey0115/hwang4/projects/deepwind/runs` |

```
deepwindData/
  train/            # 127,297 pretraining series (.npy)
  eval/             # held-out eval series
  test/             # 8 WindBench benchmark datasets (.npy)
  train_metadata.csv / eval_metadata.csv / test_metadata.csv
```

### Entry-point scripts

| Task | Script | Config | Setonix launcher |
|---|---|---|---|
| Pretrain | `train.py` | `configs/train.yaml` + `configs/training/deepwind_{small,base,large}.yaml` | `scripts/setonix/train_paper.sbatch` (`MODEL=small\|base\|large`) |
| Finetune | `finetune.py` | `configs/finetune.yaml` + `configs/training/finetune.yaml` | — (run directly) |
| Inference | `infer.py` | `configs/infer.yaml` | — (run directly) |
| Evaluate | `evaluate.py` | `configs/eval.yaml` | `scripts/setonix/eval_paper.sbatch` |

### Source (`src/`)

| Package | Contents |
|---|---|
| `src/models/` | `deepwind.py`, `backbone.py`, `configuration.py`, `adapter.py`, `layers.py` |
| `src/data/` | datasets & dataloaders |
| `src/evaluation/` | eval protocol (`protocol.py`), metrics & harness |
| `src/inference/` | inference helpers |
| `src/losses/` | losses (CRPS / quantile) |
| `src/layers/` | Transformer building blocks |
| `src/utils/` | `provenance.py`, `trainer.py`, `registry.py`, `metrics.py`, `distributed.py`, `cache.py`, `constants.py`, `vis.py` |

### Configs (`configs/`)

| Dir | Contents |
|---|---|
| `configs/model/` | architecture: `deepwind_{small,base,large,debug}.yaml` |
| `configs/training/` | `deepwind_{small,base,large}.yaml`, `finetune.yaml` |
| `configs/data/` | `train.yaml`, `finetune.yaml`, `wind_test.yaml` |
| `configs/data_eval/` | per-dataset eval configs |
| top-level | `train.yaml`, `eval.yaml`, `finetune.yaml`, `infer.yaml` |

### Model weights (checkpoints)

Each training run lives under `runs/DeepWind-Research/<run_name>/` and holds the
weights in `checkpoints/`:

```
DeepWind-Research/<run_name>/
  checkpoints/
    checkpoint-100000/    # HF-format checkpoint dir
    model.safetensors     # consolidated weights
    config.json           # serialized DeepWindConfig
    trainer_state.json    # global_step + loss history
    run_info.json         # resolved Hydra config + git/provenance snapshot
  logs/                   # train.log
  metadata/               # git_commit.txt, pip_freeze.txt, command.txt
  wandb/                  # offline wandb artifacts
```

Example (Small):
`runs/DeepWind-Research/deepwind-small-paper-seed42-20260914-164253/checkpoints/checkpoint-100000/`

### Experiment records

| Record | Location |
|---|---|
| Evaluation outputs | `runs/results/deepwind/<eval_name>/` (per-dataset dirs + `_aggregate.json` + `run_info.json`) |
| Leaderboard (git-committed source of truth) | `results/leaderboard.jsonl` |
| Pipeline state (cron driver) | `runs/pipeline_state.json` |
| wandb (online curves) | `https://wandb.ai/hexian-2001-shanxi-university/DeepWind-Research` |

### Tooling (`tools/`) & docs

| Script | Purpose |
|---|---|
| `aggregate_eval.py` | macro-average an eval into `_aggregate.json` |
| `register_eval.py` | register one eval into the leaderboard |
| `compare_models.py` | rank / compare / CSV export |
| `export_report.py` | paper tables (latex / markdown / csv) |
| `plot_results.py` | publication figures (ranking / macro / per-dataset / per-horizon) |
| `plot_forecasts.py` | per-sample forecast plots (history + point + 50%/90% PI) |
| `plot_rolling_forecasts.py` | stitched rolling forecast (backtest, fixed-length, PI bands) |
| `run_seed_sweep.py` | submit multi-seed training |
| `audit_data_splits.py` / `audit_dataset.py` / `clean_data_manifests.py` | data hygiene |

Paper figure/table scripts live in `reproduction/`; longer docs in `docs/`
(`architecture-audit.md`, `asset-inventory.md`, `data.md`, `pawsey.md`,
`refactor-roadmap.md`).

## Published specification

DeepWind-Large is described as an 18-layer, 1024-dimensional decoder-only
Transformer with 16 heads, SwiGLU width 2816, 8 experts, Top-2 routing, patch
size 16, context length 8192, and 21 quantiles. The paper reports AdamW,
`beta1=0.9`, `beta2=0.95`, weight decay 0.01, peak learning rate `1e-4`, 3,000
warmup steps, cosine decay, gradient clipping at 1.0, BF16, global batch 256,
and 100,000 steps.

## Recovered final run

> Historical note (pre-retrain): this describes the legacy `deepwind_large_v5`
> checkpoint as recovered before the 2026-09 retrain campaign. The retrain
> (Small/Base/Large to paper spec) uses `use_rotary_emb=true` and
> `aux_loss_weight=0.02`, matching the paper. See the leaderboard (below) for
> the authoritative per-model configuration.

The final `deepwind_large_v5` run reached step 100,000. Its saved configuration
and Slurm log agree on the main dimensions, 8 experts, Top-2 routing, 21-quantile
head, learning rate, global batch, warmup ratio, and step count. The run used
16 distributed accelerator devices (two Setonix nodes, eight visible devices
per node), per-device batch 4, and gradient accumulation 4.

Two material differences are preserved rather than rewritten:

| Setting | Paper | Recovered checkpoint |
|---|---:|---:|
| Rotary/xPOS attention | enabled/described | `use_rotary_emb=false` |
| MoE auxiliary weight | `0.02` | `0.01` |

The current development YAML files also contained later experimental settings
(Student-t heads, 4 experts for Base/Large, and 0.9/0.1 source weighting). They
must not be presented as the paper checkpoint configuration.

## Provenance requirements for every new run

Every run directory must retain:

- resolved Hydra config and override list;
- Git commit and dirty-worktree status;
- Slurm job ID, node list, container digest, Python/package versions;
- random seed, world size, effective global batch, dataset manifest checksum;
- W&B run ID or an offline export;
- train/eval metrics, trainer state, and checkpoint checksums.

No result is publication-ready until a fresh-process load test and at least one
fixed-sample inference regression pass.

---

# Model registry & leaderboard

The sections below describe how a completed train → eval run is registered into
a **git-committed JSONL leaderboard** (`results/leaderboard.jsonl`), and how to
compare, export paper tables, and run seed sweeps.

## 0. Step-by-step quick start

Whenever you have a new model (new architecture / hyperparameters / seed / a
finished training run), repeat these 5 steps. All commands run from the repo root.

> **Python — required setup (read first).** The base conda env is broken: its
> Python 3.14 crashes at startup (`init_fs_encoding` codec error), and
> `conda activate` is unusable for the same reason (conda itself runs on that
> Python). Always use the DeepWind venv Python. Run this once per shell, then the
> `$PY` commands below work as written:
> ```bash
> export PY=/scratch/pawsey0115/hwang4/conda_envs/deepwind/bin/python
> ```
> Equivalent alternative (no `$PY` needed): prepend it to PATH so plain `python`
> resolves to it — `export PATH=/scratch/pawsey0115/hwang4/conda_envs/deepwind/bin:$PATH`.
> The leaderboard tools are pure stdlib, but they still need a *working* Python;
> the venv provides Python 3.11.16.

**Step 1 — Evaluate** (skip if already evaluated; `CKPT` points at the training
run's `checkpoints` directory)
```bash
CKPT=/scratch/pawsey0115/hwang4/projects/deepwind/runs/DeepWind-Research/deepwind-small-paper-seed42/checkpoints
sbatch --export=ALL,MODEL=small,CKPT=${CKPT},EVAL_NAME=eval-small-paper scripts/setonix/eval_paper.sbatch
# after the eval job reaches COMPLETED in squeue, aggregate:
"$PY" tools/aggregate_eval.py \
  /scratch/pawsey0115/hwang4/projects/deepwind/runs/results/deepwind/eval-small-paper
```

**Step 2 — Register** (one row = one model)
```bash
"$PY" tools/register_eval.py \
  /scratch/pawsey0115/hwang4/projects/deepwind/runs/results/deepwind/eval-small-paper \
  --model-id deepwind-small-paper-seed42 --variant small --tags paper-spec,baseline,seed42
```

**Step 3 — Quick ranking** (which model looks best, at a glance)
```bash
"$PY" tools/compare_models.py --summary                    # all models, sorted by nCRPS ascending
"$PY" tools/compare_models.py --summary --variant small,base,large
"$PY" tools/compare_models.py --summary --csv ranking.csv  # export as CSV (open in Excel)
```

**Step 4 — Detailed comparison** (per-dataset / per-horizon, to locate the gap)
```bash
"$PY" tools/compare_models.py --variant small,base,large
"$PY" tools/compare_models.py --family <config_hash> --group-family   # same design, multiple seeds -> mean±std
```

**Step 5 — Export paper table**
```bash
"$PY" tools/export_report.py --format latex --out results_table.tex   # or markdown / csv
```

**Quick reference** (remember `export PY=...` first)

| Want to ... | Command |
|---|---|
| Inspect raw records (JSONL) | `cat results/leaderboard.jsonl` |
| Quick ranking | `"$PY" tools/compare_models.py --summary` |
| Ranking as CSV | `"$PY" tools/compare_models.py --summary --csv ranking.csv` |
| Detailed comparison | `"$PY" tools/compare_models.py --variant small,base,large` |
| Paper table | `"$PY" tools/export_report.py --format latex` |
| Plot figures | `"$PY" tools/plot_results.py` |
| Seed variance (mean±std) | `"$PY" tools/compare_models.py --family <hash> --group-family` |
| Multi-seed training in one shot | `"$PY" tools/run_seed_sweep.py --model small --seeds 42,43,44` |

## 1. Concept

- **Source of truth = `results/leaderboard.jsonl`** (append-only, one evaluated
  model per line, committed to git — reviewable, diffable, traceable). wandb and
  `run_info.json` are its upstream inputs.
- **One line = one model**: architecture + training hyperparameters + seed + git
  commit + all metrics (macro / per-dataset / per-horizon).
- **config_hash grouping**: the same design (architecture + training + data)
  with different seeds hashes identically → a seed sweep automatically groups as
  one family, ready for mean±std.

## 2. Pipeline

```
train (train_paper.sbatch)
   └─> evaluate (eval_paper.sbatch + tools/aggregate_eval.py)
          └─> register (tools/register_eval.py)          # writes results/leaderboard.jsonl
                 ├─> compare (tools/compare_models.py)
                 └─> paper table (tools/export_report.py)
```

### 2.1 Train

```bash
MODEL=small sbatch --export=ALL,MODEL=small scripts/setonix/train_paper.sbatch
```

The run name is fixed to `deepwind-small-paper-seed42` (a `--requeue` reuses the
same `output_dir`, so training auto-resumes across the 24h wall-clock limit).

### 2.2 Evaluate

```bash
CKPT=/scratch/pawsey0115/hwang4/projects/deepwind/runs/DeepWind-Research/deepwind-small-paper-seed42/checkpoints
sbatch --export=ALL,MODEL=small,CKPT=... scripts/setonix/eval_paper.sbatch
# after completion:
"$PY" tools/aggregate_eval.py <eval_dir>
```

### 2.3 Data protocol (unified target grid)

Every model — DeepWind and every baseline — is scored on the **same test
targets**, produced by `src/evaluation/protocol.py`:

- `test_target_starts(T, pred_len, train_ratio=0.7, val_ratio=0.1)` returns the
  non-overlapping target start indices over the test slice, anchored at
  `t_val = int(T * (train_ratio + val_ratio))` (= `0.8T`).
- `make_windows(series, target_starts, ctx, pred_len)` builds each (input,
  target) pair as `X = series[s-ctx:s]`, `y = series[s:s+pred_len]`.

The targets are therefore byte-identical across models; only `ctx` (the lookback
context length) is a per-model choice.

**Context length.** DeepWind's default `context_length` is 8192. On the two short
series — `gefc12_wind_7` (T = 18,756) and `gefc14_wind_10` (T = 16,800) — a full
8192-step context reaches back roughly half the series and disadvantages DeepWind
against the 1024-step baselines, so these two use `context_length = 1024` via
`configs/eval.yaml` `data.context_length_override` (resolved per dataset in
`evaluate.py::_build_dataloader`). Every other dataset keeps 8192.

The baseline models (statistical / gradient-boosted / deep-learning / TSFMs) run
on the identical protocol and windowing — see `docs/baselines.md` for the harness
and the full ranked comparison table.

### 2.4 Register

```bash
"$PY" tools/register_eval.py \
  /scratch/pawsey0115/hwang4/projects/deepwind/runs/results/deepwind/eval-small-paper \
  --model-id deepwind-small-paper-seed42 --variant small --tags paper-spec,baseline,seed42
```

`register_eval.py` automatically reads:

| Source | Fields |
|------|-----------|
| `<eval_dir>/run_info.json` | `checkpoint_path`, eval-time git commit, `run_name` |
| `checkpoints/run_info.json` (training snapshot) | `config.model` (architecture), `config.training`, `config.seed`, `config.data.seed`, training git commit |
| `<eval_dir>/_aggregate.json` | macro means + per-horizon nCRPS |
| `<eval_dir>/*/H_*.json` | per-dataset metrics |
| `checkpoint-*/trainer_state.json` | `global_step` + final loss |

## 3. Leaderboard schema

Each line of `results/leaderboard.jsonl` is one JSON object:

```json
{
  "model_id": "deepwind-small-paper-seed42",
  "variant": "small",
  "checkpoint": "…/checkpoints/checkpoint-100000",
  "eval_name": "eval-small-paper",
  "eval_dir": "…/results/deepwind/eval-small-paper",
  "git_commit": "3a0ef04…",
  "training_git_commit": "5e167d3…",
  "created_at": "2026-09-15T…Z",
  "config_hash": "sha256…",
  "architecture": { "d_model": 384, "num_layers": 6, "…": "…" },
  "training":     { "learning_rate": 1e-4, "per_device_train_batch_size": 8, "…": "…" },
  "seed": 42,
  "data_seed": null,
  "metrics": {
    "mean": { "nCRPS": 0.1097, "nMAE": 0.1453, "Accuracy": 0.7838,
              "Qualified_Rate": 0.7403, "MAE_Coverage": 0.1082, "R2": 0.5453,
              "mean_wQuantileLoss": 0.3064 },
    "per_dataset": { "30651": { "nCRPS": 0.1083, "…": "…" }, "…": "…" },
    "per_horizon_nCRPS": { "nCRPS_H1": 0.0567, "…": "…" },
    "n_cells": 48
  },
  "training_outcome": { "global_step": 100000, "final_loss": 0.0458 },
  "tags": ["paper-spec", "baseline", "seed42"]
}
```

Metric directions (used by `compare_models.py` / `export_report.py` to mark the
winner automatically):

- **Lower is better**: `nCRPS`, `nMAE`, `MAE_Coverage`, `mean_wQuantileLoss`
- **Higher is better**: `Accuracy`, `Qualified_Rate`, `R2`

## 4. Compare

```bash
"$PY" tools/compare_models.py --summary                       # quick ranking (one line per model, nCRPS ascending)
"$PY" tools/compare_models.py --summary --csv ranking.csv     # ranking as CSV
"$PY" tools/compare_models.py --all
"$PY" tools/compare_models.py --variant small,base,large
"$PY" tools/compare_models.py deepwind-small-paper-seed42 deepwind-base-paper-seed42
"$PY" tools/compare_models.py --family <config_hash> --group-family   # same design, multiple seeds -> mean±std
```

`--summary` prints a compact ranking table; the default output prints the macro
metric table + per-dataset nCRPS/nMAE tables + per-horizon nCRPS, marking each
row's best value with ` <--`.

## 5. Export paper table

```bash
"$PY" tools/export_report.py --format latex    --out results_table.tex
"$PY" tools/export_report.py --format markdown --variant small,base,large
"$PY" tools/export_report.py --format csv      --all --out results_table.csv
```

The LaTeX output is a booktabs table (`\toprule`/`\midrule`/`\bottomrule`) with
the best value bolded via `\textbf{}`.

## 6. Visualize (figures)

```bash
"$PY" tools/plot_results.py                            # all models -> reports/figures/
"$PY" tools/plot_results.py --variant small,base,large
"$PY" tools/plot_results.py --formats png,svg --dpi 200
```

`plot_results.py` renders publication-quality figures from the leaderboard into
`reports/figures/`:

| File | Content |
|---|---|
| `ranking_ncrps.png` | mean nCRPS per model, sorted (headline "which is best") |
| `macro_metrics.png` | one panel per HEADLINE metric, best value outlined |
| `per_dataset_ncrps.png` | nCRPS grouped by the 8 benchmark datasets |
| `per_horizon_ncrps.png` | nCRPS vs horizon (H1–H12), one line per model |
| `seed_sweep_<variant>.png` | per-horizon lines per seed (only when >1 seed shares a config_hash) |

Colors are fixed per variant (small=blue, base=orange, large=green). Metrics that
are "lower is better" are marked `↓`, "higher is better" `↑`. Requires matplotlib
(installed in the DeepWind venv; the `reproduction/` figures use it too).

The figures are **regenerated views** over `results/leaderboard.jsonl` — the
JSONL stays the committed source of truth, and `reports/` is git-ignored
(regenerable on demand).

### 6.1 Forecast visualisations (prediction intervals)

```bash
"$PY" tools/plot_forecasts.py --model-id deepwind-small-paper-seed42
"$PY" tools/plot_forecasts.py --model-id deepwind-base-paper-seed42 --horizons H1,H6
```

For each (dataset, horizon) this renders one figure with a few stacked forecast
windows — history, ground truth, point (median) forecast, and the 50% / 90%
prediction intervals — straight from the eval raw results
(`<eval_dir>/raw_results/<dataset>/raw_H<h>.npz`, physical MW). It reuses the
codebase's own `src/utils/vis.py::visualize_forecasts`.

**Output layout** (git-ignored, regenerable):

```
reports/forecasts/<dataset>/H<h>__<model_id>.png
```

**Cross-model consistency:** the exact sample windows are pinned in
`results/forecast_samples.json` (git-committed). Every model plots the *same*
windows, so `H1__deepwind-small-…png` vs `H1__deepwind-base-…png` are directly
comparable. The manifest is auto-created on first run (deterministic, `seed=42`)
and reused verbatim thereafter — re-run with the same `--horizons` and the
selection never changes.

| Flag | Default | Meaning |
|---|---|---|
| `--horizons` | `H1,H6` | which horizons to plot per dataset |
| `--num-plots` | `3` | sample windows per (dataset, horizon) |
| `--plot-hist-len` | `144` | trailing history steps shown |
| `--datasets` | all 8 | subset of benchmark datasets |
| `--seed` | `42` | seed for the initial sample selection |

### 6.2 Rolling (stitched) forecast visualisations

```bash
"$PY" tools/plot_rolling_forecasts.py --model-id deepwind-small-paper-seed42
"$PY" tools/plot_rolling_forecasts.py --model-id deepwind-base-paper-seed42 --steps 128
"$PY" tools/plot_rolling_forecasts.py --model-id deepwind-base-paper-seed42 --steps 256 --num-plots 3
```

A single window is hard to judge in isolation (for 60-min data H1 is *one* step), so
this tool stitches consecutive windows into one contiguous rolling forecast — a
**rolling-origin backtest**. It reads the same eval raw results and is valid because
`DeepWindTestDataset` slides its context by exactly `pred_len` (stride defaults to
`pred_len`) and the evaluator re-sorts samples by index: consecutive windows are
contiguous, so `targets.reshape(-1)` reproduces an unbroken slice of the test series.

Note the semantics: every window is still an independent forecast conditioned on its
own *true* 8192-step history (`mqd_infer=true`), so the figure is a backtest view —
not a single autonomous multi-step rollout.

**Output layout** (git-ignored, regenerable):

```
reports/forecasts_rolling/<dataset>/H<h>__<model_id>__T<steps>.png
```

**Cross-model consistency:** the stitched segment starts are pinned per (dataset,
horizon) in `results/forecast_samples.json` (`rolling.starts` — a list of
`--num-plots` start windows, deterministic `seed=42`) and reused verbatim, so every
model plots the *same* contiguous blocks.

| Flag | Default | Meaning |
|---|---|---|
| `--steps` | `128` | fixed number of steps per panel (independent of dataset resolution) |
| `--num-plots` | `3` | stacked segments (subplots) per (dataset, horizon) |
| `--horizons` | `H1,H6` | which horizons to plot per dataset |
| `--datasets` | all 8 | subset of benchmark datasets |
| `--seed` | `42` | seed for the deterministic start selection |

## 7. Seed sweep

```bash
"$PY" tools/run_seed_sweep.py --model small --seeds 42,43,44 --dry-run
"$PY" tools/run_seed_sweep.py --model base  --seeds 42,43,44
```

Each seed submits an independent training run (`DEEPWIND_RUN_NAME=deepwind-<model>-paper-seed<s>`,
`DEEPWIND_TRAIN_OVERRIDES=seed=<s>` — a Hydra top-level `seed` override that
`train.py` applies via `set_seed(cfg.seed)` to re-seed model init / data order /
RNG).

After each seed finishes training → evaluate → `register_eval.py`. Because
`config_hash` excludes the seed, the seeds group automatically, and
`compare_models.py --group-family` gives mean±std directly.

> Known limitation: `train_paper.sbatch` hardcodes `wandb.tags=[...,seed42]`, so a
> seed sweep's wandb tag still reads `seed42`; the true seed is the leaderboard's
> `seed` field and `wandb.notes`/`run_name`.

## 8. config_hash semantics

```
config_hash = sha256( canonical_json({ model, training', data', data_eval }) )
```

From `training` we strip the **run identity fields** (`run_name` / `output_dir` /
`report_to` / `overwrite_output_dir`) and the **reporting/checkpoint knobs**
(`logging_*`, `save_*`, `eval_*`, `load_best_model_at_end`,
`ddp_find_unused_parameters`, `prediction_loss_only`, etc.), and we strip `seed`
from `data`. **The seed never participates in the hash** → the same design with
different seeds yields the same `config_hash` and groups as one family.

## 9. Reuse & boundaries

- These tools **only read** `run_info.json`; they do not re-collect provenance
  (`src/utils/provenance.py` already does that).
- The existing paper figure/table scripts in `reproduction/` (reliability /
  coverage / efficiency) keep reading `<eval>/<dataset>/H_*.json`; this system is
  complementary, not a replacement — it is their upstream (registry / comparison
  / paper table).
- The leaderboard stores **absolute /scratch paths** (honest and directly
  reproducible), redirectable via `DEEPWIND_RUNS_ROOT`.
