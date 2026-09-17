# Pawsey Setonix workflow

**Contents**

- 1. Production training
- 2. Watching training with wandb

Keep Git repositories and environments under `/software/projects`, and keep
large datasets, checkpoints, logs, W&B caches, and temporary outputs under
`/scratch`.

Recommended environment variables:

```bash
export DEEPWIND_PROJECT_ROOT=/software/projects/<project>/<user>/research_projects/DeepWind
export DEEPWIND_DATA_ROOT=/scratch/<project>/<user>/deepwindData
export DEEPWIND_RUNS_ROOT=/scratch/<project>/<user>/deepwind-open/runs
export DEEPWIND_RESULTS_ROOT=/scratch/<project>/<user>/results/DeepWind-Research
export DEEPWIND_VENV=/scratch/<project>/<user>/conda_envs/deepwind
```

Use Pawsey's supported PyTorch ROCm module/container for multi-node jobs. A
virtual environment may be layered inside the container for Python-only
dependencies. Run `scripts/setonix/smoke.sbatch` before scaling to a production
allocation.

For each production launch, record `module list`, `$SINGULARITY_CONTAINER`,
`git rev-parse HEAD`, `git status --porcelain`, `pip freeze`, and Slurm metadata
inside the run directory.

## 1. Production training

The launcher streams metrics to Weights & Biases live — `WANDB_MODE=online`
is set inside the sbatch, so you can watch loss and checkpoints in the browser
as it runs (see “Watching training with wandb” below).

```bash
export DEEPWIND_RUN_NAME=large-checkpoint-replication-v1                # optional: override the default run name
sbatch --export=ALL,MODEL=large scripts/setonix/train_paper.sbatch      # gpu partition: 4 nodes, 24h, auto-requeue
```

When the `gpu` partition is drained, chain shorter `gpu-dev` chunks instead:

```bash
scripts/setonix/submit_train_chunks.sh large 8
```

## 2. Watching training with wandb

Every training run logs to the wandb project **`DeepWind-Research`**, using a
fixed run id equal to the run name (e.g. `deepwind-base-paper-seed42`).

- **Open the live run:** the training log prints `wandb run: <url>` — paste
  that URL, or browse to <https://wandb.ai> → the `DeepWind-Research` project
  (signed in to the account that owns the API key).
- **Chunked `gpu-dev` runs keep ONE wandb run, not a new one per chunk.**
  `train.py` calls `wandb.init(id=<run_name>, resume="allow")` with a *fixed*
  run name, so every chunk appends to the same run id and the step axis keeps
  growing across the 03:50 boundaries (3 chunks = 1 run).
- **The `epoch` number resets each chunk — that is expected, watch `step` not
  `epoch`.** Training is driven by `max_steps=100000` (not epochs), so the logged
  `epoch` is derived from `global_step` and restarts near 0 on every resume.
  `global_step` (the wandb x-axis) keeps growing across chunks and the loss / LR
  curves stay continuous. A resetting `epoch` does **not** mean the run restarted
  from scratch.
- **The local `wandb/` dir shows one `run-<timestamp>-<id>` folder per chunk** —
  that is just each chunk's on-disk cache; the *server-side* run is still a
  single continuous run, so don't read multiple `run-*` folders as multiple runs.
- **Offline mode:** the sbatch forces `online`. To train without the wandb
  service, edit the sbatch to `WANDB_MODE=offline`, then upload afterwards from
  a login node with `${DEEPWIND_VENV}/bin/wandb sync <run_dir>/wandb`.

```bash
# (offline mode only) push a cached run to wandb.ai from the login node
/scratch/pawsey0115/hwang4/conda_envs/deepwind/bin/wandb sync \
  /scratch/pawsey0115/hwang4/projects/deepwind/runs/DeepWind-Research/deepwind-base-paper-seed42/wandb
```

The launcher records the commit, dirty-tree patch, resolved command, loaded
modules, Python packages, and a secret-filtered environment snapshot under the
run's `metadata/` directory.

---

**Related docs:** [Architecture audit](architecture-audit.md) · [Asset inventory](asset-inventory.md) · [Baselines & model comparison](baselines.md) · [Data](data.md) · [Model cards](model-cards.md) · [Refactor roadmap](refactor-roadmap.md) · [Reproducibility record](reproducibility.md)

