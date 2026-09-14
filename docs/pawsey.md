# Pawsey Setonix workflow

Keep Git repositories and environments under `/software/projects`, and keep
large datasets, checkpoints, logs, W&B caches, and temporary outputs under
`/scratch`.

Recommended environment variables:

```bash
export DEEPWIND_PROJECT_ROOT=/software/projects/<project>/<user>/opensource_projects/github_projects/DeepWind
export DEEPWIND_DATA_ROOT=/scratch/<project>/<user>/deepwindData
export DEEPWIND_RUNS_ROOT=/scratch/<project>/<user>/deepwind-open/runs
export DEEPWIND_VENV=/software/projects/<project>/<user>/venvs/deepwind
```

Use Pawsey's supported PyTorch ROCm module/container for multi-node jobs. A
virtual environment may be layered inside the container for Python-only
dependencies. Run `scripts/setonix/smoke.sbatch` before scaling to a production
allocation.

For each production launch, record `module list`, `$SINGULARITY_CONTAINER`,
`git rev-parse HEAD`, `git status --porcelain`, `pip freeze`, and Slurm metadata
inside the run directory.

## Two-node Large training

The production launcher defaults to WandB offline mode so every run survives
an external-service outage. Use online mode only after authenticating outside
the repository.

```bash
export DEEPWIND_RUN_NAME=large-checkpoint-replication-v1
export WANDB_MODE=offline
sbatch --export=ALL scripts/setonix/train_large.sbatch
```

The launcher records the commit, dirty-tree patch, resolved command, loaded
modules, Python packages, and a secret-filtered environment snapshot under the
run's `metadata/` directory.
