#!/bin/bash --login
#SBATCH --output=/scratch/pawsey0115/hwang4/projects/deepwind/runs/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/pawsey0115/hwang4/projects/deepwind/runs/slurm-logs/%x-%j.err
#SBATCH --job-name=deepwind-smoke
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0115-gpu

set -euo pipefail

module load pytorch/2.7.1-rocm6.3.3

: "${DEEPWIND_PROJECT_ROOT:?Set DEEPWIND_PROJECT_ROOT before sbatch --export=ALL}"
: "${DEEPWIND_DATA_ROOT:?Set DEEPWIND_DATA_ROOT before sbatch --export=ALL}"
: "${DEEPWIND_RUNS_ROOT:?Set DEEPWIND_RUNS_ROOT before sbatch --export=ALL}"
: "${DEEPWIND_VENV:?Set DEEPWIND_VENV before sbatch --export=ALL}"

run_id="smoke-${SLURM_JOB_ID}"
run_dir="${DEEPWIND_RUNS_ROOT}/${run_id}"
mkdir -p "${run_dir}/logs" "${run_dir}/checkpoints" "${run_dir}/metadata"

exec > >(tee -a "${run_dir}/logs/train.log") 2>&1

git -C "$DEEPWIND_PROJECT_ROOT" rev-parse HEAD > "${run_dir}/metadata/git_commit.txt"
git -C "$DEEPWIND_PROJECT_ROOT" status --porcelain > "${run_dir}/metadata/git_status.txt"
module list 2> "${run_dir}/metadata/modules.txt"
"${DEEPWIND_VENV}/bin/python" -m pip freeze > "${run_dir}/metadata/pip_freeze.txt"

export SINGULARITYENV_DEEPWIND_DATA_ROOT="$DEEPWIND_DATA_ROOT"
export SINGULARITYENV_DEEPWIND_RUNS_ROOT="$DEEPWIND_RUNS_ROOT"
export SINGULARITYENV_PYTHONPATH="$DEEPWIND_PROJECT_ROOT"
export SINGULARITYENV_WANDB_MODE=disabled

srun -N 1 -n 1 singularity exec --rocm \
  --bind "$DEEPWIND_PROJECT_ROOT,$DEEPWIND_DATA_ROOT,$DEEPWIND_RUNS_ROOT,$DEEPWIND_VENV" \
  "$SINGULARITY_CONTAINER" \
  "$DEEPWIND_VENV/bin/python" "$DEEPWIND_PROJECT_ROOT/train.py" \
    run_name="$run_id" \
    model=deepwind_debug \
    data=train \
    data_eval=eval \
    training=deepwind_small \
    training.output_dir="${run_dir}/checkpoints" \
    training.max_steps=5 \
    training.eval_strategy=no \
    training.save_steps=5 \
    training.logging_steps=1 \
    training.per_device_train_batch_size=1 \
    training.gradient_accumulation_steps=1 \
    training.dataloader_num_workers=1 \
    training.report_to=none \
    data.npy_root="${DEEPWIND_DATA_ROOT}/train" \
    data.metadata_path="${DEEPWIND_DATA_ROOT}/train_metadata.csv"
