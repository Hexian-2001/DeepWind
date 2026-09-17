#!/bin/bash --login
#SBATCH --output=/scratch/pawsey0115/hwang4/projects/deepwind/runs/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/pawsey0115/hwang4/projects/deepwind/runs/slurm-logs/%x-%j.err
#SBATCH --job-name=deepwind-ckpt-aux
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0115-gpu

set -euo pipefail
module load pytorch/2.7.1-rocm6.3.3

: "${DEEPWIND_PROJECT_ROOT:?Set DEEPWIND_PROJECT_ROOT}"
: "${DEEPWIND_VENV:?Set DEEPWIND_VENV}"

export SINGULARITYENV_PYTHONPATH="${DEEPWIND_PROJECT_ROOT}"

srun -N 1 -n 1 singularity exec --rocm \
  --bind "${DEEPWIND_PROJECT_ROOT},${DEEPWIND_VENV}" \
  "${SINGULARITY_CONTAINER}" \
  "${DEEPWIND_VENV}/bin/python" "${DEEPWIND_PROJECT_ROOT}/test/test_checkpoint_aux_loss.py"
