#!/bin/bash --login
#SBATCH --job-name=deepwind-ckpt-smoke
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0115-gpu

set -euo pipefail
module load pytorch/2.7.1-rocm6.3.3

: "${DEEPWIND_PROJECT_ROOT:?Set DEEPWIND_PROJECT_ROOT}"
: "${DEEPWIND_DATA_ROOT:?Set DEEPWIND_DATA_ROOT}"
: "${DEEPWIND_CHECKPOINT:?Set DEEPWIND_CHECKPOINT}"
: "${DEEPWIND_REPORTS_ROOT:?Set DEEPWIND_REPORTS_ROOT}"
: "${DEEPWIND_VENV:?Set DEEPWIND_VENV}"

mkdir -p "${DEEPWIND_REPORTS_ROOT}"
report="${DEEPWIND_REPORTS_ROOT}/checkpoint-smoke-${SLURM_JOB_ID}.json"

srun singularity exec --rocm \
  --bind "${DEEPWIND_PROJECT_ROOT},${DEEPWIND_DATA_ROOT},${DEEPWIND_CHECKPOINT},${DEEPWIND_REPORTS_ROOT},${DEEPWIND_VENV}" \
  "${SINGULARITY_CONTAINER}" \
  "${DEEPWIND_VENV}/bin/python" \
  "${DEEPWIND_PROJECT_ROOT}/tools/smoke_checkpoint.py" \
  --checkpoint "${DEEPWIND_CHECKPOINT}" \
  --data-root "${DEEPWIND_DATA_ROOT}" | tee "${report}"
