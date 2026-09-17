#!/bin/bash --login
#SBATCH --job-name=deepwind-ddp-smoke
#SBATCH --partition=gpu-dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0115-gpu

set -euo pipefail
module load pytorch/2.7.1-rocm6.3.3

: "${DEEPWIND_PROJECT_ROOT:?Set DEEPWIND_PROJECT_ROOT}"
: "${DEEPWIND_DATA_ROOT:?Set DEEPWIND_DATA_ROOT}"
: "${DEEPWIND_RUNS_ROOT:?Set DEEPWIND_RUNS_ROOT}"
: "${DEEPWIND_VENV:?Set DEEPWIND_VENV}"

run_id="ddp-smoke-${SLURM_JOB_ID}"
run_dir="${DEEPWIND_RUNS_ROOT}/${run_id}"
mkdir -p "${run_dir}"/{checkpoints,logs,metadata}
exec > >(tee -a "${run_dir}/logs/train.log") 2>&1

git -C "${DEEPWIND_PROJECT_ROOT}" rev-parse HEAD > "${run_dir}/metadata/git_commit.txt"
git -C "${DEEPWIND_PROJECT_ROOT}" status --porcelain > "${run_dir}/metadata/git_status.txt"
module list 2> "${run_dir}/metadata/modules.txt"
"${DEEPWIND_VENV}/bin/python" -m pip freeze > "${run_dir}/metadata/pip_freeze.txt"

master_addr="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
master_port="$((20000 + SLURM_JOB_ID % 20000))"
# `--exclusive` can make SLURM_GPUS_ON_NODE report the node capacity rather
# than this smoke test's requested GRES. Keep the intended process count explicit.
gpus_per_node=2

export SINGULARITYENV_DEEPWIND_DATA_ROOT="${DEEPWIND_DATA_ROOT}"
export SINGULARITYENV_DEEPWIND_RUNS_ROOT="${DEEPWIND_RUNS_ROOT}"
export SINGULARITYENV_PYTHONPATH="${DEEPWIND_PROJECT_ROOT}"
export SINGULARITYENV_WANDB_MODE=disabled
export SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0
export SINGULARITYENV_NCCL_DEBUG=INFO
export SINGULARITYENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1

command="${DEEPWIND_VENV}/bin/python -m torch.distributed.run \
  --nproc_per_node=${gpus_per_node} --nnodes=${SLURM_JOB_NUM_NODES} \
  --node_rank=\${SLURM_NODEID} \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=${master_addr}:${master_port} \
  ${DEEPWIND_PROJECT_ROOT}/train.py \
  run_name=${run_id} model=deepwind_debug data=train data_eval=eval training=deepwind_small \
  training.output_dir=${run_dir}/checkpoints training.max_steps=5 \
  training.eval_strategy=no training.save_steps=5 training.logging_steps=1 \
  training.per_device_train_batch_size=1 training.gradient_accumulation_steps=1 \
  training.dataloader_num_workers=1 training.report_to=none \
  data.npy_root=${DEEPWIND_DATA_ROOT}/train \
  data.metadata_path=${DEEPWIND_DATA_ROOT}/train_metadata.csv"
printf '%s\n' "${command}" > "${run_dir}/metadata/command.txt"

srun -N "${SLURM_JOB_NUM_NODES}" -n "${SLURM_JOB_NUM_NODES}" --cpu-bind=none \
  singularity exec --rocm \
  --bind "${DEEPWIND_PROJECT_ROOT},${DEEPWIND_DATA_ROOT},${DEEPWIND_RUNS_ROOT},${DEEPWIND_VENV}" \
  "${SINGULARITY_CONTAINER}" bash -lc "${command}"
