#!/bin/bash --login
#SBATCH --job-name=deepwind-large
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --account=pawsey0115-gpu
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

module load pytorch/2.7.1-rocm6.3.3

: "${DEEPWIND_PROJECT_ROOT:?Set DEEPWIND_PROJECT_ROOT}"
: "${DEEPWIND_DATA_ROOT:?Set DEEPWIND_DATA_ROOT}"
: "${DEEPWIND_RUNS_ROOT:?Set DEEPWIND_RUNS_ROOT}"
: "${DEEPWIND_VENV:?Set DEEPWIND_VENV}"
: "${WANDB_MODE:=offline}"

run_name="${DEEPWIND_RUN_NAME:-large-${SLURM_JOB_ID}}"
run_dir="${DEEPWIND_RUNS_ROOT}/${run_name}"
mkdir -p "${run_dir}"/{checkpoints,logs,metadata,wandb}
exec > >(tee -a "${run_dir}/logs/train.log") 2>&1

requeue_on_timeout() {
  printf 'Received pre-timeout signal; requeueing job %s.\n' "${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_on_timeout USR1

git -C "${DEEPWIND_PROJECT_ROOT}" rev-parse HEAD > "${run_dir}/metadata/git_commit.txt"
git -C "${DEEPWIND_PROJECT_ROOT}" status --porcelain > "${run_dir}/metadata/git_status.txt"
git -C "${DEEPWIND_PROJECT_ROOT}" diff --binary > "${run_dir}/metadata/uncommitted.patch"
module list 2> "${run_dir}/metadata/modules.txt"
"${DEEPWIND_VENV}/bin/python" -m pip freeze > "${run_dir}/metadata/pip_freeze.txt"
env | sort | sed -E '/(TOKEN|KEY|SECRET|PASSWORD|PASS)=/Id' > "${run_dir}/metadata/environment.txt"

master_addr="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
master_port="$((20000 + SLURM_JOB_ID % 20000))"
gpus_per_node="${SLURM_GPUS_ON_NODE:-8}"

export SINGULARITYENV_DEEPWIND_DATA_ROOT="${DEEPWIND_DATA_ROOT}"
export SINGULARITYENV_DEEPWIND_RUNS_ROOT="${DEEPWIND_RUNS_ROOT}"
export SINGULARITYENV_PYTHONPATH="${DEEPWIND_PROJECT_ROOT}"
export SINGULARITYENV_WANDB_MODE="${WANDB_MODE}"
export SINGULARITYENV_WANDB_DIR="${run_dir}/wandb"
export SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0
export SINGULARITYENV_NCCL_DEBUG=WARN
export SINGULARITYENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SINGULARITYENV_OMP_NUM_THREADS=4

command="${DEEPWIND_VENV}/bin/python -m torch.distributed.run \
  --nproc_per_node=${gpus_per_node} --nnodes=${SLURM_JOB_NUM_NODES} \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=${master_addr}:${master_port} \
  ${DEEPWIND_PROJECT_ROOT}/train.py \
  run_name=${run_name} model=deepwind_large data=train data_eval=eval training=deepwind_large \
  model.num_experts=8 model.pred_head_type=quantile \
  model.use_rotary_emb=true model.use_xpos=true model.aux_loss_weight=0.02 \
  training.learning_rate=1e-4 training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=4 training.max_steps=100000 \
  training.output_dir=${run_dir}/checkpoints"
printf '%s\n' "${command}" > "${run_dir}/metadata/command.txt"

srun -N "${SLURM_JOB_NUM_NODES}" -n "${SLURM_JOB_NUM_NODES}" --cpu-bind=none \
  singularity exec --rocm \
  --bind "${DEEPWIND_PROJECT_ROOT},${DEEPWIND_DATA_ROOT},${DEEPWIND_RUNS_ROOT},${DEEPWIND_VENV}" \
  "${SINGULARITY_CONTAINER}" bash -lc "${command}"
