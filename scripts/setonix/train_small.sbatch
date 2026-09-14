#!/bin/bash --login
#SBATCH --job-name=dw-small-paper
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --account=pawsey0115-gpu
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# ---------------------------------------------------------------------------
# DeepWind-Small — paper-spec pretraining (production).
#
# Production launch (4 nodes × 8 GPUs, global batch 256, 100k steps):
#     sbatch scripts/setonix/train_small.sbatch
#
# Smoke test (2 nodes × 2 GPUs on gpu-dev, 5 steps, wandb online):
#     DEEPWIND_RUN_NAME=deepwind-small-paper-smoke \
#     GPUS_PER_NODE=2 \
#     DEEPWIND_TRAIN_OVERRIDES="training.max_steps=5 training.eval_strategy=no \
#       training.save_steps=5 training.logging_steps=1 \
#       training.per_device_train_batch_size=1 training.gradient_accumulation_steps=1 \
#       training.dataloader_num_workers=1" \
#     sbatch --partition=gpu-dev --nodes=2 --gpus-per-node=2 --time=00:30:00 \
#       scripts/setonix/train_small.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Paths (self-contained; DEEPWIND_* are NOT in ~/.bashrc) ---------------
export DEEPWIND_PROJECT_ROOT="${DEEPWIND_PROJECT_ROOT:-/software/projects/pawsey0115/hwang4/research_projects/DeepWind}"
export DEEPWIND_DATA_ROOT="${DEEPWIND_DATA_ROOT:-/scratch/pawsey0115/hwang4/deepwindData}"
export DEEPWIND_RUNS_ROOT="${DEEPWIND_RUNS_ROOT:-/scratch/pawsey0115/hwang4/projects/deepwind/runs}"
export DEEPWIND_VENV="${DEEPWIND_VENV:-/scratch/pawsey0115/hwang4/conda_envs/deepwind}"

# --- Run identity (naming convention: deepwind-{size}-{variant}-seed{seed}-{date}) ---
run_name="${DEEPWIND_RUN_NAME:-deepwind-small-paper-seed42-$(date -u +%Y%m%d-%H%M%S)}"
run_dir="${DEEPWIND_RUNS_ROOT}/DeepWind-Research/${run_name}"
mkdir -p "${run_dir}"/{checkpoints,logs,metadata,wandb}
exec > >(tee -a "${run_dir}/logs/train.log") 2>&1

# --- Requeue on preemption -------------------------------------------------
requeue_on_timeout() {
  printf 'Received pre-timeout signal; requeueing job %s.\n' "${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_on_timeout USR1

# --- Provenance snapshot (captured before any training) --------------------
git -C "${DEEPWIND_PROJECT_ROOT}" rev-parse HEAD > "${run_dir}/metadata/git_commit.txt"
git -C "${DEEPWIND_PROJECT_ROOT}" status --porcelain > "${run_dir}/metadata/git_status.txt"
git -C "${DEEPWIND_PROJECT_ROOT}" diff --binary > "${run_dir}/metadata/uncommitted.patch"
"${DEEPWIND_VENV}/bin/python" -m pip freeze > "${run_dir}/metadata/pip_freeze.txt"
env | sort | sed -E '/(TOKEN|KEY|SECRET|PASSWORD|PASS)=/Id' > "${run_dir}/metadata/environment.txt"

# --- DDP / RCCL endpoints --------------------------------------------------
master_addr="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
master_port="$((20000 + SLURM_JOB_ID % 20000))"
# Processes-per-node must match the requested GRES; --exclusive can inflate
# SLURM_GPUS_ON_NODE to node capacity, so keep this explicit (GPUS_PER_NODE).
gpus_per_node="${GPUS_PER_NODE:-8}"

# --- Runtime env (direct venv; NO singularity, NO module load) -------------
export OMP_NUM_THREADS=4
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=online
export WANDB_DIR="${run_dir}/wandb"

# Paper-exact protocol: global batch = per_device(8) × accum(1) × 4×8 = 256;
# peak LR 1e-4, warmup 3% (3000 steps), cosine, grad clip 1.0, BF16, 100k steps.
command="${DEEPWIND_VENV}/bin/python -m torch.distributed.run \
  --nproc_per_node=${gpus_per_node} --nnodes=${SLURM_JOB_NUM_NODES} \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=${master_addr}:${master_port} \
  ${DEEPWIND_PROJECT_ROOT}/train.py \
  run_name=${run_name} model=deepwind_small data=train data_eval=eval training=deepwind_small \
  training.output_dir=${run_dir}/checkpoints \
  wandb.tags=[small,paper-spec,baseline,seed42] \
  wandb.notes=DeepWind-Small_paper-spec_baseline_batch256_100ksteps \
  ${DEEPWIND_TRAIN_OVERRIDES:-}"
printf '%s\n' "${command}" > "${run_dir}/metadata/command.txt"

srun -N "${SLURM_JOB_NUM_NODES}" -n "${SLURM_JOB_NUM_NODES}" --cpu-bind=none \
  bash -c "${command}"
