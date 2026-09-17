#!/bin/bash --login
#SBATCH --output=/scratch/pawsey0115/hwang4/projects/deepwind/runs/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/pawsey0115/hwang4/projects/deepwind/runs/slurm-logs/%x-%j.err
#SBATCH --job-name=dw-paper
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
# DeepWind paper-spec pretraining — generalized for small | base | large.
#
# The run name is FIXED per model (no timestamp) so that on --requeue the job
# reuses the same output_dir and train.py auto-resumes from the last
# checkpoint (see _detect_checkpoint in train.py). This is the mechanism that
# lets a >24h run survive the 24h wall-clock limit.
#
#   MODEL=base  sbatch scripts/setonix/train_paper.sbatch     # production
#   MODEL=large sbatch scripts/setonix/train_paper.sbatch
#
# Smoke / resume test (2 nodes × 2 GPUs on gpu-dev):
#   MODEL=base GPUS_PER_NODE=2 DEEPWIND_RUN_NAME=deepwind-base-resume-test \
#   DEEPWIND_TRAIN_OVERRIDES="training.max_steps=8 training.eval_strategy=no \
#     training.save_steps=4 training.logging_steps=1 \
#     training.per_device_train_batch_size=1 training.gradient_accumulation_steps=1 \
#     training.dataloader_num_workers=1" \
#   sbatch --partition=gpu-dev --nodes=2 --gpus-per-node=2 --time=00:30:00 \
#     scripts/setonix/train_paper.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL="${MODEL:-base}"   # small | base | large
case "${MODEL}" in
  small|base|large) ;;
  *) echo "ERROR: MODEL must be small|base|large (got '${MODEL}')" >&2; exit 2 ;;
esac

# --- Paths (self-contained; DEEPWIND_* are NOT in ~/.bashrc) ---------------
export DEEPWIND_PROJECT_ROOT="${DEEPWIND_PROJECT_ROOT:-/software/projects/pawsey0115/hwang4/research_projects/DeepWind}"
export DEEPWIND_DATA_ROOT="${DEEPWIND_DATA_ROOT:-/scratch/pawsey0115/hwang4/deepwindData}"
export DEEPWIND_RUNS_ROOT="${DEEPWIND_RUNS_ROOT:-/scratch/pawsey0115/hwang4/projects/deepwind/runs}"
export DEEPWIND_VENV="${DEEPWIND_VENV:-/scratch/pawsey0115/hwang4/conda_envs/deepwind}"

# --- Run identity: FIXED name => deterministic resume across requeue -------
run_name="${DEEPWIND_RUN_NAME:-deepwind-${MODEL}-paper-seed42}"
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
gpus_per_node="${GPUS_PER_NODE:-8}"

# --- Runtime env (direct venv; NO singularity, NO module load) -------------
export OMP_NUM_THREADS=4
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=online
export WANDB_DIR="${run_dir}/wandb"
export WANDB_CONFIG_DIR="${DEEPWIND_RUNS_ROOT}/.wandb/config"
export WANDB_CACHE_DIR="${DEEPWIND_RUNS_ROOT}/.wandb/cache"
export WANDB_DATA_DIR="${DEEPWIND_RUNS_ROOT}/.wandb/data"

# Paper-exact protocol: global batch = per_device(4) × accum(2) × 4×8 = 256;
# peak LR 1e-4, warmup 3% (3000 steps), cosine, grad clip 1.0, BF16, 100k steps.
command="${DEEPWIND_VENV}/bin/python -m torch.distributed.run \
  --nproc_per_node=${gpus_per_node} --nnodes=${SLURM_JOB_NUM_NODES} \
  --node_rank=\${SLURM_NODEID} \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=${master_addr}:${master_port} \
  ${DEEPWIND_PROJECT_ROOT}/train.py \
  run_name=${run_name} model=deepwind_${MODEL} data=train data_eval=eval training=deepwind_${MODEL} \
  training.output_dir=${run_dir}/checkpoints \
  wandb.tags=[${MODEL},paper-spec,baseline,seed42] \
  wandb.notes=DeepWind-${MODEL}_paper-spec_baseline_batch256_100ksteps \
  ${DEEPWIND_TRAIN_OVERRIDES:-}"
printf '%s\n' "${command}" > "${run_dir}/metadata/command.txt"

# The c10d TCP rendezvous over the management net (10.168.x) is occasionally
# flaky (60s connect timeout); retry the torchrun launch a few times. NCCL
# (RDMA over Slingshot) is reliable once past the rendezvous.
launched=0
for attempt in 1 2 3; do
  echo "=== torchrun launch attempt ${attempt}/3 ==="
  if srun -N "${SLURM_JOB_NUM_NODES}" -n "${SLURM_JOB_NUM_NODES}" --cpu-bind=none \
      bash -c "${command}"; then
    launched=1
    break
  fi
  echo "=== torchrun attempt ${attempt} failed; retrying in 15s ==="
  sleep 15
done
if [ "${launched}" -ne 1 ]; then
  echo "ERROR: torchrun launch failed after 3 attempts" >&2
  exit 1
fi
