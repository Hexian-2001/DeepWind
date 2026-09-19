#!/bin/bash --login
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
#SBATCH --job-name=dw-paper-dev
#SBATCH --partition=gpu-dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=03:50:00
#SBATCH --account=YOUR_PROJECT-gpu
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# DeepWind paper-spec training on gpu-dev (2 nodes x 8 GPUs = 16 GPUs).
#
# Used when the main `gpu` partition is drained (QOS=exhausted). Manual
# chunked training: each submission runs ~3h50m, then the user re-submits the
# next chunk on their own schedule. The FIXED run name reuses the same
# output_dir, and train.py auto-resumes from the last checkpoint
# (_detect_checkpoint), so chained submissions simply continue.
#
# Global batch is held at the paper-exact 256 by doubling gradient
# accumulation (GPU count halved 32 -> 16):
#   small        : per_device 8 x accum 2 = 16 / card -> 16 x 16 = 256
#   base | large : per_device 4 x accum 4 = 16 / card -> 16 x 16 = 256
#
# NO --requeue / USR1 signal: the job simply ends at wall-clock. train.py
# catches Slurm's SIGTERM and saves a resumable checkpoint at the next step
# boundary (on top of the periodic save_steps=1000 checkpoints), so chained
# submissions resume from the last completed step. (Manual control keeps
# nodes free for the 03:00 weather-model inference.)
#
#   sbatch --export=ALL,MODEL=base  scripts/setonix/train_paper_dev.sbatch
#   sbatch --export=ALL,MODEL=large scripts/setonix/train_paper_dev.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL="${MODEL:-base}"   # small | base | large
case "${MODEL}" in
  small|base|large) ;;
  *) echo "ERROR: MODEL must be small|base|large (got '${MODEL}')" >&2; exit 2 ;;
esac

# --- Batch correction for 16 GPUs (global batch 256) ------------------------
case "${MODEL}" in
  small)      ACCUM=2 ;;   # paper 8 x 1 on 32 GPUs -> 8 x 2 on 16 GPUs
  base|large) ACCUM=4 ;;   # paper 4 x 2 on 32 GPUs -> 4 x 4 on 16 GPUs
esac
# More frequent checkpoints for short chunks (paper's save_steps=5000 would
# risk losing a whole 3h50m chunk if it never reaches 5000 steps).
# save_total_limit=5 (from config) bounds disk.
OVERRIDES="training.gradient_accumulation_steps=${ACCUM} training.save_steps=1000"
OVERRIDES="${OVERRIDES} ${DEEPWIND_TRAIN_OVERRIDES:-}"

# --- End-of-chunk save deadline (~2 min before the 03:50:00 wall-clock) -----
# The SIGTERM graceful save cannot finish a Base/Large 10 GB checkpoint within
# the 12-30 s SIGTERM->SIGKILL window (job 49418245), so train.py saves + stops
# at this wall-clock deadline instead (computed from this script's start time).
# Override via --export=ALL,DEEPWIND_SAVE_AFTER=<seconds> (e.g. 300 for a test).
export DEEPWIND_SAVE_DEADLINE="$(( $(date +%s) + ${DEEPWIND_SAVE_AFTER:-13680} ))"

# --- Paths (self-contained; DEEPWIND_* are NOT in ~/.bashrc) ---------------
export DEEPWIND_PROJECT_ROOT="${DEEPWIND_PROJECT_ROOT:-$MYSOFTWARE/research_projects/DeepWind}"
export DEEPWIND_DATA_ROOT="${DEEPWIND_DATA_ROOT:-$MYSCRATCH/deepwindData}"
export DEEPWIND_RUNS_ROOT="${DEEPWIND_RUNS_ROOT:-$MYSCRATCH/projects/deepwind/runs}"
export DEEPWIND_VENV="${DEEPWIND_VENV:-$MYSCRATCH/conda_envs/deepwind}"

# --- Run identity: FIXED name => deterministic resume across chunks --------
run_name="${DEEPWIND_RUN_NAME:-deepwind-${MODEL}-paper-seed42}"
run_dir="${DEEPWIND_RUNS_ROOT}/DeepWind-Research/${run_name}"
mkdir -p "${run_dir}"/{checkpoints,logs,metadata,wandb}
exec > >(tee -a "${run_dir}/logs/train.log") 2>&1

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

# Paper-exact global batch 256, peak LR 1e-4, warmup 3%, cosine, grad clip 1.0,
# BF16, 100k steps -- identical data/trajectory to the 32-GPU run.
command="${DEEPWIND_VENV}/bin/python -m torch.distributed.run \
  --nproc_per_node=${gpus_per_node} --nnodes=${SLURM_JOB_NUM_NODES} \
  --node_rank=\${SLURM_NODEID} \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=${master_addr}:${master_port} \
  ${DEEPWIND_PROJECT_ROOT}/train.py \
  run_name=${run_name} model=deepwind_${MODEL} data=train data_eval=eval training=deepwind_${MODEL} \
  training.output_dir=${run_dir}/checkpoints \
  wandb.tags=[${MODEL},paper-spec,baseline,seed42] \
  wandb.notes=DeepWind-${MODEL}_paper-spec_batch256_100ksteps_dev2x8 \
  ${OVERRIDES}"
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
