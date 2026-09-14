#!/bin/bash --login
#SBATCH --job-name=dw-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --account=pawsey0115-gpu

# ---------------------------------------------------------------------------
# DeepWind evaluation on the 8 WindBench test datasets (6 horizons).
#
#   MODEL=small CKPT=<checkpoint_dir> sbatch scripts/setonix/eval_paper.sbatch
#
# Writes per-(dataset,horizon) metric JSONs under
#   ${DEEPWIND_RUNS_ROOT}/results/deepwind/${EVAL_NAME}/
# then aggregate with:  python tools/aggregate_eval.py <that dir>
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL="${MODEL:-small}"          # small | base | large
CKPT="${CKPT:?CKPT (checkpoint dir) required}"
EVAL_NAME="${EVAL_NAME:-eval-${MODEL}-paper-seed42}"

# --- Paths (self-contained) -------------------------------------------------
export DEEPWIND_PROJECT_ROOT="${DEEPWIND_PROJECT_ROOT:-/software/projects/pawsey0115/hwang4/research_projects/DeepWind}"
export DEEPWIND_DATA_ROOT="${DEEPWIND_DATA_ROOT:-/scratch/pawsey0115/hwang4/deepwindData}"
export DEEPWIND_RUNS_ROOT="${DEEPWIND_RUNS_ROOT:-/scratch/pawsey0115/hwang4/projects/deepwind/runs}"
export DEEPWIND_VENV="${DEEPWIND_VENV:-/scratch/pawsey0115/hwang4/conda_envs/deepwind}"

out_dir="${DEEPWIND_RUNS_ROOT}/results/deepwind/${EVAL_NAME}"
mkdir -p "${out_dir}"
exec > >(tee -a "${out_dir}/eval.log") 2>&1

export NCCL_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

master_addr="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
master_port="$((20000 + SLURM_JOB_ID % 20000))"
gpus_per_node="${GPUS_PER_NODE:-8}"

command="${DEEPWIND_VENV}/bin/python -m torch.distributed.run \
  --nproc_per_node=${gpus_per_node} --nnodes=1 \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=${master_addr}:${master_port} \
  ${DEEPWIND_PROJECT_ROOT}/evaluate.py \
  model=deepwind_${MODEL} \
  run_name=${EVAL_NAME} model_name=${EVAL_NAME} \
  inference.checkpoint_path=${CKPT} \
  output.output_dir=${out_dir} \
  output.save_rolling_plots=false"
printf '%s\n' "${command}" > "${out_dir}/command.txt"

srun -N 1 -n 1 --cpu-bind=none bash -c "${command}"
