#!/bin/bash --login
#SBATCH --job-name=debug_dataloader
#SBATCH --partition=gpu-dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0115-gpu
#SBATCH --output=/scratch/pawsey0115/hwang4/results/DeepWind-Research/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/pawsey0115/hwang4/results/DeepWind-Research/slurm-logs/%x-%j.err

echo "===== Debug DataLoader started on $(hostname) at $(date) ====="

# ---------------------------------------------------------------------------
# 1. Modules (identical to training)
# ---------------------------------------------------------------------------
module load pytorch/2.7.1-rocm6.3.3
module load cray-python

# ---------------------------------------------------------------------------
# 2. Paths (identical to training)
# ---------------------------------------------------------------------------
export PROJECT_ROOT="/scratch/pawsey0115/hwang4/project_codes/deepwind_research"
export SCRATCH_ROOT="/scratch/pawsey0115/hwang4"
export RESULTS_ROOT="$SCRATCH_ROOT/results"
export VENV_PATH="/software/projects/pawsey0115/hwang4/manual/software/pythonEnvironments/pytorchContainer-environments/time-moe"
export TORCH_EXTENSIONS_DIR="$SCRATCH_ROOT/.torch_extensions_cache"

mkdir -p "$TORCH_EXTENSIONS_DIR"
mkdir -p "$RESULTS_ROOT/DeepWind-Research/slurm-logs"

# ---------------------------------------------------------------------------
# 3. Distributed setup (identical to training)
# ---------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "Master: $MASTER_ADDR:$MASTER_PORT  Nodes: $SLURM_JOB_NUM_NODES  GPUs/node: $SLURM_GPUS_ON_NODE"

# ---------------------------------------------------------------------------
# 4. Container env (identical to training)
# ---------------------------------------------------------------------------
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_OMP_NUM_THREADS=4
export SINGULARITYENV_SCRATCH_ROOT="$SCRATCH_ROOT"
export SINGULARITYENV_TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
export SINGULARITYENV_NCCL_DEBUG=WARN
export SINGULARITYENV_NCCL_P2P_DISABLE=1
export SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0
export SINGULARITYENV_TORCH_DISTRIBUTED_DEBUG=OFF
export SINGULARITYENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SINGULARITYENV_WANDB_MODE=disabled

# ---------------------------------------------------------------------------
# 5. Debug command
# ---------------------------------------------------------------------------
BIND_PATHS="$PROJECT_ROOT,$VENV_PATH,$SCRATCH_ROOT,$TORCH_EXTENSIONS_DIR"

# Place debug_dataloader.py in PROJECT_ROOT before submitting
CMD="export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
    ${VENV_PATH}/bin/python -m torch.distributed.run \
        --nproc_per_node=$SLURM_GPUS_ON_NODE \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $PROJECT_ROOT/test/debug_dataloader.py \
        --npy_root /scratch/pawsey0115/hwang4/deepwindData/train \
        --metadata_path /scratch/pawsey0115/hwang4/deepwindData/train_metadata.csv \
        --seq_len 8192 \
        --batch_size 32 \
        --grad_accum 2 \
        --num_workers 4 \
        --num_steps 20 \
        --prefetch_factor 2"

echo "CMD: $CMD"

# ---------------------------------------------------------------------------
# 6. Launch (identical to training)
# ---------------------------------------------------------------------------
srun \
    -N "$SLURM_JOB_NUM_NODES" \
    -n "$SLURM_JOB_NUM_NODES" \
    --cpu-bind=none \
    singularity exec --rocm \
        --bind "$BIND_PATHS" \
        "$SINGULARITY_CONTAINER" \
        bash -c "$CMD"

EXIT_CODE=$?
echo "===== Debug finished at $(date) with exit code $EXIT_CODE ====="
exit $EXIT_CODE