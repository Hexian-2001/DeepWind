#!/bin/bash --login
#SBATCH --job-name=deepwind_base_train
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --account=pawsey0115-gpu
#SBATCH --output=/scratch/pawsey0115/hwang4/results/DeepWind-Research/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/pawsey0115/hwang4/results/DeepWind-Research/slurm-logs/%x-%j.err

echo "===== Training job started on $(hostname) at $(date) ====="

# ---------------------------------------------------------------------------
# 1. Host modules
# ---------------------------------------------------------------------------
module load pytorch/2.7.1-rocm6.3.3
module load cray-python

echo "Container: $SINGULARITY_CONTAINER"

# ---------------------------------------------------------------------------
# 2. Paths and workspace
# ---------------------------------------------------------------------------
export PROJECT_ROOT="/scratch/pawsey0115/hwang4/project_codes/deepwind_research"
export SCRATCH_ROOT="/scratch/pawsey0115/hwang4"
export RESULTS_ROOT="$SCRATCH_ROOT/results"
export VENV_PATH="/software/projects/pawsey0115/hwang4/manual/software/pythonEnvironments/pytorchContainer-environments/time-moe"
export TORCH_EXTENSIONS_DIR="$SCRATCH_ROOT/.torch_extensions_cache"

mkdir -p "$TORCH_EXTENSIONS_DIR"
mkdir -p "$RESULTS_ROOT/DeepWind-Research/slurm-logs"
mkdir -p "$RESULTS_ROOT/wandb_cache"

# ---------------------------------------------------------------------------
# 3. Distributed setup
# ---------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "Master node:  $MASTER_ADDR:$MASTER_PORT"
echo "Nodes:        $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "All nodes:    $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

# ---------------------------------------------------------------------------
# 4. Container environment
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

export DS_BUILD_AIO=0
export LIBAIO_DISABLE=1
export SINGULARITYENV_WANDB_MODE="online"
export SINGULARITYENV_WANDB_DIR="$RESULTS_ROOT/wandb_cache"

# ---------------------------------------------------------------------------
# 5. Training command
# ---------------------------------------------------------------------------
BIND_PATHS="$PROJECT_ROOT,$VENV_PATH,$SCRATCH_ROOT,$TORCH_EXTENSIONS_DIR"

CMD="export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
    ${VENV_PATH}/bin/python -m torch.distributed.run \
        --nproc_per_node=$SLURM_GPUS_ON_NODE \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $PROJECT_ROOT/train.py \
        run_name=deepwind_base \
        model=deepwind_base \
        data=train \
        data_eval=eval \
        training=deepwind_base"

echo "Training command: $CMD"

# ---------------------------------------------------------------------------
# 6. Launch via Singularity
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
echo "===== Training job finished at $(date) with exit code $EXIT_CODE ====="
exit $EXIT_CODE