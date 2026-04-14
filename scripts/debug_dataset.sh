#!/bin/bash --login
#SBATCH --job-name=deepwind_debug_sharding
#SBATCH --partition=gpu-dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0115-gpu
#SBATCH --output=/scratch/pawsey0115/hwang4/results/DeepWind-Research/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/pawsey0115/hwang4/results/DeepWind-Research/slurm-logs/%x-%j.err

echo "===== Debug sharding job started on $(hostname) at $(date) ====="

# ---------------------------------------------------------------------------
# 1. Modules (identical to training job)
# ---------------------------------------------------------------------------
module load pytorch/2.7.1-rocm6.3.3
module load cray-python

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
export PROJECT_ROOT="/scratch/pawsey0115/hwang4/project_codes/deepwind_research"
export SCRATCH_ROOT="/scratch/pawsey0115/hwang4"
export RESULTS_ROOT="$SCRATCH_ROOT/results"
export VENV_PATH="/software/projects/pawsey0115/hwang4/manual/software/pythonEnvironments/pytorchContainer-environments/time-moe"
export TORCH_EXTENSIONS_DIR="$SCRATCH_ROOT/.torch_extensions_cache"

mkdir -p "$TORCH_EXTENSIONS_DIR"
mkdir -p "$RESULTS_ROOT/DeepWind-Research/slurm-logs"

# ---------------------------------------------------------------------------
# 3. Distributed setup (2 nodes x 8 GPUs = 16 GPUs total)
# ---------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29501   # Use a different port to avoid conflicts with training jobs
echo "Master node  : $MASTER_ADDR:$MASTER_PORT"
echo "Nodes        : $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "All nodes    : $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

# ---------------------------------------------------------------------------
# 4. Container environment variables (identical to training job)
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

# Enable DEBUG-level logging so per-worker shard info is printed
export SINGULARITYENV_PYTHONPATH="$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 5. Dataset paths — update these to match your actual data location
# ---------------------------------------------------------------------------
NPY_ROOT="/scratch/pawsey0115/hwang4/deepwindData/train"
METADATA_PATH="/scratch/pawsey0115/hwang4/deepwindData/train_metadata.csv"

# num_workers must match the value used in your real training DataLoader
NUM_WORKERS=4

# ---------------------------------------------------------------------------
# 6. Debug command
# ---------------------------------------------------------------------------
BIND_PATHS="$PROJECT_ROOT,$VENV_PATH,$SCRATCH_ROOT,$TORCH_EXTENSIONS_DIR"

CMD="export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
    ${VENV_PATH}/bin/python -m torch.distributed.run \
        --nproc_per_node=$SLURM_GPUS_ON_NODE \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $PROJECT_ROOT/test/debug_dataset_sharding.py \
        --npy_root       $NPY_ROOT \
        --metadata_path  $METADATA_PATH \
        --num_workers    $NUM_WORKERS"

echo "Debug command: $CMD"

# ---------------------------------------------------------------------------
# 7. Launch via Singularity
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

# ---------------------------------------------------------------------------
# 8. Copy the JSON result back to results directory for offline inspection
# ---------------------------------------------------------------------------
RESULT_JSON="debug_sharding_result.json"
if [ -f "$RESULT_JSON" ]; then
    DEST="$RESULTS_ROOT/DeepWind-Research/slurm-logs/debug_sharding_${SLURM_JOB_ID}.json"
    cp "$RESULT_JSON" "$DEST"
    echo "Sharding result saved to: $DEST"
fi

echo "===== Debug sharding job finished at $(date) with exit code $EXIT_CODE ====="
exit $EXIT_CODE