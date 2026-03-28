#!/bin/bash --login
#SBATCH --job-name=deepwind_debug
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --account=pawsey0115-gpu
#SBATCH --output=/scratch/pawsey0115/hwang4/results/DeepWind-Research/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/pawsey0115/hwang4/results/DeepWind-Research/slurm-logs/%x-%j.err

echo "===== Debug job started on $(hostname) at $(date) ====="

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
# 3. Distributed setup — fully dynamic, adapts to any --nodes / --gpus-per-node
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

export SINGULARITYENV_TORCH_DISTRIBUTED_DEBUG=INFO
export SINGULARITYENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export DS_BUILD_AIO=0
export LIBAIO_DISABLE=1
export SINGULARITYENV_WANDB_MODE="disabled"

# ---------------------------------------------------------------------------
# 5. Debug training command
#
# --nproc_per_node and --nnodes are driven by SLURM env vars so this script
# works correctly regardless of what --nodes / --gpus-per-node are set to
# in the #SBATCH directives above.
#
# Verification checklist (run in order):
#   [1] train/loss decreases within 50 steps       -> forward/backward pass OK
#   [2] eval_windtoolkit_loss is non-null           -> DeepWindTrainer override OK
#   [3] eval_scada_loss is non-null                 -> per-group eval OK
#   [4] checkpoint-50/ directory is created         -> checkpoint saving OK
#   [5] re-run with same run_name                   -> auto-resume OK
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
        run_name=debug \
        model=deepwind_debug \
        data=train \
        data_eval=eval \
        training=default \
        training.max_steps=500 \
        training.logging_steps=10 \
        training.eval_steps=25 \
        training.save_steps=250 \
        training.per_device_train_batch_size=4 \
        training.gradient_accumulation_steps=1 \
        training.dataloader_num_workers=2 \
        training.report_to=none"

echo "Debug command: $CMD"

# ---------------------------------------------------------------------------
# 6. Launch via Singularity — srun node count also driven by SLURM env var
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
echo "===== Debug job finished at $(date) with exit code $EXIT_CODE ====="
exit $EXIT_CODE