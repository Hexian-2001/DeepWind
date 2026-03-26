#!/bin/bash

# 1. Environment Setup
module load pytorch/2.7.1-rocm6.3.3 
module load cray-python 

# 2. Critical: Unified Container Environment Variables
# Bypasses the IOMMU hang issues observed on the compute nodes
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_NCCL_DEBUG=INFO
export SINGULARITYENV_NCCL_IB_DISABLE=1         # Stabilizes single-node runs
export SINGULARITYENV_NCCL_P2P_DISABLE=1        # REQUIRED: Fixes the 0.0W PowerCap/Hang issue
export SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0   # Setonix high-speed interconnect
export SINGULARITYENV_TORCH_DISTRIBUTED_DEBUG=INFO
export SINGULARITYENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SINGULARITYENV_TORCH_DISTRIBUTED_TIMEOUT=7200

# 3. Path & Cache Configuration
export MYENV=time-moe
export VENV_PATH=/software/projects/pawsey0115/hwang4/manual/software/pythonEnvironments/pytorchContainer-environments/${MYENV}
export PROJECT_ROOT="/scratch/pawsey0115/hwang4/project_codes/deepwind_research"
export SCRATCH_ROOT="/scratch/pawsey0115/hwang4/deepwind_experiments"
export TORCH_EXTENSIONS_DIR="/scratch/pawsey0115/hwang4/.torch_extensions_cache"

export WANDB_DIR=$SCRATCH_ROOT/wandb_cache
export SINGULARITYENV_WANDB_DIR=$WANDB_DIR
mkdir -p $WANDB_DIR $TORCH_EXTENSIONS_DIR

# 4. DDP Parameters
GPUS=8
MASTER_ADDR=localhost
MASTER_PORT=29505
# Use a random ID for rdzv if not in SLURM
RDZV_ID=${SLURM_JOB_ID:-$RANDOM}

echo "Starting Interactive Training on $(hostname) with $GPUS GPUs..."

# 5. The Training Command
CMD="export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
     ${VENV_PATH}/bin/python -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --nnodes=1 \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $PROJECT_ROOT/train.py \
    model=deepwind \
    model.d_model=384 \
    model.num_layers=6 \
    model.d_ff=1024 \
    model.num_heads=6 \
    model.num_experts=8 \
    model.num_experts_per_token=2 \
    model.dropout=0.05 \
    model.use_arcsinh=true \
    model.aux_loss_weigh=0.02 \
    model.use_load_balance_loss=true \
    data=wind_full \
    data_val=wind_val \
    training=default \
    training.learning_rate=2e-4 \
    training.per_device_train_batch_size=8 \
    training.gradient_accumulation_steps=4 \
    training.logging_steps=50 \
    training.eval_steps=5000 \
    training.save_steps=5000 \
    training.dataloader_num_workers=8 \
    run_name=deepwind_interactive_moe_v1"

# 6. Execution with Singularity
# Note: In interactive mode, we don't use 'srun' for the command itself
BIND_PATHS="$PROJECT_ROOT,$VENV_PATH,$SCRATCH_ROOT,$TORCH_EXTENSIONS_DIR"

singularity exec --rocm \
     --bind $BIND_PATHS \
     $SINGULARITY_CONTAINER \
     bash -c "$CMD"

echo "Interactive session task finished."