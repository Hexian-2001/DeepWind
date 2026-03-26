#!/bin/bash --login
#SBATCH --job-name=deepwind_large_v5
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --account=pawsey0115-gpu
#SBATCH --output=/scratch/pawsey0115/hwang4/deepwind/logs/%x-%j.out
#SBATCH --error=/scratch/pawsey0115/hwang4/deepwind/logs/%x-%j.err

echo "Job started on $(hostname) at $(date)"

# 1. Load Host Modules
module load pytorch/2.7.1-rocm6.3.3 
module load cray-python 

export SINGULARITYENV_NCCL_COMM_BLOCKING=1
export SINGULARITYENV_NCCL_ASYNC_ERROR_HANDLING=1
# 2. Critical: Unified Container Environment Variables
# Using SINGULARITYENV_ prefix ensures these are inherited inside the container
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_NCCL_DEBUG=INFO
# export SINGULARITYENV_NCCL_IB_DISABLE=1         # Disable IB for single-node stability
export SINGULARITYENV_NCCL_P2P_DISABLE=1        # REQUIRED: Bypasses IOMMU hang issue on this cluster
export SINGULARITYENV_NCCL_BUFFSIZE=2097152
export SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0   # Specify correct high-speed network interface
export SINGULARITYENV_TORCH_DISTRIBUTED_DEBUG=INFO
export SINGULARITYENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SINGULARITYENV_TORCH_NCCL_BLOCKING_WAIT=1
export SINGULARITYENV_TORCH_DISTRIBUTED_TIMEOUT=7200
#export OMP_NUM_THREADS=2
export SINGULARITYENV_NCCL_NSOCKS_PERTHREAD=4
export SINGULARITYENV_NCCL_SOCKET_NTHREADS=2

# 3. Compilation & Cache Settings
export DS_BUILD_AIO=0
export LIBAIO_DISABLE=1
export TORCH_EXTENSIONS_DIR="/scratch/pawsey0115/hwang4/.torch_extensions_cache"
mkdir -p $TORCH_EXTENSIONS_DIR
     
# 4. Distributed Master Setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "Master Node: $MASTER_ADDR:$MASTER_PORT"

# 5. Path & Workspace Setup
export MYENV=time-moe
export VENV_PATH=/software/projects/pawsey0115/hwang4/manual/software/pythonEnvironments/pytorchContainer-environments/${MYENV}
export PROJECT_ROOT="/scratch/pawsey0115/hwang4/project_codes/deepwind_research"
export SCRATCH_ROOT="/scratch/pawsey0115/hwang4/deepwind_experiments"
#export WANDB_DIR=$SCRATCH_ROOT/wandb_cache
#export SINGULARITYENV_WANDB_DIR=$WANDB_DIR
#mkdir -p $WANDB_DIR

echo "PYTHONPATH will be: $PROJECT_ROOT:$PYTHONPATH"

# 6. Training Command
CMD="export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
     ${VENV_PATH}/bin/python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $PROJECT_ROOT/train.py \
    model=deepwind \
    model.d_model=1024 \
    model.num_layers=18 \
    model.d_ff=2816 \
    model.num_heads=16 \
    model.num_experts=8 \
    model.num_experts_per_token=2 \
    model.dropout=0.1 \
    model.use_arcsinh=true \
    model.aux_loss_weight=0.01 \
    model.use_load_balance_loss=true \
    model.use_rotary_emb=false \
    model.use_moe=true \
    model.use_coord_embed=true \
    model.use_variate_embed=true \
    model.variate_atten_every_n_layers=2 \
    data=wind_full \
    data_val=wind_val \
    training=default \
    training.learning_rate=1e-4 \
    training.per_device_train_batch_size=4 \
    training.gradient_accumulation_steps=4 \
    training.logging_steps=50 \
    training.eval_steps=5000 \
    training.save_steps=5000 \
    training.dataloader_num_workers=4 \
    training.max_steps=100000 \
    run_name=deepwind_large_v5"

echo "Command: $CMD"

# 7. Execution with Singularity
BIND_PATHS="$PROJECT_ROOT,$VENV_PATH,$SCRATCH_ROOT,$TORCH_EXTENSIONS_DIR"
     
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES \
     --cpu-bind=none \
     singularity exec --rocm \
     --bind $BIND_PATHS \
     $SINGULARITY_CONTAINER \
     bash -c "$CMD"

echo "Job finished at $(date)"
