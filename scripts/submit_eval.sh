#!/bin/bash --login
#SBATCH --job-name=deepwind_eval
#SBATCH --partition=gpu-dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --account=pawsey0115-gpu
#SBATCH --output=/scratch/pawsey0115/hwang4/deepwind/logs/%x-%j.out
#SBATCH --error=/scratch/pawsey0115/hwang4/deepwind/logs/%x-%j.err

echo "Job started on $(hostname) at $(date)"

# 1. Load Host Modules

module load pytorch/2.7.1-rocm6.3.3 
module load cray-python 

# 2. Network & Debugging 
# -----------------------------------------------------------
export PYTHONUNBUFFERED=1                 
export TORCH_DISTRIBUTED_DEBUG=OFF          
export NCCL_DEBUG=WARN                    

export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn0
export TP_SOCKET_IFNAME=hsn0

#export NCCL_P2P_DISABLE=1                   
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_TIMEOUT=7200
     
# 3. Compilation Fix
export DS_BUILD_AIO=0
export LIBAIO_DISABLE=1
export TORCH_EXTENSIONS_DIR="/scratch/pawsey0115/hwang4/.torch_extensions_cache"
mkdir -p $TORCH_EXTENSIONS_DIR

# 4. Master Node Setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "Master Node: $MASTER_ADDR:$MASTER_PORT"

# 5. Paths & WANDB 
export MYENV=time-moe
export VENV_PATH=/software/projects/pawsey0115/hwang4/manual/software/pythonEnvironments/pytorchContainer-environments/${MYENV}
export PROJECT_ROOT="/scratch/pawsey0115/hwang4/project_codes/deepwind_research"
export SCRATCH_ROOT="/scratch/pawsey0115/hwang4/deepwind_experiments"

     
# 6. Command
echo "PYTHONPATH will be: $PROJECT_ROOT:$PYTHONPATH"

CMD="export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
     ${VENV_PATH}/bin/python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $PROJECT_ROOT/evaluate.py \
    run_name=deepwind_large_v5 \
    data.context_length=8192 \
    data.stride=null \
    inference.mqd_infer=true \
    inference.batch_size=8 \
    inference.num_workers=16 \
    inference.checkpoint_path="/scratch/pawsey0115/hwang4/deepwind_experiments/checkpoints/deepwind_large_v5" \
    inference.adapter_path=null \
    output.save_raw_results=true"      
echo "Command: $CMD"

# 7. Execution
BIND_PATHS="$PROJECT_ROOT,$VENV_PATH,$SCRATCH_ROOT,$TORCH_EXTENSIONS_DIR"

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES \
     --cpus-per-task=64 --cpu-bind=none \
     singularity exec --rocm \
     --bind $BIND_PATHS \
     --env WANDB_MODE=offline \
     $SINGULARITY_CONTAINER \
     bash -c "$CMD"

echo "Job finished at $(date)"
