#!/bin/bash --login
#SBATCH --job-name=deepwind_comms_test
#SBATCH --partition=gpu-dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --account=pawsey0115-gpu
#SBATCH --output=/scratch/pawsey0115/hwang4/results/DeepWind-Research/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/pawsey0115/hwang4/results/DeepWind-Research/slurm-logs/%x-%j.err

echo "===== Comms test started on $(hostname) at $(date) ====="

# ---------------------------------------------------------------------------
# 1. Host modules
# ---------------------------------------------------------------------------
module load pytorch/2.7.1-rocm6.3.3
module load cray-python

echo "Container: $SINGULARITY_CONTAINER"

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
export PROJECT_ROOT="/scratch/pawsey0115/hwang4/project_codes/deepwind_research"
export SCRATCH_ROOT="/scratch/pawsey0115/hwang4"
export RESULTS_ROOT="$SCRATCH_ROOT/results"
export VENV_PATH="/software/projects/pawsey0115/hwang4/manual/software/pythonEnvironments/pytorchContainer-environments/time-moe"
export TORCH_EXTENSIONS_DIR="$SCRATCH_ROOT/.torch_extensions_cache"

mkdir -p "$RESULTS_ROOT/DeepWind-Research/slurm-logs"

# ---------------------------------------------------------------------------
# 3. Distributed setup
# ---------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "All nodes:   $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

# ---------------------------------------------------------------------------
# 4. Container environment
# ---------------------------------------------------------------------------
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_SCRATCH_ROOT="$SCRATCH_ROOT"

export SINGULARITYENV_NCCL_DEBUG=INFO
export SINGULARITYENV_NCCL_P2P_DISABLE=1
export SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0
export SINGULARITYENV_NCCL_BUFFSIZE=2097152
export SINGULARITYENV_NCCL_NSOCKS_PERTHREAD=4
export SINGULARITYENV_NCCL_SOCKET_NTHREADS=2

export SINGULARITYENV_TORCH_DISTRIBUTED_DEBUG=DETAIL
export SINGULARITYENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SINGULARITYENV_TORCH_NCCL_BLOCKING_WAIT=1
export SINGULARITYENV_TORCH_DISTRIBUTED_TIMEOUT=300

export DS_BUILD_AIO=0
export LIBAIO_DISABLE=1
export SINGULARITYENV_WANDB_MODE="disabled"

# ---------------------------------------------------------------------------
# 5. Write comm test script to shared scratch (NOT /tmp — /tmp is node-local
#    and invisible to other nodes inside the container)
#
# Tests performed:
#   [1] dist.init_process_group  -> basic process group initialisation
#   [2] all_reduce (sum)         -> collective comms across all GPUs
#   [3] broadcast                -> rank 0 -> all ranks
#   [4] all_gather               -> each rank collects tensors from all ranks
#   [5] bandwidth (256MB tensor) -> catches slow inter-node links
# ---------------------------------------------------------------------------
TEST_SCRIPT="$SCRATCH_ROOT/comm_test_${SLURM_JOB_ID}.py"

cat > "$TEST_SCRIPT" << 'PYEOF'
import os
import time
import datetime
import torch
import torch.distributed as dist


def log(msg: str) -> None:
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node       = os.environ.get("SLURMD_NODENAME", "unknown")
    print(f"[{node} | rank={rank:02d} local={local_rank}] {msg}", flush=True)


def main() -> None:
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=300),
    )
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log(f"Process group initialised | world_size={world_size}")
    dist.barrier()

    # Test 1: all_reduce
    t = torch.ones(1, device=device) * rank
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    assert t.item() == expected, f"all_reduce FAILED: got {t.item()}, expected {expected}"
    log(f"[PASS] all_reduce  | sum={int(t.item())} (expected {expected})")
    dist.barrier()

    # Test 2: broadcast
    t = torch.zeros(1, device=device)
    if rank == 0:
        t.fill_(42.0)
    dist.broadcast(t, src=0)
    assert t.item() == 42.0, f"broadcast FAILED: got {t.item()}"
    log(f"[PASS] broadcast   | value={t.item()}")
    dist.barrier()

    # Test 3: all_gather
    send = torch.tensor([float(rank)], device=device)
    recv = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(recv, send)
    gathered = [r.item() for r in recv]
    assert gathered == list(range(world_size)), f"all_gather FAILED: {gathered}"
    log(f"[PASS] all_gather  | values={gathered}")
    dist.barrier()

    # Test 4: bandwidth — 256 MB all_reduce across all GPUs
    size_mb    = 256
    n_elements = (size_mb * 1024 * 1024) // 4  # float32
    t = torch.ones(n_elements, device=device)
    dist.barrier()
    t0 = time.perf_counter()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bw_gb   = (size_mb / 1024) / elapsed
    log(f"[PASS] bandwidth   | {size_mb}MB all_reduce in {elapsed:.3f}s (~{bw_gb:.2f} GB/s)")
    dist.barrier()

    if rank == 0:
        print("\n========================================", flush=True)
        print(f"  ALL COMM TESTS PASSED | world_size={world_size}", flush=True)
        print("========================================\n", flush=True)

    dist.destroy_process_group()


main()
PYEOF

echo "Comm test script: $TEST_SCRIPT"

# ---------------------------------------------------------------------------
# 6. Launch — bind SCRATCH_ROOT so all nodes can read the test script
# ---------------------------------------------------------------------------
BIND_PATHS="$PROJECT_ROOT,$VENV_PATH,$SCRATCH_ROOT,$TORCH_EXTENSIONS_DIR"

CMD="export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
    ${VENV_PATH}/bin/python -m torch.distributed.run \
        --nproc_per_node=8 \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $TEST_SCRIPT"

echo "Command: $CMD"

srun \
    -N "$SLURM_JOB_NUM_NODES" \
    -n "$SLURM_JOB_NUM_NODES" \
    --cpu-bind=none \
    singularity exec --rocm \
        --bind "$BIND_PATHS" \
        "$SINGULARITY_CONTAINER" \
        bash -c "$CMD"

EXIT_CODE=$?

# Clean up — only after all nodes have finished
rm -f "$TEST_SCRIPT"

if [ $EXIT_CODE -eq 0 ]; then
    echo "===== Comms test PASSED at $(date) ====="
else
    echo "===== Comms test FAILED (exit=$EXIT_CODE) at $(date) ====="
    echo "Diagnosis: check NCCL_DEBUG=INFO output above."
fi

exit $EXIT_CODE