#!/bin/bash

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/pawsey0115/hwang4/project_codes/deepwind_research"
LOG_DIR="/scratch/pawsey0115/hwang4/results/DeepWind-Research/logs/eval"

# ── Config ────────────────────────────────────────────────────────────────────
GPUS=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p $LOG_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

# ── Launch ────────────────────────────────────────────────────────────────────
cd $PROJECT_DIR

echo "=========================================="
echo "Start : $(date)"
echo "GPUs  : $GPUS"
echo "Log   : ${LOG_DIR}/eval_${TIMESTAMP}.out"
echo "=========================================="

python -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --standalone \
    evaluate.py \
    run_name=deepwind_small_scada_1_wtk_9_noleakage \
    model_name=deepwind_small_scada_1_wtk_9_noleakage \
    inference.mqd_infer=true \
    2>&1 | tee "${LOG_DIR}/eval_${TIMESTAMP}.out"

echo "=========================================="
echo "Done  : $(date)"
echo "=========================================="

# ── Collect results after evaluation ──────────────────────────────────────────
RESULTS_ROOT="/scratch/pawsey0115/hwang4/results/DeepWind-Research/results"

echo "Collecting results..."
python tools/collect_results.py \
    --results_root $RESULTS_ROOT \
    --output       ${RESULTS_ROOT}/summary/compare_all.csv

echo "Summary saved → ${RESULTS_ROOT}/summary/compare_all.csv"
