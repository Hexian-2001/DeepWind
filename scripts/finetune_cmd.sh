#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=8

MASTER_PORT=29505

PYTHON_SCRIPT="/scratch/pawsey0115/hwang4/project_codes/deepwind_research/finetune.py"
# gefc14_wind_10
# 30651
# 43458
# 75354
# 76016
SITE_NAME="76016"
DATA_PATH="/scratch/pawsey0115/hwang4/deepwindData/test/76016.npy"
METADATA_PATH="/scratch/pawsey0115/hwang4/deepwindData/train_metadata.csv"
PRETRAINED_PATH="/scratch/pawsey0115/hwang4/deepwind_experiments/checkpoints/deepwind_patch_small_v0"
OUTPUT_DIR="/scratch/pawsey0115/hwang4/deepwind_experiments/checkpoints/deepwind_small_finetune_50"
    
EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=2e-4
STRIDE=8
FINETUNE_RATE=0.5 

echo "Starting DDP Fine-tuning with $GPUS GPUs..."
echo "Total Global Batch Size: $((BATCH_SIZE * GPUS))"

python -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --master_port=$MASTER_PORT \
    $PYTHON_SCRIPT \
    --site_name "$SITE_NAME" \
    --data_path "$DATA_PATH" \
    --metadata_path "$METADATA_PATH" \
    --pretrained_path "$PRETRAINED_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --stride $STRIDE \
    --finetune_rate $FINETUNE_RATE

echo "Fine-tuning completed!"