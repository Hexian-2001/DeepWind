#!/bin/bash
export OMP_NUM_THREADS=2
GPUS=8
# Ensure the variable is set for the host shell
export SCRATCH_ROOT="/scratch/pawsey0115/hwang4/deepwind_experiments"
# Crucial: Pass it to the Singularity container
export SINGULARITYENV_SCRATCH_ROOT="$SCRATCH_ROOT"
# Distributed Training Execution
python -m torch.distributed.run --nproc_per_node=$GPUS --standalone train.py \
    model=deepwind \
    model.d_model=384 \
    model.num_layers=6 \
    model.d_ff=1024 \
    model.num_heads=6 \
    model.num_experts=4 \
    model.num_experts_per_token=2 \
    model.dropout=0.05 \
    model.use_arcsinh=true \
    model.aux_loss_weight=0.01 \
    model.use_load_balance_loss=true \
    data=wind_full \
    data_val=wind_val \
    training=default \
    training.learning_rate=2e-4 \
    training.per_device_train_batch_size=8 \
    training.gradient_accumulation_steps=1 \
    training.logging_steps=1 \
    training.eval_steps=5000 \
    training.save_steps=5000 \
    training.dataloader_num_workers=16 \
    training.ddp_find_unused_parameters=false \
    run_name=deepwind_small_ablation_without_moe
