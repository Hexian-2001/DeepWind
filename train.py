import logging
import os
import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Optional

import torch
import torch.distributed as dist
import transformers
from transformers import (
    Trainer, 
    TrainingArguments, 
    set_seed,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

from src.utils.registry import MODEL_REGISTRY, CONFIG_REGISTRY, DATASET_REGISTRY
from src.utils.trainer import DeepWindTrainer
from src.utils.distributed import is_main_process

import src.models.deepwind
import src.models.configuration
import src.data.datasets 


# Setup Logger
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # =========================================================================
    # 1. Environment & Logging Setup
    # =========================================================================
    # Resolve the config to static container
    # (Hydra uses LazyConfig by default, we resolve it to verify interpolation)
    OmegaConf.resolve(cfg)
    
    log_level = logging.INFO 
    if cfg.get("training") and cfg.training.get("log_level"):
        log_level = transformers.utils.logging.log_levels.get(cfg.training.log_level, logging.INFO)

    transformers.utils.logging.disable_default_handler()  
    transformers.utils.logging.set_verbosity(log_level)
    transformers.logging.get_logger("transformers").propagate = False
    
    # Set Seed for Reproducibility
    set_seed(cfg.seed)

    # =========================================================================
    # 2. Checkpoint Auto-Resume Logic (The Solution)
    # =========================================================================
    output_dir = cfg.training.output_dir
    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    training_args = TrainingArguments(**training_args_dict)

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            if is_main_process():
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. "
                    "To avoid this behavior, change the 'run_name' or set 'overwrite_output_dir' to True"
                )

    # =========================================================================
    # 3. Model & Config Initialization (via Registry)
    # =========================================================================
    if is_main_process():
        logger.info(f"Initializing Model: {cfg.model_name}")
    
    # Retrieve Classes
    ConfigClass = CONFIG_REGISTRY.get(cfg.model_name)
    ModelClass = MODEL_REGISTRY.get(cfg.model_name)
    
    # Convert Hydra config to Dict
    model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    config = ConfigClass(**model_config_dict)

    # Instantiate Model
    # Note: If resuming, Trainer will load weights from checkpoint later.
    # Here we just initialize the architecture.
    model = ModelClass(config)

    # Log Parameter Count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info(f"Model Parameters: {trainable_params / 1e6:.2f} M")

    # =========================================================================
    # 4. Dataset Initialization (via Registry)
    # =========================================================================
    if is_main_process():
        logger.info(f"Initializing Train Dataset: {cfg.train_dataset_name}")
    TrainDatasetClass = DATASET_REGISTRY.get(cfg.train_dataset_name)
    train_data_args = OmegaConf.to_container(cfg.data, resolve=True)
    train_dataset = TrainDatasetClass(**train_data_args)
    
    eval_dataset = None
    
    if "data_val" in cfg:
        eval_cls_name = cfg.eval_dataset_name
        if is_main_process():
            logger.info(f"Initializing Valid Dataset: {eval_cls_name}")
        EvalDatasetClass = DATASET_REGISTRY.get(eval_cls_name)
        
        if EvalDatasetClass is None:
             raise ValueError(f"Dataset '{eval_cls_name}' not found in registry!")
        
        eval_data_args = OmegaConf.to_container(cfg.data_val, resolve=True)
        eval_dataset = EvalDatasetClass(**eval_data_args)
    else:
        if is_main_process():
            logger.warning("No validation config found. Skipping evaluation.")

    trainer = DeepWindTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # =========================================================================
    # 6. Start Training
    # =========================================================================
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if is_main_process():
        logger.info("Starting Training...")

    if dist.is_initialized():
        if is_main_process():
            print(f"Rank {dist.get_rank()} is waiting at the barrier before loading checkpoint...")
        dist.barrier()

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # =========================================================================
    # 7. Final Saving & Metrics
    # =========================================================================
    if is_main_process():
        logger.info(f"Saving model to {training_args.output_dir}")

    trainer.save_model()  # Saves the model/tokenizer
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if is_main_process():
        logger.info("Training Completed Successfully.")

if __name__ == "__main__":
    main()