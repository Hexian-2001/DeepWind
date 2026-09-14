"""
finetune.py
Entry point for few-shot fine-tuning of DeepWindModel via LoRA (PEFT).

The previous hand-rolled DDP training loop is replaced with the same
industrial-grade stack used by ``train.py``:

    * Hydra config (``configs/finetune.yaml``)
    * ``DeepWindTrainer`` (handles loss extraction for embedded targets)
    * WandB experiment tracking
    * the dataset registry (``deepwind_finetune``)

Fine-tuning strategy (unchanged from the original implementation):
    * LoRA on the variate-attention QKV / output projections (r=16, α=32)
    * full fine-tuning of the MoE router and the prediction head
    * AdamW lr=1e-4, grad-clip 1.0, cosine schedule

The foundation checkpoint is loaded with ``DeepWindModel.from_pretrained`` and
only the adapter (+ router/head) weights are written back by ``Trainer``.
"""
from __future__ import annotations

import atexit
import logging
import os

import hydra
import torch.distributed as dist
import transformers
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import (
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.data.datasets import *
from src.models.configuration import *
from src.models.deepwind import *
from src.models.adapter import apply_finetune_strategy
from src.utils.distributed import cleanup_ddp, is_main_process
from src.utils.registry import DATASET_REGISTRY
from src.utils.trainer import DeepWindTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup helpers (shared semantics with train.py)
# ---------------------------------------------------------------------------


def _setup_logging(cfg: DictConfig) -> None:
    """Configure transformers logging for rank-0 only, suppressing duplicates."""
    raw_level: str = (
        cfg.get("training", {}).get("log_level", "info") or "info"
    )
    log_level: int = logging.getLevelName(raw_level.upper())
    if not isinstance(log_level, int):
        log_level = logging.INFO

    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.logging.get_logger("transformers").propagate = False


def _setup_wandb(cfg: DictConfig) -> None:
    """Initialise a wandb run on rank 0 (see train.py for rationale)."""
    wandb_cfg = cfg.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", cfg.project_name),
        name=wandb_cfg.get("name", cfg.run_name),
        id=wandb_cfg.get("id", cfg.run_name),
        tags=list(wandb_cfg.get("tags") or []),
        notes=wandb_cfg.get("notes") or "",
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
    logger.info("wandb run: %s", wandb.run.url)


def _detect_checkpoint(
    output_dir: str,
    training_args: TrainingArguments,
) -> str | None:
    """Return the path to the last checkpoint, or ``None`` (see train.py)."""
    if not os.path.isdir(output_dir):
        return None
    if training_args.overwrite_output_dir:
        return None

    last_ckpt = get_last_checkpoint(output_dir)
    if last_ckpt and training_args.resume_from_checkpoint is None:
        if is_main_process():
            logger.info(
                "Checkpoint detected: %s. Resuming fine-tuning. "
                "Set overwrite_output_dir=true to start fresh.",
                last_ckpt,
            )
    return last_ckpt


# ---------------------------------------------------------------------------
# Model / dataset factories
# ---------------------------------------------------------------------------


def _build_model(cfg: DictConfig):
    """Load the foundation checkpoint and inject the LoRA adapter.

    Returns
    -------
    (model, context_length)
        ``model`` is a ``PeftModel`` ready for training; ``context_length`` is
        read from the *loaded* checkpoint so the dataset window always matches
        the foundation model's patching (not the YAML default).
    """
    pretrained_path = cfg.finetune.pretrained_path
    if is_main_process():
        logger.info("Loading foundation model from %s", pretrained_path)

    base_model = DeepWindModel.from_pretrained(pretrained_path)
    context_length = base_model.config.context_length

    model = apply_finetune_strategy(
        base_model,
        lora_r=cfg.finetune.lora_r,
        lora_alpha=cfg.finetune.lora_alpha,
    )
    return model, context_length


def _build_train_dataset(cfg: DictConfig, context_length: int):
    """Instantiate the few-shot dataset from the registry."""
    DatasetClass = DATASET_REGISTRY.get(cfg.train_dataset_name)
    data_args: dict = OmegaConf.to_container(cfg.data, resolve=True)
    # Authoritative context length comes from the loaded checkpoint.
    data_args["context_length"] = context_length

    if is_main_process():
        logger.info(
            "Finetune dataset: %s (context_length=%d)",
            cfg.train_dataset_name,
            context_length,
        )
    return DatasetClass(**data_args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    atexit.register(cleanup_ddp)
    OmegaConf.resolve(cfg)

    # Silence wandb on non-main processes before any import side-effects.
    if not is_main_process():
        os.environ["WANDB_MODE"] = "disabled"

    # 1. Logging & reproducibility
    _setup_logging(cfg)
    set_seed(cfg.seed)

    if is_main_process():
        logger.info("Finetune run: %s", cfg.run_name)
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # 2. Training arguments & checkpoint detection
    training_args = TrainingArguments(
        **OmegaConf.to_container(cfg.training, resolve=True)
    )
    last_checkpoint = _detect_checkpoint(training_args.output_dir, training_args)

    # 3. wandb (rank-0 only)
    if is_main_process() and training_args.report_to and "wandb" in training_args.report_to:
        _setup_wandb(cfg)

    # 4. Model & dataset
    model, context_length = _build_model(cfg)
    train_dataset = _build_train_dataset(cfg, context_length)

    # 5. Trainer
    trainer = DeepWindTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    # 6. Synchronise all ranks before training begins
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if is_main_process():
        logger.info("Starting fine-tuning...")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # 7. Persist artefacts (adapter + router/head only, via PeftModel)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if is_main_process():
        logger.info("Fine-tuning completed.")
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
