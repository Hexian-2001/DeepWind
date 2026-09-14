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
from src.utils.distributed import cleanup_ddp, is_main_process
from src.utils.registry import CONFIG_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY
from src.utils.trainer import DeepWindTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup helpers
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
    """Initialise a wandb run on rank 0.

    Passes the full resolved config as hyperparameters so every run is
    self-documenting.  ``resume="allow"`` pairs with auto-checkpoint
    detection: if a run with the same ``id`` already exists on the wandb
    server the metrics will be appended rather than duplicated.

    This function is a no-op on non-main processes because
    ``WANDB_MODE=disabled`` is set for them in ``main()``.
    """
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


# ---------------------------------------------------------------------------
# Checkpoint detection
# ---------------------------------------------------------------------------


def _detect_checkpoint(
    output_dir: str,
    training_args: TrainingArguments,
) -> str | None:
    """Return the path to the last checkpoint, or ``None``.

    A checkpoint is only reported when:
    * ``output_dir`` exists and contains a valid checkpoint, AND
    * ``overwrite_output_dir`` is ``False``, AND
    * ``resume_from_checkpoint`` was not explicitly set by the user
      (in which case the user-supplied value takes precedence).
    """
    if not os.path.isdir(output_dir):
        return None
    if training_args.overwrite_output_dir:
        return None

    last_ckpt = get_last_checkpoint(output_dir)
    if last_ckpt and training_args.resume_from_checkpoint is None:
        if is_main_process():
            logger.info(
            "Checkpoint detected: %s. "
            "Resuming training. Set overwrite_output_dir=true to start fresh.",
            last_ckpt,
            )
    return last_ckpt


# ---------------------------------------------------------------------------
# Model / dataset factories
# ---------------------------------------------------------------------------


def _build_model(cfg: DictConfig):
    """Instantiate model from registry using the Hydra model config."""
    ConfigClass = CONFIG_REGISTRY.get(cfg.model_name)
    ModelClass = MODEL_REGISTRY.get(cfg.model_name)

    model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    config = ConfigClass(**model_config_dict)
    model = ModelClass(config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info("Model: %s | Trainable parameters: %.2fM", cfg.model_name, n_params / 1e6)

    return model


def _build_train_dataset(cfg: DictConfig):
    """Instantiate the training dataset from the registry."""
    DatasetClass = DATASET_REGISTRY.get(cfg.train_dataset_name)
    data_args = OmegaConf.to_container(cfg.data, resolve=True)
    if is_main_process():
        logger.info("Train dataset: %s", cfg.train_dataset_name)
    return DatasetClass(**data_args)


def _build_eval_dataset(cfg: DictConfig):
    """Instantiate eval dataset(s) from the registry.

    Returns a ``Dict[str, Dataset]`` when ``eval_groups`` is configured so
    that HF Trainer emits per-group metrics (e.g. ``eval_windtoolkit/loss``,
    ``eval_scada/loss``).  Returns a single ``Dataset`` for a combined eval,
    or ``None`` when no eval config is present.
    """
    if "data_eval" not in cfg:
        logger.warning("No data_eval config found — skipping evaluation.")
        return None

    EvalDatasetClass = DATASET_REGISTRY.get(cfg.eval_dataset_name)
    if EvalDatasetClass is None:
        raise ValueError(
            f"Eval dataset '{cfg.eval_dataset_name}' not found in DATASET_REGISTRY."
        )

    eval_args: dict = OmegaConf.to_container(cfg.data_eval, resolve=True)
    eval_groups: list[str] | None = eval_args.pop("eval_groups", None)

    if eval_groups:
        eval_dataset = {}
        for group in eval_groups:
            eval_dataset[group] = EvalDatasetClass(**eval_args, filter_dataset=group)
            if is_main_process():
                logger.info("Eval dataset [%s]: %d samples", group, len(eval_dataset[group]))
        return eval_dataset

    ds = EvalDatasetClass(**eval_args)
    if is_main_process():
        logger.info("Eval dataset: %d samples", len(ds))
    return ds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # Hugging Face initialises the process group lazily. Register cleanup before
    # any later failure so normal and exceptional exits both release RCCL state.
    atexit.register(cleanup_ddp)
    OmegaConf.resolve(cfg)

    # ── Silence all wandb output on non-main processes before any import
    #    side-effects can create a spurious run. ──────────────────────────────
    if not is_main_process():
        os.environ["WANDB_MODE"] = "disabled"
    
    # 1. Logging & reproducibility
    _setup_logging(cfg)
    set_seed(cfg.seed)

    if is_main_process():
        logger.info("Run: %s", cfg.run_name)
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # 2. Training arguments & checkpoint detection
    training_args = TrainingArguments(
        **OmegaConf.to_container(cfg.training, resolve=True)
    )
    last_checkpoint = _detect_checkpoint(training_args.output_dir, training_args)

    # 3. wandb (rank-0 only; other ranks are already disabled via env var)
    if is_main_process() and training_args.report_to and "wandb" in training_args.report_to:
        _setup_wandb(cfg)

    # 4. Model & datasets
    model = _build_model(cfg)
    train_dataset = _build_train_dataset(cfg)
    eval_dataset = _build_eval_dataset(cfg)

    # 5. Trainer
    trainer = DeepWindTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # 6. Synchronise all ranks before training begins
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if is_main_process():
        logger.info("Starting training...")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # 7. Persist artefacts
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if is_main_process():
        logger.info("Training completed.")
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
