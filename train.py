from __future__ import annotations

import atexit
import logging
import os
import signal
import time

import hydra
import torch.distributed as dist
import transformers
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import (
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.data.datasets import *  
from src.models.configuration import *  
from src.models.deepwind import *  
from src.utils.distributed import cleanup_ddp, is_main_process
from src.utils.provenance import write_run_info
from src.utils.registry import CONFIG_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY
from src.utils.trainer import DeepWindTrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown on SIGTERM (preemption / wall-clock kill)
# ---------------------------------------------------------------------------
# Slurm sends SIGTERM when a job hits its wall-clock limit or is preempted.
# HuggingFace Trainer does NOT save on SIGTERM by default, so we record the
# signal here and a callback saves a resumable checkpoint at the next step
# boundary before stopping. This lets the chunked gpu-dev jobs (no --requeue)
# and preempted 24h jobs resume from the last completed step.
_shutdown_requested = False


def _request_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning(
        "Received signal %d; saving a checkpoint at the next step boundary "
        "and stopping.",
        signum,
    )


class _GracefulSaveCallback(TrainerCallback):
    """Save a checkpoint and stop at the next step boundary after SIGTERM.

    Setting ``control.should_save`` makes the Trainer's post-step
    ``_maybe_log_save_evaluate`` write ``checkpoint-<global_step>`` (rank 0
    only, like a normal ``save_steps`` save), and ``should_training_stop``
    ends the loop right afterwards.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if not _shutdown_requested:
            return
        control.should_save = True
        control.should_training_stop = True


class _TimeBudgetSaveCallback(TrainerCallback):
    """Save + stop once a wall-clock deadline (epoch seconds) is reached.

    This is the reliable replacement for the SIGTERM graceful save: instead of
    racing a signal against Slurm's SIGKILL window (a Base/Large 10 GB
    checkpoint cannot be written within the 12-30 s SIGTERM->SIGKILL window;
    validated on job 49418245), we trigger a normal save_steps-style checkpoint
    during ordinary training flow, well before the wall-clock limit. The
    deadline is read from the ``DEEPWIND_SAVE_DEADLINE`` env var (epoch seconds;
    0 / unset = disabled).
    """

    def __init__(self, deadline: float):
        self._deadline = float(deadline)
        self._fired = False

    def on_step_end(self, args, state, control, **kwargs):
        if self._fired or self._deadline <= 0:
            return
        if time.time() >= self._deadline:
            self._fired = True
            logger.warning(
                "Wall-clock save deadline reached at global_step=%d; saving a "
                "checkpoint and stopping.",
                state.global_step,
            )
            control.should_save = True
            control.should_training_stop = True


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

    # 2b. Provenance manifest (rank-0 only; written before any training state)
    if is_main_process():
        write_run_info(training_args.output_dir, cfg)

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

    # 5b. Graceful shutdown: save + stop at the next step boundary on SIGTERM
    #     (Slurm wall-clock kill / preemption). torchrun's elastic agent traps
    #     SIGTERM and forwards it to the workers, which then save within its 30s
    #     close() grace window. SIGUSR1 is NOT used: Slurm delivers a non-SIGTERM
    #     signal to the batch shell / torchrun first, which die under the default
    #     action and let srun kill the workers before the save completes
    #     (validated on job 49413237).
    trainer.add_callback(_GracefulSaveCallback())
    signal.signal(signal.SIGTERM, _request_shutdown)

    # 5c. Wall-clock save: save + stop once DEEPWIND_SAVE_DEADLINE (epoch
    #     seconds) is reached. Reliable replacement for the SIGTERM save, which
    #     cannot finish a Base/Large 10 GB checkpoint within the 12-30 s
    #     SIGTERM->SIGKILL window (validated on job 49418245).
    _save_deadline = float(os.environ.get("DEEPWIND_SAVE_DEADLINE", "0") or "0")
    if _save_deadline > 0:
        trainer.add_callback(_TimeBudgetSaveCallback(_save_deadline))

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
