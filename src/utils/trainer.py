"""
deepwind_trainer.py

Custom HuggingFace Trainer for the DeepWind model.

Rationale
---------
HuggingFace Trainer.prediction_step() gates loss computation on the presence of
a standard ``labels`` key inside `inputs` (via ``self.label_names``).  DeepWind
embeds its targets inside the input dict (e.g. ``context`` / ``target``), so the
default gate never fires and ``eval_loss`` is always ``None``.

This subclass overrides ``prediction_step`` to perform an unconditional forward
pass and extract loss / logits from whatever return type the model produces.

Supported model output types
-----------------------------
* ``ModelOutput`` subclass  (e.g. ``DeepWindOutput``)
* ``dict``
* ``tuple``   – convention: ``(loss, logits, ...)``
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Trainer, __version__ as _transformers_version
from packaging import version

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version guard: use_legacy_prediction_loop was removed in Transformers 4.30.
# ---------------------------------------------------------------------------
_LEGACY_LOOP_SUPPORTED: bool = version.parse(_transformers_version) < version.parse("4.30.0")


class DeepWindTrainer(Trainer):
    """Trainer subclass that forces loss computation during evaluation.

    Standard :class:`~transformers.Trainer` skips loss in ``prediction_step``
    when ``inputs`` contains no recognised label keys.  DeepWind stores targets
    inside ``inputs`` under model-specific keys, so we bypass that gate entirely
    and extract loss from the model's own output.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform one evaluation step, always computing loss.

        Parameters
        ----------
        model:
            The model under evaluation.
        inputs:
            Batch dict produced by the ``DataCollator``.  For DeepWind this
            includes embedded target tensors; there is no external ``labels`` key.
        prediction_loss_only:
            When ``True`` logits are discarded before returning.
        ignore_keys:
            Output keys to exclude when filtering ``ModelOutput`` fields.
            Forwarded to the parent for completeness; unused in the non-legacy
            path because we extract fields by name rather than by exclusion.

        Returns
        -------
        loss : torch.Tensor | None
            Scalar loss detached from the computation graph, or ``None`` if the
            model did not return one.
        logits : torch.Tensor | None
            Raw predictions, or ``None`` when ``prediction_loss_only=True``.
        labels : None
            Always ``None``; DeepWind has no external label tensor to return.
        """
        # ------------------------------------------------------------------ #
        # 1. Legacy TPU path – delegate entirely to the parent.               #
        #    Preserved only for environments that pre-date Transformers 4.30.  #
        # ------------------------------------------------------------------ #
        if _LEGACY_LOOP_SUPPORTED and self.args.use_legacy_prediction_loop:
            logger.debug(
                "DeepWindTrainer: falling back to legacy prediction loop. "
                "Note – this path re-introduces the label-gate and may "
                "suppress loss computation."
            )
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )

        # ------------------------------------------------------------------ #
        # 2. Standard path                                                     #
        # ------------------------------------------------------------------ #
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)

        loss, logits = self._extract_loss_and_logits(outputs)
        loss = self._reduce_loss(loss)

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_loss_and_logits(
        outputs: Any,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract ``(loss, logits)`` from any supported output type.

        Handles three cases in priority order:

        1. ``ModelOutput`` / object with attributes
        2. ``dict``
        3. ``tuple``  – assumes ``(loss, logits, ...)`` convention
        """
        loss: Optional[torch.Tensor] = None
        logits: Optional[torch.Tensor] = None

        # --- Case 1: attribute-based (ModelOutput or custom dataclass) ---
        if hasattr(outputs, "loss"):
            loss = outputs.loss

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif hasattr(outputs, "denorm_quantile_preds"):
                # DeepWind-specific prediction tensor.
                logits = outputs.denorm_quantile_preds
            else:
                logger.debug(
                    "DeepWindTrainer: model output has 'loss' but no recognised "
                    "logits attribute ('logits' or 'denorm_quantile_preds'). "
                    "Returning logits=None."
                )

        # --- Case 2: dict ---
        elif isinstance(outputs, dict):
            loss = outputs.get("loss")
            logits = outputs.get("logits")

        # --- Case 3: tuple – (loss, logits?, ...) ---
        elif isinstance(outputs, tuple):
            if len(outputs) < 1:
                raise ValueError(
                    "Model returned an empty tuple. Expected at least a loss tensor."
                )
            loss = outputs[0]
            logits = outputs[1] if len(outputs) > 1 else None

        else:
            raise TypeError(
                f"Unsupported model output type: {type(outputs)!r}. "
                "Expected a ModelOutput subclass, dict, or tuple."
            )

        return loss, logits

    @staticmethod
    def _reduce_loss(loss: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Mean-reduce and detach loss, handling both scalar and vector outputs.

        Under DistributedDataParallel each replica may return a per-sample loss
        vector of shape ``(batch_size,)``.  Taking the mean produces a scalar
        that is consistent with single-device training.
        """
        if loss is None:
            return None

        if loss.ndim > 0:
            # Vector loss (DDP) → scalar.
            loss = loss.mean()

        return loss.detach()