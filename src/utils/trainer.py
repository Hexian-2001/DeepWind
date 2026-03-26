import torch
from transformers import Trainer
from typing import Dict, Union, Any, Tuple, Optional, List
from torch import nn
import time
from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback


class DeepWindTrainer(Trainer):
    """
    Custom Trainer for DeepWind model.
    
    This trainer overrides the `prediction_step` method to enforce loss calculation 
    during evaluation loops, bypassing the default Hugging Face behavior which 
    skips loss computation if 'labels' are missing from the inputs.
    """

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Args:
            model: The model to evaluate.
            inputs: The inputs and targets of the model.
            prediction_loss_only: Whether to return only the loss or also logits/labels.
            ignore_keys: Keys to ignore in the output (not used here but kept for signature compatibility).

        Returns:
            Tuple containing:
            - loss (torch.Tensor or None)
            - logits (torch.Tensor or None)
            - labels (torch.Tensor or None) - Always None in this implementation as Dataset has no labels.
        """
        
        # 1. Prepare inputs (move inputs to the correct device: GPU/TPU/CPU)
        inputs = self._prepare_inputs(inputs)

        # 2. Context manager for evaluation (disable gradient calculation)
        with torch.no_grad():
            if self.args.use_legacy_prediction_loop:
                # Fallback for legacy TPU setups (rarely used in modern PyTorch)
                loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            else:
                # --- CORE LOGIC: FORCE FORWARD PASS ---
                # We execute the model forward pass explicitly.
                # Unlike standard Trainer, we do not check `if has_labels:`
                outputs = model(**inputs)

                # --- LOSS EXTRACTION ---
                # Robustly extract loss regardless of return type (ModelOutput, Dict, or Tuple)
                loss = None
                logits = None

                # Case A: ModelOutput object (e.g., DeepWindOutput) or object with attributes
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    elif hasattr(outputs, "denorm_quantile_preds"):
                        # Specific support for your DeepWind model's prediction output
                        logits = outputs.denorm_quantile_preds
                
                # Case B: Dictionary output
                elif isinstance(outputs, dict):
                    loss = outputs.get("loss")
                    logits = outputs.get("logits")
                
                # Case C: Tuple output
                elif isinstance(outputs, tuple):
                    # Convention: (loss, logits, ...)
                    loss = outputs[0]
                    logits = outputs[1] if len(outputs) > 1 else None
            
                # --- LOSS PROCESSING ---
                # Handle Distributed Data Parallel (DDP) where loss might be a vector [batch_size]
                if loss is not None:
                    # If loss is a scalar (dim=0), mean() does nothing. 
                    # If it's a vector (dim=1), mean() reduces it.
                    loss = loss.mean().detach()

        # 3. Return results formatting
        if prediction_loss_only:
            return (loss, None, None)

        # Note: We return None for labels because your dataset does not provide external labels.
        # The model calculates loss internally using 'context' or 'target' inside inputs.
        return (loss, logits, None)

