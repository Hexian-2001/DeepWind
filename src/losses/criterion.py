import torch
import torch.nn as nn
from dataclasses import dataclass
from src.utils.distributed import is_main_process

@dataclass
class CriterionOutput:
    total_loss: torch.Tensor
    main_loss: torch.Tensor
    aux_loss: torch.Tensor

class DeepWindCriterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("quantiles", torch.tensor(config.quantiles))
        self.aux_weight = getattr(config, "aux_loss_weight", 0.0)

    def forward(self, pred_quantiles, target_patches, aux_loss_raw, channel_mask=None) -> CriterionOutput:
        """
        Args:
            pred_quantiles: (B, V, L, P, Q)
            target_patches: (B, V, L, P)
            aux_loss_raw: scalar tensor or None
        """
        # 1. Main Loss (Quantile)
        y_true = target_patches.unsqueeze(-1)
        diff = y_true - pred_quantiles
        indicator = (diff < 0).float()
        loss = (self.quantiles - indicator) * diff

        if channel_mask is not None:
            mask_exp = channel_mask.view(loss.shape[0], loss.shape[1], 1, 1, 1)
            loss = loss * mask_exp
            main_loss = loss.sum() / (mask_exp.sum() * loss.shape[2] * loss.shape[3] * loss.shape[4] + 1e-6)
        else:
            main_loss = loss.mean()
                
        # 2. Aux Loss
        aux_loss_val = torch.tensor(0.0, device=main_loss.device)
        if self.config.use_moe and self.config.use_load_balance_loss:
            if aux_loss_raw is not None:
                aux_loss_val = aux_loss_raw * self.aux_weight
        
        return CriterionOutput(
            total_loss=main_loss + aux_loss_val,
            main_loss=main_loss,
            aux_loss=aux_loss_val
        )
    