import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange

from src.layers.blocks import ResidualBlock 


class DeepWindQuantilePredHead(nn.Module):
    """
    Standard Quantile Prediction Head.
    
    Projects the decoder hidden states directly to quantile values per patch.
    Performs: Linear Projection -> Reshape
    
    Input:  (B, V, L, D_model)
    Output: (B, V, L, Patch_Size, Num_Quantiles)
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        patch_size: int,
        num_quantiles: int,
        act_fn_name: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_quantiles = num_quantiles
        
        # Calculate the total flattened output dimension: P * Q
        final_out_dim = self.patch_size * self.num_quantiles

        # The projection layer
        self.projector = ResidualBlock(
            in_dim=d_model,
            h_dim=d_ff,
            out_dim=final_out_dim,
            act_fn_name=act_fn_name,
            dropout_p=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states from decoder (B, V, L, D)
        Returns:
            out: Quantile predictions (B, V, L, P, Q)
        """
        # 1. Project to flattened dimension (B, V, L, P*Q)
        x_proj = self.projector(x)
        
        # 2. Reshape to separate Patch and Quantile dimensions
        # (B, V, L, P*Q) -> (B, V, L, P, Q)
        out = rearrange(x_proj, 'b v l (p q) -> b v l p q', p=self.patch_size, q=self.num_quantiles)
        
        return out
    