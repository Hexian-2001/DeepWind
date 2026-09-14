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
    

# src/layers/heads.py 

import torch.distributions as D


class StudentTHead(nn.Module):
    """
    Predicts Student-t distribution parameters per time step.

    Outputs:
        mu:    location  — linear
        sigma: scale     — softplus + eps  (> 0)
        nu:    d.o.f.    — softplus + 2    (> 2, finite variance)

    Returns stacked tensor of shape (B, V, L, P, 3).
    """
    N_PARAMS = 3

    def __init__(self, d_model: int, d_ff: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, self.N_PARAMS * patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, V, L, D)
        Returns:
            params: (B, V, L, P, 3)  — [mu, sigma, nu] along last dim
        """
        B, V, L, _ = x.shape
        out = self.proj(x)                                    # (B, V, L, 3*P)
        out = out.view(B, V, L, self.patch_size, self.N_PARAMS)

        mu_raw, sigma_raw, nu_raw = out.unbind(dim=-1)        # each (B, V, L, P)

        mu    = mu_raw
        sigma = F.softplus(sigma_raw) + 1e-4
        nu    = F.softplus(nu_raw)    + 2.0   # ν > 2 保证方差有限

        return torch.stack([mu, sigma, nu], dim=-1)           # (B, V, L, P, 3)


class ZeroInflatedBetaHead(nn.Module):
    """
    Predicts Zero-Inflated Beta distribution parameters.

    Suitable for power normalised to [0, 1] with frequent zero values.

    Outputs (pi, alpha, beta) stacked as (B, V, L, P, 3).
        pi:    P(power=0)  — sigmoid
        alpha: Beta shape  — softplus + eps
        beta:  Beta shape  — softplus + eps
    """
    N_PARAMS = 3

    def __init__(self, d_model: int, d_ff: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, self.N_PARAMS * patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, L, _ = x.shape
        out = self.proj(x).view(B, V, L, self.patch_size, self.N_PARAMS)

        pi_raw, alpha_raw, beta_raw = out.unbind(dim=-1)

        pi    = torch.sigmoid(pi_raw)
        alpha = F.softplus(alpha_raw) + 1e-4
        beta  = F.softplus(beta_raw)  + 1e-4

        return torch.stack([pi, alpha, beta], dim=-1)


class GaussianMixtureHead(nn.Module):
    """
    Predicts Mixture of K Gaussians parameters.

    Outputs (weights, mu, sigma) for K components, stacked as
    (B, V, L, P, 3*K).
    """

    def __init__(self, d_model: int, d_ff: int, patch_size: int, K: int = 3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.K          = K
        self.N_PARAMS   = 3 * K
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, self.N_PARAMS * patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, L, _ = x.shape
        out = self.proj(x).view(B, V, L, self.patch_size, self.N_PARAMS)

        # Split into 3 groups of K
        w_raw    = out[..., :self.K]
        mu_raw   = out[..., self.K: 2 * self.K]
        sig_raw  = out[..., 2 * self.K:]

        weights = torch.softmax(w_raw,   dim=-1)
        mu      = mu_raw
        sigma   = F.softplus(sig_raw) + 1e-4

        return torch.cat([weights, mu, sigma], dim=-1)        # (B, V, L, P, 3K)


def build_pred_head(config) -> nn.Module:
    """Factory: return the appropriate head based on config.pred_head_type."""
    head_type = getattr(config, "pred_head_type", "quantile")

    if head_type == "quantile":
        return DeepWindQuantilePredHead(
            d_model=config.d_model,
            d_ff=config.d_ff,
            patch_size=config.input_patch_size,
            num_quantiles=len(config.quantiles),
        )
    elif head_type == "student_t":
        return StudentTHead(config.d_model, config.d_ff, config.input_patch_size)
    elif head_type == "zi_beta":
        return ZeroInflatedBetaHead(config.d_model, config.d_ff, config.input_patch_size)
    elif head_type == "gmm":
        return GaussianMixtureHead(
            config.d_model, config.d_ff,
            config.input_patch_size,
            K=config.gmm_components,
        )
    else:
        raise ValueError(f"Unknown pred_head_type: {head_type}")