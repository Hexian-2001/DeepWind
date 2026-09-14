"""
src/losses/criterion.py
Loss functions for DeepWindModel.

Two criterion classes:
    DeepWindCriterion     — pinball (quantile) loss for quantile head.
    DistributionCriterion — NLL loss for parametric distribution heads
                            (student_t, zi_beta, gmm).

Both share the same CriterionOutput dataclass and follow the same
interface conventions:
    - Accept patch-shaped inputs (B, V, L, P, ...) and flatten internally.
    - Apply channel_mask to exclude padded variates.
    - Handle MoE auxiliary loss via the same config flags.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.distributions as D


# ── Output ─────────────────────────────────────────────────────────────────────

@dataclass
class CriterionOutput:
    total_loss: torch.Tensor
    main_loss:  torch.Tensor
    aux_loss:   torch.Tensor


# ── Shared aux loss helper ─────────────────────────────────────────────────────

def _compute_aux_loss(
    config:       any,
    aux_loss_raw: Optional[torch.Tensor],
    device:       torch.device,
    aux_weight:   float,
) -> torch.Tensor:
    """
    Apply MoE load-balance auxiliary loss if enabled in config.

    Returns a zero tensor if MoE or load-balance loss is disabled,
    or if aux_loss_raw is None.
    """
    val = torch.tensor(0.0, device=device)
    if config.use_moe and config.use_load_balance_loss:
        if aux_loss_raw is not None:
            val = aux_loss_raw * aux_weight
    return val


# ── Quantile Criterion ─────────────────────────────────────────────────────────

class DeepWindCriterion(nn.Module):
    """
    Pinball (quantile) loss for the quantile prediction head.

    Args:
        config: DeepWindConfig. Must contain:
                    config.quantiles             — list of quantile levels
                    config.use_moe               — whether MoE is active
                    config.use_load_balance_loss  — whether to apply aux loss
                    config.aux_loss_weight        — weight for aux loss
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config     = config
        self.aux_weight = getattr(config, "aux_loss_weight", 0.0)
        self.register_buffer("quantiles", torch.tensor(config.quantiles))

    def forward(
        self,
        pred_quantiles: torch.Tensor,            # (B, V, L, P, Q)
        target_patches: torch.Tensor,            # (B, V, L, P)
        aux_loss_raw:   Optional[torch.Tensor] = None,
        channel_mask:   Optional[torch.Tensor] = None,   # (B, V)
    ) -> CriterionOutput:
        """
        Args:
            pred_quantiles: Predicted quantiles in normalised space.
            target_patches: Ground truth patches in normalised space.
            aux_loss_raw:   Raw MoE load-balance loss (unweighted).
            channel_mask:   Binary mask, 1 = valid variate, 0 = padding.
        """
        # ── Pinball loss ──────────────────────────────────────────────────────
        y_true    = target_patches.unsqueeze(-1)            # (B, V, L, P, 1)
        diff      = y_true - pred_quantiles                 # (B, V, L, P, Q)
        indicator = (diff < 0).float()
        loss      = (self.quantiles - indicator) * diff     # (B, V, L, P, Q)

        # ── Channel mask ──────────────────────────────────────────────────────
        if channel_mask is not None:
            mask_exp  = channel_mask.view(loss.shape[0], loss.shape[1], 1, 1, 1)
            loss      = loss * mask_exp
            main_loss = loss.sum() / (
                mask_exp.sum() * loss.shape[2] * loss.shape[3] * loss.shape[4] + 1e-6
            )
        else:
            main_loss = loss.mean()

        # ── Aux loss ──────────────────────────────────────────────────────────
        aux_loss_val = _compute_aux_loss(
            self.config, aux_loss_raw, main_loss.device, self.aux_weight
        )

        return CriterionOutput(
            total_loss = main_loss + aux_loss_val,
            main_loss  = main_loss,
            aux_loss   = aux_loss_val,
        )


# ── Distribution Criterion ─────────────────────────────────────────────────────

class DistributionCriterion(nn.Module):
    """
    Negative log-likelihood loss for parametric distribution heads.

    Supports:
        "student_t" — location-scale Student-t  (μ, σ, ν)
        "zi_beta"   — Zero-Inflated Beta         (π, α, β)
                      Requires targets normalised to [0, 1].
        "gmm"       — Mixture of K Gaussians     (w_k, μ_k, σ_k) × K

    Interface matches DeepWindCriterion:
        - Accepts patch-shaped inputs and flattens internally.
        - Applies channel_mask identically.
        - Handles MoE aux loss via the same config flags.

    Args:
        config: DeepWindConfig. Must contain:
                    config.pred_head_type         — distribution type
                    config.gmm_components         — K for GMM
                    config.use_moe
                    config.use_load_balance_loss
                    config.aux_loss_weight
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config     = config
        self.head_type  = getattr(config, "pred_head_type", "student_t")
        self.K          = getattr(config, "gmm_components", 3)
        self.aux_weight = getattr(config, "aux_loss_weight", 0.0)

    def forward(
        self,
        pred_params:    torch.Tensor,            # (B, V, L, P, n_params)
        target_patches: torch.Tensor,            # (B, V, L, P)
        aux_loss_raw:   Optional[torch.Tensor] = None,
        channel_mask:   Optional[torch.Tensor] = None,   # (B, V)
    ) -> CriterionOutput:
        """
        Args:
            pred_params:    Distribution parameters in normalised space.
            target_patches: Ground truth patches in normalised space.
            aux_loss_raw:   Raw MoE load-balance loss (unweighted).
            channel_mask:   Binary mask, 1 = valid variate, 0 = padding.
        """
        # ── Flatten patches → time steps ─────────────────────────────────────
        B, V, L, P, n = pred_params.shape
        T             = L * P
        params_flat   = pred_params.reshape(B, V, T, n)    # (B, V, T, n)
        target_flat   = target_patches.reshape(B, V, T)    # (B, V, T)

        # ── NLL ───────────────────────────────────────────────────────────────
        nll = self._nll(params_flat, target_flat)           # (B, V, T)

        # ── Channel mask ──────────────────────────────────────────────────────
        if channel_mask is not None:
            mask_exp  = channel_mask.view(B, V, 1).float()  # (B, V, 1)
            nll       = nll * mask_exp
            main_loss = nll.sum() / (mask_exp.sum() * T + 1e-6)
        else:
            main_loss = nll.mean()

        # ── Aux loss ──────────────────────────────────────────────────────────
        aux_loss_val = _compute_aux_loss(
            self.config, aux_loss_raw, main_loss.device, self.aux_weight
        )

        return CriterionOutput(
            total_loss = main_loss + aux_loss_val,
            main_loss  = main_loss,
            aux_loss   = aux_loss_val,
        )

    # ── NLL dispatcher ────────────────────────────────────────────────────────

    def _nll(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.head_type == "student_t":
            return self._student_t_nll(params, y)
        elif self.head_type == "zi_beta":
            return self._zi_beta_nll(params, y)
        elif self.head_type == "gmm":
            return self._gmm_nll(params, y)
        else:
            raise ValueError(
                f"Unknown head_type '{self.head_type}'. "
                f"Expected one of: student_t, zi_beta, gmm."
            )

    # ── NLL implementations ───────────────────────────────────────────────────

    @staticmethod
    def _student_t_nll(params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        NLL of Student-t distribution.

        params[..., 0] = μ   (location,          linear)
        params[..., 1] = σ   (scale,              softplus + eps)
        params[..., 2] = ν   (degrees of freedom, softplus + 2)
        """
        mu, sigma, nu = params[..., 0], params[..., 1], params[..., 2]
        return -D.StudentT(df=nu, loc=mu, scale=sigma).log_prob(y)

    @staticmethod
    def _zi_beta_nll(
        params: torch.Tensor,
        y:      torch.Tensor,
        eps:    float = 1e-6,
    ) -> torch.Tensor:
        """
        NLL of Zero-Inflated Beta distribution.

        params[..., 0] = π   (zero-inflation probability, sigmoid)
        params[..., 1] = α   (Beta shape alpha,            softplus + eps)
        params[..., 2] = β   (Beta shape beta,             softplus + eps)

        Requires y ∈ [0, 1]. Targets must be normalised to unit capacity
        before using this head. Training will raise ValueError otherwise.
        """
        if y.min() < 0.0 or y.max() > 1.0:
            raise ValueError(
                f"ZI-Beta requires targets in [0, 1], "
                f"got min={y.min().item():.4f}  max={y.max().item():.4f}. "
                f"Normalise power to unit capacity before training."
            )

        pi, alpha, beta_ = params[..., 0], params[..., 1], params[..., 2]

        log_p_zero    = torch.log(pi + eps)
        log_p_nonzero = (
            torch.log(1 - pi + eps)
            + D.Beta(alpha, beta_).log_prob(y.clamp(eps, 1 - eps))
        )
        return -torch.where(y < eps, log_p_zero, log_p_nonzero)

    def _gmm_nll(
        self,
        params: torch.Tensor,
        y:      torch.Tensor,
        eps:    float = 1e-8,
    ) -> torch.Tensor:
        """
        NLL of Mixture of K Gaussians.

        params[..., :K]    = w_k   (mixture weights — already softmax in head)
        params[..., K:2K]  = μ_k   (component means)
        params[..., 2K:3K] = σ_k   (component scales — already softplus in head)
        """
        K = self.K

        # weights are already softmax-normalised (done in GaussianMixtureHead)
        weights = params[..., :K]                           # (B, V, T, K)
        mu      = params[..., K:  2 * K]
        sigma   = params[..., 2 * K: 3 * K]

        y_exp     = y.unsqueeze(-1)                         # (B, V, T, 1)
        log_probs = D.Normal(mu, sigma).log_prob(y_exp)     # (B, V, T, K)
        log_mix   = torch.logsumexp(
            log_probs + torch.log(weights + eps), dim=-1    # (B, V, T)
        )
        return -log_mix