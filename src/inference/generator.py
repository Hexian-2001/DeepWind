"""
src/inference/generator.py
Autoregressive forecasting strategies for DeepWindModel.

Supports two prediction head modes:
    "quantile"    — direct quantile regression (original behaviour)
    "student_t"   — parametric Student-t distribution
    "zi_beta"     — Zero-Inflated Beta
    "gmm"         — Mixture of K Gaussians

For distribution heads, inference-time quantile levels can be freely
specified and are independent of the quantile levels used during training.
"""
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from scipy.stats import t as scipy_t, beta as scipy_beta

from src.models.deepwind import DeepWindModel

logger = logging.getLogger(__name__)


# ── Output ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Forecast:
    """
    Container for forecasting outputs.

    Attributes:
        point_preds:    Median (point) forecast.      Shape: (B, V, H)
        quantile_preds: Quantile predictions.         Shape: (B, V, H, Q)
                        Q is determined by inference_quantiles, not training quantiles.
    """
    point_preds:    Float[torch.Tensor, "batch variates future_time_steps"]
    quantile_preds: Optional[Float[torch.Tensor, "batch variates future_time_steps quantiles"]] = None


# ── Distribution → Quantile converter ─────────────────────────────────────────

class DistributionSampler:
    """
    Converts distribution parameters (in normalised space) to quantile
    predictions (in original data space).

    This is called once after AR decoding is complete, not at every step.
    Supports analytical inversion (Student-t, ZI-Beta) and Monte Carlo
    sampling (GMM).

    Inference-time quantile levels are fully decoupled from training quantiles:
    you can request any set of quantiles regardless of what was used during
    training.

    Args:
        head_type:          Distribution type: "student_t" | "zi_beta" | "gmm".
        inference_quantiles: Quantile levels to compute at inference time. (Q,)
        gmm_components:     Number of Gaussian components (GMM only).
        n_mc_samples:       Monte Carlo samples for GMM quantile estimation.
    """

    # Standard quantile grids for convenience
    STANDARD_GRIDS = {
        "decile":     np.linspace(0.1,  0.9,  9),
        "ventile":    np.linspace(0.05, 0.95, 19),
        "percentile": np.linspace(0.01, 0.99, 99),
        "pi_80":      np.array([0.10, 0.50, 0.90]),
        "pi_90":      np.array([0.05, 0.50, 0.95]),
        "pi_95":      np.array([0.025, 0.50, 0.975]),
    }

    def __init__(
        self,
        head_type:           str,
        inference_quantiles: np.ndarray,
        gmm_components:      int = 3,
        n_mc_samples:        int = 2000,
    ) -> None:
        self.head_type           = head_type
        self.inference_quantiles = np.asarray(inference_quantiles, dtype=np.float64)
        self.K                   = gmm_components
        self.n_mc_samples        = n_mc_samples

    # ── Public ────────────────────────────────────────────────────────────────

    def params_to_quantiles(
        self,
        params:    torch.Tensor,              # (B, V, T, n_params) normalised space
        loc_scale: Tuple[torch.Tensor, torch.Tensor],  # each (B, V, 1)
    ) -> torch.Tensor:                        # (B, V, T, Q) original space
        """
        Convert normalised distribution parameters to quantile predictions
        in the original data space.

        Args:
            params:    Distribution parameters from forward(). Normalised space.
            loc_scale: (loc, scale) tuple from instance norm for inverse transform.

        Returns:
            Quantile predictions of shape (B, V, T, Q).
        """
        if self.head_type == "student_t":
            return self._student_t_quantiles(params, loc_scale)
        elif self.head_type == "zi_beta":
            return self._zi_beta_quantiles(params, loc_scale)
        elif self.head_type == "gmm":
            return self._gmm_quantiles(params, loc_scale)
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

    def extract_median(
        self,
        params:    torch.Tensor,
        loc_scale: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract the median (point forecast) from distribution parameters.
        Used at each AR step to obtain the next input patch.

        For Student-t: median = μ (symmetric distribution).
        For ZI-Beta:   median computed analytically.
        For GMM:       median via Monte Carlo.

        Returns: (B, V, T) in original data space.
        """
        loc, scale = loc_scale   # each (B, V, 1)

        if self.head_type == "student_t":
            # Median of Student-t = μ (symmetric)
            mu = params[..., 0]                    # (B, V, T)
            return mu * scale.squeeze(-1) + loc.squeeze(-1)

        elif self.head_type == "zi_beta":
            median_idx = int(np.argmin(np.abs(self.inference_quantiles - 0.5)))
            return self.params_to_quantiles(params, loc_scale)[..., median_idx]

        elif self.head_type == "gmm":
            tmp_sampler = DistributionSampler(
                head_type           = "gmm",
                inference_quantiles = np.array([0.5]),
                gmm_components      = self.K,
                n_mc_samples        = self.n_mc_samples,
            )
            return tmp_sampler.params_to_quantiles(params, loc_scale)[..., 0]

    # ── Private ───────────────────────────────────────────────────────────────

    def _student_t_quantiles(self, params, loc_scale):
        """Analytical quantiles: q = μ + σ * t_ppf(q, ν)"""
        mu_n  = params[..., 0].cpu().numpy()
        sigma = params[..., 1].cpu().numpy()
        nu    = params[..., 2].cpu().numpy()

        loc, scale = loc_scale
        loc   = loc.cpu().numpy()    # (B, V, 1) — 不 squeeze
        scale = scale.cpu().numpy()  # (B, V, 1) — 不 squeeze

        q = self.inference_quantiles                           # (Q,)
        t_ppf  = scipy_t.ppf(q, df=nu[..., np.newaxis])       # (B, V, T, Q)
        q_norm = mu_n[..., np.newaxis] + sigma[..., np.newaxis] * t_ppf

        # scale: (B, V, 1) → (B, V, 1, 1), broadcasts with (B, V, T, Q)
        result = q_norm * scale[..., np.newaxis] + loc[..., np.newaxis]
        return torch.from_numpy(result.astype(np.float32)).to(params.device)

    def _zi_beta_quantiles(self, params, loc_scale):
        """
        Analytical quantiles for Zero-Inflated Beta.

        P(Y <= y) = π + (1-π) * CDF_Beta(y | α, β)
        Invert:
            q <= π       → y = 0
            q >  π       → y = Beta_ppf((q - π) / (1 - π), α, β)

        ZI-Beta is defined on [0, 1] (normalised power).
        Inverse norm is applied after quantile computation.
        """
        pi    = params[..., 0].cpu().numpy()
        alpha = params[..., 1].cpu().numpy()
        beta_ = params[..., 2].cpu().numpy()

        loc, scale = loc_scale
        loc   = loc.cpu().numpy()
        scale = scale.cpu().numpy()

        q = self.inference_quantiles                           # (Q,)

        adjusted = np.clip(
            (q - pi[..., np.newaxis]) / (1 - pi[..., np.newaxis] + 1e-8),
            0.0, 1.0,
        )
        y_q = np.where(
            q <= pi[..., np.newaxis],
            0.0,
            scipy_beta.ppf(adjusted, alpha[..., np.newaxis], beta_[..., np.newaxis]),
        )                                                      # (B, V, T, Q)

        # Inverse norm
        result = y_q * scale[..., np.newaxis] + loc[..., np.newaxis]
        return torch.from_numpy(result.astype(np.float32)).to(params.device)

    def _gmm_quantiles(self, params, loc_scale):
        """
        Monte Carlo quantiles for Gaussian Mixture Model.

        Sample n_mc_samples from the mixture, then compute empirical quantiles.
        """
        K       = self.K
        weights = params[..., :K].cpu().numpy()
        mu_n    = params[..., K:2*K].cpu().numpy()    # normalised
        sigma   = params[..., 2*K:3*K].cpu().numpy()

        loc, scale = loc_scale
        loc   = loc.cpu().numpy()
        scale = scale.cpu().numpy()

        B, V, T, _ = weights.shape
        rng         = np.random.default_rng(0)

        # Vectorised sampling
        flat_w = weights.reshape(-1, K)                        # (B*V*T, K)
        flat_mu    = mu_n.reshape(-1, K)
        flat_sigma = sigma.reshape(-1, K)
        flat_loc   = np.broadcast_to(loc,   (B, V, T)).reshape(-1)   # (B*V*T,)
        flat_scale = np.broadcast_to(scale, (B, V, T)).reshape(-1)   # (B*V*T,)

        N  = B * V * T
        S  = self.n_mc_samples

        # Sample component for each position and sample
        cumsum   = np.cumsum(flat_w, axis=-1)                  # (N, K)
        u        = rng.random((N, S))                          # (N, S)
        comp_idx = (u[:, :, np.newaxis] > cumsum[:, np.newaxis, :]).sum(axis=-1)
        comp_idx = comp_idx.clip(0, K - 1)                     # (N, S)
        
        i_idx        = np.arange(N)[:, np.newaxis]
        samples_norm = (
            flat_mu[i_idx, comp_idx]
            + flat_sigma[i_idx, comp_idx] * rng.standard_normal((N, S))
        )                                                     # (N, S) normalised

        # Inverse norm per position
        samples_orig = (
            samples_norm * flat_scale[:, np.newaxis]
            + flat_loc[:, np.newaxis]
        )

        result = np.quantile(samples_orig, self.inference_quantiles, axis=-1).T  # (N, Q)
        result = result.reshape(B, V, T, len(self.inference_quantiles))
        return torch.from_numpy(result.astype(np.float32)).to(params.device)


# ── Forecaster ─────────────────────────────────────────────────────────────────

class DeepWindForecaster:
    """
    Patch-wise autoregressive forecaster for DeepWindModel.

    Quantile head:
        - Greedy: feeds median quantile at each AR step.
        - MQD:    Expand-Collapse multi-trajectory decoding.

    Distribution head:
        - Greedy only (MQD not supported for distribution heads).
        - At each AR step, median is extracted analytically from parameters.
        - After all AR steps, DistributionSampler converts parameters to
          quantiles at any requested inference_quantiles.

    Args:
        model:               Trained DeepWindModel.
        median_q:            Quantile level for point forecast (quantile head only).
        cfg:                 Hydra config. cfg.data.context_length required.
        inference_quantiles: Quantile levels to output at inference time.
                             For quantile heads: must match training quantiles.
                             For distribution heads: any levels are valid.
                             Accepts np.ndarray or one of DistributionSampler.STANDARD_GRIDS keys.
    """

    def __init__(
        self,
        model:               DeepWindModel,
        median_q:            float = 0.5,
        cfg:                 Optional[DictConfig] = None,
        inference_quantiles: Optional[np.ndarray | str] = None,
    ) -> None:
        if cfg is None:
            raise ValueError("cfg must be provided.")

        self.model  = model
        self.device = next(model.parameters()).device
        self.cfg    = cfg

        # Registered quantiles (training-time, used for quantile head)
        q = torch.as_tensor(model.config.quantiles, dtype=torch.float32)
        self.registered_quantiles = q
        self.median_idx = int(torch.argmin(torch.abs(q - median_q)).item())

        # Head type
        self._head_type = getattr(model.config, "pred_head_type", "quantile")

        # Inference-time quantile levels
        self._inference_quantiles = self._resolve_inference_quantiles(
            inference_quantiles
        )

        # Distribution sampler (only for non-quantile heads)
        if self._head_type != "quantile":
            self._dist_sampler = DistributionSampler(
                head_type           = self._head_type,
                inference_quantiles = self._inference_quantiles,
                gmm_components      = getattr(model.config, "gmm_components", 3),
                n_mc_samples        = 2000,
            )
            logger.info(
                f"[DeepWindForecaster] Distribution head: {self._head_type}  "
                f"inference_quantiles={self._inference_quantiles}"
            )

    def _resolve_inference_quantiles(
        self, inference_quantiles: Optional[np.ndarray | str]
    ) -> np.ndarray:
        """
        Resolve inference_quantiles to a numpy array.

        Priority:
            1. Explicit np.ndarray argument.
            2. String key from DistributionSampler.STANDARD_GRIDS.
            3. cfg.inference.inference_quantiles (list in yaml).
            4. Fall back to training quantiles (quantile head) or decile grid.
        """
        if isinstance(inference_quantiles, np.ndarray):
            return inference_quantiles

        if isinstance(inference_quantiles, str):
            grids = DistributionSampler.STANDARD_GRIDS
            if inference_quantiles not in grids:
                raise ValueError(
                    f"Unknown grid name '{inference_quantiles}'. "
                    f"Available: {list(grids.keys())}"
                )
            return grids[inference_quantiles]

        # Try cfg
        cfg_q = self.cfg.inference.get("inference_quantiles", None)
        if cfg_q is not None:
            return np.asarray(cfg_q, dtype=np.float64)

        # Default
        if self._head_type == "quantile":
            return self.registered_quantiles.numpy()
        else:
            return DistributionSampler.STANDARD_GRIDS["ventile"]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare_inputs(self, context, site_coords, variate_ids, channel_mask, has_coords):
        d = self.device
        return (
            context.to(d),
            site_coords.to(d),
            variate_ids.to(d),
            channel_mask.to(d) if channel_mask is not None else None,
            has_coords.to(d)   if has_coords   is not None else None,
        )

    def _model_forward(self, context, site_coords, variate_ids,
                       channel_mask, has_coords, kv_cache,
                       output_hidden_states, output_attentions, output_router_logits):
        return self.model(
            context=context, site_coords=site_coords, variate_ids=variate_ids,
            channel_mask=channel_mask, has_coords=has_coords, kv_cache=kv_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
        )

    def _slide_context(self, context, new_patch):
        ctx     = torch.cat([context, new_patch], dim=-1)
        max_len = self.cfg.data.context_length
        return ctx[..., -max_len:] if ctx.shape[-1] > max_len else ctx

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forecast(
        self,
        context:              Float[torch.Tensor, "batch variate time_steps"],
        site_coords:          torch.Tensor,
        variate_ids:          torch.Tensor,
        channel_mask:         Optional[torch.Tensor] = None,
        has_coords:           Optional[torch.Tensor] = None,
        prediction_length:    int  = 16,
        kv_cache                   = None,
        output_hidden_states: bool = False,
        output_attentions:    bool = False,
        output_router_logits: bool = False,
        mqd_infer:            bool = False,
    ) -> Forecast:
        
        shared = dict(
            context=context, site_coords=site_coords, variate_ids=variate_ids,
            channel_mask=channel_mask, has_coords=has_coords,
            prediction_length=prediction_length, kv_cache=kv_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
        )

        if self._head_type == "quantile":
            if mqd_infer:
                point_preds, quantile_preds = self._generate_mqd(**shared)
            else:
                point_preds, quantile_preds = self._generate_greedy(**shared)
        else:
            # Distribution heads: greedy only
            if mqd_infer:
                logger.warning(
                    "MQD decoding is not supported for distribution heads. "
                    "Falling back to greedy."
                )
            point_preds, quantile_preds = self._generate_greedy_dist(**shared)

        return Forecast(point_preds=point_preds, quantile_preds=quantile_preds)

    # ── Quantile head: Greedy ─────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_greedy(
        self, context, site_coords, variate_ids, channel_mask, has_coords,
        prediction_length, kv_cache,
        output_hidden_states, output_attentions, output_router_logits,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context, site_coords, variate_ids, channel_mask, has_coords = \
            self._prepare_inputs(context, site_coords, variate_ids, channel_mask, has_coords)

        patch_size    = int(self.model.config.input_patch_stride)
        num_iters     = math.ceil(prediction_length / patch_size)
        collected     = []
        cur_context   = context

        for _ in range(num_iters):
            output = self._model_forward(
                cur_context, site_coords, variate_ids, channel_mask, has_coords,
                kv_cache, output_hidden_states, output_attentions, output_router_logits,
            )
            preds, _ = torch.sort(
                output.pred_params[..., -patch_size:, :], dim=-1
            )                                                  # (B, V, P, Q)
            collected.append(preds)
            cur_context = self._slide_context(cur_context, preds[..., self.median_idx])

        full_preds     = torch.cat(collected, dim=2)[..., :prediction_length, :]
        point_forecast = full_preds[..., self.median_idx]
        return point_forecast, full_preds

    # ── Quantile head: MQD ───────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_mqd(
        self, context, site_coords, variate_ids, channel_mask, has_coords,
        prediction_length, kv_cache,
        output_hidden_states, output_attentions, output_router_logits,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context, site_coords, variate_ids, channel_mask, has_coords = \
            self._prepare_inputs(context, site_coords, variate_ids, channel_mask, has_coords)

        B, V, _       = context.shape
        patch_size    = int(self.model.config.input_patch_stride)
        num_iters     = math.ceil(prediction_length / patch_size)
        Q             = len(self.registered_quantiles)
        target_q      = self.registered_quantiles.to(self.device)

        # Step 0
        output = self._model_forward(
            context, site_coords, variate_ids, channel_mask, has_coords,
            kv_cache, output_hidden_states, output_attentions, output_router_logits,
        )
        first_patch, _ = torch.sort(
            output.pred_params[..., -patch_size:, :], dim=-1
        )

        # Expand B → B*Q
        cur_context = context.repeat_interleave(Q, dim=0)
        cur_coords  = site_coords.repeat_interleave(Q, dim=0)
        cur_var_ids = variate_ids.repeat_interleave(Q, dim=0)
        cur_ch_mask = channel_mask.repeat_interleave(Q, dim=0) if channel_mask is not None else None
        cur_has_c   = has_coords.repeat_interleave(Q, dim=0)   if has_coords   is not None else None

        first_input = first_patch.permute(0, 3, 1, 2).reshape(B * Q, V, patch_size)
        collected   = [first_input]
        cur_context = self._slide_context(cur_context, first_input)

        # Steps 1..N-1
        for _ in range(num_iters - 1):
            output = self._model_forward(
                cur_context, cur_coords, cur_var_ids, cur_ch_mask, cur_has_c,
                kv_cache, output_hidden_states, output_attentions, output_router_logits,
            )
            candidates, _ = torch.sort(
                output.pred_params[..., -patch_size:, :], dim=-1
            )
            candidates = candidates.view(B, Q, V, patch_size, Q)
            pool       = candidates.permute(0, 2, 3, 1, 4).reshape(B, V, patch_size, Q * Q)
            sorted_pool, _ = torch.sort(pool, dim=-1)
            indices    = (target_q * (Q * Q - 1)).long().clamp(0, Q * Q - 1)
            next_q     = sorted_pool[..., indices]
            next_input = next_q.permute(0, 3, 1, 2).reshape(B * Q, V, patch_size)
            collected.append(next_input)
            cur_context = self._slide_context(cur_context, next_input)

        full_pred    = torch.cat(collected, dim=-1)[..., :prediction_length]
        full_pred    = full_pred.view(B, Q, V, prediction_length).permute(0, 2, 3, 1)
        pred_samples, _ = torch.sort(full_pred, dim=-1)
        return pred_samples[..., self.median_idx], pred_samples

    # ── Distribution head: Greedy ─────────────────────────────────────────────

    @torch.no_grad()
    def _generate_greedy_dist(
        self, context, site_coords, variate_ids, channel_mask, has_coords,
        prediction_length, kv_cache,
        output_hidden_states, output_attentions, output_router_logits,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        context, site_coords, variate_ids, channel_mask, has_coords = \
            self._prepare_inputs(context, site_coords, variate_ids, channel_mask, has_coords)

        patch_size = int(self.model.config.input_patch_stride)
        num_iters  = math.ceil(prediction_length / patch_size)
        median_idx = int(np.argmin(np.abs(self._inference_quantiles - 0.5)))

        collected_quantiles = []
        cur_context         = context

        for _ in range(num_iters):
            output = self._model_forward(
                cur_context, site_coords, variate_ids, channel_mask, has_coords,
                kv_cache, output_hidden_states, output_attentions, output_router_logits,
            )

            params_patch = output.pred_params[..., -patch_size:, :]   # (B, V, P, n)
            loc_scale    = (output.loc_scale[0], output.loc_scale[1])
        
            # Convert with current step's loc_scale → original space (B, V, P, Q)
            q_patch, _ = torch.sort(
                self._dist_sampler.params_to_quantiles(params_patch, loc_scale),
                dim=-1,
            )
            collected_quantiles.append(q_patch)

            # Median is already in original space — feed directly as next patch
            cur_context = self._slide_context(cur_context, q_patch[..., median_idx])

        # All patches already in original space — safe to concatenate
        quantile_preds = torch.cat(collected_quantiles, dim=2)[..., :prediction_length, :]
        point_preds    = quantile_preds[..., median_idx]               # (B, V, pred_len)

        return point_preds, quantile_preds