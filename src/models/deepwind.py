"""
src/models/deepwind.py
DeepWind forecasting model.

Supports two prediction head modes controlled by config.pred_head_type:
    "quantile"  — direct quantile regression (original behaviour)
    "student_t" — parametric Student-t distribution
    "zi_beta"   — Zero-Inflated Beta (for power normalised to [0, 1])
    "gmm"       — Mixture of K Gaussians
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from einops import rearrange

from src.utils.registry import register_model
from src.models.configuration import DeepWindConfig
from src.models.backbone import DeepWindBackbone, DeepWindBackboneOutput
from src.layers.embeddings import DeepWindEmbeddings, DeepWindEmbeddingsOutput
from src.layers.heads import build_pred_head
from src.losses.criterion import DeepWindCriterion, DistributionCriterion, CriterionOutput


# ── Output dataclass ───────────────────────────────────────────────────────────

@dataclass
class DeepWindOutput(ModelOutput):
    loss:            Optional[torch.Tensor]    = None
    pred_params:     Optional[torch.Tensor]    = None   # (B,V,T,Q) denormed if quantile
                                                         # (B,V,T,n) normed  if distribution
    loc_scale:       Optional[torch.Tensor]    = None
    scaled_context:  Optional[torch.Tensor]    = None
    aux_loss:        Optional[torch.Tensor]    = None
    all_ffn_details: Optional[Tuple[Any, ...]] = None


# ── Model ──────────────────────────────────────────────────────────────────────

@register_model("deepwind")
class DeepWindModel(PreTrainedModel):
    config_class = DeepWindConfig

    def __init__(self, config: DeepWindConfig) -> None:
        super().__init__(config)
        self.config = config

        # 1. Embeddings
        self.embeddings = DeepWindEmbeddings(config)

        # 2. Backbone
        self.backbone = DeepWindBackbone(config)

        # 3. Head — selected via config.pred_head_type
        self.head = build_pred_head(config)

        # 4. Criterion — quantile uses existing class, distributions use new class
        self._head_type = getattr(config, "pred_head_type", "quantile")
        if self._head_type == "quantile":
            self.criterion = DeepWindCriterion(config)
        else:
            self.criterion = DistributionCriterion(config)

        self.post_init()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        context:              torch.Tensor,
        site_coords:          Optional[torch.Tensor] = None,
        variate_ids:          Optional[torch.Tensor] = None,
        channel_mask:         Optional[torch.Tensor] = None,
        has_coords:           Optional[torch.Tensor] = None,
        kv_cache                                     = None,
        output_hidden_states: bool                   = False,
        output_attentions:    bool                   = False,
        output_router_logits: bool                   = False,
        labels                                       = None,
        **kwargs,
    ) -> DeepWindOutput:

        # ── 1. Embed ──────────────────────────────────────────────────────────
        embeddings_output: DeepWindEmbeddingsOutput = self.embeddings(
            context, site_coords, variate_ids, has_coords
        )

        # ── 2. Backbone ───────────────────────────────────────────────────────
        backbone_output: DeepWindBackboneOutput = self.backbone(
            embeddings_output.embeddings,
            channel_mask,
            kv_cache,
            output_hidden_states,
            output_attentions,
            output_router_logits,
        )

        # ── 3. Head → (B, V, L, P, n_params) ─────────────────────────────────
        patch_preds = self.head(backbone_output.last_hidden_state)

        # ── 4. MoE aux loss ───────────────────────────────────────────────────
        moe_aux_loss = (
            backbone_output.aux_loss
            if self.config.use_moe and backbone_output.aux_loss is not None
            else None
        )

        # ── 5. Loss (computed in normalised space for both modes) ─────────────
        # AR shift: targets are patches [1:], preds align with [:-1]
        targets        = self.embeddings.patcher(embeddings_output.scaled_context)[:, :, 1:]
        preds_for_loss = patch_preds[:, :, :-1]

        if self._head_type == "quantile":
            loss_out: CriterionOutput = self.criterion(
                pred_quantiles = preds_for_loss,
                target_patches = targets,
                aux_loss_raw   = moe_aux_loss,
                channel_mask   = channel_mask,
            )
        else:
            # Flatten is handled inside DistributionCriterion
            # NLL computed in normalised space — targets are already normalised
            loss_out: CriterionOutput = self.criterion(
                pred_params    = preds_for_loss,   # (B, V, L-1, P, n) patch shape
                target_patches = targets,          # (B, V, L-1, P)
                aux_loss_raw   = moe_aux_loss,
                channel_mask   = channel_mask,
            )

        # ── 6. Post-process: flatten patches → time steps ─────────────────────
        # (B, V, L, P, n) → (B, V, T, n)
        preds_flat = rearrange(patch_preds, "b v l p n -> b v (l p) n")

        # ── 7. Inverse normalise (mode-dependent) ─────────────────────────────
        if self._head_type == "quantile":
            # All Q columns are power predictions in normalised space
            # → full inverse norm on every column
            pred_params = self.embeddings.instance_norm.inverse(
                preds_flat, embeddings_output.loc_scale
            )
        else:
            # Distribution params: only μ (index 0) lives in data space.
            # σ is a scale (no location shift). Shape params are dimensionless.
            # Denorm is NOT done here — generator.py handles it after AR decoding,
            # using loc_scale which is carried in the output.
            pred_params = preds_flat   # normalised space, shape (B, V, T, n_params)

        return DeepWindOutput(
            loss           = loss_out.total_loss,
            pred_params    = pred_params,
            loc_scale      = embeddings_output.loc_scale,
            scaled_context = embeddings_output.scaled_context,
            aux_loss       = moe_aux_loss,
            all_ffn_details= backbone_output.all_ffn_details,
        )