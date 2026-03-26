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
from src.layers.heads import DeepWindQuantilePredHead
from src.losses.criterion import DeepWindCriterion, CriterionOutput


@dataclass
class DeepWindOutput(ModelOutput):
    """
    Output type for the DeepWind forecasting model.
    """
    loss: Optional[torch.Tensor] = None
    
    # Prediction results (Optional, for memory efficiency)
    denorm_quantile_preds: Optional[torch.Tensor] = None
    loc_scale: Optional[torch.Tensor] = None
    scaled_context: Optional[torch.Tensor] = None

    # MoE Aux Loss (separated for logging)
    # Propagated MoE details from Backbone (for analysis/visualization)
    aux_loss: Optional[torch.Tensor] = None
    all_ffn_details: Optional[Tuple[Any, ...]] = None


@register_model("deepwind")
class DeepWindModel(PreTrainedModel):
    config_class = DeepWindConfig

    def __init__(self, config: DeepWindConfig):
        super().__init__(config)
        self.config = config

        # 1. Embeddings Module
        self.embeddings = DeepWindEmbeddings(config)

        # 2. Backbone (Transformer Decoder)
        self.backbone = DeepWindBackbone(config)

        # 3. Output Head
        self.head = DeepWindQuantilePredHead(
            d_model=config.d_model,
            d_ff=config.d_ff,
            patch_size=config.input_patch_size,
            num_quantiles=len(config.quantiles)
        )

        # 4. Loss
        self.criterion = DeepWindCriterion(config)

        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        context: torch.Tensor,
        site_coords: Optional[torch.Tensor] = None,
        variate_ids: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        has_coords: Optional[torch.Tensor] = None,
        kv_cache = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        labels = None,
        **kwargs,
    ) -> DeepWindOutput:
        
        # 1. Embed & Preprocess
        # Return embeds (B, V, L, D) and context info for inverse norm/loss
        embeddings_output: DeepWindEmbeddingsOutput = self.embeddings(
            context, site_coords, variate_ids, has_coords
        )

        # 2. Transformer Forward
        backbone_output: DeepWindBackboneOutput = self.backbone(embeddings_output.embeddings,
                                       channel_mask,
                                       kv_cache,
                                       output_hidden_states,
                                       output_attentions,
                                       output_router_logits
                                       )

        # 3. Predict Quantiles
        patch_quantiles = self.head(backbone_output.last_hidden_state)   # (B, V, L, P, Q)

        loss = None
        moe_aux_loss = None
        
        if self.config.use_moe and backbone_output.aux_loss is not None:
                moe_aux_loss = backbone_output.aux_loss

        targets = self.embeddings.patcher(embeddings_output.scaled_context)[:, :, 1:]
        preds_for_loss = patch_quantiles[:, :, :-1]
        loss_out: CriterionOutput = self.criterion(
            pred_quantiles=preds_for_loss,
            target_patches=targets,
            aux_loss_raw=moe_aux_loss,
            channel_mask=channel_mask
        )
        loss = loss_out.total_loss

        # 5. Post-process (Flatten patches -> Time steps)
        # (B, V, L, P, Q) -> (B, V, T, Q)
        quantile_preds = rearrange(patch_quantiles, "b v l p q -> b v (l p) q")

        # Inverse Normalize
        denorm_quantile_preds = self.embeddings.instance_norm.inverse(
            quantile_preds, embeddings_output.loc_scale
        )
        
        return DeepWindOutput(
            loss=loss,
            denorm_quantile_preds=denorm_quantile_preds,
            loc_scale=embeddings_output.loc_scale,
            scaled_context=embeddings_output.scaled_context,
            aux_loss=moe_aux_loss,
            all_ffn_details=backbone_output.all_ffn_details
        )
    