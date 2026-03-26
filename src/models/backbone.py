from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.utils import ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, cast
from einops import rearrange
from jaxtyping import Float, Bool

from src.models.configuration import DeepWindConfig
from src.models.layers import DeepWindLayer, DeepWindLayerOutput
from src.utils.constants import AttentionAxis
from src.layers.rope import TimeAwareRotaryEmbedding 
from src.layers.ffn import FeedForwardOutput
from src.utils.constants import prepare_variate_atten_mask, generate_attention_axes


@dataclass
class DeepWindBackboneOutput(ModelOutput):
    """
    Output container for the DeepWind Backbone.
    """
    last_hidden_state: torch.Tensor = None
    # Optional: Return all hidden states (e.g. for U-Net connections or analysis)
    all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    # Optional: Return all attention weights (for visualization)
    all_attentions: Optional[Tuple[torch.Tensor, ...]] = None
    # Aggregated Auxiliary Loss (e.g. MoE Load Balancing)
    aux_loss: Optional[torch.Tensor] = None
    all_ffn_details: Optional[Tuple[Optional[FeedForwardOutput], ...]] = None


class DeepWindBackbone(nn.Module):
    """
    Industrial-grade Transformer Backbone for Spatio-Temporal Modeling.
    
    Features:
    - Alternating Time/Space Attention.
    - Config-driven initialization.
    - MoE support with loss aggregation.
    - Gradient Checkpointing support (via config).
    """

    def __init__(self, config: DeepWindConfig):
        super().__init__()
        self.config = config

        # 1. Validation
        if config.d_model % config.num_heads != 0:
            raise ValueError(f"d_model ({config.d_model}) must be divisible by num_heads ({config.num_heads})")

        # 2. Rotary Embeddings (Time-Axis only)
        # Shared across layers to save memory, or per-layer if learned. Usually shared.
        self.rotary_emb = None
        if config.use_rotary_emb:
            self.rotary_emb = TimeAwareRotaryEmbedding(
                dim=config.d_model // config.num_heads,
                use_xpos=True,                        
                cache_if_possible=True
            ) # config.use_xpos,

        # 3. Layer Strategy (Time vs Space)
        attention_axes = generate_attention_axes(
            num_layers=config.num_layers,
            every_n=config.variate_atten_every_n_layers,
            variate_first=config.variate_atten_first
        )

        # 4. Construct Layers
        # Note: passing `config` directly to Layer is optional. 
        # Here we unpack explicitly to show dependency, but passing config is also fine.
        self.layers = nn.ModuleList([
            DeepWindLayer(
                embed_dim=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                attention_axis=attention_axes[i], 
                rms_norm=config.use_rms_norm,
                rotary_emb=self.rotary_emb if attention_axes[i] == AttentionAxis.TIME else None,
                                
                # FFN Config
                use_moe=config.use_moe,
                activation=config.dense_act_fn,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                use_load_balance_loss=config.use_load_balance_loss,
            )
            for i in range(config.num_layers)
        ])
        
        # 5. Gradient Checkpointing
        self.gradient_checkpointing = False


    def forward(
        self,
        inputs: Float[torch.Tensor, "batch variate seq_len embed_dim"],
        id_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> DeepWindBackboneOutput:
        
        batch_size, num_variates, seq_len, embed_dim = inputs.shape
        # 1. Prepare Masks
        time_wise_mask = None # Causal mask usually handled internally by SDPA or RoPE logic
        
        variate_wise_mask = None
        if id_mask is not None:
            # Generate the additive mask for space-wise attention.
            # We must pass `seq_len` to correctly expand the (Batch, Variate) mask
            # across all time steps.
            variate_wise_mask = prepare_variate_atten_mask(
                id_mask=id_mask, 
                dtype=inputs.dtype,
                seq_len=seq_len
            )

        # 2. Init Containers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_ffn_details = () if output_router_logits else None

        total_aux_loss = torch.tensor(0.0, device=inputs.device) if self.training else None
        num_layers_with_loss = 0

        hidden_states = inputs

        # 3. Layer Loop
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Select correct mask based on layer type
            current_mask = variate_wise_mask if layer.attention_axis == AttentionAxis.VARIATE else time_wise_mask

            # Call Layer
            # We use gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*args):
                        # Wrapper to unpack and return only hidden_state for grad checkpointing compatibility
                        return module(*args).hidden_state
                    return custom_forward

                # Note: Gradient Checkpointing usually breaks Aux Loss return. 
                # If you need Aux Loss + Checkpointing, you need a more complex wrapper.
                # Here we simplify for standard usage.
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    i, 
                    hidden_states, 
                    current_mask,
                    kv_cache,
                    output_attentions
                )
            else:
                # Standard Forward
                layer_output: DeepWindLayerOutput = layer(
                    layer_idx=i,
                    inputs=hidden_states,
                    attention_mask=current_mask,
                    kv_cache=kv_cache,
                    output_attentions=output_attentions
                )
                
                # 4. Unpack Result
                hidden_states = layer_output.hidden_state

                if layer_output.aux_loss is not None:
                    if total_aux_loss is not None:
                        total_aux_loss = total_aux_loss + layer_output.aux_loss
                        num_layers_with_loss += 1
                
                if output_router_logits:
                    all_ffn_details = all_ffn_details + (layer_output.moe_metadata,)      
                
                if output_attentions:
                    all_attentions = all_attentions + (layer_output.attn_weights,)

        # 6. Finalize Outputs
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Average loss over layers to keep magnitude independent of depth
        if total_aux_loss is not None and num_layers_with_loss > 0:
            total_aux_loss = total_aux_loss / num_layers_with_loss

        return DeepWindBackboneOutput(
            last_hidden_state=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            all_ffn_details=all_ffn_details, 
            aux_loss=total_aux_loss
        )
    