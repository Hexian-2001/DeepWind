import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Callable, cast
from dataclasses import dataclass
from transformers.utils import ModelOutput
from einops import rearrange
from jaxtyping import Float, Bool

from src.layers import RMSNorm, TimeWiseMultiheadAttention, VariateWiseMultiheadAttention, FeedForwardOutput, build_ffn
from src.utils import AttentionAxis
from src.utils.cache import KVCache

@dataclass
class DeepWindLayerOutput(ModelOutput):
    """
    Output for a single Transformer Layer.
    Strictly typing what comes out of a layer block.
    """
    hidden_state: torch.Tensor
    # Auxiliary loss from this specific layer (e.g. MoE load balancing)
    aux_loss: Optional[torch.Tensor] = None
    # Attention weights (optional, for visualization)
    attn_weights: Optional[torch.Tensor] = None
    moe_metadata: FeedForwardOutput = None


class DeepWindLayer(nn.Module):
    """
    A single Transformer Block supporting:
    1. Dual-Axis Attention (Time-wise OR Variate-wise).
    2. Mixture-of-Experts (MoE) or Standard FFN (GLU/Dense).
    3. Pre-Norm Architecture with Residual Connections.
    4. Rotary Embeddings (RoPE) integration.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_axis: AttentionAxis = AttentionAxis.TIME,
        rms_norm: bool = True,
        # --- Rotary Embeddings ---
        rotary_emb: Optional[nn.Module] = None, 
        # --- FFN Configuration ---
        use_moe: bool = False,
        activation: Union[str, Callable] = "silu",
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        use_load_balance_loss: bool = True,
    ) -> None:
        super().__init__()
            
        self.embed_dim = embed_dim
        self.attention_axis = attention_axis
        
        # 1. Normalization Layers (Pre-Norm)
        # Using factory pattern for Norm type
        norm_cls = RMSNorm if rms_norm else nn.LayerNorm
        self.norm1 = norm_cls(embed_dim)
        self.norm2 = norm_cls(embed_dim)

        # 2. Attention Mechanism Dispatcher
        # Selects the correct attention module based on the axis (Time vs Variate/Space)
        if self.attention_axis == AttentionAxis.TIME:
            self.attention = TimeWiseMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                rotary_emb=rotary_emb, 
            )
        elif self.attention_axis == AttentionAxis.VARIATE:
            self.attention = VariateWiseMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                rotary_emb=None, 
            )
        else:
            raise ValueError(f"Invalid attention axis: {attention_axis}")

        # 3. Feed Forward Network (FFN) construction
        # Uses the factory function to switch between Dense/GLU/MoE
        get_ffn = build_ffn(
            d_model=embed_dim,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            use_moe=use_moe,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            use_load_balance_loss=use_load_balance_loss,
        )
        self.ffn = get_ffn()

    def forward(
        self,
        layer_idx: int,
        inputs: Float[torch.Tensor, "batch variate seq_len embed_dim"],
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
    ) -> DeepWindLayerOutput:
        """
        Forward pass for the Transformer Layer.

        Args:
            layer_idx: Index of the current layer (needed for RoPE/Cache).
            inputs: Input tensor (B, V, T, D).
            attention_mask: Mask for attention. Shape depends on the axis:
                            - Time: (B, 1, T, T) or similar
                            - Variate: (B, 1, V, V)
            kv_cache: Key-Value cache for autoregressive inference.
            output_attentions: Whether to return attention weights.

        Returns:
            DeepWindLayerOutput object containing hidden states and auxiliary data.
        """
        
        # --- Block 1: Attention ---
        # 1.1. Pre-Normalization
        residual = inputs
        hidden_states = self.norm1(inputs)

        # 1.2. Attention Forward
        # We rely on the specific attention class (Time/Variate) to handle
        # internal dimension permutation (rearrange).
        attn_outputs = self.attention(
            layer_idx=layer_idx,
            inputs=hidden_states,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            output_attentions=output_attentions
        )

        # 1.3. Handle Output format (Tuple vs Tensor)
        attn_weights = None
        if isinstance(attn_outputs, tuple):
            attn_output, attn_weights = attn_outputs
        else:
            attn_output = attn_outputs

        # 1.4. Residual Connection
        hidden_states = residual + attn_output

        # --- Block 2: Feed Forward (FFN/MoE) ---
        # 2.1. Pre-Normalization
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)

        # 2.2. FFN Forward
        ffn_result: FeedForwardOutput = self.ffn(hidden_states)
        
        # 2. Residual Connection
        # Explicitly access .outputs
        hidden_states = residual + ffn_result.outputs

        # 3. Retrieve Aux Loss (Side Channel)
        # Consistent for both types (Dense FFN simply won't have this attribute or it will be None)
        aux_loss = getattr(self.ffn, "aux_loss", None)

        # 4. Return Layer Output
        return DeepWindLayerOutput(
            hidden_state=hidden_states,
            aux_loss=aux_loss,
            attn_weights=attn_weights,
            moe_metadata=ffn_result 
        )
    