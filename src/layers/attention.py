import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Bool

from src.utils.constants import AttentionAxis
from src.utils.cache import KVCache
from src.layers.rope import TimeAwareRotaryEmbedding 


class BaseMultiheadAttention(nn.Module):
    """
    Base multi-head attention supporting both Time-wise and Variate-wise axes.
    
    - Uses PyTorch 2.0+ SDPA (Scaled Dot Product Attention) for automatic FlashAttention support.
    - internal shape convention: (Batch_X, Num_Heads, Seq_Len, Head_Dim).
    """
    attention_axis: AttentionAxis

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rotary_emb: Optional[TimeAwareRotaryEmbedding] = None,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.rotary_emb = rotary_emb

        # QKV Projection: [..., D] -> [..., 3*D]
        self.wQKV = nn.Linear(embed_dim, 3 * embed_dim)
        self.wO = nn.Linear(embed_dim, embed_dim)

        # Safety check for subclassing
        if not hasattr(self, "attention_axis"):
            raise ValueError("Subclasses must define `attention_axis`.")

    def _rearrange_to_batch_axis(
        self, 
        inputs: Float[torch.Tensor, "batch variate seq_len embed_dim"]
    ) -> Float[torch.Tensor, "merged_batch seq_len embed_dim"]:
        """
        Flatten the non-attention axis into the batch dimension.
        """
        if self.attention_axis == AttentionAxis.TIME:
            # Time-wise: Flatten Variate into Batch -> (B*V, T, D)
            return rearrange(inputs, "b v t d -> (b v) t d")
        elif self.attention_axis == AttentionAxis.VARIATE:
            # Variate-wise: Flatten Time into Batch -> (B*T, V, D)
            # Note: Here 'seq_len' in output refers to the attention length (which is V)
            return rearrange(inputs, "b v t d -> (b t) v d")
        else:
            raise ValueError(f"Unknown axis: {self.attention_axis}")

    def _rearrange_from_batch_axis(
        self, 
        output: torch.Tensor, 
        original_batch: int, 
        original_variate: int, 
        original_time: int
    ) -> Float[torch.Tensor, "batch variate seq_len embed_dim"]:
        """
        Restore original (B, V, T, D) shape.
        """
        if self.attention_axis == AttentionAxis.TIME:
            return rearrange(output, "(b v) t d -> b v t d", b=original_batch, v=original_variate)
        elif self.attention_axis == AttentionAxis.VARIATE:
            return rearrange(output, "(b t) v d -> b v t d", b=original_batch, t=original_time)
        return output

    def forward(
        self,
        layer_idx: int,
        inputs: Float[torch.Tensor, "batch variate seq_len embed_dim"],
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        B, V, T, D = inputs.shape
        
        # 1. Reshape to (Merged_Batch, Length, Dim)
        # Length is T for Time-wise, V for Variate-wise
        x = self._rearrange_to_batch_axis(inputs)
        
        # 2. QKV Projection
        # Shape: (Merged_Batch, Length, 3 * Dim)
        qkv = self.wQKV(x)
        
        # 3. Split Heads
        # Shape: (Merged_Batch, Heads, Length, Head_Dim)
        # PyTorch SDPA expects (B, H, L, D)
        qkv = rearrange(qkv, "mb l (three h d) -> three mb h l d", three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)

        # 4. Rotary Embeddings & Caching (Time-Axis Only)
        if self.attention_axis == AttentionAxis.TIME:
            # Only apply RoPE if configured
            if self.rotary_emb is not None:
                # Offset for AR decoding
                seq_pos_offset = kv_cache.seq_len(layer_idx) if kv_cache is not None else 0
                q, k = self.rotary_emb.rotate_queries_and_keys(q, k, seq_pos_offset)
            
            # KV Cache Update
            if kv_cache is not None:
                kv_cache.append(layer_idx, (k, v))
                k, v = kv_cache[layer_idx] # Retrieve full sequence K, V
        
        # 5. Scaled Dot Product Attention
        # Automatically handles FlashAttention if available
        # Dropout is handled internally
        dropout_p = self.dropout if self.training else 0.0
        
        # Align mask shape if necessary. 
        # SDPA supports broadcasting: Mask (B, 1, L, L) can broadcast to (B, H, L, L)
        
        # [Critical] PyTorch SDPA expects different Causal flags:
        # If we provide a manual causal mask, is_causal must be False.
        # If mask is None and we want causal, is_causal must be True.
        
        is_causal = False
        if self.attention_axis == AttentionAxis.TIME:
            # If explicit mask is NOT provided, and we are training or prefilling, use built-in causal
            if attention_mask is None and (kv_cache is None or kv_cache.seq_len(layer_idx) == 0):
                 is_causal = True
        
        # Call Native PyTorch SDPA
        # Returns: (Merged_Batch, Heads, Length, Head_Dim)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
        
        # 6. Merge Heads
        # (MB, H, L, D) -> (MB, L, H*D)
        attn_out = rearrange(attn_out, "mb h l d -> mb l (h d)")
        
        # 7. Output Projection
        attn_out = self.wO(attn_out)
        
        # 8. Restore Dimensions
        output = self._rearrange_from_batch_axis(attn_out, B, V, T)
        
        if output_attentions:
            return output, None # SDPA usually doesn't return weights for efficiency
        
        return output


class TimeWiseMultiheadAttention(BaseMultiheadAttention):
    """
    Time-wise attention: Batch = (B * V), Sequence = T
    """
    attention_axis = AttentionAxis.TIME


class VariateWiseMultiheadAttention(BaseMultiheadAttention):
    """
    Variate-wise attention: Batch = (B * T), Sequence = V
    """
    attention_axis = AttentionAxis.VARIATE