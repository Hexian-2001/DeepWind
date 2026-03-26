import torch
import torch.nn as nn
from typing import Optional, Tuple
from jaxtyping import Float


class TimeAwareRotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) with optional xPOS support.
    
    This module encodes the absolute position of tokens by rotating the query and key 
    vectors in the embedding space. It captures relative positions effectively.
    
    Features:
    - Standard RoPE (Su et al., 2021).
    - xPOS (Sun et al., 2022) for better length extrapolation (optional).
    - Caching of sin/cos tables for performance.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        scale_base: int = 1.05,
        use_xpos: bool = True,
        cache_if_possible: bool = True,
    ):
        """
        Args:
            dim: Embedding dimension (head_dim).
            base: Base for the geometric progression of frequencies.
            scale_base: Base for xPOS decay scaling.
            use_xpos: Whether to apply exponential decay scaling (xPOS).
            cache_if_possible: Whether to cache cos/sin/scale tables.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale_base = scale_base
        self.use_xpos = use_xpos
        self.cache_if_possible = cache_if_possible

        # Compute the inverse frequencies: theta_i = base^(-2i/d)
        # We assume dim is even.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Caching containers
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self.register_buffer("scale_cached", None, persistent=False)

    def _update_cos_sin_tables(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

            if self.use_xpos:
                # power goes from 0 to 1
                power = (torch.arange(0, self.dim, 2).float() / self.dim).to(x.device)
                
                # BUG FIX: Change from growth to decay
                scale = 1.0 / (self.scale_base ** power) 
                
                # (T, D/2)
                scale_t = scale.unsqueeze(0) ** t.unsqueeze(1)
                scale_t = torch.cat((scale_t, scale_t), dim=-1)
                self.scale_cached = scale_t[None, None, :, :]
            else:
                self.scale_cached = None

    def rotate_queries_and_keys(
        self, 
        q: Float[torch.Tensor, "batch heads seq_len dim"], 
        k: Float[torch.Tensor, "batch heads seq_len dim"],
        seq_pos_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embedding to queries and keys.
        
        Args:
            q: Queries (Batch, Heads, Seq_Len, Dim)
            k: Keys (Batch, Heads, Seq_Len, Dim)
            seq_pos_offset: Starting position (for autoregressive decoding / KV cache).
        
        Returns:
            q_rot, k_rot
        """
        seq_len = q.shape[2]
        total_len = seq_len + seq_pos_offset

        # Update cache if needed
        if self.cos_cached is None or total_len > self.max_seq_len_cached:
            self._update_cos_sin_tables(q, total_len)

        # Slice the cached tables for current positions
        # Slicing: [0, 0, start:end, :]
        cos = self.cos_cached[:, :, seq_pos_offset:total_len, :]
        sin = self.sin_cached[:, :, seq_pos_offset:total_len, :]
        scale = self.scale_cached[:, :, seq_pos_offset:total_len, :] if self.use_xpos else None

        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        # Apply xPOS scaling
        if self.use_xpos and scale is not None:
            # xPOS: Apply scale to Q and inverse scale to K
            # This simulates decay as distance increases: q * scale * k * (1/scale)
            # Note: Implementation details vary, here we assume symmetric handling or standard xPOS.
            # Usually: q uses scale, k uses 1/scale to decay attention based on relative distance (m-n).
            q_rot = q_rot * scale
            k_rot = k_rot / scale

        return q_rot.type_as(q), k_rot.type_as(k)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input.
        [-x2, x1, -x4, x3, ...]
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    