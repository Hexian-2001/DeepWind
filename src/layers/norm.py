import torch
import torch.nn as nn

from typing import List, Optional, TypeAlias, Union, Tuple


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class InstanceNorm(nn.Module):
    """
    Instance-wise normalization for time series.

    Expected input shape: (batch, variate, seq_len).
    For each (batch, variate) pair, we normalize along the last dimension (seq_len):

        x_norm[b, v, :] = arcsinh( (x[b, v, :] - loc[b, v, 1]) / scale[b, v, 1] )

    if use_arcsinh is True, otherwise just standardization.
    """

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : Tensor
            Shape (B, V, T). Normalization is along dim=-1 (T).
        loc_scale : (loc, scale), optional
            If provided, both should be broadcastable to shape (B, V, 1).

        Returns
        -------
        x_norm : Tensor
            Same shape as x.
        (loc, scale) : tuple
            loc, scale with shape (B, V, 1), can be reused for inverse().
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)

        if loc_scale is None:
            # mean over last dim (seq_len)
            loc = torch.nanmean(x, dim=-1, keepdim=True)
            # std over last dim (seq_len)
            var = torch.nanmean((x - loc) ** 2, dim=-1, keepdim=True)
            scale = torch.sqrt(var)

            # handle NaN 和 0：NaN -> (loc=0, scale=1)，并确保 scale >= eps
            loc = torch.nan_to_num(loc, nan=0.0)
            scale = torch.nan_to_num(scale, nan=1.0)
            scale = scale.clamp_min(self.eps)
        else:
            loc, scale = loc_scale
            
            loc = loc.to(x.dtype)
            scale = scale.to(x.dtype)

        x_norm = (x - loc) / scale

        if self.use_arcsinh:
            x_norm = torch.arcsinh(x_norm)

        return x_norm.to(orig_dtype), (loc, scale)

    def inverse(
        self,
        x: torch.Tensor,
        loc_scale: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Inverse transform.

        x: normalized tensor, same shape as in forward.
        loc_scale: (loc, scale) from forward, shape (B, V, 1) or broadcastable to it.
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale
        loc = loc.to(x.dtype)
        scale = scale.to(x.dtype)
        while loc.ndim < x.ndim:
            loc = loc.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
        
        if self.use_arcsinh:
            x = torch.sinh(x)

        x_rec = x * scale + loc
        return x_rec.to(orig_dtype)
