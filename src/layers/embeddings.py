import torch
import numpy as np
from torch import nn
from typing import Tuple, Optional
from transformers.utils import ModelOutput
from dataclasses import dataclass
from jaxtyping import Float, Int, Bool

# Assuming these exist in your project based on original code
from src.layers import InstanceNorm, Patch
from src.layers import ResidualBlock 
from src.models import DeepWindConfig


@dataclass
class DeepWindEmbeddingsOutput(ModelOutput):
    embeddings: Float[torch.Tensor, "batch variate seq embed"]
    scaled_context: Optional[Float[torch.Tensor, "batch variate context_len"]] = None
    loc_scale: Optional[Tuple[
        Float[torch.Tensor, "batch variate 1"],
        Float[torch.Tensor, "batch variate 1"],
    ]] = None
    


class DeepWindEmbeddings(nn.Module):
    """
    Handles input preprocessing:
    1. Truncation
    2. Instance Normalization
    3. Patching & Stats Extraction
    4. Time/Variate/Spatial Embeddings
    5. Projection to d_model
    """
    def __init__(self, config: DeepWindConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.input_patch_size

        # 1. Normalization & Patching
        self.instance_norm = InstanceNorm(use_arcsinh=config.use_arcsinh)
        self.patcher = Patch(
            patch_size=config.input_patch_size,
            patch_stride=config.input_patch_stride,
        )
        
        # 2. Input Projection
        # Input dim = (TimeEnc + Value + Mask) * PatchSize + Stats
        # TimeEnc(P) + Value(P) + Mask(Nothing? usually implied) + Stats(9)
        # Based on original code: 2 * P + 9
        if config.use_patch_stats:
            in_dim = self.patch_size * 2 + config.num_patch_stats
        else:
            in_dim = self.patch_size * 2
        
        self.projector = ResidualBlock(
            in_dim=in_dim,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout,
        )

        # 3. Variate Embeddings
        if config.use_variate_embed:
            # +1 for padding index
            self.variate_emb = nn.Embedding(
                config.num_known_variates + 1, 
                config.d_model, 
                padding_idx=config.num_known_variates
            )

        # 4. Spatial Embeddings
        if config.use_coord_embed:
            self.spatial_mlp = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, config.d_model)
            )
            # Learnable embedding for sites without coordinates
            self.unknown_site_embed = nn.Parameter(torch.randn(1, 1, 1, config.d_model) * 0.02)

    def _compute_patch_stats(self, patched_x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Computes statistical features for each patch."""
        # Mean / Std
        mean = patched_x.mean(dim=-1, keepdim=True)
        var = patched_x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)

        # Min / Max / Range
        x_min = patched_x.amin(dim=-1, keepdim=True)
        x_max = patched_x.amax(dim=-1, keepdim=True)
        rng = x_max - x_min

        # Net change
        net_change = patched_x[..., -1:] - patched_x[..., :1]

        # Diff-based stats
        dx = patched_x[..., 1:] - patched_x[..., :-1]
        mean_abs_dx = dx.abs().mean(dim=-1, keepdim=True)
        max_abs_dx = dx.abs().amax(dim=-1, keepdim=True)

        # Energy
        energy = (patched_x ** 2).mean(dim=-1, keepdim=True)

        return torch.cat([
            mean, std, x_min, x_max, rng, net_change, mean_abs_dx, max_abs_dx, energy
        ], dim=-1)

    def _get_spatial_xyz(self, coords: torch.Tensor) -> torch.Tensor:
        """Converts Lat/Lon to 3D Cartesian coordinates."""
        rads = coords * (np.pi / 180.0)
        lon, lat = rads[:, 0], rads[:, 1]
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return torch.stack([x, y, z], dim=-1)

    def forward(
        self, 
        context: torch.Tensor,
        site_coords: Optional[torch.Tensor] = None,
        variate_ids: Optional[torch.Tensor] = None,
        has_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        
        B, V, T = context.shape
        device = context.device
        dtype = context.dtype

        # A. Truncate
        if T > self.config.context_length:
            context = context[..., -self.config.context_length:]
            T = self.config.context_length

        # B. Normalize
        context_norm, loc_scale = self.instance_norm(context)
        context_norm = context_norm.to(dtype)
        scaled_context = context_norm # Keep for AR targets later
            
        # C. Patch
        patched_context = self.patcher(context_norm) # (B, V, L, P)
        num_patches = patched_context.shape[-2]
        
        # D. Stats
        if self.config.use_patch_stats:
            patch_stats = self._compute_patch_stats(patched_context)
        
        # E. Time Encoding
        final_context_len = num_patches * self.patch_size
        time_indices = torch.arange(
            start=-final_context_len, end=0, device=device, dtype=torch.float32
        ).view(num_patches, self.patch_size)
        
        time_enc = (
            time_indices
            .view(1, 1, num_patches, self.patch_size)
            .expand(B, V, num_patches, self.patch_size)
            .div(float(T))
            .to(dtype)
        )

        # F. Concat & Project
        if self.config.use_patch_stats:
            concat_input = torch.cat([time_enc, patched_context, patch_stats], dim=-1)
        else:
            concat_input = torch.cat([time_enc, patched_context], dim=-1)
        embeds = self.projector(concat_input) # (B, V, L, D)

        # G. Add Variate Embeddings
        if self.config.use_variate_embed and variate_ids is not None:
            variate_ids = variate_ids.to(device).long()
            v_emb = self.variate_emb(variate_ids)
            embeds = embeds + v_emb.unsqueeze(2)

        # H. Add Spatial Embeddings
        if self.config.use_coord_embed and site_coords is not None:
            site_coords = site_coords.to(device)
            xyz = self._get_spatial_xyz(site_coords)
            s_emb = self.spatial_mlp(xyz).unsqueeze(1).unsqueeze(2)
            
            if has_coords is not None:
                has_coords = has_coords.to(device).view(B, 1, 1, 1)
                s_emb = has_coords * s_emb + (1 - has_coords) * self.unknown_site_embed
            
            embeds = embeds + s_emb
         
        return DeepWindEmbeddingsOutput(embeds, scaled_context, loc_scale)
    