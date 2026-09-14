import torch
import torch.nn as nn


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        """
        Construct the Patch module.
        
        Args:
            patch_size: The length of each patch.
            patch_stride: The stride (step size) between patches.
        """
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Variate, Time).
        
        Returns:
            Tensor of shape (Batch, Variate, Num_Patches, Patch_Size).
        """
        length = x.shape[-1]
        
        # 1. Padding Logic
        # Calculate the remainder to determine if padding is needed.
        # We ensure the total length is a multiple of patch_size.
        remainder = length % self.patch_size
        
        if remainder != 0:
            pad_len = self.patch_size - remainder
            
            # Create padding tensor.
            # CRITICAL FIX: Changed fill_value from torch.nan to 0.0.
            # Using NaN will cause the gradients to fail (NaN loss).
            padding_size = (
                *x.shape[:-1],
                pad_len
            )
            padding = torch.full(
                size=padding_size, 
                fill_value=0.0, 
                dtype=x.dtype, 
                device=x.device
            )

            # Left Padding:
            # We concatenate padding to the left (beginning) of the sequence.
            # This ensures the last patch ends exactly at the last timestamp of the original data,
            # which preserves the alignment of the most recent data (important for forecasting).
            x = torch.cat((padding, x), dim=-1)

        # 2. Unfold (Patching)
        # Reshape the time dimension into patches.
        # Output shape: (Batch, Variate, Num_Patches, Patch_Size)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
                                    
        return x
    