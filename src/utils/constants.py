from enum import Enum, auto
from typing import Optional, List, Union, cast
import torch


class AttentionAxis(Enum):
    TIME = auto()
    VARIATE = auto()


def generate_attention_axes(
    num_layers: int, 
    every_n: int, 
    variate_first: bool
) -> List[AttentionAxis]:
    """
    Generates the list of attention types for each layer.
    Flexible logic:
    - -1: All Time
    - 1:  All Variate
    - n:  Insert Variate every n layers
    """
    # Case A: Pure Time-Series
    if every_n == -1:
        return [AttentionAxis.TIME] * num_layers

    # Case B: Pure Variate-Series (New logic)
    if every_n == 1:
        return [AttentionAxis.VARIATE] * num_layers

    # Case C: Mixed Temporal-Variate
    if num_layers % every_n != 0:
        raise ValueError(
            f"num_layers ({num_layers}) must be divisible by every_n ({every_n})."
        )
    
    # Define one block pattern
    # e.g., if n=4, variate_first=True:  [VARIATE, TIME, TIME, TIME]
    # e.g., if n=4, variate_first=False: [TIME, TIME, TIME, VARIATE]
    block = [AttentionAxis.TIME] * (every_n - 1)
    if variate_first:
        block.insert(0, AttentionAxis.VARIATE)
    else:
        block.append(AttentionAxis.VARIATE)
    
    # Repeat block
    num_blocks = num_layers // every_n
    return block * num_blocks


def prepare_variate_atten_mask(
        id_mask: torch.Tensor, 
        dtype: torch.dtype, 
        seq_len: int
    ) -> torch.Tensor:
        """
        Creates an additive attention mask for space-wise attention.

        The mask is broadcastable to shape (Batch * Seq_Len, Num_Heads, Variate, Variate).
        It masks out padding variates (where id_mask == 0).

        Args:
            id_mask: Tensor of shape (Batch, Variate). 1 indicates valid, 0 indicates padding.
            dtype: The target data type (e.g., torch.bfloat16).
            seq_len: The current sequence length to expand the mask along the time axis.

        Returns:
            torch.Tensor: Additive mask of shape (Batch * Seq_Len, 1, 1, Variate).
                          Values are 0.0 for valid positions and -inf for masked positions.
        """
        batch_size, num_variates = id_mask.shape

        # 1. Expand the mask along the time dimension: (Batch, Variate) -> (Batch, Seq_Len, Variate)
        # We use .unsqueeze and .expand to avoid memory copy.
        mask = id_mask.unsqueeze(1).expand(batch_size, seq_len, num_variates)

        # 2. Merge Batch and Seq_Len dimensions to match the "effective batch size"
        # used in space-wise attention (Batch * Seq_Len, Variate).
        mask = mask.reshape(-1, num_variates)

        # 3. Reshape for attention broadcasting: (Batch * Seq_Len, 1, 1, Variate)
        # This allows the mask to be broadcasted across Num_Heads (dim 1) and Queries (dim 2).
        mask = mask.unsqueeze(1).unsqueeze(1)

        # 4. Convert to additive mask:
        # id_mask: 1 = keep, 0 = mask
        # Additive: 0.0 = keep, -inf = mask
        inverted_mask = 1.0 - mask

        # Use the minimum representable value for the dtype to ensure softmax is zero.
        return inverted_mask.to(dtype) * torch.finfo(dtype).min

    