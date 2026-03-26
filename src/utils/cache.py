import torch
from typing import Optional, Tuple


class KVCache:
    """
    A unified Key-Value Cache manager for autoregressive generation.

    This implementation uses pre-allocated tensors (Static Cache) to avoid 
    dynamic memory allocation overhead during the generation loop.
    It manages the KV cache for all layers in the model.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """
        Parameters
        ----------
        max_batch_size : int
            Maximum batch size expected during inference.
        max_seq_len : int
            Maximum sequence length (context window + generation length).
        num_layers : int
            Number of Transformer layers in the model.
        num_heads : int
            Number of attention heads per layer.
        head_dim : int
            Dimension of each attention head.
        dtype : torch.dtype
            Data type for the cache (usually float16 or bfloat16).
        device : torch.device
            Device to store the cache on.
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Initialize the current sequence length counter.
        # This points to the next empty slot in the sequence dimension.
        self._current_seq_len = 0

        # Pre-allocate memory for all layers.
        # Shape: (num_layers, 2, max_batch_size, num_heads, max_seq_len, head_dim)
        # The '2' dimension corresponds to (Key, Value).
        # We use a single monolithic tensor or a list of tensors. 
        # A list of tensors is often preferred to allow layers to be on different devices 
        # (pipeline parallelism), but here we assume single device for simplicity.
        self.cache = [
            torch.zeros(
                (2, max_batch_size, num_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=device
            )
            for _ in range(num_layers)
        ]

    def get_seq_len(self) -> int:
        """Returns the current sequence length stored in the cache."""
        return self._current_seq_len

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache for a specific layer with new Key/Value states 
        and returns the full (past + current) Key/Value states for attention.

        Parameters
        ----------
        layer_idx : int
            Index of the current transformer layer.
        key_states : torch.Tensor
            Current step's Key tensor. Shape: (batch, num_heads, 1, head_dim)
        value_states : torch.Tensor
            Current step's Value tensor. Shape: (batch, num_heads, 1, head_dim)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The full concatenated Key and Value tensors valid for the current step.
            Shape: (batch, num_heads, current_seq_len + 1, head_dim)
        """
        # Ensure we haven't exceeded the buffer
        batch_size = key_states.shape[0]
        if self._current_seq_len >= self.max_seq_len:
            raise ValueError(
                f"KV Cache is full. Current length: {self._current_seq_len}, "
                f"Max length: {self.max_seq_len}"
            )

        # Update cache at the specific slot
        # self.cache[layer_idx] shape: (2, max_bs, n_heads, max_seq, head_dim)
        
        # Insert Key
        self.cache[layer_idx][0, :batch_size, :, self._current_seq_len : self._current_seq_len + 1, :] = key_states
        # Insert Value
        self.cache[layer_idx][1, :batch_size, :, self._current_seq_len : self._current_seq_len + 1, :] = value_states

        # Retrieve the valid portion for attention computation
        # Note: We slice up to current_len + 1 because we just added the new token
        k_out = self.cache[layer_idx][0, :batch_size, :, : self._current_seq_len + 1, :]
        v_out = self.cache[layer_idx][1, :batch_size, :, : self._current_seq_len + 1, :]

        return k_out, v_out

    def increment(self) -> None:
        """
        Increments the sequence length pointer. 
        Must be called ONCE per generation step, after all layers have been processed.
        """
        self._current_seq_len += 1

    def reset(self) -> None:
        """Resets the cache pointer to zero (does not free memory, just logically resets)."""
        self._current_seq_len = 0
