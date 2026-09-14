from typing import List, Tuple

import torch


class KVCache:
    """
    Key-Value cache for autoregressive (expand-collapse) decoding.

    Stores, per layer, the Key/Value tensors produced at each decoding step so
    past keys/values are reused instead of recomputed. This is a simple dynamic
    cache: chunks are appended per step and concatenated along the sequence
    dimension on read, avoiding any pre-allocation or max-length bookkeeping.

    The interface mirrors what ``src.layers.attention.BaseMultiheadAttention``
    expects:

        - ``seq_len(layer_idx)``      -> number of tokens cached for that layer
        - ``append(layer_idx, (k, v))`` -> store the current step's Key/Value
        - ``cache[layer_idx]``        -> full (past + current) Key/Value
    """

    def __init__(self, num_layers: int) -> None:
        self._keys:   List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        self._values: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        self._lengths: List[int] = [0] * num_layers

    def __len__(self) -> int:
        return len(self._keys)

    def seq_len(self, layer_idx: int) -> int:
        """Number of tokens currently cached for ``layer_idx``."""
        return self._lengths[layer_idx]

    def append(self, layer_idx: int, kv: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Store the current step's (Key, Value) for ``layer_idx``."""
        k, v = kv
        self._keys[layer_idx].append(k)
        self._values[layer_idx].append(v)
        self._lengths[layer_idx] += k.shape[-2]

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the full (past + current) Key/Value for ``layer_idx``."""
        if not self._keys[layer_idx]:
            raise IndexError(f"KVCache layer {layer_idx} is empty.")
        return (
            torch.cat(self._keys[layer_idx], dim=-2),
            torch.cat(self._values[layer_idx], dim=-2),
        )

    def reset(self) -> None:
        """Drop all cached tensors, logically resetting the cache to empty."""
        for i in range(len(self._keys)):
            self._keys[i].clear()
            self._values[i].clear()
            self._lengths[i] = 0
