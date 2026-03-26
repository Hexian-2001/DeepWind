import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union, Dict
from functools import partial
from dataclasses import dataclass
from jaxtyping import Float
from transformers.utils import ModelOutput


ACT2FN: Dict[str, Callable] = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu, # Swish
    "swish": F.silu,
}


@dataclass
class FeedForwardOutput(ModelOutput):
    """
    Container for MoE FFN outputs.

    Attributes
    ----------
    outputs:
        Mixed expert outputs with shape (..., out_dim or in_dim).

    gate_probs:
        Per-token distribution over all experts with shape (B, V, T, E),
        or None when not computed (e.g., in training mode).

    selected_experts:
        Per-token top-k expert indices with shape (B, V, T, K),
        or None when not computed.

    selected_weights:
        Per-token top-k expert weights with shape (B, V, T, K),
        or None when not computed.

    expert_usage:
        Batch-wise fraction of routing slots assigned to each expert D_i,
        with shape (E,), or None when not computed.

    expert_prob_mean:
        Batch-wise mean gating probability P_i over tokens,
        with shape (E,), or None when not computed.
    """
    outputs: torch.Tensor
    gate_probs: Optional[torch.Tensor] = None
    selected_experts: Optional[torch.Tensor] = None
    selected_weights: Optional[torch.Tensor] = None
    expert_usage: Optional[torch.Tensor] = None
    expert_prob_mean: Optional[torch.Tensor] = None


class SwiGLUFFN(nn.Module):
    """
    LLaMA-style SwiGLU Feed-Forward Network.
    
    Structure:
        Gate = Linear(in, hidden)
        Up   = Linear(in, hidden)
        Down = Linear(hidden, out)
        Output = (Act(Gate) * Up) @ Down
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,  # Typically 4 * in_dim * 2/3
        out_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = False,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.act_fn = activation
        
        # LLaMA uses 3 linear layers
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x: torch.Tensor, return_dict=True) -> torch.Tensor:
        # SwiGLU: (Swish(Gate) * Up) -> Down
        x_gate = self.act_fn(self.gate_proj(x))
        x_up = self.up_proj(x)
        x = x_gate * x_up
        x = self.down_proj(x)
        x = self.dropout(x)
        if not return_dict:
            return x 
        return FeedForwardOutput(outputs=x)


class MoEFeedForward(nn.Module):
    """
    MoE FFN with top-k routing and optional load-balancing loss.

    Routing:
        gate_logits = router(x)              # [N, E]
        gate_probs  = softmax(gate_logits)   # [N, E]
        selected_experts = topk(gate_logits) # [N, K]
        weights = softmax(topk_logits)       # [N, K]
        output = sum_k weights_k * expert_k(x)

    Load-balancing loss (per layer):
        D_i = (1/T) * sum_t 1{token t selects expert i}
        P_i = (1/T) * sum_t gate_probs[t, i]
        L_load = E * sum_i D_i * P_i
    """

    def __init__(
            self,
            num_experts: int,
            num_experts_per_token: int,
            in_dim: int,
            hidden_dim: Optional[int] = None,
            out_dim: Optional[int] = None,
            activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
            bias: bool = False,
            ffn_dropout_p: float = 0.0,
            router_bias: bool = False,
            # load-balance loss switch
            use_load_balance_loss: bool = False,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.num_experts_per_token = int(num_experts_per_token)
        self.use_load_balance_loss = use_load_balance_loss

        # Expose the last computed load loss for external aggregation
        self.aux_loss: Optional[torch.Tensor] = None

        self.experts = nn.ModuleList(
            [
                SwiGLUFFN(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    activation=activation,
                    bias=bias,
                    ffn_dropout_p=ffn_dropout_p,
                )
                for _ in range(self.num_experts)
            ]
        )

        # Router: maps token representation -> expert logits
        self.router = nn.Linear(in_dim, self.num_experts, bias=router_bias)

    def _compute_load_balance_loss(
            self,
            gate_probs: torch.Tensor,  # [N, E], softmax over ALL experts
            selected_experts: torch.Tensor,  # [N, K], top-k expert indices per token
    ) -> torch.Tensor:
        """
        Common load-balancing loss implementation (works for K>1).

        Definitions (for a batch with N tokens and E experts):
        - P_i = (1/N) * sum_t gate_probs[t, i]
        - D_i = (1/(N*K)) * sum_t sum_{k=1..K} 1{selected_experts[t, k] == i}
               (i.e., fraction of routing *slots* assigned to expert i)
        - L_load = E * sum_i D_i * P_i

        Properties:
        - sum_i D_i = 1 (normalized over top-k slots)
        - sum_i P_i = 1 (since gate_probs is a distribution per token)
        """

        N, E = gate_probs.shape
        K = selected_experts.shape[-1]

        # P_i: mean gating probability allocated to expert i
        P = gate_probs.mean(dim=0)  # [E]

        # D_i: fraction of top-k routing slots assigned to expert i
        # one_hot: [N, K, E] -> sum over N and K -> [E]
        flat = selected_experts.reshape(-1)  # [N*K]
        slot_counts = torch.bincount(flat, minlength=E).to(gate_probs.dtype)  # [E]
        D = slot_counts / float(N * K)
        
        # L_load: scalar
        return E * torch.sum(D * P)

    #def forward(
    #        self,
    #        x: Float[torch.Tensor, "... in_dim"],
    #) -> Float[torch.Tensor, "... dim"]:
    #    """
    #    Parameters
    #    ----------
    #    x: Token representations with shape (..., in_dim).
#
    #    Returns
    #    -------
    #    outputs:
    #        Weighted mixture of expert outputs with shape (..., out_dim or in_dim).
    #    """
    #    # Flatten all leading dims into a token dimension: [N, in_dim]
    #    x_flat = x.reshape(-1, x.shape[-1])
#
    #    # Compute routing logits: [N, E]
    #    gate_logits = self.router(x_flat)
#
    #    # Top-k routing: select experts per token
    #    topk_logits, selected_experts = torch.topk(
    #        gate_logits, k=self.num_experts_per_token, dim=-1
    #    )  # [N, K], [N, K]
#
    #    # Convert top-k logits -> weights via softmax over selected experts: [N, K]
    #    weights = F.softmax(topk_logits, dim=-1, dtype=torch.float).type_as(x_flat)
#
    #    # Optional: compute load-balance loss (store for external aggregation)
    #    if self.use_load_balance_loss and self.training:
    #        gate_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float).type_as(x_flat)
    #        self.aux_loss = self._compute_load_balance_loss(gate_probs, selected_experts)
    #    else:
    #        self.aux_loss = None
    #        
    #    # Dispatch to experts
    #    N, K = selected_experts.shape
    #    E = self.num_experts
#
    #    # Flatten routing slots
    #    slot_expert = selected_experts.reshape(-1)  # [N*K]
    #    slot_token = torch.arange(N, device=x_flat.device).repeat_interleave(K)  # [N*K]
    #    slot_weight = weights.reshape(-1)  # [N*K]
#
    #    # Sort by expert id so each expert gets a contiguous segment
    #    perm = torch.argsort(slot_expert)  # [N*K]
    #    slot_expert_s = slot_expert[perm]
    #    slot_token_s = slot_token[perm]
    #    slot_weight_s = slot_weight[perm]
#
    #    # Prepare output accumulator
    #    results = torch.zeros_like(x_flat)
#
    #    # Find segment boundaries for each expert
    #    # counts[e] = number of slots assigned to expert e
    #    counts = torch.bincount(slot_expert_s, minlength=E)  # [E]
#
    #    offset = 0
    #    for e, cnt in enumerate(counts.tolist()):
    #        if cnt == 0:
    #            continue
#
    #        idx = slice(offset, offset + cnt)
    #        tok_ids = slot_token_s[idx]  # [cnt]
    #        w = slot_weight_s[idx].unsqueeze(-1)  # [cnt, 1]
#
    #        # Gather tokens for this expert (may contain duplicates if same token routes multiple times to same #expert, rare but possible)
    #        x_e = x_flat[tok_ids]  # [cnt, in_dim]
#
    #        y_e = self.experts[e](x_e, return_dict=False)  # [cnt, out_dim]
    #        y_e = y_e * w
#
    #        # Scatter-add back to token positions
    #        results.index_add_(0, tok_ids, y_e)
#
    #        offset += cnt
#
    #    # Restore output to the original leading dimensions: (..., dim)
    #    outputs = results.view_as(x)
#
    #    # During training, we only return the mixed expert outputs without extra routing info.
    #    if self.training:
    #        return FeedForwardOutput(outputs=outputs)
#
    #    # ---- Additional routing information for analysis/visualization ----
    #    # We want per-token expert distribution and which experts are selected.
    #    # Original input prefix shape, e.g. (batch_size, variates, context_len)
    #    prefix_shape = x.shape[:-1]
#
    #    # Full gating distribution over all experts: [..., E]
    #    gate_probs_full = F.softmax(gate_logits.float(), dim=-1).to(x_flat.dtype)
    #    gate_probs_view = gate_probs_full.view(*prefix_shape, E)           # (B, V, T, E)
#
    #    # Selected experts (top-k indices) per token: [..., K]
    #    selected_experts_view = selected_experts.view(*prefix_shape, K)    # (B, V, T, K)
#
    #    # Selected expert weights (top-k softmax weights) per token: [..., K]
    #    selected_weights_view = weights.view(*prefix_shape, K)             # (B, V, T, K)
#
    #    # ---- Expert-level statistics (batch-wise) ----
    #    # expert_prob_mean: mean gating probability P_i over tokens
    #    expert_prob_mean = gate_probs_full.mean(dim=0)  # (E,)
#
    #    # expert_usage: fraction of routing slots assigned to each expert D_i
    #    flat_idx = selected_experts.reshape(-1)  # [N*K]
    #    slot_counts = torch.bincount(flat_idx, minlength=E).to(gate_probs_full.dtype)
    #    expert_usage = slot_counts / float(N * K)  # (E,)
#
    #    return FeedForwardOutput(
    #        outputs=outputs,
    #        gate_probs=gate_probs_view,             # (B, V, T, E)
    #        selected_experts=selected_experts_view, # (B, V, T, K)
    #        selected_weights=selected_weights_view, # (B, V, T, K)
    #        expert_usage=expert_usage,              # (E,)
    #        expert_prob_mean=expert_prob_mean,      # (E,)
    #    )

    def forward(
        self,
        x: Float[torch.Tensor, "... in_dim"],
    ) -> Float[torch.Tensor, "... dim"]:
        """
        Complete MoE forward pass optimized for DDP.
        Ensures all experts remain in the autograd graph to prevent synchronization hangs.
        """
        # 1. Flatten all leading dims into a token dimension: [N, in_dim]
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        N, D = x_flat.shape

        # 2. Compute routing logits and probabilities: [N, E]
        gate_logits = self.router(x_flat)

        # 3. Top-k routing: select k experts per token
        topk_logits, selected_experts = torch.topk(
            gate_logits, k=self.num_experts_per_token, dim=-1
        )  # [N, K], [N, K]

        # 4. Convert top-k logits to weights via softmax: [N, K]
        weights = F.softmax(topk_logits, dim=-1, dtype=torch.float).type_as(x_flat)

        # 5. Optional: compute load-balance loss (stored for trainer aggregation)
        if self.use_load_balance_loss and self.training:
            gate_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float).type_as(x_flat)
            self.aux_loss = self._compute_load_balance_loss(gate_probs, selected_experts)
        else:
            self.aux_loss = None
            
        # 6. Prepare Expert Dispatching
        K = self.num_experts_per_token
        E = self.num_experts

        # Flatten routing slots for efficient batching
        slot_expert = selected_experts.reshape(-1)  # [N*K]
        slot_token = torch.arange(N, device=x_flat.device).repeat_interleave(K)  # [N*K]
        slot_weight = weights.reshape(-1)  # [N*K]

        # Sort by expert id to process each expert's assigned tokens contiguously
        perm = torch.argsort(slot_expert)  # [N*K]
        slot_expert_s = slot_expert[perm]
        slot_token_s = slot_token[perm]
        slot_weight_s = slot_weight[perm]

        # Accumulator for final combined outputs
        results = torch.zeros_like(x_flat)

        # Count tokens per expert [E]
        counts = torch.bincount(slot_expert_s, minlength=E)

        # --- DDP FIX: Dummy Gradient Sink ---
        # Forces a dependency on every expert in the ModuleList to avoid DDP deadlocks.
        dummy_grad_sink = 0.0

        # 7. Execute Expert Computation
        offset = 0
        for e in range(E):
            cnt = counts[e].item()
            
            if cnt > 0:
                # Normal path: tokens assigned to this expert
                idx = slice(offset, offset + cnt)
                tok_ids = slot_token_s[idx]
                w = slot_weight_s[idx].unsqueeze(-1)  # [cnt, 1]

                # Gather tokens and process through current expert
                x_e = x_flat[tok_ids]
                y_e = self.experts[e](x_e, return_dict=False)
                
                # Weighted scatter-add back to token indices
                results.index_add_(0, tok_ids, y_e * w)
                offset += cnt
            elif self.training:
                # DDP path: expert is idle but must participate in the graph
                # Process 1 zeroed token to register expert params in autograd
                dummy_input = x_flat[0:1] * 0.0
                dummy_out = self.experts[e](dummy_input, return_dict=False)
                dummy_grad_sink = dummy_grad_sink + dummy_out.sum() * 0.0

        # 8. Combine results and restore original shape
        # Adding dummy_grad_sink ensures idle experts are linked to the loss
        outputs = (results + dummy_grad_sink).view(orig_shape)

        # 9. Return for training vs detailed inference
        if self.training:
            return FeedForwardOutput(outputs=outputs)

        # --- Detailed routing info for analysis (Inference only) ---
        prefix_shape = orig_shape[:-1]
        gate_probs_full = F.softmax(gate_logits.float(), dim=-1).to(x_flat.dtype)
        
        return FeedForwardOutput(
            outputs=outputs,
            gate_probs=gate_probs_full.view(*prefix_shape, E),
            selected_experts=selected_experts.view(*prefix_shape, K),
            selected_weights=weights.view(*prefix_shape, K),
            expert_usage=counts.to(x_flat.dtype) / float(N * K),
            expert_prob_mean=gate_probs_full.mean(dim=0),
        )


def build_ffn(
    d_model: int,
    d_ff: int,            # Renamed from intermediate_size to match Config
    dropout: float,
    activation: Union[str, Callable] = "silu",
    use_moe: bool = False,
    num_experts: int = 8,
    num_experts_per_token: int = 2,
    use_load_balance_loss: bool = True,
):
    """
    Factory function to create FFN modules.
    Ensures both Dense and MoE paths return compatible modules (same input/output signature).
    """
        
    # 1. Resolve Activation
    if isinstance(activation, str):
        activation_fn = ACT2FN.get(activation, F.silu)
    else:
        activation_fn = activation

    # 2. Return Factory
    if use_moe:
        return partial(
            MoEFeedForward,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            in_dim=d_model,
            hidden_dim=d_ff,
            out_dim=d_model,
            activation=activation_fn,
            bias=False,
            ffn_dropout_p=dropout,
            use_load_balance_loss=use_load_balance_loss
        )
    else:
        return partial(
            SwiGLUFFN,
            in_dim=d_model,
            hidden_dim=d_ff,
            out_dim=d_model,
            activation=activation_fn,
            bias=False,
            ffn_dropout_p=dropout
        )

