"""Regression test: gradient checkpointing must not drop the MoE aux loss.

Compares the non-checkpointed and checkpointed forward/backward paths for a
tiny MoE model and asserts the router receives a non-zero gradient from the
load-balancing term in both cases.
"""
import torch

from src.models.configuration import DeepWindConfig
from src.models.deepwind import DeepWindModel


def make_model() -> DeepWindModel:
    cfg = DeepWindConfig(
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        dropout=0.0,
        context_length=64,
        input_patch_size=16,
        input_patch_stride=16,
        quantiles=[0.1, 0.5, 0.9],
        use_variate_atten=True,
        use_variate_embed=True,
        use_coord_embed=True,
        use_load_balance_loss=True,
        aux_loss_weight=0.02,
        use_rotary_emb=True,
        use_xpos=True,
        use_rms_norm=True,
        use_moe=True,
        num_experts=4,
        num_experts_per_token=2,
        num_known_variates=6,
        num_patch_stats=9,
        dense_act_fn="silu",
        use_arcsinh=True,
        pred_head_type="quantile",
    )
    return DeepWindModel(cfg)


def run(ckpt: bool):
    torch.manual_seed(0)
    model = make_model().cuda()
    model.train()
    model.backbone.gradient_checkpointing = ckpt

    context = torch.randn(2, 3, 64, device="cuda")
    channel_mask = torch.ones(2, 3, device="cuda")
    variate_ids = torch.randint(0, 6, (2, 3), device="cuda")
    site_coords = torch.randn(2, 2, device="cuda")
    has_coords = torch.ones(2, 1, device="cuda")

    out = model(
        context=context,
        channel_mask=channel_mask,
        variate_ids=variate_ids,
        site_coords=site_coords,
        has_coords=has_coords,
    )
    out.loss.backward()

    router = model.backbone.layers[0].ffn.router
    router_grad = (
        router.weight.grad.norm().item() if router.weight.grad is not None else 0.0
    )
    aux = out.aux_loss.item() if out.aux_loss is not None else None
    return out.loss.item(), aux, router_grad


def main() -> None:
    l0, a0, g0 = run(False)
    l1, a1, g1 = run(True)
    print(f"no-ckpt: loss={l0:.6f} aux_loss={a0} router_grad_norm={g0:.6f}")
    print(f"ckpt:    loss={l1:.6f} aux_loss={a1} router_grad_norm={g1:.6f}")

    assert a0 is not None and a0 > 0.0, "baseline aux loss should be positive"
    assert g0 > 0.0, "baseline router should receive gradient"
    assert a1 is not None and a1 > 0.0, "checkpointing dropped aux loss"
    assert g1 > 0.0, "checkpointing dropped router gradient"
    print("PASS")


if __name__ == "__main__":
    main()
