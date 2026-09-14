"""Regression tests for configuration switches and layer schedules."""

from src.models.backbone import DeepWindBackbone
from src.models.configuration import DeepWindConfig
from src.utils.constants import AttentionAxis


def _config(**overrides):
    values = dict(
        d_model=32,
        num_layers=6,
        num_heads=4,
        d_ff=64,
        use_moe=False,
        use_rotary_emb=True,
        use_xpos=False,
        variate_atten_every_n_layers=2,
    )
    values.update(overrides)
    return DeepWindConfig(**values)


def test_xpos_switch_is_respected():
    model = DeepWindBackbone(_config())
    assert model.rotary_emb is not None
    assert model.rotary_emb.use_xpos is False


def test_variate_attention_can_be_disabled():
    model = DeepWindBackbone(_config(use_variate_atten=False))
    assert all(layer.attention_axis == AttentionAxis.TIME for layer in model.layers)


def test_default_mixed_schedule():
    model = DeepWindBackbone(_config())
    assert [layer.attention_axis for layer in model.layers] == [
        AttentionAxis.TIME,
        AttentionAxis.VARIATE,
    ] * 3
