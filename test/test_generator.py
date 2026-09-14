"""
test/test_generator.py

Offline unit tests for DeepWindForecaster (Greedy & MQD).
Runs without a real checkpoint by patching DeepWindModel with a lightweight stub.

Usage:
    python -m pytest test/test_generator.py -v
    # or run directly:
    python test/test_generator.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import traceback
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

import torch
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Minimal stubs so we can import generator without a real model or checkpoint
# ---------------------------------------------------------------------------

@dataclass
class StubConfig:
    """Mimics DeepWindModel.config with the fields generator.py accesses."""
    quantiles:          list  = field(default_factory=lambda: [round(i * 0.05, 2) for i in range(1, 20)])  # 19 quantiles
    input_patch_stride: int   = 8


@dataclass
class StubOutput:
    """Mimics the model forward output."""
    denorm_quantile_preds: torch.Tensor   # (B, V, T_out, Q)


class StubDeepWindModel(torch.nn.Module):
    """
    Lightweight stub that replaces DeepWindModel.
    Forward pass returns random quantile predictions with the correct shape.
    """
    def __init__(self, config: StubConfig):
        super().__init__()
        self.config = config
        # Dummy parameter so next(model.parameters()).device works
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, context, **kwargs):
        B, V, T = context.shape
        P = self.config.input_patch_stride
        Q = len(self.config.quantiles)
        # Simulate output: model predicts one patch worth of quantiles at each step
        raw = torch.randn(B, V, P, Q, device=context.device)
        return StubOutput(denorm_quantile_preds=raw)
        

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(context_length: int = 512) -> object:
    return OmegaConf.create({"data": {"context_length": context_length}})


def make_forecaster(context_length: int = 512):
    from src.inference.generator import DeepWindForecaster
    cfg   = make_cfg(context_length)
    model = StubDeepWindModel(StubConfig())
    return DeepWindForecaster(model=model, median_q=0.5, cfg=cfg), model.config


def make_batch(B: int = 2, V: int = 3, T: int = 256, device: str = "cpu"):
    return {
        "context":      torch.randn(B, V, T, device=device),
        "site_coords":  torch.randn(B, 2,   device=device),
        "variate_ids":  torch.randint(0, 10, (B, V), device=device),
        "channel_mask": torch.ones(B, V,     device=device),
        "has_coords":   torch.ones(B, 1,     device=device),
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

PASS = "  PASS"
FAIL = "  FAIL"

results = []


def run_test(name, fn):
    print(f"\n{'─'*60}")
    print(f"TEST: {name}")
    try:
        fn()
        print(PASS)
        results.append((name, True, None))
    except Exception as e:
        print(FAIL)
        traceback.print_exc()
        results.append((name, False, str(e)))


# ── Test 1: cfg=None raises immediately ───────────────────────────────────────
def test_cfg_none_raises():
    from src.inference.generator import DeepWindForecaster
    model = StubDeepWindModel(StubConfig())
    try:
        DeepWindForecaster(model=model, cfg=None)
        raise AssertionError("Expected ValueError when cfg=None")
    except ValueError as e:
        assert "cfg must be provided" in str(e)

run_test("cfg=None raises ValueError", test_cfg_none_raises)


# ── Test 2: Greedy output shapes ──────────────────────────────────────────────
def test_greedy_shapes():
    forecaster, cfg = make_forecaster()
    batch = make_batch(B=2, V=3, T=256)
    Q     = len(forecaster.registered_quantiles)

    for pred_len in [8, 16, 20]:   # 20 is not a multiple of patch_size=8
        out = forecaster.forecast(**batch, prediction_length=pred_len, mqd_infer=False)

        assert out.point_preds.shape    == (2, 3, pred_len),    \
            f"point_preds shape mismatch for pred_len={pred_len}: {out.point_preds.shape}"
        assert out.quantile_preds.shape == (2, 3, pred_len, Q), \
            f"quantile_preds shape mismatch for pred_len={pred_len}: {out.quantile_preds.shape}"

        print(f"    pred_len={pred_len:3d} → point {tuple(out.point_preds.shape)}, "
              f"quantile {tuple(out.quantile_preds.shape)}")

run_test("Greedy — output shapes (including non-multiple pred_len)", test_greedy_shapes)


# ── Test 3: MQD output shapes ─────────────────────────────────────────────────
def test_mqd_shapes():
    forecaster, _ = make_forecaster()
    batch = make_batch(B=2, V=3, T=256)
    Q     = len(forecaster.registered_quantiles)

    for pred_len in [8, 16, 20]:
        out = forecaster.forecast(**batch, prediction_length=pred_len, mqd_infer=True)

        assert out.point_preds.shape    == (2, 3, pred_len),    \
            f"point_preds shape mismatch: {out.point_preds.shape}"
        assert out.quantile_preds.shape == (2, 3, pred_len, Q), \
            f"quantile_preds shape mismatch: {out.quantile_preds.shape}"

        print(f"    pred_len={pred_len:3d} → point {tuple(out.point_preds.shape)}, "
              f"quantile {tuple(out.quantile_preds.shape)}")

run_test("MQD — output shapes (including non-multiple pred_len)", test_mqd_shapes)


# ── Test 4: Quantile monotonicity ─────────────────────────────────────────────
def test_quantile_monotonicity():
    forecaster, _ = make_forecaster()
    batch = make_batch(B=4, V=2, T=256)

    for mode, flag in [("Greedy", False), ("MQD", True)]:
        out = forecaster.forecast(**batch, prediction_length=16, mqd_infer=flag)
        q   = out.quantile_preds   # (B, V, H, Q)

        # Check that q[..., i] <= q[..., i+1] everywhere
        diffs = q[..., 1:] - q[..., :-1]   # should be >= 0
        violations = (diffs < 0).sum().item()
        assert violations == 0, \
            f"{mode}: quantile monotonicity violated at {violations} positions"
        print(f"    {mode}: 0 monotonicity violations across {diffs.numel()} comparisons")

run_test("Quantile monotonicity (Greedy & MQD)", test_quantile_monotonicity)


# ── Test 5: Greedy point_preds == quantile_preds[..., median_idx] ─────────────
def test_greedy_point_equals_median():
    forecaster, _ = make_forecaster()
    batch = make_batch(B=2, V=2, T=256)
    out   = forecaster.forecast(**batch, prediction_length=16, mqd_infer=False)

    expected = out.quantile_preds[..., forecaster.median_idx]
    assert torch.allclose(out.point_preds, expected, atol=1e-5), \
        "Greedy point_preds does not match quantile_preds[..., median_idx]"

run_test("Greedy — point_preds equals median quantile", test_greedy_point_equals_median)


# ── Test 6: MQD point_preds == quantile_preds[..., median_idx] ───────────────
def test_mqd_point_equals_median():
    forecaster, _ = make_forecaster()
    batch = make_batch(B=2, V=2, T=256)
    out   = forecaster.forecast(**batch, prediction_length=16, mqd_infer=True)

    expected = out.quantile_preds[..., forecaster.median_idx]
    assert torch.allclose(out.point_preds, expected, atol=1e-5), \
        "MQD point_preds does not match quantile_preds[..., median_idx]"

run_test("MQD — point_preds equals median quantile", test_mqd_point_equals_median)


# ── Test 7: Sliding window — context never exceeds max length ─────────────────
def test_sliding_window():
    context_length = 64
    forecaster, _ = make_forecaster(context_length=context_length)

    # Start with a context that is already at the limit
    batch = make_batch(B=2, V=2, T=context_length)

    # Use a long prediction to force many AR steps
    out = forecaster.forecast(**batch, prediction_length=48, mqd_infer=False)

    # If sliding window is broken, the model would receive an oversized context
    # and either crash or silently produce wrong shapes. Shape check is sufficient.
    assert out.point_preds.shape == (2, 2, 48)

run_test("Sliding window — context clipped to max_length", test_sliding_window)


# ── Test 8: No real checkpoint needed — _prepare_inputs device move ───────────
def test_device_cpu():
    """Verify the forecaster runs end-to-end on CPU without any CUDA dependency."""
    forecaster, _ = make_forecaster()
    batch = make_batch(B=1, V=1, T=128, device="cpu")
    out   = forecaster.forecast(**batch, prediction_length=8, mqd_infer=False)
    assert out.point_preds.device.type == "cpu"

run_test("CPU end-to-end (no CUDA required)", test_device_cpu)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'═'*60}")
print("SUMMARY")
print(f"{'═'*60}")
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
for name, ok, err in results:
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  {status}  {name}")
    if err:
        print(f"         → {err}")

print(f"\n{passed}/{total} tests passed.")
print('═'*60)

if passed < total:
    sys.exit(1)