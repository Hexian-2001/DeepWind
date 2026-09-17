#!/usr/bin/env python
"""Print the model card for a registered leaderboard model.

Usage:
    python tools/model_card.py --list
    python tools/model_card.py <model_id> [<model_id> ...]

The registry is ``results/leaderboard.jsonl`` (source of truth for metrics,
architecture and training config). ``--list`` prints every registered model;
passing a ``model_id`` (or a ``small``/``base``/``large`` shorthand) renders
that model's full card: model name, metrics, architecture, parameter count,
hyper-parameters and training config. Parameter counts are recorded from the
training log ("Trainable parameters: ...") and are not stored in the registry;
baselines carry no parameter count.
"""
from __future__ import annotations

import sys
from pathlib import Path

from leaderboard_lib import (
    HEADLINE, HIGHER_IS_BETTER, LOWER_IS_BETTER, load_leaderboard,
)

REPO = Path(__file__).resolve().parents[1]
LEADERBOARD = REPO / "results" / "leaderboard.jsonl"

# Reference parameter counts (measured from the training log). A property of the
# *design*, not the run, so it is kept here rather than in the registry.
DEEPWIND_PARAMS = {
    "small": "33.21 M (trainable)",
    "base": "888.24 M (trainable)",
    "large": "~1.3 B (estimate - not yet measured)",
}

# Curated spec for baselines (mirrors docs/baselines.md). Baselines are stored
# in the registry with empty architecture/training, so their design lives here.
BASELINES = {
    "baseline-arima": ("statistical", "ARIMA(2,1,1), refit per window on last 384 steps, Gaussian residual std"),
    "baseline-lightgbm": ("gradient-boosted", "300 trees, num_leaves=31, lr=0.05, last 96 lags + seasonal lags, direct multi-step"),
    "baseline-dlinear": ("deep learning", "decomposition kernel 25, two linear heads (mean + log-std), Gaussian NLL"),
    "baseline-nbeats": ("deep learning", "2 stacks x 1 block, hidden 256, backcast/forecast residual, Gaussian NLL"),
    "baseline-patchtst": ("deep learning", "patch 32 / stride 16, d_model=96, 8 heads, 2 layers, Gaussian NLL"),
    "baseline-deepar": ("deep learning", "LSTM (hidden 128, 2 layers) over last 256 steps -> linear Gaussian head"),
    "baseline-chronos-2": ("TSFM (zero-shot)", "native 9-quantile grid [0.1 .. 0.9]"),
    "baseline-timesfm-2.5": ("TSFM (zero-shot)", "native 9-quantile grid [0.1 .. 0.9]"),
    "baseline-moirai-2": ("TSFM (zero-shot)", "Moirai-2 (R-small), native 9-quantile grid [0.1 .. 0.9]"),
}

BASELINE_COMMON = "Adam lr=1e-3, batch 64, <=25 epochs, early-stop patience 6, <=8000 windows, z-score normalisation."

# Architecture fields surfaced on a DeepWind card. A value may be a key name or
# a callable taking the architecture dict.
ARCH_FIELDS = [
    ("d_model", "d_model"),
    ("d_ff", "d_ff"),
    ("layers / heads", lambda a: "%s / %s" % (a.get("num_layers"), a.get("num_heads"))),
    ("dropout", "dropout"),
    ("context_length", "context_length"),
    ("patch size / stride", lambda a: "%s / %s" % (a.get("input_patch_size"), a.get("input_patch_stride"))),
    ("MoE", lambda a: ("%s experts, top-%s" % (a.get("num_experts"), a.get("num_experts_per_token"))) if a.get("use_moe") else "off"),
    ("variate attention", lambda a: ("every %s layers" % a.get("variate_atten_every_n_layers")) if a.get("use_variate_atten") else "off"),
    ("positional", lambda a: ("RoPE + xPos" if a.get("use_xpos") else "RoPE") if a.get("use_rotary_emb") else "learned"),
    ("norm / activation", lambda a: "%s / %s" % ("RMSNorm" if a.get("use_rms_norm") else "LayerNorm", a.get("dense_act_fn"))),
    ("prediction head", lambda a: "%s (%s-quantile)" % (a.get("pred_head_type"), len(a.get("quantiles", [])))),
    ("load-balance aux loss", "aux_loss_weight"),
    ("use_arcsinh", "use_arcsinh"),
    ("use_variate_embed", "use_variate_embed"),
    ("use_coord_embed", "use_coord_embed"),
]

# Training fields surfaced on a DeepWind card.
TRAIN_FIELDS = [
    ("learning_rate", "learning_rate"),
    ("scheduler", lambda t: "%s, warmup %s" % (t.get("lr_scheduler_type"), t.get("warmup_ratio"))),
    ("max_steps", "max_steps"),
    ("per-device batch x accum", lambda t: "%s x %s" % (t.get("per_device_train_batch_size"), t.get("gradient_accumulation_steps"))),
    ("adam betas", lambda t: "(%s, %s)" % (t.get("adam_beta1"), t.get("adam_beta2"))),
    ("weight_decay", "weight_decay"),
    ("grad clip", "max_grad_norm"),
    ("precision", lambda t: "BF16" if t.get("bf16") else "FP32"),
]


def _fmt_metric(name, val):
    if val is None:
        return "n/a"
    s = "%.4f" % val if isinstance(val, float) else str(val)
    if name in LOWER_IS_BETTER:
        return "%s  (lower better)" % s
    if name in HIGHER_IS_BETTER:
        return "%s  (higher better)" % s
    return s


def _kv_block(table, width=20, indent="  "):
    out = []
    for label, val in table:
        s = "%g" % val if isinstance(val, float) else str(val)
        out.append("%s%-*s %s" % (indent, width, label, s))
    return "\n".join(out)


def _render(rows, fields):
    out = []
    for label, key in fields:
        if callable(key):
            try:
                val = key(rows)
            except Exception:
                val = "?"
        else:
            val = rows.get(key)
        if val is None:
            continue
        out.append((label, val))
    return out


def list_models():
    rows = load_leaderboard(LEADERBOARD)
    if not rows:
        print("No models registered in %s" % LEADERBOARD)
        return 0
    order = {"small": 0, "base": 1, "large": 2, "baseline": 3}
    rows.sort(key=lambda r: (order.get(str(r.get("variant")), 9), str(r.get("model_id"))))
    print("Registered models (%s):" % LEADERBOARD)
    print("  %-30s %-10s %-10s %s" % ("model_id", "variant", "nCRPS", "step / final_loss"))
    for r in rows:
        mean = (r.get("metrics") or {}).get("mean") or {}
        ncrps = "%.4f" % mean["nCRPS"] if mean.get("nCRPS") is not None else "n/a"
        out = r.get("training_outcome") or {}
        extra = ""
        if out.get("global_step"):
            extra = "step=%s  loss=%s" % (out["global_step"], out.get("final_loss"))
        print("  %-30s %-10s %-10s %s" % (r["model_id"], r.get("variant"), ncrps, extra))
    return 0


def show_card(rec):
    mid = rec["model_id"]
    variant = rec.get("variant", "?")
    print("=" * 78)
    print("%s    [variant: %s]" % (mid, variant))
    print("=" * 78)

    params = DEEPWIND_PARAMS.get(variant)
    if params:
        print("parameters:   %s" % params)
    if rec.get("seed") is not None:
        print("seed:         %s" % rec["seed"])
    if rec.get("git_commit"):
        print("git commit:   %s" % rec["git_commit"])
    if rec.get("config_hash"):
        print("config hash:  %s" % rec["config_hash"])
    if rec.get("checkpoint"):
        print("checkpoint:   %s" % rec["checkpoint"])
    if rec.get("eval_name"):
        print("eval:         %s" % rec["eval_name"])

    out = rec.get("training_outcome") or {}
    if out.get("global_step") is not None or out.get("final_loss") is not None:
        print("\nTraining outcome")
        print("  global_step   %s" % (out.get("global_step", "n/a")))
        print("  final_loss    %s" % (out.get("final_loss", "n/a")))

    m = rec.get("metrics") or {}
    mean = m.get("mean") or {}
    if mean:
        print("\nMetrics (mean over %s datasets / %s cells)" % (
            (m.get("n_datasets") or len(m.get("per_dataset") or {})), m.get("n_cells", "?")))
        for name in HEADLINE:
            print("  %-20s %s" % (name, _fmt_metric(name, mean.get(name))))

    per_ds = m.get("per_dataset") or {}
    if per_ds:
        print("\nnCRPS / nMAE per dataset")
        for ds in sorted(per_ds):
            d = per_ds[ds]
            print("  %-16s nCRPS=%.4f  nMAE=%.4f" % (ds, d.get("nCRPS", float("nan")), d.get("nMAE", float("nan"))))

    per_h = m.get("per_horizon_nCRPS") or {}
    if per_h:
        def _hnum(k):
            try:
                return int(k.split("_")[-1].lstrip("H"))
            except ValueError:
                return 99
        items = sorted(per_h.items(), key=lambda kv: _hnum(kv[0]))
        print("\nnCRPS per horizon")
        print("  " + "  ".join("%s=%.4f" % (k.replace("nCRPS_", ""), v) for k, v in items))

    arch = rec.get("architecture") or {}
    train = rec.get("training") or {}

    if variant in ("small", "base", "large"):
        print("\nArchitecture")
        if arch:
            print(_kv_block(_render(arch, ARCH_FIELDS)))
        else:
            print("  (not stored; see configs/model/deepwind_%s.yaml)" % variant)
        print("\nTraining (hyper-parameters)")
        if train:
            print(_kv_block(_render(train, TRAIN_FIELDS)))
        else:
            print("  (not stored; see configs/training/deepwind_%s.yaml)" % variant)
    else:
        family, spec = BASELINES.get(mid, (str(variant), ""))
        print("\nfamily:       %s" % family)
        print("spec:         %s" % spec)
        print("training:     %s" % BASELINE_COMMON)


def resolve(mid, rows):
    """Exact model_id, else a variant shorthand (small/base/large)."""
    for r in rows:
        if r["model_id"] == mid:
            return r
    for r in rows:
        if r.get("variant") == mid:
            return r
    return None


def main(argv):
    if not argv or argv[0] in ("--list", "-l"):
        return list_models()
    rows = load_leaderboard(LEADERBOARD)
    rc = 0
    for mid in argv:
        rec = resolve(mid, rows)
        if rec is None:
            print("unknown model_id: %r (run with --list)" % mid, file=sys.stderr)
            rc = 2
        else:
            show_card(rec)
            print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
