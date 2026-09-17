#!/usr/bin/env python
"""Print the model card for a leaderboard model.

Usage:
    python tools/model_card.py --list
    python tools/model_card.py deepwind-base
    python tools/model_card.py baseline-deepar deepwind-small

DeepWind variants mirror configs/model/*.yaml + configs/training/*.yaml;
baselines mirror docs/baselines.md. Parameter counts are recorded from the
training log ("Trainable parameters: ..."). Keep this registry in sync with
those configs when they change.
"""

import sys

DEEPWIND = {
    "deepwind-small": {
        "params": "33.21 M (trainable)",
        "configs": "configs/model/deepwind_small.yaml, configs/training/deepwind_small.yaml",
        "arch": {
            "d_model/d_ff": "384 / 1024",
            "layers/heads": "6 / 6",
            "dropout": "0.05",
            "context": "8192",
            "patch": "16 / 16",
            "moe": "4 experts, top-2",
            "variate_attn": "every 2 layers",
            "positional": "RoPE + xPos",
            "norm/act": "RMSNorm / SiLU",
            "head": "21-quantile",
        },
        "train": {
            "lr": "1e-4 (cosine, warmup 3% = 3000 steps)",
            "max_steps": "100000",
            "global_batch": "256 (8 x 1 x 32)",
            "adam": "(0.9, 0.95)",
            "weight_decay": "0.01",
            "grad_clip": "1.0",
            "precision": "BF16",
        },
    },
    "deepwind-base": {
        "params": "888.24 M (trainable)",
        "configs": "configs/model/deepwind_base.yaml, configs/training/deepwind_base.yaml",
        "arch": {
            "d_model/d_ff": "1024 / 2816",
            "layers/heads": "12 / 16",
            "dropout": "0.05",
            "context": "8192",
            "patch": "16 / 16",
            "moe": "8 experts, top-2",
            "variate_attn": "every 2 layers",
            "positional": "RoPE + xPos",
            "norm/act": "RMSNorm / SiLU",
            "head": "21-quantile",
        },
        "train": {
            "lr": "1e-4 (cosine, warmup 3% = 3000 steps)",
            "max_steps": "100000",
            "global_batch": "256 (4 x 2 x 32)",
            "adam": "(0.9, 0.95)",
            "weight_decay": "0.01",
            "grad_clip": "1.0",
            "precision": "BF16",
        },
    },
    "deepwind-large": {
        "params": "~1.3 B (estimate - not yet measured)",
        "configs": "configs/model/deepwind_large.yaml, configs/training/deepwind_large.yaml",
        "arch": {
            "d_model/d_ff": "1024 / 2816",
            "layers/heads": "18 / 16",
            "dropout": "0.1",
            "context": "8192",
            "patch": "16 / 16",
            "moe": "8 experts, top-2",
            "variate_attn": "every 2 layers",
            "positional": "RoPE + xPos",
            "norm/act": "RMSNorm / SiLU",
            "head": "21-quantile",
        },
        "train": {
            "lr": "1e-4 (cosine, warmup 3% = 3000 steps)",
            "max_steps": "100000",
            "global_batch": "256 (4 x 2 x 32)",
            "adam": "(0.9, 0.95)",
            "weight_decay": "0.01",
            "grad_clip": "1.0",
            "precision": "BF16",
        },
    },
}

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


def _kv(table, width=16):
    return "\n".join("  %-*s %s" % (width, k, v) for k, v in table.items())


def show_deepwind(mid):
    d = DEEPWIND[mid]
    print("== %s ==" % mid)
    print("parameters:  %s" % d["params"])
    print("architecture:")
    print(_kv(d["arch"]))
    print("training:")
    print(_kv(d["train"]))
    print("configs:     %s" % d["configs"])


def show_baseline(mid):
    family, spec = BASELINES[mid]
    print("== %s ==" % mid)
    print("family:      %s" % family)
    print("spec:        %s" % spec)
    print("training:    %s" % BASELINE_COMMON)


def main(argv):
    if not argv or argv[0] in ("--list", "-l"):
        print("DeepWind variants:")
        for m in DEEPWIND:
            print("  %s" % m)
        print("Baselines:")
        for m in BASELINES:
            print("  %s" % m)
        return 0
    rc = 0
    for mid in argv:
        if mid in DEEPWIND:
            show_deepwind(mid)
        elif mid in BASELINES:
            show_baseline(mid)
        else:
            print("unknown model_id: %r (run with --list)" % mid, file=sys.stderr)
            rc = 2
        print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
