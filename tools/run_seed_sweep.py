#!/usr/bin/env python
# tools/run_seed_sweep.py
"""Submit a seed sweep of a DeepWind variant via train_paper.sbatch.

For each seed ``s`` in ``--seeds``, submits (or prints with ``--dry-run``)::

    sbatch --export=ALL,MODEL=<model>,DEEPWIND_RUN_NAME=deepwind-<model>-paper-seed<s>,DEEPWIND_TRAIN_OVERRIDES=seed=<s> scripts/setonix/train_paper.sbatch

The ``seed=<s>`` is a Hydra top-level override: ``train.py`` calls
``set_seed(cfg.seed)`` so it re-seeds model init, data ordering and RNG.

Why the seed is in ``DEEPWIND_RUN_NAME``: train_paper.sbatch derives the output
dir from a *fixed* run name so ``--requeue`` can resume deterministically. Two
seeds sharing a name would collide on the same ``output_dir``; embedding the
seed keeps each sweep member in its own directory.

Usage:
    python tools/run_seed_sweep.py --model small --seeds 42,43,44 --dry-run
    python tools/run_seed_sweep.py --model base  --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SBATCH = "scripts/setonix/train_paper.sbatch"
VALID_MODELS = {"small", "base", "large"}


def _err(msg):
    print(msg, file=sys.stderr)
    return 2


def build_command(model: str, seed: int) -> list[str]:
    run_name = f"deepwind-{model}-paper-seed{seed}"
    export = f"ALL,MODEL={model},DEEPWIND_RUN_NAME={run_name},DEEPWIND_TRAIN_OVERRIDES=seed={seed}"
    return ["sbatch", f"--export={export}", str(REPO_ROOT / TRAIN_SBATCH)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=sorted(VALID_MODELS))
    ap.add_argument("--seeds", required=True, help="comma-separated seeds, e.g. 42,43,44")
    ap.add_argument("--dry-run", action="store_true", help="print commands without submitting")
    args = ap.parse_args()

    try:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        return _err(f"invalid --seeds value: {args.seeds!r}")
    if not seeds:
        return _err("--seeds produced no values")

    for s in seeds:
        cmd = build_command(args.model, s)
        print(" ".join(cmd))
        if not args.dry_run:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stderr, file=sys.stderr)
                return _err(f"sbatch failed for seed {s} (exit {r.returncode})")
            # sbatch prints "Submitted batch job <id>" on stdout.
            print("   " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "submitted"))

    print(f"\n{len(seeds)} seed job(s) {'would be' if args.dry_run else ''} submitted for {args.model}.", file=sys.stderr)
    print("After each finishes: eval + `python tools/register_eval.py <eval_dir> --variant "
          f"{args.model} --tags paper-spec,seed-sweep` to backfill the leaderboard.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
