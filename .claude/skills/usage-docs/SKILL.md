---
name: usage-docs
description: After changing DeepWind code/scripts/configs, keep the usage docs (docs/*.md) in sync with a teach-me guide - but only when the change warrants it.
---

# usage-docs - keep DeepWind's docs in sync

You are helping maintain the DeepWind repo (Pawsey HPC). After you change code,
scripts, or configs, apply this policy.

## 1. Judge whether a doc update is warranted

Update the docs only when the change affects how someone runs or understands
the project. Update if you changed any of:

- a command / script someone runs (`sbatch`, `python`, hydra overrides)
- a flag, env var, or config key (`--export`, `WANDB_MODE`, `training.*`)
- run behavior (resume/checkpointing, eval protocol, batch size, partitions)
- a new script/file a user must know about

Skip trivial or internal-only changes (typos, a pure refactor with no external
effect, private helper edits). Do not add a doc section for every change.

## 2. Where to write

- Pawsey launch / workflow / monitoring -> `docs/pawsey.md`
- Reproduce-the-paper step-by-step -> `docs/reproducibility.md`
- data / model / baseline specifics -> `docs/data.md`, `docs/baselines.md`

## 3. How to write ("teach me" style)

- Every command must be copy-paste ready and end with a `#` comment saying what
  it does and why (not just what it is).
- Show the full command with its flags, not a fragment.
- Keep it concise: a 3-10 line section beats an unexplained one-liner.

## 4. When in doubt

Prefer to add the doc. A short, correct note is cheap; a missing command is a
support ticket.
