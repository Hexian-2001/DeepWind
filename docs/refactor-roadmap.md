# DeepWind Engineering Refactor Roadmap

> Status: **in progress** — phases are executed top-to-bottom; each phase is
> independently committable and does not change model/training/inference *logic*.

This document is the working plan for the open-source release hardening of the
DeepWind repository. It maps the 9-point directive to concrete, reviewable
work items. The overriding constraint is **"do not break existing
architecture, training, or inference logic first"** — optimisation and logic
changes come later, in a separate pass.

---

## Guiding principles

1. **Code on `/software`, data on `/scratch`.** Repositories and environments
   live under `$DEEPWIND_PROJECT_ROOT`; the corpus, checkpoints, logs, and
   results live under `/scratch`. `/software` has a 250k-inode quota, so conda
   environments must stay on `/scratch`.
2. **Everything in English** — comments, docstrings, log messages, and
   user-facing strings. Internationalisation is a hard requirement.
3. **Path-free code.** No hardcoded `/scratch/pawsey0115/hwang4/...` or
   `/software/...` literals in source; all locations resolve from environment
   variables with documented defaults.
4. **Industrial-grade experiment standards.** Every run is self-documenting:
   a run name convention, a provenance manifest (git commit, resolved config,
   `pip freeze`, env snapshot), and WandB integration. Runs must be
   reproducible and comparable.

---

## Standard environment variables

| Variable                   | Purpose                        | Default (Pawsey)                                        |
|----------------------------|--------------------------------|---------------------------------------------------------|
| `DEEPWIND_PROJECT_ROOT`    | Repository root (software)     | `/software/projects/pawsey0115/hwang4/research_projects/DeepWind` |
| `DEEPWIND_DATA_ROOT`       | Corpus root (scratch)          | `/scratch/pawsey0115/hwang4/deepwindData`               |
| `DEEPWIND_RUNS_ROOT`       | Training runs + logs (scratch) | `/scratch/pawsey0115/hwang4/deepwind-open/runs`         |
| `DEEPWIND_RESULTS_ROOT`    | Evaluation results (scratch)   | `/scratch/pawsey0115/hwang4/results/DeepWind-Research`  |
| `DEEPWIND_VENV`            | Conda env (scratch)            | `/scratch/pawsey0115/hwang4/conda_envs/deepwind`        |

These are documented in `docs/pawsey.md` and consumed uniformly by Hydra
configs (`${oc.env:...}`) and analysis scripts (`os.environ.get(...)`).

---

## Phase 0 — Environment (point 8) — *done / in verification*

- [x] Build a self-hosted conda env at `$DEEPWIND_VENV` on `/scratch`
      (miniforge3 + pip `torch==2.7.1` ROCm 6.3 wheels + full stack).
- [x] Pin the verified transformer stack (`transformers`, `peft`,
      `accelerate`).
- [ ] GPU smoke test outside the Singularity container (in flight).

## Phase 1 — Internationalise to English (point 9)

- [ ] Translate the 11 Chinese comments across 6 files
      (`notebooks/debug_model.ipynb`, `paper/gen_fig_13.py`,
      `src/inference/generator.py`, `src/layers/heads.py`,
      `src/layers/norm.py`, `src/utils/metrics.py`).

## Phase 2 — Packaging hygiene (industrial-grade)

- [ ] Add missing `accelerate` dependency to `pyproject.toml`.
- [ ] Fix copy-paste header comments in `configs/training/*.yaml`.
- [ ] Make `deepwind_base.yaml` `output_dir` consistent with small/large
      (include `${model_name}`).

## Phase 3 — De-hardcode paths (points 3, 6)

- [ ] Replace hardcoded `/scratch/...` defaults in `paper/*.py` and
      `tools/*.py` with `os.environ.get("DEEPWIND_*", <legacy default>)`.
- [ ] Fix stale `project_codes/deepwind_research/configs/eval.yaml` references
      to point at `$DEEPWIND_PROJECT_ROOT/configs/eval.yaml`.

## Phase 4 — Finetune entry-point refactor (points 4, 5)

- [ ] Rewrite `finetune.py` to use Hydra + `DeepWindTrainer` + WandB + the
      registry, matching `train.py`'s standard. Preserve LoRA r=16 / α=32,
      AdamW lr=1e-4, grad-clip 1.0, cosine schedule.
- [ ] Add `configs/finetune.yaml` (and a data config).

## Phase 5 — User-facing inference CLI (point 4)

- [ ] Add a standalone `infer.py` (or `scripts/infer.py`) that loads a
      checkpoint (+ optional LoRA adapter) and produces a single-site forecast
      from a `.npy` file or inline array, with `--output` JSON/NPZ.

## Phase 6 — Provenance & experiment standards (point 5)

- [ ] Add `src/utils/provenance.py`: writes `metadata/run_info.json` per run
      (git commit, dirty-tree flag, resolved Hydra config, `pip freeze`,
      filtered env snapshot).
- [ ] Wire it into `train.py` / `finetune.py` / `evaluate.py`.
- [ ] Adopt a run-name convention: `deepwind-{size}-{variant}-{seed}-{date}`.

## Phase 7 — Data hygiene (points 3, 6)

- [ ] Split `train_metadata.csv` into `train_metadata.csv` /
      `eval_metadata.csv` / `test_metadata.csv` manifests.
- [ ] Fix 3 trailing-space `dataset` values (`windtoolkit `, `windtoolkit  `,
      `scada `).
- [ ] Add a split-isolation audit script.

## Phase 8 — Scratch path standardisation & de-dup (point 6)

- [ ] Verify canonical symlinks under `/scratch/.../projects/deepwind/`.
- [ ] De-dup the two ~80 GB `deepwind_large_v5` trees (checksums first).
- [ ] Remove legacy paths only after verification. **Do not move the corpus.**

## Phase 9 — Security & release (point 7)

- [ ] Revoke the plaintext HF token in `hf_utils/upload_to_hf.py`.
- [ ] Push `release/open-source-v1` to GitHub (needs auth).
- [ ] Retrain Small/Base/Large to paper spec (RoPE+xPOS=true, λ=0.02).
