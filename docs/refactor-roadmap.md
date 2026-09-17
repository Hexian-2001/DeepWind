# DeepWind Engineering Refactor Roadmap

**Contents**

- 1. Guiding principles
- 2. Standard environment variables
- 3. Phase 0 — Environment (point 8) — *done*
- 4. Phase 1 — Internationalise to English (point 9)
- 5. Phase 2 — Packaging hygiene (industrial-grade)
- 6. Phase 3 — De-hardcode paths (points 3, 6)
- 7. Phase 4 — Finetune entry-point refactor (points 4, 5)
- 8. Phase 5 — User-facing inference CLI (point 4)
- 9. Phase 6 — Provenance & experiment standards (point 5)
- 10. Phase 7 — Data hygiene (points 3, 6)
- 11. Phase 8 — Scratch path standardisation & de-dup (point 6)
- 12. Phase 9 — Security & release (point 7)

> Status: **in progress** — phases are executed top-to-bottom; each phase is
> independently committable and does not change model/training/inference *logic*.

This document is the working plan for the open-source release hardening of the
DeepWind repository. It maps the 9-point directive to concrete, reviewable
work items. The overriding constraint is **"do not break existing
architecture, training, or inference logic first"** — optimisation and logic
changes come later, in a separate pass.

---

## 1. Guiding principles

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

## 2. Standard environment variables

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

## 3. Phase 0 — Environment (point 8) — *done*

- [x] Build a self-hosted conda env at `$DEEPWIND_VENV` on `/scratch`
      (miniforge3 + pip `torch==2.7.1` ROCm 6.3 wheels + full stack).
- [x] Pin the verified transformer stack (`transformers`, `peft`,
      `accelerate`).
- [x] GPU smoke test outside the Singularity container — MI250X,
      torch 2.7.1+rocm6.3: fwd/bwd + `save_pretrained`/`from_pretrained`
      round-trip + LoRA injection all OK.

## 4. Phase 1 — Internationalise to English (point 9)

- [x] Translate the 11 Chinese comments across 6 files
      (`notebooks/debug_model.ipynb`, `paper/gen_fig_13.py`,
      `src/inference/generator.py`, `src/layers/heads.py`,
      `src/layers/norm.py`, `src/utils/metrics.py`).

## 5. Phase 2 — Packaging hygiene (industrial-grade)

- [x] Add missing `accelerate` dependency to `pyproject.toml`.
- [x] Fix copy-paste header comments in `configs/training/*.yaml`.
- [x] Make `deepwind_base.yaml` `output_dir` consistent with small/large
      (include `${model_name}`).

## 6. Phase 3 — De-hardcode paths (points 3, 6)

- [x] Replace hardcoded `/scratch/...` defaults in `paper/*.py` and
      `tools/*.py` with `os.environ.get("DEEPWIND_*", <legacy default>)`.
- [x] Fix stale `project_codes/deepwind_research/configs/eval.yaml` references
      to point at `$DEEPWIND_PROJECT_ROOT/configs/eval.yaml`.

## 7. Phase 4 — Finetune entry-point refactor (points 4, 5)

- [x] Rewrite `finetune.py` to use Hydra + `DeepWindTrainer` + WandB + the
      registry, matching `train.py`'s standard. Preserve LoRA r=16 / α=32,
      AdamW lr=1e-4, grad-clip 1.0, cosine schedule.
- [x] Add `configs/finetune.yaml` (and a data config).

## 8. Phase 5 — User-facing inference CLI (point 4)

- [x] Add a standalone `infer.py` (or `scripts/infer.py`) that loads a
      checkpoint (+ optional LoRA adapter) and produces a single-site forecast
      from a `.npy` file or inline array, with `--output` JSON/NPZ.

## 9. Phase 6 — Provenance & experiment standards (point 5)

- [x] Add `src/utils/provenance.py`: writes `run_info.json` per run
      (git commit, dirty-tree flag, resolved Hydra config, `pip freeze`,
      filtered env snapshot).
- [x] Wire it into `train.py` / `finetune.py` / `evaluate.py`.
- [x] Adopt a run-name convention: `deepwind-{size}-{variant}-{seed}-{date}`
      (exposed as `provenance.make_run_name`).

## 10. Phase 7 — Data hygiene (points 3, 6)

- [x] Train/eval/test manifests already exist on scratch (127455 / 150 / 15
      rows) — no split needed.
- [x] Fix 3 trailing-space `dataset` values + drop 158 stale train-manifest
      rows + resolve the 7 leaked test files. Applied with
      `tools/clean_data_manifests.py` (idempotent, `--dry-run`, timestamped
      backups). The 7 leaked files (`15321`, `47663`, `115915`,
      `csg_wind_6`, `opsd_eu_15t_{18,19}`, `kelmarsh_6`) are **kept in
      `train/`** (participate in pretraining) and removed from `test/`; the
      byte-identical `test/` copies were quarantined to `_removed_from_test/`.
      Final: train 127297, eval 150, test 8 (paper's WindBench). Audit now
      PASSES.
- [x] Add a split-isolation audit script (`tools/audit_data_splits.py`).

## 11. Phase 8 — Scratch path standardisation & de-dup (point 6)

- [x] Verify canonical symlinks under `/scratch/.../projects/deepwind/`
      (`legacy/` holds symlinks to `deepwind`, `deepwind_experiments`,
      `baselines`, `results/DeepWind-Research`, `results/summary`).
- [x] De-dup the two ~80 GB `deepwind_large_v5` checkpoint trees: full md5
      (128 files × 2) byte-identical; deleted the old
      `deepwind_experiments/checkpoints/deepwind_large_v5` copy, kept
      `results/DeepWind-Research/checkpoints/deepwind/deepwind_large_v5`.
- [x] Removed only the verified-duplicate; corpus untouched.

## 12. Phase 9 — Security & release (point 7)

- [x] Remove the plaintext HF token in `hf_utils/upload_to_hf.py`
      (replaced with `os.environ["HF_TOKEN"]`; verified not in repo or git
      history). **Action required: rotate the token on huggingface.co.**
- [ ] Push `release/open-source-v1` to GitHub (needs auth).
- [ ] Retrain Small/Base/Large to paper spec (RoPE+xPOS=true, λ=0.02).

---

**Related docs:** [Architecture audit](architecture-audit.md) · [Asset inventory](asset-inventory.md) · [Baselines & model comparison](baselines.md) · [Data](data.md) · [Model cards](model-cards.md) · [Pawsey workflow](pawsey.md) · [Reproducibility record](reproducibility.md)

