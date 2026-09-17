# Architecture and reproducibility audit

<details>
<summary>Contents</summary>

- [1. Confirmed implementation](#1-confirmed-implementation)
- [2. Material provenance differences](#2-material-provenance-differences)
- [3. Issues found and disposition](#3-issues-found-and-disposition)
- [4. Gates before full retraining](#4-gates-before-full-retraining)

</details>

This audit distinguishes three targets: the Energy paper, the configuration
stored with `deepwind_large_v5`, and the repository defaults. They must not be
silently treated as identical.

## 1. Confirmed implementation

- Input series are instance-normalised (optionally with `arcsinh`), split into
  patches of 16, concatenated with relative time values, and projected to the
  model width.
- Variable IDs and spherical site-coordinate embeddings are added when present.
- Transformer layers alternate time-wise causal attention and variate-wise
  attention according to the configured schedule.
- Each layer uses pre-RMSNorm, residual attention, and a top-k SwiGLU MoE FFN.
- The quantile head predicts 21 values per point. Training uses next-patch
  pinball loss plus the averaged MoE load-balancing loss.

## 2. Material provenance differences

| Item | Paper | Final checkpoint (`deepwind_large_v5`) | Earlier repository default |
|---|---:|---:|---:|
| Large experts | 8 | 8 | 4 |
| Prediction head | 21 quantiles | 21 quantiles | Student-t |
| RoPE | described as enabled | disabled | enabled |
| xPOS | described as enabled | inactive because RoPE is off | enabled |
| MoE auxiliary weight | 0.02 | 0.01 | 0.02 |
| Corpus mixture | 0.7 WTK / 0.3 other SCADA | absent from recovered config | 0.9 / 0.1 |

The immutable recovered configuration is recorded in
`configs/reproduction/paper_large_actual.yaml`. A future run must explicitly
choose between **checkpoint replication** and **paper-spec replication** and
receive a new experiment name.

## 3. Issues found and disposition

- `use_xpos` was ignored and xPOS was always enabled with RoPE. Fixed by
  plumbing the switch; the final checkpoint behaviour is unchanged because its
  RoPE switch is false.
- `use_variate_atten` was stored but ignored. Fixed so disabling it creates an
  all-time-attention ablation; enabled configurations are unchanged.
- LoRA targeted six hard-coded Base-model layer indices. Fixed to derive all
  variate-attention layers from the instantiated model.
- Gradient checkpointing currently drops MoE auxiliary loss. Do not enable it
  for scientific reproduction until upgraded and regression-tested.
- Window starts are sampled with replacement, whereas the paper describes
  without-replacement cycles. This scientific behaviour difference is
  intentionally not changed during the preservation refactor.
- The paper presents a conceptual block containing both attention axes, while
  the implementation assigns one axis to each Transformer layer. Public model
  depth documentation must use the implementation convention.

## 4. Gates before full retraining

1. Decide checkpoint-replication vs paper-spec configuration.
2. Freeze and checksum train/eval manifests; document WindBench exclusion.
3. Pass unit, one-GPU, multi-GPU, and two-node smoke tests.
4. Verify resume equivalence and WandB artifact persistence.
5. Run a fixed-seed Small-model baseline before committing to Large.

The two-node, four-process DDP gate passed on Setonix as Slurm job `48776871`.

---

**Related docs:** [Asset inventory](asset-inventory.md) · [Baselines & model comparison](baselines.md) · [Data](data.md) · [Model cards](model-cards.md) · [Pawsey workflow](pawsey.md) · [Refactor roadmap](refactor-roadmap.md) · [Reproducibility record](reproducibility.md)

