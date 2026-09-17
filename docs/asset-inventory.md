# Pawsey asset inventory

<details>
<summary>Contents</summary>

- [1. De-duplication candidate (no deletion performed)](#1-de-duplication-candidate-no-deletion-performed)

</details>

The metadata-only inventory generated on 2026-09-14 is stored outside the public
repository at:

```text
/software/projects/pawsey0115/hwang4/project_admin/deepwind/inventory-2026-09-14.json
```

It completed without traversal errors and recorded:

| Legacy root | Files | Logical size | Classification |
|---|---:|---:|---|
| `deepwindData` | 127,485 | 2,239,549,241,822 B | processed corpus |
| `deepwind_experiments` | 3,491 | 268,614,752,953 B | historical runs/checkpoints |
| `results/DeepWind-Research` | 2,394 | 117,324,191,424 B | paper-era weights/results |
| `deepwind` | 16 | 2,031,870 B | early Slurm logs |
| `project_codes/deepwind_research` | 1,054 | 72,893,273 B | legacy code and generated figures |

The canonical non-destructive view is now:

```text
/scratch/pawsey0115/hwang4/datasets/wind_power/deepwind_corpus_v1
/scratch/pawsey0115/hwang4/projects/deepwind/
├── checkpoints/paper/deepwind-large
├── legacy/{experiments,paper-results,early-logs,comparison-baselines,summary}
├── logs/
├── release/
├── reports/
├── runs/
└── wandb/
```

These entries are symlinks or new empty directories. No legacy asset has been
deleted, renamed, or copied. De-duplication is deferred until checksums, release
artifacts, and paper-result provenance are verified.

The legacy view also links `comparison-baselines` and `summary`, which were
found outside directories containing the DeepWind name. The experiment
catalogue at
`/scratch/pawsey0115/hwang4/projects/deepwind/reports/experiment-catalog-2026-09-14.json`
currently identifies 23 Trainer runs.

## 1. De-duplication candidate (no deletion performed)

The two `deepwind_large_v5` checkpoint trees below are each approximately
80 GB, contain the same 128 relative file names and sizes, and have matching
hashes for the root model configuration and safetensors index:

```text
/scratch/pawsey0115/hwang4/deepwind_experiments/checkpoints/deepwind_large_v5
/scratch/pawsey0115/hwang4/results/DeepWind-Research/checkpoints/deepwind/deepwind_large_v5
```

They remain untouched. Payload checksums for every file and explicit owner
approval are required before replacing one tree with a symlink or removing it.

---

**Related docs:** [Architecture audit](architecture-audit.md) · [Baselines & model comparison](baselines.md) · [Data](data.md) · [Model cards](model-cards.md) · [Pawsey workflow](pawsey.md) · [Refactor roadmap](refactor-roadmap.md) · [Reproducibility record](reproducibility.md)

