# Pawsey asset inventory

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
├── legacy/{experiments,paper-results,early-logs}
├── logs/
├── release/
├── reports/
├── runs/
└── wandb/
```

These entries are symlinks or new empty directories. No legacy asset has been
deleted, renamed, or copied. De-duplication is deferred until checksums, release
artifacts, and paper-result provenance are verified.
