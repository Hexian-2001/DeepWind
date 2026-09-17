# Data

**Contents**

- 1. Expected processed layout
- 2. Corpus and sampling
- 3. Split isolation
- 4. Redistribution

## 1. Expected processed layout

```text
$DEEPWIND_DATA_ROOT/
├── train/                 # one NumPy array per training series
├── eval/                  # held-out validation arrays
├── test/                  # WindBench arrays
├── train_metadata.csv
└── eval_metadata.csv
```

Each array is `float32` with shape `(variate, time)`. Metadata columns are
`filename`, `latitude`, `longitude`, `variate_ids`, and `dataset`. The loader
pads heterogeneous inputs to `max_vars` and supplies a channel mask. Instance
normalisation is computed online from each sampled context window; capacity is
kept in the evaluation metadata/constants rather than this training CSV.

## 2. Corpus and sampling

The paper reports approximately 562.4 billion observations from WIND Toolkit
and 19 additional sources. It specifies source-level sampling probabilities of
0.7 for WIND Toolkit and 0.3 for the remaining SCADA sources, followed by
adaptive per-file window counts and without-replacement cycles.

The recovered final training configuration predates the current explicit
`dataset_weights` fields. Reproduction work must therefore distinguish:

- the published 0.7/0.3 sampling specification;
- the recovered 2026 training run configuration;
- later experimental defaults (currently 0.9/0.1 in `configs/data/train.yaml`).

Do not silently substitute one for another.

## 3. Split isolation

WindBench sites must not occur in pretraining. For WIND Toolkit targets, retain
the paper's 10 km spatial exclusion buffer. For SCADA sources, exclude the exact
held-out farm or turbine before window generation. A future public preprocessing
release must emit a machine-readable split manifest and contamination audit.

## 4. Redistribution

This repository does not redistribute the 2.1 TB processed corpus. Users must
obtain each source under its own terms. Shanxi Wind is proprietary and cannot be
published. Release download/preprocessing scripts and checksums for public
sources instead of uploading derived data without a license review.

---

**Related docs:** [Architecture audit](architecture-audit.md) · [Asset inventory](asset-inventory.md) · [Baselines & model comparison](baselines.md) · [Model cards](model-cards.md) · [Pawsey workflow](pawsey.md) · [Refactor roadmap](refactor-roadmap.md) · [Reproducibility record](reproducibility.md)

