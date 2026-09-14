# Data

## Expected processed layout

```text
$DEEPWIND_DATA_ROOT/
├── train/                 # one NumPy array per training series
├── eval/                  # held-out validation arrays
├── test/                  # WindBench arrays
├── train_metadata.csv
└── eval_metadata.csv
```

Each array has shape `(time, variate)`. Metadata maps each file to its dataset,
available variates, capacity, coordinates, and normalization information. The
loader pads heterogeneous inputs to `max_vars` and supplies a channel mask.

## Corpus and sampling

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

## Split isolation

WindBench sites must not occur in pretraining. For WIND Toolkit targets, retain
the paper's 10 km spatial exclusion buffer. For SCADA sources, exclude the exact
held-out farm or turbine before window generation. A future public preprocessing
release must emit a machine-readable split manifest and contamination audit.

## Redistribution

This repository does not redistribute the 2.1 TB processed corpus. Users must
obtain each source under its own terms. Shanxi Wind is proprietary and cannot be
published. Release download/preprocessing scripts and checksums for public
sources instead of uploading derived data without a license review.
