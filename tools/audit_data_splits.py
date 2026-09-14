# tools/audit_data_splits.py
"""
Audit DeepWind data manifests for split isolation and hygiene.

Checks (each line is a distinct class of problem):
  1. non-normalised ``dataset`` values (trailing whitespace)
  2. duplicate filenames within a manifest
  3. cross-manifest filename overlap (train / eval / test)
  4. manifest filenames missing from their expected ``<split>/`` directory
  5. eval / test files that also physically exist in ``train/`` (leakage)
  6. per-group (dataset) row counts

Exit code is non-zero if any check fails, so the script can gate CI.

Usage:
    export DEEPWIND_DATA_ROOT=/scratch/pawsey0115/hwang4/deepwindData
    python tools/audit_data_splits.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

DATA_ROOT = os.environ.get(
    "DEEPWIND_DATA_ROOT",
    "/scratch/pawsey0115/hwang4/deepwindData",
)

MANIFESTS = {
    "train": "train_metadata.csv",
    "eval": "eval_metadata.csv",
    "test": "test_metadata.csv",
}

_SPLITS = list(MANIFESTS)


def load_manifest(split: str) -> pd.DataFrame | None:
    path = Path(DATA_ROOT) / MANIFESTS[split]
    if not path.is_file():
        print(f"[warn] missing manifest: {path}")
        return None
    return pd.read_csv(path, dtype={"filename": str})


def check_dataset_values(manifests: dict[str, pd.DataFrame]) -> list[str]:
    """Return a list of problem strings for dirty ``dataset`` values."""
    problems = []
    for split, df in manifests.items():
        if df is None or "dataset" not in df.columns:
            continue
        raw = df["dataset"].astype(str)
        dirty = raw[raw != raw.str.strip()]
        if len(dirty):
            problems.append(
                f"{split}: {len(dirty)} non-normalised dataset value(s) "
                f"{sorted(dirty.unique())}"
            )
    return problems


def check_duplicate_filenames(manifests: dict[str, pd.DataFrame]) -> list[str]:
    problems = []
    for split, df in manifests.items():
        if df is None:
            continue
        dup = df["filename"].duplicated().sum()
        if dup:
            problems.append(f"{split}: {dup} duplicate filename(s)")
    return problems


def check_cross_split_overlap(manifests: dict[str, pd.DataFrame]) -> list[str]:
    """Report filename sets that appear in more than one manifest."""
    problems = []
    sets = {s: set(df["filename"]) for s, df in manifests.items() if df is not None}
    for i, a in enumerate(_SPLITS):
        for b in _SPLITS[i + 1 :]:
            if a in sets and b in sets:
                n = len(sets[a] & sets[b])
                if n:
                    problems.append(f"{a} ∩ {b} = {n} filename(s)")
    return problems


def check_missing_files(manifests: dict[str, pd.DataFrame]) -> list[str]:
    """Report manifest filenames absent from their expected directory.

    Lists each directory once (rather than one ``is_file()`` per row) to stay
    cheap on Lustre with the ~127k-entry ``train/`` directory.
    """
    problems = []
    for split, df in manifests.items():
        if df is None:
            continue
        dir_path = Path(DATA_ROOT) / split
        if not dir_path.is_dir():
            problems.append(f"{split}: directory missing: {dir_path}")
            continue
        on_disk = set(os.listdir(dir_path))
        missing = [f for f in df["filename"] if f not in on_disk]
        if missing:
            problems.append(
                f"{split}: {len(missing)} manifest filename(s) missing from "
                f"{split}/ (e.g. {missing[:3]})"
            )
    return problems


def check_train_leakage(manifests: dict[str, pd.DataFrame]) -> list[str]:
    """Report eval/test filenames that also exist physically in ``train/``."""
    problems = []
    train_dir = Path(DATA_ROOT) / "train"
    if manifests.get("train") is None:
        return problems
    train_files = set(manifests["train"]["filename"])

    for split in ("eval", "test"):
        df = manifests.get(split)
        if df is None:
            continue
        leaked = [
            f
            for f in df["filename"]
            if f in train_files and (train_dir / f).is_file()
        ]
        if leaked:
            problems.append(
                f"{split}: {len(leaked)} filename(s) also present in train/ "
                f"(e.g. {leaked[:3]})"
            )
    return problems


def main() -> int:
    manifests = {s: load_manifest(s) for s in _SPLITS}

    print(f"DATA_ROOT = {DATA_ROOT}")
    for split, df in manifests.items():
        if df is not None:
            print(f"  {split}: {len(df)} rows, columns={list(df.columns)}")

    # Per-group counts
    for split, df in manifests.items():
        if df is not None and "dataset" in df.columns:
            counts = df["dataset"].astype(str).str.strip().value_counts()
            print(f"  [{split} groups] " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    print()

    all_problems: list[str] = []
    for label, fn in (
        ("dataset values", check_dataset_values),
        ("duplicate filenames", check_duplicate_filenames),
        ("cross-split overlap", check_cross_split_overlap),
        ("missing files", check_missing_files),
        ("train leakage", check_train_leakage),
    ):
        problems = fn(manifests)
        status = "FAIL" if problems else "ok"
        print(f"[{status:>4}] {label}")
        for p in problems:
            print(f"          - {p}")
        all_problems.extend(problems)

    print()
    if all_problems:
        print(f"AUDIT FAILED: {len(all_problems)} problem(s)")
        return 1
    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
