# tools/clean_data_manifests.py
"""
One-time data-manifest cleanup (Phase 7 data hygiene).

Three idempotent operations, each reversible via a timestamped backup:

  1. Strip trailing whitespace from ``dataset`` values.
  2. Drop stale ``train_metadata.csv`` rows whose file is NOT in ``train/``
     (these are the 150 eval + 8 benchmark-test files that were moved out of
     ``train/`` into ``eval/`` / ``test/`` but left behind in the pre-split
     manifest).
  3. Remove test files that also physically live in ``train/`` (kept in
     ``train/`` to participate in pretraining, per release decision) — the
     ``test/`` copy is moved to a quarantine dir, not deleted.

Usage:
    export DEEPWIND_DATA_ROOT=/scratch/pawsey0115/hwang4/deepwindData
    python tools/clean_data_manifests.py --dry-run   # preview only
    python tools/clean_data_manifests.py             # apply (with backups)
"""
import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_ROOT = os.environ.get(
    "DEEPWIND_DATA_ROOT",
    "/scratch/pawsey0115/hwang4/deepwindData",
)

SPLIT_DIRS = {
    "train": "train_metadata.csv",
    "test": "test_metadata.csv",
}


def list_npy(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {f for f in os.listdir(directory) if f.endswith(".npy")}


def load(split: str) -> pd.DataFrame:
    path = Path(DATA_ROOT) / SPLIT_DIRS[split]
    return pd.read_csv(path, dtype={"filename": str})


def clean_dataset_column(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" in df.columns:
        df = df.copy()
        df["dataset"] = df["dataset"].astype(str).str.strip()
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(DATA_ROOT)
    train_files = list_npy(root / "train")
    test_files = list_npy(root / "test")

    train = clean_dataset_column(load("train"))
    test = clean_dataset_column(load("test"))

    # --- Operation 2: drop stale train rows (file not in train/) ------------
    stale_mask = ~train["filename"].isin(train_files)
    n_stale = int(stale_mask.sum())

    # --- Operation 3: leaked test files (also physically in train/) ----------
    leaked_mask = test["filename"].isin(train_files)
    leaked = test.loc[leaked_mask, "filename"].tolist()
    # Verify the train/ copy exists for every leaked file before touching test/
    missing_in_train = [f for f in leaked if f not in train_files]
    assert not missing_in_train, f"leaked files missing from train/: {missing_in_train}"

    train_clean = train.loc[~stale_mask]
    test_clean = test.loc[~leaked_mask]

    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"train rows: {len(train)} -> {len(train_clean)}  (drop {n_stale} stale)")
    print(f"test  rows: {len(test)}  -> {len(test_clean)}   (drop {len(leaked)} leaked)")
    print(f"leaked test files to keep in train/ and quarantine from test/:")
    for f in leaked:
        tr_size = (root / "train" / f).stat().st_size
        te_size = (root / "test" / f).stat().st_size
        match = "SAME_SIZE" if tr_size == te_size else "SIZE-MISMATCH"
        print(f"    {f:22s} train={tr_size:>12,}  test={te_size:>12,}  {match}")

    if args.dry_run:
        print("\n[DRY-RUN] no changes written.")
        return 0

    # --- Apply ---------------------------------------------------------------
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    backup_dir = root / f"_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Back up the two manifests before overwriting
    for split, fname in SPLIT_DIRS.items():
        src = root / fname
        shutil.copy2(src, backup_dir / fname)

    train_clean.to_csv(root / SPLIT_DIRS["train"], index=False)
    test_clean.to_csv(root / SPLIT_DIRS["test"], index=False)

    # Quarantine (move, not delete) the leaked test copies
    quarantine = root / "_removed_from_test"
    quarantine.mkdir(parents=True, exist_ok=True)
    for f in leaked:
        src = root / "test" / f
        if src.exists():
            shutil.move(str(src), str(quarantine / f))

    print(f"\nApplied. Backups in {backup_dir}")
    print(f"Quarantined {len(leaked)} leaked file(s) -> {quarantine}")
    print("Re-run tools/audit_data_splits.py to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
