#!/usr/bin/env python3
"""Audit a DeepWind corpus without loading array payloads into memory."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def read_metadata(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def audit_split(root: Path, metadata_path: Path, sample_size: int, seed: int) -> dict:
    rows, columns = read_metadata(metadata_path)
    files = sorted(root.glob("*.npy"))
    disk_names = {p.name for p in files}
    metadata_names = [row.get("filename", "") for row in rows]
    metadata_set = set(metadata_names)

    rng = random.Random(seed)
    sampled = rng.sample(files, min(sample_size, len(files)))
    shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    invalid: list[dict[str, str]] = []
    for path in sampled:
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            shapes[str(tuple(array.shape))] += 1
            dtypes[str(array.dtype)] += 1
            if array.ndim != 2:
                invalid.append({"file": path.name, "reason": f"ndim={array.ndim}"})
        except Exception as exc:  # pragma: no cover - reports damaged external data
            invalid.append({"file": path.name, "reason": repr(exc)})

    groups = Counter((row.get("dataset") or "unknown").strip().lower() for row in rows)
    missing_coords = sum(
        not row.get("longitude", "").strip() or not row.get("latitude", "").strip()
        for row in rows
    )
    return {
        "root": str(root),
        "metadata": str(metadata_path),
        "metadata_columns": columns,
        "file_count": len(files),
        "metadata_row_count": len(rows),
        "dataset_groups": dict(sorted(groups.items())),
        "duplicate_metadata_filenames": len(metadata_names) - len(metadata_set),
        "files_missing_metadata_count": len(disk_names - metadata_set),
        "metadata_missing_files_count": len(metadata_set - disk_names),
        "missing_coordinate_rows": missing_coords,
        "sample_size": len(sampled),
        "sample_shapes": dict(shapes),
        "sample_dtypes": dict(dtypes),
        "invalid_sample_files": invalid,
        "filenames": sorted(disk_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = {"schema_version": 1, "splits": {}}
    for split in ("train", "eval"):
        root = args.data_root / split
        metadata = args.data_root / f"{split}_metadata.csv"
        result["splits"][split] = audit_split(root, metadata, args.sample_size, args.seed)

    train_names = set(result["splits"]["train"].pop("filenames"))
    eval_names = set(result["splits"]["eval"].pop("filenames"))
    overlap = sorted(train_names & eval_names)
    result["train_eval_filename_overlap_count"] = len(overlap)
    result["train_eval_filename_overlap_examples"] = overlap[:20]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
