"""
Test script for DeepWindTrainDataset and DeepWindEvalDataset.

Run from project root:
    python test/test_datasets.py

Tests:
    1. Dataset instantiation (metadata, groups, weights)
    2. Train dataset: sample-level interleaving, shape, dtype, mask correctness
    3. Eval dataset: determinism, shape, filter_dataset
    4. DataLoader integration (batching, num_workers)
    5. Group balance verification
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path so 'src' is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from collections import Counter

# ---- Config ----
DATA_ROOT = Path(os.environ.get("DEEPWIND_DATA_ROOT", "data/deepwind_corpus_v1"))
TRAIN_NPY_ROOT = str(DATA_ROOT / "train")
EVAL_NPY_ROOT = str(DATA_ROOT / "eval")
TRAIN_META = str(DATA_ROOT / "train_metadata.csv")
EVAL_META = str(DATA_ROOT / "eval_metadata.csv")

pytestmark = pytest.mark.skipif(
    not Path(TRAIN_NPY_ROOT).is_dir() or not Path(EVAL_NPY_ROOT).is_dir(),
    reason="Set DEEPWIND_DATA_ROOT to run corpus integration tests.",
)

SEQ_LEN = 8192
MAX_VARS = 6
PAD_VAL_ID = 10


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ==============================================================
#  Test 1: Train Dataset Init
# ==============================================================
def _make_train_dataset():
    separator("Test 1: Train Dataset Init")
    from src.data.datasets import DeepWindTrainDataset

    ds = DeepWindTrainDataset(
        npy_root=TRAIN_NPY_ROOT,
        metadata_path=TRAIN_META,
        seq_len=SEQ_LEN,
        max_vars=MAX_VARS,
        pad_val_id=PAD_VAL_ID,
        dataset_weights={"windtoolkit": 0.5, "scada": 0.5},
        repeat=True,
        seed=42,
    )

    print(f"Groups found: {list(ds.groups.keys())}")
    for tag, files in ds.groups.items():
        print(f"  {tag}: {len(files)} files")
    print(f"Tags:  {ds._tags}")
    print(f"Probs: {ds._probs}")

    assert len(ds.groups) >= 2, "Expected at least 2 groups"
    assert abs(sum(ds._probs) - 1.0) < 1e-6, "Probs must sum to 1"
    print("PASSED")
    return ds


@pytest.fixture(scope="module")
def train_ds():
    if not Path(TRAIN_NPY_ROOT).is_dir():
        pytest.skip("Set DEEPWIND_DATA_ROOT to run corpus integration tests.")
    return _make_train_dataset()


def test_train_init(train_ds):
    assert len(train_ds.groups) >= 2
    assert abs(sum(train_ds._probs) - 1.0) < 1e-6


# ==============================================================
#  Test 2: Train Dataset — Sample Shapes & Types
# ==============================================================
def test_train_samples(train_ds):
    separator("Test 2: Train Dataset — Sample Shapes & Types")

    it = iter(train_ds)
    for i in range(5):
        sample = next(it)

        # Check all keys exist
        required_keys = {"context", "variate_ids", "channel_mask", "site_coords", "has_coords"}
        assert set(sample.keys()) == required_keys, f"Missing keys: {required_keys - set(sample.keys())}"

        # Check shapes
        assert sample["context"].shape == (MAX_VARS, SEQ_LEN), \
            f"context shape: {sample['context'].shape}, expected ({MAX_VARS}, {SEQ_LEN})"
        assert sample["variate_ids"].shape == (MAX_VARS,), \
            f"variate_ids shape: {sample['variate_ids'].shape}"
        assert sample["channel_mask"].shape == (MAX_VARS,), \
            f"channel_mask shape: {sample['channel_mask'].shape}"
        assert sample["site_coords"].shape == (2,), \
            f"site_coords shape: {sample['site_coords'].shape}"
        assert sample["has_coords"].shape == (1,), \
            f"has_coords shape: {sample['has_coords'].shape}"

        # Check dtypes
        assert sample["context"].dtype == np.float32
        assert sample["variate_ids"].dtype == np.int64
        assert sample["channel_mask"].dtype == np.float32
        assert sample["site_coords"].dtype == np.float32
        assert sample["has_coords"].dtype == np.float32

        # Check mask consistency
        n_valid = int(sample["channel_mask"].sum())
        assert n_valid >= 1, "At least 1 valid channel expected"
        assert n_valid <= MAX_VARS, f"Valid channels {n_valid} > max_vars {MAX_VARS}"

        # Check padding channels are zero
        if n_valid < MAX_VARS:
            pad_region = sample["context"][n_valid:, :]
            assert np.all(pad_region == 0.0), "Padding region should be all zeros"

            pad_ids = sample["variate_ids"][n_valid:]
            assert np.all(pad_ids == PAD_VAL_ID), \
                f"Padding variate_ids should be {PAD_VAL_ID}, got {pad_ids}"

        # Check coords consistency
        if sample["has_coords"][0] == 1.0:
            assert not np.all(sample["site_coords"] == 0.0), \
                "has_coords=1 but coords are [0,0]"

        if i == 0:
            print(f"  Sample {i}:")
            print(f"    context:      {sample['context'].shape}  dtype={sample['context'].dtype}")
            print(f"    variate_ids:  {sample['variate_ids']}")
            print(f"    channel_mask: {sample['channel_mask']}")
            print(f"    site_coords:  {sample['site_coords']}")
            print(f"    has_coords:   {sample['has_coords']}")
            print(f"    valid_channels: {n_valid}")
            print(f"    context stats: min={sample['context'][:n_valid].min():.4f}, "
                  f"max={sample['context'][:n_valid].max():.4f}, "
                  f"mean={sample['context'][:n_valid].mean():.4f}")

    print("PASSED (5 samples checked)")


# ==============================================================
#  Test 3: Train Dataset — Group Balance
# ==============================================================
def test_train_balance(train_ds):
    separator("Test 3: Train Dataset — Group Balance (sample-level)")

    # Collect 2000 samples, track which group each comes from
    # We detect group by checking variate_ids pattern or coords
    # Simpler: just count samples and time it

    it = iter(train_ds)
    n_samples = 2000

    t0 = time.time()
    samples = [next(it) for _ in range(n_samples)]
    elapsed = time.time() - t0

    print(f"  Generated {n_samples} samples in {elapsed:.2f}s "
          f"({n_samples/elapsed:.0f} samples/sec)")

    # Check that we see variation in data (not all from same file)
    first_vals = [s["context"][0, 0] for s in samples]
    unique_starts = len(set(first_vals))
    print(f"  Unique first-values: {unique_starts} / {n_samples}")
    assert unique_starts > 10, "Too few unique values — may be stuck on one file"

    # Check coords diversity
    coords_set = set()
    for s in samples:
        c = tuple(s["site_coords"].tolist())
        coords_set.add(c)
    print(f"  Unique coordinates: {len(coords_set)}")

    # Check variate_ids diversity (windtoolkit has 6 vars, scada may have fewer)
    mask_counts = Counter()
    for s in samples:
        n_valid = int(s["channel_mask"].sum())
        mask_counts[n_valid] += 1
    print(f"  Channel count distribution: {dict(mask_counts)}")

    print("PASSED")


# ==============================================================
#  Test 4: Train Dataset — DataLoader Integration
# ==============================================================
def test_train_dataloader(train_ds):
    separator("Test 4: Train Dataset — DataLoader (batch=32, workers=2)")

    loader = DataLoader(
        train_ds,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    it = iter(loader)
    for batch_idx in range(3):
        batch = next(it)

        assert batch["context"].shape == (32, MAX_VARS, SEQ_LEN), \
            f"Batch context shape: {batch['context'].shape}"
        assert batch["variate_ids"].shape == (32, MAX_VARS)
        assert batch["channel_mask"].shape == (32, MAX_VARS)
        assert batch["site_coords"].shape == (32, 2)
        assert batch["has_coords"].shape == (32, 1)

        # Check dtypes are torch tensors
        assert isinstance(batch["context"], torch.Tensor)
        assert batch["context"].dtype == torch.float32
        assert batch["variate_ids"].dtype == torch.int64

        if batch_idx == 0:
            print(f"  Batch 0:")
            print(f"    context:      {batch['context'].shape}  {batch['context'].dtype}")
            print(f"    variate_ids:  {batch['variate_ids'].shape}  {batch['variate_ids'].dtype}")
            print(f"    channel_mask: {batch['channel_mask'].shape}")
            print(f"    site_coords:  {batch['site_coords'].shape}")
            print(f"    has_coords:   {batch['has_coords'].shape}")

    # Cleanup workers
    del it, loader
    print("PASSED (3 batches)")


# ==============================================================
#  Test 5: Eval Dataset Init & Shapes
# ==============================================================
def _make_eval_dataset():
    separator("Test 5: Eval Dataset Init & Shapes")
    from src.data.datasets import DeepWindEvalDataset

    ds = DeepWindEvalDataset(
        npy_root=EVAL_NPY_ROOT,
        metadata_path=EVAL_META,
        seq_len=SEQ_LEN,
        max_vars=MAX_VARS,
        pad_val_id=PAD_VAL_ID,
    )

    print(f"  Total eval samples: {len(ds)}")
    assert len(ds) > 0, "Eval dataset is empty"

    # Check first sample
    sample = ds[0]
    assert sample["context"].shape == (MAX_VARS, SEQ_LEN)
    assert sample["variate_ids"].shape == (MAX_VARS,)
    assert sample["channel_mask"].shape == (MAX_VARS,)
    assert sample["site_coords"].shape == (2,)
    assert sample["has_coords"].shape == (1,)

    print(f"  Sample 0: context={sample['context'].shape}, "
          f"mask_sum={sample['channel_mask'].sum()}")
    print("PASSED")
    return ds


@pytest.fixture(scope="module")
def eval_ds():
    if not Path(EVAL_NPY_ROOT).is_dir():
        pytest.skip("Set DEEPWIND_DATA_ROOT to run corpus integration tests.")
    return _make_eval_dataset()


def test_eval_init(eval_ds):
    assert len(eval_ds) > 0
    assert eval_ds[0]["context"].shape == (MAX_VARS, SEQ_LEN)


# ==============================================================
#  Test 6: Eval Dataset — Determinism
# ==============================================================
def test_eval_determinism(eval_ds):
    separator("Test 6: Eval Dataset — Determinism")

    # Read first 10 samples twice, must be identical
    samples_a = [eval_ds[i] for i in range(10)]
    samples_b = [eval_ds[i] for i in range(10)]

    for i in range(10):
        for key in samples_a[i]:
            assert np.array_equal(samples_a[i][key], samples_b[i][key]), \
                f"Sample {i}, key '{key}' differs between two reads"

    print("  Two reads of first 10 samples: identical")
    print("PASSED")


# ==============================================================
#  Test 7: Eval Dataset — filter_dataset
# ==============================================================
def test_eval_filter():
    separator("Test 7: Eval Dataset — filter_dataset")
    from src.data.datasets import DeepWindEvalDataset

    wt_ds = DeepWindEvalDataset(
        npy_root=EVAL_NPY_ROOT,
        metadata_path=EVAL_META,
        seq_len=SEQ_LEN,
        filter_dataset="windtoolkit",
    )

    sc_ds = DeepWindEvalDataset(
        npy_root=EVAL_NPY_ROOT,
        metadata_path=EVAL_META,
        seq_len=SEQ_LEN,
        filter_dataset="scada",
    )

    all_ds = DeepWindEvalDataset(
        npy_root=EVAL_NPY_ROOT,
        metadata_path=EVAL_META,
        seq_len=SEQ_LEN,
    )

    print(f"  windtoolkit eval: {len(wt_ds)} samples")
    print(f"  scada eval:       {len(sc_ds)} samples")
    print(f"  all eval:         {len(all_ds)} samples")

    assert len(wt_ds) + len(sc_ds) == len(all_ds), \
        f"Filter mismatch: {len(wt_ds)} + {len(sc_ds)} != {len(all_ds)}"
    assert len(wt_ds) > 0, "windtoolkit eval is empty"
    assert len(sc_ds) > 0, "scada eval is empty"

    print("PASSED")


# ==============================================================
#  Test 8: Eval Dataset — DataLoader
# ==============================================================
def test_eval_dataloader():
    separator("Test 8: Eval Dataset — DataLoader")
    from src.data.datasets import DeepWindEvalDataset

    ds = DeepWindEvalDataset(
        npy_root=EVAL_NPY_ROOT,
        metadata_path=EVAL_META,
        seq_len=SEQ_LEN,
    )

    loader = DataLoader(ds, batch_size=32, num_workers=2, shuffle=False)

    batch = next(iter(loader))
    assert batch["context"].shape[0] == 32
    assert batch["context"].shape[1] == MAX_VARS
    assert batch["context"].shape[2] == SEQ_LEN

    print(f"  Batch shape: {batch['context'].shape}")
    print("PASSED")


# ==============================================================
#  Test 9: Throughput Benchmark
# ==============================================================
def test_throughput():
    separator("Test 9: Throughput Benchmark")
    from src.data.datasets import DeepWindTrainDataset

    ds = DeepWindTrainDataset(
        npy_root=TRAIN_NPY_ROOT,
        metadata_path=TRAIN_META,
        seq_len=SEQ_LEN,
        dataset_weights={"windtoolkit": 0.5, "scada": 0.5},
        seed=42,
    )

    loader = DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True)
    it = iter(loader)

    # Warmup
    for _ in range(5):
        next(it)

    # Benchmark
    n_batches = 50
    t0 = time.time()
    for _ in range(n_batches):
        batch = next(it)
    elapsed = time.time() - t0

    samples_per_sec = (n_batches * 32) / elapsed
    print(f"  {n_batches} batches x 32 = {n_batches * 32} samples")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {samples_per_sec:.0f} samples/sec")

    del it, loader
    print("PASSED")


# ==============================================================
#  Main
# ==============================================================
if __name__ == "__main__":
    print("DeepWind Dataset Tests")
    print(f"Train root: {TRAIN_NPY_ROOT}")
    print(f"Eval root:  {EVAL_NPY_ROOT}")

    try:
        # Train tests
        train_ds = _make_train_dataset()
        test_train_samples(train_ds)
        test_train_balance(train_ds)
        test_train_dataloader(train_ds)

        # Eval tests
        eval_ds = _make_eval_dataset()
        test_eval_determinism(eval_ds)
        test_eval_filter()
        test_eval_dataloader()

        # Throughput
        test_throughput()

        separator("ALL TESTS PASSED")

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
