"""
DeepWind Datasets for Pre-training and Evaluation.

Classes:
    DeepWindTrainDataset  — IterableDataset, sample-level balanced interleaving
    DeepWindEvalDataset   — Map-style Dataset, deterministic fixed-stride windows

Shared utilities:
    MetadataStore         — CSV metadata parser, O(1) lookup by filename
    _ShuffledFileIter     — infinite shuffled file iterator with full-coverage guarantee
"""

import random
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src.utils.registry import register_dataset
from src.utils.distributed import get_rank_world, is_main_process


logger = logging.getLogger(__name__)


# ===================================================================
#  Metadata Store (shared by train and eval)
# ===================================================================

class MetadataStore:
    """
    Parses metadata CSV once at init, provides O(1) lookup by filename.

    Required columns : filename
    Optional columns : longitude, latitude, variate_ids, dataset
    """

    def __init__(self, metadata_path: Optional[str | Path] = None):
        self.lookup: Dict[str, Dict[str, Any]] = {}
        if metadata_path is None:
            return
        
        path = Path(metadata_path)
        if not path.is_file():
            logger.warning(f"[MetadataStore] File not found: {path}")
            return

        df = pd.read_csv(path, dtype={"filename": str})
        for _, row in df.iterrows():
            fname = str(row["filename"]).strip()

            # Coordinates
            coords = None
            if pd.notna(row.get("longitude")) and pd.notna(row.get("latitude")):
                coords = np.array(
                    [float(row["longitude"]), float(row["latitude"])],
                    dtype=np.float32,
                )
            
            # Variate IDs
            v_ids = None
            raw = row.get("variate_ids")
            if pd.notna(raw):
                s = str(raw).strip()
                if s and s.lower() != "nan":
                    try:
                        v_ids = np.array(
                            [int(x) for x in s.split(",")], dtype=np.int64
                        )
                    except ValueError:
                        pass

            # Dataset tag (drives balanced sampling in training)
            dataset_tag = str(row.get("dataset", "unknown")).strip().lower()

            self.lookup[fname] = {
                "coords": coords,
                "variate_ids": v_ids,
                "dataset_tag": dataset_tag,
            }

        if is_main_process():
            logger.info(f"[MetadataStore] Loaded {len(self.lookup)} entries.")

    def get(self, filename: str) -> Dict[str, Any]:
        return self.lookup.get(
            filename,
            {"coords": None, "variate_ids": None, "dataset_tag": "unknown"},
        )


# ===================================================================
#  Infinite Shuffled File Iterator
# ===================================================================

class _ShuffledFileIter:
    """
    Yields files in shuffled order, looping forever.

    Guarantee: every file is seen exactly once per cycle before any
    repetition.  Even if training stops mid-cycle, the files seen so
    far are a uniform random subset.

        cycle 0: [f3, f1, f5, f2, f4]   <- all 5, shuffled
        cycle 1: [f2, f5, f4, f1, f3]   <- all 5, re-shuffled
    """

    __slots__ = ("_files", "_seed", "_cycle", "_pos", "_order")

    def __init__(self, files: List[Path], seed: int):
        self._files = files
        self._seed = seed
        self._cycle = 0
        self._pos = 0
        self._order = self._make_order(0)

    def _make_order(self, cycle: int) -> List[int]:
        idx = list(range(len(self._files)))
        random.Random(self._seed + cycle * 100003).shuffle(idx)
        return idx

    def __next__(self) -> Path:
        if self._pos >= len(self._order):
            self._cycle += 1
            self._order = self._make_order(self._cycle)
            self._pos = 0
        path = self._files[self._order[self._pos]]
        self._pos += 1
        return path


# ===================================================================
#  Shared Padding & Masking
# ===================================================================

def _pad_and_mask(
    window: np.ndarray,
    meta: Dict[str, Any],
    max_vars: int,
    seq_len: int,
    pad_val_id: int,
) -> Dict[str, np.ndarray]:
    """
    Apply channel padding, variate ID padding, coordinate handling.

    Used by both DeepWindTrainDataset and DeepWindEvalDataset to
    guarantee identical preprocessing.

    Args:
        window:     [C, seq_len] float32
        meta:       dict with 'coords' and 'variate_ids'
        max_vars:   pad channel dim to this size
        seq_len:    window length
        pad_val_id: embedding id for padded variate slots

    Returns:
        dict with keys: context, variate_ids, channel_mask,
                        site_coords, has_coords
    """
    C = window.shape[0]
    limit_c = min(C, max_vars)

    # 1. Values -> [max_vars, seq_len]
    if C >= max_vars:
        padded = window[:max_vars, :]
    else:
        padded = np.zeros((max_vars, seq_len), dtype=np.float32)
        padded[:limit_c, :] = window[:limit_c, :]

    # 2. Channel mask -> [max_vars]  (1=real, 0=padding)
    ch_mask = np.zeros(max_vars, dtype=np.float32)
    ch_mask[:limit_c] = 1.0

    # 3. Variate IDs -> [max_vars]
    v_ids = np.full(max_vars, pad_val_id, dtype=np.int64)
    raw_ids = meta["variate_ids"]
    if raw_ids is not None:
        n = min(len(raw_ids), max_vars)
        v_ids[:n] = raw_ids[:n]

    # 4. Coordinates -> [2],  flag -> [1]
    coords = meta["coords"]
    if coords is not None:
        out_coords = coords.copy()
        has_coords = np.array([1.0], dtype=np.float32)
    else:
        out_coords = np.zeros(2, dtype=np.float32)
        has_coords = np.array([0.0], dtype=np.float32)

    return {
        "context": padded,
        "variate_ids": v_ids,
        "channel_mask": ch_mask,
        "site_coords": out_coords,
        "has_coords": has_coords,
    }


# ===================================================================
#  Training Dataset
# ===================================================================

@register_dataset("deepwind_train")
class DeepWindTrainDataset(IterableDataset):
    """
    Streaming window dataset with sample-level balanced interleaving.

    Features:
        - Dataset-level weighted sampling (explicit weights or temperature scaling)
        - Sample-level interleaving for balanced batches
        - Adaptive window count proportional to series length
        - Smart IO: bulk-read vs mmap per file based on coverage ratio
        - Full file coverage guarantee via _ShuffledFileIter
        - Proper DDP + multi-worker sharding

    Usage:
        # Explicit weights
        ds = DeepWindTrainDataset(
            npy_root="./data/npy",
            metadata_path="./data/metadata.csv",
            dataset_weights={"windtoolkit": 0.5, "scada": 0.5},
        )

        # Automatic temperature scaling
        ds = DeepWindTrainDataset(
            npy_root="./data/npy",
            metadata_path="./data/metadata.csv",
            sampling_temperature=0.3,
        )

    metadata CSV format:
        filename,longitude,latitude,variate_ids,dataset
        site_001.npy,-97.5,35.2,"0,1,2,3,4,5",windtoolkit
        turbine_A.npy,-88.1,41.7,"0,1,2",scada
    """

    def __init__(
        self,
        npy_root: str | Path,
        metadata_path: str | Path,
        seq_len: int = 8192,
        max_vars: int = 6,
        pad_val_id: int = 10,
        # ---- Balanced sampling ----
        dataset_weights: Optional[Dict[str, float]] = None,
        sampling_temperature: float = 0.3,
        # ---- Adaptive windowing ----
        sample_ratio: float = 0.005,
        min_windows_per_file: int = 4,
        max_windows_per_file: int = 128,
        # ---- IO strategy ----
        full_read_coverage_threshold: float = 0.3,
        # ---- Iteration ----
        repeat: bool = True,
        seed: int = 42,
    ):  
        """
        Args:
            npy_root:  directory of *.npy files, each [C, T] float32.  
            metadata_path: CSV with columns   
                (filename, longitude, latitude, variate_ids, dataset).   
            seq_len:   window length.
            max_vars:  pad channel dim to this size.  
            pad_val_id: embedding id for padded variate slots.  

            dataset_weights: explicit {tag: weight}.
                If None -> temperature scaling.
            sampling_temperature: alpha for p_i ~ n_i^alpha.
                alpha=1 -> proportional (big group dominates).
                alpha=0 -> uniform across groups.

            sample_ratio:  fraction of candidate windows per file.
            min_windows_per_file / max_windows_per_file: clamp.

            full_read_coverage_threshold:
                if (num_windows * seq_len / T) > threshold ->
                bulk-read file into RAM; else keep mmap.
            
            repeat: loop forever (for step-based training).
            seed:   base seed.
        """
        super().__init__()
        self.npy_root = Path(npy_root)
        self.seq_len = seq_len
        self.max_vars = max_vars
        self.pad_val_id = pad_val_id

        self.dataset_weights = dataset_weights
        self.sampling_temperature = sampling_temperature

        self.sample_ratio = sample_ratio
        self.min_windows = min_windows_per_file
        self.max_windows = max_windows_per_file

        self.coverage_thresh = full_read_coverage_threshold
        self.repeat = repeat
        self.seed = seed

        # Metadata
        self.meta = MetadataStore(metadata_path)

        # Discover & group files
        all_files = sorted(self.npy_root.glob("*.npy"))
        if not all_files:
            raise RuntimeError(f"No .npy files in {self.npy_root}")

        self.groups: Dict[str, List[Path]] = {}
        for p in all_files:
            tag = self.meta.get(p.name)["dataset_tag"]
            self.groups.setdefault(tag, []).append(p)

        # Compute weights
        self._build_weights()

        if is_main_process():
            total = sum(len(v) for v in self.groups.values())
            logger.info(
                f"[DeepWindTrain] {total} files, "
                f"{len(self.groups)} groups:"
            )
            for i, tag in enumerate(self._tags):
                logger.info(
                    f"  {tag:20s}  files={len(self.groups[tag]):>7d}"
                    f"  weight={self._probs[i]:.4f}"
                )

    def _build_weights(self):
        """
        Compute per-group sampling probabilities.

        Mode 1: explicit dataset_weights -> normalize directly.
        Mode 2: temperature scaling      -> p_i ~ n_i^alpha.
        """
        if self.dataset_weights is not None:
            total = sum(self.dataset_weights.values())
            normed = {k: v / total for k, v in self.dataset_weights.items()}
            for tag in self.groups:
                if tag not in normed:
                    normed[tag] = 0.0
                    if is_main_process():
                        logger.warning(
                            f"  group '{tag}' not in dataset_weights -> 0"
                        )
            self._tags = sorted(normed.keys())
            self._probs = np.array(
                [normed[t] for t in self._tags], dtype=np.float64
            )
        else:
            sizes = {tag: len(fs) for tag, fs in self.groups.items()}
            raw = {
                tag: n ** self.sampling_temperature
                for tag, n in sizes.items()
            }
            total = sum(raw.values())
            self._tags = sorted(raw.keys())
            self._probs = np.array(
                [raw[t] / total for t in self._tags], dtype=np.float64
            )
        self._probs /= self._probs.sum()

    def _num_windows(self, T: int) -> int:
        """Adaptive window count: proportional to T, clamped to [min, max]."""
        n_cand = max(1, T - self.seq_len + 1)
        n = int(n_cand * self.sample_ratio)
        return int(np.clip(n, self.min_windows, self.max_windows))
    
    def _generate_from_file(
        self,
        path: Path,
        file_rng: np.random.Generator,
    ) -> Iterator[Dict[str, np.ndarray]]:
        """
        Load one .npy, choose IO strategy, sample windows, yield dicts.

        IO decision based on coverage ratio:
            HIGH coverage -> bulk-read to RAM, slice in memory
            LOW  coverage -> mmap, OS reads only touched pages
        """
        values = np.load(path, mmap_mode="r")
        if values.ndim != 2:
            return

        C, T = values.shape
        if T < self.seq_len:
            return

        n_all = T - self.seq_len + 1
        if n_all <= 0:
            return

        num_win = self._num_windows(T)
        starts = file_rng.integers(0, n_all, size=num_win)
        meta = self.meta.get(path.name)
        
        coverage = (num_win * self.seq_len) / T
        if coverage > self.coverage_thresh:
            buf = values.astype(np.float32)
            for s in starts:
                yield _pad_and_mask(
                    buf[:, s : s + self.seq_len],
                    meta, self.max_vars, self.seq_len, self.pad_val_id,
                )
        else:
            for s in starts:
                w = values[:, s : s + self.seq_len].astype(np.float32)
                yield _pad_and_mask(
                    w, meta, self.max_vars, self.seq_len, self.pad_val_id,
                )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Sample-level weighted interleaving.

        Architecture:
            Per group, an independent window generator runs forever:
                _ShuffledFileIter (files) -> _generate_from_file (windows)

            The main loop picks ONE sample at a time:
                select_rng.choice(p=probs) -> tag
                next(window_gens[tag])     -> one sample dict

            Each batch ~ Binomial(batch_size, group_prob) per group.
        """
        rank, world_size = get_rank_world()
        info = get_worker_info()
        n_workers = info.num_workers if info else 1
        w_id = info.id if info else 0

        gw_id = rank * n_workers + w_id
        gw_total = world_size * n_workers
        
        logger.info(
            "[DeepWindTrain] rank=%d/%d worker=%d/%d gw_id=%d/%d",
            rank, world_size, w_id, n_workers, gw_id, gw_total,
        )
        
        # Per-group: shard files -> infinite shuffled iter
        group_file_iters: Dict[str, _ShuffledFileIter] = {}
        active_tags: List[str] = []

        for tag in self._tags:
            shard = self.groups[tag][gw_id::gw_total]
            if not shard:
                continue
            group_file_iters[tag] = _ShuffledFileIter(
                shard,
                seed=self.seed + gw_id + hash(tag) % 9973,
            )
            active_tags.append(tag)

        if not active_tags:
            logger.warning(
                f"[DeepWindTrain] worker {gw_id}/{gw_total} has 0 files."
            )
            return

        # Probs restricted to active tags
        probs = np.array(
            [self._probs[self._tags.index(t)] for t in active_tags],
            dtype=np.float64,
        )
        probs /= probs.sum()

        # Per-group independent window generators
        def _make_window_gen(tag: str) -> Iterator[Dict[str, np.ndarray]]:
            """Infinite stream of windows for one group."""
            fc = 0  
            while True:
                path = next(group_file_iters[tag])
                file_rng = np.random.default_rng(
                    self.seed
                    + fc * 100003
                    + gw_id * 997
                    + hash(tag) % 9973
                )
                fc += 1
                try:
                    yield from self._generate_from_file(path, file_rng)
                except Exception as e:
                    if is_main_process():
                        logger.warning(f"[DeepWindTrain] {path.name}: {e}")
                    continue

        window_gens = {tag: _make_window_gen(tag) for tag in active_tags}

        # RNG for group selection
        select_rng = np.random.default_rng(self.seed + gw_id * 997)

        # Main loop: one sample at a time
        while True:
            tag = active_tags[
                select_rng.choice(len(active_tags), p=probs)
            ]
            yield next(window_gens[tag])


# ===================================================================
#  Evaluation Dataset
# ===================================================================

@register_dataset("deepwind_eval")
class DeepWindEvalDataset(Dataset):
    """
    Map-style evaluation dataset for DeepWind pre-training.

    Scans all .npy files in npy_root, generates fixed-stride
    non-overlapping windows, and pre-indexes them as (file_path, start_idx)
    pairs for deterministic, reproducible evaluation.

    Padding and masking logic is shared with DeepWindTrainDataset
    via the _pad_and_mask function.

    Usage:
        # Single eval dataset
        eval_ds = DeepWindEvalDataset(
            npy_root="./data/eval",
            metadata_path="./data/eval_metadata.csv",
            seq_len=8192,
        )

        # Per-group eval via HF Trainer
        trainer = Trainer(
            ...
            eval_dataset={
                "windtoolkit": DeepWindEvalDataset(..., filter_dataset="windtoolkit"),
                "scada": DeepWindEvalDataset(..., filter_dataset="scada"),
            },
        )
    """

    def __init__(
        self,
        npy_root: str | Path,
        metadata_path: str | Path,
        seq_len: int = 8192,
        stride: Optional[int] = None,
        max_vars: int = 6,
        pad_val_id: int = 10,
        max_samples_per_file: Optional[int] = None,
        filter_dataset: Optional[str] = None,
    ):
        """
        Args:
            npy_root:  directory of eval .npy files, each [C, T] float32.
            metadata_path: CSV with columns
                (filename, longitude, latitude, variate_ids, dataset).
            seq_len:   window length (must match training).
            stride:    step between windows. Default = seq_len (no overlap).
            max_vars:  must match training.
            pad_val_id: must match training.
            max_samples_per_file: cap windows per file to limit eval time.
                None = no limit.
            filter_dataset: if set (e.g. "windtoolkit" or "scada"),
                only include files whose metadata dataset tag matches.
                Used for per-group eval with HF Trainer dict.
        """
        super().__init__()
        self.npy_root = Path(npy_root)
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.max_vars = max_vars
        self.pad_val_id = pad_val_id
        self.max_samples_per_file = max_samples_per_file
        self.filter_dataset = (
            filter_dataset.lower().strip() if filter_dataset else None
        )

        if not self.npy_root.is_dir():
            raise NotADirectoryError(f"npy_root not found: {self.npy_root}")

        # Reuse the same MetadataStore as training
        self.meta = MetadataStore(metadata_path)

        # Pre-index all (file, start_idx) pairs
        self.samples: List[Tuple[Path, int]] = []
        self._scan_files()

    def _scan_files(self):
        """
        Scan npy_root, generate fixed-stride window indices.
        Deterministic: sorted file order, fixed stride, no randomness.
        """
        npy_files = sorted(self.npy_root.glob("*.npy"))
        if not npy_files:
            logger.warning(f"[DeepWindEval] No .npy files in {self.npy_root}")
            return

        valid_files = 0
        total_windows = 0

        for npy_path in npy_files:
            # Filter by dataset tag if requested
            if self.filter_dataset is not None:
                file_tag = self.meta.get(npy_path.name)["dataset_tag"]
                if file_tag != self.filter_dataset:
                    continue

            try:
                data = np.load(npy_path, mmap_mode="r")
                if data.ndim != 2:
                    continue

                _, T = data.shape
                if T < self.seq_len:
                    continue

                # Fixed-stride window starts
                last_start = T - self.seq_len
                starts = list(range(0, last_start + 1, self.stride))

                # Optional cap per file
                if self.max_samples_per_file is not None:
                    starts = starts[: self.max_samples_per_file]

                for s in starts:
                    self.samples.append((npy_path, s))

                valid_files += 1
                total_windows += len(starts)

            except Exception as e:
                logger.warning(
                    f"[DeepWindEval] Skipping {npy_path.name}: {e}"
                )
                continue

        if is_main_process():
            tag_info = (
                f" (filter={self.filter_dataset})"
                if self.filter_dataset
                else ""
            )
            logger.info(
                f"[DeepWindEval] {total_windows} samples from "
                f"{valid_files} files{tag_info}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        npy_path, start = self.samples[idx]

        # Load window
        data = np.load(npy_path, mmap_mode="r")
        window = np.array(
            data[:, start : start + self.seq_len], dtype=np.float32
        )

        meta = self.meta.get(npy_path.name)

        return _pad_and_mask(
            window, meta, self.max_vars, self.seq_len, self.pad_val_id,
        )


@register_dataset("deepwind_test")
class DeepWindTestDataset(Dataset):
    """
    Map-style sliding-window dataset for DeepWindModel evaluation.

    The time axis of each .npy file is split into train / val / test segments
    according to fixed ratios before any windowing is applied. Only the
    requested split is indexed, preventing any data leakage across splits.

    Split layout (time axis):
        |<------ train (0.7) ------>|<- val (0.1) ->|<--- test (0.2) --->|
        0                        t_train          t_val                   T

    Window layout within the selected split:
        |<-------- context_length -------->|<-- prediction_length -->|
        ^                                  ^
        start (within split)               split point

    Args:
        npy_path:          Path to a single .npy file of shape (C, T), float32.
        metadata_path:     Path to the metadata CSV.
        context_length:    Number of historical time steps fed to the model.
        prediction_length: Number of future time steps to predict.
        split:             Which segment to use: "train", "val", or "test".
        train_ratio:       Fraction of timesteps for training.   Default 0.7.
        val_ratio:         Fraction of timesteps for validation. Default 0.1.
        stride:            Step between consecutive window start positions.
                           Defaults to prediction_length (non-overlapping targets).
        max_vars:          Channel dimension after padding. Must match training.
        pad_val_id:        Variate embedding ID used for padded channels.
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        npy_path:          str | Path,
        metadata_path:     str | Path,
        context_length:    int,
        prediction_length: int,
        split:             str = "test",
        train_ratio:       float = 0.7,
        val_ratio:         float = 0.1,
        stride:            Optional[int] = None,
        max_vars:          int = 6,
        pad_val_id:        int = 10,
    ) -> None:
        super().__init__()

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self.VALID_SPLITS}, got '{split}'"
            )
        if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0):
            raise ValueError("train_ratio and val_ratio must be in (0, 1).")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be less than 1.0.")

        self.npy_path          = Path(npy_path)
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.window_length     = context_length + prediction_length
        self.split             = split
        self.stride            = stride if stride is not None else prediction_length
        self.max_vars          = max_vars
        self.pad_val_id        = pad_val_id

        if not self.npy_path.is_file():
            raise FileNotFoundError(f"[DeepWindTest] File not found: {self.npy_path}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

        # Metadata
        self.meta_store = MetadataStore(metadata_path)
        self.meta       = self.meta_store.get(self.npy_path.name)

        # Shape check
        data = np.load(self.npy_path, mmap_mode="r")
        if data.ndim != 2:
            raise ValueError(
                f"[DeepWindTest] Expected (C, T), got {data.shape} "
                f"for {self.npy_path.name}"
            )
        self._C, self._T = data.shape

        # ── Compute split boundaries (on raw time axis) ───────────────────────
        t_train = int(self._T * train_ratio)
        t_val   = int(self._T * (train_ratio + val_ratio))
        # test runs from t_val to T

        self._split_start, self._split_end = {
            "train": (0,       t_train),
            "val":   (t_train, t_val),
            "test":  (t_val,   self._T),
        }[split]

        split_len = self._split_end - self._split_start
        if split_len < self.window_length:
            fallback_ctx = 512
            fallback_window = fallback_ctx + self.prediction_length
            if split_len < fallback_window:
                raise ValueError(
                    f"[DeepWindTest] '{split}' split too short even for fallback: "
                    f"split_len={split_len} < fallback_window={fallback_window} "
                    f"({self.npy_path.name})."
                )
            if is_main_process():
                logger.warning(
                    "[DeepWindTest] %s  '%s' split too short for ctx=%d "
                    "(split_len=%d < window=%d). "
                    "Falling back to context_length=%d.",
                    self.npy_path.name, split,
                    self.context_length, split_len, self.window_length,
                    fallback_ctx,
                )
            self.context_length = fallback_ctx
            self.window_length  = fallback_window

        # ── Pre-compute window start positions (absolute indices) ─────────────
        # Windows must not cross the split boundary:
        #   start >= split_start
        #   start + window_length <= split_end
        first_start = self._split_start
        last_start  = self._split_end - self.window_length
        self._starts = list(range(first_start, last_start + 1, self.stride))

        if is_main_process():
            logger.info(
                "[DeepWindTest] %s  split=%s  "
                "range=[%d, %d)  split_len=%d  windows=%d  "
                "ctx=%d  pred=%d  stride=%d",
                self.npy_path.name, split,
                self._split_start, self._split_end, split_len,
                len(self._starts),
                self.context_length, self.prediction_length, self.stride,
            )

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start = self._starts[idx]
        split = start + self.context_length
        end   = split + self.prediction_length

        data        = np.load(self.npy_path, mmap_mode="r")
        context_raw = np.array(data[:, start:split], dtype=np.float32)
        target_raw  = np.array(data[:, split:end],   dtype=np.float32)

        sample = _pad_and_mask(
            window     = context_raw,
            meta       = self.meta,
            max_vars   = self.max_vars,
            seq_len    = self.context_length,
            pad_val_id = self.pad_val_id,
        )
        
        C             = min(self._C, self.max_vars)
        target_padded = np.zeros((self.max_vars, self.prediction_length), dtype=np.float32)
        target_padded[:C, :] = target_raw[:C, :]
        sample["target"] = target_padded
        sample["index"] = np.array(idx, dtype=np.int64)

        return sample
