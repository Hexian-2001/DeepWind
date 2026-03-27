import torch
import numpy as np
import pandas as pd
import random
import torch.distributed as dist
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Union, Tuple
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data import Dataset

from src.utils.registry import register_dataset
from src.utils.distributed import is_main_process

import logging

logger = logging.getLogger(__name__)

@register_dataset("deepwind_train")
class DeepWindTrainDataset(IterableDataset):
    """
    Streaming window dataset for huge wind time-series stored as .npy files.

    Each .npy file:
        values.shape == [N, T], dtype float32

    This dataset:
        - Supports multi-GPU (DDP) by sharding files by rank.
        - Supports multi-worker DataLoader by further sharding by worker id.
        - Streams windows on-the-fly (no precomputed windows, no Arrow).
        - Can generate *very* large numbers of training samples.
    """

    def __init__(
        self,
        npy_root: str | Path,
        metadata_path: str | Path = None,  # <--- Path to metadata CSV
        seq_len: int = 8192,
        max_windows_per_series: Optional[int] = None,
        shuffle_files: bool = True,
        repeat: bool = True,
        seed: int = 12,
        max_vars: int = 6,
        pad_val_id = 10,
    ):
        """
        Args:
            npy_root: directory containing *.npy files (each [6, T])
            seq_len: window length (L)
            stride: stride between candidate window starts
            max_windows_per_series: if not None, randomly sample at most
                this many windows per series to avoid explosion.
            shuffle_files: shuffle file order at each epoch
            repeat: if True, loop over the dataset infinitely (for large-scale training)
            seed: base random seed used for per-epoch shuffling and sampling
        """
        self.npy_root = Path(npy_root)
        self.seq_len = seq_len
        self.max_windows_per_series = max_windows_per_series
        self.shuffle_files = shuffle_files
        self.repeat = repeat
        self.seed = seed
        
        self.max_vars = max_vars
        self.pad_val_id = pad_val_id

        self.npy_files: List[Path] = sorted(self.npy_root.glob("*.npy"))
        if not self.npy_files:
            raise RuntimeError(f"No .npy files found under {self.npy_root}")
        if is_main_process():
            logger.info(f"[WindNPYWindowDataset] Found {len(self.npy_files)} npy files in {self.npy_root}")

        # --------- Metadata Loading (Optimized) --------- #
        self.meta_lookup = {}
        if metadata_path:
            if is_main_process():
                logger.info(f"[DeepWindTrainDataset] Loading metadata from {metadata_path}...")
            # Load with specific dtypes to ensure filename is string, others can be float/object
            df = pd.read_csv(metadata_path, dtype={'filename': str})

            success_count = 0
            for _, row in df.iterrows():
                fname = str(row['filename']).strip()

                # --- 1. Parse Coordinates (Robust Check) ---
                # Check if lat/lon are not NaN (using pandas pd.notna is safer than np.isnan)
                if pd.notna(row.get('longitude')) and pd.notna(row.get('latitude')):
                    coords = np.array([row['longitude'], row['latitude']], dtype=np.float32)
                else:
                    coords = None

                # --- 2. Parse Variate IDs (Robust Check) ---
                v_ids = None
                raw_v_ids = row.get('variate_ids')

                # Check if it exists, is not NaN, and is not an empty string
                if pd.notna(raw_v_ids):
                    v_ids_str = str(raw_v_ids).strip()
                    # Handle case where CSV might have "nan" string or empty string
                    if v_ids_str and v_ids_str.lower() != 'nan':
                        try:
                            v_ids = np.array([int(x) for x in v_ids_str.split(',')], dtype=np.int64)
                        except ValueError:
                            # Catch cases like "1,2,error"
                            if is_main_process():
                                logger.info(f"[Warning] Invalid variate_ids format for {fname}: {v_ids_str}")
                            v_ids = None

                # Only store if at least one piece of info exists (optional, depends on your needs)
                # Here we store even if both are None, so we know the file is "known"
                self.meta_lookup[fname] = (coords, v_ids)
                success_count += 1
            if is_main_process():    
                logger.info(f"[DeepWindTrainDataset] Loaded metadata for {success_count} sites.")
        else:
            if is_main_process(): 
                logger.info("[DeepWindTrainDataset] Warning: No metadata_path provided. Coords and IDs will be None.")

    # --------- DDP helpers --------- #
    @staticmethod
    def _get_rank_world() -> tuple[int, int]:
        """
        Get (rank, world_size) for DDP.
        If DDP is not initialized, return (0, 1).
        """
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        else:
            return 0, 1

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Main iterator:
            - determines this (rank, worker) shard of files
            - loops over epochs
            - for each file:
                * loads values via np.load(mmap_mode="r")
                * generates candidate window starts
                * (optionally) subsamples windows
                * shuffles starts
                * yields PADDEED window dicts (Uniform Shape)
        """
        rank, world_size = self._get_rank_world()
        worker_info = get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        # Global (rank, worker) id → used for file sharding
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        # Base file list (will be shuffled each epoch if required)
        files: List[Path] = self.npy_files.copy()
        
        # --- Padding Configuration ---
        epoch = 0
        # Infinite or single-pass loop over epochs
        while True:
            # 1) Shuffle file order once per epoch (same for all workers in this process)
            if self.shuffle_files:
                rng = random.Random(self.seed + epoch)
                rng.shuffle(files)

            # 2) Shard files by (rank, worker)
            #    Each global worker sees a disjoint subset of files.
            shard_files = files[global_worker_id::global_num_workers]

            # 3) Iterate over this shard of files
            for path in shard_files:
                try:
                    # --- Metadata Lookup ---
                    # Look for the filename (e.g., 'site_123.npy') in the lookup table
                    meta_data = self.meta_lookup.get(path.name)

                    if meta_data is not None:
                        site_coords, variate_ids = meta_data
                    else:
                        # Fallback if site not found in metadata
                        site_coords, variate_ids = None, None

                    # --- Load Data ---
                    values = np.load(path, mmap_mode="r")  # [N, T]
                    if values.ndim != 2:
                        if is_main_process():
                            logger.info(f"[Warning] {path} has unexpected shape {values.shape}, skipping.")
                        continue

                    # C = Current number of variates in this specific file
                    C, T = values.shape

                    if T < self.seq_len:
                        continue

                    # candidate start count
                    n_all = (T - self.seq_len) + 1
                    if n_all <= 0:
                        continue

                    # your downsampling: only use 1% of possible windows
                    num_samples = max(1, 32)

                    # per-(epoch, worker, file) RNG
                    rng_np = np.random.default_rng(
                        self.seed + epoch + global_worker_id + (hash(path.stem) % 1000003)
                    )

                    # --- Pre-calculate Static Padding Info for this file ---

                    # 1. Pad Variate IDs
                    # Create array filled with self.pad_val_id
                    padded_v_ids = np.full((self.max_vars,), self.pad_val_id, dtype=np.int64)
                    if variate_ids is not None:
                        # Fill valid IDs
                        valid_len = min(len(variate_ids), self.max_vars)
                        padded_v_ids[:valid_len] = variate_ids[:valid_len]

                    # 2. Generate Channel Mask
                    # 1.0 = valid data, 0.0 = padding
                    channel_mask = np.zeros((self.max_vars,), dtype=np.float32)
                    valid_channels = min(C, self.max_vars)
                    # valid_channels = self.max_vars
                    channel_mask[:valid_channels] = 1.0

                    # 3. Handle Coordinates
                    # If coords exist, use them and set flag=1. Else use 0.0 and flag=0.
                    if site_coords is not None:
                        out_coords = site_coords.astype(np.float32)
                        has_coords_flag = np.array([1.0], dtype=np.float32)
                    else:
                        out_coords = np.array([0.0, 0.0], dtype=np.float32)
                        has_coords_flag = np.array([0.0], dtype=np.float32)
                    needs_padding = (C < self.max_vars)
                    # --- Generate Windows ---
                    for _ in range(num_samples):
                        s_int = rng_np.integers(0, n_all)  # choose index
                        window = values[:, s_int: s_int + self.seq_len].astype(np.float32)

                        if not needs_padding:
                            padded_context = window
                        else:
                            limit_c = min(C, self.max_vars)
                            padded_context = np.zeros((self.max_vars, self.seq_len), dtype=np.float32)
                            padded_context[:limit_c, :] = window
                            

                        yield {
                            "context": padded_context,    # [self.max_vars, seq_len] (Padded with target data)
                            "variate_ids": padded_v_ids,  # [self.max_vars] (Keep padding IDs for these rows)
                            "channel_mask": channel_mask, # [self.max_vars] (Keep 0.0 for these rows to avoid duplicate loss)
                            "site_coords": out_coords,
                            "has_coords": has_coords_flag
                        }

                except Exception as e:
                    if is_main_process():
                        logger.info(f"[Error] Failed to read {path}: {e}")
                    continue

            epoch += 1
            if not self.repeat:
                break


@register_dataset("deepwind_test")
class DeepWindTestDataset(Dataset):
    """
    Map-style evaluation dataset for a single .npy file.
    Aligned with DeepWindTrainDataset logic (Padding, Masking, Coordinates).

    Produces:
        {
            "context":      [MAX_VARS, context_length], # Padded
            "target":       [MAX_VARS, prediction_length], # Padded (for consistent loss calc)
            "variate_ids":  [MAX_VARS],
            "channel_mask": [MAX_VARS],
            "site_coords":  [2],
            "has_coords":   [1]
        }
    """

    def __init__(
            self,
            npy_path: str | Path,
            metadata_path: str | Path,
            context_length: int,
            prediction_length: int,
            stride: Optional[int] = None,
            split: str = "test",  # 'test' (last 20%), 'val' (10%), or 'train' 
            max_vars: int = 6,
            pad_val_id = 10,
    ) -> None:
        """
        Args:
            npy_path: Path to the .npy file with shape [V, T] (Variates, Time).
            metadata_path: Path to the metadata CSV file.
            context_length: Length of history sequence (L_c).
            prediction_length: Length of prediction sequence (L_p).
            stride: Stride between sliding windows. Defaults to prediction_length.
            split: Dataset split to use.
                   - 'test': Uses the last 20% (0.8 -> 1.0) for zero-shot evaluation.
                   - 'val': Uses the middle 10% (0.7 -> 0.8) for validation.
                   - 'all': Uses the entire dataset (0.0 -> 1.0).
                   * Note: If the selected split is shorter than the sequence length,
                     it automatically falls back to 'all'.
        """
        super().__init__()
        self.npy_path = Path(npy_path)
        self.metadata_path = Path(metadata_path)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.seq_len = self.context_length + self.prediction_length
        self.stride = int(stride or prediction_length)
        self.split = split.lower()
        
        # --- Consistency Config (Must match Train Dataset) ---
        self.max_vars = max_vars
        self.pad_val_id = pad_val_id

        if not self.npy_path.is_file():
            raise FileNotFoundError(f"File not found: {self.npy_path}")

        # 1. Load Data (Memory Map)
        # Keep mmap open to access data without loading the entire file into RAM.
        self.data = np.load(self.npy_path, mmap_mode="r")

        if self.data.ndim != 2:
            raise RuntimeError(f"Expected [V, T] shape, got {self.data.shape}")

        self.V, self.T = self.data.shape

        # 2. Define Split Boundaries (7:1:2 Strategy)
        # Train: 0% - 70% | Valid: 70% - 80% | Test: 80% - 100%
        train_end = int(self.T * 0.7)
        val_end = int(self.T * 0.8)

        if self.split == "test":
            start_idx = val_end
            end_idx = self.T
        elif self.split == "val":
            start_idx = train_end
            end_idx = val_end
        elif self.split == "train":
            start_idx = 0
            end_idx = train_end
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'test', 'val', or 'all'.")

        # --- Fallback Logic ---
        # If the selected split segment is shorter than the required sequence length,
        # fallback to using the entire dataset to ensure we can evaluate something.
        current_segment_len = end_idx - start_idx

        if current_segment_len < self.seq_len + self.prediction_length:
            if is_main_process():
                logger.info(f"[DeepWindTestDataset] Warning: Split '{self.split}' length ({current_segment_len}) "
                  f"is shorter than required seq_len+pred_len ({self.seq_len + self.prediction_length}).")
                logger.info(f"[DeepWindTestDataset] -> Fallback: Using FULL dataset range [0, {self.T}) for evaluation.")

            # Reset to full range
            start_idx = 0
            end_idx = self.T

            # Final check: if even the full dataset is too short, we cannot proceed.
            if (end_idx - start_idx) < self.seq_len + self.prediction_length:
                raise RuntimeError(f"Dataset too small! Total time steps {self.T} < required seq_len+pred_len {self.seq_len + self.prediction_length}.")

        if is_main_process():
            logger.info(
            f"[DeepWindTestDataset] Final Split: {self.split.upper()} (adjusted) | Range: [{start_idx}, {end_idx}) | Total Time: {self.T}")

        # 3. Generate Valid Start Indices
        # We generate indices such that the window [start, start + seq_len) fits strictly within [start_idx, end_idx).
        last_possible_start = end_idx - self.seq_len

        # range is inclusive for start, exclusive for end, so we add 1
        self.starts = list(range(start_idx, last_possible_start + 1, self.stride))

        if len(self.starts) == 0:
            if is_main_process():
                logger.info(f"[Warning] No samples generated. Check sequence length vs dataset size.")

        # 3. Load Metadata (Aligned with Train logic)
        self._load_metadata()

    def _load_metadata(self):
        """Helper to load and format metadata exactly like the training set."""
        self.site_coords = None
        self.variate_ids = None

        if self.metadata_path.is_file():
            df = pd.read_csv(
                self.metadata_path,
                dtype={'filename': str, 'latitude': np.float32, 'longitude': np.float32}
            )
            target_filename = self.npy_path.name
            row = df[df['filename'] == target_filename]

            if not row.empty:
                data = row.iloc[0]

                # --- Coordinates ---
                lon, lat = data['longitude'], data['latitude']
                if pd.notna(lon) and pd.notna(lat):
                    self.site_coords = np.array([lon, lat], dtype=np.float32)

                # --- Variate IDs ---
                v_ids_str = str(data.get('variate_ids', ''))
                if pd.notna(data.get('variate_ids')) and v_ids_str.strip() and v_ids_str.lower() != 'nan':
                    try:
                        self.variate_ids = np.array([int(x) for x in v_ids_str.split(',')], dtype=np.int64)
                    except ValueError:
                        if is_main_process():
                            logger.info(f"[Warning] Invalid variate_ids format for {target_filename}")
        
    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches the i-th window, applies padding and masking to match training data.
        """
        s = self.starts[idx]
        e = s + self.seq_len

        # 1. Load Raw Window [V, seq_len]
        # Use copy=True to ensure we have a writable array for padding later if needed
        full_window = np.array(self.data[:, s:e], dtype=np.float32)

        C = full_window.shape[0]

        # --- Logic from DeepWindTrainDataset ---

        # 2. Pad/Truncate Data (values)
        limit_c = min(C, self.max_vars)

        # Initialize container with zeros [MAX_VARS, seq_len]
        # padded_window = np.zeros((self.max_vars, self.seq_len), dtype=np.float32)
        padded_window = np.zeros((limit_c, self.seq_len), dtype=np.float32)
        # Fill valid channels
        padded_window[:limit_c, :] = full_window[:limit_c, :]

        # If we need padding (C < MAX_VARS), Train logic repeats the 0-th channel
        # if C < self.max_vars:
        #     # Train code: padded_context[limit_c:, :] = window[0, :]
        #     target_data = full_window[0, :]
        #     padded_window[limit_c:, :] = target_data

        # 3. Prepare Channel Mask
        # channel_mask = np.zeros((self.max_vars,), dtype=np.float32)
        channel_mask = np.zeros((limit_c,), dtype=np.float32)
        channel_mask[:limit_c] = 1.0

        # 4. Prepare Variate IDs (Pad with PAD_VAL_ID)
        # padded_v_ids = np.full((self.max_vars,), self.pad_val_id, dtype=np.int64)
        padded_v_ids = np.full((limit_c,), self.pad_val_id, dtype=np.int64)
        if self.variate_ids is not None:
            valid_len_ids = min(len(self.variate_ids), self.max_vars)
            padded_v_ids[:valid_len_ids] = self.variate_ids[:valid_len_ids]

        # 5. Prepare Coordinates & Flag
        if self.site_coords is not None:
            out_coords = self.site_coords.astype(np.float32)
            has_coords_flag = np.array([1.0], dtype=np.float32)
        else:
            out_coords = np.array([0.0, 0.0], dtype=np.float32)
            has_coords_flag = np.array([0.0], dtype=np.float32)

        # 6. Split into Context and Target
        # Note: both are now shape [MAX_VARS, ...] to allow batching
        context = padded_window[:, :self.context_length]
        target = padded_window[:, self.context_length:]

        return {
            "context": context,  # [MAX_VARS, L_c]
            "target": target,  # [MAX_VARS, L_p]
            "variate_ids": padded_v_ids,  # [MAX_VARS]
            "channel_mask": channel_mask,  # [MAX_VARS]
            "site_coords": out_coords,  # [2]
            "has_coords": has_coords_flag  # [1]
        }


@register_dataset("deepwind_eval")
class DeepWindValidDataset(Dataset):
    """
    Map-style Validation Dataset for DeepWind Model.
    
    CRITICAL FIXES:
    - Strictly handles missing/NaN coordinates in metadata.
    - Prevents NaN propagation into model embeddings.
    """

    def __init__(
            self,
            npy_root: str | Path,
            metadata_path: str | Path,
            context_length: int,
            stride: Optional[int] = None,
            max_vars: int = 6,
            pad_val_id: int = 10,
    ) -> None:
        super().__init__()
        self.npy_root = Path(npy_root)
        self.metadata_path = Path(metadata_path)
        self.context_length = int(context_length)
        self.seq_len = self.context_length
        self.stride = int(stride or self.seq_len) 
        self.max_vars = max_vars
        self.pad_val_id = pad_val_id

        if not self.npy_root.is_dir():
            raise NotADirectoryError(f"npy_root must be a directory: {self.npy_root}")

        self._load_metadata_lookup()
        self.samples: List[Tuple[Path, int]] = []
        self._scan_and_index_files()

    def _load_metadata_lookup(self):
        """
        Loads metadata with STRICT NaN handling.
        """
        self.meta_lookup = {}
        if self.metadata_path.is_file():
            try:
                # 1. Read CSV, forcing NaNs to be standard numpy NaNs
                df = pd.read_csv(
                    self.metadata_path,
                    dtype={'filename': str, 'latitude': float, 'longitude': float} # Force float
                )
                
                for _, row in df.iterrows():
                    fname = row['filename']
                    
                    # 2. Extract & Sanitize Coordinates
                    lat = row.get('latitude')
                    lon = row.get('longitude')
                    
                    # STRICT CHECK: Only accept if BOTH are valid numbers (not NaN/None)
                    # np.isnan handles both np.nan and standard float('nan')
                    has_valid_coords = (
                        isinstance(lat, (float, int)) and 
                        isinstance(lon, (float, int)) and 
                        not np.isnan(lat) and 
                        not np.isnan(lon)
                    )
                    
                    final_lat = float(lat) if has_valid_coords else 0.0
                    final_lon = float(lon) if has_valid_coords else 0.0
                    
                    # 3. Parse Variate IDs
                    v_ids_str = str(row.get('variate_ids', ''))
                    parsed_ids = None
                    if v_ids_str and v_ids_str.lower() != 'nan':
                        try:
                            parsed_ids = np.array([int(x) for x in v_ids_str.split(',')], dtype=np.int64)
                        except ValueError:
                            pass
                    
                    # 4. Store cleaned data
                    self.meta_lookup[fname] = {
                        'lat': final_lat,
                        'lon': final_lon,
                        'has_coords': has_valid_coords, # Explicit boolean flag
                        'variate_ids': parsed_ids
                    }
            except Exception as e:
                if is_main_process():
                    logger.error(f"Failed to load metadata from {self.metadata_path}: {e}")
    
    def _scan_and_index_files(self):
        """
        Scans directory for valid validation windows (Last 20%).
        """
        npy_files = list(self.npy_root.glob("*.npy"))
        if not npy_files:
            return

        valid_files_count = 0
        total_windows = 0
        
        for npy_path in npy_files:
            try:
                data = np.load(npy_path, mmap_mode="r")
                if data.ndim != 2: continue
                _, T = data.shape
                
                # Split Strategy: Last 20%
                start_valid_boundary = int(T * 0.8)
                valid_segment_len = T - start_valid_boundary

                if valid_segment_len < self.seq_len:
                    continue

                valid_files_count += 1
                last_possible_start = T - self.seq_len
                
                # Generate indices
                file_starts = range(start_valid_boundary, last_possible_start + 1, self.stride)
                for s in file_starts:
                    self.samples.append((npy_path, s))
                    total_windows += 1

            except Exception:
                continue
        if is_main_process():
            logger.info(f"Validation Set Ready: {total_windows} samples from {valid_files_count} files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        npy_path, start_idx = self.samples[idx]
        end_idx = start_idx + self.seq_len

        # 1. Load Data
        try:
            data_mmap = np.load(npy_path, mmap_mode="r")
            raw_window = np.array(data_mmap[:, start_idx:end_idx], dtype=np.float32)
        except Exception as e:
            if is_main_process():
                logger.error(f"Error loading {npy_path}: {e}")
            raw_window = np.zeros((self.max_vars, self.seq_len), dtype=np.float32)
        
        C = raw_window.shape[0]
        
        # 2. Metadata Retrieval (From pre-cleaned lookup)
        meta = self.meta_lookup.get(npy_path.name, {})
        
        # --- FIX STARTS HERE ---
        # Explicitly check the boolean flag we computed during loading
        has_coords_bool = meta.get('has_coords', False)
        
        if has_coords_bool:
            # Metadata guarantees these are not NaN
            out_coords = np.array([meta['lon'], meta['lat']], dtype=np.float32)
            has_coords_flag = np.array([1.0], dtype=np.float32)
        else:
            # Fallback for missing or NaN coordinates
            out_coords = np.array([0.0, 0.0], dtype=np.float32)
            has_coords_flag = np.array([0.0], dtype=np.float32)
        # --- FIX ENDS HERE ---

        variate_ids = meta.get('variate_ids', None)

        # 3. Padding & Masking
        limit_c = min(C, self.max_vars)
        
        padded_window = np.zeros((self.max_vars, self.seq_len), dtype=np.float32)
        padded_window[:limit_c, :] = raw_window[:limit_c, :]

        channel_mask = np.zeros((self.max_vars,), dtype=np.float32)
        channel_mask[:limit_c] = 1.0

        padded_v_ids = np.full((self.max_vars,), self.pad_val_id, dtype=np.int64)
        if variate_ids is not None:
            valid_len_ids = min(len(variate_ids), self.max_vars)
            padded_v_ids[:valid_len_ids] = variate_ids[:valid_len_ids]

        return {
            "context": padded_window,
            "variate_ids": padded_v_ids,
            "channel_mask": channel_mask,
            "site_coords": out_coords,     # Guaranteed no NaNs
            "has_coords": has_coords_flag  # Correctly 0.0 if coords missing
        }


@register_dataset("deepwind_train_2")
class DeepWindTrainDataset_2(Dataset):
    """
    Map-style Validation Dataset for DeepWind Model.
    
    CRITICAL FIXES:
    - Strictly handles missing/NaN coordinates in metadata.
    - Prevents NaN propagation into model embeddings.
    """

    def __init__(
            self,
            npy_root: str | Path,
            metadata_path: str | Path,
            context_length: int,
            stride: Optional[int] = None,
            max_vars: int = 6,
            pad_val_id: int = 10,
    ) -> None:
        super().__init__()
        self.npy_root = Path(npy_root)
        self.metadata_path = Path(metadata_path)
        self.context_length = int(context_length)
        self.seq_len = self.context_length
        self.stride = int(stride or self.seq_len) 
        self.max_vars = max_vars
        self.pad_val_id = pad_val_id

        if not self.npy_root.is_dir():
            raise NotADirectoryError(f"npy_root must be a directory: {self.npy_root}")

        self._load_metadata_lookup()
        self.samples: List[Tuple[Path, int]] = []
        self._scan_and_index_files()

    def _load_metadata_lookup(self):
        """
        Loads metadata with STRICT NaN handling.
        """
        self.meta_lookup = {}
        if self.metadata_path.is_file():
            try:
                # 1. Read CSV, forcing NaNs to be standard numpy NaNs
                df = pd.read_csv(
                    self.metadata_path,
                    dtype={'filename': str, 'latitude': float, 'longitude': float} # Force float
                )
                
                for _, row in df.iterrows():
                    fname = row['filename']
                    
                    # 2. Extract & Sanitize Coordinates
                    lat = row.get('latitude')
                    lon = row.get('longitude')
                    
                    # STRICT CHECK: Only accept if BOTH are valid numbers (not NaN/None)
                    # np.isnan handles both np.nan and standard float('nan')
                    has_valid_coords = (
                        isinstance(lat, (float, int)) and 
                        isinstance(lon, (float, int)) and 
                        not np.isnan(lat) and 
                        not np.isnan(lon)
                    )
                    
                    final_lat = float(lat) if has_valid_coords else 0.0
                    final_lon = float(lon) if has_valid_coords else 0.0
                    
                    # 3. Parse Variate IDs
                    v_ids_str = str(row.get('variate_ids', ''))
                    parsed_ids = None
                    if v_ids_str and v_ids_str.lower() != 'nan':
                        try:
                            parsed_ids = np.array([int(x) for x in v_ids_str.split(',')], dtype=np.int64)
                        except ValueError:
                            pass
                    
                    # 4. Store cleaned data
                    self.meta_lookup[fname] = {
                        'lat': final_lat,
                        'lon': final_lon,
                        'has_coords': has_valid_coords, # Explicit boolean flag
                        'variate_ids': parsed_ids
                    }
            except Exception as e:
                logger.error(f"Failed to load metadata from {self.metadata_path}: {e}")

    def _scan_and_index_files(self):
        """
        Scans directory for valid validation windows (Last 20%).
        """
        npy_files = list(self.npy_root.glob("*.npy"))
        if not npy_files:
            return

        valid_files_count = 0
        total_windows = 0
        
        for npy_path in npy_files:
            try:
                data = np.load(npy_path, mmap_mode="r")
                if data.ndim != 2: continue
                _, T = data.shape
                
                # Split Strategy: Last 20%
                start_valid_boundary = int(T * 0.0)
                valid_segment_len = T - start_valid_boundary

                if valid_segment_len < self.seq_len:
                    continue

                valid_files_count += 1
                last_possible_start = T - self.seq_len
                
                # Generate indices
                file_starts = range(start_valid_boundary, last_possible_start + 1, self.stride)
                for s in file_starts:
                    self.samples.append((npy_path, s))
                    total_windows += 1

            except Exception:
                continue
        
        logger.info(f"Validation Set Ready: {total_windows} samples from {valid_files_count} files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        npy_path, start_idx = self.samples[idx]
        end_idx = start_idx + self.seq_len

        # 1. Load Data
        try:
            data_mmap = np.load(npy_path, mmap_mode="r")
            raw_window = np.array(data_mmap[:, start_idx:end_idx], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error loading {npy_path}: {e}")
            raw_window = np.zeros((self.max_vars, self.seq_len), dtype=np.float32)

        C = raw_window.shape[0]
        
        # 2. Metadata Retrieval (From pre-cleaned lookup)
        meta = self.meta_lookup.get(npy_path.name, {})
        
        # --- FIX STARTS HERE ---
        # Explicitly check the boolean flag we computed during loading
        has_coords_bool = meta.get('has_coords', False)
        
        if has_coords_bool:
            # Metadata guarantees these are not NaN
            out_coords = np.array([meta['lon'], meta['lat']], dtype=np.float32)
            has_coords_flag = np.array([1.0], dtype=np.float32)
        else:
            # Fallback for missing or NaN coordinates
            out_coords = np.array([0.0, 0.0], dtype=np.float32)
            has_coords_flag = np.array([0.0], dtype=np.float32)
        # --- FIX ENDS HERE ---

        variate_ids = meta.get('variate_ids', None)

        # 3. Padding & Masking
        limit_c = min(C, self.max_vars)
        
        padded_window = np.zeros((self.max_vars, self.seq_len), dtype=np.float32)
        padded_window[:limit_c, :] = raw_window[:limit_c, :]

        channel_mask = np.zeros((self.max_vars,), dtype=np.float32)
        channel_mask[:limit_c] = 1.0

        padded_v_ids = np.full((self.max_vars,), self.pad_val_id, dtype=np.int64)
        if variate_ids is not None:
            valid_len_ids = min(len(variate_ids), self.max_vars)
            padded_v_ids[:valid_len_ids] = variate_ids[:valid_len_ids]

        return {
            "context": padded_window,
            "variate_ids": padded_v_ids,
            "channel_mask": channel_mask,
            "site_coords": out_coords,     # Guaranteed no NaNs
            "has_coords": has_coords_flag  # Correctly 0.0 if coords missing
        }        


class DeepWindFinetuneDataset(Dataset):
    """
    Map-style evaluation dataset for a single .npy file.
    Aligned with DeepWindTrainDataset logic (Padding, Masking, Coordinates).

    Produces:
        {
            "context":      [MAX_VARS, context_length], # Padded
            "variate_ids":  [MAX_VARS],
            "channel_mask": [MAX_VARS],
            "site_coords":  [2],
            "has_coords":   [1]
        }
    """

    def __init__(
            self,
            npy_path: str | Path,
            metadata_path: str | Path,
            context_length: int,
            stride: Optional[int] = None,
            finetune_rate: str = float,  
            max_vars: int = 6,
            pad_val_id = 10,
    ) -> None:
        """
        Args:
            npy_path: Path to the .npy file with shape [V, T] (Variates, Time).
            metadata_path: Path to the metadata CSV file.
            context_length: Length of history sequence (L_c).
            prediction_length: Length of prediction sequence (L_p).
            stride: Stride between sliding windows. Defaults to prediction_length.
            split: Dataset split to use.
                   - 'test': Uses the last 20% (0.8 -> 1.0) for zero-shot evaluation.
                   - 'val': Uses the middle 10% (0.7 -> 0.8) for validation.
                   - 'all': Uses the entire dataset (0.0 -> 1.0).
                   * Note: If the selected split is shorter than the sequence length,
                     it automatically falls back to 'all'.
        """
        super().__init__()
        self.npy_path = Path(npy_path)
        self.metadata_path = Path(metadata_path)
        self.context_length = int(context_length)
        self.seq_len = self.context_length
        self.stride = stride

        # --- Consistency Config (Must match Train Dataset) ---
        self.max_vars = max_vars
        self.pad_val_id = pad_val_id

        if not self.npy_path.is_file():
            raise FileNotFoundError(f"File not found: {self.npy_path}")

        # 1. Load Data (Memory Map)
        # Keep mmap open to access data without loading the entire file into RAM.
        self.data = np.load(self.npy_path, mmap_mode="r")

        if self.data.ndim != 2:
            raise RuntimeError(f"Expected [V, T] shape, got {self.data.shape}")

        self.V, self.T = self.data.shape
        
        start_idx = 0
        end_idx = int(finetune_rate * self.T)

        # --- Fallback Logic ---
        # If the selected split segment is shorter than the required sequence length,
        # fallback to using the entire dataset to ensure we can evaluate something.
        current_segment_len = end_idx - start_idx

        if current_segment_len < self.seq_len:
            raise ValueError(
                "(data_length * finetune_rate) < context_length, which means finetune_date is too small. "
            )

        # 3. Generate Valid Start Indices
        # We generate indices such that the window [start, start + seq_len) fits strictly within [start_idx, end_idx).
        last_possible_start = end_idx - self.seq_len

        # range is inclusive for start, exclusive for end, so we add 1
        self.starts = list(range(start_idx, last_possible_start + 1, self.stride))

        if len(self.starts) == 0:
            if is_main_process():
                logger.info(f"[Warning] No samples generated. Check sequence length vs dataset size.")

        # 3. Load Metadata (Aligned with Train logic)
        self._load_metadata()

    def _load_metadata(self):
        """Helper to load and format metadata exactly like the training set."""
        self.site_coords = None
        self.variate_ids = None

        if self.metadata_path.is_file():
            df = pd.read_csv(
                self.metadata_path,
                dtype={'filename': str, 'latitude': np.float32, 'longitude': np.float32}
            )
            target_filename = self.npy_path.name
            row = df[df['filename'] == target_filename]

            if not row.empty:
                data = row.iloc[0]

                # --- Coordinates ---
                lon, lat = data['longitude'], data['latitude']
                if pd.notna(lon) and pd.notna(lat):
                    self.site_coords = np.array([lon, lat], dtype=np.float32)

                # --- Variate IDs ---
                v_ids_str = str(data.get('variate_ids', ''))
                if pd.notna(data.get('variate_ids')) and v_ids_str.strip() and v_ids_str.lower() != 'nan':
                    try:
                        self.variate_ids = np.array([int(x) for x in v_ids_str.split(',')], dtype=np.int64)
                    except ValueError:
                        if is_main_process():
                            logger.info(f"[Warning] Invalid variate_ids format for {target_filename}")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches the i-th window, applies padding and masking to match training data.
        """
        s = self.starts[idx]
        e = s + self.seq_len

        # 1. Load Raw Window [V, seq_len]
        # Use copy=True to ensure we have a writable array for padding later if needed
        full_window = np.array(self.data[:, s:e], dtype=np.float32)

        C = full_window.shape[0]

        # --- Logic from DeepWindTrainDataset ---

        # 2. Pad/Truncate Data (values)
        limit_c = min(C, self.max_vars)

        # Initialize container with zeros [MAX_VARS, seq_len]
        padded_window = np.zeros((self.max_vars, self.seq_len), dtype=np.float32)

        # Fill valid channels
        padded_window[:limit_c, :] = full_window[:limit_c, :]

        # If we need padding (C < MAX_VARS), Train logic repeats the 0-th channel
        # if C < self.max_vars:
        #     # Train code: padded_context[limit_c:, :] = window[0, :]
        #     target_data = full_window[0, :]
        #     padded_window[limit_c:, :] = target_data

        # 3. Prepare Channel Mask
        channel_mask = np.zeros((self.max_vars,), dtype=np.float32)
        channel_mask[:limit_c] = 1.0

        # 4. Prepare Variate IDs (Pad with PAD_VAL_ID)
        padded_v_ids = np.full((self.max_vars,), self.pad_val_id, dtype=np.int64)
        if self.variate_ids is not None:
            valid_len_ids = min(len(self.variate_ids), self.max_vars)
            padded_v_ids[:valid_len_ids] = self.variate_ids[:valid_len_ids]

        # 5. Prepare Coordinates & Flag
        if self.site_coords is not None:
            out_coords = self.site_coords.astype(np.float32)
            has_coords_flag = np.array([1.0], dtype=np.float32)
        else:
            out_coords = np.array([0.0, 0.0], dtype=np.float32)
            has_coords_flag = np.array([0.0], dtype=np.float32)

        # 6. Split into Context and Target
        # Note: both are now shape [MAX_VARS, ...] to allow batching
        context = padded_window[:, :self.context_length]

        return {
            "context": context,  # [MAX_VARS, L_c]
            "variate_ids": padded_v_ids,  # [MAX_VARS]
            "channel_mask": channel_mask,  # [MAX_VARS]
            "site_coords": out_coords,  # [2]
            "has_coords": has_coords_flag  # [1]
        }
        