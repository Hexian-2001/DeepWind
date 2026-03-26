import os
import logging
import json
import hydra
import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from peft import PeftModel

from src.models.deepwind import DeepWindModel
from src.inference.generator import DeepWindForecaster, Forecast
from src.data.datasets import DeepWindTestDataset  
from src.utils.metrics import ForecastingEvaluator 
from src.utils.vis import visualize_forecasts


DATASET_RES_CONFIG = {
    "30651": 5,
    "43458": 5,
    "75354": 5,
    "76016": 5,
    "csg_wind_5": 15,
    "gefc12_wind_7": 60,
    "gefc14_wind_10": 60,
    "penmanshiel_15": 10
}

DATASET_CAPACITY_CONFIG = {
    "30651": 16.0,
    "43458": 16.0,
    "75354": 16.0,
    "76016": 4.0,
    "csg_wind_5": 35,
    "gefc12_wind_7": 1.0,
    "gefc14_wind_10": 1.0,
    "penmanshiel_15": 2080.0
}


logger = logging.getLogger(__name__)

def get_dataset_capacity(dataset_name):
    if dataset_name in DATASET_CAPACITY_CONFIG:
        return DATASET_CAPACITY_CONFIG[dataset_name]
    for key, cap in DATASET_CAPACITY_CONFIG.items():
        if key in dataset_name:
            return cap
    raise ValueError(f"Capacity for dataset '{dataset_name}' not defined.")

def get_pred_len(dataset_name, horizon_hours):
    res = DATASET_RES_CONFIG.get(dataset_name)
    if res is None:
        raise ValueError(f"Resolution for {dataset_name} not defined.")
    return int(horizon_hours * 60 / res)

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def to_device(batch: Dict, device: torch.device):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v
    return new_batch

def gather_and_merge(local_tensor: torch.Tensor, world_size: int, device: torch.device) -> torch.Tensor:
    
    if world_size == 1:
        return local_tensor

    # 1. Move to GPU for NCCL communication
    tensor_gpu = local_tensor.to(device)
    
    # 2. Gather list of tensors
    gathered_list = [torch.zeros_like(tensor_gpu) for _ in range(world_size)]
    dist.all_gather(gathered_list, tensor_gpu)
    
    # 3. Concatenate and move back to CPU to save VRAM
    merged = torch.cat(gathered_list, dim=0).cpu()
    return merged

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class DistributedEvaluator:
    def __init__(self, cfg: DictConfig, device: torch.device, rank: int, world_size: int):
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size
       
        self.model = self._load_model()
        self.forecaster = DeepWindForecaster(self.model, cfg=cfg)
    
    def _load_model(self):
        logger.info(f"Rank {self.rank}: Loading base model from {self.cfg.inference.checkpoint_path}")
        model = DeepWindModel.from_pretrained(self.cfg.inference.checkpoint_path)
        # 2. Check if a fine-tuned adapter path is provided in config
        # Assuming your config has a field like self.cfg.inference.adapter_path
        adapter_path = self.cfg.inference.get("adapter_path", None)
        if adapter_path and os.path.exists(adapter_path):
            logger.info(f"Rank {self.rank}: Fine-tuned adapter detected at {adapter_path}. Merging weights...")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            logger.info(f"Rank {self.rank}: Adapter merged successfully.")
        else:
            logger.info(f"Rank {self.rank}: No adapter found. Using base model for inference.")
        # 3. Finalize model state
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def run_inference(self, dataloader: DataLoader, dataset_len: int, pred_len: int) -> Dict[str, np.ndarray]:
        
        local_preds_point = []
        local_preds_quantile = []
        local_targets = []
        local_history = []
        
        iterator = dataloader
        if self.rank == 0:
            iterator = tqdm(dataloader, desc="Inference", leave=False)

        for batch in iterator:
            batch = to_device(batch, self.device)

            output: Forecast = self.forecaster.forecast(
                context=batch["context"],
                site_coords=batch.get("site_coords"),
                variate_ids=batch.get("variate_ids"),
                channel_mask=batch.get("channel_mask"),
                has_coords=batch.get("has_coords"),
                prediction_length=pred_len,
                kv_cache=self.cfg.inference.use_kv_cache,
                mqd_infer=self.cfg.inference.mqd_infer,
            )
            
            p_preds = output.point_preds[:, 0, :].cpu()
            q_preds = output.quantile_preds[:, 0, :, :].cpu()
            
            targets = batch["target"][:, 0, :].cpu()
            history = batch["context"][:, 0, :].cpu()

            local_preds_point.append(p_preds)
            local_preds_quantile.append(q_preds)
            local_targets.append(targets)
            local_history.append(history)

        if not local_preds_point:
            return {}

        local_point_cat = torch.cat(local_preds_point, dim=0)     # [Local_B, T]
        local_quantile_cat = torch.cat(local_preds_quantile, dim=0) # [Local_B, T, Q]
        local_target_cat = torch.cat(local_targets, dim=0)
        local_history_cat = torch.cat(local_history, dim=0)

        logger.info(f"Rank {self.rank}: Gathering results...")
        all_point = gather_and_merge(local_point_cat, self.world_size, self.device)
        all_quantile = gather_and_merge(local_quantile_cat, self.world_size, self.device)
        all_target = gather_and_merge(local_target_cat, self.world_size, self.device)
        all_history = gather_and_merge(local_history_cat, self.world_size, self.device)

        
        if self.rank == 0:
            logger.info(f"Gathered size: {all_point.shape[0]}, True dataset size: {dataset_len}")
            all_point = all_point[:dataset_len]
            all_quantile = all_quantile[:dataset_len]
            all_target = all_target[:dataset_len]
            all_history = all_history[:dataset_len]
            
            return {
                "point_preds": all_point.numpy(),
                "quantile_preds": all_quantile.numpy(),
                "targets": all_target.numpy(),
                "history": all_history.numpy()
            }
        
        return {}

    def compute_metrics(
        self, 
        results: Dict[str, np.ndarray], 
        capacity: np.ndarray
    ) -> Dict:
        """
        Computes forecasting metrics.
        
        Different normalization logic:
        1. Capacity-based metrics (nRMSE, nMAE, Accuracy, Q-Rate):
           Normalized by installed capacity (MW). No need for Naive scaling.
        2. Relative metrics (MASE, CRPS):
           Normalized by Naive baseline to show relative skill (Model / Naive).
        
        Args:
            results: Dictionary containing 'targets', 'point_preds', 'quantile_preds', 'history'.
            capacity: The installed capacity array corresponding to the current dataset.
                      Shape: [N] or [N, 1].
        """
        if self.rank != 0 or not results:
            return {}

        logger.info("Computing Metrics...")

        # ---------------------------------------------------
        # 1. Unpack Data
        # ---------------------------------------------------
        y_true = results["targets"]              # [N, Pred_Len]
        y_pred = results["point_preds"]          # [N, Pred_Len]
        y_quantiles = results["quantile_preds"]  # [N, Pred_Len, Num_Quantiles]
        y_hist = results["history"]              # [N, Context_Len]
        
        # Ensure capacity is [N, 1] for broadcasting
        if capacity.ndim == 1:
            capacity = capacity[:, np.newaxis]

        # ---------------------------------------------------
        # 2. Initialize Evaluator & Naive Baseline
        # ---------------------------------------------------
        evaluator = ForecastingEvaluator(
            season_length=self.cfg.inference.seasonality,
            quantiles=np.array(self.model.config.quantiles)
        )

        # ---------------------------------------------------
        # 3. Metric Calculation Helper
        # ---------------------------------------------------
        def get_metrics_values(target, mean, quantiles, cap):
            """Calculates absolute metric values."""
            return {
                "ncrps": evaluator.calc_ncrps(target, quantiles, cap),
                "nMAE": evaluator.calc_nmae(target, mean, cap),
                "Accuracy": evaluator.calc_grid_accuracy(target, mean, cap),
                "Qualified_Rate": evaluator.calc_qualified_rate(target, mean, cap, threshold=0.25)
            }

        # Calculate Absolute Values
        model_metrics = get_metrics_values(y_true, y_pred, y_quantiles, capacity)

        logger.info(f"\nEvaluation Report:\n{json.dumps(model_metrics, indent=4, cls=NumpyEncoder)}")
        
        return model_metrics


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if rank == 0 else logging.WARN
    )

    if rank == 0:
        logger.info(f"Starting Evaluation on Rank {rank}/{world_size}")
        logger.info(f"Context Len: {cfg.data.context_length}, Horizon: {cfg.inference.horizon_h} hour.")

    evaluator = DistributedEvaluator(cfg, device, rank, world_size)

    test_files = cfg.data.get("test_npy_paths", [])
    horizon_hs = cfg.inference.get("horizon_h", [])
    if not test_files:
        logger.warning("No test files found in cfg.data.test_npy_paths!")
    
    for npy_path in test_files:
        for horizon_h in horizon_hs:
            dataset_name = os.path.basename(npy_path).replace(".npy", "")
        
            if rank == 0:
                logger.info(f"Processing Dataset: {dataset_name} | Path: {npy_path}")
        
            try:
                pred_len = get_pred_len(dataset_name, horizon_h)
            except ValueError as e:
                if is_main_process(): print(f"{e}, skipping...")
                continue

            dataset = DeepWindTestDataset(
                npy_path=npy_path,
                metadata_path=cfg.data.metadata_path,
                context_length=cfg.data.context_length, 
                prediction_length=pred_len,
                stride=cfg.data.stride or pred_len,  
            )

            # ---------------------------------------------------------------------
            # 1. Dynamic Capacity Extraction (Rank 0 only)
            # ---------------------------------------------------------------------
            # Default fallback
            try:
                capacity_val = get_dataset_capacity(dataset_name)
            except ValueError as e:
                if is_main_process(): print(f"{e}, skipping dataset...")
                continue

            sampler = DistributedSampler(
                dataset, 
                num_replicas=world_size, 
                rank=rank, 
                shuffle=False, 
                drop_last=False
            )

            dataloader = DataLoader(
                dataset,
                batch_size=cfg.inference.batch_size,
                num_workers=cfg.inference.num_workers,
                sampler=sampler,
                pin_memory=True
            )
        
            results = evaluator.run_inference(dataloader, len(dataset), pred_len)
        
            if rank == 0:
                logger.info(f"Calculating metrics for {dataset_name}...")
                total_samples = results["targets"].shape[0]
                capacity_array = np.full(total_samples, capacity_val, dtype=np.float32)
                metrics = evaluator.compute_metrics(results, capacity=capacity_array)

                final_result = {
                    "dataset": dataset_name,
                    "horizon_hours": horizon_h,
                    "pred_len_steps": pred_len,
                    "metrics": metrics, 
                    "timestamp": os.popen('date').read().strip()
                }

                save_dir = os.path.join(cfg.output.output_dir, dataset_name)
                os.makedirs(save_dir, exist_ok=True)

                save_file = f"H_{horizon_h}.json" 
                save_path = os.path.join(save_dir, save_file)    

                with open(save_path, "w") as f:
                    json.dump(final_result, f, indent=4,cls=NumpyEncoder)

                print(f"[Saved] {save_path}")
                
                # ---------------------------------------------------------------------
                # Save raw inference data for custom plotting/analysis
                # ---------------------------------------------------------------------
                if cfg.output.get("save_raw_results", False):
                    raw_data_dir = os.path.join(cfg.output.output_dir, "raw_results", dataset_name)
                    os.makedirs(raw_data_dir, exist_ok=True)
                    
                    # Save as compressed numpy archive (.npz)
                    raw_data_path = os.path.join(raw_data_dir, f"raw_H{horizon_h}.npz")
                    
                    np.savez_compressed(
                        raw_data_path,
                        point_preds=results["point_preds"],      # Shape: [Samples, Pred_Len]
                        quantile_preds=results["quantile_preds"], # Shape: [Samples, Pred_Len, Quantiles]
                        targets=results["targets"],              # Shape: [Samples, Pred_Len]
                        history=results["history"],              # Shape: [Samples, Context_Len]
                        capacity=capacity_val,                    # Scalar meta-info
                        dataset_name=dataset_name,
                        horizon_h=horizon_h
                    )
                    logger.info(f"[Raw Data Saved] {raw_data_path}")

                if cfg.output.save_plots:
                    logger.info(f"Generating plots for {dataset_name}...")
                    quantiles_list = evaluator.model.config.quantiles
                
                    visualize_forecasts(
                        results=results,
                        quantiles_list=quantiles_list,
                        save_dir=os.path.join(cfg.output.output_dir, "plots"),
                        dataset_name=dataset_name,
                        num_plots=cfg.output.num_plots,
                        plot_hist_len=cfg.output.plot_hist_len,
                        seed=cfg.seed
                    )
        
            torch.distributed.barrier()
    
    cleanup_ddp()

if __name__ == "__main__":
    main()