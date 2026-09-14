# tools/plot_rolling.py
import os

import numpy as np
from omegaconf import OmegaConf
from src.evaluation.reporter import EvaluationReporter

RESULTS_ROOT = os.environ.get(
    "DEEPWIND_RESULTS_ROOT",
    "/scratch/pawsey0115/hwang4/results/DeepWind-Research",
)
NPZ_PATH = os.path.join(
    RESULTS_ROOT,
    "results/deepwind/deepwind_patch_16_no_stats/raw_results/csg_wind_5/raw_H12.npz",
)
OUTPUT_DIR = os.path.join(
    RESULTS_ROOT, "results/deepwind/deepwind_patch_16_no_stats"
)

cfg = OmegaConf.create({
    "output": {
        "rolling_plot_segment":  1440,
        "rolling_plot_max_segs": 10,
        "save_plots":            False,
        "num_plots":             4,
        "plot_hist_len":         144,
    },
    "seed": 42,
})

res          = np.load(NPZ_PATH, allow_pickle=True)
dataset_name = str(res["dataset_name"])
horizon_h    = int(res["horizon_h"])
pred_len     = res["targets"].shape[1]
quantiles    = res["quantiles"]  

results = {
    "targets":        res["targets"],
    "point_preds":    res["point_preds"],
    "quantile_preds": res["quantile_preds"],
    "history":        res["history"],
}

reporter = EvaluationReporter(
    output_dir = OUTPUT_DIR,
    cfg        = cfg,
    quantiles  = quantiles,
)

reporter._save_rolling_plots(results, dataset_name, horizon_h, pred_len)