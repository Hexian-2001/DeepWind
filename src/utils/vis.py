import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_forecasts(
    results: dict, 
    quantiles_list: list, 
    save_dir: str, 
    dataset_name: str,
    num_plots: int = 8, 
    plot_hist_len: int = 144, 
    seed: int = 42            
):
    """
    Visualizes forecast results with history, ground truth, point forecast, and confidence intervals (50% & 90%).
    """
    # 1. Extract Data
    y_true = results["targets"]       # [N, P]
    y_pred = results["point_preds"]   # [N, P]
    y_quantiles = results["quantile_preds"] # [N, P, Q]
    y_hist = results["history"]       # [N, H]
    
    total_samples = y_true.shape[0]
    num_plots = min(num_plots, total_samples)
    
    # 2. Select Fixed Samples (Deterministic)
    rng = np.random.RandomState(seed)
    indices = rng.choice(total_samples, size=num_plots, replace=False)
    
    # 3. Identify Quantile Indices
    # Helper to find closest index for a probability (e.g., 0.05 for 5th percentile)
    q_arr = np.array(quantiles_list)
    def get_q_idx(target_q):
        return (np.abs(q_arr - target_q)).argmin()

    # 90% Confidence Interval -> P5 to P95
    idx_p05 = get_q_idx(0.05)
    idx_p95 = get_q_idx(0.95)
    
    # 50% Confidence Interval -> P25 to P75
    idx_p25 = get_q_idx(0.25)
    idx_p75 = get_q_idx(0.75)

    # 4. Create Plot
    # Adjust figure height based on number of plots
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3.5 * num_plots), constrained_layout=True)
    if num_plots == 1: axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # --- Prepare Data Snippets ---
        # Crop history to the last 'plot_hist_len' steps
        hist_segment = y_hist[idx, -plot_hist_len:]
        true_segment = y_true[idx]
        pred_segment = y_pred[idx]
        
        # Quantiles
        q_p05 = y_quantiles[idx, :, idx_p05]
        q_p95 = y_quantiles[idx, :, idx_p95]
        q_p25 = y_quantiles[idx, :, idx_p25]
        q_p75 = y_quantiles[idx, :, idx_p75]
        
        # Time Axis (Indices)
        T_hist = len(hist_segment)
        T_pred = len(true_segment)
        t_hist = np.arange(T_hist)
        t_pred = np.arange(T_hist, T_hist + T_pred)
        
        # --- Plotting ---
        
        # 1. History (Black)
        ax.plot(t_hist, hist_segment, color="black", alpha=0.7, label="History (Context)", linewidth=1.5)
        
        # 2. Ground Truth (Green)
        ax.plot(t_pred, true_segment, color="green", alpha=0.8, label="Ground Truth", linewidth=1.5)
        
        # 3. 90% Confidence Interval (Light Blue)
        ax.fill_between(t_pred, q_p05, q_p95, color="dodgerblue", alpha=0.2, label="90% Prediction Interval", zorder=1)
        
        # 4. 50% Confidence Interval (Darker Blue)
        ax.fill_between(t_pred, q_p25, q_p75, color="dodgerblue", alpha=0.4, label="50% Prediction Interval", zorder=2)
        
        # 5. Point Forecast (Red Dashed)
        ax.plot(t_pred, pred_segment, color="firebrick", linestyle="--", linewidth=2, label="Point Forecast (Median)", zorder=3)
        
        # --- Styling ---
        ax.set_title(f"Sample {idx} | {dataset_name}", fontsize=10, fontweight="bold", loc="left")
        ax.axvline(x=T_hist, color="gray", linestyle=":", alpha=0.5) # Separation line
        ax.grid(True, alpha=0.2)
        
        # Legend only on the first plot to save space
        if i == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=True, fontsize=9)

    # 5. Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_vis.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to: {save_path}")
    