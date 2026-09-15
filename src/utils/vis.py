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
    seed: int = 42,
    indices: list | None = None,
    fname: str | None = None,
    ylabel: str | None = None,
    dpi: int = 300,
):
    """
    Visualizes forecast results with history, ground truth, point forecast,
    and confidence intervals (50% & 90%).

    Args:
        results:         Dict with ``targets``, ``point_preds``, ``quantile_preds``,
                         ``history`` arrays.
        quantiles_list:  Quantile levels (e.g. ``[0.05, ..., 0.95]``).
        save_dir:        Output directory.
        dataset_name:    Dataset identifier (used in the per-panel title).
        num_plots:       Number of samples to draw (ignored when ``indices`` given).
        plot_hist_len:   Number of trailing history steps to show.
        seed:            RNG seed for deterministic sample selection.
        indices:         Explicit sample indices (overrides seed-based choice).
                         Enables reproducible cross-model comparison via a shared
                         manifest.
        fname:           Output filename stem (default ``<dataset_name>_vis``).
        ylabel:          y-axis label (default None).
        dpi:             Save resolution.

    Returns:
        Absolute path of the written PNG.
    """
    # 1. Extract data
    y_true = results["targets"]       # [N, P]
    y_pred = results["point_preds"]   # [N, P]
    y_quantiles = results["quantile_preds"]  # [N, P, Q]
    y_hist = results["history"]       # [N, H]

    total_samples = y_true.shape[0]

    # 2. Select samples — explicit indices (manifest) or deterministic seed.
    if indices is None:
        num_plots = min(num_plots, total_samples)
        rng = np.random.RandomState(seed)
        indices = rng.choice(total_samples, size=num_plots, replace=False)
        indices = sorted(int(i) for i in indices)
    else:
        indices = sorted(int(i) for i in indices)
        num_plots = len(indices)

    # 3. Quantile indices: P05/P95 -> 90% PI, P25/P75 -> 50% PI.
    q_arr = np.asarray(quantiles_list)

    def get_q_idx(target_q):
        return int((np.abs(q_arr - target_q)).argmin())

    idx_p05 = get_q_idx(0.05)
    idx_p95 = get_q_idx(0.95)
    idx_p25 = get_q_idx(0.25)
    idx_p75 = get_q_idx(0.75)

    # 4. Plot one stacked panel per sample.
    fig, axes = plt.subplots(
        num_plots, 1, figsize=(12, 3.5 * num_plots), constrained_layout=True
    )
    if num_plots == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]

        hist_segment = y_hist[idx, -plot_hist_len:]
        true_segment = y_true[idx]
        pred_segment = y_pred[idx]

        q_p05 = y_quantiles[idx, :, idx_p05]
        q_p95 = y_quantiles[idx, :, idx_p95]
        q_p25 = y_quantiles[idx, :, idx_p25]
        q_p75 = y_quantiles[idx, :, idx_p75]

        T_hist = len(hist_segment)
        T_pred = len(true_segment)
        t_hist = np.arange(T_hist)
        t_pred = np.arange(T_hist, T_hist + T_pred)

        ax.plot(t_hist, hist_segment, color="black", alpha=0.7,
                label="History (Context)", linewidth=1.5)
        ax.plot(t_pred, true_segment, color="green", alpha=0.8,
                label="Ground Truth", linewidth=1.5)
        ax.fill_between(t_pred, q_p05, q_p95, color="dodgerblue", alpha=0.2,
                        label="90% Prediction Interval", zorder=1)
        ax.fill_between(t_pred, q_p25, q_p75, color="dodgerblue", alpha=0.4,
                        label="50% Prediction Interval", zorder=2)
        ax.plot(t_pred, pred_segment, color="firebrick", linestyle="--",
                linewidth=2, label="Point Forecast (Median)", zorder=3)

        ax.set_title(f"Sample {idx} | {dataset_name}", fontsize=10,
                     fontweight="bold", loc="left")
        ax.axvline(x=T_hist, color="gray", linestyle=":", alpha=0.5)
        ax.grid(True, alpha=0.2)
        if ylabel:
            ax.set_ylabel(ylabel)

        if i == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3,
                      frameon=True, fontsize=9)

    # 5. Save
    os.makedirs(save_dir, exist_ok=True)
    stem = fname if fname else f"{dataset_name}_vis"
    save_path = os.path.join(save_dir, f"{stem}.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path
