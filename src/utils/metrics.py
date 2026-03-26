import numpy as np
from scipy.stats import norm


class ForecastingEvaluator:
    def __init__(self, season_length=24, quantiles=None):
        self.season_length = season_length

        # Use the specific quantiles provided by the user
        if quantiles is None:
            # Default quantiles
            self.quantiles = np.array([
                0.01, 0.05, 0.1, 0.15, 0.2,
                0.25, 0.3, 0.35, 0.4, 0.45,
                0.5,
                0.55, 0.6, 0.65, 0.7, 0.75,
                0.8, 0.85, 0.9, 0.95, 0.99,
            ])
        else:
            self.quantiles = np.array(quantiles)
    
        # Pre-calculate Z-scores for Gaussian distribution (used for Naive baseline)
        self.z_scores = norm.ppf(self.quantiles)

    def get_seasonal_naive_forecast(self, history, pred_len):
        """
        Generates probabilistic Seasonal Naive forecasts.
        Returns:
            naive_mean: [N, pred_len]
            naive_quantiles: [N, pred_len, Q]
        """
        N, H = history.shape
        naive_mean = np.zeros((N, pred_len))
        
        # 1. Point Forecast
        for i in range(pred_len):
            lookback_idx = -self.season_length + (i % self.season_length)
            naive_mean[:, i] = history[:, lookback_idx]

        # 2. Probabilistic Forecast (Gaussian assumption)
        if H > self.season_length:
            seasonal_diff = history[:, self.season_length:] - history[:, :-self.season_length]
            sigma = np.std(seasonal_diff, axis=1, keepdims=True) + 1e-8
        else:
            sigma = np.std(history, axis=1, keepdims=True) + 1e-8

        # Broadcast to [N, pred_len, Q]
        naive_quantiles = (
                naive_mean[:, :, np.newaxis] +
                self.z_scores.reshape(1, 1, -1) * sigma[:, :, np.newaxis]
        )
        return naive_mean, naive_quantiles

    def calc_nrmse(self, y_true, y_pred_mean, capacity):
        """
        Normalized Root Mean Square Error (nRMSE).
        Formula: sqrt( mean( ((y_true - y_pred) / capacity)^2 ) )
        Range: [0, inf), lower is better.
        """
        self._check_capacity(capacity)
        
        # 1. Calculate squared normalized error per sample [N, T]
        #    (y - y_hat)^2 / C^2  =  ((y - y_hat) / C)^2
        #    Broadcasting capacity: [N] -> [N, 1]
        err_sq = ((y_true - y_pred_mean) / capacity[:, np.newaxis]) ** 2
        
        # 2. Mean over all samples and time steps
        mse_norm = np.nanmean(err_sq)
        
        # 3. Square root
        return np.sqrt(mse_norm)

    def calc_nmae(self, y_true, y_pred_mean, capacity):
        """
        Normalized Mean Absolute Error (nMAE).
        Formula: mean( |y_true - y_pred| / capacity )
        Range: [0, inf), lower is better.
        """
        self._check_capacity(capacity)
        
        # 1. Calculate absolute normalized error
        err_abs = np.abs(y_true - y_pred_mean) / capacity[:, np.newaxis]
        
        # 2. Mean over all samples and time steps
        return np.nanmean(err_abs)

    def calc_grid_accuracy(self, y_true, y_pred_mean, capacity):
        """
        Grid Accuracy (A).
        Formula: 1 - nRMSE
        Range: (-inf, 1.0], higher is better. (Ideally close to 1.0)
        """
        # Rely on calc_nrmse to ensure consistency
        nrmse = self.calc_nrmse(y_true, y_pred_mean, capacity)
        return 1.0 - nrmse

    def calc_qualified_rate(self, y_true, y_pred_mean, capacity, threshold=0.25):
        """
        Qualified Rate (Q).
        Formula: Percentage of points where |y_true - y_pred| <= capacity * threshold
        Range: [0.0, 1.0], higher is better.
        
        Args:
            threshold: 0.25 for short-term (day-ahead), 0.15 for ultra-short-term.
        """
        self._check_capacity(capacity)
        
        # 1. Absolute Error [N, T]
        abs_err = np.abs(y_true - y_pred_mean)
        
        # 2. Allowed Deviation [N, 1]
        allowed_dev = capacity[:, np.newaxis] * threshold
        
        # 3. Boolean check [N, T]
        is_qualified = abs_err <= allowed_dev
        
        # 4. Mean over all points
        return np.nanmean(is_qualified.astype(float))

    def calc_ncrps(self, y_true, y_pred_quantiles, capacity):
        """
        Normalized Continuous Ranked Probability Score (nCRPS).
        Formula: Sum( Pinball_Loss(q) / Capacity ) / (N * T * Num_Quantiles)
        
        Args:
            y_true: np.ndarray, shape [N, T]
            y_pred_quantiles: np.ndarray, shape [N, T, Num_Quantiles]
            capacity: np.ndarray, shape [N] or scalar. The capacity C_i for each sample.
        """
        loss_sum = 0.0
        
        # Ensure y_true is 2D [N, T]
        if y_true.ndim == 3:
            y_true = y_true[:, :, 0]
            
        N, T = y_true.shape
        
        self._check_capacity(capacity)

        # Loop over quantiles
        for idx, q in enumerate(self.quantiles):
            # y_pred_q: [N, T]
            y_pred_q = y_pred_quantiles[:, :, idx]
            
            errors = y_true - y_pred_q
            
            # Pinball Loss: max(q * e, (q-1) * e)
            # Shape: [N, T]
            loss = np.maximum(q * errors, (q - 1) * errors)
            
            # --- Key Change: Normalize by Capacity immediately ---
            # Broadcasting: [N, T] / [N, 1]
            norm_loss = loss * 2 / capacity[:, np.newaxis]
            
            loss_sum += np.sum(norm_loss)
            
        # Average over (N * T * Num_Quantiles)
        # Note: The sum already includes the summation over i, t, and q.
        # We just divide by the total count.
        total_steps = N * T * len(self.quantiles)
        
        return loss_sum / total_steps

    def calc_mase(self, y_true, y_pred_mean, history):
        """
        Mean Absolute Scaled Error (MASE).
        Benchmark against Seasonal Naive.
        Range: [0, inf), < 1 means better than naive.
        """
        # Numerator: MAE of forecast
        forecast_mae = np.mean(np.abs(y_true - y_pred_mean), axis=1) # [N]

        # Denominator: MAE of naive in-sample
        if history.shape[1] > self.season_length:
            seasonal_diff = np.abs(history[:, self.season_length:] - history[:, :-self.season_length])
            scale = np.mean(seasonal_diff, axis=1)
        else:
            scale = np.mean(np.abs(np.diff(history, axis=1)), axis=1)

        # Masking invalid scales
        valid_mask = scale > 1e-8
        if np.sum(valid_mask) == 0:
            return np.nan

        valid_mase = forecast_mae[valid_mask] / scale[valid_mask]
        return np.mean(valid_mase)

    def _check_capacity(self, capacity):
        """Helper to ensure capacity is valid for broadcasting."""
        if capacity is None:
            raise ValueError("Capacity cannot be None for normalized metrics.")
        if np.any(capacity <= 0):
             # You might want to handle 0 capacity gracefully or warn
             pass
