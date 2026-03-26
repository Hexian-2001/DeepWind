from dataclasses import dataclass
from typing import cast, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int


from src.models.deepwind import DeepWindModel

import math
from torch import nn


@dataclass(frozen=True)
class Forecast:
    point_preds: Float[torch.Tensor, "batch variates future_time_steps"]
    quantile_preds: Float[torch.Tensor, "batch variates future_time_steps samples"] | None = None


class DeepWindForecaster:
    """
    Forecaster for DeepWindModel with quantile outputs and patch-wise AR decoding.

    - Deterministic forecast  : use median (q=0.5) at each AR step.
    - Sampling forecast (TODO): sample from quantile grid (see generate_samples stub).
    """

    def __init__(
            self,
            model: DeepWindModel,
            median_q: float = 0.5,
            cfg=None
    ) -> None:
        """
        Parameters
        ----------
        model : DeepWindModel
            The trained DeepWindModel instance.
        median_q : float, optional
            Which quantile to use as "median"/point forecast.
            Typically 0.5. We will pick the nearest quantile in model.quantiles.
        """
        self.model: DeepWindModel = model
        self.device = next(model.parameters()).device

        # Model must expose its quantile levels as a tensor or list
        if not hasattr(model.config, "quantiles"):
            raise ValueError("DeepWindModel.config must define `self.quantiles` (list or tensor of quantile levels).")

        q = torch.as_tensor(model.config.quantiles, dtype=torch.float32)
        self.registered_quantiles = q

        # Find index of quantile closest to `median_q`
        self.median_idx = int(torch.argmin(torch.abs(q - median_q)).item())
        
        self.cfg = cfg

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    @torch.no_grad()
    def forecast(
            self,
            context: Float[torch.Tensor, "batch variate time_steps"],
            site_coords,
            variate_ids,
            channel_mask: Optional[torch.Tensor] = None,  # (B, MAX_VARS) [1=Valid, 0=Pad]
            has_coords: Optional[torch.Tensor] = None,  # (B, 1) [1=HasCoords, 0=None]
            prediction_length: int = 16,
            kv_cache = None,
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            output_router_logits: bool = False,
            mqd_infer: bool = False
    ) -> Forecast:
        """
        Generate forecast for a batch of time series.

        Parameters
        ----------
        context : (B, V, T)
            Raw input context.
        group_ids : optional
            Group identifiers for each (B, V, L) patch in the decoder.
            If None, DeepWindModel will typically create a default.
        prediction_length : int
            Number of future time steps to predict.
        num_samples : int | None
            Number of sample trajectories. If None → deterministic (median) forecast.
        use_kv_cache : bool
            Kept for API compatibility. Current implementation does not
            yet exploit KV cache.

        Returns
        -------
        Forecast
            mean : (B, V, H)
            samples : (B, V, H, S) or None
        """

        if mqd_infer:
            point_preds, quantile_preds = self.generate_samples_mqd(
                context=context,
                site_coords=site_coords,
                variate_ids=variate_ids,
                channel_mask=channel_mask,
                has_coords=has_coords,
                prediction_length=prediction_length,
                kv_cache=kv_cache,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
            ) 
        else:
            point_preds, quantile_preds = self.generate_samples_greedy(
                context=context,
                site_coords=site_coords,
                variate_ids=variate_ids,
                channel_mask=channel_mask,
                has_coords=has_coords,
                prediction_length=prediction_length,
                kv_cache=kv_cache,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
            ) 
        

        return Forecast(point_preds=point_preds, quantile_preds=quantile_preds)


    @torch.no_grad()
    def generate_samples_mqd(
            self,
            context: Float[torch.Tensor, "batch variate time_steps"],
            site_coords,
            variate_ids,
            channel_mask,
            has_coords,
            prediction_length: int,
            kv_cache,
            output_hidden_states,
            output_attentions,
            output_router_logits,
    ) -> Tuple[
        Float[torch.Tensor, "batch variate pred_len"],
        Float[torch.Tensor, "batch variate pred_len samples"],
    ]:
        """
        Autoregressive Multi-Quantile Decoding (AMQD).

        This method implements an 'Expand-Collapse' strategy to maintain forecast uncertainty
        without exponential computation growth (similar to Beam Search depth-2).

        Algorithm Steps:
        1. Expand: Maintain Q parallel trajectories (where Q is the number of quantiles).
        2. Predict: Each of the Q trajectories predicts Q future quantiles (resulting in Q*Q candidates).
        3. Collapse: Pool the Q*Q candidates into a single distribution and re-extract the Q target quantiles.
        4. Update: Append these Q new values to the context and repeat.
        """
        model = self.model
        model.eval()
        device = self.device

        # 1. Setup Quantiles
        # ---------------------------------------------------------
        # Use the model's configured quantiles (e.g., the 21 values)
        quantiles_list = self.registered_quantiles
        # Convert to tensor: [0.01, ..., 0.99]
        target_quantiles = quantiles_list
        num_quantiles = len(quantiles_list) # Q

        # 2. Prepare Input Data
        # ---------------------------------------------------------
        B, V, T = context.shape
        context = context.to(device)
        site_coords = site_coords.to(device)
        variate_ids = variate_ids.to(device)
        if channel_mask is not None:
            channel_mask = channel_mask.to(device)
        if has_coords is not None:
            has_coords = has_coords.to(device)

        patch_size = int(model.config.input_patch_stride)
        # Calculate how many patch steps are needed
        rounded_steps = int(math.ceil(prediction_length / patch_size) * patch_size)
        num_iters = rounded_steps // patch_size

        collected_patches = []

        # 3. First Step (Initialization)
        # ---------------------------------------------------------
        # Initial prediction based on the original context (B, V, T)
        output = model(
            context=context,
            site_coords=site_coords,
            variate_ids=variate_ids,
            channel_mask=channel_mask,
            has_coords=has_coords,
            kv_cache=kv_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
        )

        # Get the prediction for the last patch: (B, V, P, Q)
        current_samples = output.denorm_quantile_preds[..., -patch_size:, :]

        # [IMPROVEMENT] Sort immediately to ensure monotonicity
        # This prevents quantile crossing (e.g., P10 > P90) before we fork trajectories.
        # It ensures Trajectory 0 gets the lowest value, and Trajectory Q gets the highest.
        current_samples, _ = torch.sort(current_samples, dim=-1)

        # 4. Expand Batch Dimension (B -> B * Q)
        # ---------------------------------------------------------
        # We now fork the context into Q parallel paths.

        # Expand Context: (B, V, T) -> (B*Q, V, T)
        cur_context = context.repeat_interleave(num_quantiles, dim=0)

        # Expand Metadata
        cur_site_coords = site_coords.repeat_interleave(num_quantiles, dim=0)
        cur_variate_ids = variate_ids.repeat_interleave(num_quantiles, dim=0)
        cur_channel_mask = channel_mask.repeat_interleave(num_quantiles, dim=0) if channel_mask is not None else None
        cur_has_coords = has_coords.repeat_interleave(num_quantiles, dim=0) if has_coords is not None else None

        # Prepare the data to be appended to the context
        # Transformation: (B, V, P, Q) -> (B, Q, V, P) -> (B*Q, V, P)
        next_input_patch = current_samples.permute(0, 3, 1, 2).reshape(B * num_quantiles, V, patch_size)

        collected_patches.append(next_input_patch)

        # Update Context with the first prediction
        cur_context = torch.cat([cur_context, next_input_patch], dim=-1)

        # Sliding Window: Clip context to max length
        max_seq_len = self.cfg.data.context_length
        if cur_context.shape[-1] > max_seq_len:
            cur_context = cur_context[..., -max_seq_len:]

        # 5. Autoregressive Loop (Expand -> Collapse)
        # ---------------------------------------------------------
        # Loop for the remaining steps (subtract 1 because first step is done)
        for _ in range(num_iters - 1):
            # Run model on expanded context: (B*Q, V, T_curr)
            output = model(
                context=cur_context,
                site_coords=cur_site_coords,
                variate_ids=cur_variate_ids,
                channel_mask=cur_channel_mask,
                has_coords=cur_has_coords,
                kv_cache=kv_cache,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
            )

            # Get predictions: (B*Q, V, P, Q)
            candidates = output.denorm_quantile_preds[..., -patch_size:, :]

            # [IMPROVEMENT] Sort raw outputs immediately
            # Although we sort again during collapse, sorting here ensures the 'candidates'
            # tensor itself is valid (monotonic) if inspected or used elsewhere.
            candidates, _ = torch.sort(candidates, dim=-1)

            # --- COLLAPSE STEP ---
            # We now have Q*Q candidates per original sample. We need to mix them
            # and reduce them back to Q quantiles.

            # 1. Reshape to separate the two Quantile dimensions
            # Shape: (B, Q (path), V, P, Q (pred))
            candidates = candidates.view(B, num_quantiles, V, patch_size, num_quantiles)

            # 2. Pool/Flatten the candidates
            # Permute to bring both Q dims to the end: (B, V, P, Q, Q)
            # Flatten to (B, V, P, Q*Q)
            candidates_pool = candidates.permute(0, 2, 3, 1, 4).reshape(B, V, patch_size, num_quantiles * num_quantiles)

            # 3. Sort the pooled candidates
            # This sorts the Q*Q values (e.g., 441 values) in ascending order.
            sorted_candidates, _ = torch.sort(candidates_pool, dim=-1) # (B, V, P, Q*Q)

            # 4. Select indices for the target quantiles
            # We want to pick values from the sorted array that correspond to [0.01, 0.05, ... 0.99]
            pool_size = num_quantiles * num_quantiles
            indices = target_quantiles * (pool_size - 1)
            indices = indices.long() # Shape: (Q,)

            # Extract the new Q values
            # Shape: (B, V, P, Q)
            next_quantiles = sorted_candidates[..., indices]

            # --- PREPARE FOR NEXT STEP ---
            # Reshape back to (B*Q, V, P) to match the expanded context batch size
            next_input_patch = next_quantiles.permute(0, 3, 1, 2).reshape(B * num_quantiles, V, patch_size)

            collected_patches.append(next_input_patch)

            # Append to context
            cur_context = torch.cat([cur_context, next_input_patch], dim=-1)

            # Sliding Window
            if cur_context.shape[-1] > max_seq_len:
                cur_context = cur_context[..., -max_seq_len:]

        # 6. Finalize Output
        # ---------------------------------------------------------
        # [FIX] Do not use absolute index T, because cur_context might have been cropped.
        # Instead, we retrieve the generated part from the end of the tensor.

        # We generated exactly 'rounded_steps' (e.g., 96 steps),
        # but the user might have asked for 'prediction_length' (e.g., 90 steps).

        # 1. Extract all generated patches (the suffix of the sequence)
        # Shape: (B*Q, V, rounded_steps)
        full_prediction = torch.cat(collected_patches, dim=-1)
        # 2. Crop to the exact requested prediction length
        # Shape: (B*Q, V, prediction_length)
        full_prediction = full_prediction[..., :prediction_length]

        # Reshape logic: (B*Q, ...) -> (B, Q, ...)
        full_prediction = full_prediction.view(B, num_quantiles, V, prediction_length)

        # Permute to standard format: (B, V, H, Samples/Quantiles)
        pred_samples = full_prediction.permute(0, 2, 3, 1)
        pred_samples, _ = torch.sort(pred_samples, dim=-1)

        # Select the median prediction as the point forecast
        median_idx = self.median_idx

        point_forecast = pred_samples[..., median_idx] # (B, V, Pred_Len)

        return point_forecast, pred_samples
    

    @torch.no_grad()
    def generate_samples_greedy(
            self,
            context: Float[torch.Tensor, "batch variate time_steps"],
            site_coords,
            variate_ids,
            channel_mask,
            has_coords,
            prediction_length: int,
            kv_cache,
            output_hidden_states,
            output_attentions,
            output_router_logits,
    ) -> Tuple[
        Float[torch.Tensor, "batch variate pred_len"],
        Float[torch.Tensor, "batch variate pred_len samples"],
    ]:
        """
        Greedy Autoregressive Decoding (Deterministic Baseline).

        Strategy:
        Always feed the 'median' (or specified median_idx) prediction of step t
        as the input for step t+1.

        Returns:
            point_preds: The sequence of medians.
            quantile_preds: The quantiles PREDICTED at each step (conditioned on the median path).
                            Note: This underestimates uncertainty compared to MQD.
        """
        model = self.model
        model.eval()
        device = self.device

        # 1. Setup
        B, V, T = context.shape
        context = context.to(device)
        # Move other inputs to device... (omitted for brevity, same as MQD)
        site_coords = site_coords.to(device)
        variate_ids = variate_ids.to(device)
        if channel_mask is not None: channel_mask = channel_mask.to(device)
        if has_coords is not None: has_coords = has_coords.to(device)

        patch_size = int(model.config.input_patch_stride)
        rounded_steps = int(math.ceil(prediction_length / patch_size) * patch_size)
        num_iters = rounded_steps // patch_size
        
        # We will collect the outputs here
        # Shape: List of [B, V, P, Q]
        collected_patches = []

        cur_context = context
        cur_kv_cache = kv_cache # If you implement KV cache later

        # 2. Autoregressive Loop
        for i in range(num_iters):
            output = model(
                context=cur_context,
                site_coords=site_coords,
                variate_ids=variate_ids,
                channel_mask=channel_mask,
                has_coords=has_coords,
                kv_cache=cur_kv_cache,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
            )
            
            # shape: (B, V, P, Q)
            # Ensure we use the SAME attribute as in MQD (e.g., denorm_quantile_preds)
            # assuming context is raw values.
            preds = output.denorm_quantile_preds[..., -patch_size:, :] 
            
            # Sort to ensure median index is valid
            preds, _ = torch.sort(preds, dim=-1)

            # Store the full distribution for this step
            collected_patches.append(preds)

            # --- GREEDY SELECTION ---
            # Pick the single deterministic value (Median) to feed back
            # Shape: (B, V, P)
            next_input_patch = preds[..., self.median_idx]

            # Update Context
            cur_context = torch.cat([cur_context, next_input_patch], dim=-1)

            # Sliding Window
            max_seq_len = self.cfg.data.context_length
            if cur_context.shape[-1] > max_seq_len:
                cur_context = cur_context[..., -max_seq_len:]
            
            # Handle KV Cache Update if applicable
            # cur_kv_cache = output.past_key_values

        # 3. Finalize Output
        # Concatenate all generated patches: (B, V, rounded_steps, Q)
        full_preds = torch.cat(collected_patches, dim=2)
        
        # Crop to prediction_length
        full_preds = full_preds[..., :prediction_length, :]

        # Extract Point Forecast (Median)
        point_forecast = full_preds[..., self.median_idx] # (B, V, pred_len)

        # Return both
        # Note: full_preds here represents the quantiles around the median path,
        # not the mixture of all possible paths.
        return point_forecast, full_preds
