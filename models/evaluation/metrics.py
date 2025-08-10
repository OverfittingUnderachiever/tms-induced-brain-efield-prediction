import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# --- Keep original vector functions for potential future use ---
def calculate_magnitude_error_vector(pred_vec: np.ndarray, target_vec: np.ndarray) -> Tuple[float, float, float]:
    """Calculate magnitude error metrics FROM VECTOR fields.
    Args:
        pred_vec: Predicted E-fields vectors, shape (batch_size, 3, D, H, W) or (3, D, H, W)
        target_vec: Target E-fields vectors, shape (batch_size, 3, D, H, W) or (3, D, H, W)
    Returns: Tuple[float, float, float]: MAE, RMSE, and relative error """
    if pred_vec.shape != target_vec.shape:
        raise ValueError(f"Vector shape mismatch: pred {pred_vec.shape} vs target {target_vec.shape}")
    if pred_vec.ndim == 5:
        pred_mag = np.sqrt(np.sum(pred_vec**2, axis=1))
        target_mag = np.sqrt(np.sum(target_vec**2, axis=1))
    elif pred_vec.ndim == 4:
        pred_mag = np.sqrt(np.sum(pred_vec**2, axis=0))
        target_mag = np.sqrt(np.sum(target_vec**2, axis=0))
    else:
        raise ValueError(f"Input vectors must be 4D or 5D, got {pred_vec.ndim}D")

    abs_error = np.abs(pred_mag - target_mag)
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(abs_error**2))
    rel_error = np.mean(abs_error / (target_mag + 1e-8)) * 100.0
    return float(mae), float(rmse), float(rel_error)

def calculate_angular_error(pred_vec: np.ndarray, target_vec: np.ndarray) -> Tuple[float, float, float]:
    """Calculate angular error metrics FROM VECTOR fields. """
    # (Keep original implementation as is)
    # Ensure pred and target have the same shape
    if pred_vec.shape != target_vec.shape:
        raise ValueError(f"Shape mismatch: pred {pred_vec.shape} vs target {target_vec.shape}")

    # Reshape to handle both 4D and 5D tensors
    if pred_vec.ndim == 5:  # (batch_size, 3, D, H, W)
        batch_size, channels, d, h, w = pred_vec.shape
        pred_reshaped = pred_vec.reshape(batch_size, channels, -1).transpose(0, 2, 1)  # (batch_size, D*H*W, 3)
        target_reshaped = target_vec.reshape(batch_size, channels, -1).transpose(0, 2, 1)  # (batch_size, D*H*W, 3)

        # Calculate magnitudes for each voxel
        pred_mag = np.sqrt(np.sum(pred_reshaped**2, axis=2, keepdims=True))  # (batch_size, D*H*W, 1)
        target_mag = np.sqrt(np.sum(target_reshaped**2, axis=2, keepdims=True))  # (batch_size, D*H*W, 1)

        # Normalize to unit vectors (avoid division by zero)
        pred_norm = pred_reshaped / (pred_mag + 1e-8)
        target_norm = target_reshaped / (target_mag + 1e-8)

        # Calculate dot product
        dot_product = np.sum(pred_norm * target_norm, axis=2)  # (batch_size, D*H*W)

        # Clip to valid range for arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate angular error in radians
        angular_error_rad = np.arccos(dot_product)

        # Convert to degrees
        angular_error_deg = np.degrees(angular_error_rad)

        # Flatten for statistics
        angular_error_flat = angular_error_deg.flatten()

    elif pred_vec.ndim == 4:  # (3, D, H, W)
        channels, d, h, w = pred_vec.shape
        pred_reshaped = pred_vec.reshape(channels, -1).transpose(1, 0)  # (D*H*W, 3)
        target_reshaped = target_vec.reshape(channels, -1).transpose(1, 0)  # (D*H*W, 3)

        # Calculate magnitudes for each voxel
        pred_mag = np.sqrt(np.sum(pred_reshaped**2, axis=1, keepdims=True))  # (D*H*W, 1)
        target_mag = np.sqrt(np.sum(target_reshaped**2, axis=1, keepdims=True))  # (D*H*W, 1)

        # Normalize to unit vectors (avoid division by zero)
        pred_norm = pred_reshaped / (pred_mag + 1e-8)
        target_norm = target_reshaped / (target_mag + 1e-8)

        # Calculate dot product
        dot_product = np.sum(pred_norm * target_norm, axis=1)  # (D*H*W,)

        # Clip to valid range for arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate angular error in radians
        angular_error_rad = np.arccos(dot_product)

        # Convert to degrees
        angular_error_deg = np.degrees(angular_error_rad)

        # Use as is for statistics
        angular_error_flat = angular_error_deg
    else:
         raise ValueError(f"Input vectors must be 4D or 5D, got {pred_vec.ndim}D")

    # Calculate statistics
    mean_error = np.mean(angular_error_flat)
    median_error = np.median(angular_error_flat)
    percentile_95 = np.percentile(angular_error_flat, 95)

    return float(mean_error), float(median_error), float(percentile_95)


def calculate_correlation_vector(pred_vec: np.ndarray, target_vec: np.ndarray) -> float:
    """Calculate correlation coefficient BETWEEN VECTOR fields (flattened)."""
    if pred_vec.shape != target_vec.shape:
        raise ValueError(f"Vector shape mismatch: pred {pred_vec.shape} vs target {target_vec.shape}")
    pred_flat = pred_vec.flatten()
    target_flat = target_vec.flatten()
    with np.errstate(invalid='ignore'): # Ignore warnings if std dev is zero
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


# --- New/Adapted functions for MAGNITUDE evaluation ---

def calculate_magnitude_metrics_scalar(pred_mag: np.ndarray, target_mag: np.ndarray) -> Tuple[float, float, float]:
    """Calculate magnitude error metrics DIRECTLY from scalar magnitude fields.

    Args:
        pred_mag: Predicted magnitude field, shape (batch_size, D, H, W) or (D, H, W)
        target_mag: Target magnitude field, shape (batch_size, D, H, W) or (D, H, W)

    Returns:
        Tuple[float, float, float]: MAE, RMSE, and relative error
    """
    if pred_mag.shape != target_mag.shape:
        # Attempt to fix if one has batch dim and other doesn't
        if pred_mag.ndim == target_mag.ndim + 1 and pred_mag.shape[0] == 1:
            pred_mag = pred_mag.squeeze(0)
        elif target_mag.ndim == pred_mag.ndim + 1 and target_mag.shape[0] == 1:
            target_mag = target_mag.squeeze(0)

        # Final check
        if pred_mag.shape != target_mag.shape:
            raise ValueError(f"Magnitude shape mismatch: pred {pred_mag.shape} vs target {target_mag.shape}")

    abs_error = np.abs(pred_mag - target_mag)
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(abs_error**2))

    # Calculate relative error, handle potential division by zero or near-zero target
    # Use mean of target magnitude in denominator for stability if individual values are zero
    target_mean = np.mean(target_mag)
    if target_mean > 1e-8:
         rel_error = np.mean(abs_error / (target_mag + 1e-8)) * 100.0 # Original approach
         # Alternative: rel_error = mae / target_mean * 100.0
    else:
         rel_error = np.inf # Or some indicator that relative error is not meaningful

    return float(mae), float(rmse), float(rel_error)


def calculate_correlation_scalar(
    pred_mag: np.ndarray,
    target_mag: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Calculate correlation coefficient DIRECTLY between scalar magnitude fields, applying an optional mask.

    Args:
        pred_mag: Predicted magnitude field, shape (batch_size, D, H, W) or (D, H, W)
        target_mag: Target magnitude field, shape (batch_size, D, H, W) or (D, H, W)
        mask: Optional boolean mask indicating valid voxels (True).

    Returns:
        float: Pearson correlation coefficient over the masked region.
    """
    # --- Shape validation and alignment ---
    if pred_mag.shape != target_mag.shape:
        if pred_mag.ndim == target_mag.ndim + 1 and pred_mag.shape[0] == 1: pred_mag = pred_mag.squeeze(0)
        elif target_mag.ndim == pred_mag.ndim + 1 and target_mag.shape[0] == 1: target_mag = target_mag.squeeze(0)
    if pred_mag.shape != target_mag.shape:
        raise ValueError(f"Magnitude shape mismatch: pred {pred_mag.shape} vs target {target_mag.shape}")

    # --- Mask Handling ---
    if mask is not None:
        if mask.shape != target_mag.shape:
            if mask.ndim == target_mag.ndim - 1 and target_mag.ndim > 3: mask = np.expand_dims(mask, axis=0) # Broadcast attempt
            if mask.shape != target_mag.shape: raise ValueError(f"Mask shape {mask.shape} incompatible with target shape {target_mag.shape}")
        pred_flat = pred_mag[mask].flatten()
        target_flat = target_mag[mask].flatten()
        if pred_flat.size < 2: # Need at least 2 points for correlation
             logger.warning(f"Correlation calculation skipped: not enough valid elements after masking (count={pred_flat.size}).")
             return 0.0
    else:
        pred_flat = pred_mag.flatten()
        target_flat = target_mag.flatten()
        if pred_flat.size < 2:
             logger.warning(f"Correlation calculation skipped: not enough elements in input arrays (count={pred_flat.size}).")
             return 0.0

    # --- Correlation Calculation ---
    if np.std(pred_flat) < 1e-8 or np.std(target_flat) < 1e-8:
        logger.debug("Correlation calculation skipped: zero standard deviation in masked predicted or target magnitude.")
        return 0.0

    with np.errstate(invalid='ignore'): # Ignore warnings for constant input
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]

    return float(corr) if not np.isnan(corr) else 0.0

def calculate_hotspot_iou_scalar(
    pred_mag: np.ndarray,
    target_mag: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold_percentile: float = 95.0
) -> float:
    """Calculate Intersection over Union (IoU) for hotspots DIRECTLY from scalar magnitude fields, applying an optional mask.

    Args:
        pred_mag: Predicted magnitude field, shape (batch_size, D, H, W) or (D, H, W)
        target_mag: Target magnitude field, shape (batch_size, D, H, W) or (D, H, W)
        mask: Optional boolean mask indicating valid voxels (True).
        threshold_percentile: Percentile threshold (0-100) to define hotspots within *masked*, non-zero values.

    Returns:
        float: Hotspot IoU score (0-1, higher is better) calculated over the masked region.
    """
    # --- Shape validation and alignment ---
    if pred_mag.shape != target_mag.shape:
        if pred_mag.ndim == target_mag.ndim + 1 and pred_mag.shape[0] == 1: pred_mag = pred_mag.squeeze(0)
        elif target_mag.ndim == pred_mag.ndim + 1 and target_mag.shape[0] == 1: target_mag = target_mag.squeeze(0)
    if pred_mag.shape != target_mag.shape:
        raise ValueError(f"Magnitude shape mismatch: pred {pred_mag.shape} vs target {target_mag.shape}")

    # --- Mask Handling ---
    if mask is not None:
        if mask.shape != target_mag.shape:
            if mask.ndim == target_mag.ndim - 1 and target_mag.ndim > 3: mask = np.expand_dims(mask, axis=0) # Broadcast attempt
            if mask.shape != target_mag.shape: raise ValueError(f"Mask shape {mask.shape} incompatible with target shape {target_mag.shape}")
        # Get masked values for percentile calculation
        pred_mag_masked_vals = pred_mag[mask & (pred_mag > 1e-8)]
        target_mag_masked_vals = target_mag[mask & (target_mag > 1e-8)]
        if pred_mag_masked_vals.size == 0 or target_mag_masked_vals.size == 0:
             logger.warning(f"Cannot calculate hotspot IoU: No valid non-zero elements after masking (pred count={pred_mag_masked_vals.size}, target count={target_mag_masked_vals.size}).")
             return 0.0
    else:
        # No mask, consider all non-zero values
        mask = np.ones_like(target_mag, dtype=bool) # Create dummy mask for consistency later
        pred_mag_masked_vals = pred_mag[pred_mag > 1e-8]
        target_mag_masked_vals = target_mag[target_mag > 1e-8]
        if pred_mag_masked_vals.size == 0 or target_mag_masked_vals.size == 0:
             logger.warning("Cannot calculate hotspot IoU: No non-zero values in prediction or target magnitude (no mask applied).")
             return 0.0

    # --- Hotspot Definition ---
    pred_threshold = np.percentile(pred_mag_masked_vals, threshold_percentile)
    target_threshold = np.percentile(target_mag_masked_vals, threshold_percentile)

    # Create boolean hotspot maps over the original grid shape
    pred_hotspots_bool = pred_mag > pred_threshold
    target_hotspots_bool = target_mag > target_threshold

    # --- Calculate IoU within the masked region ---
    # Intersection: True in pred hotspot, target hotspot, AND mask
    intersection = np.sum(np.logical_and.reduce([pred_hotspots_bool, target_hotspots_bool, mask]))
    # Union: True in (pred hotspot OR target hotspot) AND mask
    union = np.sum(np.logical_and(np.logical_or(pred_hotspots_bool, target_hotspots_bool), mask))

    if union == 0:
        # If the union *within the mask* is zero, check if intersection is also zero
        iou = 1.0 if intersection == 0 else 0.0 # Perfect match (of nothing within mask)
    else:
        iou = intersection / union

    return float(iou)

# --- Main Metrics Function for Magnitude Prediction ---

def calculate_magnitude_metrics_scalar(
    pred_mag: np.ndarray,
    target_mag: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """Calculate magnitude error metrics DIRECTLY from scalar magnitude fields, applying an optional mask.

    Args:
        pred_mag: Predicted magnitude field, shape (batch_size, D, H, W) or (D, H, W)
        target_mag: Target magnitude field, shape (batch_size, D, H, W) or (D, H, W)
        mask: Optional boolean mask of the same shape as target_mag, indicating valid voxels (True).

    Returns:
        Tuple[float, float, float]: MAE, RMSE, and relative error calculated over the masked region.
    """
    # --- Shape validation and alignment ---
    if pred_mag.shape != target_mag.shape:
        if pred_mag.ndim == target_mag.ndim + 1 and pred_mag.shape[0] == 1: pred_mag = pred_mag.squeeze(0)
        elif target_mag.ndim == pred_mag.ndim + 1 and target_mag.shape[0] == 1: target_mag = target_mag.squeeze(0)
    if pred_mag.shape != target_mag.shape:
        raise ValueError(f"Magnitude shape mismatch: pred {pred_mag.shape} vs target {target_mag.shape}")

    # --- Mask Handling ---
    if mask is not None:
        if mask.shape != target_mag.shape:
             # Try to broadcast mask if it lacks batch dim but data has it
             if mask.ndim == target_mag.ndim - 1 and target_mag.ndim > 3:
                  mask = np.expand_dims(mask, axis=0) # Add batch dim
                  if mask.shape != target_mag.shape: # Check again
                      raise ValueError(f"Mask shape {mask.shape} incompatible with target shape {target_mag.shape} even after broadcasting attempt.")
             else:
                  raise ValueError(f"Mask shape {mask.shape} incompatible with target shape {target_mag.shape}")
        # Apply mask
        pred_masked = pred_mag[mask]
        target_masked = target_mag[mask]
        if pred_masked.size == 0: # Check if mask is empty
             logger.warning("Mask is empty or leaves no valid elements for scalar metrics calculation.")
             return 0.0, 0.0, 0.0 # Or np.nan
    else:
        # No mask, use all elements
        pred_masked = pred_mag.flatten()
        target_masked = target_mag.flatten()
        if pred_masked.size == 0:
             logger.warning("Input arrays are empty for scalar metrics calculation.")
             return 0.0, 0.0, 0.0 # Or np.nan


    # --- Metric Calculations on Masked Data ---
    abs_error = np.abs(pred_masked - target_masked)
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(abs_error**2))

    # Relative error: calculated using masked target values
    target_masked_mean = np.mean(target_masked)
    if target_masked_mean > 1e-8:
        # Avoid division by zero for individual elements in the mean calculation
        rel_error = np.mean(abs_error / (target_masked + 1e-8)) * 100.0
    else:
        rel_error = np.inf

    return float(mae), float(rmse), float(rel_error)


# --- Original Main Metrics Function (for Vector Fields - kept for reference) ---

def calculate_vector_metrics(pred_vec: np.ndarray, target_vec: np.ndarray) -> Dict[str, float]:
    """Calculate all evaluation metrics for VECTOR field prediction.

    Args:
        pred_vec: Predicted E-fields vectors, shape (batch_size, 3, D, H, W) or (3, D, H, W)
        target_vec: Target E-fields vectors, shape (batch_size, 3, D, H, W) or (3, D, H, W)

    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    # Ensure inputs are numpy arrays
    pred_vec = np.asarray(pred_vec)
    target_vec = np.asarray(target_vec)

    # Calculate magnitude metrics
    mae, rmse, rel_error = calculate_magnitude_error_vector(pred_vec, target_vec)

    # Calculate angular metrics
    mean_angular, median_angular, p95_angular = calculate_angular_error(pred_vec, target_vec)

    # Calculate correlation (on flattened vectors)
    correlation = calculate_correlation_vector(pred_vec, target_vec)

    # Calculate hotspot IoU (using original magnitude calculation inside)
    # Note: This internal calculation could be replaced by calculate_hotspot_iou_scalar if needed
    # For now, keep the original logic tied to vector inputs.
    target_mag_for_hotspot = np.sqrt(np.sum(target_vec**2, axis=1 if target_vec.ndim == 5 else 0))
    pred_mag_for_hotspot = np.sqrt(np.sum(pred_vec**2, axis=1 if pred_vec.ndim == 5 else 0))
    hotspot_iou = calculate_hotspot_iou_scalar(pred_mag_for_hotspot, target_mag_for_hotspot, threshold_percentile=95.0)


    # Combine into metrics dictionary
    metrics = {
        'magnitude_mae': mae,
        'magnitude_rmse': rmse,
        'magnitude_rel_error': rel_error,
        'mean_angular_error': mean_angular,
        'median_angular_error': median_angular,
        'p95_angular_error': p95_angular,
        'vector_correlation': correlation, # Renamed to avoid clash
        'hotspot_iou': hotspot_iou,
    }

    if np.isinf(metrics['magnitude_rel_error']):
        logger.warning("Relative error is infinite (likely zero mean target magnitude). Setting to -1.0.")
        metrics['magnitude_rel_error'] = -1.0

    return metrics

# --- Deprecated/Modified original functions ---
# Rename original calculate_metrics to avoid conflicts if needed later
# We now primarily use calculate_magnitude_metrics or calculate_vector_metrics

# Keep the original function signature but call the vector version
# This maintains backward compatibility if anything external calls it directly
# But the trainer should now call calculate_magnitude_metrics
def calculate_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
     """
     Original calculate_metrics function. Now defaults to calling
     calculate_vector_metrics for backward compatibility.
     For magnitude-only models, use calculate_magnitude_metrics.

     Args:
         pred: Predicted E-fields, shape (batch_size, C, D, H, W) or (C, D, H, W). C=3 for vector.
         target: Target E-fields, shape (batch_size, C, D, H, W) or (C, D, H, W). C=3 for vector.

     Returns:
         Dict[str, float]: Dictionary of metrics.
     """
     logger.warning("Calling deprecated calculate_metrics. Assuming vector input. "
                    "Use calculate_magnitude_metrics for magnitude models.")
     # Check if input looks like magnitude (C=1) or vector (C=3)
     pred_channels = pred.shape[1] if pred.ndim > 3 else pred.shape[0]
     if pred_channels == 1:
          logger.error("Input appears to be magnitude, but calculate_metrics (vector version) was called.")
          # Decide how to handle: raise error, or call magnitude version?
          # Let's call magnitude version with a warning
          logger.warning("Redirecting to calculate_magnitude_metrics based on input shape.")
          # Need to calculate target magnitude first if target is vector
          target_channels = target.shape[1] if target.ndim > 3 else target.shape[0]
          if target_channels == 3:
                target_mag = np.sqrt(np.sum(np.asarray(target)**2, axis=1 if target.ndim==5 else 0, keepdims=True))
                # Squeeze channel dims before passing
                pred_squeezed = np.asarray(pred).squeeze(axis=1 if pred.ndim==5 else 0)
                target_mag_squeezed = target_mag.squeeze(axis=1 if target_mag.ndim==5 else 0)
                return calculate_magnitude_metrics(pred_squeezed, target_mag_squeezed)
          else:
                # Both seem to be magnitude already
                pred_squeezed = np.asarray(pred).squeeze(axis=1 if pred.ndim==5 else 0)
                target_squeezed = np.asarray(target).squeeze(axis=1 if target.ndim==5 else 0)
                return calculate_magnitude_metrics(pred_squeezed, target_squeezed)

     elif pred_channels == 3:
        # Assume vector input
        return calculate_vector_metrics(pred, target)
     else:
          raise ValueError(f"Cannot determine if input is vector (C=3) or magnitude (C=1). Shape: {pred.shape}")
     
def calculate_magnitude_metrics(
    pred_mag: np.ndarray,
    target_mag: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate all relevant evaluation metrics for MAGNITUDE prediction, applying an optional mask.

    Args:
        pred_mag: Predicted magnitude field, shape (batch_size, D, H, W) or (D, H, W)
        target_mag: Target magnitude field, shape (batch_size, D, H, W) or (D, H, W)
        mask: Optional boolean mask indicating valid voxels (True).

    Returns:
        Dict[str, float]: Dictionary of metrics (MAE, RMSE, Relative Error, Correlation, Hotspot IoU) calculated over the masked region.
    """
    # Ensure inputs are numpy arrays
    pred_mag = np.asarray(pred_mag)
    target_mag = np.asarray(target_mag)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

    # Calculate basic magnitude errors (passing mask)
    mae, rmse, rel_error = calculate_magnitude_metrics_scalar(pred_mag, target_mag, mask=mask)

    # Calculate correlation (passing mask)
    correlation = calculate_correlation_scalar(pred_mag, target_mag, mask=mask)

    # Calculate hotspot IoU (using a default percentile, passing mask)
    hotspot_iou = calculate_hotspot_iou_scalar(pred_mag, target_mag, mask=mask, threshold_percentile=95.0)

    # Combine into metrics dictionary
    metrics = {
        'magnitude_mae': mae,
        'magnitude_rmse': rmse,
        'magnitude_rel_error': rel_error,
        'magnitude_correlation': correlation,
        'hotspot_iou': hotspot_iou,
    }

    # Filter out potential inf values from rel_error if necessary
    if np.isinf(metrics['magnitude_rel_error']):
        logger.warning("Relative error is infinite (likely zero mean target magnitude in masked region). Setting to -1.0.")
        metrics['magnitude_rel_error'] = -1.0

    return metrics