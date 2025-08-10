
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.gridspec as gridspec
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Already configured in train_evaluate_test
logger = logging.getLogger('tms_magnitude_visualization')


# --- Modified for MAGNITUDE ---
def visualize_prediction_vs_ground_truth(
    pred_mag: np.ndarray, # Predicted magnitude scalar field
    target_mag: np.ndarray, # Target magnitude scalar field
    slice_idx: Optional[Tuple[int, int, int]] = None,
    mask: Optional[np.ndarray] = None, # Mask should be spatial [D, H, W]
    save_path: Optional[str] = None,
    display_error: bool = True,
    colormap: str = 'viridis',
    error_colormap: str = 'hot',
    title_prefix: str = "" # Optional prefix for title
):
    """
    Visualize predicted magnitude vs. ground truth magnitude for a single sample.

    Args:
        pred_mag: Predicted magnitude field, shape (D, H, W) or (B, D, H, W)
        target_mag: Ground truth magnitude field, shape (D, H, W) or (B, D, H, W)
        slice_idx: Optional indices for slices to show (x, y, z) - referring to D, H, W axes
        mask: Optional binary spatial mask, shape (D, H, W)
        save_path: Optional path to save the visualization
        display_error: Whether to display error maps
        colormap: Colormap for magnitude visualization
        error_colormap: Colormap for error visualization
        title_prefix: Optional prefix for the plot title.

    Returns:
        matplotlib.figure.Figure: The generated figure (or None if error).
    """
    fig = None # Initialize fig to None
    try:
        # Handle batch dimension if present (select first sample)
        if pred_mag.ndim == 4 and pred_mag.shape[0] >= 1:
            pred_mag = pred_mag[0]
        if target_mag.ndim == 4 and target_mag.shape[0] >= 1:
            target_mag = target_mag[0]

        # Ensure we're working with 3D tensors (D, H, W)
        if pred_mag.ndim != 3 or target_mag.ndim != 3:
            raise ValueError(f"Expected 3D magnitude fields (D, H, W), got pred: {pred_mag.shape}, target: {target_mag.shape}")
        if pred_mag.shape != target_mag.shape:
             # Try to interpolate prediction to match target if shapes differ slightly
             if np.prod(pred_mag.shape) > 0 and np.prod(target_mag.shape) > 0:
                 logger.warning(f"Shape mismatch between pred_mag {pred_mag.shape} and target_mag {target_mag.shape}. "
                                f"Attempting to interpolate prediction.")
                 import torch # Use torch for interpolation convenience
                 pred_mag_t = torch.from_numpy(pred_mag).unsqueeze(0).unsqueeze(0).float() # Add B, C dims
                 pred_mag_interp = torch.nn.functional.interpolate(pred_mag_t, size=target_mag.shape, mode='trilinear', align_corners=False)
                 pred_mag = pred_mag_interp.squeeze(0).squeeze(0).numpy()
                 if pred_mag.shape != target_mag.shape: # Check again after interpolation
                     raise ValueError(f"Shape mismatch persists after interpolation: pred {pred_mag.shape}, target {target_mag.shape}")
             else:
                 raise ValueError(f"Shape mismatch: pred {pred_mag.shape} vs target {target_mag.shape}")


        # Calculate absolute error magnitude
        error_mag = np.abs(pred_mag - target_mag)

        # Apply mask if provided
        if mask is not None:
            if mask.shape != pred_mag.shape:
                raise ValueError(f"Mask shape {mask.shape} must match magnitude shape {pred_mag.shape}")
            pred_mag = pred_mag * mask
            target_mag = target_mag * mask
            error_mag = error_mag * mask

        # Set default slice indices if not provided (center slices)
        if slice_idx is None:
            d_idx = pred_mag.shape[0] // 2 # Index for Depth (Sagittal view needs Y,Z)
            h_idx = pred_mag.shape[1] // 2 # Index for Height (Coronal view needs X,Z)
            w_idx = pred_mag.shape[2] // 2 # Index for Width (Axial view needs X,Y)
            # Note: Plotting axes conventions can be tricky. Let's stick to D, H, W indexing.
            # Sagittal shows H vs W at index D = d_idx
            # Coronal shows D vs W at index H = h_idx
            # Axial shows D vs H at index W = w_idx
            slice_idx = (d_idx, h_idx, w_idx)
        else:
            d_idx, h_idx, w_idx = slice_idx

        # Create figure
        rows, cols = 3, 3 if display_error else 2
        fig = plt.figure(figsize=(cols * 5, rows * 4)) # Adjusted size
        gs = gridspec.GridSpec(rows, cols, figure=fig)

        # Common color normalization for magnitude (using robust min/max)
        vmin_robust = np.percentile(target_mag[target_mag > 1e-8], 1) if np.any(target_mag > 1e-8) else 0
        vmax_robust = np.percentile(target_mag[target_mag > 1e-8], 99) if np.any(target_mag > 1e-8) else 1
        if np.any(pred_mag > 1e-8):
             vmin_robust = min(vmin_robust, np.percentile(pred_mag[pred_mag > 1e-8], 1))
             vmax_robust = max(vmax_robust, np.percentile(pred_mag[pred_mag > 1e-8], 99))

        # Ensure vmin < vmax
        if vmin_robust >= vmax_robust:
             vmin_robust = 0
             vmax_robust = max(np.max(target_mag), np.max(pred_mag))
             if vmax_robust < 1e-8 : vmax_robust = 1.0 # Handle all zero case


        norm = Normalize(vmin=vmin_robust, vmax=vmax_robust)

        # Error normalization (robust max)
        error_vmax_robust = np.percentile(error_mag[error_mag > 1e-8], 99) if np.any(error_mag > 1e-8) else 1
        if error_vmax_robust < 1e-8: error_vmax_robust = 1.0 # Handle all zero error
        error_norm = Normalize(vmin=0, vmax=error_vmax_robust)

        # --- Plotting Slices ---
        # Sagittal view (plot H vs W plane at specified D index)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(np.rot90(pred_mag[d_idx, :, :]), cmap=colormap, norm=norm)
        ax1.set_title(f'Predicted - Sagittal (D={d_idx})')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(np.rot90(target_mag[d_idx, :, :]), cmap=colormap, norm=norm)
        ax2.set_title(f'Ground Truth - Sagittal (D={d_idx})')
        ax2.axis('off')

        if display_error:
            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(np.rot90(error_mag[d_idx, :, :]), cmap=error_colormap, norm=error_norm)
            ax3.set_title(f'Error - Sagittal (D={d_idx})')
            ax3.axis('off')

        # Coronal view (plot D vs W plane at specified H index)
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(np.rot90(pred_mag[:, h_idx, :]), cmap=colormap, norm=norm)
        ax4.set_title(f'Predicted - Coronal (H={h_idx})')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(np.rot90(target_mag[:, h_idx, :]), cmap=colormap, norm=norm)
        ax5.set_title(f'Ground Truth - Coronal (H={h_idx})')
        ax5.axis('off')

        if display_error:
            ax6 = fig.add_subplot(gs[1, 2])
            im6 = ax6.imshow(np.rot90(error_mag[:, h_idx, :]), cmap=error_colormap, norm=error_norm)
            ax6.set_title(f'Error - Coronal (H={h_idx})')
            ax6.axis('off')

        # Axial view (plot D vs H plane at specified W index)
        ax7 = fig.add_subplot(gs[2, 0])
        im7 = ax7.imshow(np.rot90(pred_mag[:, :, w_idx]), cmap=colormap, norm=norm)
        ax7.set_title(f'Predicted - Axial (W={w_idx})')
        ax7.axis('off')

        ax8 = fig.add_subplot(gs[2, 1])
        im8 = ax8.imshow(np.rot90(target_mag[:, :, w_idx]), cmap=colormap, norm=norm)
        ax8.set_title(f'Ground Truth - Axial (W={w_idx})')
        ax8.axis('off')

        if display_error:
            ax9 = fig.add_subplot(gs[2, 2])
            im9 = ax9.imshow(np.rot90(error_mag[:, :, w_idx]), cmap=error_colormap, norm=error_norm)
            ax9.set_title(f'Error - Axial (W={w_idx})')
            ax9.axis('off')

        # Add colorbar for magnitude
        cbar_ax_mag = fig.add_axes([0.92, 0.35, 0.02, 0.5]) # Position adjusted
        cbar_mag = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cbar_ax_mag)
        cbar_mag.set_label('E-field Magnitude')

        # Add error colorbar if displaying error
        if display_error:
            cbar_ax_err = fig.add_axes([0.92, 0.1, 0.02, 0.2]) # Position adjusted
            cbar_err = plt.colorbar(cm.ScalarMappable(norm=error_norm, cmap=error_colormap), cax=cbar_ax_err)
            cbar_err.set_label('Magnitude Error')

        fig.suptitle(f'{title_prefix}E-field Magnitude: Prediction vs. Ground Truth'.strip(), fontsize=16)
        fig.tight_layout(rect=[0, 0, 0.9, 0.95]) # Adjust rectangle to prevent overlap with colorbars

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved magnitude visualization to {save_path}")

    except Exception as e:
        logger.error(f"Failed during magnitude visualization: {e}", exc_info=True)
        if fig:
             plt.close(fig) # Close figure if it was created but failed later
        return None # Return None on error

    return fig


# --- Modified for MAGNITUDE ---
def visualize_hotspots(
    pred_mag: np.ndarray, # Predicted magnitude scalar field
    target_mag: np.ndarray, # Target magnitude scalar field
    mask: Optional[np.ndarray] = None, # Mask should be spatial [D, H, W]
    threshold_percentile: float = 95.0,
    slice_idx: Optional[Tuple[int, int, int]] = None,
    save_path: Optional[str] = None,
    title_prefix: str = "" # Optional prefix for title
):
    """
    Visualize hotspots (high-intensity regions) in predicted vs. ground truth magnitude fields.

    Args:
        pred_mag: Predicted magnitude field, shape (D, H, W) or (B, D, H, W)
        target_mag: Ground truth magnitude field, shape (D, H, W) or (B, D, H, W)
        mask: Optional binary spatial mask, shape (D, H, W)
        threshold_percentile: Percentile threshold (0-100) for defining hotspots within non-zero values.
        slice_idx: Optional indices for slices to show (d, h, w)
        save_path: Optional path to save the visualization
        title_prefix: Optional prefix for the plot title.

    Returns:
        Tuple[matplotlib.figure.Figure, float]: The generated figure and the calculated IoU score (or None, 0.0 if error).
    """
    fig = None
    iou = 0.0
    try:
        # Handle batch dimension if present
        if pred_mag.ndim == 4 and pred_mag.shape[0] >= 1:
            pred_mag = pred_mag[0]
        if target_mag.ndim == 4 and target_mag.shape[0] >= 1:
            target_mag = target_mag[0]

        # Ensure we're working with 3D tensors (D, H, W)
        if pred_mag.ndim != 3 or target_mag.ndim != 3:
             raise ValueError(f"Expected 3D magnitude fields (D, H, W), got pred: {pred_mag.shape}, target: {target_mag.shape}")
        if pred_mag.shape != target_mag.shape:
             # Try interpolation as fallback
             if np.prod(pred_mag.shape) > 0 and np.prod(target_mag.shape) > 0:
                 logger.warning(f"Shape mismatch for hotspots {pred_mag.shape} vs {target_mag.shape}. Interpolating prediction.")
                 import torch
                 pred_mag_t = torch.from_numpy(pred_mag).unsqueeze(0).unsqueeze(0).float()
                 pred_mag_interp = torch.nn.functional.interpolate(pred_mag_t, size=target_mag.shape, mode='trilinear', align_corners=False)
                 pred_mag = pred_mag_interp.squeeze(0).squeeze(0).numpy()
                 if pred_mag.shape != target_mag.shape:
                      raise ValueError(f"Shape mismatch persists after interpolation: pred {pred_mag.shape}, target {target_mag.shape}")
             else:
                 raise ValueError(f"Shape mismatch: pred {pred_mag.shape} vs target {target_mag.shape}")


        # Apply mask if provided
        if mask is not None:
            if mask.shape != pred_mag.shape:
                raise ValueError(f"Mask shape {mask.shape} must match magnitude shape {pred_mag.shape}")
            pred_mag_masked = pred_mag * mask
            target_mag_masked = target_mag * mask
        else:
            pred_mag_masked = pred_mag
            target_mag_masked = target_mag

        # Define hotspots based on percentile threshold of *masked* non-zero values
        pred_mag_nonzero = pred_mag_masked[pred_mag_masked > 1e-8]
        target_mag_nonzero = target_mag_masked[target_mag_masked > 1e-8]

        if len(pred_mag_nonzero) == 0 or len(target_mag_nonzero) == 0:
            logger.warning("Cannot calculate hotspots: No non-zero values in masked prediction or target magnitude.")
            # Still create plot showing potentially empty hotspots
            pred_threshold = np.inf
            target_threshold = np.inf
            iou = 0.0 # No overlap possible
        else:
            pred_threshold = np.percentile(pred_mag_nonzero, threshold_percentile)
            target_threshold = np.percentile(target_mag_nonzero, threshold_percentile)
            iou = calculate_hotspot_iou_scalar(pred_mag_masked, target_mag_masked, threshold_percentile) # Use metric func

        pred_hotspots = pred_mag_masked > pred_threshold
        target_hotspots = target_mag_masked > target_threshold

        # Calculate overlap (intersection) visualization
        intersection = np.logical_and(pred_hotspots, target_hotspots)

        # Set default slice indices if not provided
        if slice_idx is None:
            d_idx = pred_mag.shape[0] // 2
            h_idx = pred_mag.shape[1] // 2
            w_idx = pred_mag.shape[2] // 2
            slice_idx = (d_idx, h_idx, w_idx)
        else:
            d_idx, h_idx, w_idx = slice_idx

        # Create figure
        fig, axs = plt.subplots(3, 3, figsize=(15, 12)) # Rows: Sag, Cor, Axi; Cols: Pred, Targ, Overlap

        # Sagittal views (x -> D index)
        axs[0, 0].imshow(np.rot90(pred_hotspots[d_idx, :, :]), cmap='Blues', vmin=0, vmax=1)
        axs[0, 0].set_title(f'Predicted Hotspots - Sag (D={d_idx})')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(np.rot90(target_hotspots[d_idx, :, :]), cmap='Reds', vmin=0, vmax=1)
        axs[0, 1].set_title(f'Ground Truth Hotspots - Sag (D={d_idx})')
        axs[0, 1].axis('off')

        axs[0, 2].imshow(np.rot90(intersection[d_idx, :, :]), cmap='Greens', vmin=0, vmax=1)
        axs[0, 2].set_title(f'Overlap (Intersection) - Sag (D={d_idx})')
        axs[0, 2].axis('off')

        # Coronal views (y -> H index)
        axs[1, 0].imshow(np.rot90(pred_hotspots[:, h_idx, :]), cmap='Blues', vmin=0, vmax=1)
        axs[1, 0].set_title(f'Predicted Hotspots - Cor (H={h_idx})')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(np.rot90(target_hotspots[:, h_idx, :]), cmap='Reds', vmin=0, vmax=1)
        axs[1, 1].set_title(f'Ground Truth Hotspots - Cor (H={h_idx})')
        axs[1, 1].axis('off')

        axs[1, 2].imshow(np.rot90(intersection[:, h_idx, :]), cmap='Greens', vmin=0, vmax=1)
        axs[1, 2].set_title(f'Overlap - Cor (H={h_idx})')
        axs[1, 2].axis('off')

        # Axial views (z -> W index)
        axs[2, 0].imshow(np.rot90(pred_hotspots[:, :, w_idx]), cmap='Blues', vmin=0, vmax=1)
        axs[2, 0].set_title(f'Predicted Hotspots - Axi (W={w_idx})')
        axs[2, 0].axis('off')

        axs[2, 1].imshow(np.rot90(target_hotspots[:, :, w_idx]), cmap='Reds', vmin=0, vmax=1)
        axs[2, 1].set_title(f'Ground Truth Hotspots - Axi (W={w_idx})')
        axs[2, 1].axis('off')

        axs[2, 2].imshow(np.rot90(intersection[:, :, w_idx]), cmap='Greens', vmin=0, vmax=1)
        axs[2, 2].set_title(f'Overlap - Axi (W={w_idx})')
        axs[2, 2].axis('off')

        # Add title with IoU
        fig.suptitle(f'{title_prefix}Hotspot Analysis ({threshold_percentile} percentile, IoU = {iou:.4f})'.strip(), fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved hotspot visualization to {save_path}")

    except Exception as e:
        logger.error(f"Failed during hotspot visualization: {e}", exc_info=True)
        if fig:
            plt.close(fig)
        return None, 0.0 # Return None, 0.0 on error

    return fig, iou
def plot_train_val_loss(
    train_losses: List[float],
    val_losses: List[Optional[float]],
    validation_frequency: int = 1,
    save_path: Optional[str] = None,
    title: str = "Training and Validation Loss"
) -> plt.Figure:
    """
    Create a plot of training and validation loss curves.
    
    Args:
        train_losses: List of training loss values
        val_losses: List of validation loss values (can contain None)
        validation_frequency: Frequency of validation (epochs)
        save_path: Path to save the figure
        title: Plot title
        
    Returns:
        matplotlib.pyplot.Figure: The generated figure
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Filter out None values from validation loss
    valid_val_losses = [loss for loss in val_losses if loss is not None]
    
    # Generate x-axis values
    epochs_train = np.arange(1, len(train_losses) + 1)
    epochs_val = np.arange(1, len(val_losses) + 1) * validation_frequency
    
    # Plot losses
    if train_losses:
        plt.plot(epochs_train, train_losses, 'bo-', label='Training Loss')
    if valid_val_losses:
        plt.plot(epochs_val[:len(valid_val_losses)], valid_val_losses, 'ro-', label='Validation Loss')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if train_losses or valid_val_losses:
        plt.legend()
    plt.grid(True)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved loss plot to {save_path}")
    
    return fig

def extract_history_metrics(
    history: Dict[str, List[Dict[str, float]]],
    metrics_of_interest: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract metrics from training history into a format suitable for visualization.
    
    Args:
        history: Training history dictionary from ModelTrainer
        metrics_of_interest: List of metric names to extract
        
    Returns:
        Dict with train and val metrics
    """
    full_metrics_history = {"train": {}, "val": {}}
    
    # Process training metrics
    if history.get("train"):
        for epoch_metrics in history["train"]:
            for key, value in epoch_metrics.items():
                if metrics_of_interest and key not in metrics_of_interest:
                    continue
                if key not in full_metrics_history["train"]:
                    full_metrics_history["train"][key] = []
                if value is not None:
                    full_metrics_history["train"][key].append(value)
    
    # Process validation metrics
    if history.get("val"):
        for epoch_metrics in history["val"]:
            for key, value in epoch_metrics.items():
                if metrics_of_interest and key not in metrics_of_interest:
                    continue
                if key not in full_metrics_history["val"]:
                    full_metrics_history["val"][key] = []
                full_metrics_history["val"][key].append(value)
    
    return full_metrics_history
# --- Kept mostly as is, but ensure metrics_history keys match new metrics ---
def create_metrics_summary_plot(metrics_history: Dict[str, List[float]], save_path=None, title='Training Metrics Summary', max_metrics=12):
    """
    Create a summary plot of training/validation metrics. (Assumes keys match magnitude metrics).

    Args:
        metrics_history: Dictionary with metric names as keys and lists of values as values
                         (e.g., {'loss': [...], 'magnitude_mae': [...], 'magnitude_correlation': [...]})
        save_path: Optional path to save the visualization
        title: Plot title
        max_metrics: Maximum number of metrics to plot in the grid.

    Returns:
        matplotlib.figure.Figure: The generated figure (or None if no metrics).
    """
    fig = None
    try:
        if not isinstance(metrics_history, dict) or not metrics_history:
             logger.warning("No metrics history provided for plotting.")
             return None

        # Filter out any metrics with empty lists or None values
        valid_metrics_history = {k: v for k, v in metrics_history.items() if v and all(val is not None for val in v)}

        if not valid_metrics_history:
            logger.warning("No valid metric values found in history for plotting.")
            return None

        if len(valid_metrics_history) > max_metrics:
            logger.warning(f"Too many metrics ({len(valid_metrics_history)}). Plotting only the first {max_metrics}.")
            metrics_to_plot = dict(list(valid_metrics_history.items())[:max_metrics])
        else:
            metrics_to_plot = valid_metrics_history

        n_metrics = len(metrics_to_plot)
        if n_metrics == 0:
            logger.warning("No metrics left after filtering for plotting.")
            return None

        # Determine number of rows and columns
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        # Create figure
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False) # squeeze=False ensures axs is always 2D array
        axs = axs.flatten() # Flatten for easy indexing

        # Plot each metric
        for i, (metric_name, values) in enumerate(metrics_to_plot.items()):
            ax = axs[i]
            epochs = np.arange(1, len(values) + 1)
            ax.plot(epochs, values, 'o-', label=metric_name)
            ax.set_title(f'{metric_name} vs. Epoch')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.replace('_', ' ').title()) # Nicer label
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add best value annotation (handle potential empty values just in case)
            if values:
                 is_loss = 'loss' in metric_name.lower() or 'error' in metric_name.lower() or 'mae' in metric_name.lower() or 'rmse' in metric_name.lower()
                 best_epoch_idx = np.argmin(values) if is_loss else np.argmax(values)
                 best_value = values[best_epoch_idx]
                 best_epoch = epochs[best_epoch_idx]
                 ax.annotate(f'Best: {best_value:.4f}\n(Epoch {best_epoch})',
                             xy=(best_epoch, best_value),
                             xytext=(best_epoch, best_value + 0.1 * abs(best_value) if best_value != 0 else 0.1), # Adjust text position slightly
                             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
                             fontsize=9, ha='center')

        # Hide unused subplots
        for i in range(n_metrics, len(axs)):
            axs[i].axis('off')

        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics summary plot to {save_path}")

    except Exception as e:
        logger.error(f"Failed during metrics summary plot generation: {e}", exc_info=True)
        if fig:
            plt.close(fig)
        return None

    return fig

# --- Kept as is, but ensure results keys match new metrics ---
def visualize_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    title: str = 'Model Comparison'
):
    """
    Create a comparison plot for multiple models and metrics. (Assumes keys match magnitude metrics).

    Args:
        results: Dictionary {model_name: {metric_name: value, ...}, ...}
        metrics: List of metrics to include (e.g., ['magnitude_mae', 'magnitude_correlation', 'hotspot_iou'])
                 If None, uses all metrics found in the first model's results.
        save_path: Optional path to save the visualization
        title: Plot title

    Returns:
        matplotlib.figure.Figure: The generated figure (or None if error).
    """
    fig = None
    try:
        if not isinstance(results, dict) or not results:
            logger.warning("No results provided for model comparison.")
            return None

        model_names = list(results.keys())
        if not model_names:
            logger.warning("No model names found in results.")
            return None

        # Determine metrics to plot
        if metrics is None:
            # Use metrics from the first model, filtering out any None values
            metrics = [k for k, v in results[model_names[0]].items() if v is not None]
            if not metrics:
                 # Fallback: check other models if first one has no valid metrics
                 for model in model_names[1:]:
                      metrics = [k for k, v in results[model].items() if v is not None]
                      if metrics: break

        if not metrics:
            logger.warning("No valid metrics found to plot for comparison.")
            return None

        # Filter results to only include the desired metrics and handle missing ones
        plot_data = {metric: [] for metric in metrics}
        valid_model_names = []
        for model in model_names:
             model_metrics = results.get(model, {})
             # Check if model has at least one valid metric we want to plot
             if any(metric in model_metrics and model_metrics[metric] is not None for metric in metrics):
                  valid_model_names.append(model)
                  for metric in metrics:
                       plot_data[metric].append(model_metrics.get(metric, np.nan)) # Use NaN for missing values

        if not valid_model_names:
             logger.warning("No models have valid values for the selected metrics.")
             return None

        model_names = valid_model_names # Update model names list
        n_models = len(model_names)
        n_metrics = len(metrics)

        # Create figure
        fig, axs = plt.subplots(n_metrics, 1, figsize=(max(8, n_models * 1.5), 4 * n_metrics), squeeze=False)
        axs = axs.flatten()

        # Colors for bars
        colors = plt.cm.get_cmap('tab10', max(10, n_models)) # Use a colormap

        bar_width = 0.6

        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axs[i]
            values = np.array(plot_data[metric]) # Contains NaNs for missing values

            # Create bars, handling NaNs
            valid_indices = np.arange(n_models)
            bars = ax.bar(valid_indices, np.nan_to_num(values), bar_width, color=[colors(j % 10) for j in range(n_models)]) # Use nan_to_num for plotting

            # Add values on top of bars only for non-NaNs
            for j, bar in enumerate(bars):
                if not np.isnan(values[j]): # Only add text if value is not NaN
                    height = bar.get_height()
                    # Adjust text position slightly above bar
                    text_y = height + 0.02 * np.nanmax(np.abs(values)) if height >= 0 else height - 0.05 * np.nanmax(np.abs(values))
                    if np.isnan(text_y): text_y = 0 # Fallback if max is NaN
                    ax.text(bar.get_x() + bar.get_width() / 2., text_y,
                            f'{values[j]:.4f}', ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=9)
                else:
                     # Optionally indicate missing data
                     ax.text(bar.get_x() + bar.get_width() / 2., 0, 'N/A', ha='center', va='center', fontsize=8, color='gray')


            # Set title and labels
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xticks(valid_indices)
            ax.set_xticklabels(model_names, rotation=30, ha='right') # Rotate labels for readability
            ax.grid(True, axis='y', linestyle='--', alpha=0.5) # Grid only on y-axis

            # Adjust y-limits for better visualization
            if not np.all(np.isnan(values)):
                 min_val = np.nanmin(values)
                 max_val = np.nanmax(values)
                 padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 1e-6 else 0.1
                 ax.set_ylim(min_val - padding, max_val + padding * 1.5) # Extra space for text at top


        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust bottom margin for rotated labels

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison visualization to {save_path}")

    except Exception as e:
        logger.error(f"Failed during model comparison visualization: {e}", exc_info=True)
        if fig:
            plt.close(fig)
        return None

    return fig


# --- Modified for MAGNITUDE ---
def generate_standard_visualizations(
    model_name: str,
    pred_mag: np.ndarray, # Expects predicted magnitude (B, 1, D, H, W) or (1, D, H, W) or (D, H, W)
    target_mag: np.ndarray, # Expects target magnitude (B, 1, D, H, W) or (1, D, H, W) or (D, H, W)
    mask: Optional[np.ndarray] = None, # Expects spatial mask (D, H, W)
    metrics_dict: Optional[Dict[str, float]] = None, # Use the calculated magnitude metrics
    output_dir: str = 'visualization_output'
):
    """
    Generate a standard set of visualizations for MAGNITUDE model evaluation.

    Args:
        model_name: Name of the model.
        pred_mag: Predicted magnitude field.
        target_mag: Ground truth magnitude field.
        mask: Optional binary spatial mask.
        metrics_dict: Optional dictionary of calculated magnitude metric values.
        output_dir: Directory to save visualizations.

    Returns:
        Dict[str, str]: Dictionary mapping visualization names to file paths.
    """
    visualization_paths = {}
    fig_mag, fig_hotspot, fig_metrics = None, None, None # Initialize figure vars

    try:
        # Create output directory
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        logger.debug(f"[GSV_Mag] Input pred_mag type: {type(pred_mag)}, shape: {getattr(pred_mag, 'shape', 'N/A')}")
        logger.debug(f"[GSV_Mag] Input target_mag type: {type(target_mag)}, shape: {getattr(target_mag, 'shape', 'N/A')}")

        # --- Ensure inputs are 3D spatial maps (D, H, W) ---
        if pred_mag.ndim > 3:
            if pred_mag.shape[0] == 1: # Batch dim
                 pred_mag = pred_mag.squeeze(axis=0)
            else: # More than 1 sample in batch, use first one
                 logger.warning(f"generate_standard_visualizations received batch size > 1 for pred_mag ({pred_mag.shape}). Using first sample.")
                 pred_mag = pred_mag[0]
        if pred_mag.ndim == 3 and pred_mag.shape[0] == 1 : # Channel dim remaining?
             pred_mag = pred_mag.squeeze(axis=0) # Now should be [D, H, W]

        if target_mag.ndim > 3:
             if target_mag.shape[0] == 1: # Batch dim
                 target_mag = target_mag.squeeze(axis=0)
             else:
                 logger.warning(f"generate_standard_visualizations received batch size > 1 for target_mag ({target_mag.shape}). Using first sample.")
                 target_mag = target_mag[0]
        if target_mag.ndim == 3 and target_mag.shape[0] == 1: # Channel dim remaining?
            target_mag = target_mag.squeeze(axis=0) # Now should be [D, H, W]

        # Final check after squeezing
        if pred_mag.ndim != 3:
            raise ValueError(f"Processed pred_mag must be 3D (D, H, W), but got shape {pred_mag.shape}")
        if target_mag.ndim != 3:
            raise ValueError(f"Processed target_mag must be 3D (D, H, W), but got shape {target_mag.shape}")


        # Define output paths
        pred_vs_gt_path = os.path.join(model_output_dir, 'magnitude_prediction_vs_ground_truth.png')
        hotspots_path = os.path.join(model_output_dir, 'magnitude_hotspots.png')

        logger.info(f"Generating standard magnitude visualizations for {model_name}")
        logger.debug(f"[GSV_Mag] Using pred_mag shape: {pred_mag.shape}, target_mag shape: {target_mag.shape}")

        # 1. Prediction vs. Ground Truth Magnitude
        fig_mag = visualize_prediction_vs_ground_truth(
            pred_mag, target_mag, mask=mask, save_path=pred_vs_gt_path, title_prefix=f"{model_name} - "
        )
        if fig_mag:
            visualization_paths['magnitude_prediction_vs_ground_truth'] = pred_vs_gt_path
            plt.close(fig_mag) # Close figure after saving

        # 2. Hotspot Analysis on Magnitude
        fig_hotspot, iou = visualize_hotspots(
            pred_mag, target_mag, mask=mask, save_path=hotspots_path, title_prefix=f"{model_name} - "
        )
        if fig_hotspot:
            visualization_paths['magnitude_hotspots'] = hotspots_path
            plt.close(fig_hotspot) # Close figure after saving

        # 3. Metrics Summary Plot (if metrics provided)
        if metrics_dict:
            metrics_path = os.path.join(model_output_dir, 'metrics_summary.png')
            # Convert single values in metrics_dict to lists for plotting history
            metrics_history = {k: [v] if not isinstance(v, (list, np.ndarray)) else v for k, v in metrics_dict.items() if v is not None}

            fig_metrics = create_metrics_summary_plot(
                metrics_history, save_path=metrics_path,
                title=f'{model_name} - Metrics Summary'
            )
            if fig_metrics:
                visualization_paths['metrics_summary'] = metrics_path
                plt.close(fig_metrics) # Close figure after saving

        logger.info(f"Generated {len(visualization_paths)} standard magnitude visualizations for {model_name}")

    except Exception as e:
        logger.error(f"Error generating standard visualizations for {model_name}: {e}", exc_info=True)
        # Ensure figures are closed if errors occurred
        for fig in [fig_mag, fig_hotspot, fig_metrics]:
             if fig: plt.close(fig)

    return visualization_paths


# --- Remove or Mark Vector Specific Visualizations as Deprecated ---

def visualize_vector_field(*args, **kwargs):
     logger.warning("visualize_vector_field is deprecated for magnitude prediction models.")
     return None

def visualize_directional_error(*args, **kwargs):
     logger.warning("visualize_directional_error is deprecated for magnitude prediction models.")
     return None

# --- Add Helper function from metrics.py if not already imported ---
# This is needed by the adapted visualize_hotspots
# Make sure it's available here or imported correctly.
# Assuming it's defined in the same file or imported:
# from .metrics import calculate_hotspot_iou_scalar
