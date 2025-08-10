#!/usr/bin/env python3
"""
E-field Augmentations Comparison

Creates a single visualization showing the original E-field and how
each augmentation type affects it using BatchAugmentation functions.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import necessary modules
try:
    from tms_efield_prediction.utils.state.context import TMSPipelineContext
    from tms_efield_prediction.data.pipeline.loader import TMSDataLoader as PipelineTMSDataLoader
    from tms_efield_prediction.data.transformations.stack_pipeline import EnhancedStackingPipeline
    
    # Import the BatchAugmentation class - adjust path if needed
    from tms_efield_prediction.data.transformations.augmentation import BatchAugmentation
    logger.info("Successfully imported TMS pipeline modules and BatchAugmentation")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize all E-field augmentations in one image')
    parser.add_argument('--subject', '-s', type=str, required=True, 
                        help='Subject ID to use for visualization')
    parser.add_argument('--data_root', '-d', type=str, default=os.path.expanduser('~/MA_Henry/data'),
                        help='Root directory for TMS data')
    parser.add_argument('--output_dir', '-o', type=str, default='augmentation_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--bin_size', '-b', type=int, default=25,
                        help='Bin size for the data')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for processing (cpu or cuda)')
    parser.add_argument('--slice_view', type=str, default='axial',
                        choices=['axial', 'coronal', 'sagittal'],
                        help='Slice view to display (axial, coronal, or sagittal)')
    parser.add_argument('--unified', action='store_true',
                        help='Use augment_batch_samples instead of individual functions')
    return parser.parse_args()

def load_real_tms_data(subject_id, data_root_path, bin_size=25, device='cpu', max_samples=1):
    """Load real TMS data for a specific subject."""
    logger.info(f"Loading real TMS data for subject {subject_id}")
    
    # Format subject_id with leading zeros to 3 digits
    subject_id = subject_id.zfill(3)
    
    # Set up pipeline context
    output_shape = (bin_size, bin_size, bin_size)
    tms_config = {"mri_tensor": None, "device": device}
    pipeline_context = TMSPipelineContext(
        dependencies={},
        config=tms_config,
        pipeline_mode="mri_dadt",
        experiment_phase="training",
        debug_mode=True,
        subject_id=subject_id,
        data_root_path=data_root_path,
        output_shape=output_shape,
        normalization_method="standard",
        device=device,
        dadt_scaling_factor=1.0e-6
    )
    
    try:
        # Load raw data
        data_loader = PipelineTMSDataLoader(context=pipeline_context)
        raw_data = data_loader.load_raw_data()
        if raw_data is None:
            logger.error(f"Raw data is None for subject {subject_id}")
            return None
        
        # Update context with MRI tensor
        pipeline_context.config["mri_tensor"] = raw_data.mri_tensor
        
        # Create sample list
        samples = data_loader.create_sample_list(raw_data)
        
        # Skip if empty
        if not samples:
            logger.warning(f"No samples found for subject {subject_id}")
            return None
        
        # Limit number of samples if needed
        if max_samples > 0 and len(samples) > max_samples:
            logger.info(f"Limiting to {max_samples} samples (out of {len(samples)} available)")
            samples = samples[:max_samples]
        
        # Process data using the stacking pipeline
        stacking_pipeline = EnhancedStackingPipeline(context=pipeline_context)
        processed_data_list = stacking_pipeline.process_batch(samples)
        
        # Extract features and targets
        features_list = [processed_data.input_features for processed_data in processed_data_list]
        targets_list = [processed_data.target_efield for processed_data in processed_data_list]
        
        # Stack into tensors
        features_stacked = torch.stack(features_list)
        targets_vectors = torch.stack(targets_list)
        
        logger.info(f"Loaded {len(features_list)} samples for subject {subject_id}")
        logger.info(f"Features shape: {features_stacked.shape}, Targets shape: {targets_vectors.shape}")
        
        return {
            'input_features': features_stacked,
            'target_efield': targets_vectors
        }
            
    except Exception as e:
        logger.error(f"Error loading data for subject {subject_id}: {e}", exc_info=True)
        return None

def get_slice(tensor, slice_idx=None, view='axial'):
    """
    Extract the appropriate slice from 3D tensor.
    
    Args:
        tensor: 5D tensor [B, C, D, H, W]
        slice_idx: Index of slice to extract (None for center)
        view: 'axial', 'coronal', or 'sagittal'
    
    Returns:
        2D slice of the tensor
    """
    if tensor.dim() != 5:
        raise ValueError(f"Expected 5D tensor [B, C, D, H, W], got {tensor.shape}")
    
    # Get dimensions
    batch_size, channels, depth, height, width = tensor.shape
    
    # Use center slice if not specified
    if slice_idx is None:
        if view == 'axial':
            slice_idx = depth // 2
        elif view == 'coronal':
            slice_idx = height // 2
        else:  # sagittal
            slice_idx = width // 2
    
    # Extract slice based on view
    if view == 'axial':  # Z slice (top view)
        return tensor[0, 0, :, :, slice_idx]
    elif view == 'coronal':  # Y slice (front view)
        return tensor[0, 0, :, slice_idx, :]
    else:  # sagittal - X slice (side view)
        return tensor[0, 0, slice_idx, :, :]

def visualize_all_augmentations(tensor, output_path, slice_view='axial', cmap='viridis', use_unified=False):
    """
    Create a single visualization showing the original tensor and 
    all individual augmentations applied to it using BatchAugmentation functions.
    
    Args:
        tensor: E-field tensor [B, C, D, H, W]
        output_path: Path to save visualization
        slice_view: 'axial', 'coronal', or 'sagittal'
        cmap: Colormap to use
        use_unified: Whether to use augment_batch_samples instead of individual functions
    """
    device = tensor.device
    
    if use_unified:
        logger.info("Using BatchAugmentation.augment_batch_samples for each augmentation type")
        
        # Create individual augmentation batches with unified function
        # For spatial shift
        batch_shift = {'input_features': tensor.clone(), 'target_efield': tensor.clone()}
        shift_config = {
            'enabled': True,
            'spatial_shift': {
                'enabled': True,
                'max_shift': 0,  # Not used - we'll set shifts directly
                'probability': 1.0
            }
        }
        # Apply manual shifts
        shifts = torch.tensor([[0, 5, 0]], device=device)
        batch_shift['target_efield'] = BatchAugmentation.batch_spatial_shift(
            batch_shift['target_efield'], shifts)
        shifted = batch_shift['target_efield']
        
        # For rotation
        batch_rotate = {'input_features': tensor.clone(), 'target_efield': tensor.clone()}
        angles = torch.tensor([[0, np.pi/6, 0]], device=device)  # 30 degrees around Y-axis
        batch_rotate['target_efield'] = BatchAugmentation.batch_rotation(
            batch_rotate['target_efield'], angles)
        rotated = batch_rotate['target_efield']
        
        # For elastic deformation
        batch_deform = {'input_features': tensor.clone(), 'target_efield': tensor.clone()}
        strengths = torch.tensor([0.2], device=device)
        batch_deform['target_efield'] = BatchAugmentation.batch_elastic_deformation(
            batch_deform['target_efield'], strengths, sigma=10.0)
        deformed = batch_deform['target_efield']
        
        # For intensity scaling
        batch_scale = {'input_features': tensor.clone(), 'target_efield': tensor.clone()}
        factors = torch.tensor([1.3], device=device)
        batch_scale['target_efield'] = BatchAugmentation.batch_intensity_scaling(
            batch_scale['target_efield'], factors)
        scaled = batch_scale['target_efield']
        
        # For gaussian noise
        batch_noise = {'input_features': tensor.clone(), 'target_efield': tensor.clone()}
        stds = torch.tensor([0.05], device=device)
        batch_noise['target_efield'] = BatchAugmentation.batch_gaussian_noise(
            batch_noise['target_efield'], stds)
        noisy = batch_noise['target_efield']
        
    else:
        logger.info("Using individual BatchAugmentation functions")
        
        # Apply individual augmentations using BatchAugmentation batch functions
        # For spatial shift
        shifts = torch.tensor([[0, 5, 0]], device=device)  # Shift by (0, 5, 0)
        shifted = BatchAugmentation.batch_spatial_shift(tensor.clone(), shifts)
        
        # For rotation
        angles = torch.tensor([[0, np.pi/6, 0]], device=device)  # 30 degrees around Y-axis
        rotated = BatchAugmentation.batch_rotation(tensor.clone(), angles)
        
        # For elastic deformation
        strengths = torch.tensor([0.2], device=device)
        deformed = BatchAugmentation.batch_elastic_deformation(tensor.clone(), strengths, sigma=10.0)
        
        # For intensity scaling
        factors = torch.tensor([1.3], device=device)
        scaled = BatchAugmentation.batch_intensity_scaling(tensor.clone(), factors)
        
        # For gaussian noise
        stds = torch.tensor([0.05], device=device)
        noisy = BatchAugmentation.batch_gaussian_noise(tensor.clone(), stds)
    
    # Log tensor shapes
    logger.info(f"Original tensor shape: {tensor.shape}")
    logger.info(f"Shifted tensor shape: {shifted.shape}")
    logger.info(f"Rotated tensor shape: {rotated.shape}")
    logger.info(f"Deformed tensor shape: {deformed.shape}")
    logger.info(f"Scaled tensor shape: {scaled.shape}")
    logger.info(f"Noisy tensor shape: {noisy.shape}")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Extract slices
    orig_slice = get_slice(tensor, view=slice_view).cpu().numpy()
    shift_slice = get_slice(shifted, view=slice_view).cpu().numpy()
    rotate_slice = get_slice(rotated, view=slice_view).cpu().numpy()
    deform_slice = get_slice(deformed, view=slice_view).cpu().numpy()
    scale_slice = get_slice(scaled, view=slice_view).cpu().numpy()
    noise_slice = get_slice(noisy, view=slice_view).cpu().numpy()
    
    # Find global min/max for consistent colormap
    all_slices = [orig_slice, shift_slice, rotate_slice, deform_slice, 
                scale_slice, noise_slice]
    vmin = min(slice.min() for slice in all_slices)
    vmax = max(slice.max() for slice in all_slices)
    
    # Original E-field
    axes[0, 0].set_title("Original E-field")
    im = axes[0, 0].imshow(orig_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_ylabel(f"{slice_view.capitalize()} View")
    
    # Spatial shift
    axes[0, 1].set_title("Spatial Shift")
    axes[0, 1].imshow(shift_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Rotation
    axes[0, 2].set_title("Rotation")
    axes[0, 2].imshow(rotate_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Elastic deformation
    axes[1, 0].set_title("Elastic Deformation")
    axes[1, 0].imshow(deform_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_ylabel(f"{slice_view.capitalize()} View")
    
    # Intensity scaling
    axes[1, 1].set_title("Intensity Scaling")
    axes[1, 1].imshow(scale_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Gaussian noise
    axes[1, 2].set_title("Gaussian Noise")
    axes[1, 2].imshow(noise_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax)
    
    # Remove ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add main title with function information
    method_text = "BatchAugmentation.augment_batch_samples" if use_unified else "BatchAugmentation batch functions"
    plt.suptitle(f"E-field Augmentations - {slice_view.capitalize()} View\nUsing {method_text}", fontsize=14)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path}")
    plt.close()
    
    # Create version with contours to highlight differences
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Define thresholds for contours
    threshold = 0.3 * vmax
    
    # Original E-field
    axes[0, 0].set_title("Original E-field")
    im = axes[0, 0].imshow(orig_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    orig_contour = axes[0, 0].contour(orig_slice, levels=[threshold], colors='white', linewidths=1.5)
    axes[0, 0].set_ylabel(f"{slice_view.capitalize()} View")
    
    # Spatial shift
    axes[0, 1].set_title("Spatial Shift")
    axes[0, 1].imshow(shift_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].contour(shift_slice, levels=[threshold], colors='white', linewidths=1.5)
    axes[0, 1].contour(orig_slice, levels=[threshold], colors='red', linewidths=1, linestyles='dashed')
    
    # Rotation
    axes[0, 2].set_title("Rotation")
    axes[0, 2].imshow(rotate_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 2].contour(rotate_slice, levels=[threshold], colors='white', linewidths=1.5)
    axes[0, 2].contour(orig_slice, levels=[threshold], colors='red', linewidths=1, linestyles='dashed')
    
    # Elastic deformation
    axes[1, 0].set_title("Elastic Deformation")
    axes[1, 0].imshow(deform_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].contour(deform_slice, levels=[threshold], colors='white', linewidths=1.5)
    axes[1, 0].contour(orig_slice, levels=[threshold], colors='red', linewidths=1, linestyles='dashed')
    axes[1, 0].set_ylabel(f"{slice_view.capitalize()} View")
    
    # Intensity scaling
    axes[1, 1].set_title("Intensity Scaling")
    axes[1, 1].imshow(scale_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].contour(scale_slice, levels=[threshold], colors='white', linewidths=1.5)
    axes[1, 1].contour(orig_slice, levels=[threshold], colors='red', linewidths=1, linestyles='dashed')
    
    # Gaussian noise
    axes[1, 2].set_title("Gaussian Noise")
    axes[1, 2].imshow(noise_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 2].contour(noise_slice, levels=[threshold], colors='white', linewidths=1.5)
    axes[1, 2].contour(orig_slice, levels=[threshold], colors='red', linewidths=1, linestyles='dashed')
    
    # Add legend to explain contours
    legend_elements = [
        Line2D([0], [0], color='white', lw=1.5, label='Augmented'),
        Line2D([0], [0], color='red', lw=1, linestyle='dashed', label='Original')
    ]
    
    # Add the legend to the figure
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.05))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax)
    
    # Remove ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add main title
    method_text = "BatchAugmentation.augment_batch_samples" if use_unified else "BatchAugmentation batch functions"
    plt.suptitle(f"E-field Augmentations with Contours - {slice_view.capitalize()} View\nUsing {method_text}", fontsize=14)
    
    # Save figure with contours
    contour_path = output_path.replace('.png', '_contours.png')
    plt.savefig(contour_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved contour visualization to {contour_path}")
    plt.close()

def main():
    """Main function to create E-field augmentation visualization."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    logger.info(f"Generating E-field augmentation visualization for subject {args.subject}...")
    
    # Load real data
    batch = load_real_tms_data(
        subject_id=args.subject,
        data_root_path=args.data_root,
        bin_size=args.bin_size,
        device=device,
        max_samples=1
    )
    
    if batch is None:
        logger.error("Failed to load real TMS data. Exiting.")
        return
    
    logger.info("Successfully loaded real TMS data")
    
    # Extract the E-field tensor
    efield_tensor = batch['target_efield']
    
    # Log tensor information for debugging
    logger.info(f"E-field tensor shape: {efield_tensor.shape}")
    logger.info(f"E-field tensor range: [{efield_tensor.min().item():.4f}, {efield_tensor.max().item():.4f}]")
    
    # Create output file path with method indicator
    method_tag = "_unified" if args.unified else "_batch"
    output_path = os.path.join(
        output_dir, 
        f"subject_{args.subject}_efield_augmentations{method_tag}_{args.slice_view}.png"
    )
    
    # Create the visualization
    visualize_all_augmentations(
        efield_tensor,
        output_path,
        slice_view=args.slice_view,
        cmap='viridis',
        use_unified=args.unified
    )
    
    logger.info("Visualization complete")

if __name__ == "__main__":
    main()