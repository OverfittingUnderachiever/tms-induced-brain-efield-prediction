#!/usr/bin/env python3
"""
Visualization utility for TMS E-field prediction data.

This script provides enhanced visualization capabilities for TMS data,
including 3D visualizations of MRI, dA/dt, and E-field data.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib.widgets import Slider
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tms_visualizer')


def load_data(file_path: str):
    """
    Load processed data from .npz file.
    
    Args:
        file_path: Path to the processed data file
        
    Returns:
        Tuple of (stacked_data, mask, metadata, efield)
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Extract components
        stacked_data = data['stacked_data']
        mask = data['mask']
        metadata = data['metadata'].item()
        
        # Try to load E-field if available
        efield = None
        if 'efield' in data and data['efield'].size > 1:
            efield = data['efield']
        
        logger.info(f"Loaded data from {file_path}")
        logger.info(f"Stacked data shape: {stacked_data.shape}")
        logger.info(f"Mask shape: {mask.shape}")
        
        return stacked_data, mask, metadata, efield
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def create_orthogonal_views(data: np.ndarray, axis_names=None, mask=None, cmap='viridis', title=None):
    """
    Create orthogonal slice views (axial, coronal, sagittal) of 3D volume.
    
    Args:
        data: 3D data array
        axis_names: Optional list of axis names [x, y, z]
        mask: Optional binary mask to apply
        cmap: Colormap to use
        title: Optional plot title
    """
    if axis_names is None:
        axis_names = ['x', 'y', 'z']
    
    if len(data.shape) > 3:
        # For multi-channel data, use the first channel
        data_to_plot = data[..., 0]
    else:
        data_to_plot = data
    
    # Apply mask if provided
    if mask is not None:
        data_to_plot = data_to_plot * mask
    
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get middle slices
    slice_x = data_to_plot.shape[0] // 2
    slice_y = data_to_plot.shape[1] // 2
    slice_z = data_to_plot.shape[2] // 2
    
    # Plot slices
    axs[0].imshow(data_to_plot[slice_x, :, :].T, cmap=cmap, origin='lower')
    axs[0].set_title(f'Sagittal ({axis_names[0]}={slice_x})')
    axs[0].axis('off')
    
    axs[1].imshow(data_to_plot[:, slice_y, :].T, cmap=cmap, origin='lower')
    axs[1].set_title(f'Coronal ({axis_names[1]}={slice_y})')
    axs[1].axis('off')
    
    axs[2].imshow(data_to_plot[:, :, slice_z].T, cmap=cmap, origin='lower')
    axs[2].set_title(f'Axial ({axis_names[2]}={slice_z})')
    axs[2].axis('off')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    return fig


def create_interactive_slice_viewer(data: np.ndarray, mask=None, cmap='viridis', axis_idx=0):
    """
    Create an interactive slice viewer with slider control.
    
    Args:
        data: 3D data array (or 4D with channels)
        mask: Optional binary mask to apply
        cmap: Colormap to use
        axis_idx: Which axis to slice along (0,1,2)
    """
    # Check if data has channels
    has_channels = len(data.shape) > 3
    
    if has_channels:
        # Default to first channel
        channel_idx = 0
        data_view = data[..., channel_idx]
    else:
        data_view = data
    
    # Apply mask if provided
    if mask is not None:
        data_view = data_view * mask
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # Create axis labels based on slice direction
    if axis_idx == 0:
        slice_direction = 'Sagittal'
        img_indexing = lambda i: data_view[i, :, :].T
    elif axis_idx == 1:
        slice_direction = 'Coronal'
        img_indexing = lambda i: data_view[:, i, :].T
    else:
        slice_direction = 'Axial'
        img_indexing = lambda i: data_view[:, :, i].T
    
    # Get slice dimension
    slice_dim = data_view.shape[axis_idx]
    initial_slice = slice_dim // 2
    
    # Create initial image
    img = ax.imshow(img_indexing(initial_slice), cmap=cmap, origin='lower')
    plt.title(f'{slice_direction} Slice {initial_slice}/{slice_dim-1}')
    
    # Create slider
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, slice_dim-1, valinit=initial_slice, valstep=1)
    
    # Update function for slider
    def update(val):
        slice_idx = int(slider.val)
        img.set_data(img_indexing(slice_idx))
        plt.title(f'{slice_direction} Slice {slice_idx}/{slice_dim-1}')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Add channel slider if data has channels
    if has_channels:
        ax_channel_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        channel_slider = Slider(
            ax_channel_slider, 'Channel', 0, data.shape[-1]-1, 
            valinit=channel_idx, valstep=1
        )
        
        def update_channel(val):
            nonlocal data_view
            channel_idx = int(channel_slider.val)
            data_view = data[..., channel_idx]
            if mask is not None:
                data_view = data_view * mask
            img.set_data(img_indexing(int(slider.val)))
            plt.title(f'{slice_direction} Slice {int(slider.val)}/{slice_dim-1} - Channel {channel_idx}')
            fig.canvas.draw_idle()
        
        channel_slider.on_changed(update_channel)
    
    plt.tight_layout()
    return fig


def create_3d_surface_visualization(data: np.ndarray, mask: np.ndarray, threshold=0.2, 
                                  subsample=4, cmap='viridis', title=None):
    """
    Create 3D surface visualization of the data.
    
    Args:
        data: 3D data array (if 4D, first channel is used)
        mask: Binary mask
        threshold: Value threshold for visualization
        subsample: Subsampling factor to reduce points
        cmap: Colormap to use
        title: Optional plot title
    """
    # Handle multi-channel data
    if len(data.shape) > 3:
        data_to_plot = data[..., 0]
    else:
        data_to_plot = data
    
    # Apply mask
    masked_data = data_to_plot * mask
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates of voxels above threshold and in mask
    x, y, z = np.where((masked_data > threshold) & mask)
    
    # Subsample points to make visualization manageable
    idx = np.arange(0, len(x), subsample)
    x = x[idx]
    y = y[idx]
    z = z[idx]
    
    # Get data values for color mapping
    values = masked_data[x, y, z]
    
    # Create colormap
    norm = colors.Normalize(vmin=values.min(), vmax=values.max())
    
    # Plot 3D scatter
    scatter = ax.scatter(x, y, z, c=values, cmap=cmap, norm=norm, 
                         s=5, alpha=0.5, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig


def visualize_channel_comparison(stacked_data: np.ndarray, mask: np.ndarray, efield: np.ndarray = None,
                                output_dir: str = None):
    """
    Create comprehensive visualization comparing different channels.
    
    Args:
        stacked_data: Stacked data array (MRI + dA/dt)
        mask: Binary mask for valid voxels
        efield: Optional E-field data
        output_dir: Directory to save visualizations
    """
    # Create a figure
    fig = plt.figure(figsize=(18, 12))
    
    # Get number of channels in stacked data
    if len(stacked_data.shape) > 3:
        n_channels = stacked_data.shape[-1]
    else:
        n_channels = 1
    
    # Calculate number of rows needed
    n_rows = 1 + (n_channels + 2) // 3  # +2 for mask and e-field if available
    
    # Create grid of subplots
    axes = []
    for i in range(n_rows):
        for j in range(3):
            idx = i * 3 + j
            if idx < n_channels + (2 if efield is not None else 1):
                ax = fig.add_subplot(n_rows, 3, idx + 1)
                axes.append(ax)
    
    # Get middle slice
    slice_idx = stacked_data.shape[0] // 2
    
    # Plot each channel
    for i in range(n_channels):
        channel_data = stacked_data[slice_idx, :, :, i] if n_channels > 1 else stacked_data[slice_idx, :, :]
        axes[i].imshow(channel_data.T, cmap='viridis')
        channel_name = f"Channel {i}"
        if i == 0:
            channel_name = "MRI"
        elif i in [1, 2, 3]:
            comp = ["X", "Y", "Z"][i-1] if i <= 3 else ""
            channel_name = f"dA/dt {comp}"
        axes[i].set_title(channel_name)
        axes[i].axis('off')
    
    # Plot mask
    mask_ax_idx = n_channels
    axes[mask_ax_idx].imshow(mask[slice_idx, :, :].T, cmap='binary')
    axes[mask_ax_idx].set_title('Mask')
    axes[mask_ax_idx].axis('off')
    
    # Plot E-field if available
    if efield is not None:
        efield_ax_idx = n_channels + 1
        if len(efield.shape) > 3:  # Vector E-field
            # Calculate magnitude
            efield_mag = np.linalg.norm(efield, axis=-1)
            efield_slice = efield_mag[slice_idx, :, :]
        else:  # Scalar E-field
            efield_slice = efield[slice_idx, :, :]
        
        axes[efield_ax_idx].imshow(efield_slice.T, cmap='hot')
        axes[efield_ax_idx].set_title('E-field')
        axes[efield_ax_idx].axis('off')
    
    # Add overall title
    plt.suptitle('TMS Data Channel Comparison', fontsize=16)
    plt.tight_layout()
    
    # Save figure if output directory is provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'channel_comparison.png'), dpi=300)
        logger.info(f"Channel comparison saved to {os.path.join(output_dir, 'channel_comparison.png')}")
    
    return fig


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='TMS Data Visualization Utility')
    parser.add_argument('--data', type=str, required=True, help='Processed data file (.npz)')
    parser.add_argument('--output', type=str, default='./viz_output', help='Output directory')
    parser.add_argument('--interactive', action='store_true', help='Show interactive visualizations')
    parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for 3D visualization')
    parser.add_argument('--subsample', type=int, default=4, help='Subsample factor for 3D visualization')
    parser.add_argument('--channel', type=int, default=0, help='Initial channel to visualize')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    stacked_data, mask, metadata, efield = load_data(args.data)
    
    # Create visualizations
    logger.info("Creating orthogonal views")
    
    # Orthogonal views of MRI (usually first channel)
    mri_fig = create_orthogonal_views(
        stacked_data[..., 0] if stacked_data.shape[-1] > 1 else stacked_data,
        mask=mask,
        cmap='gray',
        title='MRI Orthogonal Views'
    )
    mri_fig.savefig(os.path.join(args.output, 'mri_orthogonal.png'), dpi=300)
    
    # If we have dA/dt (usually subsequent channels)
    if stacked_data.shape[-1] > 1:
        logger.info("Creating dA/dt visualizations")
        
        # Combine dA/dt channels if we have more than one (vector)
        if stacked_data.shape[-1] >= 4:  # MRI + 3 dA/dt components
            dadt_channels = stacked_data[..., 1:4]
            # Compute magnitude
            dadt_mag = np.linalg.norm(dadt_channels, axis=-1)
            
            dadt_fig = create_orthogonal_views(
                dadt_mag,
                mask=mask,
                cmap='plasma',
                title='dA/dt Magnitude Orthogonal Views'
            )
            dadt_fig.savefig(os.path.join(args.output, 'dadt_orthogonal.png'), dpi=300)
        else:
            # Single dA/dt channel
            dadt_fig = create_orthogonal_views(
                stacked_data[..., 1],
                mask=mask,
                cmap='plasma',
                title='dA/dt Orthogonal Views'
            )
            dadt_fig.savefig(os.path.join(args.output, 'dadt_orthogonal.png'), dpi=300)
    
    # E-field visualization if available
    if efield is not None:
        logger.info("Creating E-field visualizations")
        
        # Compute magnitude if vector E-field
        if len(efield.shape) > 3:
            efield_mag = np.linalg.norm(efield, axis=-1)
        else:
            efield_mag = efield
        
        efield_fig = create_orthogonal_views(
            efield_mag,
            mask=mask,
            cmap='hot',
            title='E-field Magnitude Orthogonal Views'
        )
        efield_fig.savefig(os.path.join(args.output, 'efield_orthogonal.png'), dpi=300)
    
    # Channel comparison
    logger.info("Creating channel comparison")
    comparison_fig = visualize_channel_comparison(stacked_data, mask, efield, args.output)
    
    # 3D visualization
    logger.info("Creating 3D visualization")
    
    # Choose data for 3D viz (MRI, dA/dt magnitude or E-field)
    if efield is not None:
        if len(efield.shape) > 3:
            viz_data = np.linalg.norm(efield, axis=-1)
        else:
            viz_data = efield
        viz_title = "E-field Magnitude 3D Visualization"
    elif stacked_data.shape[-1] > 3:  # MRI + vector dA/dt
        viz_data = np.linalg.norm(stacked_data[..., 1:4], axis=-1)
        viz_title = "dA/dt Magnitude 3D Visualization"
    elif stacked_data.shape[-1] > 1:  # MRI + scalar dA/dt
        viz_data = stacked_data[..., 1]
        viz_title = "dA/dt 3D Visualization"
    else:  # Just MRI
        viz_data = stacked_data
        viz_title = "MRI 3D Visualization"
    
    viz_3d_fig = create_3d_surface_visualization(
        viz_data, 
        mask, 
        threshold=args.threshold,
        subsample=args.subsample,
        title=viz_title
    )
    viz_3d_fig.savefig(os.path.join(args.output, '3d_visualization.png'), dpi=300)
    
    # Interactive visualizations if requested
    if args.interactive:
        logger.info("Creating interactive visualizations")
        
        plt.figure()
        plt.clf()
        interactive_fig = create_interactive_slice_viewer(
            stacked_data,
            mask=mask,
            cmap='viridis'
        )
        
        plt.show()
    else:
        plt.close('all')
    
    logger.info(f"All visualizations saved to {args.output}")


if __name__ == "__main__":
    main()