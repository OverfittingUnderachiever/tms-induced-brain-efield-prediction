"""
Simplified field visualization without complicated imports.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import h5py
import logging
import glob
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tms_field_viz')

def load_dadt_data(file_path):
    """Load dA/dt data from HDF5 or numpy file."""
    try:
        if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            with h5py.File(file_path, 'r') as f:
                if 'dAdt' in f:
                    return f['dAdt'][:]
                else:
                    # Try first dataset
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            return f[key][:]
        elif file_path.endswith('.npy'):
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logger.error(f"Error loading dA/dt data: {e}")
        raise

def calculate_field_magnitude(field_data):
    """Calculate magnitude of vector field."""
    if field_data.shape[-1] != 3:
        return field_data
    return np.linalg.norm(field_data, axis=-1)

def calculate_field_direction(field_data):
    """Calculate unit direction vectors of vector field."""
    if field_data.shape[-1] != 3:
        return field_data
    magnitude = np.linalg.norm(field_data, axis=-1, keepdims=True)
    non_zero = magnitude > 1e-15
    direction = np.zeros_like(field_data)
    if np.any(non_zero):
        direction[non_zero[..., 0]] = field_data[non_zero[..., 0]] / magnitude[non_zero[..., 0]]
    return direction

def visualize_fields(dadt_data, efield_data, position=0, output_dir="./viz_output", interactive=False):
    """Create visualizations comparing dA/dt and E-field data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle multiple positions
    if len(dadt_data.shape) > 3:
        if position >= dadt_data.shape[0]:
            logger.error(f"Position {position} exceeds available dA/dt data positions")
            return
        dadt = dadt_data[position]
    else:
        dadt = dadt_data
        
    if len(efield_data.shape) > 3:
        if position >= efield_data.shape[0]:
            logger.error(f"Position {position} exceeds available E-field data positions")
            return
        efield = efield_data[position]
    else:
        efield = efield_data
    
    # Create mask based on non-zero values
    mask = np.any(np.abs(dadt) > 0, axis=-1) if len(dadt.shape) > 2 else np.abs(dadt) > 0
    
    # Calculate magnitude
    dadt_mag = calculate_field_magnitude(dadt)
    efield_mag = calculate_field_magnitude(efield)
    
    # Create middle slices
    slice_idx = dadt.shape[2] // 2  # Axial slice
    
    dadt_slice = dadt[:, :, slice_idx]
    efield_slice = efield[:, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]
    
    dadt_mag_slice = dadt_mag[:, :, slice_idx]
    efield_mag_slice = efield_mag[:, :, slice_idx]
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Magnitude plots
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(dadt_mag_slice.T, cmap='plasma', origin='lower')
    plt.colorbar(im1, ax=ax1, label='dA/dt magnitude')
    ax1.set_title('dA/dt Field')
    
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(efield_mag_slice.T, cmap='hot', origin='lower')
    plt.colorbar(im2, ax=ax2, label='E-field magnitude')
    ax2.set_title('E-field')
    
    # Comparison (correlation)
    ax3 = fig.add_subplot(133)
    
    # Normalize data
    dadt_norm = (dadt_mag_slice - np.min(dadt_mag_slice)) / (np.max(dadt_mag_slice) - np.min(dadt_mag_slice) + 1e-10)
    efield_norm = (efield_mag_slice - np.min(efield_mag_slice)) / (np.max(efield_mag_slice) - np.min(efield_mag_slice) + 1e-10)
    
    # Difference
    difference = dadt_norm - efield_norm
    im3 = ax3.imshow(difference.T, cmap='coolwarm', origin='lower', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar(im3, ax=ax3, label='Difference')
    
    # Add correlation
    correlation = np.corrcoef(dadt_mag_slice.flatten(), efield_mag_slice.flatten())[0, 1]
    ax3.set_title(f'Comparison (r={correlation:.3f})')
    
    plt.suptitle(f'TMS Field Comparison - Position {position}')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'field_comparison_pos{position}.png'), dpi=300)
    logger.info(f"Saved visualization to {os.path.join(output_dir, f'field_comparison_pos{position}.png')}")
    
    if interactive:
        plt.show()
    else:
        plt.close()
    
    return fig

def run_visualization(dadt_path, efield_path, subject_id="001", position=0, 
                      output_dir="./viz_output", all_positions=False, interactive=False):
    """Run field visualization with minimal parameters."""
    # Handle case where efield is a directory
    if os.path.isdir(efield_path):
        position_files = glob.glob(os.path.join(efield_path, f"{subject_id}_position_*.npy"))
        if not position_files:
            logger.error(f"No position files found in {efield_path}")
            return
        
        # Sort by position number
        position_files.sort(key=lambda f: int(re.search(r'position_(\d+)', f).group(1)))
        
        if all_positions:
            # Load all positions
            all_efields = []
            for pos_file in position_files:
                pos_data = np.load(pos_file)
                all_efields.append(pos_data)
            efield_data = np.stack(all_efields)
            logger.info(f"Loaded {len(position_files)} position files")
        else:
            # Load specific position
            if position < len(position_files):
                efield_data = np.load(position_files[position])
                logger.info(f"Loaded position {position} from {position_files[position]}")
            else:
                logger.error(f"Position {position} not found (max: {len(position_files)-1})")
                return
    else:
        # Direct file path
        if not os.path.exists(efield_path):
            logger.error(f"E-field file not found: {efield_path}")
            return
        efield_data = np.load(efield_path)
    
    # Load dA/dt data
    if not os.path.exists(dadt_path):
        logger.error(f"dA/dt file not found: {dadt_path}")
        return
    dadt_data = load_dadt_data(dadt_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process based on all_positions flag
    if all_positions:
        # Ensure both have multiple positions
        if len(dadt_data.shape) <= 3 or len(efield_data.shape) <= 3:
            logger.error("Both dA/dt and E-field data must have multiple positions for --all-positions")
            return
        
        # Process each position
        for pos in range(min(dadt_data.shape[0], efield_data.shape[0])):
            pos_dir = os.path.join(output_dir, f"position_{pos}")
            os.makedirs(pos_dir, exist_ok=True)
            visualize_fields(dadt_data, efield_data, position=pos, output_dir=pos_dir, interactive=interactive)
            logger.info(f"Processed position {pos}")
    else:
        # Process single position
        visualize_fields(dadt_data, efield_data, position=position, output_dir=output_dir, interactive=interactive)
    
    logger.info("Visualization complete!")