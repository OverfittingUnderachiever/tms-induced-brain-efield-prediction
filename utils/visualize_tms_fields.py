#!/usr/bin/env python3
"""
Standalone TMS field visualization script.
Updated to handle both 2D and 3D data formats.
"""

import os
import sys
import argparse
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
    # Check if last dimension is 3 (vector components)
    if len(field_data.shape) > 1 and field_data.shape[-1] == 3:
        return np.linalg.norm(field_data, axis=-1)
    return field_data

def print_data_info(data, name):
    """Print information about data array."""
    logger.info(f"{name} shape: {data.shape}")
    logger.info(f"{name} min: {np.min(data)}, max: {np.max(data)}")
    logger.info(f"{name} has NaN: {np.isnan(data).any()}")
    logger.info(f"{name} has Inf: {np.isinf(data).any()}")

def visualize_fields(dadt_data, efield_data, position=0, output_dir="./viz_output", interactive=False):
    """Create visualizations comparing dA/dt and E-field data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Print information about the data
    print_data_info(dadt_data, "dA/dt data")
    print_data_info(efield_data, "E-field data")
    
    # Handle multiple positions for dA/dt
    if len(dadt_data.shape) > 3:
        if position >= dadt_data.shape[0]:
            logger.error(f"Position {position} exceeds available dA/dt data positions")
            return
        dadt = dadt_data[position]
    else:
        dadt = dadt_data
    
    # Handle E-field data shape
    is_efield_2d = len(efield_data.shape) <= 2
    
    # For 2D E-field, no need to extract specific position
    if not is_efield_2d and len(efield_data.shape) > 3:
        if position >= efield_data.shape[0]:
            logger.error(f"Position {position} exceeds available E-field data positions")
            return
        efield = efield_data[position]
    else:
        efield = efield_data
    
    # Create mask based on non-zero values in dA/dt
    if len(dadt.shape) > 2 and dadt.shape[-1] == 3:  # Vector data
        mask = np.any(np.abs(dadt) > 0, axis=-1)
    else:
        mask = np.abs(dadt) > 0
    
    # Calculate magnitudes
    dadt_mag = calculate_field_magnitude(dadt)
    efield_mag = calculate_field_magnitude(efield)
    
    # Choose slice for 3D data
    if len(dadt.shape) == 3 and dadt.shape[-1] != 3:  # 3D volumetric data
        slice_idx = dadt.shape[2] // 2  # Middle axial slice
        dadt_slice = dadt[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]
    elif len(dadt.shape) == 4:  # 3D vector field
        slice_idx = dadt.shape[2] // 2
        dadt_slice = dadt[:, :, slice_idx, :]
        mask_slice = mask[:, :, slice_idx]
    else:  # Already 2D
        dadt_slice = dadt
        mask_slice = mask
    
    # Handle 2D vs 3D E-field
    if is_efield_2d:
        efield_slice = efield
        efield_mag_slice = efield_mag
    else:
        if len(efield.shape) == 3 and efield.shape[-1] != 3:
            efield_slice = efield[:, :, slice_idx]
        elif len(efield.shape) == 4:
            efield_slice = efield[:, :, slice_idx, :]
        else:
            efield_slice = efield
        
        if len(efield_mag.shape) == 3:
            efield_mag_slice = efield_mag[:, :, slice_idx]
        else:
            efield_mag_slice = efield_mag
    
    # Ensure data has matching dimensions for comparison
    # If necessary, reshape or resample
    if dadt_mag_slice.shape != efield_mag_slice.shape:
        logger.warning(f"Data shape mismatch: dA/dt={dadt_mag_slice.shape}, E-field={efield_mag_slice.shape}")
        
        # Either resample E-field to match dA/dt dimensions
        if hasattr(np, 'resize'):  # Use numpy.resize as a last resort
            logger.warning(f"Resizing E-field data to match dA/dt dimensions")
            efield_mag_slice = np.resize(efield_mag_slice, dadt_mag_slice.shape)
        # Or just use the data as-is and let matplotlib handle it
    
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
    
    # Comparison plot - only if shapes match
    ax3 = fig.add_subplot(133)
    
    if dadt_mag_slice.shape == efield_mag_slice.shape:
        # Normalize data for comparison
        dadt_norm = (dadt_mag_slice - np.min(dadt_mag_slice)) / (np.max(dadt_mag_slice) - np.min(dadt_mag_slice) + 1e-10)
        efield_norm = (efield_mag_slice - np.min(efield_mag_slice)) / (np.max(efield_mag_slice) - np.min(efield_mag_slice) + 1e-10)
        
        # Difference
        difference = dadt_norm - efield_norm
        im3 = ax3.imshow(difference.T, cmap='coolwarm', origin='lower', norm=Normalize(vmin=-1, vmax=1))
        plt.colorbar(im3, ax=ax3, label='Normalized difference')
        
        # Add correlation
        correlation = np.corrcoef(dadt_mag_slice.flatten(), efield_mag_slice.flatten())[0, 1]
        ax3.set_title(f'Comparison (r={correlation:.3f})')
    else:
        ax3.text(0.5, 0.5, "Cannot compare:\nShape mismatch", 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Comparison')
    
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
        # Check if we have multiple positions
        has_multiple_dadt = len(dadt_data.shape) > 3
        has_multiple_efield = len(efield_data.shape) > 2 and isinstance(efield_data, np.ndarray) and hasattr(efield_data, 'shape')
        
        if not has_multiple_dadt and not has_multiple_efield:
            logger.warning("Neither dA/dt nor E-field data have multiple positions. Processing single position.")
            visualize_fields(dadt_data, efield_data, position=0, output_dir=output_dir, interactive=interactive)
            return
        
        # Determine number of positions
        if has_multiple_dadt:
            n_positions = dadt_data.shape[0]
        else:
            n_positions = efield_data.shape[0]
        
        # Process each position
        for pos in range(n_positions):
            pos_dir = os.path.join(output_dir, f"position_{pos}")
            os.makedirs(pos_dir, exist_ok=True)
            
            # Get data for this position
            if has_multiple_dadt:
                pos_dadt = dadt_data[pos]
            else:
                pos_dadt = dadt_data
                
            if has_multiple_efield:
                pos_efield = efield_data[pos]
            else:
                pos_efield = efield_data
            
            visualize_fields(pos_dadt, pos_efield, position=pos, output_dir=pos_dir, interactive=interactive)
            logger.info(f"Processed position {pos}")
    else:
        # Process single position
        visualize_fields(dadt_data, efield_data, position=position, output_dir=output_dir, interactive=interactive)
    
    logger.info("Visualization complete!")

def parse_args():
    parser = argparse.ArgumentParser(description="TMS Field Visualization")
    
    # Input data arguments
    parser.add_argument("--dadt", type=str, 
                      default="./dadt_sims/dAdts.h5",
                      help="Path to dA/dt data file (.h5 or .npy), default: ./dadt_sims/dAdts.h5")
    parser.add_argument("--efield", type=str,
                      default="./efield_sims",
                      help="Path to E-field data file or directory (.npy), default: ./efield_sims")
    parser.add_argument("--subject-id", type=str, 
                      default="001",
                      help="Subject ID for finding files, default: 001")
    
    # Output arguments
    parser.add_argument("--output", type=str, 
                      default="./field_visualization",
                      help="Output directory for visualizations, default: ./field_visualization")
    
    # Visualization parameters
    parser.add_argument("--position", type=int, 
                      default=0,
                      help="Position index to visualize (default: 0)")
    
    # Batch processing
    parser.add_argument("--all-positions", action="store_true",
                      help="Process all positions")
    
    # Display options
    parser.add_argument("--interactive", action="store_true",
                      help="Show interactive plots")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print current directory
    print(f"Working directory: {os.getcwd()}")
    
    # Run visualization
    run_visualization(
        dadt_path=args.dadt,
        efield_path=args.efield,
        subject_id=args.subject_id,
        position=args.position,
        output_dir=args.output,
        all_positions=args.all_positions,
        interactive=args.interactive
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())