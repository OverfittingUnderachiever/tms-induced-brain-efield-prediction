#!/usr/bin/env python3
"""
Command-line tool for TMS field visualization and comparison.

This script provides a command-line interface for comparing E-field and dA/dt
data from TMS simulations using the field_comparison_visualization module.
"""

import os
import sys
import argparse
import logging
import numpy as np
import glob
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tms_field_comparison')

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
print(project_root)
# Import custom modules
# Import custom modules
from utils.field_comparison_visualization import (
    FieldVisualizer, FieldVisualizationConfig, VisualizationData
)
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TMS Field Comparison Visualization")
    
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
    parser.add_argument("--mesh", type=str, 
                      default=None,
                      help="Optional path to mesh file (.msh)")
    parser.add_argument("--mask", type=str, 
                      default=None,
                      help="Optional path to mask file (.npy)")
    
    # Output arguments
    parser.add_argument("--output", type=str, 
                      default="./field_visualization",
                      help="Output directory for visualizations, default: ./field_visualization")
    
    # Visualization parameters
    parser.add_argument("--position", type=int, 
                      default=0,
                      help="Position index to visualize (default: 0)")
    parser.add_argument("--slice-view", type=str, 
                      default="axial",
                      choices=["axial", "coronal", "sagittal"],
                      help="Slice view (default: axial)")
    parser.add_argument("--slice-index", type=int, 
                      default=None,
                      help="Slice index (default: auto)")
    parser.add_argument("--component", type=int, 
                      default=None,
                      help="Vector component to visualize (0=x, 1=y, 2=z, default: all)")
    parser.add_argument("--colormap-dadt", type=str, 
                      default="plasma",
                      help="Colormap for dA/dt data (default: plasma)")
    parser.add_argument("--colormap-efield", type=str, 
                      default="hot",
                      help="Colormap for E-field data (default: hot)")
    parser.add_argument("--alpha", type=float, 
                      default=0.7,
                      help="Transparency for overlay plots (default: 0.7)")
    parser.add_argument("--threshold", type=float, 
                      default=0.2,
                      help="Threshold for 3D visualization (default: 0.2)")
    parser.add_argument("--subsample", type=int, 
                      default=3,
                      help="Subsampling factor for 3D visualization (default: 3)")
    
    # Batch processing
    parser.add_argument("--all-positions", action="store_true",
                      help="Process all positions")
    parser.add_argument("--max-positions", type=int,
                      default=None,
                      help="Maximum number of positions to process (default: all)")
    parser.add_argument("--with-components", action="store_true",
                      help="Include vector component visualizations")
    
    # Display options
    parser.add_argument("--interactive", action="store_true",
                      help="Show interactive plots")
    
    return parser.parse_args()
def combine_efield_files(file_paths, output_path):
    """
    Combine individual E-field position files into a single array.
    
    Args:
        file_paths: List of file paths
        output_path: Path to save combined file
    """
    # Sort by position number
    def get_position_number(filename):
        match = re.search(r'position_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    file_paths.sort(key=get_position_number)
    
    # Load first file to get shape
    first_data = np.load(file_paths[0])
    
    # Create array to hold all positions
    all_data = np.zeros((len(file_paths), *first_data.shape), dtype=first_data.dtype)
    
    # Load each file
    for i, file_path in enumerate(file_paths):
        position = get_position_number(file_path)
        logger.info(f"Loading position {position} from {os.path.basename(file_path)}")
        all_data[i] = np.load(file_path)
    
    # Save combined array
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_data)
    logger.info(f"Saved combined data with shape {all_data.shape} to {output_path}")
    
    return output_path

def main():
    """Main function."""
    args = parse_args()
    
    # Handle case where efield is a directory
    efield_path = args.efield
    if os.path.isdir(efield_path):
        # Look for specific position file
        position_file = os.path.join(efield_path, f"{args.subject_id}_position_{args.position}.npy")
        if os.path.exists(position_file):
            efield_path = position_file
            logger.info(f"Using E-field file: {efield_path}")
        else:
            # Look for combined file
            combined_file = os.path.join(efield_path, f"{args.subject_id}_all_efields.npy")
            if os.path.exists(combined_file):
                efield_path = combined_file
                logger.info(f"Using combined E-field file: {efield_path}")
            else:
                # Find all position files
                position_files = glob.glob(os.path.join(efield_path, f"{args.subject_id}_position_*.npy"))
                if position_files:
                    # If all-positions mode, combine files temporarily
                    if args.all_positions:
                        combined_file = os.path.join(args.output, "temp_all_efields.npy")
                        combine_efield_files(position_files, combined_file)
                        efield_path = combined_file
                        logger.info(f"Created temporary combined E-field file: {efield_path}")
                    else:
                        # Use the specified position or default to 0
                        position_files.sort(key=lambda f: int(re.search(r'position_(\d+)', f).group(1)))
                        if args.position < len(position_files):
                            efield_path = position_files[args.position]
                            logger.info(f"Using E-field file: {efield_path}")
                        else:
                            logger.error(f"Position {args.position} not found in directory {args.efield}")
                            return 1
                else:
                    logger.error(f"No E-field files found in directory {args.efield}")
                    return 1
    # Verify input files
    if not os.path.exists(args.dadt):
        logger.error(f"dA/dt file not found: {args.dadt}")
        return 1
    
    if not os.path.exists(args.efield):
        logger.error(f"E-field file not found: {args.efield}")
        return 1
    
    if args.mesh and not os.path.exists(args.mesh):
        logger.error(f"Mesh file not found: {args.mesh}")
        return 1
    
    if args.mask and not os.path.exists(args.mask):
        logger.error(f"Mask file not found: {args.mask}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create visualization configuration
    config = FieldVisualizationConfig(
        slice_view=args.slice_view,
        colormap_dadt=args.colormap_dadt,
        colormap_efield=args.colormap_efield,
        alpha=args.alpha,
        slice_index=args.slice_index,
        vector_stride=5  # Fixed stride for clearer vector plots
    )
    
    # Create visualizer
    visualizer = FieldVisualizer(config)
    
    # Load data
    data = visualizer.load_simulation_data(
        dadt_file_path=args.dadt,
        efield_file_path=args.efield,
        mesh_file_path=args.mesh,
        mask_file_path=args.mask
    )
    
    # Process all positions if requested
    if args.all_positions:
        logger.info(f"Processing all positions (max: {args.max_positions})")
        visualizer.visualize_all_positions(
            data,
            output_dir=args.output,
            max_positions=args.max_positions,
            components=args.with_components
        )
        logger.info(f"All visualizations saved to {args.output}")
        return 0
    
    # Create a single position visualization
    logger.info(f"Creating visualizations for position {args.position}")
    
    # Magnitude comparison
    magnitude_path = os.path.join(args.output, f"position_{args.position}_magnitude.png")
    magnitude_fig = visualizer.visualize_field_comparison(
        data,
        output_path=magnitude_path if not args.interactive else None,
        position_index=args.position,
        component_index=args.component,
        show_magnitude=True,
        show_vectors=False
    )
    
    # Vector field comparison
    vector_path = os.path.join(args.output, f"position_{args.position}_vector.png")
    vector_fig = visualizer.visualize_field_comparison(
        data,
        output_path=vector_path if not args.interactive else None,
        position_index=args.position,
        component_index=args.component,
        show_magnitude=False,
        show_vectors=True
    )
    
    # Combined visualization
    combined_path = os.path.join(args.output, f"position_{args.position}_combined.png")
    combined_fig = visualizer.visualize_field_comparison(
        data,
        output_path=combined_path if not args.interactive else None,
        position_index=args.position,
        component_index=args.component,
        show_magnitude=True,
        show_vectors=True
    )
    
    # 3D comparison
    threed_path = os.path.join(args.output, f"position_{args.position}_3d.png")
    threed_fig = visualizer.visualize_3d_comparison(
        data,
        output_path=threed_path if not args.interactive else None,
        position_index=args.position,
        threshold=args.threshold,
        subsample=args.subsample
    )
    
    logger.info(f"Visualizations saved to {args.output}")
    
    # Show plots if interactive
    if args.interactive:
        import matplotlib.pyplot as plt
        plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())