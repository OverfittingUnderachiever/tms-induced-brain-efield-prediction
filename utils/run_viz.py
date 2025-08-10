#!/usr/bin/env python3
"""
Simple script for TMS field visualization.
"""

import os
import sys
import argparse
import logging
from utils.field_viz import run_visualization

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('run_viz')

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
    print(os.getcwd())
    
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