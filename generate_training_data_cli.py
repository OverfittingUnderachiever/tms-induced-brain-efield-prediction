#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training data generation CLI for TMS E-field prediction.

This script integrates the functionality from generate_training_data.py
with the TMS E-field prediction project architecture.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import nibabel as nib  # For loading .nii.gz files
from scipy.interpolate import RegularGridInterpolator


import scipy.io as sio # <--- Fix for NameError
import traceback       # <--- For error printing
import shutil          # <--- For cleanup
import pickle          # <--- For saving/loading pickle format
from tqdm import tqdm  # <--- For progress bars
from simnibs import mesh_io
    
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.debug.hooks import PipelineDebugHook
from utils.debug.context import PipelineDebugContext
from utils.resource.monitor import ResourceMonitor
from utils.state.context import TMSPipelineContext
from data.transformations.voxel_mapping import VoxelMapper
from data.pipeline.field_processor import FieldProcessor, FieldProcessingConfig
from data.transformations.element_to_node import interpolate_element_to_node
from extract_efields_from_mat import extract_and_save_efields, print_attrs
from generate_dadt import generate_dadt_from_mat

def parse_args():
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate training data for TMS E-field prediction")


    parser.add_argument('--mri-type', type=str, default='dti', choices=['dti', 'conductivity'],
                   help='Type of MRI data to use: "dti" for DTI tensor or "conductivity" for tissue type')
    parser.add_argument("--subjects", "-s", nargs='+', default=["006"],
                        help="List of subject IDs (e.g., 006 007 008)")

    parser.add_argument("--bin_size", "-b", type=int, default=25,
                        help="Bin size for voxelization (default: 25)")

    parser.add_argument("--processes", "-p", type=int, default=None,
                        help="Number of processes to use (default: all available)")

    parser.add_argument("--clean", "-c", action="store_true",
                        help="Clean up intermediate files after processing")

    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory (default: derived from subject)")

    parser.add_argument("--data_root", "-d", type=str, default="/home/freyhe/MA_Henry/data",
                        help="Root directory for data (default: /home/freyhe/MA_Henry/data)")

    parser.add_argument("--formats", "-f", type=str, default="torch",
                        help="Output formats (comma-separated: torch,numpy,pickle; default: torch)")

    parser.add_argument("--debug_level", type=int, default=1,
                        help="Debug verbosity level (0-3, default: 1)")

    # --- ADDED Arguments for MRI Mode Selection ---
    parser.add_argument("--mri_mode", type=str, default="conductivity", choices=["conductivity", "dti"],
                        help="Type of MRI data to process ('conductivity' or 'dti'). Default: conductivity")

    parser.add_argument("--dti_file_pattern", type=str, default="{subject_id}_dti_node_data.npy",
                        help="Filename pattern for DTI node data relative to experiment/all dir (use {subject_id} placeholder). Only used if --mri_mode=dti. Default: '{subject_id}_dti_node_data.npy'")
    # --- END ADDED ---

    return parser.parse_args()

def _get_required_file_path(description: str, base_path: str, *path_components: str) -> str:
    """Constructs and verifies existence of a required file path."""
    path = os.path.join(base_path, *path_components)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required {description} file not found at expected location: {path}")
    return path

# Add this code to the beginning of generate_training_data_cli.py right after your imports
# This will import the extraction functionality directly

# Import extract_efields_from_mat.py functionality
try:
    # Import the extraction functions directly
    from extract_efields_from_mat import extract_and_save_efields, print_attrs
    EXTRACTION_AVAILABLE = True
except ImportError:
    print("Warning: extract_efields_from_mat.py not found in path. E-field extraction will not be available.")
    EXTRACTION_AVAILABLE = False

# Add this function to search for and extract E-fields
def find_and_extract_efields(subject_id, data_root):
    """
    Search for .mat files containing E-fields and extract them.
    
    Args:
        subject_id: Subject ID with or without 'sub-' prefix
        data_root: Root directory for data
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    if not EXTRACTION_AVAILABLE:
        print("E-field extraction not available. Please ensure extract_efields_from_mat.py is in the same directory.")
        return False
    
    # Format subject ID consistently
    subject_id_clean = subject_id.replace("sub-", "")
    subject_id_full = f"sub-{subject_id_clean}"
    
    # Find the experiment directory
    base_dirs = [
        os.path.join(data_root, subject_id_full),
        os.path.join(data_root, subject_id_clean)
    ]
    
    experiment_dir = None
    for base_dir in base_dirs:
        exp_dir = os.path.join(base_dir, "experiment")
        if os.path.exists(exp_dir):
            experiment_dir = exp_dir
            break
    
    if not experiment_dir:
        print(f"ERROR: Could not find experiment directory for subject {subject_id_full}")
        return False
    
    # Try various potential locations for the .mat file
    mat_files = []
    
    # Common locations
    potential_locations = [
        os.path.join(experiment_dir, "all"),
        experiment_dir,
        os.path.join(experiment_dir, "efields"),
        os.path.join(experiment_dir, "e_field")
    ]
    
    # Check common locations first
    for location in potential_locations:
        if os.path.exists(location):
            # Try different naming patterns
            patterns = [
                f"{subject_id_full}_middle_gray_matter_efields.mat",
                f"{subject_id_clean}_middle_gray_matter_efields.mat",
                f"middle_gray_matter_efields_{subject_id_full}.mat",
                f"middle_gray_matter_efields_{subject_id_clean}.mat",
                f"{subject_id_full}_efields.mat",
                f"{subject_id_clean}_efields.mat",
                "efields.mat"
            ]
            
            for pattern in patterns:
                mat_path = os.path.join(location, pattern)
                if os.path.exists(mat_path):
                    mat_files.append(mat_path)
    
    # If not found in common locations, search more broadly
    if not mat_files:
        print("Searching for .mat files containing E-fields in experiment directory...")
        for root, _, files in os.walk(experiment_dir):
            for file in files:
                if file.endswith(".mat") and ("efield" in file.lower() or "e_field" in file.lower()):
                    mat_files.append(os.path.join(root, file))
    
    if not mat_files:
        print(f"ERROR: Could not find any .mat files containing E-fields for subject {subject_id_full}")
        return False
    
    # Try to extract E-fields from each found .mat file
    for mat_file in mat_files:
        print(f"\nAttempting to extract E-fields from: {mat_file}")
        try:
            # Call the extract_and_save_efields function
            extract_and_save_efields(mat_file, experiment_dir, subject_id_full, explore_only=False)
            
            # Check if the extraction was successful by looking for the output file
            expected_output = os.path.join(experiment_dir, "multi_sim_100", f"{subject_id_full}_all_efields.npy")
            if os.path.exists(expected_output):
                print(f"Extraction successful! E-field data saved to: {expected_output}")
                return True
            else:
                print(f"Warning: Extraction didn't produce the expected output file: {expected_output}")
                # Continue trying other files if available
        except Exception as e:
            print(f"Error extracting from {mat_file}: {e}")
            import traceback
            traceback.print_exc()
            # Continue trying other files if available
    
    print("Failed to extract E-fields from any of the found .mat files.")
    return False

# Now modify the derive_paths function to use this function when E-field data is not found
def derive_paths(subject_id, data_root, mri_mode, dti_file_pattern):
    """Derive file paths based on subject ID and MRI mode.

    Args:
        subject_id: Subject ID
        data_root: Root directory for data
        mri_mode: The type of MRI data ('conductivity' or 'dti').
        dti_file_pattern: Filename pattern for DTI node data (used if mri_mode='dti').

    Returns:
        Dictionary of file paths. Includes DTI paths only if mri_mode is 'dti'.
    """
    # Remove 'sub-' prefix if present for path construction
    subject_id_clean = subject_id.replace("sub-", "")
    subject_id_full = f"sub-{subject_id_clean}" # Ensure consistent prefix usage

    # Base directory
    base_dir = os.path.join(data_root, subject_id_full, "experiment")

    # Check if base directory exists
    if not os.path.exists(base_dir):
        # Try alternative path with clean subject ID
        alt_base = os.path.join(data_root, subject_id_clean, "experiment")
        if os.path.exists(alt_base):
            base_dir = alt_base
            print(f"Using alternative base directory: {base_dir}")
        else:
             # Try alternative path by adding sub- prefix back to clean id
             alt_base_prefix = os.path.join(data_root, f"sub-{subject_id_clean}", "experiment")
             if os.path.exists(alt_base_prefix):
                  base_dir = alt_base_prefix
                  print(f"Using alternative base directory: {alt_base_prefix}")
             else:
                  raise ValueError(f"Base directory not found: Tried {os.path.join(data_root, subject_id_full)}, {alt_base}, {alt_base_prefix}")

    # Paths for E-fields
    efield_data_dir = os.path.join(base_dir, "multi_sim_100")
    efield_data_path = os.path.join(efield_data_dir, f"{subject_id_full}_all_efields.npy")
    if not os.path.exists(efield_data_path):
        # Try alternative paths
        alt_paths = [
            os.path.join(efield_data_dir, f"{subject_id_clean}_all_efields.npy")
        ]
        for path in alt_paths:
            if os.path.exists(path):
                efield_data_path = path
                print(f"Using alternative E-field path: {efield_data_path}")
                break

        if not os.path.exists(efield_data_path):
            # Before giving up, try extracting E-fields
            print(f"\nE-field data not found at any expected location.")
            print(f"Attempting to extract E-fields from .mat file for subject {subject_id_full}...")

            extraction_success = find_and_extract_efields(subject_id_full, data_root) # Use full id

            if extraction_success:
                 # Use the original expected path first, as extraction should create it
                 expected_extracted_path = os.path.join(efield_data_dir, f"{subject_id_full}_all_efields.npy")
                 if os.path.exists(expected_extracted_path):
                     efield_data_path = expected_extracted_path
                     print(f"Using freshly extracted E-field data from: {efield_data_path}")
                 else:
                     # Check alternatives again if main one still missing
                     for path in alt_paths:
                         if os.path.exists(path):
                             efield_data_path = path
                             print(f"Using freshly extracted E-field data (alt path): {efield_data_path}")
                             break
                     if not os.path.exists(efield_data_path):
                         print(f"Warning: Extraction reported success but E-field data still not found at expected locations.")
                         print(f"Please check the extraction output for actual file location.")
                         raise FileNotFoundError(f"E-field data not found at expected locations after extraction.")
            else:
                raise FileNotFoundError(f"E-field data not found and extraction failed.")

    # Use the middle gray matter ROI mesh for E-fields, dA/dt, and DTI
    all_dir = os.path.join(base_dir, "all")
    roi_mesh_path = os.path.join(all_dir, f"{subject_id_full}_middle_gray_matter_roi.msh")
    if not os.path.exists(roi_mesh_path):
        # Try alternative paths
        alt_paths = [
            os.path.join(all_dir, f"{subject_id_clean}_middle_gray_matter_roi.msh")
        ]
        for path in alt_paths:
            if os.path.exists(path):
                roi_mesh_path = path
                print(f"Using alternative ROI mesh path: {roi_mesh_path}")
                break

        if not os.path.exists(roi_mesh_path):
            # Check mesh_outputs directory created by extraction script
            mesh_outputs_dir = os.path.join(efield_data_dir, "mesh_outputs")
            if os.path.exists(mesh_outputs_dir):
                potential_meshes = [
                    os.path.join(mesh_outputs_dir, f"{subject_id_full}_efield_first.msh"),
                    os.path.join(mesh_outputs_dir, f"{subject_id_clean}_efield_first.msh")
                ]
                for path in potential_meshes:
                    if os.path.exists(path):
                        roi_mesh_path = path
                        print(f"Using mesh from extraction script: {roi_mesh_path}")
                        break

            if not os.path.exists(roi_mesh_path):
                raise FileNotFoundError(f"ROI mesh file not found at any expected location")

    print(f"Using ROI mesh file for E-field, dA/dt, DTI: {roi_mesh_path}")

    efield_output_dir = os.path.join(base_dir, "E_arrays")

    # Paths for dAdt fields
    dadt_data_dir = os.path.join(base_dir, "dadt_roi_maps")
    dadt_data_path = os.path.join(dadt_data_dir, f"{subject_id_full}_roi_dadts.npy")
    if not os.path.exists(dadt_data_path):
         alt_path = os.path.join(dadt_data_dir, f"{subject_id_clean}_roi_dadts.npy")
         if os.path.exists(alt_path):
              dadt_data_path = alt_path
              print(f"Using alternative dA/dt path: {dadt_data_path}")
         else:
             print(f"dA/dt data not found at {dadt_data_path} or {alt_path}. Attempting to generate...")
             # Make sure generate_dadt_from_mat can handle different subject ID formats if needed
             generate_dadt_from_mat(subject_id_full) # Pass full ID like sub-001
             expected_generated_path = os.path.join(dadt_data_dir, f"{subject_id_full}_roi_dadts.npy")
             if not os.path.exists(expected_generated_path) and not os.path.exists(alt_path):
                  raise FileNotFoundError(f"dA/dt data not found even after attempting generation.")
             elif os.path.exists(expected_generated_path):
                 dadt_data_path = expected_generated_path
                 print(f"Using generated dA/dt data from {dadt_data_path}")
             else:
                 dadt_data_path = alt_path
                 print(f"Using generated dA/dt data from {alt_path}")

    # Use the same ROI mesh for dA/dt
    dadt_mesh_path = roi_mesh_path
    dadt_output_dir = os.path.join(base_dir, "dAdt_arrays")

    # Path for stacked results
    stacked_output_dir = os.path.join(base_dir, "stacked_arrays")

    # Verify mesh file exists
    try:
        test_mesh = mesh_io.read_msh(roi_mesh_path)
        print(f"Loaded ROI mesh ({os.path.basename(roi_mesh_path)}) with {len(test_mesh.nodes.node_coord)} nodes and {len(test_mesh.elm.elm_number)} elements")
    except Exception as e:
        print(f"WARNING: Could not load ROI mesh file: {e}")
        print("This might cause problems in the subsequent processing steps.")

    # Base paths dictionary
    paths = {
        'efield_data_path': efield_data_path,
        'efield_mesh_path': roi_mesh_path,
        'efield_output_dir': efield_output_dir,
        'dadt_data_path': dadt_data_path,
        'dadt_mesh_path': roi_mesh_path,
        'dadt_output_dir': dadt_output_dir,
        'stacked_output_dir': stacked_output_dir,
        'base_dir': base_dir
    }

    # --- Conditionally Add DTI Paths ---
    if mri_mode == 'dti':
        print(f"Deriving paths for DTI mode...")
        # Set up output directory for DTI data
        dti_output_dir = os.path.join(base_dir, "MRI_arrays")
        os.makedirs(dti_output_dir, exist_ok=True)
        
        # Set up paths for DTI processed node data (if/when generated)
        dti_node_data_path = os.path.join(all_dir, f"{subject_id_full}_dti_node_data.npy")
        
        # Look for DTI tensor file
        tensor_paths = [
            os.path.join(data_root, subject_id_full, "headmodel", f"d2c_{subject_id_full}", "dti_results_T1space", "DTI_conf_tensor.nii.gz"),
            os.path.join(data_root, subject_id_clean, "headmodel", f"d2c_{subject_id_clean}", "dti_results_T1space", "DTI_conf_tensor.nii.gz"),
            os.path.join(data_root, subject_id_full, "headmodel", "dti_results", "DTI_conf_tensor.nii.gz"),
            os.path.join(data_root, subject_id_clean, "headmodel", "dti_results", "DTI_conf_tensor.nii.gz")
        ]
        
        tensor_file_path = None
        for path in tensor_paths:
            if os.path.exists(path):
                tensor_file_path = path
                print(f"Found DTI tensor file: {tensor_file_path}")
                break
                
        if tensor_file_path is None:
            print(f"WARNING: DTI tensor file not found. Checked paths: {tensor_paths}")
            print(f"Will attempt to find it during processing.")

        # Add DTI paths to the dictionary
        paths['dti_data_path'] = dti_node_data_path
        paths['dti_mesh_path'] = roi_mesh_path  # Use same ROI mesh
        paths['dti_output_dir'] = dti_output_dir
        if tensor_file_path:
            paths['dti_tensor_file'] = tensor_file_path
        
        print(f"Using DTI node data path: {paths['dti_data_path']}")
        print(f"Using DTI mesh path: {paths['dti_mesh_path']}")
        if tensor_file_path:
            print(f"Using DTI tensor file: {paths['dti_tensor_file']}")
    else:  # conductivity mode
        # Add path for conductivity output dir
        paths['mri_output_dir'] = os.path.join(base_dir, "MRI_arrays")
        print("Conductivity mode selected. No DTI paths derived.")
    # --- End Conditional DTI Paths ---

    return paths



from scipy.interpolate import RegularGridInterpolator  # For DTI interpolation

# Add this function for DTI processing
def process_dti_tensor_data(subject_id, data_root, target_coordinate, region_dimensions, bin_size, 
                           debug_hook=None, resource_monitor=None):
    """
    Process DTI tensor data following the provided code snippet approach:
    1. Load raw tensor file using nibabel
    2. Build 3x3 symmetric tensor at each point
    3. Map to the voxel grid using the existing VoxelMapper
    
    Args:
        subject_id: Subject ID (with or without 'sub-' prefix)
        data_root: Root directory for data
        target_coordinate: Target coordinate for transformation
        region_dimensions: Dimensions of the region of interest
        bin_size: Size of bins for voxelization
        debug_hook: Optional debug hook
        resource_monitor: Optional resource monitor
        
    Returns:
        Processed tensor data mapped to the voxel grid and processing time
    """
    print(f"\nProcessing DTI tensor data for subject {subject_id}")
    start_time = time.time()
    
    # Format subject ID consistently
    subject_id_clean = subject_id.replace("sub-", "")
    subject_id_full = f"sub-{subject_id_clean}"
    
    # Generate possible paths for the DTI tensor file
    tensor_paths = [
        os.path.join(data_root, subject_id_full, "headmodel", f"d2c_{subject_id_full}", "dti_results_T1space", "DTI_conf_tensor.nii.gz"),
        os.path.join(data_root, subject_id_clean, "headmodel", f"d2c_{subject_id_clean}", "dti_results_T1space", "DTI_conf_tensor.nii.gz"),
        os.path.join(data_root, subject_id_full, "headmodel", "dti_results", "DTI_conf_tensor.nii.gz"),
        os.path.join(data_root, subject_id_clean, "headmodel", "dti_results", "DTI_conf_tensor.nii.gz")
    ]
    
    # Find the tensor file
    tensor_file = None
    for path in tensor_paths:
        if os.path.exists(path):
            tensor_file = path
            print(f"Found DTI tensor file: {tensor_file}")
            break
    
    if tensor_file is None:
        raise FileNotFoundError(f"Could not find DTI tensor file for subject {subject_id}. Checked {tensor_paths}")
    
    # Load the tensor file using nibabel
    print(f"Loading DTI tensor file: {tensor_file}")
    sigma_img = nib.load(tensor_file)
    sigma_data = sigma_img.get_fdata()
    shape = sigma_data.shape[:3]
    print(f"Loaded DTI tensor data with shape: {sigma_data.shape}")
    
    # Generate grid of voxel indices
    print("Generating voxel indices and coordinates...")
    I, J, K = np.meshgrid(
        np.arange(shape[0]), 
        np.arange(shape[1]), 
        np.arange(shape[2]), 
        indexing='ij'
    )
    voxel_indices = np.stack([I, J, K, np.ones_like(I)], axis=-1)  # shape: (X, Y, Z, 4)
    coords = voxel_indices @ sigma_img.affine.T  # shape: (X, Y, Z, 4)
    coords = coords[..., :3]  # Drop homogeneous coordinate
    
    # Build symmetric tensor
    print("Building symmetric tensors...")
    anisotropic_cond_vals = np.zeros(shape + (3, 3))
    dxx = sigma_data[..., 0]
    dyy = sigma_data[..., 1]
    dzz = sigma_data[..., 2]
    dxy = sigma_data[..., 3]
    dxz = sigma_data[..., 4]
    dyz = sigma_data[..., 5]
    anisotropic_cond_vals[..., 0, 0] = dxx
    anisotropic_cond_vals[..., 1, 1] = dyy
    anisotropic_cond_vals[..., 2, 2] = dzz
    anisotropic_cond_vals[..., 0, 1] = anisotropic_cond_vals[..., 1, 0] = dxy
    anisotropic_cond_vals[..., 0, 2] = anisotropic_cond_vals[..., 2, 0] = dxz
    anisotropic_cond_vals[..., 1, 2] = anisotropic_cond_vals[..., 2, 1] = dyz
    
    # Now we have:
    # 1. Full 3D coordinates for each voxel in world space: coords
    # 2. Tensor data for each voxel: anisotropic_cond_vals
    
    # Let's reshape the tensor data to a more manageable format:
    # From (X, Y, Z, 3, 3) to (N, 9) where N is the number of voxels
    print("Reshaping tensor data...")
    flattened_coords = coords.reshape(-1, 3)  # (N, 3)
    
    # Get upper triangular values (6 unique components)
    tensor_components = np.zeros((np.prod(shape), 6))
    tensor_components[:, 0] = dxx.flatten()  # xx
    tensor_components[:, 1] = dyy.flatten()  # yy
    tensor_components[:, 2] = dzz.flatten()  # zz
    tensor_components[:, 3] = dxy.flatten()  # xy
    tensor_components[:, 4] = dxz.flatten()  # xz
    tensor_components[:, 5] = dyz.flatten()  # yz
    
    # Create pipeline context for VoxelMapper
    from utils.state.context import TMSPipelineContext
    pipeline_context = TMSPipelineContext(
        dependencies={}, config={}, subject_id=subject_id, 
        data_root_path=data_root,
        normalization_method="minmax", output_shape=(bin_size, bin_size, bin_size),
        debug_mode=debug_hook is not None, pipeline_mode="mri_dti",
        experiment_phase="preprocessing"
    )
    
    # We need to load the mesh to get gm_point and skin_vec for rotation
    print("Loading subject mesh for rotation parameters...")
    try:
        # Find mesh file
        from simnibs import mesh_io
        mesh_path = os.path.join(data_root, subject_id_full, "headmodel", f"{subject_id_full}.msh")
        if not os.path.exists(mesh_path):
            alt_path = os.path.join(data_root, subject_id_clean, "headmodel", f"{subject_id_clean}.msh")
            if os.path.exists(alt_path):
                mesh_path = alt_path
        
        msh = mesh_io.read_msh(mesh_path)
        print(f"Loaded mesh with {len(msh.nodes.node_coord)} nodes")
        
        # Load ROI center for gm_point and skin_vec
        roi_center_path = os.path.join(data_root, subject_id_full, "experiment", f"{subject_id_full}_roi_center.mat")
        if not os.path.exists(roi_center_path):
            alt_path = os.path.join(data_root, subject_id_clean, "experiment", f"{subject_id_clean}_roi_center.mat")
            if os.path.exists(alt_path):
                roi_center_path = alt_path
        
        import scipy.io as sio
        roi_data = sio.loadmat(roi_center_path, struct_as_record=False, squeeze_me=True)
        roi_center = roi_data['roi_center']
        
        gm_point = np.asarray(roi_center.gm).flatten().astype(float)
        skin_vec = np.asarray(roi_center.skin_vec).flatten().astype(float)
        
        print(f"Using gm_point: {gm_point}")
        print(f"Using skin_vec: {skin_vec}")
        
    except Exception as e:
        print(f"Error loading mesh data: {e}")
        traceback.print_exc()
        raise
    
    # Now create a spatial transform for the DTI data
    # This is a bit different than the normal workflow since we're not starting with mesh nodes
    # We already have coordinates in world space, but we need to:
    # 1. Create a rotation matrix to align skin_vec with Y-axis
    # 2. Apply this rotation to the coordinates
    # 3. Translate to center gm_point at the target coordinate
    
    print("Creating spatial transformation...")
    from data.transformations.voxel_mapping import VoxelMapper
    
    # Create a temporary VoxelMapper to compute the transformation matrix
    temp_mapper = VoxelMapper(
        context=pipeline_context, 
        mesh_path=mesh_path,  # Just needed for initialization
        gm_point=gm_point,
        skin_vec=skin_vec,
        target_coordinate=target_coordinate,
        region_dimensions=region_dimensions,
        bin_size=bin_size
    )
    
    # Preprocess to compute the transformation matrix
    print("Computing transform matrix...")
    temp_mapper.preprocess(save_path=None)  # We don't need to save this
    transform_matrix = temp_mapper.get_current_transform_matrix()
    
    # Apply the transformation to the coordinates
    print("Applying transformation to coordinates...")
    homogeneous_coords = np.ones((flattened_coords.shape[0], 4))
    homogeneous_coords[:, :3] = flattened_coords
    transformed_homogeneous = np.dot(homogeneous_coords, transform_matrix.T)
    transformed_coords = transformed_homogeneous[:, :3]
    
    # Create a grid for the transformed coordinates
    print("Creating voxel grid...")
    x_bounds = (target_coordinate[0] - region_dimensions[0]/2, target_coordinate[0] + region_dimensions[0]/2)
    y_bounds = (target_coordinate[1] - region_dimensions[1]/2, target_coordinate[1] + region_dimensions[1]/2)
    z_bounds = (target_coordinate[2] - region_dimensions[2]/2, target_coordinate[2] + region_dimensions[2]/2)
    
    x_bins = np.linspace(x_bounds[0], x_bounds[1], bin_size + 1)
    y_bins = np.linspace(y_bounds[0], y_bounds[1], bin_size + 1)
    z_bins = np.linspace(z_bounds[0], z_bounds[1], bin_size + 1)
    
    # Filter points within ROI bounds
    print("Filtering points within ROI...")
    roi_mask = (
        (transformed_coords[:, 0] >= x_bounds[0]) & (transformed_coords[:, 0] <= x_bounds[1]) &
        (transformed_coords[:, 1] >= y_bounds[0]) & (transformed_coords[:, 1] <= y_bounds[1]) &
        (transformed_coords[:, 2] >= z_bounds[0]) & (transformed_coords[:, 2] <= z_bounds[1])
    )
    
    roi_coords = transformed_coords[roi_mask]
    roi_tensors = tensor_components[roi_mask]
    
    print(f"Points within ROI: {np.sum(roi_mask)} out of {len(transformed_coords)}")
    
    # Now map the tensor components to the target grid
    print("Mapping tensor components to voxel grid...")
    
    # Initialize output tensor grid (bin_size, bin_size, bin_size, 6)
    tensor_grid = np.zeros((bin_size, bin_size, bin_size, 6))
    
    # Count number of points in each voxel for averaging
    voxel_counts = np.zeros((bin_size, bin_size, bin_size), dtype=int)
    
    # Assign points to voxels and accumulate tensor values
    for i in range(len(roi_coords)):
        # Determine voxel indices for this point
        x_idx = np.clip(np.digitize(roi_coords[i, 0], x_bins) - 1, 0, bin_size - 1)
        y_idx = np.clip(np.digitize(roi_coords[i, 1], y_bins) - 1, 0, bin_size - 1)
        z_idx = np.clip(np.digitize(roi_coords[i, 2], z_bins) - 1, 0, bin_size - 1)
        
        # Add tensor components to the accumulator
        tensor_grid[x_idx, y_idx, z_idx] += roi_tensors[i]
        voxel_counts[x_idx, y_idx, z_idx] += 1
    
    # Average the accumulated tensor components
    print("Computing average tensor per voxel...")
    valid_voxels = voxel_counts > 0
    
    # Replace loop with broadcasting for efficiency
    for c in range(6):  # For each tensor component
        component = tensor_grid[..., c]
        component[valid_voxels] /= voxel_counts[valid_voxels]
        tensor_grid[..., c] = component
    
    print(f"Final tensor grid shape: {tensor_grid.shape}")
    print(f"Occupied voxels: {np.sum(valid_voxels)} out of {bin_size**3}")
    
    # Create voxel validity mask
    voxel_mask = voxel_counts > 0
    
    dti_time = time.time() - start_time
    print(f"Processed DTI tensor data in {dti_time:.2f} seconds")
    
    return tensor_grid, voxel_mask, dti_time

def load_subject_data(sub_path: str) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Loads mesh, gm point, and gm normal vector for a subject from FIXED expected paths.

    This function assumes the following structure relative to sub_path:
    - Mesh: <sub_path>/headmodel/<sub_id>.msh
    - ROI:  <sub_path>/experiment/<sub_id>_roi_center.mat
    It raises errors immediately if files are missing or malformed.

    Args:
        sub_path: Path to the subject's main directory (e.g., /path/to/data/sub-001).
                  The directory name itself should be the subject ID (e.g., 'sub-001').

    Returns:
        Tuple of (mesh object, gm_point (3,) ndarray, skin_vec (3,) ndarray).

    Raises:
        FileNotFoundError: If the mesh or ROI .mat file is not found at the expected location.
        ValueError: If the subject path doesn't seem to represent a valid subject ID,
                    or if loaded gm_point/skin_vec have incorrect shapes.
        KeyError: If 'roi_center' key is missing in the .mat file.
        AttributeError: If 'gm' or 'skin_vec' attributes are missing within the 'roi_center' struct.
        RuntimeError: For other mesh/mat file loading or parsing errors.
    """
    sub_id = os.path.basename(sub_path) # Get subject ID like 'sub-001' from the path

    # --- Basic validation of the subject ID derived from the path ---
    # Add more specific checks if needed (e.g., must start with 'sub-')
    if not sub_id:
        raise ValueError(f"Could not derive subject ID from path: {sub_path}")
    print(f"Debug: Loading data for subject ID: {sub_id} from path: {sub_path}")

    # --- 1. Load Mesh (Strict Path) ---
    try:
        mesh_filename = f"{sub_id}.msh"
        msh_path = _get_required_file_path("mesh", sub_path, 'headmodel', mesh_filename)
        msh = mesh_io.read_msh(msh_path)
        print(f"Debug: Successfully loaded mesh from {msh_path}")
    except FileNotFoundError:
        raise # Re-raise the specific error from the helper
    except Exception as e:
        # Catch potential errors from mesh_io.read_msh
        raise RuntimeError(f"Failed to load or parse mesh file {msh_path}: {e}") from e

    # --- 2. Load ROI Data from .mat file (Strict Path) ---
    try:
        roi_filename = f"{sub_id}_roi_center.mat"
        roi_center_file = _get_required_file_path("ROI", sub_path, 'experiment', roi_filename)

        # Load using struct_as_record=False for object-like access
        mat_data = sio.loadmat(roi_center_file, struct_as_record=False, squeeze_me=True)
        print(f"Debug: Successfully loaded .mat file: {roi_center_file}")

        # Check for 'roi_center' key - raise KeyError if missing
        if 'roi_center' not in mat_data:
            raise KeyError(f"'roi_center' key not found in {roi_center_file}")

        roi_struct = mat_data['roi_center']

        # Check for 'gm' and 'skin_vec' attributes - raise AttributeError if missing
        if not hasattr(roi_struct, 'gm'):
            raise AttributeError(f"'gm' attribute not found in 'roi_center' struct within {roi_center_file}")
        if not hasattr(roi_struct, 'skin_vec'):
            raise AttributeError(f"'skin_vec' attribute not found in 'roi_center' struct within {roi_center_file}")

        # Access fields, convert, and validate format/shape
        gm_point = np.asarray(roi_struct.gm).flatten().astype(float)
        skin_vec = np.asarray(roi_struct.skin_vec).flatten().astype(float)

        if gm_point.shape != (3,):
            raise ValueError(f"Extracted gm_point from {roi_center_file} has incorrect shape: {gm_point.shape}. Expected (3,).")
        if skin_vec.shape != (3,):
             raise ValueError(f"Extracted skin_vec from {roi_center_file} has incorrect shape: {skin_vec.shape}. Expected (3,).")

        print(f"Debug: Extracted gm_point: {gm_point}")
        print(f"Debug: Extracted skin_vec: {skin_vec}")

    except FileNotFoundError:
        raise # Re-raise specific error
    except (KeyError, AttributeError, ValueError) as e:
        # Re-raise specific data structure or content errors
        raise e
    except Exception as e:
        # Catch other potential scipy.io errors during loading/parsing
        raise RuntimeError(f"Failed to load or parse ROI data from {roi_center_file}: {e}") from e

    # --- 3. Return successfully loaded data ---
    return msh, gm_point, skin_vec



def check_orientation_data(subject_id, data_root):
    """Check if orientation data exists and load it.
    
    Args:
        subject_id: Subject ID
        data_root: Root directory for data
        
    Returns:
        Orientation data if it exists, None otherwise
    """
    # First try the global orientation file
    global_orientation_path = f"{data_root}/orientation_data.json"
    
    # Then fall back to subject-specific paths if needed
    subject_id_clean = subject_id.replace("sub-", "")
    subject_orientation_path = f"{data_root}/{subject_id}/experiment/orientation_data.json"
    alt_subject_path = f"{data_root}/{subject_id_clean}/experiment/orientation_data.json"
    
    # Try global path first, then subject-specific paths
    if os.path.exists(global_orientation_path):
        orientation_path = global_orientation_path
        print(f"Using global orientation data from {orientation_path}")
    else:
        print(f"ERROR: Orientation data not found at:")
        print(f"  - {global_orientation_path}")

        return None
    
    try:
        with open(orientation_path, 'r') as f:
            orientation_data = json.load(f)
            
        required_keys = ["original_center", "original_normal", "target_center", "target_normal"]
        for key in required_keys:
            if key not in orientation_data:
                print(f"ERROR: Orientation data is missing required key: {key}")
                return None
                
        print(f"Loaded orientation data from {orientation_path}")
        return orientation_data
    
    except Exception as e:
        print(f"ERROR: Failed to load orientation data: {str(e)}")
        return None

def load_mesh_and_roi(sub_path: str) -> Tuple[Any, np.ndarray]:
    """
    Load mesh and ROI center for a subject.
    
    Args:
        sub_path: Path to subject directory
        
    Returns:
        Tuple of (mesh, roi_center)
    """
    from simnibs import mesh_io
    import scipy.io as sio
    
    sub = sub_path.split('/')[-1]
    msh_name = sub + ".msh"
    msh_path = os.path.join(sub_path, 'headmodel', msh_name)
    
    # Try alternative paths if needed
    if not os.path.exists(msh_path):
        alt_paths = [
            os.path.join(sub_path, 'headmodel', f"{sub.replace('sub-', '')}.msh"),
            os.path.join(sub_path, msh_name),
            os.path.join(sub_path, f"{sub.replace('sub-', '')}.msh")
        ]
        for path in alt_paths:
            if os.path.exists(path):
                msh_path = path
                break
                
    if not os.path.exists(msh_path):
        raise FileNotFoundError(f"Cannot find mesh file for {sub}")
    
    # Load mesh
    msh = mesh_io.read_msh(msh_path)

    # Load ROI center
    roi_center_path = os.path.join(sub_path, 'experiment')
    roi_center_name = sub + "_roi_center.mat"
    roi_center_file = os.path.join(roi_center_path, roi_center_name)
    
    # Try alternative paths if needed
    if not os.path.exists(roi_center_file):
        alt_paths = [
            os.path.join(roi_center_path, f"{sub.replace('sub-', '')}_roi_center.mat"),
            os.path.join(sub_path, f"{sub}_roi_center.mat"),
            os.path.join(sub_path, "experiment", f"{sub.replace('sub-', '')}_roi_center.mat")
        ]
        for path in alt_paths:
            if os.path.exists(path):
                roi_center_file = path
                break
                
    if not os.path.exists(roi_center_file):
        raise FileNotFoundError(f"Cannot find ROI center file for {sub}")
    
    roi_center = sio.loadmat(roi_center_file)['roi_center']
    
    return msh, roi_center

def process_and_stack_fields(args):
    """
    Process E-fields and dAdt fields, voxelize them by rotating skin_vec to
    align with the Y-axis, align MRI conductivity, and stack the results.
    """
    subject_id = args.subject         # e.g., sub-001
    bin_size = args.bin_size
    n_processes = args.processes      # Note: VoxelMapper currently runs sequentially in preprocess
    clean_intermediate = args.clean
    data_root = args.data_root

    # Parse output formats
    formats = args.formats.split(',')
    save_torch = 'torch' in formats
    save_numpy = 'numpy' in formats
    save_pickle = 'pickle' in formats

    # --- Load Subject Specific Data (mesh, gm_point, skin_vec) ---
    try:
        subject_path = os.path.join(data_root, subject_id)
        if not os.path.exists(subject_path):
             # Try alternative name like '001' if 'sub-001' not found
             clean_id = subject_id.replace("sub-", "")
             alt_subject_path = os.path.join(data_root, clean_id)
             if os.path.exists(alt_subject_path):
                  subject_path = alt_subject_path
                  print(f"Info: Using alternative subject path: {alt_subject_path}")
             else:
                  # Try another common pattern if subject_id was already clean
                  alt_subject_path_prefix = os.path.join(data_root, f"sub-{clean_id}")
                  if os.path.exists(alt_subject_path_prefix):
                       subject_path = alt_subject_path_prefix
                       print(f"Info: Using alternative subject path: {alt_subject_path_prefix}")
                  else:
                       raise FileNotFoundError(f"Subject directory not found at {os.path.join(data_root, subject_id)}, {alt_subject_path}, or {alt_subject_path_prefix}")

        print(f"Debug: Using subject data path: {subject_path}")
        # Call the modified loading function (ensure it's defined in this file or imported)
        msh, gm_point, skin_vec = load_subject_data(subject_path)

        if gm_point is None or skin_vec is None:
             print(f"ERROR: Could not load gm_point or skin_vec for subject {subject_id}. Cannot proceed with rotation.")
             # Optionally: return or raise a specific error instead of sys.exit
             return None # Indicate failure for this subject
             # sys.exit(1)

    except FileNotFoundError as e:
         print(f"ERROR: Required file not found. {e}")
         # Optionally: return or raise
         return None # Indicate failure
         # sys.exit(1)
    except Exception as e:
         print(f"ERROR: Failed to load subject data (mesh/gm_point/skin_vec) for {subject_id}: {e}")
         traceback.print_exc()
         # Optionally: return or raise
         return None # Indicate failure
         # sys.exit(1)
    # --- End Load Subject Specific Data ---


    # --- Derive Paths for Field Data & Output Directories ---
    try:
        paths = derive_paths(subject_id, data_root, args.mri_mode, args.dti_file_pattern)
        efield_mesh_path = paths['efield_mesh_path'] # VoxelMapper still needs mesh *path* internally
        dadt_mesh_path = paths['dadt_mesh_path']
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: Could not derive necessary paths for {subject_id}: {e}")
        return None # Indicate failure

    # Extract paths
    efield_data_path = paths['efield_data_path']
    efield_output_dir = paths['efield_output_dir']
    dadt_data_path = paths['dadt_data_path']
    dadt_output_dir = paths['dadt_output_dir']
    stacked_output_dir_base = paths['stacked_output_dir'] # Base dir for stacked results
    base_dir = paths['base_dir']

    # Set or create output directory (potentially override from args)
    if args.output_dir:
        stacked_output_dir = args.output_dir
    else:
        # Append transformation type to default output dir
        stacked_output_dir = f"{stacked_output_dir_base}"

    os.makedirs(stacked_output_dir, exist_ok=True)
    os.makedirs(efield_output_dir, exist_ok=True) # Intermediate dirs
    os.makedirs(dadt_output_dir, exist_ok=True)   # Intermediate dirs

    # Create subdirectories for output formats
    torch_dir, numpy_dir, pickle_dir = None, None, None
    if save_torch:
        os.makedirs(os.path.join(efield_output_dir, "torch"), exist_ok=True)
        os.makedirs(os.path.join(dadt_output_dir, "torch"), exist_ok=True)
        torch_dir = os.path.join(stacked_output_dir, "torch")
        os.makedirs(torch_dir, exist_ok=True)
    if save_numpy:
        os.makedirs(os.path.join(efield_output_dir, "numpy"), exist_ok=True)
        os.makedirs(os.path.join(dadt_output_dir, "numpy"), exist_ok=True)
        numpy_dir = os.path.join(stacked_output_dir, "numpy")
        os.makedirs(numpy_dir, exist_ok=True)
    if save_pickle:
        os.makedirs(os.path.join(efield_output_dir, "pickle"), exist_ok=True)
        os.makedirs(os.path.join(dadt_output_dir, "pickle"), exist_ok=True)
        pickle_dir = os.path.join(stacked_output_dir, "pickle")
        os.makedirs(pickle_dir, exist_ok=True)
    # --- End Path Setup ---


    # --- Setup Contexts, Monitors, Hooks ---
    start_time_total = time.time()
    debug_context = PipelineDebugContext(
        verbosity_level=args.debug_level, memory_limit=8000,
        sampling_rate=0.1 if args.debug_level > 0 else 0.0
    )
    debug_hook = PipelineDebugHook(debug_context)
    resource_monitor = ResourceMonitor()
    resource_monitor.start_monitoring()
    pipeline_context = TMSPipelineContext(
        dependencies={}, config={}, subject_id=subject_id, data_root_path=data_root,
        normalization_method="minmax", dadt_scaling_factor=1.0e-6,
        output_shape=(bin_size, bin_size, bin_size), debug_mode=args.debug_level > 0,
        pipeline_mode="mri_efield", experiment_phase="preprocessing"
    )
    # --- End Setup ---


    print(f"Processing data for subject {subject_id} with bin size {bin_size}")
    print(f"Transformation: Rotate skin_vec ({skin_vec.round(3)}) to [0,1,0] around gm_point ({gm_point.round(3)})")
    print("="*80)


    # --- Initialize Field Processor ---
    field_config = FieldProcessingConfig(
        bin_size=bin_size, n_processes=n_processes,
        save_torch=True, save_numpy=save_numpy, save_pickle=save_pickle,
        clean_intermediate=clean_intermediate
    )
    field_processor = FieldProcessor(
        context=pipeline_context, config=field_config,
        debug_hook=debug_hook, resource_monitor=resource_monitor
    ) # Note: FieldProcessor itself isn't heavily used here, VoxelMapper does the work
    target_coord = np.array([23.0, 32, 23.0])
    roi_dims = (46.0, 37.0, 46.0) # Example ROI dimensions

    efield_mapper = VoxelMapper(
        context=pipeline_context, mesh_path=efield_mesh_path,
        gm_point=gm_point,              # Loaded from subject data
        skin_vec=skin_vec,            # Loaded from subject data
        target_coordinate=target_coord, # Explicitly set target
        region_dimensions=roi_dims,     # Explicitly set ROI dimensions
        bin_size=bin_size,
        debug_hook=debug_hook, resource_monitor=resource_monitor
    )
    # Similarly for dadt_mapper
    dadt_mapper = VoxelMapper(
        context=pipeline_context, mesh_path=dadt_mesh_path, # Use correct mesh path
        gm_point=gm_point,              # Same gm_point
        skin_vec=skin_vec,            # Same skin_vec
        target_coordinate=target_coord, # Same target
        region_dimensions=roi_dims,     # Same ROI dimensions
        bin_size=bin_size,
        debug_hook=debug_hook, resource_monitor=resource_monitor
    )
    # --- End Voxel Mapper Init ---


    # --- STEP 1: Process E-fields ---
    print(f"\nSTEP 1: Processing E-fields for {subject_id}")
    start_time_efield = time.time()
    try:
        efield_data = np.load(efield_data_path)
        print(f"Debug - Loaded E-field data shape: {efield_data.shape}")

        # Preprocess: Load mesh, calculate rotation, transform nodes, determine ROI, create mapping
        efield_stats = efield_mapper.preprocess(
            save_path=os.path.join(efield_output_dir, f"efield_preprocessing_b{bin_size}.pkl")
        )
        print(f"Debug - E-field preprocessing stats: {efield_stats}")


        # Process each E-field configuration using the preprocessed mapper
        efield_grids = []
        n_efields = efield_data.shape[0]
        print(f"Mapping {n_efields} E-field configurations to voxel grid...")
        for i in tqdm(range(n_efields), desc="Processing E-fields"):
            # process_field uses the stored mapping and transformed nodes
            efield_grid = efield_mapper.process_field(efield_data[i], output_grid=True)
            efield_grids.append(efield_grid)

            # Save intermediate rotated/voxelized E-field grids
            if save_torch:
                torch.save(torch.from_numpy(efield_grid).float(),
                           os.path.join(efield_output_dir, "torch", f"efield_{i}_b{bin_size}.pt"))
            if save_numpy:
                np.save(os.path.join(efield_output_dir, "numpy", f"efield_{i}_b{bin_size}.npy"),
                       efield_grid)

        efield_time = time.time() - start_time_efield
        print(f"Processed {n_efields} E-fields in {efield_time:.2f} seconds")

    except Exception as e:
        print(f"ERROR during E-field processing for {subject_id}: {e}")
        traceback.print_exc()
        resource_monitor.stop_monitoring()
        return None # Indicate failure
    # --- End STEP 1 ---


      
# # --- STEP 2: Process dA/dt fields ---
    print(f"\nSTEP 2: Processing dA/dt fields for {subject_id}")
    start_time_dadt = time.time()
    try:
        # Import the interpolation function
        from data.transformations.element_to_node import interpolate_element_to_node
        
        dadt_data = np.load(dadt_data_path)
        print(f"Debug - Loaded dA/dt data shape: {dadt_data.shape}") # Will be (N_sim, N_elem, 3)

        # --- MESH LOADING FOR INTERPOLATION ---
        print(f"Debug - Loading mesh for element-to-node interpolation...")

        # Load the mesh for dA/dt fields
        try:
            from simnibs import mesh_io # Ensure imported
            dadt_mesh = mesh_io.read_msh(dadt_mesh_path)
            num_dadt_nodes = len(dadt_mesh.nodes.node_coord)
            num_elements = len(dadt_mesh.elm.node_number_list)
            print(f"Debug - Loaded mesh with {num_dadt_nodes} nodes and {num_elements} elements.")
        except ImportError:
            print("ERROR: simnibs.mesh_io needed for dA/dt processing but not found.")
            raise
        except Exception as e:
            print(f"ERROR: Failed to load dA/dt mesh {dadt_mesh_path}: {e}")
            raise

        # For each simulation, create a node-based field
        dadt_data_node_based = []
        num_simulations = dadt_data.shape[0]

        # Option 1: If you have node-based field data available in another file, load it here
        # Additional path for node-based field data if available
        dadt_node_fields_path = os.path.join(os.path.dirname(dadt_mesh_path), f"{subject_id}_roi_node_fields.npy")
        if os.path.exists(dadt_node_fields_path):
            print(f"Debug - Found node-based field data at {dadt_node_fields_path}, loading directly...")
            node_field_data = np.load(dadt_node_fields_path)
            if node_field_data.shape[0] == num_simulations and node_field_data.shape[1] == num_dadt_nodes:
                dadt_data_node_based = [node_field_data[i] for i in range(num_simulations)]
                print(f"Debug - Loaded {num_simulations} node-based fields directly.")
        else:
            # Option 2: Perform element-to-node interpolation
            print(f"Debug - No direct node fields found. Interpolating from element-based data...")
            print(f"Debug - dA/dt data is for {dadt_data.shape[1]} elements, mesh has {num_elements} elements")
            
            # Get element types if available
            element_types = None
            if hasattr(dadt_mesh.elm, 'elm_type'):
                element_types = dadt_mesh.elm.elm_type
                unique_types = np.unique(element_types)
                print(f"Debug - Element types in mesh: {unique_types}")
            
            # Interpolate element-based data to node-based data
            node_field_data = interpolate_element_to_node(
                dadt_data,                      # Element-based field data
                num_dadt_nodes,                 # Number of nodes in the mesh
                dadt_mesh.elm.node_number_list, # Element-to-node connectivity
                element_types=element_types,    # Element type information
                one_indexed=True,               # SimNIBS uses 1-indexed nodes
                verbose=True
            )
            
            # Save interpolated data for future use
            print(f"Debug - Saving interpolated node-based fields to {dadt_node_fields_path}")
            os.makedirs(os.path.dirname(dadt_node_fields_path), exist_ok=True)
            np.save(dadt_node_fields_path, node_field_data)
            
            # Use the interpolated data
            dadt_data_node_based = [node_field_data[i] for i in range(num_simulations)]
            print(f"Debug - Created {num_simulations} node-based fields through interpolation.")

        print(f"Debug - Final dadt_data_node_based is a list of {len(dadt_data_node_based)} arrays, each with shape ({len(dadt_data_node_based[0])}, 3)")
        # --- INTERPOLATION END ---


        # Preprocess: Load mesh, calculate rotation, transform nodes, determine ROI, create mapping
        dadt_stats = dadt_mapper.preprocess(
            save_path=os.path.join(dadt_output_dir, f"dadt_preprocessing_b{bin_size}.pkl")
        )
        print(f"Debug - dA/dt preprocessing stats: {dadt_stats}")

        # Process each dA/dt configuration using the preprocessed mapper
        dadt_grids = []
        n_dadts = len(dadt_data_node_based) # Number of successfully interpolated fields
        print(f"Mapping {n_dadts} INTERPOLATED dA/dt configurations to voxel grid...")
        for i in tqdm(range(n_dadts), desc="Processing dA/dt fields"):
            # Use the interpolated node-based data
            dadt_grid = dadt_mapper.process_field(dadt_data_node_based[i], output_grid=True)
            dadt_grids.append(dadt_grid)

            # Save intermediate rotated/voxelized dA/dt grids (if desired)
            if save_torch:
                torch.save(torch.from_numpy(dadt_grid).float(),
                            os.path.join(dadt_output_dir, "torch", f"dadt_{i}_b{bin_size}.pt"))
            if save_numpy:
                np.save(os.path.join(dadt_output_dir, "numpy", f"dadt_{i}_b{bin_size}.npy"),
                        dadt_grid)

        dadt_time = time.time() - start_time_dadt
        print(f"Processed {n_dadts} dA/dt fields in {dadt_time:.2f} seconds")

    except Exception as e:
        print(f"ERROR during dA/dt processing for {subject_id}: {e}")
        traceback.print_exc()
        resource_monitor.stop_monitoring()
        return None # Indicate failure
    # --- End STEP 2 ---

    

    # --- STEP 3: Load and Process MRI Data (Conductivity or DTI) ---
    print(f"\nSTEP 3: Loading and Processing {args.mri_mode.upper()} data for {subject_id}")
    start_time_mri = time.time()

    # Create output dir based on MRI mode
    if args.mri_mode == 'conductivity':
        mri_output_dir = os.path.join(base_dir, "MRI_arrays")
    else:  # dti mode
        mri_output_dir = paths['dti_output_dir']

    os.makedirs(os.path.join(mri_output_dir, "torch"), exist_ok=True)
    os.makedirs(os.path.join(mri_output_dir, "numpy"), exist_ok=True)
    cache_dir = os.path.join(base_dir, "cached_data")
    os.makedirs(cache_dir, exist_ok=True)

    # Choose processing method based on MRI mode
    if args.mri_mode == 'conductivity':
        cache_path = os.path.join(cache_dir, f"mri_conductivity_b{bin_size}.pt")
        
        try:
            # Original conductivity processing code
            if os.path.exists(cache_path) and not args.clean:
                print(f"Loading cached ROTATED MRI conductivity data from {cache_path}")
                mri_data = torch.load(cache_path).numpy()
            else:
                print("Cached rotated MRI conductivity not found or clean requested. Processing from mesh...")
                
                # Define conductivity tag to value mapping
                cond_map = {1: 0.126, 2: 0.275, 3: 1.654, 4: 0.01, 5: 0.465,
                            6: 0.5, 7: 0.008, 8: 0.025, 9: 0.6, 10: 0.16,
                            100: 29.4, 500: 1.0}
                default_conductivity = 0.0
                
                # Check what tissue tags exist in the mesh
                if hasattr(msh.elm, 'tag1'):
                    unique_tags = np.unique(msh.elm.tag1)
                    print(f"Debug: Found unique tissue tags in mesh: {unique_tags}")
                
                # 1. Create VoxelMapper with the full mesh
                print("Debug: Setting up VoxelMapper for conductivity data...")
                msh_path = getattr(msh, 'fn', None)  # Get mesh filename if available
                if msh_path is None:
                    print("Debug: Mesh object doesn't have filename attribute. Using temporary path.")
                    msh_path = os.path.join(base_dir, f"{subject_id}_head.msh")
                    
                    # If mesh doesn't have a path, save it temporarily
                    try:
                        from simnibs import mesh_io
                        mesh_io.write_msh(msh, msh_path)
                        print(f"Debug: Saved mesh to temporary file: {msh_path}")
                    except Exception as e:
                        print(f"Warning: Could not save mesh to temporary file: {e}")
                        # Create a fallback - this is only needed for VoxelMapper initialization
                        # It will use our directly provided nodes anyway
                        with open(msh_path, 'w') as f:
                            f.write("Placeholder")
                
                conductivity_mapper = VoxelMapper(
                    context=pipeline_context,
                    mesh_path=msh_path,
                    gm_point=gm_point,
                    skin_vec=skin_vec,
                    target_coordinate=target_coord,
                    region_dimensions=roi_dims,
                    bin_size=bin_size,
                    debug_hook=debug_hook,
                    resource_monitor=resource_monitor
                )
                
                # Override with direct node data if needed
                conductivity_mapper.original_nodes = msh.nodes.node_coord
                conductivity_mapper.total_nodes = len(msh.nodes.node_coord)
                
                # 2. Preprocess to transform nodes and create voxel mapping
                print("Debug: Preprocessing for conductivity...")
                conductivity_mapper.preprocess(
                    save_path=os.path.join(mri_output_dir, f"mri_preprocessing_b{bin_size}.pkl")
                )
                
                # 3. Create node-based conductivity values from mesh elements
                print("Debug: Creating node-based conductivity values...")
                
                # Get nodes that are within ROI after transformation
                roi_node_indices = np.where(conductivity_mapper.node_mask)[0]
                num_roi_nodes = len(roi_node_indices)
                print(f"Debug: Found {num_roi_nodes} nodes within ROI after transformation")
                
                # Initialize arrays to track sum and count of conductivity values per node
                node_cond_sum = np.zeros(conductivity_mapper.total_nodes, dtype=float)
                node_cond_count = np.zeros(conductivity_mapper.total_nodes, dtype=int)
                
                # Process all mesh elements - don't filter by type
                element_count = len(msh.elm.node_number_list)
                print(f"Debug: Processing conductivity from {element_count} mesh elements...")
                
                for elm_idx in range(element_count):
                    # Get conductivity tag for this element
                    if hasattr(msh.elm, 'tag1') and elm_idx < len(msh.elm.tag1):
                        cond_tag = msh.elm.tag1[elm_idx]
                        cond_value = cond_map.get(cond_tag, default_conductivity)
                        
                        # Get nodes for this element
                        node_indices = msh.elm.node_number_list[elm_idx]
                        
                        # Add conductivity value to each node of this element
                        for node_idx in node_indices:
                            # Adjust index if 1-based (SimNIBS sometimes uses 1-indexed nodes)
                            node_idx_0based = node_idx - 1 if node_idx >= conductivity_mapper.total_nodes else node_idx
                            
                            # Only count if within valid range
                            if 0 <= node_idx_0based < conductivity_mapper.total_nodes:
                                node_cond_sum[node_idx_0based] += cond_value
                                node_cond_count[node_idx_0based] += 1
                
                # Calculate average conductivity per node (avoid division by zero)
                node_conductivity = np.zeros(conductivity_mapper.total_nodes, dtype=float)
                valid_nodes = node_cond_count > 0
                node_conductivity[valid_nodes] = node_cond_sum[valid_nodes] / node_cond_count[valid_nodes]
                
                # Report on nodes with no conductivity
                num_missing = np.sum(node_cond_count == 0)
                if num_missing > 0:
                    print(f"Warning: {num_missing} nodes ({num_missing/conductivity_mapper.total_nodes*100:.1f}%) have no conductivity information")
                
                print(f"Debug: Node conductivity range: Min={np.min(node_conductivity[valid_nodes]):.3f}, Max={np.max(node_conductivity[valid_nodes]):.3f}")
                
                # 4. Process the node conductivity values using VoxelMapper
                print("Debug: Voxelizing conductivity values...")
                mri_data = conductivity_mapper.process_field(node_conductivity)
                
                # Save result to cache for future use
                torch.save(torch.from_numpy(mri_data).float(), cache_path)
                print(f"Saved ROTATED MRI conductivity to cache: {cache_path}")
                print(f"Debug: Final MRI conductivity grid shape: {mri_data.shape}")
                print(f"Debug: Value range: Min={np.min(mri_data):.3f}, Max={np.max(mri_data):.3f}")

            # Save final (rotated) MRI conductivity data to output formats
            if save_torch:
                torch.save(torch.from_numpy(mri_data).float(),
                        os.path.join(mri_output_dir, "torch", f"mri_conductivity_b{bin_size}.pt"))
            if save_numpy:
                np.save(os.path.join(mri_output_dir, "numpy", f"mri_conductivity_b{bin_size}.npy"),
                    mri_data)
            
            # Set the data type for stacking
            mri_type = 'conductivity'
            
        except Exception as e:
            print(f"ERROR: Failed to load or process MRI conductivity data: {str(e)}")
            traceback.print_exc()
            resource_monitor.stop_monitoring()
            return None  # Indicate failure

    else:  # DTI mode
        cache_path = os.path.join(cache_dir, f"mri_dti_tensor_b{bin_size}.pt")
        
        try:
            # Check for cached DTI data
            if os.path.exists(cache_path) and not args.clean:
                print(f"Loading cached ROTATED DTI tensor data from {cache_path}")
                mri_data = torch.load(cache_path).numpy()
                voxel_mask = np.any(mri_data != 0, axis=-1)  # Create mask from non-zero voxels
            else:
                print("Processing DTI tensor data...")
                # Process DTI tensor data
                mri_data, voxel_mask, _ = process_dti_tensor_data(
                    subject_id, 
                    data_root, 
                    target_coord, 
                    roi_dims, 
                    bin_size, 
                    debug_hook, 
                    resource_monitor
                )
                
                # Save to cache
                torch.save(torch.from_numpy(mri_data).float(), cache_path)
                print(f"Saved ROTATED DTI tensor data to cache: {cache_path}")
            
            # Save final processed DTI data to output formats
            if save_torch:
                torch.save(torch.from_numpy(mri_data).float(),
                        os.path.join(mri_output_dir, "torch", f"mri_dti_tensor_b{bin_size}.pt"))
            if save_numpy:
                np.save(os.path.join(mri_output_dir, "numpy", f"mri_dti_tensor_b{bin_size}.npy"),
                    mri_data)
                    
            # Set the data type for stacking
            mri_type = 'dti_tensor'
            
        except Exception as e:
            print(f"ERROR: Failed to load or process DTI data: {str(e)}")
            traceback.print_exc()
            resource_monitor.stop_monitoring()
            return None  # Indicate failure

    mri_time = time.time() - start_time_mri
    print(f"Processed MRI {args.mri_mode} data in {mri_time:.2f} seconds")

    # Replace the STEP 4 in process_and_stack_fields function with this code:

    # --- STEP 4: Stack the Results ---
    print(f"\nSTEP 4: Stacking ALIGNED {args.mri_mode.upper()} and dA/dt fields for {subject_id}")
    start_time_stacking = time.time()
    try:
        # Determine the number of fields to stack (minimum of E-field and dA/dt)
        num_fields_to_stack = min(len(efield_grids), len(dadt_grids))
        if len(efield_grids) != len(dadt_grids):
            print(f"Warning: Number of E-fields ({len(efield_grids)}) and dA/dt fields ({len(dadt_grids)}) doesn't match.")
            print(f"Stacking only the first {num_fields_to_stack} fields.")

        # Normalize MRI data (using method from context)
        mri_normalized = normalize_mri(mri_data, method=pipeline_context.normalization_method)
        print(f"Debug: Normalized MRI {args.mri_mode}. Shape: {mri_normalized.shape}")
        
        # Get min/max values for each component
        if len(mri_normalized.shape) > 3:
            for i in range(mri_normalized.shape[-1]):
                print(f"   Component {i}: Min={np.min(mri_normalized[..., i]):.3f}, Max={np.max(mri_normalized[..., i]):.3f}")
        else:
            print(f"   Min={np.min(mri_normalized):.3f}, Max={np.max(mri_normalized):.3f}")

        # Stack fields individually
        print(f"Stacking {num_fields_to_stack} field pairs...")
        for i in tqdm(range(num_fields_to_stack), desc="Stacking fields"):
            # Handle MRI data format based on MRI mode
            if args.mri_mode == 'conductivity':
                # Conductivity is scalar, add channel dimension
                if len(mri_normalized.shape) == 3:  # [bin, bin, bin]
                    mri_expanded = np.expand_dims(mri_normalized, axis=-1)  # -> [bin, bin, bin, 1]
                else:
                    mri_expanded = mri_normalized  # Already has channel dim
                    
                # Define channel names
                mri_channel_names = ['mri_conductivity']
            else:  # DTI tensor
                # DTI tensor has 6 components
                if len(mri_normalized.shape) == 4 and mri_normalized.shape[-1] == 6:
                    mri_expanded = mri_normalized  # Already has channel dim
                else:
                    raise ValueError(f"Unexpected DTI tensor data shape: {mri_normalized.shape}, expected 4D with 6 components")
                    
                # Define tensor component channel names
                mri_channel_names = ['dti_xx', 'dti_yy', 'dti_zz', 'dti_xy', 'dti_xz', 'dti_yz']

            # dA/dt field might be scalar [bin,bin,bin] or vector [bin,bin,bin,3]
            current_dadt = dadt_grids[i]
            if len(current_dadt.shape) == 3:  # Scalar dA/dt
                dadt_expanded = np.expand_dims(current_dadt, axis=-1)  # -> [bin, bin, bin, 1]
                dadt_channel_names = ['dadt_mag']
            elif len(current_dadt.shape) == 4:  # Vector dA/dt
                dadt_expanded = current_dadt
                dadt_channel_names = [f'dadt_{ax}' for ax in 'xyz']
            else:
                raise ValueError(f"Unexpected dA/dt grid shape for stacking: {current_dadt.shape}")

            # Stack MRI and dA/dt along the last (channel) dimension
            input_features = np.concatenate([mri_expanded, dadt_expanded], axis=-1)

            # Target E-field (can be scalar or vector)
            target_efield = efield_grids[i]
            
            # Determine target channel names
            if len(target_efield.shape) == 4 and target_efield.shape[-1] == 3:
                target_channel_names = [f'efield_{ax}' for ax in 'xyz']
            else:
                target_channel_names = ['efield_mag']

            # Create structured data dictionary for numpy/pickle
            structured_data = {
                'input_features': input_features.astype(np.float32),  # Use float32 for saving space
                'target_efield': target_efield.astype(np.float32),
                'metadata': {
                    'subject_id': subject_id,
                    'coil_position_idx': i,
                    'bin_size': bin_size,
                    'transformation': 'rotation_gmvec_to_y',
                    'mri_type': args.mri_mode,
                    'input_channels': mri_channel_names + dadt_channel_names,
                    'target_channels': target_channel_names
                }
            }

            # Save stacked data in requested formats
            file_suffix = f"stacked_{i}_b{bin_size}"
            if save_torch and torch_dir:
                torch.save({
                    'input_features': torch.from_numpy(input_features).float(),
                    'target_efield': torch.from_numpy(target_efield).float(),
                    'metadata': structured_data['metadata']
                }, os.path.join(torch_dir, f"{file_suffix}.pt"))
            if save_numpy and numpy_dir:
                # Saving dict requires allow_pickle=True for numpy
                np.save(os.path.join(numpy_dir, f"{file_suffix}.npy"), structured_data)
            if save_pickle and pickle_dir:
                import pickle
                with open(os.path.join(pickle_dir, f"{file_suffix}.pkl"), 'wb') as f:
                    pickle.dump(structured_data, f)

        stacking_time = time.time() - start_time_stacking
        print(f"Stacked {num_fields_to_stack} field pairs in {stacking_time:.2f} seconds")

    except Exception as e:
        print(f"ERROR during stacking for {subject_id}: {e}")
        traceback.print_exc()
        resource_monitor.stop_monitoring()
        return None  # Indicate failure


    # --- Create Manifest ---
    # Example PyTorch loading code (adjust paths and details)
    pytorch_example = f"""
import torch
import os

# Example loading a single sample
file_path = os.path.join('{torch_dir or "path/to/torch/data"}', 'stacked_0_b{bin_size}.pt')
try:
    stacked_data = torch.load(file_path)
    input_features = stacked_data['input_features'] # Shape: [bin, bin, bin, channels]
    target_efield = stacked_data['target_efield']   # Shape: [bin, bin, bin] or [bin, bin, bin, 3]
    metadata = stacked_data['metadata']
    print(f"Loaded sample 0: Input shape {{input_features.shape}}, Target shape {{target_efield.shape}}")
    print(f"Metadata: {{metadata}}")
except FileNotFoundError:
    print(f"Example file not found: {{file_path}}")

# Example loading a batch (requires a Dataset class usually)
# Assuming files are named stacked_0_b{bin_size}.pt, stacked_1_b{bin_size}.pt, ...
batch_inputs = []
batch_targets = []
num_to_load = min(5, {num_fields_to_stack}) # Load first few
data_dir = '{torch_dir or "path/to/torch/data"}'

print(f"\\nAttempting to load first {{num_to_load}} samples...")
for i in range(num_to_load):
    fpath = os.path.join(data_dir, f'stacked_{{i}}_b{bin_size}.pt')
    if os.path.exists(fpath):
        try:
            sample = torch.load(fpath)
            batch_inputs.append(sample['input_features'])
            batch_targets.append(sample['target_efield'])
        except Exception as e:
            print(f"Could not load {{fpath}}: {{e}}")
    else:
        print(f"File not found: {{fpath}}")

if batch_inputs:
    inputs_tensor = torch.stack(batch_inputs)  # Batch shape: [N, bin, bin, bin, channels]
    targets_tensor = torch.stack(batch_targets) # Batch shape: [N, bin, bin, bin] or [N, bin, bin, bin, 3]
    print(f"\\nSuccessfully loaded {{len(batch_inputs)}} samples.")
    print(f"Batch inputs shape: {{inputs_tensor.shape}}")
    print(f"Batch targets shape: {{targets_tensor.shape}}")
else:
    print("\\nCould not load any samples for batch example.")
"""

    stacked_manifest = {
        'subject_id': subject_id,
        'date_processed': time.strftime('%Y-%m-%d %H:%M:%S'),
        'bin_size': bin_size,
        'num_fields_processed': {'efield': n_efields, 'dadt': n_dadts},
        'num_fields_stacked': num_fields_to_stack,
        'efield_source_file': os.path.basename(paths['efield_data_path']),
        'dadt_source_file': os.path.basename(paths['dadt_data_path']),
        'mesh_source_file': os.path.basename(efield_mesh_path), # Assuming same mesh basis
        'mri_source_description': 'Conductivity derived from mesh elements, aligned to rotated space',
        'transformation_applied': {
             'type': 'rotation_gmvec_to_y',
             'target_vector': [0, 1, 0],
             'source_vector_key': 'skin_vec',
             'center_of_rotation_key': 'gm_point',
             'original_gm_point': gm_point.tolist(),
             'original_skin_vec': skin_vec.tolist(),
        },
        'roi_bounds_after_transform': efield_mapper.region_bounds,
        'data_structure': {
            'input_features': 'Aligned MRI conductivity stacked with rotated dA/dt',
            'target_efield': 'Rotated E-field ground truth',
            'channel_info': structured_data['metadata']['input_channels'] + structured_data['metadata']['target_channels'] # Example, refine as needed
        },
        'formats_saved': {
            'torch': save_torch, 'numpy': save_numpy, 'pickle': save_pickle
        },
        'output_paths': {
            'torch': torch_dir if save_torch else None,
            'numpy': numpy_dir if save_numpy else None,
            'pickle': pickle_dir if save_pickle else None,
            'intermediate_efield': efield_output_dir,
            'intermediate_dadt': dadt_output_dir,
            'aligned_mri': mri_output_dir
        },
        'pytorch_loading_example': pytorch_example
    }

    manifest_filename = f"stacked_manifest_b{bin_size}.json"
    manifest_path = os.path.join(stacked_output_dir, manifest_filename)
    try:
        with open(manifest_path, 'w') as f:
            json.dump(stacked_manifest, f, indent=4, cls=NumpyEncoder) # Use encoder for numpy arrays in metadata if any
        print(f"Saved manifest to: {manifest_path}")
    except Exception as e:
        print(f"ERROR saving manifest: {e}")
    # --- End Manifest ---


    # --- Cleanup Intermediate Files ---
    if clean_intermediate:
        print("\nCleaning up intermediate files...")
        try:
            if os.path.exists(efield_output_dir):
                shutil.rmtree(efield_output_dir)
                print(f"Removed intermediate E-field directory: {efield_output_dir}")
            if os.path.exists(dadt_output_dir):
                shutil.rmtree(dadt_output_dir)
                print(f"Removed intermediate dA/dt directory: {dadt_output_dir}")
            # Optionally remove the rotated MRI dir if only stacked is needed
            # if os.path.exists(mri_output_dir):
            #     shutil.rmtree(mri_output_dir)
            #     print(f"Removed intermediate aligned MRI directory: {mri_output_dir}")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    # --- End Cleanup ---


    # --- Final Summary ---
    resource_monitor.stop_monitoring()
    total_time = time.time() - start_time_total
    print("\n" + "="*80)
    print(f"Finished processing subject {subject_id}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"  - E-field processing: {efield_time:.2f} s")
    print(f"  - dA/dt processing: {dadt_time:.2f} s")
    print(f"  - MRI alignment: {mri_time:.2f} s")
    print(f"  - Stacking: {stacking_time:.2f} s")
    print(f"Fields stacked: {num_fields_to_stack}")
    print(f"Final output directory: {stacked_output_dir}")
    print("="*80 + "\n")

    # Return processing statistics
    return {
        'subject_id': subject_id,
        'bin_size': bin_size,
        'efield_count': n_efields,
        'dadt_count': n_dadts,
        'stacked_count': num_fields_to_stack,
        'efield_time': efield_time,
        'dadt_time': dadt_time,
        'mri_align_time': mri_time,
        'stacking_time': stacking_time,
        'total_time': total_time,
        'output_dir': stacked_output_dir,
        'manifest_path': manifest_path,
        'status': 'success'
    }

# Helper class for JSON dumping numpy arrays (if needed in manifest)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Convert arrays to lists for JSON
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


def normalize_mri(mri_data, method="minmax"):
    """Normalize MRI data using the specified method.
    Works with scalar, vector, and tensor (6-component) data.
    
    Args:
        mri_data: MRI data array (scalar, vector, or tensor)
        method: Normalization method ("minmax", "standard", "robust")
        
    Returns:
        Normalized MRI data
    """
    # Check if we have multi-component data
    is_multicomponent = len(mri_data.shape) >= 4 and mri_data.shape[-1] > 1
    
    if is_multicomponent:
        # Normalize each component separately
        normalized_data = np.zeros_like(mri_data)
        
        for i in range(mri_data.shape[-1]):
            component_data = mri_data[..., i]
            
            if method == "standard":
                # Z-score normalization
                mean = np.mean(component_data)
                std = np.std(component_data)
                if std > 0:
                    normalized_data[..., i] = (component_data - mean) / std
                else:
                    normalized_data[..., i] = component_data - mean
                    
            elif method == "minmax":
                # Min-max normalization
                min_val = np.min(component_data)
                max_val = np.max(component_data)
                if max_val > min_val:
                    normalized_data[..., i] = (component_data - min_val) / (max_val - min_val)
                else:
                    normalized_data[..., i] = np.zeros_like(component_data)
                    
            elif method == "robust":
                # Robust normalization using percentiles
                p10 = np.percentile(component_data, 10)
                p90 = np.percentile(component_data, 90)
                if p90 > p10:
                    normalized_data[..., i] = (component_data - p10) / (p90 - p10)
                else:
                    normalized_data[..., i] = np.zeros_like(component_data)
            else:
                # No normalization
                normalized_data[..., i] = component_data
        
        return normalized_data
    else:
        # Original scalar normalization
        if method == "standard":
            # Z-score normalization
            mean = np.mean(mri_data)
            std = np.std(mri_data)
            if std > 0:
                return (mri_data - mean) / std
            else:
                return mri_data - mean
                
        elif method == "minmax":
            # Min-max normalization
            min_val = np.min(mri_data)
            max_val = np.max(mri_data)
            if max_val > min_val:
                return (mri_data - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(mri_data)
                
        elif method == "robust":
            # Robust normalization using percentiles
            p10 = np.percentile(mri_data, 10)
            p90 = np.percentile(mri_data, 90)
            if p90 > p10:
                return (mri_data - p10) / (p90 - p10)
            else:
                return np.zeros_like(mri_data)
                
        else:
            # No normalization
            return mri_data
        

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Process each subject
    results = []
    
    for subject_num in args.subjects:
        # Format subject ID by adding 'sub-' prefix
        subject_id = f"sub-{subject_num}"
            
        print(f"\n{'='*80}")
        print(f"Processing subject: {subject_id}")
        print(f"{'='*80}\n")
        
        # Create a copy of args with the current subject
        import copy
        subject_args = copy.deepcopy(args)
        subject_args.subject = subject_id
        
        try:
            # Process and stack fields for this subject
            subject_result = process_and_stack_fields(subject_args)
            results.append(subject_result)
            print(f"Successfully processed subject: {subject_id}")
        except Exception as e:
            print(f"ERROR processing subject {subject_id}: {str(e)}")
            # Continue with next subject instead of terminating
            continue
    
    # Print summary of all processed subjects
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY FOR ALL SUBJECTS")
    print(f"{'='*80}")
    failed_subjects = len(args.subjects) - len([r for r in results if r is not None])
    processed_count = len(results) - failed_subjects
    print(f"Total subjects attempted: {len(args.subjects)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_subjects}")
    for result in results:
        if result and result.get('status') == 'success': # Check if result is not None and status is success
            print(f"\nSubject: {result.get('subject_id', 'N/A')}") # Use .get for safety
            print(f"  Fields stacked: {result.get('stacked_count', 'N/A')}")
            print(f"  Processing time: {result.get('total_time', -1):.2f} seconds")
            print(f"  Output directory: {result.get('output_dir', 'N/A')}")
        elif result: # If result exists but status wasn't success (or status key missing)
            print(f"\nSubject: {result.get('subject_id', 'N/A')} - Processing Incomplete or Failed")
            print(f"  Status: {result.get('status', 'unknown_error')}")
            if 'error_message' in result: # Optional: Add error message to result dict on failure
                print(f"  Error: {result['error_message']}")
    # Return success
    sys.exit(0)