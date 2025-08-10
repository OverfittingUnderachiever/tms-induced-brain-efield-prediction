# tms_efield_prediction/data/pipeline/training_data_generator.py
"""
Training data generator for TMS E-field prediction.

This module provides components for generating training data from raw TMS simulation results,
with functionalities for processing and stacking E-fields and dA/dt fields.
"""

import os
import numpy as np
import time
import json
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import multiprocessing as mp
from tqdm import tqdm

from utils.debug.hooks import PipelineDebugHook
from utils.resource.monitor import ResourceMonitor
from utils.state.context import TMSPipelineContext, PipelineState
from .tms_data_types import TMSRawData, TMSProcessedData, TMSSample
from ..transformations.voxel_mapping import VoxelMapper, create_transform_matrix


@dataclass
class OrientationData:
    """Orientation data for processing."""
    original_center: List[float]
    original_normal: List[float]
    target_center: List[float]
    target_normal: List[float]


@dataclass
class ProcessingConfig:
    """Configuration for processing E-fields and dA/dt fields."""
    subject_id: str
    bin_size: int = 25
    n_processes: Optional[int] = None
    save_torch: bool = True
    save_numpy: bool = False
    save_pickle: bool = False
    clean_intermediate: bool = False
    base_dir: Optional[str] = None
    output_dir: Optional[str] = None


class TrainingDataGenerator:
    """
    Generator for TMS E-field prediction training data.
    
    This class processes raw E-field and dA/dt data, transforms them into voxelized format,
    and stacks them together to create input features for neural networks.
    """
    
    def __init__(
        self,
        context: TMSPipelineContext,
        debug_hook: Optional[PipelineDebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """
        Initialize the training data generator.
        
        Args:
            context: TMS-specific pipeline context
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
        """
        self.context = context
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Register with resource monitor if provided
        if self.resource_monitor:
            self.resource_monitor.register_component(
                "training_data_generator",
                self._reduce_memory,
                priority=10  # Higher priority for this component
            )
        
        # State tracking
        self.processing_state = {}
        self._memory_usage = 0
    
    def load_orientation_data(self, subject_id: str, orientation_path: Optional[str] = None) -> Optional[OrientationData]:
        """
        Load orientation data for the subject.
        
        Args:
            subject_id: Subject ID
            orientation_path: Optional path to orientation data file. If None, uses default path.
            
        Returns:
            OrientationData object if successful, None otherwise
        """
        # Determine path if not provided
        if orientation_path is None:
            orientation_path = self._get_orientation_path(subject_id)
        
        # Log orientation data loading attempt
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "loading_orientation_data",
                {'subject_id': subject_id, 'path': orientation_path}
            )
        
        # Check if orientation data exists
        if not os.path.exists(orientation_path):
            if self.debug_hook:
                self.debug_hook.record_error(
                    ValueError(f"Orientation data not found at {orientation_path}"),
                    {'subject_id': subject_id, 'operation': 'load_orientation_data'}
                )
            return None
        
        try:
            # Load orientation data
            with open(orientation_path, 'r') as f:
                data = json.load(f)
            
            # Check required keys
            required_keys = ["original_center", "original_normal", "target_center", "target_normal"]
            for key in required_keys:
                if key not in data:
                    if self.debug_hook:
                        self.debug_hook.record_error(
                            ValueError(f"Orientation data is missing required key: {key}"),
                            {'subject_id': subject_id, 'operation': 'load_orientation_data'}
                        )
                    return None
            
            # Create OrientationData object
            orientation_data = OrientationData(
                original_center=data["original_center"],
                original_normal=data["original_normal"],
                target_center=data["target_center"],
                target_normal=data["target_normal"]
            )
            
            # Log successful loading
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "orientation_data_loaded",
                    {
                        'subject_id': subject_id,
                        'original_center': orientation_data.original_center,
                        'original_normal': orientation_data.original_normal,
                        'target_center': orientation_data.target_center,
                        'target_normal': orientation_data.target_normal
                    }
                )
            
            return orientation_data
            
        except Exception as e:
            # Log error
            if self.debug_hook:
                self.debug_hook.record_error(
                    e,
                    {'subject_id': subject_id, 'operation': 'load_orientation_data'}
                )
            return None
    
    def _get_orientation_path(self, subject_id: str) -> str:
        """
        Get the path to orientation data for the subject.
        
        Args:
            subject_id: Subject ID
            
        Returns:
            Path to orientation data file
        """
        # Try to derive from context if possible
        if hasattr(self.context, 'data_root_path') and self.context.data_root_path:
            # Handle different subject ID formats
            if subject_id.startswith('sub-'):
                base_subject_id = subject_id
            else:
                base_subject_id = f"sub-{subject_id}"
            
            # Check different potential locations for orientation data
            potential_paths = [
                os.path.join(self.context.data_root_path, base_subject_id, "experiment", "orientation_data.json"),
                os.path.join(self.context.data_root_path, "data", base_subject_id, "experiment", "orientation_data.json"),
                os.path.join(self.context.data_root_path, "data", base_subject_id, "orientation_data.json")
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    return path
        
        # Default path (original implementation)
        return f"/home/freyhe/MA_Henry/data/{subject_id}/experiment/orientation_data.json"

    def _normalize_mri(self, mri_data: np.ndarray) -> np.ndarray:
        """
        Normalize MRI data.
        
        Args:
            mri_data: MRI data array
            
        Returns:
            Normalized MRI data
        """
        method = self.context.normalization_method if hasattr(self.context, 'normalization_method') else "minmax"
        
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


    def load_mri_conductivity(self, subject_id: str) -> np.ndarray:
        """
        Load and process MRI conductivity data.
        
        Args:
            subject_id: Subject ID
            
        Returns:
            Processed conductivity map as numpy array
        """
        from simnibs import mesh_io
        import scipy.io as sio
        
        # Determine paths
        base_dir = self._get_base_dir(subject_id)
        
        # Check if processed conductivity already exists
        cache_dir = os.path.join(base_dir, "cached_data")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"mri_conductivity_b{self.config.bin_size}.pt")
        
        # If cache exists, load it directly
        if os.path.exists(cache_path):
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("loading_cached_conductivity", {'path': cache_path})
            print(f"Loading cached MRI conductivity data from {cache_path}")
            return torch.load(cache_path).numpy()
        
        # Otherwise, process from mesh
        print("Cached MRI conductivity not found. Processing from mesh...")
        
        sub_parts = subject_id.split('-')
        sub_id = sub_parts[-1] if len(sub_parts) > 1 else subject_id
        
        # Get mesh and ROI center
        sub_path = os.path.join(os.path.dirname(base_dir), subject_id)
        msh_name = f"{subject_id}.msh"
        mesh_path = os.path.join(sub_path, 'headmodel', msh_name)
        
        # Try alternative paths if needed
        if not os.path.exists(mesh_path):
            alt_mesh_paths = [
                os.path.join(sub_path, 'headmodel', f"sub-{sub_id}.msh"),
                os.path.join(sub_path, f"{subject_id}.msh"),
                os.path.join(sub_path, f"sub-{sub_id}.msh")
            ]
            for path in alt_mesh_paths:
                if os.path.exists(path):
                    mesh_path = path
                    break
        
        # Load mesh
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("loading_head_mesh", {'path': mesh_path})
        
        try:
            msh = mesh_io.read_msh(mesh_path)
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(
                    e, {'subject_id': subject_id, 'operation': 'load_mri_conductivity'}
                )
            raise ValueError(f"Failed to load head mesh: {str(e)}")
        
        # Load ROI center
        roi_center_path = os.path.join(sub_path, 'experiment', f"{subject_id}_roi_center.mat")
        if not os.path.exists(roi_center_path):
            alt_roi_paths = [
                os.path.join(sub_path, 'experiment', f"sub-{sub_id}_roi_center.mat"),
                os.path.join(base_dir, f"{subject_id}_roi_center.mat"),
                os.path.join(base_dir, f"sub-{sub_id}_roi_center.mat")
            ]
            for path in alt_roi_paths:
                if os.path.exists(path):
                    roi_center_path = path
                    break
        
        try:
            roi_center = sio.loadmat(roi_center_path)['roi_center']
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(
                    e, {'subject_id': subject_id, 'operation': 'load_mri_conductivity'}
                )
            raise ValueError(f"Failed to load ROI center: {str(e)}")
        
        # Crop mesh to element type 4
        cropped = msh.crop_mesh(elm_type=4)
        
        # Get field binning parameters
        bin_size = self.config.bin_size
        
        # Create grid coordinates
        x = np.linspace(-bin_size/2, bin_size/2, bin_size)
        y = np.linspace(-bin_size/2, bin_size/2, bin_size)
        z = np.linspace(-bin_size/2, bin_size/2, bin_size)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
        grid_coords = np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T
        
        # Find closest elements and get conductivities
        idcs = cropped.find_closest_element(grid_coords, return_index=True)[1]
        cond = cropped.elm.tag1[idcs].astype(float).reshape(bin_size, bin_size, bin_size)
        
        # Map conductivity numbers to values
        cond_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 500]
        cond_values = [0.126, 0.275, 1.654, 0.01, 0.465, 0.5, 0.008, 0.025, 0.6, 0.16, 29.4, 1.0]
        
        for n, v in zip(cond_number, cond_values):
            cond[cond == n] = v
        
        # Save to cache for future use
        torch.save(torch.from_numpy(cond).float(), cache_path)
        print(f"Saved MRI conductivity to cache: {cache_path}")
        
        return cond
    def process_and_stack_fields(self, config: ProcessingConfig) -> Dict[str, Any]:
        """
        Process both E-fields and dAdt fields, voxelize them, and stack MRI with dAdt.
        
        Args:
            config: Processing configuration
            
        Returns:
            Dictionary with processing statistics
        """
        # Ensure subject_id is in the right format
        subject_id = config.subject_id
        if not subject_id.startswith('sub-'):
            subject_id = f"sub-{subject_id}"
        
        # Log processing start
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "processing_and_stacking_start",
                {
                    'subject_id': subject_id,
                    'bin_size': config.bin_size,
                    'n_processes': config.n_processes
                }
            )
        
        start_time_total = time.time()
        
        # First, check for orientation data
        orientation_data = self.load_orientation_data(subject_id)
        if orientation_data is None:
            if self.debug_hook:
                self.debug_hook.record_error(
                    ValueError(f"Cannot proceed without orientation data for subject {subject_id}"),
                    {'subject_id': subject_id, 'operation': 'process_and_stack_fields'}
                )
            return {'error': 'Cannot proceed without orientation data'}
        
        # Define paths based on subject ID
        base_dir = config.base_dir or self._get_base_dir(subject_id)
        
        # Paths for E-fields
        efield_data_path = os.path.join(base_dir, "multi_sim_100", f"{subject_id}_all_efields.npy")
        efield_mesh_path = os.path.join(base_dir, "multi_sim_100", "mesh_outputs", f"{subject_id}_efield_first.msh")
        efield_output_dir = os.path.join(base_dir, "E_arrays")
        
        # Paths for dAdt fields
        dadt_data_path = os.path.join(base_dir, "dadt_roi_maps", f"{subject_id}_roi_dadts.npy")
        dadt_mesh_path = os.path.join(base_dir, "all", f"{subject_id}_middle_gray_matter_roi.msh")
        dadt_output_dir = os.path.join(base_dir, "dAdt_arrays")
        
        # Path for stacked results
        stacked_output_dir = config.output_dir or os.path.join(base_dir, "stacked_arrays")
        os.makedirs(stacked_output_dir, exist_ok=True)
        
        # Create subdirectories for stacked results
        if config.save_torch:
            torch_dir = os.path.join(stacked_output_dir, "torch")
            os.makedirs(torch_dir, exist_ok=True)
        
        if config.save_numpy:
            numpy_dir = os.path.join(stacked_output_dir, "numpy")
            os.makedirs(numpy_dir, exist_ok=True)
        
        if config.save_pickle:
            pickle_dir = os.path.join(stacked_output_dir, "pickle")
            os.makedirs(pickle_dir, exist_ok=True)
        
        # Log directories
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "processing_directories",
                {
                    'base_dir': base_dir,
                    'efield_data_path': efield_data_path,
                    'efield_mesh_path': efield_mesh_path,
                    'dadt_data_path': dadt_data_path,
                    'dadt_mesh_path': dadt_mesh_path,
                    'output_dir': stacked_output_dir
                }
            )
        
        # 1. Process E-fields
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("efield_processing_start", {'subject_id': subject_id})
                
        efield_results = self.process_fields(
            field_path=efield_data_path,
            mesh_path=efield_mesh_path,
            output_dir=efield_output_dir,
            field_type="efield",
            orientation_data=orientation_data,
            bin_size=config.bin_size,
            n_processes=config.n_processes,
            save_torch=True,  # Always save intermediate results
            save_numpy=True,
            save_pickle=False
        )
        
        # Update processing state
        self.processing_state['efield_processing'] = efield_results
        
        # 2. Process dAdt fields
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("dadt_processing_start", {'subject_id': subject_id})
                
        dadt_results = self.process_fields(
            field_path=dadt_data_path,
            mesh_path=dadt_mesh_path,
            output_dir=dadt_output_dir,
            field_type="dadt",
            orientation_data=orientation_data,
            bin_size=config.bin_size,
            n_processes=config.n_processes,
            save_torch=True,  # Always save intermediate results
            save_numpy=True,
            save_pickle=False
        )
        
        # Update processing state
        self.processing_state['dadt_processing'] = dadt_results
        
        # 3. Load and process MRI conductivity data
        print(f"STEP 3: Loading MRI conductivity data for {subject_id}")
        mri_output_dir = os.path.join(base_dir, "MRI_arrays")
        os.makedirs(os.path.join(mri_output_dir, "torch"), exist_ok=True)
        os.makedirs(os.path.join(mri_output_dir, "numpy"), exist_ok=True)
        
        try:
            # Load MRI conductivity data
            mri_conductivity = self.load_mri_conductivity(subject_id)
            
            # Save MRI data
            if config.save_torch:
                torch.save(torch.from_numpy(mri_conductivity).float(), 
                        os.path.join(mri_output_dir, "torch", f"mri_conductivity_b{config.bin_size}.pt"))
            
            if config.save_numpy:
                np.save(os.path.join(mri_output_dir, "numpy", f"mri_conductivity_b{config.bin_size}.npy"), 
                    mri_conductivity)
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(
                    e, {'subject_id': subject_id, 'operation': 'process_mri_conductivity'}
                )
            return {'error': f'Failed to load MRI data: {str(e)}'}
        
        # 4. Stack MRI and dAdt for input, keep E-field as target
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("stacking_start", {'subject_id': subject_id})
                
        stacking_start_time = time.time()
        
        # Get the number of fields for each type
        num_efields = efield_results['total_fields']
        num_dadt_fields = dadt_results['total_fields']
        
        # Determine the number of fields to stack (minimum of both)
        num_fields_to_stack = min(num_efields, num_dadt_fields)
        
        if num_efields != num_dadt_fields:
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "field_count_mismatch",
                    {
                        'num_efields': num_efields,
                        'num_dadt_fields': num_dadt_fields,
                        'will_stack': num_fields_to_stack
                    }
                )
        
        # Stack results: MRI + dAdt as input, E-field as target
        stacked_results = []
        
        # Normalize MRI data
        mri_normalized = self._normalize_mri(mri_conductivity)
        
        # Use tqdm for progress tracking
        for i in tqdm(range(num_fields_to_stack), desc="Stacking fields"):
            # Load fields
            efield_torch_path = os.path.join(efield_output_dir, "torch", f"efield_{i}_b{config.bin_size}.pt")
            dadt_torch_path = os.path.join(dadt_output_dir, "torch", f"dadt_{i}_b{config.bin_size}.pt")
            
            efield = torch.load(efield_torch_path)
            dadt = torch.load(dadt_torch_path)
            
            # Check shapes
            efield_shape = efield.shape
            dadt_shape = dadt.shape
            
            # Prepare output paths
            if config.save_torch:
                stacked_torch_path = os.path.join(torch_dir, f"stacked_{i}_b{config.bin_size}.pt")
            if config.save_numpy:
                stacked_numpy_path = os.path.join(numpy_dir, f"stacked_{i}_b{config.bin_size}.npy")
            if config.save_pickle:
                stacked_pickle_path = os.path.join(pickle_dir, f"stacked_{i}_b{config.bin_size}.pkl")
            
            # Prepare MRI data for stacking
            if len(mri_normalized.shape) == 3:  # [bin, bin, bin]
                mri_tensor = torch.from_numpy(mri_normalized).float().unsqueeze(-1)  # [bin, bin, bin, 1]
            else:
                mri_tensor = torch.from_numpy(mri_normalized).float()
            
            # Stack MRI and dAdt as input features
            if len(dadt_shape) == 3 and len(mri_tensor.shape) == 4:
                # dAdt is scalar [bin, bin, bin], MRI is [bin, bin, bin, 1]
                dadt_expanded = dadt.unsqueeze(-1)
                input_features = torch.cat([mri_tensor, dadt_expanded], dim=-1)
            elif len(dadt_shape) == 4 and len(mri_tensor.shape) == 4:
                # Both are tensor formats (MRI [bin, bin, bin, 1], dAdt [bin, bin, bin, 3])
                input_features = torch.cat([mri_tensor, dadt], dim=-1)
            else:
                # Handle incompatible shapes
                if self.debug_hook:
                    self.debug_hook.record_event(
                        "shape_mismatch",
                        {
                            'mri_shape': list(mri_tensor.shape),
                            'dadt_shape': list(dadt_shape)
                        }
                    )
                # Create a dictionary format
                input_features = {
                    'mri': mri_tensor,
                    'dadt': dadt
                }
            
            # Create structured data with input_features and target_efield
            structured_data = {
                'input_features': input_features,
                'target_efield': efield,
                'metadata': {
                    'subject_id': subject_id,
                    'coil_position_idx': i,
                    'bin_size': config.bin_size
                }
            }
            
            # Save structured results
            if config.save_torch:
                torch.save(structured_data, stacked_torch_path)
            
            if config.save_numpy:
                if isinstance(input_features, dict):
                    # For dictionary format, save as numpy dictionary
                    np_dict = {
                        'input_features': {
                            'mri': mri_tensor.numpy(),
                            'dadt': dadt.numpy()
                        },
                        'target_efield': efield.numpy(),
                        'metadata': structured_data['metadata']
                    }
                    np.save(stacked_numpy_path, np_dict)
                else:
                    # For tensor format, convert to numpy dictionary
                    np_dict = {
                        'input_features': input_features.numpy(),
                        'target_efield': efield.numpy(),
                        'metadata': structured_data['metadata']
                    }
                    np.save(stacked_numpy_path, np_dict)
            
            if config.save_pickle:
                import pickle
                pickle_data = {
                    'input_features': input_features.numpy() if not isinstance(input_features, dict) else {
                        'mri': mri_tensor.numpy(),
                        'dadt': dadt.numpy()
                    },
                    'target_efield': efield.numpy(),
                    'metadata': structured_data['metadata']
                }
                with open(stacked_pickle_path, 'wb') as f:
                    pickle.dump(pickle_data, f)
            
            # Store info about stacked data
            if isinstance(input_features, dict):
                stacked_info = {
                    'shape': 'dict',
                    'mri_shape': list(mri_tensor.shape),
                    'dadt_shape': list(dadt.shape),
                    'efield_shape': list(efield_shape)
                }
            else:
                stacked_info = {
                    'shape': 'tensor',
                    'input_features_shape': list(input_features.shape),
                    'target_efield_shape': list(efield_shape),
                    'channel_info': ['mri'] + (['dadt_x', 'dadt_y', 'dadt_z'] if len(dadt_shape) == 4 else ['dadt'])
                }
            
            stacked_results.append(stacked_info)
        
        # Create manifest for stacked results
        stacked_manifest = {
            'subject_id': subject_id,
            'date_processed': time.strftime('%Y-%m-%d %H:%M:%S'),
            'bin_size': config.bin_size,
            'num_fields': num_fields_to_stack,
            'efield_source': efield_data_path,
            'dadt_source': dadt_data_path,
            'mri_source': 'conductivity data from mesh',
            'orientation_data': {
                'original_center': orientation_data.original_center,
                'original_normal': orientation_data.original_normal,
                'target_center': orientation_data.target_center,
                'target_normal': orientation_data.target_normal
            },
            'data_structure': {
                'input_features': 'MRI conductivity stacked with dA/dt',
                'target_efield': 'E-field ground truth'
            },
            'stacked_info': stacked_results[0] if stacked_results else None,
            'formats': {
                'torch': config.save_torch,
                'numpy': config.save_numpy,
                'pickle': config.save_pickle
            },
            'output_paths': {
                'torch': os.path.join(stacked_output_dir, "torch") if config.save_torch else None,
                'numpy': os.path.join(stacked_output_dir, "numpy") if config.save_numpy else None,
                'pickle': os.path.join(stacked_output_dir, "pickle") if config.save_pickle else None
            },
            'pytorch_loading_example': """
    import torch

    # Load a single stacked field
    stacked_data = torch.load('path/to/stacked_0_b25.pt')

    # Access input features and target
    input_features = stacked_data['input_features']
    target_efield = stacked_data['target_efield']

    print(f"Input features shape: {input_features.shape}")
    print(f"Target E-field shape: {target_efield.shape}")

    # For a batch of stacked data
    batch = []
    for i in range(90):
        sample = torch.load(f'path/to/stacked_{i}_b25.pt')
        batch.append({
            'input': sample['input_features'],
            'target': sample['target_efield']
        })
    """
        }
        
        # Save manifest
        manifest_path = os.path.join(stacked_output_dir, f"stacked_manifest_b{config.bin_size}.json")
        with open(manifest_path, 'w') as f:
            json.dump(stacked_manifest, f, indent=4)
        
        # Clean up intermediate files if requested
        if config.clean_intermediate:
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("cleaning_intermediate_files", {})
                    
            import shutil
            if os.path.exists(efield_output_dir):
                shutil.rmtree(efield_output_dir)
            if os.path.exists(dadt_output_dir):
                shutil.rmtree(dadt_output_dir)
        
        stacking_time = time.time() - stacking_start_time
        total_time = time.time() - start_time_total
        
        # Log completion
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "processing_and_stacking_complete",
                {
                    'subject_id': subject_id,
                    'bin_size': config.bin_size,
                    'num_stacked': num_fields_to_stack,
                    'stacking_time': stacking_time,
                    'total_time': total_time
                }
            )
        
        # Return processing statistics
        return {
            'subject_id': subject_id,
            'bin_size': config.bin_size,
            'efield_results': efield_results,
            'dadt_results': dadt_results,
            'num_stacked': num_fields_to_stack,
            'stacking_time': stacking_time,
            'total_time': total_time,
            'manifest_path': manifest_path
        }

    def process_fields(
        self,
        field_path: str,
        mesh_path: str,
        output_dir: str,
        field_type: str,
        orientation_data: OrientationData,
        bin_size: int = 25,
        n_processes: Optional[int] = None,
        save_torch: bool = True,
        save_numpy: bool = True,
        save_pickle: bool = False
    ) -> Dict[str, Any]:
        """
        Process all fields (E-fields or dAdt) in parallel, converting them to voxel grids.
        
        Args:
            field_path: Path to the file containing all fields
            mesh_path: Path to the mesh file
            output_dir: Directory to save voxelized fields
            field_type: Type of field to process ('efield' or 'dadt')
            orientation_data: Dictionary containing orientation parameters
            bin_size: Number of bins for voxelization
            n_processes: Number of processes to use. If None, uses all available cores.
            save_torch: Whether to save data as PyTorch tensors
            save_numpy: Whether to save data as NumPy arrays
            save_pickle: Whether to save data as pickle files with metadata
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        
        # Log processing start
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                f"{field_type}_processing_start",
                {
                    'field_path': field_path,
                    'mesh_path': mesh_path,
                    'bin_size': bin_size
                }
            )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create format-specific subdirectories if needed
        if save_torch:
            torch_dir = os.path.join(output_dir, "torch")
            os.makedirs(torch_dir, exist_ok=True)
        
        if save_numpy:
            numpy_dir = os.path.join(output_dir, "numpy")
            os.makedirs(numpy_dir, exist_ok=True)
        
        if save_pickle:
            pickle_dir = os.path.join(output_dir, "pickle")
            os.makedirs(pickle_dir, exist_ok=True)
        
        # 1. Load mesh to get centroid for transformation
        from simnibs import mesh_io
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("loading_mesh", {'path': mesh_path})
            
        msh = mesh_io.read_msh(mesh_path)
        node_coords = msh.nodes.node_coord
        mesh_centroid = np.mean(node_coords, axis=0)
        
        # 2. Create transformation matrix with parameters from orientation data
        original_center = np.array(orientation_data.original_center)
        original_normal = np.array(orientation_data.original_normal)
        target_center = np.array(orientation_data.target_center)
        target_normal = np.array(orientation_data.target_normal)
        
        transform_matrix = create_transform_matrix(
            original_center, original_normal, target_center, target_normal
        )
        
        # 3. Create VoxelMapper
        mapper = VoxelMapper(
            mesh_path=mesh_path,
            transform_matrix=transform_matrix,
            bin_size=bin_size,
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        # 4. Run preprocessing and save the results
        preprocessing_path = os.path.join(output_dir, f"preprocessing_b{bin_size}.pkl")
        preprocessing_stats = mapper.preprocess(save_path=preprocessing_path)
        
        # 5. Load all fields
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(f"loading_{field_type}_fields", {'path': field_path})
            
        all_fields = np.load(field_path)
        
        # Determine data structure
        if len(all_fields.shape) == 3:  # [num_fields, num_nodes, 3]
            num_fields = all_fields.shape[0]
            num_nodes = all_fields.shape[1]
            
            # Verify that the node count matches the mesh
            if num_nodes != mapper.total_nodes:
                if self.debug_hook:
                    self.debug_hook.record_event(
                        "node_count_mismatch",
                        {
                            'field_nodes': num_nodes,
                            'mesh_nodes': mapper.total_nodes
                        }
                    )
                
                # Try to determine if we need to transpose
                if all_fields.shape[1] == 3 and all_fields.shape[0] == mapper.total_nodes:
                    if self.debug_hook:
                        self.debug_hook.record_event("transposing_fields", {})
                        
                    all_fields = np.transpose(all_fields, (2, 0, 1))
                    num_fields = all_fields.shape[0]
        
        elif len(all_fields.shape) == 2:
            # Could be [num_fields, num_nodes] or [num_nodes, 3]
            if all_fields.shape[1] == 3:
                # Single field with vector values [num_nodes, 3]
                num_fields = 1
                all_fields = np.array([all_fields])
            else:
                # Multiple fields with scalar values [num_fields, num_nodes]
                num_fields = all_fields.shape[0]
        else:
            error_msg = f"Unexpected {field_type} array shape: {all_fields.shape}"
            if self.debug_hook:
                self.debug_hook.record_error(
                    ValueError(error_msg),
                    {'field_type': field_type, 'operation': 'process_fields'}
                )
            raise ValueError(error_msg)
        
        # Log field structure
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                f"{field_type}_structure",
                {
                    'shape': all_fields.shape,
                    'num_fields': num_fields
                }
            )
        
        # Save some processing data for workers to reuse
        worker_data = {
            'voxel_coordinates': mapper.voxel_coordinates,
            'node_to_voxel_map': mapper.node_to_voxel_map,
            'voxel_node_counts': mapper.voxel_node_counts,
            'bin_size': bin_size,
            'region_bounds': mapper.region_bounds,
            'output_dir': output_dir,
            'save_torch': save_torch,
            'save_numpy': save_numpy,
            'save_pickle': save_pickle,
            'torch_dir': torch_dir if save_torch else None,
            'numpy_dir': numpy_dir if save_numpy else None,
            'pickle_dir': pickle_dir if save_pickle else None,
            'field_type': field_type  # Pass field type to workers
        }
        
        # Save worker data to a temporary file
        import pickle
        worker_data_path = os.path.join(output_dir, f"worker_data_b{bin_size}.pkl")
        with open(worker_data_path, 'wb') as f:
            pickle.dump(worker_data, f)
        
        # Determine number of processes
        if n_processes is None:
            n_processes = mp.cpu_count()
        
        # Log parallel processing start
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                f"{field_type}_parallel_processing_start",
                {
                    'num_fields': num_fields,
                    'n_processes': n_processes
                }
            )
        
        # Prepare worker arguments
        worker_args = [(i, all_fields[i], worker_data_path) for i in range(num_fields)]
        
        # Use multiprocessing for parallel execution
        with mp.Pool(processes=n_processes) as pool:
            # Process fields in parallel with progress bar
            list(tqdm(
                pool.imap(process_field_worker, worker_args),
                total=num_fields,
                desc=f"Processing {field_type} fields"
            ))
        
        # Clean up temporary data
        os.remove(worker_data_path)
        
        # Create data manifest
        # Determine if this is vector data based on shape
        is_vector = len(all_fields.shape) == 3 and all_fields.shape[2] == 3
        self._create_data_manifest(
            output_dir=output_dir, 
            num_fields=num_fields, 
            bin_size=bin_size, 
            save_torch=save_torch, 
            save_numpy=save_numpy, 
            save_pickle=save_pickle, 
            field_type=field_type, 
            is_vector=is_vector, 
            orientation_data=orientation_data
        )
        
        elapsed_time = time.time() - start_time
        
        # Log completion
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                f"{field_type}_processing_complete",
                {
                    'num_fields': num_fields,
                    'elapsed_time': elapsed_time
                }
            )
        
        return {
            'bin_size': bin_size,
            'preprocessing_stats': preprocessing_stats,
            'total_fields': num_fields,
            'elapsed_time': elapsed_time,
            'n_processes': n_processes
        }
    
    def _create_data_manifest(
        self, 
        output_dir: str,
        num_fields: int,
        bin_size: int,
        save_torch: bool,
        save_numpy: bool,
        save_pickle: bool,
        field_type: str,
        is_vector: bool,
        orientation_data: OrientationData
    ) -> None:
        """
        Create a manifest file with information about the processed data.
        
        Args:
            output_dir: Directory where data is saved
            num_fields: Number of fields processed
            bin_size: Voxelization bin size
            save_torch: Whether PyTorch tensors were saved
            save_numpy: Whether NumPy arrays were saved
            save_pickle: Whether pickle files were saved
            field_type: Type of field processed ('efield' or 'dadt')
            is_vector: Whether the data contains vector fields
            orientation_data: Orientation data used for processing
        """
        manifest = {
            'date_processed': time.strftime('%Y-%m-%d %H:%M:%S'),
            'field_type': field_type,
            'num_fields': num_fields,
            'bin_size': bin_size,
            'grid_shape': f"{bin_size}×{bin_size}×{bin_size}" + ("×3" if is_vector else ""),
            'is_vector': is_vector,
            'orientation_data': {
                'original_center': orientation_data.original_center,
                'original_normal': orientation_data.original_normal,
                'target_center': orientation_data.target_center,
                'target_normal': orientation_data.target_normal
            },
            'formats': {
                'torch': save_torch,
                'numpy': save_numpy,
                'pickle': save_pickle
            },
            'output_paths': {
                'torch': os.path.join(output_dir, "torch") if save_torch else None,
                'numpy': os.path.join(output_dir, "numpy") if save_numpy else None,
                'pickle': os.path.join(output_dir, "pickle") if save_pickle else None
            },
            'pytorch_loading_example': f"""
import torch

# Load a single {field_type} field
field = torch.load('path/to/{field_type}_0_b{bin_size}.pt')
print(f"{field_type} shape: {{field.shape}}")  # Should be [{bin_size}, {bin_size}, {bin_size}{', 3' if is_vector else ''}]

# For a batch of {field_type} fields
fields = [torch.load(f'path/to/{field_type}_{{i}}_b{bin_size}.pt') for i in range({num_fields})]
batch = torch.stack(fields)
print(f"Batch shape: {{batch.shape}}")  # Should be [{num_fields}, {bin_size}, {bin_size}, {bin_size}{', 3' if is_vector else ''}]
"""
        }
        
        # Save manifest as JSON
        manifest_path = os.path.join(output_dir, f"data_manifest_b{bin_size}.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        
        # Log manifest creation
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "manifest_created",
                {'path': manifest_path}
            )
    
    def _get_base_dir(self, subject_id: str) -> str:
        """
        Get the base directory for subject data.
        
        Args:
            subject_id: Subject ID
            
        Returns:
            Base directory path
        """
        # Try to derive from context if possible
        if hasattr(self.context, 'data_root_path') and self.context.data_root_path:
            # Handle different subject ID formats
            if subject_id.startswith('sub-'):
                base_subject_id = subject_id
            else:
                base_subject_id = f"sub-{subject_id}"
            
            # Check different potential locations
            potential_paths = [
                os.path.join(self.context.data_root_path, base_subject_id, "experiment"),
                os.path.join(self.context.data_root_path, "data", base_subject_id, "experiment")
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    return path
        
        # Default path (original implementation)
        return f"/home/freyhe/MA_Henry/data/{subject_id}/experiment"
    
    def create_training_samples_from_stacked(
        self, 
        subject_id: str,
        stacked_dir: Optional[str] = None,
        bin_size: int = 25
    ) -> List[TMSProcessedData]:
        """
        Create TMS processed data samples from stacked data.
        
        Args:
            subject_id: Subject ID
            stacked_dir: Directory containing stacked data. If None, uses default path.
            bin_size: Bin size used for voxelization
            
        Returns:
            List of TMS processed data objects
        """
        # Ensure subject_id is in the right format
        if not subject_id.startswith('sub-'):
            subject_id = f"sub-{subject_id}"
        
        # Get stacked directory if not provided
        if stacked_dir is None:
            base_dir = self._get_base_dir(subject_id)
            stacked_dir = os.path.join(base_dir, "stacked_arrays")
        
        # Check if directory exists
        if not os.path.exists(stacked_dir):
            error_msg = f"Stacked data directory not found: {stacked_dir}"
            if self.debug_hook:
                self.debug_hook.record_error(
                    ValueError(error_msg),
                    {'subject_id': subject_id, 'operation': 'create_training_samples_from_stacked'}
                )
            raise ValueError(error_msg)
        
        # Load manifest
        manifest_path = os.path.join(stacked_dir, f"stacked_manifest_b{bin_size}.json")
        if not os.path.exists(manifest_path):
            error_msg = f"Stacked manifest not found: {manifest_path}"
            if self.debug_hook:
                self.debug_hook.record_error(
                    ValueError(error_msg),
                    {'subject_id': subject_id, 'operation': 'create_training_samples_from_stacked'}
                )
            raise ValueError(error_msg)
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Determine paths and format
        if manifest['formats']['torch']:
            format_dir = "torch"
            loader = lambda path: torch.load(path)
            is_torch = True
        elif manifest['formats']['numpy']:
            format_dir = "numpy"
            loader = lambda path: np.load(path)
            is_torch = False
        else:
            error_msg = "No supported format (torch or numpy) found in manifest"
            if self.debug_hook:
                self.debug_hook.record_error(
                    ValueError(error_msg),
                    {'subject_id': subject_id, 'operation': 'create_training_samples_from_stacked'}
                )
            raise ValueError(error_msg)
        
        data_dir = os.path.join(stacked_dir, format_dir)
        num_fields = manifest['num_fields']
        
        # Log sample creation start
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "training_sample_creation_start",
                {
                    'subject_id': subject_id,
                    'num_fields': num_fields,
                    'data_dir': data_dir
                }
            )
        
        # Create processed data samples
        processed_samples = []
        
        for i in range(num_fields):
            # Load stacked data
            data_path = os.path.join(data_dir, f"stacked_{i}_b{bin_size}.{'pt' if is_torch else 'npy'}")
            stacked_data = loader(data_path)
            
            # Convert to numpy if needed
            if is_torch:
                if isinstance(stacked_data, dict):
                    input_features = stacked_data['dadt'].numpy()
                    target_efield = stacked_data['efield'].numpy()
                else:
                    # Extract E-field and dA/dt based on stacked format
                    stacked_info = manifest['stacked_info']
                    if stacked_info['shape'] == 'scalar_efield_vector_dadt':
                        # Format: [bin, bin, bin, 4] with [efield_magnitude, dadt_x, dadt_y, dadt_z]
                        input_features = stacked_data[:, :, :, 1:].numpy()  # dA/dt channels
                        target_efield = stacked_data[:, :, :, 0].unsqueeze(-1).numpy()  # E-field magnitude
                    elif stacked_info['shape'] == 'vector_efield_vector_dadt':
                        # Format: [bin, bin, bin, 6] with [efield_x, efield_y, efield_z, dadt_x, dadt_y, dadt_z]
                        input_features = stacked_data[:, :, :, 3:].numpy()  # dA/dt channels
                        target_efield = stacked_data[:, :, :, :3].numpy()  # E-field vector
                    else:
                        # Unknown format
                        if self.debug_hook:
                            self.debug_hook.record_error(
                                ValueError(f"Unknown stacked format: {stacked_info['shape']}"),
                                {'subject_id': subject_id, 'sample_idx': i}
                            )
                        continue
            else:
                # NumPy format handling
                if isinstance(stacked_data, dict):
                    input_features = stacked_data.item()['dadt']
                    target_efield = stacked_data.item()['efield']
                else:
                    # Extract based on shape
                    if stacked_data.shape[-1] == 4:
                        # [bin, bin, bin, 4] format
                        input_features = stacked_data[:, :, :, 1:]
                        target_efield = stacked_data[:, :, :, 0:1]
                    elif stacked_data.shape[-1] == 6:
                        # [bin, bin, bin, 6] format
                        input_features = stacked_data[:, :, :, 3:]
                        target_efield = stacked_data[:, :, :, :3]
                    else:
                        # Unknown format
                        if self.debug_hook:
                            self.debug_hook.record_error(
                                ValueError(f"Unknown stacked shape: {stacked_data.shape}"),
                                {'subject_id': subject_id, 'sample_idx': i}
                            )
                        continue
            
            # Create mask (all True for now, could be improved later)
            mask = np.ones_like(input_features[:, :, :, 0], dtype=bool)
            
            # Create processed data sample
            sample = TMSProcessedData(
                subject_id=subject_id,
                input_features=input_features,
                target_efield=target_efield,
                mask=mask,
                metadata={
                    'sample_id': f"{subject_id}_stacked_{i}",
                    'coil_position_idx': i,
                    'bin_size': bin_size,
                    'stacked_path': data_path
                }
            )
            
            processed_samples.append(sample)
        
        # Log completion
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "training_sample_creation_complete",
                {
                    'subject_id': subject_id,
                    'num_samples': len(processed_samples)
                }
            )
        
        return processed_samples
    
    def _reduce_memory(self, target_reduction: float) -> None:
        """
        Callback for memory reduction requests.
        
        Args:
            target_reduction: Fraction of memory to reduce (0.0-1.0)
        """
        # Log memory reduction request
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "memory_reduction_requested",
                {
                    'target_reduction': target_reduction,
                    'component': "training_data_generator"
                }
            )
        
        # Clear processing state to free memory
        self.processing_state = {}
        
        # Log completion
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "memory_reduction_complete",
                {'component': "training_data_generator"}
            )


# Function for multiprocessing worker
def process_field_worker(args: Tuple[int, np.ndarray, str]) -> int:
    """
    Worker function for parallel processing of fields.
    
    Args:
        args: Tuple containing (field_index, field_data, worker_data_path)
    
    Returns:
        Field index that was processed
    """
    field_index, field_data, worker_data_path = args
    
    # Load worker data
    import pickle
    with open(worker_data_path, 'rb') as f:
        worker_data = pickle.load(f)
    
    # Extract worker data
    voxel_coordinates = worker_data['voxel_coordinates']
    node_to_voxel_map = worker_data['node_to_voxel_map']
    voxel_node_counts = worker_data['voxel_node_counts']
    bin_size = worker_data['bin_size']
    region_bounds = worker_data['region_bounds']
    save_torch = worker_data['save_torch']
    save_numpy = worker_data['save_numpy']
    save_pickle = worker_data['save_pickle']
    torch_dir = worker_data['torch_dir']
    numpy_dir = worker_data['numpy_dir']
    pickle_dir = worker_data['pickle_dir']
    field_type = worker_data['field_type']
    
    # Check if field data is vector (3D) or scalar
    is_vector = len(field_data.shape) == 2 and field_data.shape[1] == 3
    
    # Process field data
    voxelized_data = process_field_optimized(
        field_data, node_to_voxel_map, voxel_node_counts, is_vector
    )
    
    # Get grid data (numpy array)
    grid_data = get_voxelized_data_as_grid(
        voxelized_data, bin_size, is_vector
    )
    
    # Save outputs in requested formats
    if save_pickle:
        # Save full metadata pickle
        import pickle
        output_path = os.path.join(pickle_dir, f"{field_type}_{field_index}_b{bin_size}.pkl")
        save_voxelized_data(
            voxelized_data, 
            voxel_coordinates, 
            is_vector,
            bin_size,
            region_bounds,
            output_path
        )
    
    if save_numpy:
        # Save NumPy array
        grid_output_path = os.path.join(numpy_dir, f"{field_type}_{field_index}_b{bin_size}.npy")
        np.save(grid_output_path, grid_data)
    
    if save_torch:
        # Convert to PyTorch tensor and save
        import torch
        tensor_data = torch.from_numpy(grid_data.copy())  # Use copy to ensure memory is contiguous
        tensor_output_path = os.path.join(torch_dir, f"{field_type}_{field_index}_b{bin_size}.pt")
        torch.save(tensor_data, tensor_output_path)
    
    return field_index


def process_field_optimized(
    field_array: np.ndarray, 
    node_to_voxel_map: Dict[int, int], 
    voxel_node_counts: Dict[int, int], 
    is_vector: bool
) -> Dict[int, np.ndarray]:
    """
    Optimized function to process a field array directly.
    
    Args:
        field_array: Field array (can be vector or scalar)
        node_to_voxel_map: Mapping from node indices to voxel indices
        voxel_node_counts: Number of nodes in each voxel
        is_vector: Whether the data is vector or scalar
    
    Returns:
        Voxelized field data with voxel indices as keys and aggregated values as values
    """
    # Group nodes by voxel for more efficient processing
    voxel_to_nodes = {}
    for node_idx, voxel_idx in node_to_voxel_map.items():
        if voxel_idx not in voxel_to_nodes:
            voxel_to_nodes[voxel_idx] = []
        voxel_to_nodes[voxel_idx].append(node_idx)
    
    # Initialize voxelized data dictionary
    voxelized_data = {}
    
    # Process each voxel
    for voxel_idx, node_indices in voxel_to_nodes.items():
        # Get values for all nodes in this voxel
        if is_vector:
            # For vector data, stack the vectors
            voxel_values = np.array([field_array[idx] for idx in node_indices])
            # Calculate mean vector
            voxel_mean = np.mean(voxel_values, axis=0)
        else:
            # For scalar data, get scalar values
            voxel_values = np.array([field_array[idx] for idx in node_indices])
            # Calculate mean scalar
            voxel_mean = np.mean(voxel_values)
        
        # Store result
        voxelized_data[voxel_idx] = voxel_mean
    
    return voxelized_data


def get_voxelized_data_as_grid(
    voxelized_data: Dict[int, np.ndarray], 
    bin_size: int, 
    is_vector: bool
) -> np.ndarray:
    """
    Convert voxelized data from dictionary format to a 3D grid.
    
    Args:
        voxelized_data: Voxelized data with voxel indices as keys and values as values
        bin_size: Number of bins in each dimension
        is_vector: Whether the data is vector or scalar
    
    Returns:
        3D grid with voxelized data
    """
    # Create empty grid
    if is_vector:
        vector_size = next(iter(voxelized_data.values())).size
        grid = np.zeros((bin_size, bin_size, bin_size, vector_size), dtype=float)
    else:
        grid = np.zeros((bin_size, bin_size, bin_size), dtype=float)
    
    # Fill grid with data
    for voxel_idx, value in voxelized_data.items():
        # Convert flat index to 3D index
        z = voxel_idx % bin_size
        y = (voxel_idx // bin_size) % bin_size
        x = voxel_idx // (bin_size**2)
        
        grid[x, y, z] = value
    
    return grid


def save_voxelized_data(
    voxelized_data: Dict[int, np.ndarray], 
    voxel_coordinates: Dict[int, Tuple[float, float, float]], 
    is_vector: bool, 
    bin_size: int, 
    region_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], 
    output_path: str
) -> None:
    """
    Save voxelized data to a pickle file with metadata.
    
    Args:
        voxelized_data: Voxelized data with voxel indices as keys and values as values
        voxel_coordinates: Voxel coordinates with voxel indices as keys and (x, y, z) as values
        is_vector: Whether the data is vector or scalar
        bin_size: Number of bins used for voxelization
        region_bounds: Bounds of the region of interest
        output_path: Path to save the voxelized data
    """
    # Create output dict with metadata
    output_data = {
        'voxelized_data': voxelized_data,
        'voxel_coordinates': voxel_coordinates,
        'is_vector': is_vector,
        'bin_size': bin_size,
        'region_bounds': region_bounds
    }
    
    # Save as pickle
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)