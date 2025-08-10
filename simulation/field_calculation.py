"""
Field calculation utilities for TMS simulation.

This module provides functions for calculating and processing
TMS-induced vector fields with resource monitoring.
"""

import os
import numpy as np
import h5py
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from joblib import Parallel, delayed
from tqdm import tqdm

# Import fmm3dpy if available (for fast field calculations)
try:
    import fmm3dpy
    FMM3D_AVAILABLE = True
except ImportError:
    FMM3D_AVAILABLE = False

# Project imports
from tms_efield_prediction.utils.debug.hooks import DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.simulation.tms_simulation import SimulationContext


@dataclass
class FieldCalculationConfig:
    """Configuration for field calculations."""
    didt: float = 1.49e6  # dI/dt value in A/s
    precision: float = 1e-3  # Precision for FMM calculation
    use_fmm: bool = True  # Whether to use FMM acceleration
    parallel_chunks: int = 1  # Number of chunks for parallel processing


class FieldCalculator:
    """Calculates TMS-induced vector fields with resource monitoring."""
    
    def __init__(
        self, 
        context: SimulationContext,
        config: FieldCalculationConfig,
        debug_hook: Optional[DebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """
        Initialize the field calculator.
        
        Args:
            context: Simulation context
            config: Field calculation configuration
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
        """
        self.context = context
        self.config = config
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Check if FMM is requested but not available
        if self.config.use_fmm and not FMM3D_AVAILABLE:
            if self.debug_hook:
                self.debug_hook.record_event("fmm_not_available", {
                    "message": "FMM3D not available, falling back to direct calculation"
                })
            self.config.use_fmm = False
        
        # Register with resource monitor if provided
        if resource_monitor:
            resource_monitor.register_component(
                "FieldCalculator",
                self._reduce_memory
            )
    
    def _reduce_memory(self, target_reduction: float) -> None:
        """
        Reduce memory usage.
        
        Args:
            target_reduction: Target reduction percentage
        """
        # Clear any cached results
        if hasattr(self, '_cached_results'):
            del self._cached_results
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def get_dAdt_from_coord(
        self,
        coil_matrices: np.ndarray,
        target_positions: np.ndarray,
        n_jobs: int = 1
    ) -> np.ndarray:
        """
        Calculate dA/dt at target positions for given coil matrices.
        
        Args:
            coil_matrices: Array of coil transformation matrices
            target_positions: Array of target positions
            n_jobs: Number of parallel jobs
            
        Returns:
            Array of dA/dt values at target positions
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("FieldCalculator.get_dAdt_from_coord", "start")
        
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("get_dAdt_from_coord_start", {
                "coil_matrices_count": len(coil_matrices),
                "target_positions_count": len(target_positions),
                "n_jobs": n_jobs
            })
        
        try:
            # Initialize result array
            A = np.zeros((len(coil_matrices), len(target_positions), 3))
            
            # Read coil data from file
            try:
                from simnibs.simulation import coil_numpy as coil_lib
                d_position, d_moment = coil_lib.read_ccd(self.context.coil_file_path)
            except ImportError:
                raise ImportError("SimNIBS coil_numpy module not available")
            
            # Convert dipole positions to homogeneous coordinates
            d_position = np.hstack([d_position * 1e3, np.ones((d_position.shape[0], 1))])
            
            # Precompute transformed positions for each coil matrix
            d_pos_all = np.array([
                coil_matrix.dot(d_position.T).T[:, :3] * 1e-3  # back to meters
                for coil_matrix in coil_matrices
            ])
            
            # Precompute rotated moments for each coil matrix
            d_mom_all = np.array([
                np.dot(d_moment, coil_matrix[:3, :3].T)
                for coil_matrix in coil_matrices
            ])
            
            # Calculate dA/dt for each coil position
            if n_jobs == 1 or len(coil_matrices) == 1:
                # Sequential processing
                for i in range(len(coil_matrices)):
                    A[i] = self._calculate_single_dAdt(
                        d_pos_all[i], 
                        d_mom_all[i], 
                        target_positions
                    )
            else:
                # Parallel processing
                chunks = min(n_jobs, len(coil_matrices))
                coil_chunks = np.array_split(np.arange(len(coil_matrices)), chunks)
                
                def process_chunk(chunk_indices):
                    chunk_results = np.zeros((len(chunk_indices), len(target_positions), 3))
                    for j, i in enumerate(chunk_indices):
                        chunk_results[j] = self._calculate_single_dAdt(
                            d_pos_all[i], 
                            d_mom_all[i], 
                            target_positions
                        )
                    return chunk_results, chunk_indices
                
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_chunk)(chunk) for chunk in coil_chunks
                )
                
                # Merge results
                for chunk_result, chunk_indices in results:
                    for j, i in enumerate(chunk_indices):
                        A[i] = chunk_result[j]
            
            # Store in cache
            self._cached_results = A
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "get_dAdt_from_coord_complete", 
                    {
                        "result_shape": A.shape,
                        "min_value": float(np.min(A)),
                        "max_value": float(np.max(A)),
                        "mean_value": float(np.mean(np.abs(A)))
                    }
                )
            
            return A
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "FieldCalculator.get_dAdt_from_coord",
                    "coil_matrices_count": len(coil_matrices),
                    "target_positions_count": len(target_positions)
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("FieldCalculator.get_dAdt_from_coord", "end")
    
    def _calculate_single_dAdt(
        self,
        d_pos: np.ndarray,
        d_mom: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        Calculate dA/dt for a single coil position.
        
        Args:
            d_pos: Dipole positions in meters
            d_mom: Dipole moments
            targets: Target positions in mm
            
        Returns:
            dA/dt values at targets
        """
        # Convert targets to meters
        targets_m = targets * 1e-3
        
        if self.config.use_fmm and FMM3D_AVAILABLE:
            # Use FMM for fast calculation
            out = fmm3dpy.lfmm3d(
                eps=self.config.precision,
                sources=d_pos.T,
                charges=d_mom.T,
                targets=targets_m.T,
                pgt=2,  # Include gradient
                nd=d_mom.shape[1]  # Number of dimensions
            )
            
            # Compute curl of A to get dA/dt components
            dAdt = np.zeros((len(targets), 3))
            dAdt[:, 0] = (out.gradtarg[1, 2] - out.gradtarg[2, 1])
            dAdt[:, 1] = (out.gradtarg[2, 0] - out.gradtarg[0, 2])
            dAdt[:, 2] = (out.gradtarg[0, 1] - out.gradtarg[1, 0])
            
        else:
            # Use direct calculation (slower but doesn't require FMM3D)
            dAdt = np.zeros((len(targets), 3))
            mu0 = 4 * np.pi * 1e-7  # Vacuum permeability
            
            # Calculate field for each dipole
            for j in range(len(d_pos)):
                r = targets_m - d_pos[j]
                r_norm = np.linalg.norm(r, axis=1)[:, np.newaxis]
                r_unit = r / r_norm
                
                # Dipole field formula
                m_cross_r = np.cross(d_mom[j], r_unit)
                field = 3 * np.einsum('ij,ij->i', d_mom[j], r_unit)[:, np.newaxis] * r_unit
                field -= d_mom[j]
                field *= mu0 / (4 * np.pi * r_norm**3)
                
                # Add to total field
                dAdt += field
        
        # Apply dI/dt scaling and return
        dAdt *= -1e-7 * self.config.didt
        
        return dAdt
    
    def save_dAdt_to_hdf5(
        self,
        dAdt_data: np.ndarray,
        save_path: Optional[str] = None,
        dataset_name: str = 'dAdt'
    ) -> str:
        """
        Save dA/dt data to HDF5 file.
        
        Args:
            dAdt_data: Array of dA/dt values
            save_path: Path to save file (default: context.output_path)
            dataset_name: Name of dataset in HDF5 file
            
        Returns:
            Path to saved file
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("FieldCalculator.save_dAdt_to_hdf5", "start")
        
        if save_path is None:
            save_path = self.context.output_path
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Generate filename
            h5_path = os.path.join(save_path, 'dAdts.h5')
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("save_dAdt_to_hdf5_start", {
                    "h5_path": h5_path,
                    "dataset_name": dataset_name,
                    "data_shape": dAdt_data.shape
                })
            
            # Save to HDF5 file
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset(dataset_name, data=dAdt_data)
            
            if self.debug_hook and self.debug_hook.should_sample():
                file_size_mb = 0
                if os.path.exists(h5_path):
                    file_size_mb = os.path.getsize(h5_path) / (1024 * 1024)
                
                self.debug_hook.record_event(
                    "save_dAdt_to_hdf5_complete", 
                    {
                        "h5_path": h5_path,
                        "file_size_mb": file_size_mb
                    }
                )
            
            return h5_path
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "FieldCalculator.save_dAdt_to_hdf5",
                    "save_path": save_path,
                    "dataset_name": dataset_name
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("FieldCalculator.save_dAdt_to_hdf5", "end")
    
    def load_dAdt_from_hdf5(
        self,
        file_path: str,
        dataset_name: str = 'dAdt'
    ) -> np.ndarray:
        """
        Load dA/dt data from HDF5 file.
        
        Args:
            file_path: Path to HDF5 file
            dataset_name: Name of dataset in HDF5 file
            
        Returns:
            Array of dA/dt values
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("FieldCalculator.load_dAdt_from_hdf5", "start")
        
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("load_dAdt_from_hdf5_start", {
                "file_path": file_path,
                "dataset_name": dataset_name
            })
        
        try:
            # Load from HDF5 file
            with h5py.File(file_path, 'r') as f:
                if dataset_name not in f:
                    available_datasets = list(f.keys())
                    raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file. "
                                    f"Available datasets: {available_datasets}")
                
                dAdt_data = f[dataset_name][:]
            
            # Store in cache
            self._cached_results = dAdt_data
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "load_dAdt_from_hdf5_complete", 
                    {
                        "data_shape": dAdt_data.shape,
                        "file_size_mb": os.path.getsize(file_path) / (1024 * 1024)
                    }
                )
            
            return dAdt_data
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "FieldCalculator.load_dAdt_from_hdf5",
                    "file_path": file_path,
                    "dataset_name": dataset_name
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("FieldCalculator.load_dAdt_from_hdf5", "end")


def calculate_field_magnitude(field_data: np.ndarray) -> np.ndarray:
    """
    Calculate magnitude of vector field.
    
    Args:
        field_data: Vector field data with shape [..., 3]
        
    Returns:
        Field magnitude with shape [...]
    """
    if field_data.shape[-1] != 3:
        raise ValueError(f"Expected vector field with last dimension 3, got {field_data.shape}")
    
    return np.linalg.norm(field_data, axis=-1)


def calculate_field_direction(field_data: np.ndarray) -> np.ndarray:
    """
    Calculate unit direction vectors of vector field.
    
    Args:
        field_data: Vector field data with shape [..., 3]
        
    Returns:
        Field direction with shape [..., 3]
    """
    if field_data.shape[-1] != 3:
        raise ValueError(f"Expected vector field with last dimension 3, got {field_data.shape}")
    
    # Calculate magnitude along last axis
    magnitude = np.linalg.norm(field_data, axis=-1, keepdims=True)
    
    # Handle zero magnitude vectors - reshape to match field_data dimensions
    non_zero = magnitude > 1e-15
    
    # Initialize result with zeros
    direction = np.zeros_like(field_data)
    
    # Only normalize non-zero vectors - handle broadcasting properly
    if np.any(non_zero):
        # Need to reshape indices for proper broadcasting
        indices = np.where(non_zero[..., 0])
        for idx in zip(*indices):
            direction[idx] = field_data[idx] / magnitude[idx]
    
    return direction