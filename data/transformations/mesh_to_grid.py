# tms_efield_prediction/data/transformations/mesh_to_grid.py
"""
Mesh to grid transformation implementation.

This module provides functionality to transform mesh-based data to regular grid format.
The transformations maintain resource awareness and debug capability integration.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tms_efield_prediction.utils.debug.hooks import PipelineDebugHook, DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.utils.pipeline.implementation_unit import ImplementationUnit, UnitResult, UnitPipeline
from tms_efield_prediction.utils.state.context import TMSPipelineContext, PipelineContext, PipelineState
from tms_efield_prediction.utils.debug.context import PipelineDebugState


class MeshToGridTransformer:
    """Transforms mesh-based data to regular 3D grid format."""
    
    def __init__(
    self, 
    context: TMSPipelineContext,
    debug_hook: Optional[PipelineDebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
    ):
        """Initialize with appropriate context and hooks.
        
        Args:
            context: TMS-specific pipeline context
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
        """
        self.context = context
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Track internal memory usage
        self._memory_usage = 0
        self._intermediate_data = {}
        
    def create_grid(self, node_centers: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create regular grid for data voxelization.
        
        Args:
            node_centers: Centers of mesh nodes
            n_bins: Number of bins in each dimension
            
        Returns:
            Tuple of (grid_coords, bin_edges, bin_centers)
        """
        # Create implementation unit
        grid_creator = ImplementationUnit(
            transform_fn=lambda x: self._make_grid(x, n_bins),
            name="grid_creator",
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        # Execute transformation
        result = grid_creator(node_centers)
        
        # Update memory tracking
        self._update_memory_usage("grid_data", result.output)
        
        return result.output
    
    def voxelize_data(
        self, 
        data: np.ndarray, 
        node_centers: np.ndarray, 
        grid_coords: np.ndarray, 
        bin_edges: np.ndarray
    ) -> np.ndarray:
        """
        Convert mesh data to regular grid format.
        
        Args:
            data: Data values at mesh nodes
            node_centers: Centers of mesh nodes
            grid_coords: Regular grid coordinates
            bin_edges: Bin edges for grid
            
        Returns:
            Data values on regular grid
        """
        # Create implementation unit
        voxelizer = ImplementationUnit(
            transform_fn=lambda inputs: self._voxelize_data_impl(*inputs),
            name="data_voxelizer",
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        # Execute transformation
        result = voxelizer((data, node_centers, grid_coords, bin_edges))
        
        # Update memory tracking
        self._update_memory_usage("voxelized_data", result.output)
        
        return result.output
    
    def generate_mask(
        self, 
        node_centers: np.ndarray, 
        grid_coords: np.ndarray, 
        bin_edges: np.ndarray
    ) -> np.ndarray:
        """
        Generate binary mask for valid voxels.
        
        Args:
            node_centers: Centers of mesh nodes
            grid_coords: Regular grid coordinates
            bin_edges: Bin edges for grid
            
        Returns:
            Binary mask
        """
        # Create implementation unit
        mask_generator = ImplementationUnit(
            transform_fn=lambda inputs: self._generate_mask_impl(*inputs),
            name="mask_generator",
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        # Execute transformation
        result = mask_generator((node_centers, grid_coords, bin_edges))
        
        # Update memory tracking
        self._update_memory_usage("mask_data", result.output)
        
        return result.output
    
    def create_pipeline(self) -> UnitPipeline:
        """
        Create a complete transformation pipeline.
        
        Returns:
            UnitPipeline: Complete transformation pipeline
        """
        # Create implementation units
        grid_creator = ImplementationUnit(
            transform_fn=lambda x: self._make_grid(x['node_centers'], x['n_bins']),
            name="grid_creator",
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        voxelizer = ImplementationUnit(
            transform_fn=lambda x: self._voxelize_data_impl(
                x['data'], x['node_centers'], x['grid_output'][0], x['grid_output'][1]
            ),
            name="data_voxelizer",
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        mask_generator = ImplementationUnit(
            transform_fn=lambda x: self._generate_mask_impl(
                x['node_centers'], x['grid_output'][0], x['grid_output'][1]
            ),
            name="mask_generator",
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        # Build pipeline
        pipeline = UnitPipeline(
            units=[grid_creator, voxelizer, mask_generator],
            name="mesh_to_grid_pipeline",
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        return pipeline
    
    def transform(
        self, 
        mesh_data: np.ndarray, 
        node_centers: np.ndarray, 
        n_bins: int = 64
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Complete mesh to grid transformation with single call.
        
        Args:
            mesh_data: Data values at mesh nodes
            node_centers: Centers of mesh nodes
            n_bins: Number of bins in each dimension
            
        Returns:
            Tuple of (voxelized_data, mask, metadata)
        """
        start_time = time.time()
        
        # Create grid
        grid_coords, bin_edges, bin_centers = self.create_grid(node_centers, n_bins)
        
        # Voxelize data (might be vector or scalar)
        is_vector = len(mesh_data.shape) > 1 and mesh_data.shape[1] > 1
        
        if is_vector:
            # Handle vector data (e.g., E-field)
            voxelized_data = np.zeros((n_bins, n_bins, n_bins, mesh_data.shape[1]), dtype=np.float32)
            for i in range(mesh_data.shape[1]):
                voxelized_data[..., i] = self.voxelize_data(
                    mesh_data[:, i], node_centers, grid_coords, bin_edges
                )
        else:
            # Handle scalar data
            voxelized_data = self.voxelize_data(mesh_data, node_centers, grid_coords, bin_edges)
        
        # Generate mask
        mask = self.generate_mask(node_centers, grid_coords, bin_edges)
        
        # Create metadata
        metadata = {
            'grid_shape': (n_bins, n_bins, n_bins),
            'is_vector': is_vector,
            'bin_centers': bin_centers,
            'execution_time': time.time() - start_time,
            'n_bins': n_bins
        }
        
        # Log if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "mesh_to_grid_transform_complete",
                {
                    'mesh_data_shape': mesh_data.shape,
                    'output_shape': voxelized_data.shape,
                    'mask_shape': mask.shape,
                    'execution_time': metadata['execution_time']
                }
            )
        
        return voxelized_data, mask, metadata
    
    def _make_grid(self, node_centers: np.ndarray, n_bins: int) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Implementation of grid creation.
        
        Args:
            node_centers: Centers of mesh nodes
            n_bins: Number of bins in each dimension
            
        Returns:
            Tuple of (grid_coords, bin_edges, bin_centers)
        """
        # Find min and max extent of nodes
        mins = np.min(node_centers, axis=0)
        maxs = np.max(node_centers, axis=0)
        
        # Create bin edges along each dimension
        bin_edges = []
        bin_centers = []
        for dim in range(3):
            edges = np.linspace(mins[dim], maxs[dim], n_bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            bin_edges.append(edges)
            bin_centers.append(centers)
        
        # Create the grid coordinates using the centers
        grid = np.meshgrid(*bin_centers, indexing='ij')
        grid_coords = np.stack(grid).reshape(3, -1).T
        
        # Log if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "grid_creation_complete",
                {
                    'mins': mins,
                    'maxs': maxs,
                    'n_bins': n_bins,
                    'bin_edges_shapes': [edges.shape for edges in bin_edges],
                    'grid_coords_shape': grid_coords.shape
                }
            )
        
        return grid_coords, bin_edges, bin_centers
    
    def _voxelize_data_impl(
        self, 
        data: np.ndarray, 
        node_centers: np.ndarray, 
        grid_coords: np.ndarray, 
        bin_edges: List[np.ndarray]
    ) -> np.ndarray:
        """
        Implementation of data voxelization.
        
        Args:
            data: Data values at mesh nodes
            node_centers: Centers of mesh nodes
            grid_coords: Regular grid coordinates
            bin_edges: List of bin edges for each dimension
            
        Returns:
            Voxelized data
        """
        n_bins = len(bin_edges[0]) - 1
        voxelized = np.zeros((n_bins, n_bins, n_bins), dtype=np.float32)
        counts = np.zeros((n_bins, n_bins, n_bins), dtype=np.int32)
        
        # Process data in chunks to reduce memory pressure
        chunk_size = 10000  # Adjust based on available memory
        for i in range(0, len(node_centers), chunk_size):
            end_idx = min(i + chunk_size, len(node_centers))
            chunk_centers = node_centers[i:end_idx]
            chunk_data = data[i:end_idx]
            
            # Find bin indices for each node
            bin_indices = np.zeros((end_idx - i, 3), dtype=np.int32)
            for dim in range(3):
                bin_indices[:, dim] = np.digitize(chunk_centers[:, dim], bin_edges[dim]) - 1
            
            # Filter out points outside the grid
            valid_points = (
                (bin_indices[:, 0] >= 0) & (bin_indices[:, 0] < n_bins) &
                (bin_indices[:, 1] >= 0) & (bin_indices[:, 1] < n_bins) &
                (bin_indices[:, 2] >= 0) & (bin_indices[:, 2] < n_bins)
            )
            
            valid_indices = bin_indices[valid_points]
            valid_data = chunk_data[valid_points]
            
            # Accumulate data and counts
            for j in range(len(valid_indices)):
                x, y, z = valid_indices[j]
                voxelized[x, y, z] += valid_data[j]
                counts[x, y, z] += 1
            
            # Free memory for next chunk
            del chunk_centers, chunk_data, bin_indices, valid_points, valid_indices, valid_data
        
        # Compute averages for cells with data
        mask = counts > 0
        voxelized[mask] /= counts[mask]
        
        # Handle NaN values (shouldn't happen with proper masking)
        voxelized = np.nan_to_num(voxelized, nan=0.0)
        
        # Log if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "voxelization_complete",
                {
                    'input_shape': data.shape,
                    'output_shape': voxelized.shape,
                    'non_zero_voxels': np.count_nonzero(mask),
                    'max_value': np.max(voxelized),
                    'min_value': np.min(voxelized[mask]) if np.any(mask) else 0.0
                }
            )
        
        return voxelized
    
    def _generate_mask_impl(
        self, 
        node_centers: np.ndarray, 
        grid_coords: np.ndarray, 
        bin_edges: List[np.ndarray]
    ) -> np.ndarray:
        """
        Implementation of mask generation.
        
        Args:
            node_centers: Centers of mesh nodes
            grid_coords: Regular grid coordinates
            bin_edges: List of bin edges for each dimension
            
        Returns:
            Binary mask
        """
        n_bins = len(bin_edges[0]) - 1
        mask = np.zeros((n_bins, n_bins, n_bins), dtype=bool)
        
        # Process data in chunks to reduce memory pressure
        chunk_size = 10000  # Adjust based on available memory
        for i in range(0, len(node_centers), chunk_size):
            end_idx = min(i + chunk_size, len(node_centers))
            chunk_centers = node_centers[i:end_idx]
            
            # Find bin indices for each node
            bin_indices = np.zeros((end_idx - i, 3), dtype=np.int32)
            for dim in range(3):
                bin_indices[:, dim] = np.digitize(chunk_centers[:, dim], bin_edges[dim]) - 1
            
            # Filter out points outside the grid
            valid_points = (
                (bin_indices[:, 0] >= 0) & (bin_indices[:, 0] < n_bins) &
                (bin_indices[:, 1] >= 0) & (bin_indices[:, 1] < n_bins) &
                (bin_indices[:, 2] >= 0) & (bin_indices[:, 2] < n_bins)
            )
            
            valid_indices = bin_indices[valid_points]
            
            # Set mask where nodes exist
            for j in range(len(valid_indices)):
                x, y, z = valid_indices[j]
                mask[x, y, z] = True
            
            # Free memory for next chunk
            del chunk_centers, bin_indices, valid_points, valid_indices
        
        # Optional: Dilate mask to fill small gaps
        if self.context.config.get('mask_dilation', False):
            from scipy import ndimage
            mask = ndimage.binary_dilation(mask, iterations=self.context.config.get('dilation_iterations', 1))
        
        # Log if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "mask_generation_complete",
                {
                    'mask_shape': mask.shape,
                    'non_zero_voxels': np.count_nonzero(mask),
                    'coverage_percent': 100 * np.count_nonzero(mask) / mask.size
                }
            )
        
        return mask
        
        
    def _update_memory_usage(self, key: str, data: Any) -> None:
        """
        Update memory usage tracking.
        
        Args:
            key: Identifier for the data object
            data: Data to track (can be array or tuple)
        """
        # Calculate memory usage (approximate)
        if isinstance(data, tuple):
            # Handle tuple of arrays
            mem_bytes = sum(arr.nbytes for arr in data if hasattr(arr, 'nbytes'))
        elif hasattr(data, 'nbytes'):
            # Handle single array
            mem_bytes = data.nbytes
        else:
            # Handle other data types
            mem_bytes = sys.getsizeof(data)
        
        # Store in intermediate data dictionary
        if key in self._intermediate_data:
            # Subtract old usage
            old_usage = self._intermediate_data[key]['bytes']
            self._memory_usage -= old_usage
        
        # Add new usage
        shape = data.shape if hasattr(data, 'shape') else str(type(data))
        dtype = data.dtype if hasattr(data, 'dtype') else str(type(data))
        
        self._intermediate_data[key] = {
            'shape': shape,
            'dtype': dtype,
            'bytes': mem_bytes
        }
        self._memory_usage += mem_bytes
        
        # Update resource monitor if available
        if self.resource_monitor:
            self.resource_monitor.update_component_usage(
                "mesh_to_grid_transformer", 
                self._memory_usage
            )
    
        def _reduce_memory(target_reduction: float) -> None:
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
                        'current_memory': self._memory_usage
                    }
                )
            
            # Clear non-essential intermediate data
            keys_to_clear = []
            for key, info in self._intermediate_data.items():
                # Skip essential data (could be more sophisticated)
                if key in ['voxelized_data', 'mask_data']:
                    continue
                    
                keys_to_clear.append(key)
                self._memory_usage -= info['bytes']
                
                # Check if we've met the target
                if self._memory_usage / (1.0 - target_reduction) <= self._memory_usage:
                    break
            
            # Actually clear the data
            for key in keys_to_clear:
                del self._intermediate_data[key]
            
            # Update resource monitor
            if self.resource_monitor:
                self.resource_monitor.update_component_usage(
                    "mesh_to_grid_transformer", 
                    self._memory_usage
                )
            
            # Log results
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "memory_reduction_complete",
                    {
                        'cleared_keys': keys_to_clear,
                        'new_memory': self._memory_usage
                    }
                )
        
        # Store the _reduce_memory function as an instance method
        self._reduce_memory = _reduce_memory
        
        # Register with resource monitor if provided
        if self.resource_monitor:
            self.resource_monitor.register_component(
                "mesh_to_grid_transformer",
                self._reduce_memory,
                priority=10  # Higher priority for this component
            )