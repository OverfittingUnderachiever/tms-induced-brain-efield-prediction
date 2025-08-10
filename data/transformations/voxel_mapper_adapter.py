# tms_efield_prediction/data/transformations/voxel_mapper_adapter.py
"""
Adapter to use VoxelMapper as a drop-in replacement for MeshToGridTransformer.

This adapter allows the VoxelMapper to be used wherever MeshToGridTransformer is expected,
maintaining compatibility with existing code while leveraging the improved implementation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import os

from utils.debug.hooks import PipelineDebugHook
from utils.resource.monitor import ResourceMonitor
from utils.state.context import TMSPipelineContext
from .voxel_mapping import VoxelMapper, create_transform_matrix


class VoxelMapperAdapter:
    """
    Adapter class that provides the MeshToGridTransformer interface using VoxelMapper
    implementation for improved performance and memory usage.
    """
    
    def __init__(
        self, 
        context: TMSPipelineContext,
        debug_hook: Optional[PipelineDebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """
        Initialize the adapter with appropriate context and hooks.
        
        Args:
            context: TMS-specific pipeline context
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
        """
        self.context = context
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Will create VoxelMapper instances as needed
        self.mapper_cache = {}
        
        # Register with resource monitor if provided
        if self.resource_monitor:
            self.resource_monitor.register_component(
                "voxel_mapper_adapter",
                self._reduce_memory,
                priority=15  # Medium priority
            )
    
    def create_grid(self, node_centers: np.ndarray, n_bins: int) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Create regular grid for data voxelization.
        
        Args:
            node_centers: Centers of mesh nodes
            n_bins: Number of bins in each dimension
            
        Returns:
            Tuple of (grid_coords, bin_edges, bin_centers)
        """
        # Get cache key
        key = f"grid_{n_bins}"
        
        # Check cache
        if key in self.mapper_cache:
            mapper = self.mapper_cache[key]
        else:
            # Calculate bounds based on node centers
            mins = np.min(node_centers, axis=0)
            maxs = np.max(node_centers, axis=0)
            region_bounds = tuple((mins[i], maxs[i]) for i in range(3))
            
            # Create VoxelMapper
            mapper = VoxelMapper(
                bin_size=n_bins,
                region_bounds=region_bounds,
                context=self.context,
                debug_hook=self.debug_hook,
                resource_monitor=self.resource_monitor
            )
            
            # Cache for future use
            self.mapper_cache[key] = mapper
        
        # Create bins
        x_bounds, y_bounds, z_bounds = mapper.region_bounds
        bin_edges = [
            np.linspace(x_bounds[0], x_bounds[1], n_bins + 1),
            np.linspace(y_bounds[0], y_bounds[1], n_bins + 1),
            np.linspace(z_bounds[0], z_bounds[1], n_bins + 1)
        ]
        
        # Create bin centers
        bin_centers = [
            (edges[:-1] + edges[1:]) / 2 for edges in bin_edges
        ]
        
        # Create grid coordinates
        grid = np.meshgrid(*bin_centers, indexing='ij')
        grid_coords = np.stack(grid).reshape(3, -1).T
        
        return grid_coords, bin_edges, bin_centers
    
    def transform(
        self, 
        mesh_data: np.ndarray, 
        node_centers: np.ndarray, 
        n_bins: int = 64
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Complete mesh to grid transformation with single call.
        Mimics the interface of MeshToGridTransformer.transform().
        
        Args:
            mesh_data: Data values at mesh nodes
            node_centers: Centers of mesh nodes
            n_bins: Number of bins in each dimension
            
        Returns:
            Tuple of (voxelized_data, mask, metadata)
        """
        import time
        start_time = time.time()
        
        # Log transformation start
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "voxel_mapper_adapter_transform_start",
                {
                    'n_bins': n_bins,
                    'mesh_data_shape': mesh_data.shape,
                    'node_centers_shape': node_centers.shape
                }
            )
        
        # Calculate bounds based on node centers
        mins = np.min(node_centers, axis=0)
        maxs = np.max(node_centers, axis=0)
        region_bounds = tuple((mins[i], maxs[i]) for i in range(3))
        
        # Create a temporary VoxelMapper for this transform
        # We don't save any preprocessing since this is a one-off transform
        temp_key = f"temp_{n_bins}_{hash(str(mins))}"
        
        if temp_key in self.mapper_cache:
            mapper = self.mapper_cache[temp_key]
        else:
            mapper = VoxelMapper(
                bin_size=n_bins,
                region_bounds=region_bounds,
                context=self.context,
                debug_hook=self.debug_hook,
                resource_monitor=self.resource_monitor
            )
            
            # Create basic preprocessing (node_to_voxel_map)
            # Since we have node_centers directly, we can create a simplified mapping
            
            # Create voxel indices
            x_indices = np.clip(
                np.digitize(node_centers[:, 0], mapper.x_bins) - 1, 
                0, n_bins - 1
            )
            y_indices = np.clip(
                np.digitize(node_centers[:, 1], mapper.y_bins) - 1, 
                0, n_bins - 1
            )
            z_indices = np.clip(
                np.digitize(node_centers[:, 2], mapper.z_bins) - 1, 
                0, n_bins - 1
            )
            
            # Flat indices
            voxel_indices = (
                x_indices * n_bins**2 + 
                y_indices * n_bins + 
                z_indices
            )
            
            # Create node-to-voxel map
            mapper.node_to_voxel_map = {
                i: voxel_indices[i] for i in range(len(node_centers))
            }
            
            # Count nodes per voxel
            mapper.voxel_node_counts = {}
            for voxel_idx in voxel_indices:
                if voxel_idx in mapper.voxel_node_counts:
                    mapper.voxel_node_counts[voxel_idx] += 1
                else:
                    mapper.voxel_node_counts[voxel_idx] = 1
            
            # Cache for future use
            self.mapper_cache[temp_key] = mapper
        
        # Process data
        is_vector = len(mesh_data.shape) > 1 and mesh_data.shape[1] > 1
        
        if is_vector:
            # Handle vector data
            voxelized_data = np.zeros((n_bins, n_bins, n_bins, mesh_data.shape[1]), dtype=np.float32)
            
            # Process each vector component
            for i in range(mesh_data.shape[1]):
                component_data = mesh_data[:, i]
                voxelized_component = mapper.process_field(component_data, is_vector=False)
                component_grid = mapper.get_voxelized_data_as_grid(voxelized_component, is_vector=False)
                voxelized_data[..., i] = component_grid
        else:
            # Handle scalar data
            voxelized_dict = mapper.process_field(mesh_data, is_vector=False)
            voxelized_data = mapper.get_voxelized_data_as_grid(voxelized_dict, is_vector=False)
        
        # Create mask (voxels with data)
        mask = np.zeros((n_bins, n_bins, n_bins), dtype=bool)
        for voxel_idx in mapper.voxel_node_counts.keys():
            z = voxel_idx % n_bins
            y = (voxel_idx // n_bins) % n_bins
            x = voxel_idx // (n_bins**2)
            mask[x, y, z] = True
        
        # Create metadata
        metadata = {
            'grid_shape': (n_bins, n_bins, n_bins),
            'is_vector': is_vector,
            'bin_centers': [
                (mapper.x_bins[:-1] + mapper.x_bins[1:]) / 2,
                (mapper.y_bins[:-1] + mapper.y_bins[1:]) / 2,
                (mapper.z_bins[:-1] + mapper.z_bins[1:]) / 2
            ],
            'execution_time': time.time() - start_time,
            'n_bins': n_bins,
            'implementation': 'VoxelMapperAdapter'
        }
        
        # Log transformation completion
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "voxel_mapper_adapter_transform_complete",
                {
                    'output_shape': voxelized_data.shape,
                    'mask_shape': mask.shape,
                    'execution_time': metadata['execution_time']
                }
            )
        
        return voxelized_data, mask, metadata
    
    def voxelize_data(
        self, 
        data: np.ndarray, 
        node_centers: np.ndarray, 
        grid_coords: np.ndarray, 
        bin_edges: List[np.ndarray]
    ) -> np.ndarray:
        """
        Convert mesh data to regular grid format.
        Mimics the interface of MeshToGridTransformer.voxelize_data().
        
        Args:
            data: Data values at mesh nodes
            node_centers: Centers of mesh nodes
            grid_coords: Regular grid coordinates
            bin_edges: Bin edges for grid
            
        Returns:
            Data values on regular grid
        """
        # Calculate n_bins from bin_edges
        n_bins = len(bin_edges[0]) - 1
        
        # Calculate bounds from bin_edges
        region_bounds = tuple((edges[0], edges[-1]) for edges in bin_edges)
        
        # Create mapper
        key = f"voxelize_{n_bins}"
        if key in self.mapper_cache:
            mapper = self.mapper_cache[key]
        else:
            mapper = VoxelMapper(
                bin_size=n_bins,
                region_bounds=region_bounds,
                context=self.context,
                debug_hook=self.debug_hook,
                resource_monitor=self.resource_monitor
            )
            
            # Create mapping from node centers
            x_indices = np.clip(
                np.digitize(node_centers[:, 0], bin_edges[0]) - 1, 
                0, n_bins - 1
            )
            y_indices = np.clip(
                np.digitize(node_centers[:, 1], bin_edges[1]) - 1, 
                0, n_bins - 1
            )
            z_indices = np.clip(
                np.digitize(node_centers[:, 2], bin_edges[2]) - 1, 
                0, n_bins - 1
            )
            
            voxel_indices = (
                x_indices * n_bins**2 + 
                y_indices * n_bins + 
                z_indices
            )
            
            mapper.node_to_voxel_map = {
                i: voxel_indices[i] for i in range(len(node_centers))
            }
            
            mapper.voxel_node_counts = {}
            for voxel_idx in voxel_indices:
                if voxel_idx in mapper.voxel_node_counts:
                    mapper.voxel_node_counts[voxel_idx] += 1
                else:
                    mapper.voxel_node_counts[voxel_idx] = 1
            
            self.mapper_cache[key] = mapper
        
        # Process field
        is_vector = len(data.shape) > 1 and data.shape[1] > 1
        voxelized_dict = mapper.process_field(data, is_vector=is_vector)
        voxelized_data = mapper.get_voxelized_data_as_grid(voxelized_dict, is_vector=is_vector)
        
        return voxelized_data
    
    def generate_mask(
        self, 
        node_centers: np.ndarray, 
        grid_coords: np.ndarray, 
        bin_edges: List[np.ndarray]
    ) -> np.ndarray:
        """
        Generate binary mask for valid voxels.
        Mimics the interface of MeshToGridTransformer.generate_mask().
        
        Args:
            node_centers: Centers of mesh nodes
            grid_coords: Regular grid coordinates
            bin_edges: Bin edges for grid
            
        Returns:
            Binary mask
        """
        # Calculate n_bins from bin_edges
        n_bins = len(bin_edges[0]) - 1
        
        # Similar to voxelize_data, create mapping and calculate mask
        key = f"mask_{n_bins}"
        if key in self.mapper_cache:
            mapper = self.mapper_cache[key]
        else:
            # Calculate bounds from bin_edges
            region_bounds = tuple((edges[0], edges[-1]) for edges in bin_edges)
            
            mapper = VoxelMapper(
                bin_size=n_bins,
                region_bounds=region_bounds,
                context=self.context,
                debug_hook=self.debug_hook,
                resource_monitor=self.resource_monitor
            )
            
            # Create mapping from node centers
            x_indices = np.clip(
                np.digitize(node_centers[:, 0], bin_edges[0]) - 1, 
                0, n_bins - 1
            )
            y_indices = np.clip(
                np.digitize(node_centers[:, 1], bin_edges[1]) - 1, 
                0, n_bins - 1
            )
            z_indices = np.clip(
                np.digitize(node_centers[:, 2], bin_edges[2]) - 1, 
                0, n_bins - 1
            )
            
            voxel_indices = (
                x_indices * n_bins**2 + 
                y_indices * n_bins + 
                z_indices
            )
            
            mapper.voxel_node_counts = {}
            for voxel_idx in voxel_indices:
                if voxel_idx in mapper.voxel_node_counts:
                    mapper.voxel_node_counts[voxel_idx] += 1
                else:
                    mapper.voxel_node_counts[voxel_idx] = 1
            
            self.mapper_cache[key] = mapper
        
        # Create mask
        mask = np.zeros((n_bins, n_bins, n_bins), dtype=bool)
        for voxel_idx in mapper.voxel_node_counts.keys():
            z = voxel_idx % n_bins
            y = (voxel_idx // n_bins) % n_bins
            x = voxel_idx // (n_bins**2)
            mask[x, y, z] = True
        
        return mask
    
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
                    'component': "voxel_mapper_adapter"
                }
            )
        
        # Clear temporary mappers from cache to free memory
        temp_keys = [key for key in list(self.mapper_cache.keys()) if key.startswith('temp_')]
        for key in temp_keys:
            del self.mapper_cache[key]
        
        # If still needed, clear all but the most recently used mappers
        if target_reduction > 0.5 and len(self.mapper_cache) > 1:
            # Keep just one mapper
            keep_key = list(self.mapper_cache.keys())[0]
            keys_to_clear = [key for key in self.mapper_cache if key != keep_key]
            for key in keys_to_clear:
                del self.mapper_cache[key]
        
        # Log completion
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "memory_reduction_complete",
                {
                    'component': "voxel_mapper_adapter",
                    'remaining_mappers': len(self.mapper_cache)
                }
            )