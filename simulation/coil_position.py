"""
Coil position handling module.

This module provides utilities for managing TMS coil positions
with explicit state tracking and resource monitoring.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R

# Project imports
from tms_efield_prediction.utils.debug.hooks import DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.simulation.tms_simulation import (
    SimulationContext, 
    rotate_grid, 
    generate_grid, 
    calc_matsimnibs
)


@dataclass
class CoilPositioningConfig:
    """Configuration for coil positioning."""
    search_radius: float = 50.0  # Radius of search area in mm
    spatial_resolution: float = 2.0  # Spacing between grid points in mm
    distance: float = 2.0  # Distance from skin in mm
    rotation_angles: np.ndarray = field(default_factory=lambda: np.arange(0, 360, 10))
    head_z_axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    grid_x_axis: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0]))
    grid_z_axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))


class CoilPositionGenerator:
    """Generates TMS coil positions with resource monitoring."""
    
    def __init__(
        self, 
        context: SimulationContext,
        config: CoilPositioningConfig,
        debug_hook: Optional[DebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """
        Initialize the coil position generator.
        
        Args:
            context: Simulation context
            config: Coil positioning configuration
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
        """
        self.context = context
        self.config = config
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Register with resource monitor if provided
        if resource_monitor:
            resource_monitor.register_component(
                "CoilPositionGenerator",
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
    
    def generate_positions(
        self, 
        mesh: Any, 
        roi_center: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate coil positions for TMS simulation.
        
        Args:
            mesh: SimNIBS mesh
            roi_center: ROI center information
            
        Returns:
            Tuple of (matsimnibs matrices, grid coordinates)
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("CoilPositionGenerator.generate_positions", "start")
        
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("generate_positions_start", {
                "subject_id": self.context.subject_id,
                "config": {
                    "search_radius": self.config.search_radius,
                    "spatial_resolution": self.config.spatial_resolution,
                    "distance": self.config.distance,
                    "rotation_angles": len(self.config.rotation_angles)
                }
            })
        
        try:
            # Calculate rotation matrix to align grid with skin normal
            from tms_efield_prediction.simulation.tms_simulation import rotate_grid, generate_grid
            
            rot_matrix = rotate_grid(
                self.config.grid_z_axis, 
                roi_center['skin_vec'],
                debug_hook=self.debug_hook,
                resource_monitor=self.resource_monitor
            )
            
            # Calculate new grid axes
            gridx_new = rot_matrix @ self.config.grid_x_axis
            gridy_new = np.cross(roi_center['skin_vec'], gridx_new)
            
            # Generate grid points
            points, s_grid = generate_grid(
                roi_center['skin'], 
                gridx_new, 
                gridy_new, 
                self.config.search_radius, 
                self.config.spatial_resolution,
                debug_hook=self.debug_hook,
                resource_monitor=self.resource_monitor
            )
            
            # Skip matsimnibs calculation for tests
            if len(points) == 0:
                # For tests, just return empty arrays
                matsimnibs_flat = np.zeros((len(self.config.rotation_angles), 0, 4, 4))
                grid = np.zeros((0, 3))
            else:
                # Calculate matsimnibs matrices - use a direct import to bypass the issue
                from tms_efield_prediction.simulation.tms_simulation import calc_matsimnibs
                
                matsimnibs = calc_matsimnibs(
                    mesh, 
                    points, 
                    distance=self.config.distance, 
                    rot_angles=self.config.rotation_angles, 
                    headz=self.config.head_z_axis,
                    debug_hook=self.debug_hook,
                    resource_monitor=self.resource_monitor
                )
                
                # Reshape to flatten the angles and positions
                matsimnibs_flat = matsimnibs.reshape(-1, 4, 4)
                
                # Create grid with angle as first dimension
                grid = np.stack([
                    [np.array([a, *s]) for s in s_grid] 
                    for a in self.config.rotation_angles
                ]).reshape(-1, 3)
            
            # Cache results
            self._cached_results = (matsimnibs_flat, grid)
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "generate_positions_complete", 
                    {
                        "matsimnibs_shape": matsimnibs_flat.shape,
                        "grid_shape": grid.shape,
                        "points_count": len(points),
                        "rotation_angles": len(self.config.rotation_angles)
                    }
                )
            
            return matsimnibs_flat, grid
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "CoilPositionGenerator.generate_positions",
                    "subject_id": self.context.subject_id,
                    "config": str(self.config)
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("CoilPositionGenerator.generate_positions", "end")
    
    def save_positions(
        self, 
        matsimnibs: np.ndarray, 
        grid: np.ndarray,
        save_path: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Save generated positions to files.
        
        Args:
            matsimnibs: Matsimnibs transformation matrices
            grid: Grid coordinates
            save_path: Path to save files (default: context.output_path)
            
        Returns:
            Tuple of (matsimnibs_path, grid_path)
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("CoilPositionGenerator.save_positions", "start")
        
        if save_path is None:
            save_path = self.context.output_path
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Generate filenames
            sub_id = self.context.subject_id
            matsimnibs_path = os.path.join(save_path, f"sub-{sub_id}_matsimnibs.npy")
            grid_path = os.path.join(save_path, f"sub-{sub_id}_grid.npy")
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("save_positions_start", {
                    "matsimnibs_path": matsimnibs_path,
                    "grid_path": grid_path
                })
            
            # Save files
            np.save(matsimnibs_path, matsimnibs)
            np.save(grid_path, grid)
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "save_positions_complete", 
                    {
                        "matsimnibs_path": matsimnibs_path,
                        "grid_path": grid_path,
                        "matsimnibs_shape": matsimnibs.shape,
                        "grid_shape": grid.shape
                    }
                )
            
            return matsimnibs_path, grid_path
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "CoilPositionGenerator.save_positions",
                    "subject_id": self.context.subject_id,
                    "save_path": save_path
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("CoilPositionGenerator.save_positions", "end")


def batch_positions(
    positions: np.ndarray, 
    n_batches: int,
    batch_index: int
) -> np.ndarray:
    """
    Split positions into batches for parallel processing.
    
    Args:
        positions: Array of positions
        n_batches: Number of batches
        batch_index: Index of batch to return (0-based)
        
    Returns:
        Batch of positions
    """
    if batch_index >= n_batches:
        raise ValueError(f"Batch index {batch_index} out of range (0-{n_batches-1})")
    
    # Split into batches
    batches = np.array_split(positions, n_batches)
    
    return batches[batch_index]