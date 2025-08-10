"""
Field comparison visualization module for TMS E-field prediction.

This module provides specialized visualization capabilities for comparing
dA/dt and E-field data from TMS simulations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import logging
import h5py

# Project imports
from utils.debug.hooks import DebugHook
from utils.resource.monitor import ResourceMonitor
from data.formats.simnibs_io import load_mesh, load_dadt_data
from simulation.field_calculation import calculate_field_magnitude, calculate_field_direction
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tms_field_visualizer')


@dataclass
class FieldVisualizationConfig:
    """Configuration for field visualization."""
    slice_view: str = 'axial'  # 'axial', 'coronal', 'sagittal'
    colormap_dadt: str = 'plasma'
    colormap_efield: str = 'hot'
    colormap_comparison: str = 'coolwarm'
    alpha: float = 0.7  # Transparency for overlays
    slice_index: Optional[int] = None  # Auto-select middle slice if None
    roi_radius: float = 20.0  # ROI radius in mm
    vector_stride: int = 5  # Stride for vector field visualization


@dataclass
class VisualizationData:
    """Container for visualization data."""
    dadt_data: Optional[np.ndarray] = None
    efield_data: Optional[np.ndarray] = None
    mesh_data: Optional[Dict] = None
    mask_data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FieldVisualizer:
    """Field visualization class with resource monitoring and debug hooks."""
    
    def __init__(
        self, 
        config: FieldVisualizationConfig,
        debug_hook: Optional[DebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """
        Initialize field visualizer.
        
        Args:
            config: Visualization configuration
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
        """
        self.config = config
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Register with resource monitor if provided
        if resource_monitor:
            resource_monitor.register_component(
                "FieldVisualizer",
                self._reduce_memory
            )
    
    def _reduce_memory(self, target_reduction: float) -> None:
        """
        Reduce memory usage when requested by resource monitor.
        
        Args:
            target_reduction: Target reduction percentage
        """
        # Clear any cached data
        if hasattr(self, '_cache'):
            del self._cache
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def load_simulation_data(
        self,
        dadt_file_path: Optional[str] = None,
        efield_file_path: Optional[str] = None,
        mesh_file_path: Optional[str] = None,
        mask_file_path: Optional[str] = None
    ) -> VisualizationData:
        """
        Load simulation data for visualization.
        
        Args:
            dadt_file_path: Path to dA/dt data file (.h5 or .npy)
            efield_file_path: Path to E-field data file (.npy)
            mesh_file_path: Path to mesh file (.msh)
            mask_file_path: Path to mask file (.npy)
            
        Returns:
            VisualizationData object
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("FieldVisualizer.load_simulation_data", "start")
        
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("load_simulation_data_start", {
                "dadt_file": dadt_file_path,
                "efield_file": efield_file_path,
                "mesh_file": mesh_file_path,
                "mask_file": mask_file_path
            })
        
        visualization_data = VisualizationData()
        
        try:
            # Load dA/dt data if provided
            if dadt_file_path and os.path.exists(dadt_file_path):
                try:
                    dadt_data = load_dadt_data(
                        dadt_file_path,
                        debug_hook=self.debug_hook,
                        resource_monitor=self.resource_monitor
                    )
                    visualization_data.dadt_data = dadt_data
                    visualization_data.metadata['dadt_shape'] = dadt_data.shape
                    logger.info(f"Loaded dA/dt data with shape {dadt_data.shape}")
                except Exception as e:
                    logger.error(f"Error loading dA/dt data: {e}")
            
            # Load E-field data if provided
            if efield_file_path and os.path.exists(efield_file_path):
                try:
                    efield_data = np.load(efield_file_path)
                    visualization_data.efield_data = efield_data
                    visualization_data.metadata['efield_shape'] = efield_data.shape
                    logger.info(f"Loaded E-field data with shape {efield_data.shape}")
                except Exception as e:
                    logger.error(f"Error loading E-field data: {e}")
            
            # Load mesh data if provided
            if mesh_file_path and os.path.exists(mesh_file_path):
                try:
                    mesh_data = load_mesh(
                        mesh_file_path,
                        debug_hook=self.debug_hook,
                        resource_monitor=self.resource_monitor
                    )
                    visualization_data.mesh_data = mesh_data
                    visualization_data.metadata['node_count'] = len(mesh_data.nodes)
                    logger.info(f"Loaded mesh data with {len(mesh_data.nodes)} nodes")
                except Exception as e:
                    logger.error(f"Error loading mesh data: {e}")
            
            # Load mask data if provided
            if mask_file_path and os.path.exists(mask_file_path):
                try:
                    mask_data = np.load(mask_file_path)
                    visualization_data.mask_data = mask_data
                    visualization_data.metadata['mask_shape'] = mask_data.shape
                    logger.info(f"Loaded mask data with shape {mask_data.shape}")
                except Exception as e:
                    logger.error(f"Error loading mask data: {e}")
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("load_simulation_data_complete", {
                    "dadt_loaded": visualization_data.dadt_data is not None,
                    "efield_loaded": visualization_data.efield_data is not None,
                    "mesh_loaded": visualization_data.mesh_data is not None,
                    "mask_loaded": visualization_data.mask_data is not None
                })
            
            return visualization_data
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "FieldVisualizer.load_simulation_data",
                    "dadt_file": dadt_file_path,
                    "efield_file": efield_file_path
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("FieldVisualizer.load_simulation_data", "end")
    
    def visualize_field_comparison(
        self,
        data: VisualizationData,
        output_path: Optional[str] = None,
        position_index: int = 0,
        component_index: Optional[int] = None,
        show_magnitude: bool = True,
        show_vectors: bool = False
    ) -> plt.Figure:
        """
        Create comparative visualization of dA/dt and E-field data.
        
        Args:
            data: Visualization data
            output_path: Optional path to save visualization
            position_index: Index of position to visualize
            component_index: Optional component index (0=x, 1=y, 2=z)
            show_magnitude: Whether to show magnitude
            show_vectors: Whether to show vector field
            
        Returns:
            Matplotlib figure
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("FieldVisualizer.visualize_field_comparison", "start")
        
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("visualize_field_comparison_start", {
                "position_index": position_index,
                "component_index": component_index,
                "show_magnitude": show_magnitude,
                "show_vectors": show_vectors
            })
        
        try:
            # Check if we have both dA/dt and E-field data
            if data.dadt_data is None or data.efield_data is None:
                error_msg = "Both dA/dt and E-field data are required for comparison"
                if self.debug_hook:
                    self.debug_hook.record_error(ValueError(error_msg), {
                        "component": "FieldVisualizer.visualize_field_comparison"
                    })
                raise ValueError(error_msg)
            
            # Handle multiple positions
            dadt = data.dadt_data
            efield = data.efield_data
            
            # Check if these are arrays of multiple positions
            if len(dadt.shape) > 3:
                if position_index >= dadt.shape[0]:
                    raise ValueError(f"Position index {position_index} is out of range for dA/dt data with shape {dadt.shape}")
                dadt = dadt[position_index]
            
            if len(efield.shape) > 3:
                if position_index >= efield.shape[0]:
                    raise ValueError(f"Position index {position_index} is out of range for E-field data with shape {efield.shape}")
                efield = efield[position_index]
            
            # Create mask if not provided
            mask = data.mask_data
            if mask is None:
                # Create a simple mask based on non-zero values in dA/dt data
                mask = np.any(np.abs(dadt) > 0, axis=-1) if len(dadt.shape) > 2 else np.abs(dadt) > 0
            
            # Determine which slice to show based on configuration
            slice_view = self.config.slice_view.lower()
            if slice_view not in ['axial', 'coronal', 'sagittal']:
                slice_view = 'axial'  # Default to axial
            
            # Determine slice indices
            if slice_view == 'axial':
                slice_axis = 2
            elif slice_view == 'coronal':
                slice_axis = 1
            else:  # sagittal
                slice_axis = 0
            
            # Determine slice index if not specified
            slice_index = self.config.slice_index
            if slice_index is None or slice_index >= dadt.shape[slice_axis]:
                slice_index = dadt.shape[slice_axis] // 2
            
            # Extract slices
            if slice_axis == 0:
                dadt_slice = dadt[slice_index, :, :]
                efield_slice = efield[slice_index, :, :]
                mask_slice = mask[slice_index, :, :]
            elif slice_axis == 1:
                dadt_slice = dadt[:, slice_index, :]
                efield_slice = efield[:, slice_index, :]
                mask_slice = mask[:, slice_index, :]
            else:  # slice_axis == 2
                dadt_slice = dadt[:, :, slice_index]
                efield_slice = efield[:, :, slice_index]
                mask_slice = mask[:, :, slice_index]
            
            # Handle vector field data (if shape includes components)
            dadt_has_components = len(dadt.shape) > 3 or (len(dadt.shape) == 3 and dadt.shape[-1] == 3)
            efield_has_components = len(efield.shape) > 3 or (len(efield.shape) == 3 and efield.shape[-1] == 3)
            
            # Calculate magnitude if showing vectors
            if dadt_has_components:
                dadt_mag = calculate_field_magnitude(dadt_slice)
            else:
                dadt_mag = dadt_slice
            
            if efield_has_components:
                efield_mag = calculate_field_magnitude(efield_slice)
            else:
                efield_mag = efield_slice
            
            # Apply mask
            dadt_mag = dadt_mag * mask_slice
            efield_mag = efield_mag * mask_slice
            
            # Create figure and axes
            fig = plt.figure(figsize=(18, 10))
            
            if show_vectors and show_magnitude:
                # Show magnitudes and vectors
                gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
                ax_dadt_mag = plt.subplot(gs[0, 0])
                ax_efield_mag = plt.subplot(gs[0, 1])
                ax_comparison = plt.subplot(gs[0, 2])
                ax_dadt_vec = plt.subplot(gs[1, 0])
                ax_efield_vec = plt.subplot(gs[1, 1])
                ax_overlay = plt.subplot(gs[1, 2])
            else:
                # Show either magnitudes or vectors
                gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
                ax_dadt = plt.subplot(gs[0, 0])
                ax_efield = plt.subplot(gs[0, 1])
                ax_comparison = plt.subplot(gs[0, 2])
            
            # Plot data based on what we're showing
            if show_magnitude:
                if show_vectors:
                    # Magnitude plots
                    im_dadt = ax_dadt_mag.imshow(dadt_mag.T, cmap=self.config.colormap_dadt, origin='lower')
                    plt.colorbar(im_dadt, ax=ax_dadt_mag, label='dA/dt magnitude')
                    ax_dadt_mag.set_title('dA/dt Magnitude')
                    
                    im_efield = ax_efield_mag.imshow(efield_mag.T, cmap=self.config.colormap_efield, origin='lower')
                    plt.colorbar(im_efield, ax=ax_efield_mag, label='E-field magnitude')
                    ax_efield_mag.set_title('E-field Magnitude')
                    
                    # Vector plots
                    if component_index is not None and dadt_has_components and efield_has_components:
                        self._plot_vector_component(
                            ax_dadt_vec, dadt_slice, mask_slice, component_index, 
                            title=f'dA/dt Component {component_index}'
                        )
                        self._plot_vector_component(
                            ax_efield_vec, efield_slice, mask_slice, component_index, 
                            title=f'E-field Component {component_index}'
                        )
                    else:
                        self._plot_vector_field(
                            ax_dadt_vec, dadt_slice, mask_slice, 
                            title='dA/dt Vector Field'
                        )
                        self._plot_vector_field(
                            ax_efield_vec, efield_slice, mask_slice, 
                            title='E-field Vector Field'
                        )
                    
                    # Overlay plot
                    self._plot_field_overlay(
                        ax_overlay, dadt_mag, efield_mag, mask_slice, 
                        title='Magnitude Overlay'
                    )
                else:
                    # Only magnitude plots
                    im_dadt = ax_dadt.imshow(dadt_mag.T, cmap=self.config.colormap_dadt, origin='lower')
                    plt.colorbar(im_dadt, ax=ax_dadt, label='dA/dt magnitude')
                    ax_dadt.set_title('dA/dt Magnitude')
                    
                    im_efield = ax_efield.imshow(efield_mag.T, cmap=self.config.colormap_efield, origin='lower')
                    plt.colorbar(im_efield, ax=ax_efield, label='E-field magnitude')
                    ax_efield.set_title('E-field Magnitude')
            elif show_vectors:
                # Only vector plots
                if component_index is not None and dadt_has_components and efield_has_components:
                    self._plot_vector_component(
                        ax_dadt, dadt_slice, mask_slice, component_index, 
                        title=f'dA/dt Component {component_index}'
                    )
                    self._plot_vector_component(
                        ax_efield, efield_slice, mask_slice, component_index, 
                        title=f'E-field Component {component_index}'
                    )
                else:
                    self._plot_vector_field(
                        ax_dadt, dadt_slice, mask_slice, 
                        title='dA/dt Vector Field'
                    )
                    self._plot_vector_field(
                        ax_efield, efield_slice, mask_slice, 
                        title='E-field Vector Field'
                    )
            
            # Comparison plot (correlation or difference)
            # Normalize the data to [0, 1] range for better comparison
            dadt_norm = (dadt_mag - np.min(dadt_mag)) / (np.max(dadt_mag) - np.min(dadt_mag) + 1e-10)
            efield_norm = (efield_mag - np.min(efield_mag)) / (np.max(efield_mag) - np.min(efield_mag) + 1e-10)
            
            # Calculate difference
            difference = dadt_norm - efield_norm
            
            # Plot difference
            im_diff = ax_comparison.imshow(
                difference.T, 
                cmap=self.config.colormap_comparison, 
                origin='lower',
                norm=Normalize(vmin=-1, vmax=1)
            )
            plt.colorbar(im_diff, ax=ax_comparison, label='Normalized difference (dA/dt - E-field)')
            ax_comparison.set_title('Field Comparison')
            
            # Add text annotations with statistics
            correlation = np.corrcoef(dadt_mag.flatten(), efield_mag.flatten())[0, 1]
            textstr = f"Correlation: {correlation:.3f}\n"
            
            # Calculate cosine similarity if vectors
            if dadt_has_components and efield_has_components:
                cosine_sim = self._calculate_cosine_similarity(dadt_slice, efield_slice, mask_slice)
                textstr += f"Cosine Similarity: {cosine_sim:.3f}\n"
            
            # Add RMS difference
            rms_diff = np.sqrt(np.mean((dadt_norm - efield_norm) ** 2))
            textstr += f"RMS Difference: {rms_diff:.3f}"
            
            ax_comparison.text(
                0.05, 0.95, textstr,
                transform=ax_comparison.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
            )
            
            # Add overall title
            slice_name = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}[slice_axis]
            plt.suptitle(
                f"{slice_name} Slice {slice_index} - TMS Field Comparison (Position {position_index})",
                fontsize=16
            )
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save figure if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved field comparison to {output_path}")
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("visualize_field_comparison_complete", {
                    "correlation": correlation,
                    "rms_difference": rms_diff,
                    "slice_axis": slice_axis,
                    "slice_index": slice_index
                })
            
            return fig
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "FieldVisualizer.visualize_field_comparison",
                    "position_index": position_index,
                    "component_index": component_index
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("FieldVisualizer.visualize_field_comparison", "end")
    
    def _plot_vector_field(
        self,
        ax: plt.Axes,
        vector_slice: np.ndarray,
        mask_slice: np.ndarray,
        title: str
    ) -> None:
        """
        Plot vector field.
        
        Args:
            ax: Matplotlib axes
            vector_slice: Vector field slice (must have component dimension)
            mask_slice: Mask slice
            title: Plot title
        """
        # Check if we have vector components
        if vector_slice.shape[-1] != 3:
            ax.text(0.5, 0.5, "Vector field not available", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Extract X and Y components for quiver plot
        vx = vector_slice[..., 0] * mask_slice
        vy = vector_slice[..., 1] * mask_slice
        
        # Calculate magnitude for color mapping
        magnitude = np.sqrt(vx**2 + vy**2)
        
        # Create coordinate meshgrid
        y, x = np.mgrid[:vx.shape[0], :vx.shape[1]]
        
        # Subsample for clearer visualization
        stride = self.config.vector_stride
        
        # Plot vector field
        quiver = ax.quiver(
            x[::stride, ::stride].T,
            y[::stride, ::stride].T,
            vx[::stride, ::stride].T,
            vy[::stride, ::stride].T,
            magnitude[::stride, ::stride].T,
            cmap='viridis',
            scale=30,
            width=0.002
        )
        
        plt.colorbar(quiver, ax=ax, label='Magnitude')
        ax.set_title(title)
        ax.set_aspect('equal')
    
    def _plot_vector_component(
        self,
        ax: plt.Axes,
        vector_slice: np.ndarray,
        mask_slice: np.ndarray,
        component_index: int,
        title: str
    ) -> None:
        """
        Plot single component of vector field.
        
        Args:
            ax: Matplotlib axes
            vector_slice: Vector field slice (must have component dimension)
            mask_slice: Mask slice
            component_index: Component index (0=x, 1=y, 2=z)
            title: Plot title
        """
        # Check if we have vector components
        if vector_slice.shape[-1] != 3:
            ax.text(0.5, 0.5, "Vector components not available", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Check component index
        if component_index < 0 or component_index >= 3:
            component_index = 0
        
        # Extract component and apply mask
        component = vector_slice[..., component_index] * mask_slice
        
        # Plot component
        im = ax.imshow(component.T, cmap='RdBu_r', origin='lower')
        plt.colorbar(im, ax=ax, label=f'Component {component_index}')
        ax.set_title(title)
    
    def _plot_field_overlay(
        self,
        ax: plt.Axes,
        field1: np.ndarray,
        field2: np.ndarray,
        mask_slice: np.ndarray,
        title: str
    ) -> None:
        """
        Plot overlay of two fields.
        
        Args:
            ax: Matplotlib axes
            field1: First field (dA/dt)
            field2: Second field (E-field)
            mask_slice: Mask slice
            title: Plot title
        """
        # Normalize fields for overlay
        field1_norm = field1 * mask_slice
        max_val1 = np.max(field1_norm)
        if max_val1 > 0:
            field1_norm = field1_norm / max_val1
        
        field2_norm = field2 * mask_slice
        max_val2 = np.max(field2_norm)
        if max_val2 > 0:
            field2_norm = field2_norm / max_val2
        
        # Plot first field in red channel
        red_channel = np.zeros((*field1_norm.shape, 3))
        red_channel[..., 0] = field1_norm
        
        # Plot second field in blue channel
        blue_channel = np.zeros((*field2_norm.shape, 3))
        blue_channel[..., 2] = field2_norm
        
        # Combine fields
        combined = np.clip(red_channel + blue_channel, 0, 1)
        
        # Plot combined image
        ax.imshow(combined.transpose(1, 0, 2), origin='lower')
        ax.set_title(title)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='dA/dt'),
            Patch(facecolor='blue', label='E-field'),
            Patch(facecolor='magenta', label='Overlap')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _calculate_cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between vector fields.
        
        Args:
            vec1: First vector field
            vec2: Second vector field
            mask: Mask
            
        Returns:
            Cosine similarity
        """
        # Check dimensions
        if vec1.shape[-1] != 3 or vec2.shape[-1] != 3:
            return 0.0
        
        # Flatten vectors and apply mask
        mask_flat = mask.flatten()
        valid_indices = np.where(mask_flat > 0)[0]
        
        if len(valid_indices) == 0:
            return 0.0
        
        vec1_flat = vec1.reshape(-1, 3)[valid_indices]
        vec2_flat = vec2.reshape(-1, 3)[valid_indices]
        
        # Calculate dot product
        dot_product = np.sum(vec1_flat * vec2_flat, axis=1)
        
        # Calculate magnitudes
        vec1_mag = np.sqrt(np.sum(vec1_flat**2, axis=1))
        vec2_mag = np.sqrt(np.sum(vec2_flat**2, axis=1))
        
        # Calculate cosine similarity
        cos_sim = np.mean(dot_product / (vec1_mag * vec2_mag + 1e-10))
        
        return cos_sim
    
    def visualize_3d_comparison(
        self,
        data: VisualizationData,
        output_path: Optional[str] = None,
        position_index: int = 0,
        threshold: float = 0.2,
        subsample: int = 3
    ) -> plt.Figure:
        """
        Create 3D comparison visualization of dA/dt and E-field data.
        
        Args:
            data: Visualization data
            output_path: Optional path to save visualization
            position_index: Index of position to visualize
            threshold: Threshold value for visualization
            subsample: Subsampling factor to reduce points
            
        Returns:
            Matplotlib figure
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("FieldVisualizer.visualize_3d_comparison", "start")
        
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("visualize_3d_comparison_start", {
                "position_index": position_index,
                "threshold": threshold,
                "subsample": subsample
            })
        
        try:
            # Check if we have both dA/dt and E-field data
            if data.dadt_data is None or data.efield_data is None:
                error_msg = "Both dA/dt and E-field data are required for comparison"
                if self.debug_hook:
                    self.debug_hook.record_error(ValueError(error_msg), {
                        "component": "FieldVisualizer.visualize_3d_comparison"
                    })
                raise ValueError(error_msg)
            
            # Handle multiple positions
            dadt = data.dadt_data
            efield = data.efield_data
            
            # Check if these are arrays of multiple positions
            if len(dadt.shape) > 3:
                if position_index >= dadt.shape[0]:
                    raise ValueError(f"Position index {position_index} is out of range for dA/dt data with shape {dadt.shape}")
                dadt = dadt[position_index]
            
            if len(efield.shape) > 3:
                if position_index >= efield.shape[0]:
                    raise ValueError(f"Position index {position_index} is out of range for E-field data with shape {efield.shape}")
                efield = efield[position_index]
            
            # Create mask if not provided
            mask = data.mask_data
            if mask is None:
                # Create a simple mask based on non-zero values in dA/dt data
                mask = np.any(np.abs(dadt) > 0, axis=-1) if len(dadt.shape) > 2 else np.abs(dadt) > 0
            
            # Calculate magnitudes if needed
            dadt_has_components = len(dadt.shape) > 2 and dadt.shape[-1] == 3
            efield_has_components = len(efield.shape) > 2 and efield.shape[-1] == 3
            
            if dadt_has_components:
                dadt_mag = calculate_field_magnitude(dadt)
            else:
                dadt_mag = dadt
            
            if efield_has_components:
                efield_mag = calculate_field_magnitude(efield)
            else:
                efield_mag = efield
            
            # Apply mask
            dadt_mag = dadt_mag * mask
            efield_mag = efield_mag * mask
            
            # Create figure
            fig = plt.figure(figsize=(18, 6))
            
            # Create masks above threshold
            dadt_mask = dadt_mag > threshold * np.max(dadt_mag)
            efield_mask = efield_mag > threshold * np.max(efield_mag)
            
            # Create 3D plots
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            ax3 = fig.add_subplot(133, projection='3d')
            
            # Get coordinates of voxels above threshold
            x_dadt, y_dadt, z_dadt = np.where(dadt_mask & mask)
            x_efield, y_efield, z_efield = np.where(efield_mask & mask)
            
            # Subsample points
            idx_dadt = np.arange(0, len(x_dadt), subsample)
            idx_efield = np.arange(0, len(x_efield), subsample)
            
            x_dadt = x_dadt[idx_dadt]
            y_dadt = y_dadt[idx_dadt]
            z_dadt = z_dadt[idx_dadt]
            
            x_efield = x_efield[idx_efield]
            y_efield = y_efield[idx_efield]
            z_efield = z_efield[idx_efield]
            
            # Get values for color mapping
            values_dadt = dadt_mag[x_dadt, y_dadt, z_dadt]
            values_efield = efield_mag[x_efield, y_efield, z_efield]
            
            # Plot dA/dt
            sc1 = ax1.scatter(x_dadt, y_dadt, z_dadt, c=values_dadt, 
                             cmap=self.config.colormap_dadt, alpha=0.5)
            plt.colorbar(sc1, ax=ax1, shrink=0.5, label='dA/dt magnitude')
            ax1.set_title('dA/dt Field')
            
            # Plot E-field
            sc2 = ax2.scatter(x_efield, y_efield, z_efield, c=values_efield, 
                             cmap=self.config.colormap_efield, alpha=0.5)
            plt.colorbar(sc2, ax=ax2, shrink=0.5, label='E-field magnitude')
            ax2.set_title('E-field')
            
            # Find overlap points
            overlap_mask = dadt_mask & efield_mask & mask
            x_overlap, y_overlap, z_overlap = np.where(overlap_mask)
            
            # Subsample overlap points
            idx_overlap = np.arange(0, len(x_overlap), subsample)
            x_overlap = x_overlap[idx_overlap]
            y_overlap = y_overlap[idx_overlap]
            z_overlap = z_overlap[idx_overlap]
            
            # Calculate normalized difference for color mapping
            dadt_norm = dadt_mag / np.max(dadt_mag)
            efield_norm = efield_mag / np.max(efield_mag)
            diff = dadt_norm - efield_norm
            
            values_overlap = diff[x_overlap, y_overlap, z_overlap]
            
            # Plot overlap
            sc3 = ax3.scatter(x_overlap, y_overlap, z_overlap, c=values_overlap, 
                             cmap=self.config.colormap_comparison, alpha=0.5,
                             vmin=-1, vmax=1)
            plt.colorbar(sc3, ax=ax3, shrink=0.5, label='Field difference')
            ax3.set_title('Field Comparison')
            
            # Set axis labels
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
                # Set equal aspect ratio
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()
                z_lim = ax.get_zlim()
                
                max_range = max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]) / 2
                mid_x = (x_lim[1] + x_lim[0]) / 2
                mid_y = (y_lim[1] + y_lim[0]) / 2
                mid_z = (z_lim[1] + z_lim[0]) / 2
                
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Add overall title
            plt.suptitle(f"3D TMS Field Comparison (Position {position_index})", fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save figure if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved 3D field comparison to {output_path}")
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("visualize_3d_comparison_complete", {
                    "dadt_points": len(x_dadt),
                    "efield_points": len(x_efield),
                    "overlap_points": len(x_overlap)
                })
            
            return fig
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "FieldVisualizer.visualize_3d_comparison",
                    "position_index": position_index,
                    "threshold": threshold
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("FieldVisualizer.visualize_3d_comparison", "end")
    
    def visualize_all_positions(
        self,
        data: VisualizationData,
        output_dir: str,
        max_positions: Optional[int] = None,
        components: bool = False
    ) -> None:
        """
        Create visualizations for all positions in the data.
        
        Args:
            data: Visualization data
            output_dir: Directory to save visualizations
            max_positions: Maximum number of positions to visualize
            components: Whether to visualize vector components
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("FieldVisualizer.visualize_all_positions", "start")
        
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event("visualize_all_positions_start", {
                "output_dir": output_dir,
                "max_positions": max_positions,
                "components": components
            })
        
        try:
            # Check if we have both dA/dt and E-field data
            if data.dadt_data is None or data.efield_data is None:
                error_msg = "Both dA/dt and E-field data are required for comparison"
                if self.debug_hook:
                    self.debug_hook.record_error(ValueError(error_msg), {
                        "component": "FieldVisualizer.visualize_all_positions"
                    })
                raise ValueError(error_msg)
            
            # Determine number of positions
            if len(data.dadt_data.shape) > 3:
                n_positions = data.dadt_data.shape[0]
            else:
                n_positions = 1
            
            if len(data.efield_data.shape) > 3:
                n_positions = min(n_positions, data.efield_data.shape[0])
            
            # Limit number of positions if requested
            if max_positions is not None and max_positions > 0:
                n_positions = min(n_positions, max_positions)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Loop through positions
            for i in range(n_positions):
                # Create position directory
                pos_dir = os.path.join(output_dir, f"position_{i}")
                os.makedirs(pos_dir, exist_ok=True)
                
                # Create magnitude comparison
                magnitude_path = os.path.join(pos_dir, "magnitude_comparison.png")
                self.visualize_field_comparison(
                    data,
                    output_path=magnitude_path,
                    position_index=i,
                    show_magnitude=True,
                    show_vectors=False
                )
                
                # Create vector field comparison
                vector_path = os.path.join(pos_dir, "vector_comparison.png")
                self.visualize_field_comparison(
                    data,
                    output_path=vector_path,
                    position_index=i,
                    show_magnitude=False,
                    show_vectors=True
                )
                
                # Create 3D comparison
                threed_path = os.path.join(pos_dir, "3d_comparison.png")
                self.visualize_3d_comparison(
                    data,
                    output_path=threed_path,
                    position_index=i
                )
                
                # Visualize components if requested
                if components:
                    for comp_idx in range(3):
                        component_path = os.path.join(pos_dir, f"component_{comp_idx}_comparison.png")
                        self.visualize_field_comparison(
                            data,
                            output_path=component_path,
                            position_index=i,
                            component_index=comp_idx,
                            show_magnitude=False,
                            show_vectors=True
                        )
                
                logger.info(f"Completed visualizations for position {i}")
            
            # Create a summary file
            summary_path = os.path.join(output_dir, "visualization_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"TMS Field Visualization Summary\n")
                f.write(f"==============================\n\n")
                f.write(f"Total positions visualized: {n_positions}\n")
                
                if data.dadt_data is not None:
                    f.write(f"dA/dt data shape: {data.dadt_data.shape}\n")
                
                if data.efield_data is not None:
                    f.write(f"E-field data shape: {data.efield_data.shape}\n")
                
                if data.mesh_data is not None:
                    f.write(f"Mesh node count: {len(data.mesh_data.nodes)}\n")
                
                f.write(f"\nVisualization completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Completed all visualizations. Output saved to {output_dir}")
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("visualize_all_positions_complete", {
                    "positions_visualized": n_positions,
                    "output_dir": output_dir
                })
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "FieldVisualizer.visualize_all_positions",
                    "output_dir": output_dir,
                    "max_positions": max_positions
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("FieldVisualizer.visualize_all_positions", "end")