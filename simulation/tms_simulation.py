"""
TMS simulation module.

This module provides functionality for simulating TMS-induced E-fields
with explicit state management, resource monitoring, and debugging hooks.
"""

import os
import time
import numpy as np
import h5py
import shutil
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed
from tqdm import tqdm

# SimNIBS imports
from simnibs import mesh_io, run_simnibs, sim_struct
from simnibs.simulation import coil_numpy as coil_lib

# Project imports
from tms_efield_prediction.utils.debug.hooks import DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.utils.state.context import ModuleState, PipelineContext
import logging
logging.basicConfig(level=logging.INFO)

# Create your logger
logger = logging.getLogger(__name__)

@dataclass
class SimulationState(ModuleState):
    """State for TMS simulation operations."""
    simulation_phase: str = "initialization"
    mesh_data: Optional[Dict[str, Any]] = None
    coil_data: Optional[Dict[str, Any]] = None
    matsimnibs: Optional[np.ndarray] = None
    dadt_data: Optional[np.ndarray] = None
    efield_data: Optional[np.ndarray] = None
    
    def transition_to(self, new_phase: str) -> 'SimulationState':
        """Transition to a new simulation phase.
        
        Args:
            new_phase: Target phase
            
        Returns:
            New simulation state
        """
        valid_phases = [
            "initialization", 
            "mesh_loading", 
            "coil_positioning", 
            "dadt_calculation", 
            "efield_simulation",
            "data_extraction",
            "completed"
        ]
        
        if new_phase not in valid_phases:
            raise ValueError(f"Invalid simulation phase: {new_phase}")
        
        # Create new state
        new_state = SimulationState(
            version=self.version + 1,
            data=self.data.copy(),
            simulation_phase=new_phase,
            mesh_data=self.mesh_data,
            coil_data=self.coil_data,
            matsimnibs=self.matsimnibs,
            dadt_data=self.dadt_data,
            efield_data=self.efield_data
        )
        
        return new_state


@dataclass
class SimulationContext(PipelineContext):
    """Context for TMS simulation operations."""
    subject_id: str = ""
    data_root_path: str = ""
    coil_file_path: str = ""
    output_path: str = ""
    tensor_nifti_path: str = ""
    anisotropy_type: str = "vn"
    aniso_maxratio: float = 10.0
    aniso_maxcond: float = 5.0
    didt: float = 1.49e6  # Default dI/dt value
    map_to_surf: bool = True
    map_to_vol: bool = False
    fields: str = "eE"  # Fields to simulate
    
    def validate(self) -> bool:
        """Validate simulation context configuration.
        
        Returns:
            bool: True if valid, raises exception otherwise
        """
        # Basic validation from PipelineContext
        super().validate()
        
        # Additional validation
        if not self.subject_id:
            raise ValueError("Subject ID must be specified")
        
        if not self.data_root_path or not os.path.exists(self.data_root_path):
            raise ValueError(f"Invalid data root path: {self.data_root_path}")
            
        if not self.coil_file_path or not os.path.exists(self.coil_file_path):
            raise ValueError(f"Invalid coil file path: {self.coil_file_path}")
            
        # Create output directory if it doesn't exist
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
        
        return True


def load_mesh_and_roi(
    context: SimulationContext,
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Load mesh and ROI information with proper resource tracking.
    
    Args:
        context: Simulation context
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        
    Returns:
        Tuple of (SimNIBS mesh, ROI center dictionary)
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simulation.load_mesh_and_roi", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("load_mesh_and_roi_start", {"subject_id": context.subject_id})
    
    try:
        # Derive file paths
        sub_path = context.data_root_path
        sub_id = context.subject_id
        
        # Load mesh
        msh_name = f"{sub_id}.msh"
        msh_path = os.path.join(sub_path, 'headmodel', msh_name)
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event("loading_mesh", {"path": msh_path})
        
        msh = mesh_io.read_msh(msh_path)
        
        # Load ROI center
        roi_center_path = os.path.join(sub_path, 'experiment')
        roi_center_name = f"{sub_id}_roi_center.mat"
        roi_center_file = os.path.join(roi_center_path, roi_center_name)
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event("loading_roi_center", {"path": roi_center_file})
        
        # Load MAT file - use h5py to handle modern MATLAB format
        with h5py.File(roi_center_file, 'r') as f:
            if 'roi_center' in f:
                roi_center_ref = f['roi_center']
                roi_center = {
                    'gm': np.array(f[roi_center_ref['gm'][0, 0]]),
                    'skin': np.array(f[roi_center_ref['skin'][0, 0]]),
                    'skin_vec': np.array(f[roi_center_ref['skin_vec'][0, 0]])
                }
            else:
                # Fallback for older format
                roi_center_data = np.array(f[next(iter(f.keys()))])
                roi_center = {
                    'gm': roi_center_data[:3],
                    'skin': roi_center_data[3:6],
                    'skin_vec': roi_center_data[6:9]
                }
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "load_mesh_and_roi_complete", 
                {
                    "subject_id": context.subject_id,
                    "mesh_node_count": len(msh.nodes),
                    "roi_center_keys": list(roi_center.keys())
                }
            )
        
        return msh, roi_center
        
    except Exception as e:
        if debug_hook:
            debug_hook.record_error(e, {
                "component": "simulation.get_skin_average_normal_vector",
                "roi_radius": roi_radius
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simulation.get_skin_average_normal_vector", "end")


def rotate_grid(
    gridz: np.ndarray, 
    centernormal_skin: np.ndarray,
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
) -> np.ndarray:
    """
    Calculate rotation matrix to align grid with skin normal.
    
    Args:
        gridz: Z-axis unit vector of the grid
        centernormal_skin: Normal vector at the skin center
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        
    Returns:
        Rotation matrix as numpy array
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simulation.rotate_grid", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("rotate_grid_start", {
            "gridz": gridz.tolist(), 
            "centernormal_skin": centernormal_skin.tolist()
        })
    
    try:
        # Calculate rotation angle and axis
        theta = np.arccos(np.dot(gridz, centernormal_skin))
        axis_rot = np.cross(gridz, centernormal_skin)
        axis_rot_norm = np.linalg.norm(axis_rot)
        
        # Handle parallel vectors case
        if axis_rot_norm < 1e-10:
            if np.dot(gridz, centernormal_skin) > 0:
                # Vectors are parallel, no rotation needed
                rot_matrix = np.eye(3)
            else:
                # Vectors are anti-parallel, rotate 180° around any perpendicular axis
                # Choose x-axis if gridz is not parallel to it, otherwise choose y-axis
                if abs(np.dot(gridz, [1, 0, 0])) < 0.99:
                    perp_axis = np.cross(gridz, [1, 0, 0])
                else:
                    perp_axis = np.cross(gridz, [0, 1, 0])
                perp_axis = perp_axis / np.linalg.norm(perp_axis)
                rot_matrix = R.from_rotvec(np.pi * perp_axis).as_matrix()
        else:
            # Normal case: compute rotation matrix from angle and axis
            axis_rot = axis_rot / axis_rot_norm
            rot_matrix = R.from_rotvec(theta * axis_rot).as_matrix()
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "rotate_grid_complete", 
                {
                    "theta_degrees": np.degrees(theta),
                    "rotation_matrix_det": np.linalg.det(rot_matrix)
                }
            )
        
        return rot_matrix
        
    except Exception as e:
        if debug_hook:
            debug_hook.record_error(e, {
                "component": "simulation.rotate_grid",
                "gridz": gridz.tolist(),
                "centernormal_skin": centernormal_skin.tolist()
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simulation.rotate_grid", "end")


def generate_grid(
    centerpoint_skin: np.ndarray, 
    gridx_new: np.ndarray, 
    gridy_new: np.ndarray, 
    search_radius: float, 
    spatial_resolution: float,
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a grid of points around the center point.
    
    Args:
        centerpoint_skin: Center point on the skin
        gridx_new: New x-axis unit vector
        gridy_new: New y-axis unit vector
        search_radius: Radius of search area in mm
        spatial_resolution: Spacing between grid points in mm
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        
    Returns:
        Tuple of (3D points, 2D grid coordinates)
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simulation.generate_grid", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("generate_grid_start", {
            "centerpoint": centerpoint_skin.tolist(),
            "search_radius": search_radius,
            "spatial_resolution": spatial_resolution
        })
    
    try:
        # Calculate number of grid points needed
        n = np.ceil(search_radius / spatial_resolution)
        
        # Create grid of x,y values
        x_vals = np.arange(-n * spatial_resolution, n * spatial_resolution + spatial_resolution, spatial_resolution)
        y_vals = np.arange(-n * spatial_resolution, n * spatial_resolution + spatial_resolution, spatial_resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Calculate 3D positions for each grid point
        points = centerpoint_skin.reshape(1, 3) + X.reshape(-1, 1) * gridx_new + Y.reshape(-1, 1) * gridy_new
        
        # Create 2D grid coordinates for reference
        grid = np.stack((X.flatten(), Y.flatten())).T
        
        # Keep only points within the search radius
        distances = np.sqrt(X.flatten()**2 + Y.flatten()**2)
        keep = distances <= search_radius
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "generate_grid_complete", 
                {
                    "total_points": len(points),
                    "points_in_radius": np.sum(keep),
                    "grid_shape": (len(x_vals), len(y_vals))
                }
            )
        
        return points[keep], grid[keep]
        
    except Exception as e:
        if debug_hook:
            debug_hook.record_error(e, {
                "component": "simulation.generate_grid",
                "centerpoint": centerpoint_skin.tolist(),
                "search_radius": search_radius,
                "spatial_resolution": spatial_resolution
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simulation.generate_grid", "end")


def calc_matsimnibs(
    mesh: Any, 
    grid_centers: np.ndarray, 
    distance: float, 
    rot_angles: np.ndarray, 
    headz: np.ndarray = np.array([0, 0, 1]),
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
) -> np.ndarray:
    """
    Calculate matsimnibs transformation matrices for TMS coil positions.
    
    Args:
        mesh: SimNIBS mesh
        grid_centers: Array of center points on grid
        distance: Distance from skin in mm
        rot_angles: Array of rotation angles in degrees
        headz: Head z-axis direction
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        
    Returns:
        Array of transformation matrices
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simulation.calc_matsimnibs", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("calc_matsimnibs_start", {
            "grid_center_count": len(grid_centers),
            "rot_angle_count": len(rot_angles),
            "distance": distance
        })
    
    try:
        # Extract skin surface
        skin_surface = [5, 1005]  # Surface IDs for skin
        msh_surf = mesh.crop_mesh(elm_type=2)  # Extract surface elements
        msh_skin = msh_surf.crop_mesh(skin_surface)  # Extract skin surface
        
        # Find closest skin elements and their normals
        centers, closest = msh_skin.find_closest_element(grid_centers, return_index=True)
        z_vectors = -msh_skin.triangle_normals()[closest]  # Normals point outward, we need inward
        
        # Calculate coil centers by moving from skin surface along normal
        coil_centers = centers - distance * z_vectors
        
        # Initialize matsimnibs array
        matsimnibs = np.zeros((len(rot_angles), len(grid_centers), 4, 4), dtype=float)
        matsimnibs[:, :, 3, 3] = 1  # Set homogeneous coordinate to 1
        
        # Calculate transformation matrices for each rotation angle
        for a, rot_angle_deg in enumerate(rot_angles):
            angle_rad = np.deg2rad(rot_angle_deg)
            
            # Create rotation vectors and matrices
            rotation_vectors = angle_rad * z_vectors
            rot_matrix = R.from_rotvec(rotation_vectors).as_matrix().transpose(0, 2, 1)
            
            # Project headz onto orthogonal complement of z_vectors
            dot_product = np.einsum('ij,j->i', z_vectors, headz)
            norm_squared = np.einsum('ij,ij->i', z_vectors, z_vectors)
            projection = (dot_product / norm_squared)[:, np.newaxis] * z_vectors
            headz_projected = headz - projection
            
            # Normalize headz_projected if not zero
            headz_norms = np.linalg.norm(headz_projected, axis=1)
            non_zero = headz_norms > 1e-10
            headz_projected[non_zero] = headz_projected[non_zero] / headz_norms[non_zero, np.newaxis]
            
            # For zero vectors, generate a perpendicular vector to z
            for i in np.where(~non_zero)[0]:
                if abs(z_vectors[i, 0]) < 0.9:
                    perp = np.array([1, 0, 0])
                else:
                    perp = np.array([0, 1, 0])
                headz_projected[i] = np.cross(z_vectors[i], perp)
                headz_projected[i] = headz_projected[i] / np.linalg.norm(headz_projected[i])
            
            # Rotate headz_projected using rotation matrices
            y_vectors = np.einsum('ijk,ik->ij', rot_matrix, headz_projected)
            
            # Ensure unit vectors
            y_vectors = y_vectors / np.linalg.norm(y_vectors, axis=1)[:, None]
            
            # Determine x vectors using cross product
            x_vectors = np.cross(y_vectors, z_vectors)
            
            # Build matsimnibs matrices
            matsimnibs[a, :, :3, 0] = x_vectors  # First column is x vector
            matsimnibs[a, :, :3, 1] = y_vectors  # Second column is y vector
            matsimnibs[a, :, :3, 2] = z_vectors  # Third column is z vector
            matsimnibs[a, :, :3, 3] = coil_centers  # Fourth column is translation
        
        if debug_hook and debug_hook.should_sample():
            sample_idx = min(len(grid_centers) - 1, 5)  # Sample the 5th point or the last one
            debug_hook.record_event(
                "calc_matsimnibs_complete", 
                {
                    "output_shape": matsimnibs.shape,
                    "sample_position": coil_centers[sample_idx].tolist(),
                    "sample_normal": z_vectors[sample_idx].tolist()
                }
            )
        
        return matsimnibs
        
    except Exception as e:
        if debug_hook:
            debug_hook.record_error(e, {
                "component": "simulation.calc_matsimnibs",
                "grid_center_count": len(grid_centers),
                "rot_angle_count": len(rot_angles),
                "distance": distance
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simulation.calc_matsimnibs", "end")


def get_matsimnibs(
    context: SimulationContext,
    msh: Any,
    roi_center: Dict[str, np.ndarray],
    params: Dict[str, Any],
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None,
    gridx: np.ndarray = np.array([1, 0, 0]),
    gridz: np.ndarray = np.array([0, 0, 1]),
    headz: np.ndarray = np.array([0, 0, 1])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate matsimnibs matrices and grid for coil positions.
    
    Args:
        context: Simulation context
        msh: SimNIBS mesh
        roi_center: ROI center information
        params: Parameters for matsimnibs generation
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        gridx: X-axis direction for grid
        gridz: Z-axis direction for grid
        headz: Head z-axis direction
        
    Returns:
        Tuple of (matsimnibs matrices, grid coordinates)
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simulation.get_matsimnibs", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("get_matsimnibs_start", {
            "subject_id": context.subject_id,
            "params": params
        })
    
    try:
        # Extract parameters
        search_radius = params.get('search_radius', 50)
        spatial_resolution = params.get('spatial_resolution', 2)
        rotation_angles = params.get('rotation_angles', np.array([0]))
        distance = params.get('distance', 2)
        
        # Handle scalar rotation angle
        if np.isscalar(rotation_angles):
            rotation_angles = np.array([rotation_angles])
        
        # Calculate rotation matrix to align grid with skin normal
        rot_matrix = rotate_grid(
            gridz, 
            roi_center['skin_vec'],
            debug_hook=debug_hook,
            resource_monitor=resource_monitor
        )
        
        # Calculate new grid axes
        gridx_new = rot_matrix @ gridx
        gridy_new = np.cross(roi_center['skin_vec'], gridx_new)
        
        # Generate grid points
        points, s_grid = generate_grid(
            roi_center['skin'], 
            gridx_new, 
            gridy_new, 
            search_radius, 
            spatial_resolution,
            debug_hook=debug_hook,
            resource_monitor=resource_monitor
        )
        
        # Calculate matsimnibs matrices
        matsimnibs = calc_matsimnibs(
            msh, 
            points, 
            distance=distance, 
            rot_angles=rotation_angles, 
            headz=headz,
            debug_hook=debug_hook,
            resource_monitor=resource_monitor
        )
        
        # Reshape to flatten the angles and positions
        matsimnibs_flat = matsimnibs.reshape(-1, 4, 4)
        
        # Create grid with angle as first dimension
        grid = np.stack([[np.array([a, *s]) for s in s_grid] for a in rotation_angles]).reshape(-1, 3)
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "get_matsimnibs_complete", 
                {
                    "matsimnibs_shape": matsimnibs_flat.shape,
                    "grid_shape": grid.shape,
                    "rotation_angles": rotation_angles.tolist()
                }
            )
        
        return matsimnibs_flat, grid
        
    except Exception as e:
        if debug_hook:
            debug_hook.record_error(e, {
                "component": "simulation.get_matsimnibs",
                "subject_id": context.subject_id,
                "params": str(params)
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simulation.get_matsimnibs", "end")
def calc_dAdt(
    context: SimulationContext,
    mesh: Any,
    matsimnibs: np.ndarray,
    roi_mesh: Optional[Any] = None,
    roi_center: Optional[Dict[str, np.ndarray]] = None,
    skin_normal_avg: Optional[np.ndarray] = None,
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None,
    n_cpus: int = 1,
    to_hdf5: bool = True,
    save_path: Optional[str] = None,
    roi_radius: float = 20.0
) -> Optional[np.ndarray]:
    """
    Calculate dA/dt for given coil positions, optimized for ROI storage.
    
    This function can use either a pre-created ROI mesh or create one on the fly
    from the full mesh using the provided ROI center and skin normal vector.
    Using an ROI mesh significantly reduces computation time and memory usage
    (typically by ~98% reduction in mesh size).
    
    Args:
        context: Simulation context
        mesh: SimNIBS mesh (full brain mesh)
        matsimnibs: Array of transformation matrices
        roi_mesh: Optional pre-created ROI mesh (preferred for efficiency)
        roi_center: ROI center information dictionary (required if roi_mesh is None and ROI optimization is desired)
        skin_normal_avg: Average normal vector at skin (required if roi_mesh is None and ROI optimization is desired)
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        n_cpus: Number of CPU cores to use
        to_hdf5: Whether to save results to HDF5 file
        save_path: Path to save results, default is context.output_path
        roi_radius: Radius of ROI cylinder in mm (used if creating ROI on the fly)
        
    Returns:
        Array of dA/dt values or None if saving to file only
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simulation.calc_dAdt", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("calc_dAdt_start", {
            "subject_id": context.subject_id,
            "matsimnibs_count": len(matsimnibs),
            "n_cpus": n_cpus,
            "to_hdf5": to_hdf5,
            "roi_radius": roi_radius,
            "roi_mesh_provided": roi_mesh is not None
        })
    
    try:
        # Prepare output path
        if save_path is None:
            save_path = context.output_path
        os.makedirs(save_path, exist_ok=True)
        
        # Create temporary directory for parallel processing
        tmp_path = None
        if to_hdf5:
            tmp_path = os.path.join(save_path, 'tmp_dadt')
            os.makedirs(tmp_path, exist_ok=True)
        
        # Determine which mesh to use for calculation
        calculation_mesh = None
        use_roi_optimization = False
        
        if roi_mesh is not None:
            # Use provided ROI mesh - much more efficient
            calculation_mesh = roi_mesh
            use_roi_optimization = True
            
            if debug_hook and debug_hook.should_sample():
                node_count = len(roi_mesh.nodes) if hasattr(roi_mesh, 'nodes') else 0
                element_count = len(roi_mesh.elm.triangles) if hasattr(roi_mesh.elm, 'triangles') else 0
                
                debug_hook.record_event("calc_dAdt_using_provided_roi", {
                    "roi_mesh_node_count": node_count,
                    "roi_mesh_element_count": element_count
                })
        elif roi_center is not None and skin_normal_avg is not None:
            # Create ROI mesh on the fly
            if debug_hook and debug_hook.should_sample():
                debug_hook.record_event("calc_dAdt_create_roi", {
                    "roi_radius": roi_radius
                })
            
            # Get cylindrical ROI
            cylindrical_roi = compute_cylindrical_roi(
                mesh, 
                roi_center['gm'], 
                skin_normal_avg, 
                roi_radius=roi_radius,
                debug_hook=debug_hook,
                resource_monitor=resource_monitor
            )
            
            # Crop mesh to ROI to reduce memory usage
            cropped_mesh = crop_mesh_nodes(
                mesh, 
                cylindrical_roi,
                debug_hook=debug_hook,
                resource_monitor=resource_monitor
            )
            
            # Remove disconnected components
            roi_mesh = remove_islands(
                cropped_mesh, 
                roi_center,
                debug_hook=debug_hook,
                resource_monitor=resource_monitor
            )
            
            calculation_mesh = roi_mesh
            use_roi_optimization = True
            
            if debug_hook and debug_hook.should_sample():
                original_element_count = len(mesh.elm.triangles) if hasattr(mesh.elm, 'triangles') else 0
                roi_element_count = len(roi_mesh.elm.triangles) if hasattr(roi_mesh.elm, 'triangles') else 0
                
                reduction_percent = 0
                if original_element_count > 0:
                    reduction_percent = 100 * (1 - roi_element_count / original_element_count)
                
                debug_hook.record_event("calc_dAdt_roi_created", {
                    "original_elements": original_element_count,
                    "roi_elements": roi_element_count,
                    "reduction_percent": reduction_percent
                })
        else:
            # Use full mesh (not recommended for performance)
            calculation_mesh = mesh
            
            if debug_hook and debug_hook.should_sample():
                node_count = len(mesh.nodes) if hasattr(mesh, 'nodes') else 0
                element_count = len(mesh.elm.triangles) if hasattr(mesh.elm, 'triangles') else 0
                
                debug_hook.record_event("calc_dAdt_using_full_mesh", {
                    "warning": "Using full mesh will consume significantly more memory",
                    "mesh_node_count": node_count,
                    "mesh_element_count": element_count
                })
        
        # Define function for calculating dA/dt - more memory efficient version
        def get_dAdt(matsim, i):
            try:
                # Calculate dA/dt for this position
                dadt = coil_lib.set_up_tms(calculation_mesh, context.coil_file_path, matsim, context.didt)[:, :]
                
                if to_hdf5 and tmp_path:
                    # Save to temporary file with single precision to save space
                    np.save(os.path.join(tmp_path, f"{i}.npy"), dadt.astype(np.float32))
                
                return dadt
            except Exception as e:
                # Improve error handling with position information
                error_msg = f"Error calculating dA/dt for position {i}: {str(e)}"
                if debug_hook:
                    debug_hook.record_error(Exception(error_msg), {
                        "position_index": i,
                        "component": "simulation.calc_dAdt.get_dAdt"
                    })
                raise Exception(error_msg)
        
        # Process sequential or parallel
        if n_cpus == 1 or len(matsimnibs) == 1:
            # Sequential processing
            if debug_hook and debug_hook.should_sample():
                debug_hook.record_event("calc_dAdt_sequential", {
                    "matsimnibs_count": len(matsimnibs)
                })
            
            results = []
            for i, matsim in enumerate(tqdm(matsimnibs, desc="Calculating dA/dt")):
                results.append(get_dAdt(matsim, i))
        else:
            # Parallel processing
            if debug_hook and debug_hook.should_sample():
                debug_hook.record_event("calc_dAdt_parallel", {
                    "matsimnibs_count": len(matsimnibs),
                    "n_cpus": n_cpus
                })
            
            results = Parallel(n_jobs=n_cpus)(
                delayed(get_dAdt)(matsim, i) for i, matsim in enumerate(tqdm(matsimnibs, desc="Calculating dA/dt"))
            )
        
        # Process results
        dadt_data = None
        
        if to_hdf5 and tmp_path:
            # Create HDF5 file from temporary files with compression
            h5_path = os.path.join(save_path, 'dAdts.h5')
            
            if debug_hook and debug_hook.should_sample():
                debug_hook.record_event("calc_dAdt_saving_hdf5", {
                    "path": h5_path,
                    "compression": True,
                    "optimization": "ROI-based" if use_roi_optimization else "Full mesh"
                })
            
            with h5py.File(h5_path, 'w') as f:
                # Load first file to get shape
                first_dadt = np.load(os.path.join(tmp_path, "0.npy"))
                
                # Create dataset with compression
                dadt_shape = (len(matsimnibs), *first_dadt.shape)
                dadt_dataset = f.create_dataset(
                    'dAdt', 
                    shape=dadt_shape,
                    dtype=np.float32,  # Use single precision
                    compression='gzip',
                    compression_opts=4
                )
                
                # Fill dataset
                for i in range(len(matsimnibs)):
                    dadt_dataset[i] = np.load(os.path.join(tmp_path, f"{i}.npy"))
                
                # Store metadata about the calculation
                f.create_dataset('calculation_type', data=np.string_("roi" if use_roi_optimization else "full_mesh"))
                
                # Store ROI information if used
                if use_roi_optimization and roi_center is not None:
                    f.create_dataset('roi_center_gm', data=roi_center['gm'])
                    f.create_dataset('roi_center_skin', data=roi_center['skin'])
                    f.create_dataset('roi_center_skin_vec', data=roi_center['skin_vec'])
                    f.create_dataset('roi_radius', data=roi_radius)
                    
                    # Store element mapping information for reconstruction if needed
                    if roi_mesh is not None and hasattr(roi_mesh.elm, 'triangles'):
                        try:
                            elm_ids = np.array([e.id for e in roi_mesh.elm.triangles])
                            f.create_dataset('roi_element_ids', data=elm_ids)
                        except Exception as e:
                            if debug_hook:
                                debug_hook.record_error(e, {
                                    "component": "simulation.calc_dAdt",
                                    "action": "storing_element_ids"
                                })
            
            # Clean up temporary files
            try:
                shutil.rmtree(tmp_path)
            except Exception as e:
                if debug_hook:
                    debug_hook.record_error(e, {
                        "component": "simulation.calc_dAdt",
                        "action": "cleanup_temp_files",
                        "tmp_path": tmp_path
                    })
            
            # Load the data from HDF5 to return
            try:
                with h5py.File(h5_path, 'r') as f:
                    dadt_data = f['dAdt'][:]
            except Exception as e:
                if debug_hook:
                    debug_hook.record_error(e, {
                        "component": "simulation.calc_dAdt",
                        "action": "loading_hdf5_after_save",
                        "file_path": h5_path
                    })
                # Continue without loading data if there's an error
        else:
            # Stack results from memory
            dadt_data = np.stack(results)
            if save_path is not None and not to_hdf5:
                np.save(os.path.join(save_path, 'dAdts.npy'), dadt_data)
        
        if debug_hook and debug_hook.should_sample():
            element_reduction = "N/A"
            if use_roi_optimization and roi_mesh is not None and hasattr(mesh.elm, 'triangles') and hasattr(roi_mesh.elm, 'triangles'):
                element_reduction = f"{len(roi_mesh.elm.triangles)}/{len(mesh.elm.triangles)}"
            
            debug_hook.record_event(
                "calc_dAdt_complete", 
                {
                    "dadt_shape": dadt_data.shape if dadt_data is not None else None,
                    "to_hdf5": to_hdf5,
                    "save_path": save_path,
                    "element_reduction": element_reduction,
                    "optimization": "ROI-based" if use_roi_optimization else "Full mesh"
                }
            )
        
        return dadt_data
        
    except Exception as e:
        if debug_hook:
            debug_hook.record_error(e, {
                "component": "simulation.calc_dAdt",
                "subject_id": context.subject_id,
                "matsimnibs_count": len(matsimnibs),
                "n_cpus": n_cpus
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simulation.calc_dAdt", "end")

def compute_cylindrical_roi(self, mesh, roi_center_gm, skin_normal_avg, roi_radius=20.0):
    """Load ROI boolean nodes from file - raises error if not found."""
    import numpy as np
    import scipy.io as sio
    import os
    
    # Try to load existing ROI boolean nodes
    experiment_all_dir = os.path.join(self.context.data_root_path, "experiment", "all")
    roi_bool_file = os.path.join(experiment_all_dir, f"sub-{self.context.subject_id}_roi_bool_nodes_middle_gray_matter.mat")
    
    if not os.path.exists(roi_bool_file):
        error_msg = f"Required ROI boolean nodes file not found: {roi_bool_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.log(logging.INFO, f"Loading ROI boolean nodes from: {roi_bool_file}")
        mat_data = sio.loadmat(roi_bool_file)
        
        # Find the boolean array in the mat file - try different possible keys
        roi_bool = None
        for key in mat_data.keys():
            if key not in ['__header__', '__version__', '__globals__']:
                data = mat_data[key]
                if isinstance(data, np.ndarray) and data.dtype == bool:
                    roi_bool = data
                    logger.log(logging.INFO, f"Found boolean array with key: {key}")
                    break
        
        if roi_bool is None:
            error_msg = f"No boolean array found in {roi_bool_file}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check if the size matches the mesh
        if len(roi_bool) != len(mesh.nodes[:]):
            error_msg = f"ROI boolean array size ({len(roi_bool)}) doesn't match mesh node count ({len(mesh.nodes[:])})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.log(logging.INFO, f"Using pre-computed ROI boolean nodes with {np.sum(roi_bool)} selected nodes")
        return roi_bool
        
    except Exception as e:
        error_msg = f"Error loading ROI boolean nodes: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def remove_islands(
    cropped: Any,
    roi_center: Dict[str, np.ndarray],
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
) -> Any:
    """
    Remove disconnected components from the mesh.
    
    Args:
        cropped: Cropped SimNIBS mesh
        roi_center: ROI center information
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        
    Returns:
        Mesh with only the connected component containing ROI center
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simulation.remove_islands", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("remove_islands_start", {
            "node_count": len(cropped.nodes)
        })
    
    try:
        # Find closest element to ROI center
        _, center_id = cropped.find_closest_element(roi_center['gm'], return_index=True)
        
        # Get connected components
        comps = cropped.elm.connected_components()
        
        # Find the component containing the center point
        valid_comps = [c for c in comps if np.isin(center_id, c)]
        
        if not valid_comps:
            raise ValueError("ROI center not found in any connected component")
        
        valid_elms = valid_comps[0]
        
        # Crop mesh to keep only the valid component
        result = cropped.crop_mesh(elements=valid_elms)
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "remove_islands_complete", 
                {
                    "original_elements": len(cropped.elm.node_number_list),
                    "remaining_elements": len(result.elm.node_number_list),
                    "component_count": len(comps)
                }
            )
        
        return result
        
    except Exception as e:
        if debug_hook:
            debug_hook.record_error(e, {
                "component": "simulation.remove_islands"
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simulation.remove_islands", "end")


def crop_mesh_nodes(
    mesh: Any,
    nodes_bool: np.ndarray,
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
) -> Any:
    """
    Crop mesh to keep only elements with all nodes in the selection.
    
    Args:
        mesh: SimNIBS mesh
        nodes_bool: Boolean mask of nodes to keep
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        
    Returns:
        Cropped mesh
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simulation.crop_mesh_nodes", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("crop_mesh_nodes_start", {
            "node_count": len(mesh.nodes),
            "nodes_to_keep": np.sum(nodes_bool)
        })
    
    try:
        # Get node numbers to keep (add 1 for SimNIBS 1-based indexing)
        node_keep_indexes = np.append(np.where(nodes_bool)[0] + 1, -1)
        
        # Find elements where all nodes are in the selection
        node_numbers = mesh.elm.node_number_list
        elements_to_keep = []
        
        # Check each element type
        for elm_type, node_lists in node_numbers.items():
            # Get number of nodes per element for this type
            if elm_type == 'tetrahedra':
                n_nodes = 4
            elif elm_type == 'triangles':
                n_nodes = 3
            else:
                continue
                
            # Reshape for easier checking
            reshaped = node_lists.reshape(-1, n_nodes)
            
            # Check if all nodes of each element are in the selection
            element_mask = np.all(np.isin(reshaped, node_keep_indexes), axis=1)
            
            # Get element indices
            element_indices = np.where(element_mask)[0]
            
            # Add to keep list with 1-based indexing
            elements_to_keep.extend(element_indices + 1)
        
        # Crop mesh
        cropped = mesh.crop_mesh(elements=elements_to_keep)
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "crop_mesh_nodes_complete", 
                {
                    "original_elements": len(mesh.elm.node_number_list),
                    "remaining_elements": len(cropped.elm.node_number_list)
                }
            )
        
        return cropped
        
    except Exception as e:
        if debug_hook:
            debug_hook.record_error(e, {
                "component": "simulation.crop_mesh_nodes",
                "node_count": len(mesh.nodes),
                "nodes_to_keep": np.sum(nodes_bool)
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simulation.crop_mesh_nodes", "end")

import logging
logging.basicConfig(level=logging.INFO)

# Create your logger
logger = logging.getLogger(__name__)
def extract_save_efield(self, sim_path, roi_center, skin_normal_avg, clean_path=False, save_final_mesh=False, save_path_mesh=None):
                """
                Custom version of extract_save_efield that handles different mesh structures.
                
                Args:
                    sim_path: Path to simulation result mesh
                    roi_center: ROI center information
                    skin_normal_avg: Average normal vector at skin
                    clean_path: Whether to clean up simulation directory
                    save_final_mesh: Whether to save the final ROI mesh
                    save_path_mesh: Path to save the final mesh
                    
                Returns:
                    E-field data array
                """
                logger.info("Using custom E-field extraction function")
                
                try:
                    from simnibs import mesh_io
                    import shutil
                    import numpy as np
                    
                    # Load simulation mesh
                    msh = mesh_io.read_msh(sim_path)
                    
                    # Get the E-field data directly from the mesh
                    if not msh.nodedata:
                        raise ValueError("No node data found in mesh")
                    
                    # Try to create a simplified ROI if possible
                    try:
                        logger.info("Attempting to create ROI for the E-field")
                        
                        # Just use the nodes close to the GM center
                        nodes = np.array([node.coordinates for node in msh.nodes])
                        distances = np.linalg.norm(nodes - roi_center['gm'], axis=1)
                        
                        # Use nodes within 30mm of GM center for ROI
                        roi_radius = 30.0  # mm
                        roi_nodes = distances < roi_radius
                        
                        # Count how many nodes are in ROI
                        roi_node_count = np.sum(roi_nodes)
                        logger.info(f"Found {roi_node_count} nodes within {roi_radius}mm of GM center")
                        
                        if roi_node_count > 100:  # Only use ROI if we have enough nodes
                            # Get node indices for ROI
                            roi_indices = np.where(roi_nodes)[0]
                            
                            # Create a subset of the nodedata for the ROI
                            efield_roi = np.zeros_like(msh.nodedata[0])
                            for i, node_idx in enumerate(roi_indices):
                                efield_roi[i] = msh.nodedata[0][node_idx]
                            
                            logger.info(f"Successfully extracted E-field data for ROI with shape {efield_roi.shape}")
                            efield = efield_roi
                        else:
                            # Just use all the data if ROI is too small
                            logger.warning("ROI is too small, using full mesh data")
                            efield = msh.nodedata[0][:]
                    except Exception as e:
                        logger.warning(f"Failed to create ROI: {str(e)}")
                        # Fallback - just use the full data
                        efield = msh.nodedata[0][:]
                        
                    logger.info(f"E-field data shape: {efield.shape}")
                    
                    # Save final mesh if requested
                    if save_final_mesh and save_path_mesh:
                        os.makedirs(os.path.dirname(save_path_mesh), exist_ok=True)
                        try:
                            msh.write(save_path_mesh)
                            logger.info(f"Saved mesh to {save_path_mesh}")
                        except Exception as e:
                            logger.warning(f"Failed to save mesh: {str(e)}")
                    
                    # Clean up simulation directory if requested
                    if clean_path:
                        sim_dir = os.path.dirname(sim_path)
                        if os.path.exists(sim_dir):
                            try:
                                parent_dir = os.path.abspath(os.path.join(sim_dir, ".."))
                                if os.path.basename(parent_dir) == "tmp":
                                    logger.info(f"Cleaning up simulation directory: {parent_dir}")
                                    shutil.rmtree(parent_dir)
                            except Exception as e:
                                logger.warning(f"Failed to clean up directory: {str(e)}")
                    
                    return efield
                    
                except Exception as e:
                    logger.error(f"Error in custom_extract_save_efield: {str(e)}")
                    raise

def run_efield_sim(simnibs_params, matsimnibs, n_cpus=1):
    s = sim_struct.SESSION()
    s.map_to_surf = simnibs_params['map_to_surf']
    s.fields = simnibs_params['fields']
    s.fnamehead = simnibs_params['mesh_path']
    s.pathfem = simnibs_params['out_path']
    s.open_in_gmsh = False

    tms_list = s.add_tmslist()
    tms_list.fnamecoil = simnibs_params['coil_path']
    tms_list.anisotropy_type = simnibs_params['anisotropy_type']
    tms_list.fn_tensor_nifti = simnibs_params['nifti_path']
    tms_list.aniso_maxratio = simnibs_params['aniso_maxratio']
    tms_list.aniso_maxcond = simnibs_params['aniso_maxcond']
    tms_list.name = 'sim'

    if len(matsimnibs.shape)==3:
        for m in matsimnibs:
            pos = tms_list.add_position()
            pos.matsimnibs = m
            pos.didt = simnibs_params['didt']
    else:
        pos = tms_list.add_position()
        pos.matsimnibs = matsimnibs
        pos.didt = simnibs_params['didt']

    run_simnibs(s,cpus=n_cpus)

    coil_name = os.path.splitext(os.path.basename(tms_list.fnamecoil))[0]
    sim_path = s.pathfem + "/subjectoverlays/{0}-{1:0=4d}{2}_".format(tms_list.name, 1, coil_name) + tms_list.anisotropy_type + '_central.msh'

    return sim_path

def get_skin_average_normal_vector(
    mesh: Any,
    roi_center: Dict[str, np.ndarray],
    roi_radius: float = 20.0,
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
) -> np.ndarray:
    """
    Get average normal vector of skin within ROI.
    
    Args:
        mesh: SimNIBS mesh
        roi_center: Dictionary with ROI center information
        roi_radius: Radius for ROI in mm
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        
    Returns:
        Average normal vector as numpy array
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simulation.get_skin_average_normal_vector", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("get_skin_average_normal_vector_start", {"roi_radius": roi_radius})
    
    try:
        # Extract skin region
        skin_region_id = 1005  # Assuming region_idx 1005 corresponds to skin
        skin_cells = mesh.crop_mesh(tags=skin_region_id)
        
        # Get skin triangle centers and normals
        skin_centers = skin_cells.elements_baricenters()[:]
        skin_normals = skin_cells.triangle_normals()[:]
        
        # Compute skin ROI
        roi_center_skin = roi_center['skin']
        distances = np.linalg.norm(skin_centers - roi_center_skin, axis=1)
        skin_roi = distances < roi_radius
        
        # Average normal vector in the ROI
        skin_normal_avg = np.mean(skin_normals[skin_roi], axis=0)
        skin_normal_avg = skin_normal_avg / np.linalg.norm(skin_normal_avg)
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "get_skin_average_normal_vector_complete", 
                {
                    "skin_roi_count": np.sum(skin_roi),
                    "skin_normal_avg": skin_normal_avg.tolist()
                }
            )
        
    except Exception as e:
        if debug_hook:
            debug_hook.record_error(e, {
                "component": "simulation.get_skin_average_normal_vector",
                "roi_radius": roi_radius
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simulation.get_skin_average_normal_vector", "end")