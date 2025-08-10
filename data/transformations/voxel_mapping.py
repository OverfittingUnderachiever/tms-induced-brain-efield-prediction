# tms_efield_prediction/data/transformations/voxel_mapping.py
"""
Voxel mapping for mesh to grid transformation.

This module provides efficient mapping from mesh nodes to voxels
for TMS E-field prediction, including rotation and translation steps.
"""

import os
import numpy as np
import time
import pickle
import sys
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy.spatial.transform import Rotation
import nibabel as nib  # For loading .nii.gz files
from scipy.interpolate import RegularGridInterpolator 


# Assume these imports exist in your project structure
from utils.debug.hooks import PipelineDebugHook
from utils.resource.monitor import ResourceMonitor
from utils.state.context import TMSPipelineContext
from .PC_visualization import visualize_point_clouds # Make sure this utility is available
# Import mesh_io from simnibs if needed within the VoxelMapper itself
from simnibs import mesh_io


class VoxelMapper:
    """
    A class for efficient mapping from mesh nodes to voxels, including
    rotation to align skin_vec with Y-axis and translation of gm_point
    to a fixed target coordinate.
    """

    def __init__(
        self,
        context: TMSPipelineContext,
        mesh_path: Optional[str] = None,
        gm_point: Optional[np.ndarray] = None,      # Original gm_point
        skin_vec: Optional[np.ndarray] = None,     # Original skin_vec
        target_coordinate: Optional[np.ndarray] = np.array([23.0, 16.0, 23.0]), # Target for gm_point
        region_dimensions: Optional[Tuple] = (46.0, 30.0, 
        46.0), # Widths (x,y,z) for ROI
        bin_size: int = 20,
        debug_hook: Optional[PipelineDebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """Initialize the VoxelMapper.

        Args:
            context: TMS-specific pipeline context.
            mesh_path: Path to the mesh file.
            gm_point: The 3D point around which rotation occurs (from mesh space).
            skin_vec: The 3D vector to be aligned with the target vector (from mesh space).
            target_coordinate: The desired final 3D coordinate for the gm_point after all transformations.
                               Defaults to [23, 16, 23].
            region_dimensions: Dimensions (width_x, width_y, width_z) of the ROI bounding box,
                               centered around the target_coordinate. Defaults to (46, 30, 46).
            bin_size: Number of bins in each dimension for voxelization.
            debug_hook: Optional debug hook for tracking.
            resource_monitor: Optional resource monitor for memory tracking.
        """
        self.context = context
        self.mesh_path = mesh_path
        self.gm_point = gm_point
        self.skin_vec = skin_vec
        self.target_coordinate = np.array(target_coordinate) if target_coordinate is not None else None
        self.region_dimensions = region_dimensions
        self.bin_size = bin_size
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor

        if self.target_coordinate is None:
             print("Warning: VoxelMapper initialized without target_coordinate. Translation will not be performed.")
        if self.region_dimensions is None and self.target_coordinate is not None:
             print("Warning: Target coordinate provided but no region_dimensions. ROI cannot be centered.")
             self.region_bounds = None # Cannot calculate bounds yet
        elif self.region_dimensions is not None and self.target_coordinate is not None:
             # Calculate initial ROI bounds centered at target_coordinate
             # These might be recalculated if bin centers are needed before preprocess
             self._calculate_roi_bounds()
        else:
            self.region_bounds = None # Will be set during preprocess if target_coordinate is defined


        # Internal state - will be set during preprocessing
        self.node_to_voxel_map: Optional[Dict[int, int]] = None
        self.voxel_node_counts: Optional[Dict[int, int]] = None
        self.original_nodes: Optional[np.ndarray] = None
        self.transformed_nodes: Optional[np.ndarray] = None
        self.voxel_coordinates: Optional[Dict[int, Tuple[float, float, float]]] = None
        self.node_mask: Optional[np.ndarray] = None  # Mask for nodes within the calculated region of interest
        self._current_transform_matrix: Optional[np.ndarray] = None # Stores the combined final transform

        # Statistics
        self.total_nodes = 0
        self.nodes_in_roi = 0
        self.total_voxels = 0
        self.occupied_voxels = 0
        self.total_elements = 0 # Can be set if mesh is loaded with elements
        self.is_node_based = True

        # Memory tracking
        self._memory_usage = 0
        self._intermediate_data = {}

        # Bins and centers (will be properly set based on final ROI bounds in preprocess)
        self.x_bins: Optional[np.ndarray] = None
        self.y_bins: Optional[np.ndarray] = None
        self.z_bins: Optional[np.ndarray] = None
        self.x_centers: Optional[np.ndarray] = None
        self.y_centers: Optional[np.ndarray] = None
        self.z_centers: Optional[np.ndarray] = None

        # Register with resource monitor if provided
        if self.resource_monitor:
            self.resource_monitor.register_component(
                "voxel_mapper",
                self._reduce_memory,
                priority=10
            )

    def _calculate_roi_bounds(self):
        """Calculates ROI bounds based on target_coordinate and region_dimensions."""
        if self.target_coordinate is None or self.region_dimensions is None:
            self.region_bounds = None
            print("Warning: Cannot calculate ROI bounds without target_coordinate and region_dimensions.")
            return

        width_x, width_y, width_z = self.region_dimensions
        x_bounds = (0.0, width_x)
        y_bounds = (0.0, width_y)
        z_bounds = (0.0, width_z)

        self.region_bounds = (x_bounds, y_bounds, z_bounds)
        print(f"Debug: Calculated ROI bounds centered at {self.target_coordinate}: X={x_bounds}, Y={y_bounds}, Z={z_bounds}")


    def get_current_transform_matrix(self) -> Optional[np.ndarray]:
         """Returns the final 4x4 transformation matrix (rotation + translation) calculated during preprocess."""
         return self._current_transform_matrix

    def preprocess(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Preprocess the mesh:
        1. Load nodes.
        2. Calculate rotation to align skin_vec with [0,1,0] around gm_point.
        3. Calculate translation to move the (rotated) gm_point to target_coordinate.
        4. Combine rotation and translation into a single transform matrix.
        5. Apply the combined transform to all nodes.
        6. Define ROI centered around target_coordinate.
        7. Filter nodes within ROI.
        8. Create voxel mapping based on transformed nodes and ROI.
        """
        start_time = time.time()

        if self.mesh_path is None:
             raise ValueError("VoxelMapper requires mesh_path for preprocessing.")
        if self.gm_point is None or self.skin_vec is None:
             raise ValueError("VoxelMapper requires gm_point and skin_vec for rotation.")
        if self.target_coordinate is None:
             raise ValueError("VoxelMapper requires target_coordinate for translation.")
        if self.region_dimensions is None:
             raise ValueError("VoxelMapper requires region_dimensions for defining the ROI.")


        target_axis_vector = np.array([0.0, 1.0, 0.0]) # Global Y-axis

        # Log start
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "preprocessing_started",
                {
                    'mesh_path': self.mesh_path, 'bin_size': self.bin_size,
                    'transformation_type': 'rotation_then_translation',
                    'source_vector': self.skin_vec.tolist(),
                    'target_axis_vector': target_axis_vector.tolist(),
                    'rotation_center': self.gm_point.tolist(),
                    'final_target_coord': self.target_coordinate.tolist()
                }
            )

        # Load mesh
        print(f"Debug - VoxelMapper loading mesh from: {self.mesh_path}")
        msh = mesh_io.read_msh(self.mesh_path)
        self.original_nodes = msh.nodes.node_coord
        self.total_nodes = len(self.original_nodes)
        # Potentially load elements count if needed elsewhere
        # self.total_elements = len(msh.elm.elm_number)
        print(f"Debug - Loaded mesh with {self.total_nodes} nodes.")

        # --- ADDITION 1: Log Original Node Bounds ---
        if self.total_nodes > 0:
            original_min_coords = self.original_nodes.min(axis=0)
            original_max_coords = self.original_nodes.max(axis=0)
            print(f"Debug - Original Node Bounds (Min): {original_min_coords.round(3)}")
            print(f"Debug - Original Node Bounds (Max): {original_max_coords.round(3)}")
        else:
            print("Debug - Original Node Bounds: N/A (0 nodes)")
        # --- END ADDITION 1 ---


        if self.total_nodes == 0:
            print("Warning - Mesh has zero nodes. Skipping transformation and voxelization.")
            self.transformed_nodes = np.array([])
            self.node_mask = np.array([], dtype=bool)
            self.nodes_in_roi = 0
            # Set empty mappings
            self.node_to_voxel_map = {}
            self.voxel_node_counts = {}
            self.voxel_coordinates = {}
            self._current_transform_matrix = np.eye(4) # Identity transform
            self.total_voxels = self.bin_size**3
            self.occupied_voxels = 0
            # Ensure bins are defined based on target ROI even if no nodes
            self._calculate_roi_bounds()
            if self.region_bounds:
                 self._setup_bins_and_centers()
            return { # Return minimal stats
                'total_nodes': 0, 'nodes_in_roi': 0, 'total_voxels': self.total_voxels,
                'occupied_voxels': 0, 'empty_percentage': 100.0, 'preprocessing_time': time.time() - start_time,
                'roi_bounds': self.region_bounds, 'final_gm_point_coord': self.target_coordinate.tolist()
             }


        # --- Step 1: Calculate Rotation Transformation ---
        print(f"Debug - Calculating rotation: align gm_vec {self.skin_vec} to target {target_axis_vector} around center {self.gm_point}")
        v_orig = self.skin_vec / np.linalg.norm(self.skin_vec)
        v_targ = target_axis_vector # Already normalized

        dot_prod = np.dot(v_orig, v_targ)
        if np.isclose(dot_prod, 1.0):
            rotation_matrix = np.eye(3)
            print("Debug - Vectors already aligned. Using identity rotation.")
        elif np.isclose(dot_prod, -1.0):
            print("Debug - Vectors anti-parallel. Calculating 180-degree rotation.")
            axis_ortho = np.cross(v_orig, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis_ortho) < 1e-6:
                axis_ortho = np.cross(v_orig, np.array([0.0, 0.0, 1.0]))
            axis_ortho /= np.linalg.norm(axis_ortho)
            rot = Rotation.from_rotvec(axis_ortho * np.pi)
            rotation_matrix = rot.as_matrix()
        else:
            axis = np.cross(v_orig, v_targ)
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.clip(dot_prod, -1.0, 1.0)) # Clip for safety
            print(f"Debug - Calculated rotation axis: {axis.round(3)}, angle: {np.degrees(angle):.2f} degrees")
            rot = Rotation.from_rotvec(axis * angle)
            rotation_matrix = rot.as_matrix()

        # Create the 4x4 rotation-only transformation matrix (Translate -> Rotate -> Translate back)
        center_of_rotation = self.gm_point
        T_to_origin = np.eye(4)
        T_to_origin[:3, 3] = -center_of_rotation
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = rotation_matrix
        T_from_origin = np.eye(4)
        T_from_origin[:3, 3] = center_of_rotation
        rotation_transform_matrix = T_from_origin @ R_4x4 @ T_to_origin
        print("Debug - Calculated 4x4 rotation-only transform matrix.")


        # --- Step 2: Calculate Translation Transformation ---
        # Find where gm_point ends up *after* the rotation
        gm_point_h = np.append(self.gm_point, 1)
        rotated_gm_point_h = np.dot(gm_point_h, rotation_transform_matrix.T) # Apply rotation
        rotated_gm_point = rotated_gm_point_h[:3]
        print(f"Debug - Original gm_point {self.gm_point.round(3)} is rotated to {rotated_gm_point.round(3)}")

        # Calculate the translation needed to move rotated_gm_point to target_coordinate
        translation_vector = self.target_coordinate - rotated_gm_point
        print(f"Debug - Translation required to move rotated gm_point to target {self.target_coordinate.round(3)}: {translation_vector.round(3)}")

        # Create the 4x4 pure translation matrix
        translation_transform_matrix = np.eye(4)
        translation_transform_matrix[:3, 3] = translation_vector
        print("Debug - Calculated 4x4 translation-only transform matrix.")


        # --- Step 3: Combine Transformations ---
        # The final transform applies rotation FIRST, then translation.
        # Matrix multiplication order: Translation @ Rotation
        self._current_transform_matrix = translation_transform_matrix @ rotation_transform_matrix
        print("Debug - Combined final 4x4 transformation matrix (Rotation then Translation).")

        # --- Verification (Optional but Recommended) ---
        # Check where the original gm_point ends up after the *combined* transform
        final_gm_point_h = np.dot(gm_point_h, self._current_transform_matrix.T)
        final_gm_point_coord = final_gm_point_h[:3]
        print(f"Debug - Verification: Applying final transform to gm_point yields: {final_gm_point_coord.round(3)}")
        if not np.allclose(final_gm_point_coord, self.target_coordinate):
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"WARNING: Final transformed gm_point {final_gm_point_coord.round(3)} does NOT match target {self.target_coordinate.round(3)}!")
            print(f"Difference: {final_gm_point_coord - self.target_coordinate}")
            print(f"Check matrix calculations.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Check the orientation of the skin_vec after the combined transform
        gm_vec_end_point = self.gm_point + self.skin_vec
        gm_vec_end_point_h = np.append(gm_vec_end_point, 1)
        final_gm_vec_end_point_h = np.dot(gm_vec_end_point_h, self._current_transform_matrix.T)
        final_gm_vec_end_point = final_gm_vec_end_point_h[:3]
        final_skin_vec = final_gm_vec_end_point - final_gm_point_coord
        final_skin_vec_normalized = final_skin_vec / np.linalg.norm(final_skin_vec)
        print(f"Debug - Verification: Final transformed skin_vec orientation: {final_skin_vec_normalized.round(3)} (should be close to {target_axis_vector})")
        if not np.allclose(final_skin_vec_normalized, target_axis_vector, atol=1e-6):
             print(f"WARNING: Final transformed skin_vec orientation is not aligned with Y-axis!")
        # --- End Verification ---


        # --- Step 4: Apply Combined Transformation to All Nodes ---
        homogeneous_coords = np.ones((self.total_nodes, 4))
        homogeneous_coords[:, :3] = self.original_nodes
        transformed_homogeneous = np.dot(homogeneous_coords, self._current_transform_matrix.T)
        self.transformed_nodes = transformed_homogeneous[:, :3]
        print(f"Debug - Applied final (rotation + translation) transform to {self.total_nodes} nodes.")
        self._update_memory_usage("transformed_nodes", self.transformed_nodes)

        # --- ADDITION 2: Log Transformed Node Bounds ---
        if self.total_nodes > 0:
            transformed_min_coords = self.transformed_nodes.min(axis=0)
            transformed_max_coords = self.transformed_nodes.max(axis=0)
            print(f"Debug - Transformed Node Bounds (Min): {transformed_min_coords.round(3)}")
            print(f"Debug - Transformed Node Bounds (Max): {transformed_max_coords.round(3)}")
        else:
            # This case is technically handled earlier, but good practice
            print("Debug - Transformed Node Bounds: N/A (0 nodes)")
        # --- END ADDITION 2 ---


        # --- Step 5: Define ROI and Filter Nodes ---
        # ROI is now defined centered around the fixed target_coordinate
        self._calculate_roi_bounds() # Recalculate just in case, ensures it uses target_coord
        if self.region_bounds is None:
             raise RuntimeError("Failed to calculate ROI bounds after setting target_coordinate and region_dimensions.")

        x_bounds, y_bounds, z_bounds = self.region_bounds
        print(f"Debug - Using ROI bounds centered at target coord: X={x_bounds}, Y={y_bounds}, Z={z_bounds}")

        # Filter nodes using the ROI bounds applied to the *transformed* nodes
        self.node_mask = (
            (self.transformed_nodes[:, 0] >= x_bounds[0]) & (self.transformed_nodes[:, 0] <= x_bounds[1]) &
            (self.transformed_nodes[:, 1] >= y_bounds[0]) & (self.transformed_nodes[:, 1] <= y_bounds[1]) &
            (self.transformed_nodes[:, 2] >= z_bounds[0]) & (self.transformed_nodes[:, 2] <= z_bounds[1])
        )
        self.nodes_in_roi = np.sum(self.node_mask)
        print(f"Debug - Nodes within ROI: {self.nodes_in_roi} / {self.total_nodes}")
        self._update_memory_usage("node_mask", self.node_mask)

        # --- Step 6: Setup Bins and Voxel Mapping ---
        # Create bins and bin centers based on the final ROI bounds
        self._setup_bins_and_centers()

        # Create node-to-voxel mapping using transformed nodes and final bins
        self._create_voxel_mapping() # Uses self.transformed_nodes, self.node_mask, and bins

        elapsed_time = time.time() - start_time

        # --- Optional Visualization ---
        try:
            subject_id_str = getattr(self.context, 'subject_id', 'unknown_subject')
            viz_output_path = f"transform_visualization_{subject_id_str}_b{self.bin_size}.html"
            print(f"\n--- Visualizing Node Transformation (Rotation + Translation) for {subject_id_str} ---")
            print(f"    Saving to: {os.path.abspath(viz_output_path)}")

            # Prepare points for visualization
            points_data = [
                { 'name': f'{subject_id_str} Original',
                  'points': self.original_nodes, 'color': [0.2, 0.2, 1.0], 'size': 1.0 },
                { 'name': f'{subject_id_str} Transformed (Rot+Trans)',
                  'points': self.transformed_nodes, 'color': [1.0, 0.2, 0.2], 'size': 1.0 }
            ]
            # Add markers for key points
            points_data.extend([
                 { 'name': 'Original GM Point', 'points': self.gm_point.reshape(1, 3), 'color': [0, 1, 1], 'size': 10.0}, # Cyan
                 { 'name': 'Rotated GM Point', 'points': rotated_gm_point.reshape(1, 3), 'color': [1, 1, 0], 'size': 10.0}, # Yellow
                 { 'name': 'Final Target Coord (Goal)', 'points': self.target_coordinate.reshape(1, 3), 'color': [0, 1, 0], 'size': 15.0}, # Green (larger)
                 # --- ADD THIS MARKER ---
                 { 'name': 'Actual Final GM Point (Result)', 'points': final_gm_point_coord.reshape(1, 3), 'color': [1, 0, 1], 'size': 10.0} # Magenta
                 # --- END ADDITION ---
            ])

            visualize_point_clouds(
                 point_clouds_data=points_data,
                 output_path=viz_output_path
                 # point_size handled within point_clouds_data now
            )
            print(f"--- Visualization Saved ---\n")
        except NameError: print("WARNING: visualize_point_clouds function not found. Skipping visualization.")
        except Exception as e: print(f"WARNING: Failed visualization: {e}")
        # --- End Visualization ---


        # Save preprocessing results (reflects rotation + translation)
        if save_path:
            # Ensure the directory exists
            save_dir = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_dir, exist_ok=True)
            # Add transform matrix to saved data
            self._save_preprocessing(save_path)
            print(f"Debug - Saved preprocessing results to {save_path}")


        # Return statistics
        empty_percentage = ((self.total_voxels - self.occupied_voxels) / self.total_voxels * 100) if self.total_voxels > 0 else 0
        stats = {
            'total_nodes': self.total_nodes, 'nodes_in_roi': self.nodes_in_roi,
            'total_voxels': self.total_voxels, 'occupied_voxels': self.occupied_voxels,
            'empty_percentage': empty_percentage, 'preprocessing_time': elapsed_time,
            'roi_bounds': self.region_bounds,
            'original_gm_point': self.gm_point.tolist(),
            'original_skin_vec': self.skin_vec.tolist(),
            'target_axis_vector': target_axis_vector.tolist(),
            'final_gm_point_coord': final_gm_point_coord.tolist(), # Actual final coord
            'target_coordinate': self.target_coordinate.tolist() # The intended target coord
        }
        if not np.allclose(final_gm_point_coord, self.target_coordinate):
             stats['warning'] = "Final GM point position did not exactly match target coordinate."

        return stats


    def _setup_bins_and_centers(self):
        """Sets up bin edges and centers based on self.region_bounds."""
        if self.region_bounds is None:
             raise ValueError("Cannot set up bins without defined region_bounds.")
        if self.bin_size <= 0:
             raise ValueError("bin_size must be positive.")

        x_bounds, y_bounds, z_bounds = self.region_bounds
        self.x_bins = np.linspace(x_bounds[0], x_bounds[1], self.bin_size + 1)
        self.y_bins = np.linspace(y_bounds[0], y_bounds[1], self.bin_size + 1)
        self.z_bins = np.linspace(z_bounds[0], z_bounds[1], self.bin_size + 1)

        self.x_centers = (self.x_bins[:-1] + self.x_bins[1:]) / 2
        self.y_centers = (self.y_bins[:-1] + self.y_bins[1:]) / 2
        self.z_centers = (self.z_bins[:-1] + self.z_bins[1:]) / 2
        # print("Debug - Setup bins and centers based on ROI.")


    # ============================================================
    #  Keep the following methods as they were:
    #  - _create_voxel_mapping
    #  - _save_preprocessing (consider adding _current_transform_matrix to saved data)
    #  - from_saved (consider loading _current_transform_matrix if saved)
    #  - process_field
    #  - _process_field_optimized
    #  - get_voxelized_data_as_grid
    #  - generate_mask
    #  - transform
    #  - _update_memory_usage
    #  - _reduce_memory
    # ============================================================

    # --- MODIFICATION to _save_preprocessing ---
    def _save_preprocessing(self, save_path: str):
        """Save preprocessing results for future reuse.

        Args:
            save_path: Path to save preprocessing results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # Data to save
        preprocessing_data = {
            # Core mapping data
            'node_to_voxel_map': self.node_to_voxel_map,
            'voxel_node_counts': self.voxel_node_counts,
            'voxel_coordinates': self.voxel_coordinates,
            'node_mask': self.node_mask,

            # Statistics and config
            'total_nodes': self.total_nodes,
            'nodes_in_roi': self.nodes_in_roi,
            'total_voxels': self.total_voxels,
            'occupied_voxels': self.occupied_voxels,
            'bin_size': self.bin_size,
            'region_bounds': self.region_bounds,

            # Transformation info (NEW)
            'original_gm_point': self.gm_point,
            'original_skin_vec': self.skin_vec,
            'target_coordinate': self.target_coordinate,
            'final_transform_matrix': self._current_transform_matrix,
            'region_dimensions': self.region_dimensions, # Save dimensions used for ROI

            # Save original nodes if needed for inverse transform debugging etc.
            # 'original_nodes': self.original_nodes, # Optional, can be large
        }

        # Save as pickle
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(preprocessing_data, f)
        except Exception as e:
            print(f"ERROR: Failed to save preprocessing data to {save_path}: {e}")
            # Decide if you want to raise the error or just warn
            # raise

        # Log if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "preprocessing_saved",
                {'save_path': save_path}
            )

    # --- MODIFICATION to from_saved ---
    @classmethod
    def from_saved(
        cls,
        save_path: str,
        context: TMSPipelineContext,
        debug_hook: Optional[PipelineDebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """Load a VoxelMapper from saved preprocessing results.

        Args:
            save_path: Path to saved preprocessing results
            context: TMS-specific pipeline context
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking

        Returns:
            Initialized VoxelMapper with loaded preprocessing results
        """
        # Log if debug enabled
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "loading_saved_preprocessing",
                {'save_path': save_path}
            )

        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Saved preprocessing file not found: {save_path}")

        try:
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load or unpickle data from {save_path}: {e}") from e

        # Create instance - using loaded parameters
        # Make sure all necessary args for __init__ are present in 'data' or have defaults
        mapper = cls(
            context=context,
            mesh_path=None, # Mesh path not strictly needed if only using saved data
            gm_point=data.get('original_gm_point'), # Load if saved
            skin_vec=data.get('original_skin_vec'), # Load if saved
            target_coordinate=data.get('target_coordinate'), # Load if saved
            region_dimensions=data.get('region_dimensions'), # Load if saved
            bin_size=data['bin_size'], # Required key
            debug_hook=debug_hook,
            resource_monitor=resource_monitor
        )

        # Load essential saved data
        mapper.node_to_voxel_map = data.get('node_to_voxel_map')
        mapper.voxel_node_counts = data.get('voxel_node_counts')
        mapper.voxel_coordinates = data.get('voxel_coordinates')
        mapper.node_mask = data.get('node_mask')
        mapper.total_nodes = data.get('total_nodes')
        mapper.nodes_in_roi = data.get('nodes_in_roi')
        mapper.total_voxels = data.get('total_voxels')
        mapper.occupied_voxels = data.get('occupied_voxels')
        mapper.region_bounds = data.get('region_bounds')

        # Load transformation matrix (NEW)
        mapper._current_transform_matrix = data.get('final_transform_matrix')

        # Reconstruct bins and centers from loaded bounds (important!)
        if mapper.region_bounds and mapper.bin_size > 0:
             mapper._setup_bins_and_centers()
        else:
             print("Warning: Could not reconstruct bins/centers from loaded data (missing bounds or invalid bin size).")


        # --- Data Validation (Optional but Recommended) ---
        essential_keys = ['node_to_voxel_map', 'voxel_node_counts', 'voxel_coordinates',
                          'node_mask', 'total_nodes', 'nodes_in_roi', 'total_voxels',
                          'occupied_voxels', 'bin_size', 'region_bounds',
                          'final_transform_matrix'] # Add new essential keys
        missing_keys = [k for k in essential_keys if getattr(mapper, k, None) is None and data.get(k) is None]
        if missing_keys:
             print(f"Warning: Loaded data from {save_path} is missing essential keys: {missing_keys}")
             # Decide if this should be a critical error
             # raise ValueError(f"Loaded data missing keys: {missing_keys}")

        # Update memory tracking based on loaded data
        # (You might want to recalculate memory based on loaded objects)
        mapper._memory_usage = 0 # Reset memory
        for key in ['node_to_voxel_map', 'voxel_node_counts', 'voxel_coordinates', 'node_mask', '_current_transform_matrix']:
             if hasattr(mapper, key) and getattr(mapper, key) is not None:
                  mapper._update_memory_usage(key, getattr(mapper, key))


        # Log if debug enabled
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "saved_preprocessing_loaded",
                {
                    'total_nodes': mapper.total_nodes,
                    'nodes_in_roi': mapper.nodes_in_roi,
                    'occupied_voxels': mapper.occupied_voxels,
                    'transform_matrix_loaded': mapper._current_transform_matrix is not None
                }
            )

        print(f"Debug - Loaded VoxelMapper state from {save_path}")
        return mapper

    # --- Remaining methods (unchanged unless dependent on new state) ---
    def _create_voxel_mapping(self):
        """Create mapping from node indices to voxel indices."""
        # Check prerequisites
        if self.transformed_nodes is None or self.node_mask is None:
            raise RuntimeError("Transformed nodes or node mask not available for voxel mapping.")
        if self.x_bins is None or self.y_bins is None or self.z_bins is None:
             raise RuntimeError("Bins are not set up for voxel mapping.")
        if self.x_centers is None or self.y_centers is None or self.z_centers is None:
             raise RuntimeError("Bin centers are not set up for voxel mapping.")


        # Log start if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "voxel_mapping_started",
                {'bin_size': self.bin_size, 'nodes_in_roi': self.nodes_in_roi}
            )

        # Initialize data structures
        self.node_to_voxel_map = {}  # Maps original node index to flat voxel index
        self.voxel_node_counts = {}  # Counts nodes per flat voxel index
        voxel_coords = {}            # Stores center coordinates per flat voxel index

        # Only process nodes within ROI
        roi_original_indices = np.where(self.node_mask)[0]
        if len(roi_original_indices) == 0:
             print("Debug - No nodes found within ROI. Voxel map will be empty.")
             self.voxel_coordinates = {}
             self.total_voxels = self.bin_size**3
             self.occupied_voxels = 0
             # Update memory tracking for empty structures
             self._update_memory_usage("node_to_voxel_map", self.node_to_voxel_map)
             self._update_memory_usage("voxel_node_counts", self.voxel_node_counts)
             self._update_memory_usage("voxel_coordinates", self.voxel_coordinates)
             return # Nothing more to do

        roi_transformed_coords = self.transformed_nodes[roi_original_indices]

        # Digitize coordinates to get bin indices (relative to the ROI bins)
        # np.digitize returns indices starting from 1, so subtract 1
        x_indices = np.digitize(roi_transformed_coords[:, 0], self.x_bins) - 1
        y_indices = np.digitize(roi_transformed_coords[:, 1], self.y_bins) - 1
        z_indices = np.digitize(roi_transformed_coords[:, 2], self.z_bins) - 1

        # Clip indices to ensure they are within [0, bin_size - 1]
        # This handles points exactly on the upper boundary edge
        x_indices = np.clip(x_indices, 0, self.bin_size - 1)
        y_indices = np.clip(y_indices, 0, self.bin_size - 1)
        z_indices = np.clip(z_indices, 0, self.bin_size - 1)

        # Create flat voxel indices (C-style flattening: Z varies fastest)
        # Make sure indices are integer type for calculations
        voxel_indices_flat = (
            x_indices.astype(int) * (self.bin_size**2) +
            y_indices.astype(int) * self.bin_size +
            z_indices.astype(int)
        )

        # Build mapping and count nodes per voxel
        # Iterate using the original indices and the calculated flat voxel indices
        for i, original_node_idx in enumerate(roi_original_indices):
            voxel_idx_flat = voxel_indices_flat[i]
            bin_x, bin_y, bin_z = x_indices[i], y_indices[i], z_indices[i]

            # Map original node index to its flat voxel index
            self.node_to_voxel_map[original_node_idx] = voxel_idx_flat

            # Count nodes in each voxel and store coords if first node in voxel
            if voxel_idx_flat in self.voxel_node_counts:
                self.voxel_node_counts[voxel_idx_flat] += 1
            else:
                self.voxel_node_counts[voxel_idx_flat] = 1
                # Store voxel center coordinates using the bin indices
                voxel_coords[voxel_idx_flat] = (
                    self.x_centers[bin_x],
                    self.y_centers[bin_y],
                    self.z_centers[bin_z]
                )

        # Store voxel coordinates dictionary
        self.voxel_coordinates = voxel_coords

        # Update memory tracking
        self._update_memory_usage("node_to_voxel_map", self.node_to_voxel_map)
        self._update_memory_usage("voxel_node_counts", self.voxel_node_counts)
        self._update_memory_usage("voxel_coordinates", self.voxel_coordinates)

        # Calculate final statistics
        self.total_voxels = self.bin_size**3
        self.occupied_voxels = len(self.voxel_node_counts)
        occupancy_percentage = (self.occupied_voxels / self.total_voxels * 100) if self.total_voxels > 0 else 0
        print(f"Debug - Voxel mapping created. Total Voxels: {self.total_voxels}, Occupied: {self.occupied_voxels} ({occupancy_percentage:.2f}%)")


        # Log completion if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "voxel_mapping_completed",
                {
                    'total_voxels': self.total_voxels,
                    'occupied_voxels': self.occupied_voxels,
                    'occupancy_percentage': occupancy_percentage
                }
            )


    def process_field(
        self,
        field_data: np.ndarray,
        output_grid: bool = True
    ) -> Union[Dict[int, np.ndarray], np.ndarray]:
        """Process a field array, mapping node values to voxel values using mean aggregation.

        Args:
            field_data: Field data values at mesh nodes (shape [N,] or [N, D]).
            output_grid: Whether to return as regular grid (True) or dict (False).

        Returns:
            Voxelized field data (as 3D/4D grid or dictionary mapping flat voxel index to value).

        Raises:
            ValueError: If preprocessing has not been done or field data size is incorrect.
        """
        if self.node_to_voxel_map is None or self.voxel_node_counts is None:
            raise ValueError("Preprocessing has not been done. Call preprocess() or load from saved first.")

        start_time = time.time()

        # Log start if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "process_field_started",
                {'field_data_shape': field_data.shape, 'output_grid': output_grid}
            )

        # Check if field data size matches the total number of original nodes
        if len(field_data) != self.total_nodes:
             # Check if it matches the number of elements (if available)
             # if hasattr(self, 'total_elements') and len(field_data) == self.total_elements:
             #     # Handle element-based data if necessary (might require different mapping logic)
             #     raise NotImplementedError("Processing element-based field data is not yet fully supported.")
             # else:
                 raise ValueError(f"Field data size ({len(field_data)}) does not match mesh node count ({self.total_nodes}).")

        # Determine if field data is vector (e.g., E-field) or scalar (e.g., magnitude)
        is_vector = field_data.ndim > 1 and field_data.shape[1] > 1
        vector_dim = field_data.shape[1] if is_vector else 1


        # Process field data using the optimized method
        # This returns a dictionary: {flat_voxel_idx: mean_value_in_voxel}
        voxelized_data_dict = self._process_field_optimized(field_data, is_vector)


        # Convert to grid if requested
        if output_grid:
            # This creates the dense [bin, bin, bin] or [bin, bin, bin, dim] grid
            result = self.get_voxelized_data_as_grid(voxelized_data_dict, is_vector, vector_dim)
            result_shape = result.shape
        else:
            result = voxelized_data_dict
            result_shape = f"dict[{len(result)}]" # Describe shape for dict

        elapsed_time = time.time() - start_time

        # Log completion if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "process_field_completed",
                {
                    'execution_time': elapsed_time,
                    'voxelized_shape': result_shape,
                    'is_vector': is_vector
                }
            )

        return result


    def _process_field_optimized(
        self,
        field_array: np.ndarray,
        is_vector: bool
    ) -> Dict[int, np.ndarray]:
        """Optimized function to process a field array by aggregating values per voxel.

        Args:
            field_array: Field array corresponding to original nodes (shape [N,] or [N, D]).
            is_vector: Whether the data is vector or scalar.

        Returns:
            Voxelized field data as dictionary {flat_voxel_idx: mean_value}.
            For scalar data, mean_value is float. For vector, it's np.ndarray.
        """
        if not self.node_to_voxel_map: # Handle empty map case
             return {}

        # Initialize dictionary to store sums and counts for averaging
        # Using tuples: (sum_of_values, count)
        voxel_aggregates: Dict[int, Tuple[Union[float, np.ndarray], int]] = {}

        # Iterate through the node->voxel map (contains only nodes within ROI)
        for node_idx, voxel_idx_flat in self.node_to_voxel_map.items():
            # Get the field value for the current node
            value = field_array[node_idx]

            if voxel_idx_flat in voxel_aggregates:
                current_sum, current_count = voxel_aggregates[voxel_idx_flat]
                new_sum = current_sum + value
                new_count = current_count + 1
                voxel_aggregates[voxel_idx_flat] = (new_sum, new_count)
            else:
                # Initialize sum and count for this voxel
                voxel_aggregates[voxel_idx_flat] = (value, 1)

        # Calculate the mean for each voxel
        voxelized_data_mean: Dict[int, Union[float, np.ndarray]] = {}
        for voxel_idx_flat, (total_sum, count) in voxel_aggregates.items():
            if count > 0:
                 mean_value = total_sum / count
                 voxelized_data_mean[voxel_idx_flat] = mean_value
            # else: # Should not happen with this logic, but for robustness:
            #     voxelized_data_mean[voxel_idx_flat] = np.zeros_like(total_sum) if isinstance(total_sum, np.ndarray) else 0.0


        return voxelized_data_mean

    def get_voxelized_data_as_grid(
        self,
        voxelized_data_dict: Dict[int, np.ndarray],
        is_vector: bool,
        vector_dim: Optional[int] = None # Pass vector dimension if known
    ) -> np.ndarray:
        """Convert voxelized data from dictionary format to a dense 3D/4D grid.

        Args:
            voxelized_data_dict: Voxelized data {flat_voxel_idx: value}.
            is_vector: Whether the data is vector or scalar.
            vector_dim: The dimension of vectors if is_vector is True.

        Returns:
            3D grid (if scalar) or 4D grid (if vector) with voxelized data.
            Empty voxels will have a value of 0.
        """
        # Determine grid shape and data type
        if is_vector:
            if vector_dim is None:
                 # Try to infer from the first item if dict is not empty
                 if voxelized_data_dict:
                      first_val = next(iter(voxelized_data_dict.values()))
                      if isinstance(first_val, np.ndarray):
                           vector_dim = len(first_val)
                      else: # Should be ndarray for vector, but handle scalar case
                           print("Warning: is_vector=True but data seems scalar. Assuming dim=1.")
                           vector_dim = 1
                           is_vector = False # Treat as scalar if inference fails
                 else: # Dictionary is empty
                      print("Warning: is_vector=True but no data to infer dimension. Assuming dim=3.")
                      vector_dim = 3 # Default assumption or raise error?

            grid_shape = (self.bin_size, self.bin_size, self.bin_size, vector_dim)
            # Infer dtype from first element or default to float
            dtype = next(iter(voxelized_data_dict.values())).dtype if voxelized_data_dict else float
        else:
            grid_shape = (self.bin_size, self.bin_size, self.bin_size)
            dtype = type(next(iter(voxelized_data_dict.values()))) if voxelized_data_dict else float


        # Create empty grid filled with zeros
        grid = np.zeros(grid_shape, dtype=dtype)

        # Fill grid with data from the dictionary
        if not voxelized_data_dict:
             return grid # Return empty grid if dict is empty

        # Vectorized index calculation (more efficient than loop for large dicts)
        if len(voxelized_data_dict) > 0:
            flat_indices = np.array(list(voxelized_data_dict.keys()), dtype=int)
            values = np.array(list(voxelized_data_dict.values())) # Shape [N,] or [N, D]

            # Convert flat indices to 3D indices (matching C-style flattening)
            z_indices = flat_indices % self.bin_size
            y_indices = (flat_indices // self.bin_size) % self.bin_size
            x_indices = flat_indices // (self.bin_size**2)

            # Assign values to the grid using advanced indexing
            if is_vector:
                grid[x_indices, y_indices, z_indices, :] = values
            else:
                grid[x_indices, y_indices, z_indices] = values

        return grid


    def generate_mask(self) -> np.ndarray:
        """Generate a binary mask indicating valid (occupied) voxels in the grid.

        Returns:
            Binary mask of shape (bin_size, bin_size, bin_size), True for occupied voxels.
        """
        if self.voxel_node_counts is None:
            raise ValueError("Preprocessing has not been done. Cannot generate mask.")

        # Create empty mask (all False)
        mask = np.zeros((self.bin_size, self.bin_size, self.bin_size), dtype=bool)

        # Get flat indices of occupied voxels
        occupied_flat_indices = np.array(list(self.voxel_node_counts.keys()), dtype=int)

        if len(occupied_flat_indices) > 0:
            # Convert flat indices to 3D indices
            z_indices = occupied_flat_indices % self.bin_size
            y_indices = (occupied_flat_indices // self.bin_size) % self.bin_size
            x_indices = occupied_flat_indices // (self.bin_size**2)

            # Set corresponding positions in the mask to True
            mask[x_indices, y_indices, z_indices] = True

        # Optional: Dilate mask (consider if needed for your application)
        # if self.context.config.get('mask_dilation', False):
        #     try:
        #         from scipy import ndimage
        #         iterations = self.context.config.get('dilation_iterations', 1)
        #         mask = ndimage.binary_dilation(mask, iterations=iterations)
        #         print(f"Debug - Applied binary dilation to mask with {iterations} iterations.")
        #     except ImportError:
        #         print("Warning: scipy not found. Cannot perform mask dilation.")


        # Log if debug enabled
        non_zero_voxels = np.count_nonzero(mask)
        coverage_percent = 100 * non_zero_voxels / mask.size if mask.size > 0 else 0
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "mask_generated",
                {
                    'mask_shape': mask.shape,
                    'non_zero_voxels': non_zero_voxels,
                    'coverage_percent': coverage_percent
                }
            )

        return mask

    def transform(
        self,
        field_data: np.ndarray,
        # node_centers: Optional[np.ndarray] = None, # Not used in current logic
        mesh_path: Optional[str] = None # Used only if preprocess not run
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Complete mesh to grid transformation pipeline: preprocess (if needed),
        process field, generate mask. Mimics an older interface.

        Args:
            field_data: Field data values at mesh nodes (shape [N,] or [N, D]).
            mesh_path: Path to mesh file (only needed if preprocess() hasn't been called).

        Returns:
            Tuple of (voxelized_data_grid, mask, metadata).
        """
        start_time = time.time()

        # Check if preprocessing is needed (i.e., mapping doesn't exist)
        if self.node_to_voxel_map is None:
            print("Debug - Preprocessing needed within transform call.")
            if mesh_path and self.mesh_path is None:
                self.mesh_path = mesh_path
                print(f"Debug - Using provided mesh path for preprocessing: {self.mesh_path}")

            if self.mesh_path is None:
                raise ValueError("VoxelMapper.transform requires either prior preprocessing or a mesh_path.")
            if self.gm_point is None or self.skin_vec is None or self.target_coordinate is None or self.region_dimensions is None:
                 raise ValueError("VoxelMapper needs gm_point, skin_vec, target_coordinate, and region_dimensions set before preprocessing.")

            # Perform preprocessing
            preprocess_stats = self.preprocess()
            print(f"Debug - Preprocessing completed within transform: {preprocess_stats}")


        # Process the field data into a grid
        is_vector = field_data.ndim > 1 and field_data.shape[1] > 1
        voxelized_data_grid = self.process_field(field_data, output_grid=True)

        # Generate the binary mask
        mask = self.generate_mask()

        # Create metadata dictionary
        total_time = time.time() - start_time
        metadata = {
            'grid_shape': voxelized_data_grid.shape,
            'mask_shape': mask.shape,
            'is_vector': is_vector,
            'execution_time': total_time,
            'bin_size': self.bin_size,
            'total_voxels': self.total_voxels,
            'occupied_voxels': self.occupied_voxels,
            'coverage_percent': (self.occupied_voxels / self.total_voxels * 100) if self.total_voxels > 0 else 0,
            'roi_bounds': self.region_bounds,
            'transformation_details': {
                 'type': 'rotation_then_translation',
                 'target_coordinate': self.target_coordinate.tolist() if self.target_coordinate is not None else None
            }
        }

        # Log if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "transform_completed",
                {
                    'field_data_shape': field_data.shape,
                    'output_grid_shape': voxelized_data_grid.shape,
                    'mask_shape': mask.shape,
                    'execution_time': metadata['execution_time']
                }
            )

        return voxelized_data_grid, mask, metadata


    def _update_memory_usage(self, key: str, data: Any) -> None:
        """Update memory usage tracking (simplified)."""
        mem_bytes = 0
        try:
            if isinstance(data, np.ndarray):
                mem_bytes = data.nbytes
            elif isinstance(data, dict) or isinstance(data, list) or isinstance(data, tuple):
                 # Very rough estimate for containers
                 mem_bytes = sys.getsizeof(data)
                 # Add elements' size - this can be slow for large containers
                 # if isinstance(data, dict):
                 #     for k, v in data.items(): mem_bytes += sys.getsizeof(k) + sys.getsizeof(v)
                 # elif isinstance(data, (list, tuple)):
                 #      for item in data: mem_bytes += sys.getsizeof(item)
            else:
                mem_bytes = sys.getsizeof(data)
        except Exception: # Catch potential errors in getsizeof
             mem_bytes = 0 # Assign 0 if size calculation fails

        # Update total and intermediate tracking
        old_usage = self._intermediate_data.get(key, {}).get('bytes', 0)
        self._memory_usage -= old_usage
        self._memory_usage += mem_bytes

        shape_str = str(getattr(data, 'shape', type(data)))
        dtype_str = str(getattr(data, 'dtype', type(data)))

        self._intermediate_data[key] = {
            'shape': shape_str,
            'dtype': dtype_str,
            'bytes': mem_bytes
        }

        # Update resource monitor if available
        if self.resource_monitor:
            self.resource_monitor.update_component_usage(
                "voxel_mapper",
                self._memory_usage
            )


    def _reduce_memory(self, target_reduction: float) -> None:
        """Callback for memory reduction requests (basic implementation)."""
        current_memory = self._memory_usage
        target_memory = current_memory * (1.0 - target_reduction)
        memory_reduced = 0

        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "memory_reduction_requested",
                {
                    'component': 'voxel_mapper',
                    'target_reduction': target_reduction,
                    'current_memory': current_memory,
                    'target_memory': target_memory
                }
            )

        # Identify potentially clearable data (example: original nodes if kept)
        keys_to_consider = ['original_nodes', 'transformed_nodes'] # Add others if applicable
        keys_cleared = []

        for key in keys_to_consider:
            if key in self._intermediate_data:
                mem_saved = self._intermediate_data[key]['bytes']
                # Check if clearing this helps meet the target
                if current_memory - memory_reduced - mem_saved <= target_memory:
                    print(f"Debug (Memory Reduction) - Clearing '{key}' (saves approx {mem_saved} bytes)")
                    # Remove from tracking
                    del self._intermediate_data[key]
                    # Nullify the attribute in the class instance
                    if hasattr(self, key):
                        setattr(self, key, None)
                    self._memory_usage -= mem_saved
                    memory_reduced += mem_saved
                    keys_cleared.append(key)
                    # Check if target met
                    if self._memory_usage <= target_memory:
                        break # Stop clearing once target is met

        # Update resource monitor with final usage
        if self.resource_monitor:
            self.resource_monitor.update_component_usage(
                "voxel_mapper",
                self._memory_usage
            )

        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "memory_reduction_complete",
                {
                    'component': 'voxel_mapper',
                    'cleared_keys': keys_cleared,
                    'memory_reduced': memory_reduced,
                    'final_memory': self._memory_usage
                }
            )