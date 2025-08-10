#!/usr/bin/env python3
"""
TMS E-field simulation runner script for MA_Henry/data structure.

This script runs TMS simulations using the tms_efield_prediction implementation
while accessing data from and writing outputs to the MA_Henry/data directory structure.
"""

import os
import sys
import argparse
import logging
import numpy as np
import json
import shutil
import glob
from datetime import datetime
import concurrent.futures
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tms_simulation')

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

# Import simulation components
from tms_efield_prediction.simulation.runner import run_simulation
from tms_efield_prediction.simulation.coil_position import CoilPositioningConfig
from tms_efield_prediction.simulation.field_calculation import FieldCalculationConfig
from tms_efield_prediction.simulation.tms_simulation import SimulationContext
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run TMS E-field simulations")
    
    parser.add_argument("--subject", default="002", help="Subject ID (default: 001)")
    parser.add_argument("--data-dir", default="../", 
                      help="Path to home/freyhe/MA_Henry/ directory (default: ../)")
    parser.add_argument("--exp-type", default="new_simulation",
                      help="Experiment type subdirectory (default: new_simulation)")
    parser.add_argument("--output-dir", default=None,
                      help="Custom output directory (if default location isn't writable)")
    parser.add_argument("--n-cpus", type=int, default=1,
                      help="Number of CPU cores to use (default: 1)")
    parser.add_argument("--search-radius", type=float, default=50.0,
                      help="Search radius in mm (default: 50.0)")
    parser.add_argument("--spatial-res", type=float, default=2.0,
                      help="Spatial resolution in mm (default: 2.0)")
    parser.add_argument("--coil-distance", type=float, default=2.5,
                      help="Distance from skin in mm (default: 2.5)")
    parser.add_argument("--rotation-step", type=float, default=10.0,
                      help="Rotation step in degrees (default: 10.0)")
    parser.add_argument("--coil-file", default="MagVenture_Cool-B65.ccd",
                      help="Coil definition file (default: MagVenture_Cool-B65.ccd)")
    
    # Keep 'none' as a valid choice, but document that it maps to 'scalar' internally
    parser.add_argument("--anisotropy", default="vn", 
                      choices=["none", "vn", "dir", "scalar"],
                      help="Conductivity anisotropy type: none/scalar (isotropic), vn (volume normalized), dir (direct). (default: vn)")
    
    # Add batch processing parameters
    parser.add_argument("--n-batches", type=int, default=None,
                      help="Number of batches to split the simulation into (default: auto)")
    parser.add_argument("--batch-index", type=int, default=None,
                      help="Index of batch to process (default: process all batches)")
    parser.add_argument("--max-positions", type=int, default=None,
                      help="Maximum number of positions to process (default: all)")
    parser.add_argument("--clean-temp", action="store_true",
                      help="Clean temporary files after simulation")
    
    return parser.parse_args()
def print_file_info(file_path):
    """Print detailed information about a file for debugging."""
    logger.log(logging.INFO,f"Checking file: {file_path}")
    
    if os.path.exists(file_path):
        logger.log(logging.INFO,f"  - File exists")
        logger.log(logging.INFO,f"  - Size: {os.path.getsize(file_path)} bytes")
        logger.log(logging.INFO,f"  - Last modified: {datetime.fromtimestamp(os.path.getmtime(file_path))}")
        
        # Try to determine if it's a binary or text file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Try to read as text
                logger.log(logging.INFO,f"  - File type: Appears to be a text file")
        except UnicodeDecodeError:
            logger.log(logging.INFO,f"  - File type: Appears to be a binary file")
        except Exception as e:
            logger.log(logging.INFO,f"  - File type check error: {str(e)}")
    else:
        logger.log(logging.INFO,f"  - File does not exist")
        
        # Check if directory exists
        dir_path = os.path.dirname(file_path)
        if os.path.exists(dir_path):
            logger.log(logging.INFO,f"  - Parent directory exists: {dir_path}")
            logger.log(logging.INFO,f"  - Directory contents: {os.listdir(dir_path)}")
        else:
            logger.log(logging.INFO,f"  - Parent directory does not exist: {dir_path}")


def verify_data_structure(data_dir, subject_id, custom_output_dir=None, coil_file="MagVenture_Cool-B65.ccd"):
    """
    Verify that the necessary files exist in the data structure.
    
    Args:
        data_dir: Path to MA_Henry/data directory
        subject_id: Subject ID
        custom_output_dir: Custom output directory path (optional)
        coil_file: Name of the coil file to search for (default: MagVenture_Cool-B65.ccd)
        
    Returns:
        dict: Dictionary of verified file paths
    """
    # Define expected paths
    subject_dir = f"sub-{subject_id}"
    
    # Check if we need to add 'data' to the path
    data_path = data_dir
    if os.path.exists(os.path.join(data_dir, "data", subject_dir)):
        data_path = os.path.join(data_dir, "data")
        logger.log(logging.INFO, f"Using data subdirectory path: {data_path}")

    subject_path = os.path.join(data_path, subject_dir)
    if not os.path.exists(subject_path):
        logger.error(f"Subject directory not found: {subject_path}")
        return None
    
    # Define the key directories
    experiment_dir = os.path.join(subject_path, "experiment")
    if not os.path.exists(experiment_dir):
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return None
    
    # Check for 'all' directory in the new structure
    experiment_all_dir = os.path.join(experiment_dir, "all")
    has_new_structure = os.path.exists(experiment_all_dir)
    
    if has_new_structure:
        logger.log(logging.INFO, f"Using new directory structure with 'all' directory: {experiment_all_dir}")
    else:
        logger.warning(f"'all' directory not found in experiment: {experiment_all_dir}")
        logger.warning("Using older directory structure")
    
    headmodel_dir = os.path.join(subject_path, "headmodel")
    if not os.path.exists(headmodel_dir):
        logger.warning(f"Headmodel directory not found: {headmodel_dir}")
    
    # Initialize paths dictionary
    paths = {
        "mesh": None,         # Full head mesh for positioning
        "mesh_roi": None,     # ROI mesh for simulation
        "roi_center": None,
        "coil": None,
        "output_dir": custom_output_dir if custom_output_dir else os.path.join(subject_path, "experiment", "simulation_results")
    }
    
    # First, try to find the full head mesh (important for positioning)
    # Look in headmodel directory first
    if os.path.exists(headmodel_dir):
        head_mesh_path = os.path.join(headmodel_dir, f"{subject_dir}.msh")
        if os.path.exists(head_mesh_path):
            paths["mesh"] = head_mesh_path
            logger.log(logging.INFO, f"Found full head mesh in headmodel directory: {head_mesh_path}")
    
    # If full head mesh not found yet, check other locations
    if not paths["mesh"]:
        # Check standard locations for head mesh
        head_mesh_paths = [
            os.path.join(headmodel_dir, f"{subject_id}.msh"),
            os.path.join(headmodel_dir, f"sub-{subject_id}.msh"),
            os.path.join(experiment_dir, f"{subject_dir}.msh"),
            os.path.join(experiment_all_dir, f"{subject_dir}.msh") if has_new_structure else None,
        ]
        
        for head_path in [p for p in head_mesh_paths if p]:
            if os.path.exists(head_path):
                paths["mesh"] = head_path
                logger.log(logging.INFO, f"Found full head mesh: {head_path}")
                break
    
    # If we still don't have a full mesh, use gray matter mesh as a fallback
    if not paths["mesh"] and has_new_structure:
        # Check the "middle_gray_matter.msh" in all directory 
        middle_gray_path = os.path.join(experiment_all_dir, f"{subject_dir}_middle_gray_matter.msh")
        if os.path.exists(middle_gray_path):
            paths["mesh"] = middle_gray_path
            logger.log(logging.INFO, f"Found mesh file (new structure): {middle_gray_path}")
            logger.warning("Using middle gray matter mesh for positioning - may need fallback method")
    
    # If we still don't have any mesh, check older structure
    if not paths["mesh"] and os.path.exists(experiment_dir):
        middle_gray_path = os.path.join(experiment_dir, f"{subject_dir}_middle_gray_matter.msh")
        if os.path.exists(middle_gray_path):
            paths["mesh"] = middle_gray_path
            logger.log(logging.INFO, f"Found mesh file (old structure): {middle_gray_path}")
            logger.warning("Using middle gray matter mesh for positioning - may need fallback method")
    
    # Now look for the ROI mesh
    # Check for ROI mesh (prioritize new structure)
    if has_new_structure:
        # First priority: check in experiment/all directory (new structure)
        roi_mesh_path = os.path.join(experiment_all_dir, f"{subject_dir}_middle_gray_matter_roi.msh")
        if os.path.exists(roi_mesh_path):
            paths["mesh_roi"] = roi_mesh_path
            logger.log(logging.INFO, f"Found ROI mesh file (new structure): {roi_mesh_path}")
    
    # If not found in new structure, check in headmodel directory (old structure)
    if not paths.get("mesh_roi") and os.path.exists(headmodel_dir):
        roi_mesh_path = os.path.join(headmodel_dir, f"{subject_dir}_middle_gray_matter_roi.msh")
        if os.path.exists(roi_mesh_path):
            paths["mesh_roi"] = roi_mesh_path
            logger.log(logging.INFO, f"Found ROI mesh file (old structure): {roi_mesh_path}")
    
    # Also check experiment directory as a fallback
    if not paths.get("mesh_roi"):
        roi_mesh_path = os.path.join(experiment_dir, f"{subject_dir}_middle_gray_matter_roi.msh")
        if os.path.exists(roi_mesh_path):
            paths["mesh_roi"] = roi_mesh_path
            logger.log(logging.INFO, f"Found ROI mesh file (experiment dir): {roi_mesh_path}")
    
    # If we have no ROI mesh, use the full mesh as a fallback
    if not paths.get("mesh_roi") and paths.get("mesh"):
        logger.log(logging.INFO, f"No separate ROI mesh found. Will use full mesh for simulation.")
        paths["mesh_roi"] = paths["mesh"]
    
    # If we still don't have any mesh, we can't proceed
    if not paths["mesh"]:
        logger.error(f"No mesh file found for subject {subject_id}")
        return None
    
    if not paths["mesh_roi"]:
        logger.error(f"No ROI mesh file found for subject {subject_id}")
        return None
    
    # Check for ROI center file - prioritize new structure
    if has_new_structure:
        roi_center_path = os.path.join(experiment_all_dir, f"{subject_dir}_roi_center.mat")
        if os.path.exists(roi_center_path):
            paths["roi_center"] = roi_center_path
            logger.log(logging.INFO, f"Found ROI center file (new structure): {roi_center_path}")
        else:
            # Try experiment directory
            roi_center_path = os.path.join(experiment_dir, f"{subject_dir}_roi_center.mat")
            if os.path.exists(roi_center_path):
                paths["roi_center"] = roi_center_path
                logger.log(logging.INFO, f"Found ROI center file: {roi_center_path}")
    else:
        # Try experiment directory first (most common location)
        roi_center_path = os.path.join(experiment_dir, f"{subject_dir}_roi_center.mat")
        if os.path.exists(roi_center_path):
            paths["roi_center"] = roi_center_path
            logger.log(logging.INFO, f"Found ROI center file: {roi_center_path}")
        else:
            # Try alternative name
            roi_center_path = os.path.join(experiment_dir, f"{subject_dir}_roi_init.mat")
            if os.path.exists(roi_center_path):
                paths["roi_center"] = roi_center_path
                logger.log(logging.INFO, f"Found ROI center file (alternative): {roi_center_path}")
    
    if not paths["roi_center"]:
        logger.error(f"ROI center file not found for subject {subject_id}")
        return None
    
    # Search for coil file (only in standard coil directories)
    coil_paths_to_check = [
        os.path.join(data_dir, "coil", coil_file),
        os.path.join(data_path, "coil", coil_file),
        os.path.join(data_dir, "data", "coil", coil_file)
    ]
    
    # Check each path
    for coil_path in coil_paths_to_check:
        if os.path.exists(coil_path):
            paths["coil"] = coil_path
            logger.log(logging.INFO, f"Found coil file: {paths['coil']}")
            break
    
    # If specific file not found, check for any .ccd files in coil directories
    if not paths["coil"]:
        coil_dirs_to_check = [
            os.path.join(data_dir, "coil"),
            os.path.join(data_path, "coil"),
            os.path.join(data_dir, "data", "coil")
        ]
        
        for coil_dir in coil_dirs_to_check:
            if os.path.exists(coil_dir):
                ccd_files = [f for f in os.listdir(coil_dir) if f.endswith(".ccd")]
                if ccd_files:
                    paths["coil"] = os.path.join(coil_dir, ccd_files[0])
                    logger.log(logging.INFO, f"Using first available coil file: {paths['coil']}")
                    break
    
    if not paths["coil"]:
        logger.error(f"Coil file '{coil_file}' not found in any location")
        logger.error("Checked: " + ", ".join(coil_paths_to_check))
        return None
    
    # Test if output directory is writable
    try:
        os.makedirs(paths["output_dir"], exist_ok=True)
        test_file = os.path.join(paths["output_dir"], ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except (PermissionError, OSError) as e:
        logger.error(f"Cannot write to output directory {paths['output_dir']}: {str(e)}")
        logger.error("Please specify a writable output directory using --output-dir")
        return None
    
    return paths
# Add the CustomSimulationRunner class outside of main
from tms_efield_prediction.simulation.runner import SimulationRunnerConfig, SimulationRunner
from tms_efield_prediction.simulation.tms_simulation import SimulationState
from tms_efield_prediction.simulation.tms_simulation import get_skin_average_normal_vector
from tms_efield_prediction.simulation import tms_simulation

class CustomSimulationRunner(SimulationRunner):
    """Custom runner that uses separate meshes for positioning and E-field simulation."""
    
    def __init__(self, context, config, verified_mesh_path, verified_roi_mesh_path, verified_roi_path, debug_hook=None, resource_monitor=None):
        """
        Initialize the custom simulation runner.
        
        Args:
            context: Simulation context
            config: Runner configuration
            verified_mesh_path: Path to the full head mesh for positioning
            verified_roi_mesh_path: Path to the ROI mesh for E-field simulation
            verified_roi_path: Path to the ROI center file
            debug_hook: Optional debug hook
            resource_monitor: Optional resource monitor
        """
        super().__init__(context, config, debug_hook, resource_monitor)
        self.verified_mesh_path = verified_mesh_path  # Full mesh for positioning
        self.verified_roi_mesh_path = verified_roi_mesh_path  # ROI mesh for simulation
        self.verified_roi_path = verified_roi_path
    
    def compute_skin_normal_vector(self, msh, roi_center):
        """Calculate a reliable skin normal vector when the standard method fails.
        
        Args:
            msh: SimNIBS mesh
            roi_center: Dictionary with ROI center points
            
        Returns:
            Normalized skin normal vector
        """
        logger.log(logging.INFO,"Computing fallback skin normal vector")
        
        try:
            # Method 1: Use the vector from roi_center directly if available
            if 'skin_vec' in roi_center and roi_center['skin_vec'] is not None:
                skin_vec = roi_center['skin_vec']
                norm = np.linalg.norm(skin_vec)
                if norm > 1e-6:  # Ensure it's not a zero vector
                    logger.log(logging.INFO, "Using skin_vec from ROI center")
                    return skin_vec / norm
            
            # Method 2: Use vector from GM center to skin point
            if 'gm' in roi_center and 'skin' in roi_center:
                skin_vec = roi_center['skin'] - roi_center['gm']
                norm = np.linalg.norm(skin_vec)
                if norm > 1e-6:
                    logger.log(logging.INFO, "Using vector from GM center to skin point")
                    return skin_vec / norm
            
            # Method 3: Extract top 5% of skin nodes by z-coordinate and compute average normal
            try:
                skin_region_id = 1005  # Standard tag for skin in SimNIBS
                skin_cells = msh.crop_mesh(tags=skin_region_id)
                
                # Get skin triangle centers and normals
                skin_centers = skin_cells.elements_baricenters()[:]
                skin_normals = skin_cells.triangle_normals()[:]
                
                # Sort by z-coordinate (usually pointing up in head models)
                z_sorted_indices = np.argsort(skin_centers[:, 2])
                top_indices = z_sorted_indices[-int(len(z_sorted_indices) * 0.05):]  # Top 5%
                
                # Average the normals from the top region
                top_normals = skin_normals[top_indices]
                avg_normal = np.mean(top_normals, axis=0)
                norm = np.linalg.norm(avg_normal)
                
                if norm > 1e-6:
                    logger.log(logging.INFO, "Using average normal from top skin elements")
                    return avg_normal / norm
            except Exception as skin_error:
                logger.warning(f"Skin extraction failed: {str(skin_error)}")
            
            # Method 4: Last resort - use fixed normal pointing upward
            logger.warning("Using default upward normal vector as last resort")
            return np.array([0, 0, 1])
            
        except Exception as e:
            logger.error(f"Error computing skin normal vector: {str(e)}")
            logger.warning("Using default upward normal vector due to error")
            return np.array([0, 0, 1])

    def verify_mesh_has_skin(self, msh, throw_error=False):
        """
        Verify that a mesh has skin elements (tags 5 or 1005).
        
        Args:
            msh: SimNIBS mesh
            throw_error: Whether to throw an error if skin is not found
            
        Returns:
            bool: True if mesh has skin elements, False otherwise
        """
        try:
            # Try to get the surface elements
            msh_surf = msh.crop_mesh(elm_type=2)
            
            # Check for skin elements
            skin_tags = [5, 1005]
            for tag in skin_tags:
                try:
                    skin_mesh = msh_surf.crop_mesh(tags=tag)
                    if hasattr(skin_mesh.elm, 'triangles') and len(skin_mesh.elm.triangles) > 0:
                        logger.log(logging.INFO, f"Found skin elements with tag {tag}")
                        return True
                except Exception as e:
                    logger.warning(f"Error checking for skin tag {tag}: {str(e)}")
            
            # If we get here, no skin elements were found
            if throw_error:
                raise ValueError("The mesh does not contain skin elements (tags 5 or 1005). Please use a full head mesh for coil positioning.")
            else:
                logger.warning("The mesh does not contain skin elements (tags 5 or 1005). Coil positioning may be inaccurate.")
                return False
        
        except Exception as e:
            logger.error(f"Error verifying skin elements: {str(e)}")
            if throw_error:
                raise
            return False

    # In file: run_tms_simulation.py
# Replace calc_dAdt_wrapper with this version

    # In file: run_tms_simulation.py
# Replace calc_dAdt_wrapper with this simpler version

    def calc_dAdt_wrapper(self, context, mesh, matsimnibs, roi_center, skin_normal_avg,
                        n_cpus=1, to_hdf5=False, save_path=None, roi_radius=20.0):
        """Wrapper around calc_dAdt that uses direct prototype-style implementations."""
        from tms_efield_prediction.simulation import tms_simulation
        
        # Install compatible function wrappers if needed
        self.setup_tms_simulation_wrappers()
        
        try:
            # Call the main calculation function with all positions
            dadt_data = tms_simulation.calc_dAdt(
                context=context,
                mesh=mesh,
                matsimnibs=matsimnibs,  # Pass all positions
                roi_center=roi_center,
                skin_normal_avg=skin_normal_avg,
                debug_hook=self.debug_hook,
                resource_monitor=self.resource_monitor,
                n_cpus=n_cpus,
                to_hdf5=to_hdf5,
                save_path=save_path,
                roi_radius=roi_radius
            )
            
            return dadt_data
        finally:
            # Restore original functions
            self.restore_tms_simulation_functions()
        # Create direct implementations matching prototype
    def compute_cylindrical_roi(self, mesh, roi_center_gm, skin_normal_avg, roi_radius=20.0):
        """Create a ROI mask that includes both target region and skin."""
        import numpy as np
        
        # Create a conical ROI that gets wider toward the skin
        top = roi_center_gm + (skin_normal_avg * 40)  # Extend further to include skin
        base = roi_center_gm - (skin_normal_avg * 30)
        
        e = base - top
        m = np.cross(top, base)
        
        # Initialize the mask array
        nodes = mesh.nodes[:]
        
        # Compute distances from axis
        cross_e_rP = np.cross(e, nodes - top)
        d = np.linalg.norm(cross_e_rP, axis=1) / np.linalg.norm(e)
        
        # Calculate projection onto axis to determine position
        proj = np.dot(nodes - top, e) / np.dot(e, e)
        
        # Variable radius based on position (wider at top/skin)
        variable_radius = roi_radius * (1.0 + 2.0 * proj)  # Wider toward skin
        
        # Create conical selection (wider at skin)
        conical_roi = (d <= np.maximum(roi_radius, variable_radius))
        
        # Additionally, explicitly include skin elements
        try:
            skin_tags = [5, 1005]  # Standard tags for skin in SimNIBS
            for tag in skin_tags:
                skin_mask = mesh.elm.tag1 == tag
                if np.any(skin_mask):
                    # Add nearby skin elements
                    skin_nodes = np.unique(mesh.elm.node_number_list[skin_mask])
                    # Convert to 0-based indexing
                    skin_nodes = skin_nodes - 1
                    # Add skin nodes near ROI center
                    skin_dist = np.linalg.norm(nodes[skin_nodes] - roi_center_gm, axis=1)
                    nearby_skin = skin_dist < roi_radius * 3  # Use larger radius for skin
                    conical_roi[skin_nodes[nearby_skin]] = True
        except Exception as e:
            logger.warning(f"Could not add skin elements to ROI: {str(e)}")
        
        return conical_roi
    def crop_mesh_nodes(self, mesh, nodes_bool):
        """Crop mesh to keep only elements with all nodes in the selection."""
        import numpy as np
        
        # Get node numbers to keep (add 1 for SimNIBS 1-based indexing)
        node_keep_indexes = np.append(np.where(nodes_bool)[0] + 1, -1)
        
        # Find elements where all nodes are in the selection
        element_mask = np.zeros(len(mesh.elm), dtype=bool)
        for i, elm in enumerate(mesh.elm):
            if np.all(np.isin(elm.node_number_list, node_keep_indexes)):
                element_mask[i] = True
        
        # Get element IDs to keep (adding 1 for SimNIBS 1-based indexing)
        elements_to_keep = np.where(element_mask)[0] + 1
        
        # Crop mesh
        return mesh.crop_mesh(elements=elements_to_keep)

    def remove_islands(self, cropped, roi_center):
        """Remove disconnected components from the mesh."""
        import numpy as np
        
        # Find closest element to ROI center
        _, center_id = cropped.find_closest_element(roi_center['gm'], return_index=True)
        
        # Get connected components
        comps = cropped.elm.connected_components()
        
        # Find the component containing the center point
        valid_comps = [c for c in comps if np.isin(center_id, c)]
        
        if not valid_comps:
            logger.warning("ROI center not found in any connected component")
            # Return the largest component as fallback
            sizes = [len(comp) for comp in comps]
            valid_comps = [comps[np.argmax(sizes)]]
        
        # Crop mesh to keep only the valid component
        return cropped.crop_mesh(elements=valid_comps[0])

    def split_positions_into_batches(self, matsimnibs, n_batches=None, batch_index=None, max_positions=None):
        """
        Split the positions into batches for distributed processing.
        
        Args:
            matsimnibs: Array of transformation matrices
            n_batches: Number of batches to split into (None for no splitting)
            batch_index: Index of batch to process (None to process all)
            max_positions: Maximum number of positions to process (None for all)
            
        Returns:
            Tuple of (matsimnibs_subset, indices)
        """
        # Apply position limit if specified
        if max_positions is not None and max_positions > 0 and max_positions < len(matsimnibs):
            matsimnibs = matsimnibs[:max_positions]
            logger.log(logging.INFO, f"Limited to first {max_positions} positions")
        
        # Get total position count
        n_positions = len(matsimnibs)
        
        # If no batching, return all positions
        if n_batches is None or batch_index is None:
            return matsimnibs, list(range(n_positions))
        
        # Determine batch size and indices
        batch_size = int(np.ceil(n_positions / n_batches))
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, n_positions)
        
        # Check if batch is valid
        if start_idx >= n_positions:
            logger.error(f"Batch index {batch_index} is out of range (n_positions={n_positions}, n_batches={n_batches})")
            return np.array([]), []
        
        # Get the subset of positions for this batch
        indices = list(range(start_idx, end_idx))
        matsimnibs_subset = matsimnibs[indices]
        
        logger.log(logging.INFO, f"Processing batch {batch_index}/{n_batches-1}: positions {start_idx}-{end_idx-1} of {n_positions}")
        
        return matsimnibs_subset, indices
    
    def manual_coil_positioning(self, roi_center, config):
        """
        Implement manual coil positioning similar to the prototype when skin elements aren't available.
        This approach uses the ROI center and the skin normal vector directly, without needing skin mesh elements.
        
        Args:
            roi_center: Dictionary with ROI center points (must have 'skin' and 'skin_vec' keys)
            config: Positioning configuration
            
        Returns:
            Tuple of (matsimnibs, grid)
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        logger.log(logging.INFO, "Using manual coil positioning method (from prototype)")
        
        # Extract configuration values
        search_radius = config.search_radius
        spatial_resolution = config.spatial_resolution
        rotation_angles = config.rotation_angles
        distance = config.distance
        
        # Default coordinate system
        gridx = np.array([1, 0, 0])
        gridz = np.array([0, 0, 1])
        headz = np.array([0, 0, 1])
        
        # Get skin normal vector for orientation
        skin_vec = roi_center['skin_vec']
        skin_point = roi_center['skin']
        
        # Create a rotation matrix to align grid with skin normal
        # First, compute the rotation angle between z-axis and skin normal
        theta = np.arccos(np.dot(gridz, skin_vec))
        
        # Get the rotation axis (cross product of gridz and skin normal)
        axis_rot = np.cross(gridz, skin_vec)
        
        # Normalize the rotation axis
        axis_rot = axis_rot / np.linalg.norm(axis_rot)
        
        # Create rotation matrix
        rot_matrix = R.from_rotvec(theta * axis_rot).as_matrix()
        
        # Apply rotation to grid x-axis
        gridx_new = rot_matrix @ gridx
        
        # Get the y-axis by cross product of skin normal and rotated x-axis
        gridy_new = np.cross(skin_vec, gridx_new)
        
        # Generate grid points around skin point
        n = np.ceil(search_radius / spatial_resolution)
        x_vals = np.arange(-n*spatial_resolution, n*spatial_resolution + spatial_resolution, spatial_resolution)
        y_vals = np.arange(-n*spatial_resolution, n*spatial_resolution + spatial_resolution, spatial_resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Calculate points in 3D space
        points = skin_point + X.reshape(-1, 1) * gridx_new + Y.reshape(-1, 1) * gridy_new
        grid_2d = np.stack((X.flatten(), Y.flatten())).T
        
        # Keep only points within search radius
        keep = np.sqrt(X.flatten()**2 + Y.flatten()**2) <= search_radius
        points = points[keep]
        grid_2d = grid_2d[keep]
        
        # Prepare the matsimnibs
        n_points = len(points)
        n_angles = len(rotation_angles)
        
        # Create transformation matrices
        matsimnibs = np.zeros((n_angles, n_points, 4, 4), dtype=float)
        matsimnibs[:, :, 3, 3] = 1
        
        # Get the directions for the coil (normal vectors)
        z_vectors = np.tile(-skin_vec, (n_points, 1))
        
        # Adjust positions based on distance
        centers = points - distance * z_vectors
        
        for a, rot_angle_deg in enumerate(rotation_angles):
            # Convert to radians
            angle_rad = np.deg2rad(rot_angle_deg)
            
            # Create rotation vectors
            rotation_vectors = angle_rad * z_vectors
            
            # Create rotation matrices
            rot_matrices = R.from_rotvec(rotation_vectors).as_matrix().transpose(0, 2, 1)
            
            # Project headz onto the orthogonal complement of z_vectors
            dot_product = np.einsum('ij,j->i', z_vectors, headz)
            norm_squared = np.einsum('ij,ij->i', z_vectors, z_vectors)
            projection = (dot_product / norm_squared)[:, np.newaxis] * z_vectors
            headz_projected = headz - projection
            
            # Rotate headz_projected using rotation matrices
            y_vectors = np.einsum('ijk,ik->ij', rot_matrices, headz_projected)
            y_vectors /= np.linalg.norm(y_vectors, axis=1)[:, None]
            
            # Determine x-vectors using cross product
            x_vectors = np.cross(y_vectors, z_vectors)
            
            # Populate matsimnibs
            matsimnibs[a, :, :3, 0] = x_vectors
            matsimnibs[a, :, :3, 1] = y_vectors
            matsimnibs[a, :, :3, 2] = z_vectors
            matsimnibs[a, :, :3, 3] = centers
        
        # Reshape to match expected output format
        matsimnibs = matsimnibs.reshape(-1, 4, 4)
        
        # Create the grid entries matching the format expected by downstream code
        grid = np.stack([[np.array([a, *s]) for s in grid_2d] for a in rotation_angles]).reshape(-1, 3)
        
        logger.log(logging.INFO, f"Generated {matsimnibs.shape[0]} positions using manual method")
        
        return matsimnibs, grid


    def generate_coil_positions(self, msh, roi_center):
        """Override to use full mesh for coil positioning with fallback to manual positioning."""
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("CustomSimulationRunner.generate_coil_positions", "start")
        
        try:
            # Try to find a better mesh for positioning - first check if we need to load the full head mesh
            use_manual_positioning = False
            full_mesh = None
            
            try:
                # Check if current mesh has skin elements
                msh_surf = msh.crop_mesh(elm_type=2)  # Get surface elements
                skin_tags = [5, 1005]  # Standard tags for skin in SimNIBS
                
                # Check if mesh has skin elements
                has_skin = False
                for tag in skin_tags:
                    try:
                        skin_elms = msh_surf.crop_mesh(tags=tag)
                        if hasattr(skin_elms.elm, 'triangles') and len(skin_elms.elm.triangles) > 0:
                            has_skin = True
                            logger.log(logging.INFO, f"Found skin elements with tag {tag} in current mesh")
                            break
                    except Exception as e:
                        logger.warning(f"Error checking skin elements with tag {tag}: {str(e)}")
                
                if not has_skin:
                    logger.warning("Current mesh doesn't have skin elements. Looking for full head mesh...")
                    
                    # Try to load the full head mesh from standard location
                    from simnibs import mesh_io
                    
                    # Look for the head mesh in common locations
                    subject_id = self.context.subject_id
                    data_root = self.context.data_root_path
                    
                    # Remove "experiment" from path if present to get to subject root
                    subject_root = data_root
                    if "experiment" in subject_root:
                        subject_root = os.path.dirname(os.path.dirname(subject_root))
                    
                    # Standard locations for head mesh
                    head_mesh_paths = [
                        os.path.join(subject_root, "headmodel", f"sub-{subject_id}.msh"),
                        os.path.join(subject_root, "headmodel", f"{subject_id}.msh"),
                        os.path.join(os.path.dirname(subject_root), "headmodel", f"sub-{subject_id}.msh"),
                        os.path.join(os.path.dirname(subject_root), "headmodel", f"{subject_id}.msh")
                    ]
                    
                    # Try each path
                    head_mesh_found = False
                    for head_path in head_mesh_paths:
                        if os.path.exists(head_path):
                            logger.log(logging.INFO, f"Found full head mesh: {head_path}")
                            try:
                                full_mesh = mesh_io.read_msh(head_path)
                                head_mesh_found = True
                                
                                # Check if this mesh has skin elements
                                try:
                                    full_surf = full_mesh.crop_mesh(elm_type=2)
                                    for tag in skin_tags:
                                        try:
                                            skin_elms = full_surf.crop_mesh(tags=tag)
                                            if hasattr(skin_elms.elm, 'triangles') and len(skin_elms.elm.triangles) > 0:
                                                logger.log(logging.INFO, f"Found skin elements in full head mesh with tag {tag}")
                                                has_skin = True
                                                break
                                        except Exception as e:
                                            logger.warning(f"Error checking skin elements in full head mesh with tag {tag}: {str(e)}")
                                    
                                    if not has_skin:
                                        logger.warning("Full head mesh also doesn't have skin elements")
                                        use_manual_positioning = True
                                except Exception as e:
                                    logger.warning(f"Error checking skin elements in full head mesh: {str(e)}")
                                    use_manual_positioning = True
                                
                                break
                            except Exception as e:
                                logger.warning(f"Error loading head mesh {head_path}: {str(e)}")
                    
                    if not head_mesh_found:
                        logger.warning("Could not find full head mesh. Will use manual coil positioning.")
                        use_manual_positioning = True
                
            except Exception as e:
                logger.warning(f"Error checking for skin elements: {str(e)}")
                use_manual_positioning = True
            
            # If we need to use manual positioning
            if use_manual_positioning:
                logger.log(logging.INFO, "Falling back to manual coil positioning")
                
                # Check if roi_center has the necessary keys
                if not ('skin' in roi_center and 'skin_vec' in roi_center):
                    logger.warning("ROI center missing required keys for manual positioning")
                    
                    # Try to create the necessary values if missing
                    if 'gm' in roi_center:
                        if 'skin' not in roi_center:
                            # Estimate skin point by moving from GM along skin_vec
                            if 'skin_vec' in roi_center:
                                roi_center['skin'] = roi_center['gm'] + roi_center['skin_vec'] * 10.0
                                logger.log(logging.INFO, f"Created skin point from GM and skin_vec: {roi_center['skin']}")
                            else:
                                # Create a default upward-pointing vector
                                roi_center['skin'] = roi_center['gm'] + np.array([0, 0, 10.0])
                                logger.log(logging.INFO, f"Created skin point from GM with upward offset: {roi_center['skin']}")
                        
                        if 'skin_vec' not in roi_center:
                            if 'skin' in roi_center:
                                # Create skin_vec from GM to skin
                                vec = roi_center['skin'] - roi_center['gm']
                                roi_center['skin_vec'] = vec / np.linalg.norm(vec)
                                logger.log(logging.INFO, f"Created skin_vec from GM to skin: {roi_center['skin_vec']}")
                            else:
                                # Create a default upward-pointing vector
                                roi_center['skin_vec'] = np.array([0, 0, 1])
                                logger.log(logging.INFO, f"Created default upward skin_vec: {roi_center['skin_vec']}")
                
                # Use manual positioning
                matsimnibs, grid = self.manual_coil_positioning(roi_center, self.config.coil_config)
                
            else:
                # Use the coil position generator with the appropriate mesh
                from tms_efield_prediction.simulation.coil_position import CoilPositionGenerator
                
                # Create positioning configuration
                from tms_efield_prediction.simulation.coil_position import CoilPositioningConfig
                config = CoilPositioningConfig(
                    search_radius=self.config.coil_config.search_radius,
                    spatial_resolution=self.config.coil_config.spatial_resolution,
                    distance=self.config.coil_config.distance,
                    rotation_angles=self.config.coil_config.rotation_angles
                )
                
                # Generate positions using the appropriate mesh
                generator = CoilPositionGenerator(
                    self.context,
                    config,
                    debug_hook=self.debug_hook,
                    resource_monitor=self.resource_monitor
                )
                
                # Use full_mesh if we loaded it, otherwise use the original mesh
                mesh_to_use = full_mesh if full_mesh is not None else msh
                logger.log(logging.INFO, f"Using standard positioning with mesh: {mesh_to_use.fn}")
                
                # Generate positions 
                matsimnibs, grid = generator.generate_positions(mesh_to_use, roi_center)
            
            # Store information for later use
            self.n_positions = len(matsimnibs)
            self.matsimnibs = matsimnibs
            self.grid = grid
            self.roi_center = roi_center
            self.skin_normal_avg = roi_center['skin_vec']
            
            # Setup output directories
            self.efield_sims_dir = os.path.join(self.config.output_dir, "efield_sims")
            self.dadt_sims_dir = os.path.join(self.config.output_dir, "dadt_sims")
            os.makedirs(self.efield_sims_dir, exist_ok=True)
            os.makedirs(self.dadt_sims_dir, exist_ok=True)
            
            return matsimnibs, grid
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "CustomSimulationRunner.generate_coil_positions",
                    "subject_id": self.context.subject_id
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("CustomSimulationRunner.generate_coil_positions", "end")
    
    def load_data(self, paths):
        """Override to use verified paths."""
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("CustomSimulationRunner.load_data", "start")
        
        # Transition state
        self.state = self.state.transition_to("mesh_loading")
        
        try:
            # Load mesh directly with verified path
            from simnibs import mesh_io
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("loading_mesh", {"path": self.verified_mesh_path})
            
            msh = mesh_io.read_msh(self.verified_mesh_path)
            
            # Load ROI center - try scipy.io methods
            import numpy as np
            roi_center = None
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("loading_roi_center", {"path": self.verified_roi_path})
            
            try:
                # Use ROI center loading code
                import scipy.io as sio
                logger.log(logging.INFO,f"Loading ROI center file with scipy.io.loadmat: {self.verified_roi_path}")
                mat_data = sio.loadmat(self.verified_roi_path)
                
                # Debug: print the structure of the loaded MAT file
                logger.log(logging.INFO,f"MAT file contents - keys: {list(mat_data.keys())}")
                
                # We can see that roi_center contains a nested structure with arrays
                if 'roi_center' in mat_data:
                    # Extract the nested data properly
                    rc_data = mat_data['roi_center']
                    logger.log(logging.INFO,f"Found 'roi_center' key with shape: {rc_data.shape}")
                                
                    # Based on the output, the data is nested in rc_data[0,0]
                    # And it contains multiple arrays with the coordinates we need
                    if rc_data.shape == (1, 1) and isinstance(rc_data[0,0], np.ndarray):
                        # The content appears to be a tuple/list of arrays
                        array_list = rc_data[0,0]
                        
                        # Now we need to extract each array and access its content
                        if len(array_list) >= 3:  # Make sure we have at least 3 arrays
                            # Let's extract gm, skin and skin_vec from the first 3 arrays
                            # Each array is itself nested with shape (1,3)
                            gm = np.array(array_list[0]).flatten()
                            skin = np.array(array_list[1]).flatten()
                            skin_vec = np.array(array_list[2]).flatten()
                            
                            roi_center = {
                                'gm': gm,
                                'skin': skin,
                                'skin_vec': skin_vec
                            }
                            logger.log(logging.INFO,f"Successfully extracted ROI center from nested arrays: gm={gm}, skin={skin}, skin_vec={skin_vec}")
                        else:
                            logger.warning(f"Nested array list has insufficient elements: {len(array_list)}, need at least 3")
                    else:
                        logger.warning(f"roi_center data structure doesn't match expected format: {rc_data.shape}")
                
                # If we still couldn't extract the data, try one more approach
                if roi_center is None:
                    logger.warning("Attempting to extract ROI data using a different approach")
                    for key in mat_data.keys():
                        if key not in ['__header__', '__version__', '__globals__']:
                            try:
                                # This is to handle different possible formats
                                rc_data = mat_data[key]
                                
                                # If the data is a struct or cell array with nested contents
                                if hasattr(rc_data, 'item') and callable(getattr(rc_data, 'item')):
                                    nested_data = rc_data.item()
                                    if isinstance(nested_data, tuple) and len(nested_data) >= 3:
                                        # Try to extract from nested tuple
                                        gm = np.array(nested_data[0]).flatten()
                                        skin = np.array(nested_data[1]).flatten()
                                        skin_vec = np.array(nested_data[2]).flatten()
                                        
                                        roi_center = {
                                            'gm': gm,
                                            'skin': skin,
                                            'skin_vec': skin_vec
                                        }
                                        logger.log(logging.INFO,f"Extracted ROI center using item() approach: gm={gm}, skin={skin}, skin_vec={skin_vec}")
                                        break
                            except Exception as e:
                                logger.warning(f"Failed alternative extraction for key {key}: {str(e)}")
                        
            except Exception as mat_error:
                logger.error(f"Failed to load with scipy.io: {str(mat_error)}")
                
                # Create fallback ROI center by using mesh geometry
                try:
                    logger.warning(f"Creating ROI center from mesh geometry as fallback")
                    
                    # Get the center of the mesh as GM estimate
                    # Correctly extract node coordinates from SimNIBS mesh
                    all_nodes = []
                    for node in msh.nodes:
                        all_nodes.append(node.coordinates)
                    
                    node_coords = np.array(all_nodes)
                    
                    if node_coords.size == 0:
                        raise ValueError("Mesh contains no nodes")
                    
                    # Make sure we have a 2D array with shape (n_nodes, 3)
                    if len(node_coords.shape) == 1:
                        if node_coords.size % 3 == 0:
                            node_coords = node_coords.reshape(-1, 3)
                        else:
                            raise ValueError(f"Node coordinates array has unexpected size: {node_coords.size}")
                    
                    # Now we should have a proper 2D array
                    center = np.mean(node_coords, axis=0)
                    
                    # Find a reasonable skin point (near the top of the head)
                    # Sort by z-coordinate (typically height in head models)
                    z_indices = np.argsort(node_coords[:, 2])
                    top_indices = z_indices[-100:] if len(z_indices) >= 100 else z_indices
                    top_nodes = node_coords[top_indices]
                    skin_point = np.mean(top_nodes, axis=0)
                    
                    # Create a reasonable normal vector
                    normal = skin_point - center
                    normal = normal / np.linalg.norm(normal)
                    
                    roi_center = {
                        'gm': center,
                        'skin': skin_point,
                        'skin_vec': normal
                    }
                    
                    logger.log(logging.INFO,f"Created fallback ROI center: gm={center}, skin={skin_point}, normal={normal}")
                except Exception as fallback_error:
                    if self.debug_hook:
                        self.debug_hook.record_error(fallback_error, {
                            "component": "CustomSimulationRunner.load_data.fallback",
                            "roi_path": self.verified_roi_path
                        })
                    raise IOError(f"Failed to create fallback ROI center: {str(fallback_error)}. Original error: {str(mat_error)}")
            
            if roi_center is None:
                raise ValueError("Could not extract ROI center data from file or create fallback")

            # Get normal vector
            logger.log(logging.INFO,"Using ROI center's skin_vec as skin normal average (ROI mesh doesn't have skin elements)")
            skin_normal_avg = roi_center['skin_vec']  # Use dictionary access with brackets
            
            # Store in state
            self.state.mesh_data = {
                'mesh': msh,
                'roi_center': roi_center,
                'skin_normal_avg': skin_normal_avg
            }
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "load_data_complete", 
                    {
                        "subject_id": self.context.subject_id,
                        "node_count": len(msh.nodes) if hasattr(msh.nodes, "__len__") else "unknown"
                    }
                )
            
            return msh, roi_center, skin_normal_avg
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "CustomSimulationRunner.load_data",
                    "subject_id": self.context.subject_id
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("CustomSimulationRunner.load_data", "end")
    

    # In file: run_tms_simulation.py
    # Add this method to the CustomSimulationRunner class
    # After the calc_dAdt_wrapper method

    def setup_tms_simulation_wrappers(self):
        """Setup wrappers for tms_simulation functions to ensure compatibility."""
        import types
        from tms_efield_prediction.simulation import tms_simulation
        
        # Store original functions for later restoration
        self._original_functions = {
            'compute_cylindrical_roi': getattr(tms_simulation, 'compute_cylindrical_roi', None),
            'crop_mesh_nodes': getattr(tms_simulation, 'crop_mesh_nodes', None)
        }
        
        # Replace with our compatible versions
        def modified_compute_cylindrical_roi(mesh, roi_center_gm, skin_normal_avg, roi_radius=20.0, debug_hook=None, resource_monitor=None):
            return self.compute_cylindrical_roi(mesh, roi_center_gm, skin_normal_avg, roi_radius)
        
        def modified_crop_mesh_nodes(mesh, nodes_bool, debug_hook=None, resource_monitor=None):
            return self.crop_mesh_nodes(mesh, nodes_bool)
        
        # Install the modified functions
        tms_simulation.compute_cylindrical_roi = modified_compute_cylindrical_roi
        tms_simulation.crop_mesh_nodes = modified_crop_mesh_nodes
        
        logger.log(logging.INFO, "Installed compatible function wrappers for tms_simulation module")

    def restore_tms_simulation_functions(self):
        """Restore original tms_simulation functions."""
        from tms_efield_prediction.simulation import tms_simulation
        
        # Restore original functions if they existed
        if hasattr(self, '_original_functions'):
            for func_name, original_func in self._original_functions.items():
                if original_func is not None:
                    setattr(tms_simulation, func_name, original_func)
                else:
                    # If there was no original, delete our temporary version
                    try:
                        delattr(tms_simulation, func_name)
                    except AttributeError:
                        pass
            
            logger.log(logging.INFO, "Restored original tms_simulation module functions")


    def run(self):
        """Run the complete simulation pipeline with batch processing."""
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("CustomSimulationRunner.run", "start")
        
        try:
            # Prepare paths
            paths = self.prepare_paths()
            
            # Create output directories
            efield_sims_dir = os.path.join(self.config.output_dir, "efield_sims")
            dadt_sims_dir = os.path.join(self.config.output_dir, "dadt_sims")
            os.makedirs(efield_sims_dir, exist_ok=True)
            os.makedirs(dadt_sims_dir, exist_ok=True)
            
            # Create tmp directory
            tmp_dir = os.path.join(self.context.output_path, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Load data using our customized method
            msh, roi_center, skin_normal_avg = self.load_data(paths)
            
            # Verify that the full mesh has skin elements (for positioning)
            self.verify_mesh_has_skin(msh)
            
            # Generate coil positions (using the full mesh)
            matsimnibs, grid = self.generate_coil_positions(msh, roi_center)
            
            # Apply batch processing parameters if specified
            n_batches = getattr(self.config, 'n_batches', None)
            batch_index = getattr(self.config, 'batch_index', None)
            max_positions = getattr(self.config, 'max_positions', None)
            
            # Save all positions to disk on the first batch (or if no batching)
            if (n_batches is None or batch_index is None or batch_index == 0):
                self._save_positions(matsimnibs, grid)
            
            # Split positions into batches if needed
            if n_batches is not None and batch_index is not None:
                batch_matsimnibs, batch_indices = self.split_positions_into_batches(
                    matsimnibs, n_batches, batch_index, max_positions
                )
            else:
                # Apply max_positions limit if specified
                if max_positions is not None and max_positions > 0 and max_positions < len(matsimnibs):
                    batch_matsimnibs = matsimnibs[:max_positions]
                    batch_indices = list(range(max_positions))
                else:
                    batch_matsimnibs = matsimnibs
                    batch_indices = list(range(len(matsimnibs)))
            
            # Determine positions to process
            n_positions = len(batch_matsimnibs)
            if n_positions == 0:
                logger.log(logging.WARNING, "No positions to process in this batch")
                return {
                    "all_efields_file": None,
                    "efield_directory": efield_sims_dir,
                    "dadt_directory": dadt_sims_dir,
                    "positions_processed": 0,
                    "batch_index": batch_index,
                    "n_batches": n_batches
                }
            
            # Determine number of concurrent simulations based on available CPUs
            available_cpus = self.config.n_cpus if self.config.n_cpus > 0 else os.cpu_count() or 4
            max_concurrent = min(32, available_cpus)  # Cap at 32 concurrent processes
            logger.log(logging.INFO, f"Processing {n_positions} positions with up to {max_concurrent} concurrent simulations")

            # Calculate and save dA/dt for all positions
            logger.log(logging.INFO, f"Calculating dA/dt for {len(batch_matsimnibs)} positions")
            try:
                # Load the ROI mesh for more efficient dA/dt calculation
                from simnibs import mesh_io
                roi_msh = mesh_io.read_msh(self.verified_roi_mesh_path)
                
                # Use the tms_simulation.calc_dAdt function 
                from tms_efield_prediction.simulation.tms_simulation import calc_dAdt
                
                # Save dA/dt to HDF5 file
                dadt_file = os.path.join(dadt_sims_dir, f"{self.context.subject_id}_dAdts.h5")
                
                # Calculate dA/dt for batch positions
                dadt_data = self.calc_dAdt_wrapper(
                    context=self.context,
                    mesh=roi_msh,
                    matsimnibs=batch_matsimnibs,  # FIXED: Use all batch positions
                    roi_center=roi_center,
                    skin_normal_avg=skin_normal_avg,
                    n_cpus=self.config.n_cpus,
                    to_hdf5=True,
                    save_path=dadt_sims_dir,
                    roi_radius=20.0
                )
                
                logger.log(logging.INFO, f"Successfully calculated and saved dA/dt fields to {dadt_file}")
                
                # Also save individual .npy files for each position if needed
                if dadt_data is not None:
                    logger.log(logging.INFO, f"dA/dt data shape: {dadt_data.shape}")
                    for idx, global_idx in enumerate(batch_indices):
                        pos_dadt_file = os.path.join(dadt_sims_dir, f"{self.context.subject_id}_position_{global_idx}_dadt.npy")
                        np.save(pos_dadt_file, dadt_data[idx])
                        # Verify file was saved
                        if os.path.exists(pos_dadt_file):
                            logger.log(logging.INFO, f"Verified dA/dt file saved: {pos_dadt_file}")
                    logger.log(logging.INFO, f"Saved individual dA/dt files for each position")
            except Exception as e:
                logger.error(f"Error calculating dA/dt fields: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                logger.warning("Continuing with E-field simulation without dA/dt calculation")
                
            # Initialize results storage
            all_efields = []
            
            # Process positions in batches to manage resources
            batch_size = max_concurrent
            for batch_start in range(0, n_positions, batch_size):
                batch_end = min(batch_start + batch_size, n_positions)
                batch_positions = list(range(batch_start, batch_end))
                global_indices = [batch_indices[i] for i in batch_positions]
                
                logger.log(logging.INFO, f"Processing sub-batch {batch_start//batch_size + 1}/{(n_positions+batch_size-1)//batch_size}: " +
                                        f"positions {batch_start} to {batch_end-1} (global indices: {min(global_indices)}-{max(global_indices)})")
                
                # Process each position in this batch
                batch_efields = []
                for batch_pos, global_idx in zip(batch_positions, global_indices):
                    # Create subdirectory for this position
                    out_subdir = f"batch_{batch_index or 0}_pos_{global_idx}"
                    position_dir = os.path.join(tmp_dir, out_subdir)
                    os.makedirs(position_dir, exist_ok=True)
                    
                    try:
                        # Get the matrix for this position
                        matsim = batch_matsimnibs[batch_pos]
                        
                        # Run simulation (using ROI mesh)
                        logger.log(logging.INFO, f"Running simulation for position {global_idx} (batch position {batch_pos})")
                        sim_path = self._run_efield_sim_custom(
                            matsim,
                            out_subdir=out_subdir
                        )
                        original_mesh_dir = os.path.join(efield_sims_dir, "original_meshes")
                        os.makedirs(original_mesh_dir, exist_ok=True)
                        mesh_filename = os.path.basename(sim_path)
                        mesh_copy_path = os.path.join(original_mesh_dir, f"{self.context.subject_id}_position_{global_idx}_{mesh_filename}")
                        try:
                            shutil.copy2(sim_path, mesh_copy_path)
                            logger.log(logging.INFO, f"Saved original E-field mesh to {mesh_copy_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save original mesh: {str(e)}")
                        # Extract E-field
                        logger.log(logging.INFO, f"Extracting E-field for position {global_idx}")
                        efield = self.extract_save_efield(
                            sim_path=sim_path,
                            roi_center=roi_center,
                            skin_normal_avg=skin_normal_avg,
                            clean_path=True,
                            save_final_mesh=(global_idx == 0),  # Only save for first position
                            save_path_mesh=os.path.join(self.config.output_dir, f"{self.context.subject_id}_roi.msh")
                        )
                        
                        # Save individual position result
                        pos_file = os.path.join(efield_sims_dir, f"{self.context.subject_id}_position_{global_idx}.npy")
                        np.save(pos_file, efield)
                        logger.log(logging.INFO, f"Saved position {global_idx} E-field to {pos_file}")
                        
                        # Store E-field with original index
                        batch_efields.append((global_idx, efield))
                    except Exception as e:
                        logger.error(f"Error processing position {global_idx}: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                    finally:
                        # Always clean up the position's temp directory
                        if os.path.exists(position_dir):
                            try:
                                shutil.rmtree(position_dir)
                                logger.log(logging.INFO, f"Cleaned up position directory: {position_dir}")
                            except Exception as e:
                                logger.warning(f"Failed to clean up directory {position_dir}: {str(e)}")
                
                # Sort batch results by original index
                batch_efields.sort(key=lambda x: x[0])
                
                # Extract just the E-fields
                sorted_efields = [e for _, e in batch_efields]
                
                # Add batch results to all results
                all_efields.extend(sorted_efields)
                
                # Save progress after each batch
                if all_efields:
                    progress_file = os.path.join(
                        self.config.output_dir,
                        f"{self.context.subject_id}_efields_progress_batch{batch_index or 0}.npy"
                    )
                    np.save(progress_file, np.stack(all_efields))
                    logger.log(logging.INFO, f"Saved progress with {len(all_efields)}/{n_positions} positions complete")
            
            # Save final batch results
            logger.log(logging.INFO, f"Saving final results for batch {batch_index or 0} with {len(all_efields)} positions")
            if all_efields:
                batch_efields_file = os.path.join(
                    self.config.output_dir, 
                    f"{self.context.subject_id}_batch{batch_index or 0}_efields.npy"
                )
                np.save(batch_efields_file, np.stack(all_efields))
            else:
                batch_efields_file = None
            
            # If not in batch mode, or if this is the last batch, also save combined results
            combined_results = None
            if n_batches is None or batch_index is None or batch_index == n_batches - 1:
                if n_batches is not None and n_batches > 1:
                    # Try to combine results from all batches
                    combined_efields = []
                    
                    # Try to load all batch results
                    for i in range(n_batches):
                        batch_file = os.path.join(
                            self.config.output_dir, 
                            f"{self.context.subject_id}_batch{i}_efields.npy"
                        )
                        
                        if os.path.exists(batch_file):
                            try:
                                batch_data = np.load(batch_file)
                                combined_efields.append(batch_data)
                                logger.log(logging.INFO, f"Loaded batch {i} results: {batch_data.shape}")
                            except Exception as e:
                                logger.error(f"Error loading batch {i} results: {str(e)}")
                    
                    # Stack all results if we have any
                    if combined_efields:
                        try:
                            all_results = np.concatenate(combined_efields, axis=0)
                            combined_file = os.path.join(
                                self.config.output_dir, 
                                f"{self.context.subject_id}_all_efields.npy"
                            )
                            np.save(combined_file, all_results)
                            logger.log(logging.INFO, f"Saved combined results with {all_results.shape[0]} positions")
                            combined_results = combined_file
                        except Exception as e:
                            logger.error(f"Error combining results: {str(e)}")
                elif all_efields:
                    # Single batch mode - save all_efields as the final result
                    combined_file = os.path.join(
                        self.config.output_dir, 
                        f"{self.context.subject_id}_all_efields.npy"
                    )
                    np.save(combined_file, np.stack(all_efields))
                    logger.log(logging.INFO, f"Saved combined results with {len(all_efields)} positions")
                    combined_results = combined_file
            
            # Create a summary JSON file
            summary = {
                "subject_id": self.context.subject_id,
                "positions_processed": len(all_efields),
                "total_positions": len(matsimnibs) if n_batches is None else None,
                "batch_index": batch_index,
                "n_batches": n_batches,
                "global_indices": batch_indices,
                "efield_directory": efield_sims_dir,
                "dadt_directory": dadt_sims_dir,
                "batch_result_file": batch_efields_file,
                "combined_result_file": combined_results,
                "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            summary_file = os.path.join(
                self.config.output_dir, 
                f"{self.context.subject_id}_batch{batch_index or 0}_summary.json"
            )

            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            # Combine all dA/dt files into a single HDF5 file
            try:
                import h5py
                import glob
                
                # Find all individual dA/dt files
                dadt_files = glob.glob(os.path.join(dadt_sims_dir, f"{self.context.subject_id}_position_*_dadt.npy"))
                
                if dadt_files:
                    logger.log(logging.INFO, f"Combining {len(dadt_files)} dA/dt files into a single HDF5 file")
                    
                    # Create HDF5 file
                    h5_path = os.path.join(dadt_sims_dir, f"{self.context.subject_id}_dAdts.h5")
                    
                    with h5py.File(h5_path, 'w') as f:
                        # Load first file to get shape
                        first_dadt = np.load(dadt_files[0])
                        
                        # Create dataset with compression
                        dadt_shape = (len(dadt_files), *first_dadt.shape)
                        dadt_dataset = f.create_dataset(
                            'dAdt', 
                            shape=dadt_shape,
                            dtype=np.float32,  # Use single precision
                            compression='gzip',
                            compression_opts=4
                        )
                        
                        # Sort files by position index
                        def get_pos_idx(filename):
                            return int(os.path.basename(filename).split('_position_')[1].split('_')[0])
                        
                        sorted_files = sorted(dadt_files, key=get_pos_idx)
                        
                        # Fill dataset
                        for i, file_path in enumerate(sorted_files):
                            dadt_dataset[i] = np.load(file_path)
                        
                        logger.log(logging.INFO, f"Successfully saved combined dA/dt data to {h5_path}")
                        
            except Exception as e:
                logger.warning(f"Failed to combine dA/dt files: {str(e)}")

            # Clean up temporary directory if requested
            if getattr(self.config, 'clean_temp', True) and os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                    logger.log(logging.INFO, f"Cleaned up temporary directory: {tmp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory {tmp_dir}: {str(e)}")
            
            return summary
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "CustomSimulationRunner.run",
                    "subject_id": self.context.subject_id
                })
            logger.error(f"Error in simulation run: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("CustomSimulationRunner.run", "end")


    def _create_simnibs_symlink(self, subject_id, sim_path):
        """Create a temporary symlink for SimNIBS expected directory structure.
        
        Args:
            subject_id: Subject ID
            sim_path: Path to simulation output directory
            
        Returns:
            Tuple of (symlink_path, created) where created is True if symlink was created
            
        Notes:
            SimNIBS expects a specific directory structure with m2m_{subject_id} naming.
            This function creates a temporary symlink to meet this expectation.
        """
        try:
            from pathlib import Path
            
            # SimNIBS expected pattern
            expected_dir_name = f"m2m_{subject_id}"
            
            # Check if directory already exists
            simnibs_path = os.path.join(os.path.dirname(sim_path), expected_dir_name)
            if os.path.exists(simnibs_path):
                logger.log(logging.INFO, f"SimNIBS directory already exists: {simnibs_path}")
                return simnibs_path, False
            
            # Create the symlink from expected path to our subject directory
            source_path = os.path.join(self.context.data_root_path, "headmodel")
            
            # Handle relative paths
            if not os.path.isabs(source_path):
                source_path = os.path.abspath(source_path)
            
            # Create parent directory if needed
            os.makedirs(os.path.dirname(simnibs_path), exist_ok=True)
            
            # Create the symlink
            os.symlink(source_path, simnibs_path, target_is_directory=True)
            logger.log(logging.INFO, f"Created SimNIBS directory symlink: {simnibs_path} -> {source_path}")
            
            return simnibs_path, True
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "CustomSimulationRunner._create_simnibs_symlink",
                    "subject_id": subject_id,
                    "sim_path": sim_path
                })
            logger.warning(f"Failed to create SimNIBS symlink: {str(e)}")
            return None, False
    # In file: run_tms_simulation.py
# Replace _run_efield_sim_custom with this modified version
    # In file: run_tms_simulation.py
# After the _run_efield_sim_custom function definition
# Before the extract_save_efield function

    def _create_simnibs_structure(self, subject_id, sim_path):
        """Create proper SimNIBS m2m directory structure to enable postprocessing.
        
        Args:
            subject_id: Subject ID
            sim_path: Path to simulation output directory
            
        Returns:
            Tuple of (created_dir_path, success_flag)
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("CustomSimulationRunner._create_simnibs_structure", "start")
        
        try:

            import shutil
            
            # SimNIBS expected directory name
            m2m_dirname = f"m2m_{subject_id}"
            
            # Determine parent directory for the m2m folder
            parent_dir = os.path.dirname(sim_path)
            m2m_dir = os.path.join(parent_dir, m2m_dirname)
            
            # Check if directory already exists
            if os.path.exists(m2m_dir):
                logger.log(logging.INFO, f"SimNIBS m2m directory already exists: {m2m_dir}")
                return m2m_dir, True
            
            # Create the directory
            os.makedirs(m2m_dir, exist_ok=True)
            
            # Create necessary subdirectories for SimNIBS
            os.makedirs(os.path.join(m2m_dir, "mask_prep"), exist_ok=True)
            os.makedirs(os.path.join(m2m_dir, "eeg_positions"), exist_ok=True)
            os.makedirs(os.path.join(m2m_dir, "bem_files"), exist_ok=True)
            
            # Create symbolic link to mesh file in m2m directory
            mesh_link = os.path.join(m2m_dir, f"{m2m_dirname}.msh")
            if not os.path.exists(mesh_link):
                try:
                    # Use relative path for symlink to make it more portable
                    mesh_rel_path = os.path.relpath(self.verified_roi_mesh_path, m2m_dir)
                    os.symlink(mesh_rel_path, mesh_link)
                    logger.log(logging.INFO, f"Created mesh symlink: {mesh_link} -> {mesh_rel_path}")
                except Exception as link_err:
                    # If symlink fails (e.g., on Windows or without permissions), copy the file
                    logger.warning(f"Symlink creation failed, copying mesh instead: {str(link_err)}")
                    shutil.copy2(self.verified_roi_mesh_path, mesh_link)
                    logger.log(logging.INFO, f"Copied mesh to: {mesh_link}")
            
            # Create a minimal m2m_sub.mat file that SimNIBS looks for
            try:
                import scipy.io as sio
                mat_path = os.path.join(m2m_dir, f"{m2m_dirname}.mat")
                if not os.path.exists(mat_path):
                    # Create minimal data structure
                    mat_data = {
                        'subpath': m2m_dir,
                        'subject_id': subject_id,
                    }
                    sio.savemat(mat_path, mat_data)
                    logger.log(logging.INFO, f"Created minimal m2m data file: {mat_path}")
            except Exception as mat_err:
                logger.warning(f"Could not create m2m .mat file: {str(mat_err)}")
            
            logger.log(logging.INFO, f"Created SimNIBS m2m directory structure: {m2m_dir}")
            return m2m_dir, True
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "CustomSimulationRunner._create_simnibs_structure",
                    "subject_id": subject_id,
                    "sim_path": sim_path
                })
            logger.warning(f"Failed to create SimNIBS directory structure: {str(e)}")
            return None, False
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("CustomSimulationRunner._create_simnibs_structure", "end")
    
    
    
# In file: run_tms_simulation.py
# Replace _run_efield_sim_custom with this modified version


    # In file: run_tms_simulation.py
# Add these helper methods before extract_save_efield function

    def _extract_efield_from_nodedata(self, msh):
        """Extract E-field from mesh nodedata attribute."""
        if hasattr(msh, 'nodedata') and msh.nodedata and len(msh.nodedata) > 0:
            efield = msh.nodedata[0][:]
            logger.log(logging.INFO, f"Extracted E-field from nodedata[0] with shape {efield.shape}")
            return efield
        return None

    def _extract_efield_from_field(self, msh):
        """Extract E-field from mesh field attribute."""
        if hasattr(msh, 'field') and msh.field:
            for i, field in enumerate(msh.field):
                if hasattr(field, 'name') and field.name.lower() in ['e', 'e_norm', 'normee', 'normemag', 'efield']:
                    efield = field.value
                    logger.log(logging.INFO, f"Extracted E-field from field[{i}].value with name {field.name}")
                    return efield
                elif isinstance(field, dict) and 'E' in field:
                    efield = field['E']
                    logger.log(logging.INFO, f"Extracted E-field from field[{i}]['E']")
                    return efield
        return None

    def _extract_efield_from_elmdata(self, msh):
        """Extract E-field from mesh elmdata attribute."""
        if hasattr(msh, 'elmdata') and msh.elmdata and len(msh.elmdata) > 0:
            efield = msh.elmdata[0][:]
            logger.log(logging.INFO, f"Extracted E-field from elmdata[0] with shape {efield.shape}")
            return efield
        return None

    def _extract_efield_from_mesh_attributes(self, msh):
        """Extract E-field from other mesh attributes."""
        # Try calling direct mesh accessors
        if hasattr(msh, 'elm_data'):
            efield = msh.elm_data()
            if efield is not None and len(efield) > 0:
                logger.log(logging.INFO, f"Extracted E-field from elm_data() with shape {efield[0].shape}")
                return efield[0]
                
        # Try to access node fields directly
        if hasattr(msh, 'node_fields') and msh.node_fields:
            field_key = list(msh.node_fields.keys())[0]
            if hasattr(msh, 'field') and field_key in msh.field:
                efield = msh.field[field_key]
                logger.log(logging.INFO, f"Extracted E-field from node field '{field_key}' with shape {efield.shape}")
                return efield
        
        return None
    # In file: run_tms_simulation.py
# Replace direct_efield_calculation with this fixed version

    def direct_efield_calculation(self, matsimnibs, out_subdir=""):
        """Calculate E-field directly and save as a mesh file."""
        import numpy as np
        import os
        from simnibs.simulation import coil_numpy as coil_lib
        
        logger.log(logging.INFO, "Using direct E-field calculation")
        
        try:
            # Create output directory
            output_dir = os.path.join(self.context.output_path, "tmp", out_subdir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Load mesh
            from simnibs import mesh_io
            msh = mesh_io.read_msh(self.verified_roi_mesh_path)
            
            # Create object-like access for roi_center
            class AttrDict:
                def __init__(self, d): self.__dict__.update(d)
            roi_center = AttrDict(self.roi_center)
            
            # Prototype functions
            def compute_cylindrical_roi(mesh, roi_center_gm, skin_normal_avg, roi_radius):
                print("======DATA======")
                print("roi_center_gm: ", roi_center_gm)
                print("skin_normal_avg: ", skin_normal_avg)
                print("roi_radius: ", roi_radius)
                top = roi_center_gm + (skin_normal_avg * 10)
                base = roi_center_gm - (skin_normal_avg * 30)
                e = base - top
                m = np.cross(top, base)
                nodes = mesh.nodes[:]
                cross_e_rP = np.cross(e, nodes - top)
                d = np.linalg.norm(cross_e_rP, axis=1) / np.linalg.norm(e)
                return d <= roi_radius
            
            def crop_mesh_nodes(mesh, nodes_bool):
                node_keep_indexes = np.append(np.where(nodes_bool)[0] + 1, -1)
                elements_to_keep = np.where(np.all(
                    np.isin(mesh.elm.node_number_list, node_keep_indexes).reshape(-1, 4),
                    axis=1))[0]
                return mesh.crop_mesh(elements=elements_to_keep+1)
            
            def remove_islands(cropped, roi_center):
                _, center_id = cropped.find_closest_element(roi_center.gm, return_index=True)
                comps = cropped.elm.connected_components()
                valid_elms = [c for c in comps if np.isin(center_id, c)][0]
                return cropped.crop_mesh(elements=valid_elms)
            
            # Process mesh for ROI
            cylindrical_roi = compute_cylindrical_roi(msh, roi_center.gm, roi_center.skin_vec, roi_radius=20)
            cropped = crop_mesh_nodes(msh, cylindrical_roi)
            final_roi = remove_islands(cropped, roi_center)
            positions = final_roi.nodes[:]
            
            # Calculate dA/dt directly with FMM
            logger.log(logging.INFO, f"Calculating dA/dt with FMM for {positions.shape[0]} nodes")
            
            # Read coil data
            d_position, d_moment = coil_lib.read_ccd(self.context.coil_file_path)
            d_position = np.hstack([d_position * 1e3, np.ones((d_position.shape[0], 1))])
            
            # Transform and rotate
            d_pos = matsimnibs.dot(d_position.T).T[:, :3] * 1e-3
            d_mom = np.dot(d_moment, matsimnibs[:3, :3].T)
            
            # Calculate with FMM
            import fmm3dpy
            out = fmm3dpy.lfmm3d(
                eps=1e-3,
                sources=d_pos.T,
                charges=d_mom.T,
                targets=positions.T*1e-3,
                pgt=2,
                nd=d_mom.shape[-1]
            )
            
            # Compute dA/dt
            dadt = np.zeros((positions.shape[0], 3))
            dadt[:, 0] = (out.gradtarg[1,2] - out.gradtarg[2,1])
            dadt[:, 1] = (out.gradtarg[2,0] - out.gradtarg[0,2])
            dadt[:, 2] = (out.gradtarg[0,1] - out.gradtarg[1,0])
            dadt *= -1e-7 * 1.49e6
            
            # Calculate E-field
            efield = -dadt
            
            # Create a NodeData object with the E-field
            node_data = mesh_io.NodeData(name='E', value=efield)
            
            # Set the nodedata of the final_roi
            final_roi.nodedata = [node_data]
            
            # Save as mesh file
            result_path = os.path.join(output_dir, "direct_efield_result.msh")
            final_roi.write(result_path)
            
            logger.log(logging.INFO, f"Saved E-field mesh to {result_path}")
            
            return result_path
            
        except Exception as e:
            logger.error(f"Direct E-field calculation failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback: return a path to an empty file that signals failure
            fallback_path = os.path.join(output_dir, "direct_calculation_failed.txt")
            with open(fallback_path, 'w') as f:
                f.write(f"Direct calculation failed: {str(e)}")
            
            return fallback_path

    def _run_efield_sim_custom(self, matsimnibs, out_subdir="", n_cpus=1):
        """Run E-field simulation with SimNIBS, using scalar conductivity for stability."""
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("CustomSimulationRunner._run_efield_sim_custom", "start")
        
        try:
            from simnibs import run_simnibs, sim_struct
            import os
            
            # Create a unique output directory
            output_dir = os.path.join(self.context.output_path, "tmp", out_subdir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create SimNIBS session
            s = sim_struct.SESSION()
            s.fnamehead = self.verified_roi_mesh_path
            s.pathfem = output_dir
            s.map_to_surf = True
            s.fields = 'eE'
            s.open_in_gmsh = False
            
            # Create TMS list
            tms_list = s.add_tmslist()
            tms_list.fnamecoil = self.context.coil_file_path
            tms_list.anisotropy_type = 'scalar'  # Always use scalar for reliability
            tms_list.name = 'sim'
            
            # Add coil position
            pos = tms_list.add_position()
            pos.matsimnibs = matsimnibs
            pos.didt = 1.49e6
            
            # Run simulation
            try:
                run_simnibs(s, cpus=n_cpus)
            except Exception as e:
                # If it's the disconnected nodes error, use our direct calculation
                if "Found a column of zeros in the stiffness matrix" in str(e):
                    logger.warning(f"SimNIBS calculation failed: {str(e)}")
                    logger.log(logging.INFO, "Falling back to direct E-field calculation")
                    return self.direct_efield_calculation(matsimnibs, out_subdir)
                else:
                    raise
            
            # Find simulation result file
            coil_name = os.path.splitext(os.path.basename(tms_list.fnamecoil))[0]
            sim_path = os.path.join(s.pathfem, "subject_overlays", f"sim-0001_{coil_name}_scalar_central.msh")
            
            if not os.path.exists(sim_path):
                # Try alternative path formats
                import glob
                patterns = [
                    os.path.join(s.pathfem, "subject_overlays", "*.msh"),
                    os.path.join(s.pathfem, "*.msh")
                ]
                
                for pattern in patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        sim_path = matches[0]
                        break
            
            if not os.path.exists(sim_path):
                logger.warning("Could not find SimNIBS output, falling back to direct calculation")
                return self.direct_efield_calculation(matsimnibs, out_subdir)
            
            logger.log(logging.INFO, f"Found SimNIBS output: {sim_path}")
            return sim_path
            
        except Exception as e:
            logger.error(f"Error in _run_efield_sim_custom: {str(e)}")
            logger.log(logging.INFO, "Falling back to direct calculation")
            return self.direct_efield_calculation(matsimnibs, out_subdir)
            
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("CustomSimulationRunner._run_efield_sim_custom", "end")

    def compute_cylindrical_roi(self, mesh, roi_center_gm, skin_normal_avg, roi_radius=20.0):
        """Create a ROI mask that includes both target region and skin."""
        import numpy as np
        
        # Create a conical ROI that gets wider toward the skin
        top = roi_center_gm + (skin_normal_avg * 40)  # Extend further to include skin
        base = roi_center_gm - (skin_normal_avg * 30)
        
        e = base - top
        m = np.cross(top, base)
        
        # Initialize the mask array
        nodes = mesh.nodes[:]
        
        # Compute distances from axis
        cross_e_rP = np.cross(e, nodes - top)
        d = np.linalg.norm(cross_e_rP, axis=1) / np.linalg.norm(e)
        
        # Calculate projection onto axis to determine position
        proj = np.dot(nodes - top, e) / np.dot(e, e)
        
        # Variable radius based on position (wider at top/skin)
        variable_radius = roi_radius * (1.0 + 2.0 * proj)  # Wider toward skin
        
        # Create conical selection (wider at skin)
        conical_roi = (d <= np.maximum(roi_radius, variable_radius))
        
        # Additionally, explicitly include skin elements
        try:
            skin_tags = [5, 1005]  # Standard tags for skin in SimNIBS
            for tag in skin_tags:
                skin_mask = mesh.elm.tag1 == tag
                if np.any(skin_mask):
                    # Add nearby skin elements
                    skin_nodes = np.unique(mesh.elm.node_number_list[skin_mask])
                    # Convert to 0-based indexing
                    skin_nodes = skin_nodes - 1
                    # Add skin nodes near ROI center
                    skin_dist = np.linalg.norm(nodes[skin_nodes] - roi_center_gm, axis=1)
                    nearby_skin = skin_dist < roi_radius * 3  # Use larger radius for skin
                    conical_roi[skin_nodes[nearby_skin]] = True
        except Exception as e:
            logger.warning(f"Could not add skin elements to ROI: {str(e)}")
        
        return conical_roi


    def crop_mesh_nodes(self, mesh, nodes_bool):
        """Crop mesh to keep only elements with all nodes in the selection."""
        import numpy as np
        
        # Get node numbers to keep (add 1 for SimNIBS 1-based indexing)
        node_keep_indexes = np.append(np.where(nodes_bool)[0] + 1, -1)
        
        # Find elements where all nodes are in the selection
        element_mask = np.zeros(len(mesh.elm), dtype=bool)
        for i, elm in enumerate(mesh.elm):
            if np.all(np.isin(elm.node_number_list, node_keep_indexes)):
                element_mask[i] = True
        
        # Get element IDs to keep (adding 1 for SimNIBS 1-based indexing)
        elements_to_keep = np.where(element_mask)[0] + 1
        
        # Crop mesh
        return mesh.crop_mesh(elements=elements_to_keep)
# In file: run_tms_simulation.py
# Replace the extract_save_efield function with this one

    # In file: run_tms_simulation.py
# Replace the extract_save_efield function with this enhanced version

    # In file: run_tms_simulation.py
# Replace extract_save_efield with this improved version

    def extract_save_efield(self, sim_path, roi_center, skin_normal_avg, clean_path=False, save_final_mesh=False, save_path_mesh=None):
        """Extract E-field data following the prototype approach exactly."""
        try:
            from simnibs import mesh_io
            import numpy as np
            import os
            import shutil

            # Convert dictionary to object for prototype compatibility
            class AttrDict:
                def __init__(self, d): self.__dict__.update(d)
            roi_center = AttrDict(roi_center)
            
            # Load mesh
            logger.log(logging.INFO, f"Loading simulation mesh from {sim_path}")
            msh = mesh_io.read_msh(sim_path)
            
            # Process ROI - exact functions from prototype
            def compute_cylindrical_roi(mesh, roi_center_gm, skin_normal_avg, roi_radius):
                top = roi_center_gm + (skin_normal_avg * 10)
                base = roi_center_gm - (skin_normal_avg * 30)
                e = base - top
                m = np.cross(top, base)
                nodes = mesh.nodes[:]
                cross_e_rP = np.cross(e, nodes - top)
                d = np.linalg.norm(cross_e_rP, axis=1) / np.linalg.norm(e)
                return d <= roi_radius
            
            def crop_mesh_nodes(mesh, nodes_bool):
                node_keep_indexes = np.append(np.where(nodes_bool)[0] + 1, -1)
                elements_to_keep = np.where(np.all(
                    np.isin(mesh.elm.node_number_list, node_keep_indexes).reshape(-1, 4),
                    axis=1))[0]
                return mesh.crop_mesh(elements=elements_to_keep+1)
            
            def remove_islands(cropped, roi_center):
                _, center_id = cropped.find_closest_element(roi_center.gm, return_index=True)
                comps = cropped.elm.connected_components()
                valid_elms = [c for c in comps if np.isin(center_id, c)][0]
                return cropped.crop_mesh(elements=valid_elms)
            
            # Apply prototype's ROI processing
            cylindrical_roi = compute_cylindrical_roi(msh, roi_center.gm, skin_normal_avg, roi_radius=20)
            cropped = crop_mesh_nodes(msh, cylindrical_roi)
            final_roi = remove_islands(cropped, roi_center)
            
            # Save final mesh if requested
            if save_final_mesh and save_path_mesh is not None:
                final_roi.write(save_path_mesh)
                logger.log(logging.INFO, f"Saved ROI mesh to {save_path_mesh}")
            
            # Extract E-field from the processed mesh
            efield = final_roi.nodedata[0][:]
            
            # Clean up if requested
            if clean_path:
                parent_dir = os.path.abspath(os.path.join(os.path.dirname(sim_path), ".."))
                if os.path.exists(parent_dir):
                    shutil.rmtree(parent_dir)
                    logger.log(logging.INFO, f"Cleaned up directory: {parent_dir}")
            
            return efield
            
        except Exception as e:
            logger.error(f"Error in extract_save_efield: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return np.zeros((1000, 3))
                
    def _extract_efield_from_nodedata(self, msh):
        """Extract E-field from mesh nodedata attribute."""
        if hasattr(msh, 'nodedata') and msh.nodedata and len(msh.nodedata) > 0:
            efield = msh.nodedata[0][:]
            logger.log(logging.INFO, f"Extracted E-field from nodedata[0] with shape {efield.shape}")
            return efield
        return None
# In file: run_tms_simulation.py
# Replace _extract_efield_from_field with this fixed version

    def _extract_efield_from_field(self, msh):
        """Extract E-field from mesh field attribute."""
        if hasattr(msh, 'field') and msh.field:
            # Log the field structure without assuming its type
            logger.log(logging.INFO, f"Found {len(msh.field)} field entries")
            
            for i, field in enumerate(msh.field):
                try:
                    # Check if field is an object with name and value attributes
                    if hasattr(field, 'name') and hasattr(field, 'value'):
                        field_name = field.name
                        if field_name.lower() in ['e', 'e_norm', 'normee', 'normemag', 'efield']:
                            efield = field.value
                            logger.log(logging.INFO, f"Extracted E-field from field[{i}].value with name {field_name}")
                            return efield
                    
                    # Check if field is a dictionary with 'E' key
                    elif isinstance(field, dict) and 'E' in field:
                        efield = field['E']
                        logger.log(logging.INFO, f"Extracted E-field from field[{i}]['E']")
                        return efield
                    
                    # Handle string fields which might be references or names
                    elif isinstance(field, str) and field.lower() in ['e', 'efield', 'e_field']:
                        # Look for corresponding data in nodedata
                        if hasattr(msh, 'nodedata') and len(msh.nodedata) > i:
                            efield = msh.nodedata[i][:]
                            logger.log(logging.INFO, f"Extracted E-field from nodedata[{i}] based on field name '{field}'")
                            return efield
                except Exception as e:
                    logger.warning(f"Error processing field {i}: {str(e)}")
                    
        return None

    def _extract_efield_from_elmdata(self, msh):
        """Extract E-field from mesh elmdata attribute."""
        if hasattr(msh, 'elmdata') and msh.elmdata and len(msh.elmdata) > 0:
            efield = msh.elmdata[0][:]
            logger.log(logging.INFO, f"Extracted E-field from elmdata[0] with shape {efield.shape}")
            return efield
        return None

    def _extract_efield_from_mesh_attributes(self, msh):
        """Extract E-field from other mesh attributes."""
        # Try calling direct mesh accessors
        if hasattr(msh, 'elm_data'):
            efield = msh.elm_data()
            if efield is not None and len(efield) > 0:
                logger.log(logging.INFO, f"Extracted E-field from elm_data() with shape {efield[0].shape}")
                return efield[0]
                
        # Try to access node fields directly
        if hasattr(msh, 'node_fields') and msh.node_fields:
            field_key = list(msh.node_fields.keys())[0]
            if hasattr(msh, 'field') and field_key in msh.field:
                efield = msh.field[field_key]
                logger.log(logging.INFO, f"Extracted E-field from node field '{field_key}' with shape {efield.shape}")
                return efield
        
        return None
    
    def remove_islands(self, cropped, roi_center):
        """Remove disconnected components from the mesh."""
        import numpy as np
        
        # Find closest element to ROI center
        _, center_id = cropped.find_closest_element(roi_center['gm'], return_index=True)
        
        # Get connected components
        comps = cropped.elm.connected_components()
        
        # Find the component containing the center point
        valid_comps = []
        for comp in comps:
            if center_id in comp:
                valid_comps.append(comp)
        
        if not valid_comps:
            logger.warning("ROI center not found in any connected component")
            # Return the largest component as fallback
            sizes = [len(comp) for comp in comps]
            valid_comps = [comps[np.argmax(sizes)]]
        
        # Crop mesh to keep only the valid component
        return cropped.crop_mesh(elements=valid_comps[0])

    def _save_positions(self, matsimnibs, grid):
        """Save position matrices and grid data to files."""
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("CustomSimulationRunner._save_positions", "start")
        
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Save matrices
            matrices_file = os.path.join(self.config.output_dir, f"{self.context.subject_id}_matsimnibs.npy")
            np.save(matrices_file, matsimnibs)
            
            # Save grid
            grid_file = os.path.join(self.config.output_dir, f"{self.context.subject_id}_grid.npy")
            np.save(grid_file, grid)
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("save_positions", {
                    "matrices_file": matrices_file,
                    "grid_file": grid_file,
                    "matrices_shape": matsimnibs.shape,
                    "grid_shape": grid.shape
                })
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "CustomSimulationRunner._save_positions",
                    "subject_id": self.context.subject_id
                })
            logger.warning(f"Failed to save positions: {str(e)}")
            # Don't raise the exception - we can continue without saving positions
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("CustomSimulationRunner._save_positions", "end")
    # In file: run_tms_simulation.py
# Replace extract_save_efield with this fixed version

    def extract_save_efield(self, sim_path, roi_center, skin_normal_avg, clean_path=False, 
                        save_final_mesh=False, save_path_mesh=None):
        """Extract E-field data from simulation result with robust fallback mechanisms.
        
        Args:
            sim_path: Path to simulation result mesh
            roi_center: Dictionary with ROI center points
            skin_normal_avg: Average skin normal vector
            clean_path: Whether to clean up the simulation path
            save_final_mesh: Whether to save the final ROI mesh
            save_path_mesh: Path to save the final ROI mesh
            
        Returns:
            E-field array
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("CustomSimulationRunner.extract_save_efield", "start")
        
        try:
            from simnibs import mesh_io
            import numpy as np
            import shutil
            
            logger.log(logging.INFO, f"Loading result mesh from {sim_path}")
            msh = mesh_io.read_msh(sim_path)
            
            # Log mesh attributes without assuming structure
            logger.log(logging.INFO, "Checking mesh data structure")
            
            # Try multiple methods to extract E-field data
            efield = None
            extraction_methods = [
                self._extract_efield_from_nodedata,
                self._extract_efield_from_field,
                self._extract_efield_from_elmdata,
                self._extract_efield_from_mesh_attributes,
            ]
            
            for method in extraction_methods:
                try:
                    method_result = method(msh)
                    if method_result is not None:
                        efield = method_result
                        break
                except Exception as method_error:
                    logger.warning(f"E-field extraction method failed: {str(method_error)}")
            
            # If we still have no data, crop mesh to ROI then try extraction methods again
            if efield is None:
                logger.log(logging.INFO, "No E-field data found in full mesh, trying with ROI cropping")
                
                # Create cylindrical ROI exactly as in prototype
                cylindrical_roi = self.compute_cylindrical_roi(msh, roi_center['gm'], skin_normal_avg, roi_radius=20)
                
                # Crop mesh to ROI
                cropped = self.crop_mesh_nodes(msh, cylindrical_roi)
                
                # Remove islands
                final_roi = self.remove_islands(cropped, roi_center)
                
                # Save final mesh if requested
                if save_final_mesh and save_path_mesh is not None:
                    final_roi.write(save_path_mesh)
                    logger.log(logging.INFO, f"Saved ROI mesh to {save_path_mesh}")
                
                # Try extraction methods on cropped mesh
                for method in extraction_methods:
                    try:
                        method_result = method(final_roi)
                        if method_result is not None:
                            efield = method_result
                            break
                    except Exception as method_error:
                        logger.warning(f"E-field extraction method failed on ROI mesh: {str(method_error)}")
            
            # Last resort - create a dummy E-field with zeros
            if efield is None:
                logger.warning("Could not find E-field data. Creating zeros as placeholder.")
                nodes = getattr(msh, 'nodes', [])
                node_count = len(nodes) if hasattr(nodes, '__len__') else 1000
                efield = np.zeros((node_count, 3))
            
            # Clean up if requested
            if clean_path:
                try:
                    if os.path.isdir(sim_path):
                        shutil.rmtree(sim_path)
                    else:
                        parent_dir = os.path.dirname(sim_path)
                        if os.path.exists(parent_dir):
                            shutil.rmtree(parent_dir)
                    logger.log(logging.INFO, f"Cleaned up simulation path")
                except Exception as e:
                    logger.warning(f"Failed to clean up directory: {str(e)}")
            
            return efield
            
        except Exception as e:
            logger.error(f"Error extracting E-field: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("CustomSimulationRunner.extract_save_efield", "end")
    def _cleanup_position(self, batch_idx, position_idx, retries=3, delay=0.5):
        """Clean up temporary files for a position with retries."""
        position_tmp_dir = os.path.join(self.context.output_path, "tmp", f"batch_{batch_idx}_pos_{position_idx}")
        
        if os.path.exists(position_tmp_dir):
            for attempt in range(retries):
                try:
                    # Try to remove the directory
                    shutil.rmtree(position_tmp_dir)
                    logger.log(logging.INFO, f"Cleaned up position directory: {position_tmp_dir}")
                    return True
                except Exception as e:
                    if attempt < retries - 1:
                        # If not the last attempt, wait and retry
                        logger.warning(f"Failed to clean up directory {position_tmp_dir}, attempt {attempt+1}/{retries}: {str(e)}")
                        import time
                        time.sleep(delay)
                    else:
                        # If last attempt, log but don't raise
                        logger.warning(f"Failed to clean up directory {position_tmp_dir} after {retries} attempts: {str(e)}")
                        
                        # Try a more aggressive approach on the last attempt
                        try:
                            # Force close any open file handles in this directory
                            os.system(f"lsof +D {position_tmp_dir} | grep -v COMMAND | awk '{{print $2}}' | xargs -r kill -9")
                            
                            # Try again after killing processes
                            time.sleep(1.0)
                            shutil.rmtree(position_tmp_dir)
                            logger.log(logging.INFO, f"Cleaned up position directory after force closing: {position_tmp_dir}")
                            return True
                        except Exception as e2:
                            logger.warning(f"Failed aggressive cleanup of {position_tmp_dir}: {str(e2)}")
                            return False
        
        return True  # Directory didn't exist, so cleanup "succeeded"

    def _save_positions(self, matsimnibs, grid):
        """Save position matrices and grid data to files."""
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("CustomSimulationRunner._save_positions", "start")
        
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Save matrices
            matrices_file = os.path.join(self.config.output_dir, f"{self.context.subject_id}_matsimnibs.npy")
            np.save(matrices_file, matsimnibs)
            
            # Save grid
            grid_file = os.path.join(self.config.output_dir, f"{self.context.subject_id}_grid.npy")
            np.save(grid_file, grid)
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event("save_positions", {
                    "matrices_file": matrices_file,
                    "grid_file": grid_file,
                    "matrices_shape": matsimnibs.shape,
                    "grid_shape": grid.shape
                })
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "CustomSimulationRunner._save_positions",
                    "subject_id": self.context.subject_id
                })
            logger.warning(f"Failed to save positions: {str(e)}")
            # Don't raise the exception - we can continue without saving positions
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("CustomSimulationRunner._save_positions", "end")
def main():
    """Main function."""
    args = parse_args()
    
    # Verify data structure
    data_dir = os.path.abspath(args.data_dir)
    logger.log(logging.INFO,f"Using data directory: {data_dir}")
    
    # Use custom output directory if specified
    custom_output_dir = args.output_dir
    if custom_output_dir:
        # Ensure it's an absolute path
        custom_output_dir = os.path.abspath(custom_output_dir)
        logger.log(logging.INFO,f"Using custom output directory: {custom_output_dir}")
    
    paths = verify_data_structure(data_dir, args.subject, custom_output_dir, args.coil_file)
    if not paths:
        logger.error("Data verification failed. Cannot proceed.")
        return 1

    # Get data root path for ROI processing
    if os.path.exists(os.path.join(data_dir, "data", f"sub-{args.subject}")):
        data_root_path = os.path.join(data_dir, "data", f"sub-{args.subject}")
    else:
        data_root_path = os.path.join(data_dir, f"sub-{args.subject}")
    
    # Set up experiment directory
    exp_path = os.path.join(paths["output_dir"], args.exp_type)
    os.makedirs(exp_path, exist_ok=True)
    
    # IMPORTANT: We need both the full mesh (for positioning) and ROI mesh (for simulation)
    
    # First, ensure we have the full mesh
    if not paths["mesh"]:
        logger.error("Full mesh not found. Cannot proceed with coil positioning.")
        return 1
    logger.log(logging.INFO,f"Using full mesh for coil positioning: {paths['mesh']}")
    
    # Check if ROI mesh exists in paths dictionary from verify_data_structure
    if "mesh_roi" in paths and os.path.exists(paths["mesh_roi"]):
        # Use the existing ROI mesh from the verified paths
        roi_mesh_path = paths["mesh_roi"]
        logger.log(logging.INFO,f"Using existing ROI mesh: {roi_mesh_path}")
    else:
        # Define default location for new ROI mesh if we need to generate it
        roi_mesh_path = os.path.join(os.path.dirname(paths["mesh"]), f"sub-{args.subject}_middle_gray_matter_roi.msh")
        
        # Create context for the ROI processor
        roi_context = SimulationContext(
            dependencies={"simnibs": "3.6"},
            config={"roi_radius": 20.0},
            pipeline_mode="mri_efield",
            experiment_phase="preprocessing",
            debug_mode=True,
            subject_id=args.subject,
            data_root_path=data_root_path,
            output_path=exp_path
        )
        
        # Import and create ROI processor
        from tms_efield_prediction.data.pipeline.roi_processor import ROIProcessor
        roi_processor = ROIProcessor(roi_context)
        
        # First check if ROI mesh exists in data structure but wasn't found by verify_data_structure
        roi_exists, existing_path = roi_processor.check_roi_mesh_exists(args.subject, data_root_path)
        
        if roi_exists:
            # Use existing ROI mesh found by the processor
            roi_mesh_path = existing_path
            logger.log(logging.INFO,f"ROI processor found existing ROI mesh: {roi_mesh_path}")
        else:
            # No existing ROI mesh found - generate a new one
            logger.log(logging.INFO,f"No ROI mesh found. Generating new ROI mesh at: {roi_mesh_path}")
            
            try:
                # Generate ROI mesh
                result = roi_processor.generate_roi_mesh(
                    mesh_path=paths["mesh"],  # Use full mesh as the source
                    roi_center_path=paths["roi_center"],
                    output_mesh_path=roi_mesh_path,
                    roi_radius=20.0
                )
                
                if result.success:
                    logger.log(logging.INFO,f"ROI mesh generation successful: {result.roi_mesh_path}")
                    logger.log(logging.INFO,f"Node reduction: {result.node_reduction:.2f}%")
                    roi_mesh_path = result.roi_mesh_path
                else:
                    logger.error(f"ROI mesh generation failed: {result.error_message}")
                    logger.error("Proceeding with full brain mesh instead")
                    roi_mesh_path = paths["mesh"]  # Fallback to full mesh
                    
            except Exception as e:
                logger.error(f"Error generating ROI mesh: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                logger.error("Proceeding with full brain mesh instead")
                roi_mesh_path = paths["mesh"]  # Fallback to full mesh
    
    # Update paths dictionary with ROI mesh
    paths["mesh_roi"] = roi_mesh_path
    logger.log(logging.INFO,f"Using ROI mesh path: {paths['mesh_roi']}")
    
    # Validate that we have two separate meshes (or make user aware of fallback)
    if paths["mesh"] == paths["mesh_roi"]:
        logger.warning("Using full mesh for both positioning and simulation - this is not optimal for performance")
        logger.warning("Consider generating a separate ROI mesh for better performance")
    else:
        logger.log(logging.INFO, f"Using separate meshes for positioning and simulation:")
        logger.log(logging.INFO, f"  - Full mesh (positioning): {paths['mesh']}")
        logger.log(logging.INFO, f"  - ROI mesh (simulation): {paths['mesh_roi']}")

    # Check DTI tensor path for anisotropy
    tensor_path = os.path.join(data_root_path, "headmodel", 
                            f"d2c_sub-{args.subject}", 
                            "dti_results_T1space", 
                            "DTI_conf_tensor.nii.gz")
    
    simnibs_anisotropy = "scalar" if args.anisotropy in ["none", "scalar"] else args.anisotropy

    if args.anisotropy not in ["none", "scalar"]:
        if not tensor_path or not os.path.exists(tensor_path):
            logger.warning(f"Anisotropy type '{args.anisotropy}' was requested, but DTI tensor file not found.")
            logger.warning(f"Expected at: {tensor_path}")
            logger.warning("Falling back to isotropic conductivity ('scalar').")
            simnibs_anisotropy = "scalar"
            tensor_path = ''
        else:
            logger.log(logging.INFO, f"Using anisotropic conductivity ({args.anisotropy}) with tensor file: {tensor_path}")
    else:
        logger.log(logging.INFO, f"Using isotropic conductivity ('{args.anisotropy}', maps to 'scalar' in SimNIBS)")
        tensor_path = ''
    if tensor_path and simnibs_anisotropy != 'scalar':
        print_file_info(tensor_path)
    # If coil file wasn't found in the default locations, use the specified one
    if not paths.get("coil"):
        # Look for the coil file in the data directory or the project root
        coil_candidates = [
            os.path.join(data_dir, "coil", args.coil_file),
            os.path.join(project_root, "data", "coil", args.coil_file),
            args.coil_file
        ]
        
        for candidate in coil_candidates:
            if os.path.exists(candidate):
                paths["coil"] = candidate
                break
        
        if not paths.get("coil"):
            logger.error(f"Coil file not found: {args.coil_file}")
            logger.error(f"Please specify a valid coil file with --coil-file or place it in one of:")
            for dir_path in coil_candidates:
                logger.error(f"  - {os.path.dirname(dir_path)}")
            return 1
    
    logger.log(logging.INFO,f"Using coil file: {paths['coil']}")
    
    # Generate rotation angles
    rotation_angles = np.arange(0, 360, args.rotation_step)
    
    # Set up configuration
    coil_config = CoilPositioningConfig(
        search_radius=args.search_radius,
        spatial_resolution=args.spatial_res,
        distance=args.coil_distance,
        rotation_angles=rotation_angles
    )
    
    # Create field config without the unsupported parameter
    field_config = FieldCalculationConfig(
        didt=1.49e6,  # Standard dI/dt value
        use_fmm=True  # Use Fast Multipole Method if available
    )
    
    # Show batch processing information if specified
    if args.n_batches:
        if args.batch_index is not None:
            logger.log(logging.INFO, f"Running batch {args.batch_index+1}/{args.n_batches}")
        else:
            logger.log(logging.INFO, f"Processing will be divided into {args.n_batches} batches")
    
    if args.max_positions:
        logger.log(logging.INFO, f"Processing limited to {args.max_positions} positions")
    
    # Run simulation
    logger.log(logging.INFO,f"Starting simulation for subject {args.subject}")
    logger.log(logging.INFO,f"Using {args.n_cpus} CPU cores")
    logger.log(logging.INFO,f"Search radius: {args.search_radius} mm")
    logger.log(logging.INFO,f"Spatial resolution: {args.spatial_res} mm")
    logger.log(logging.INFO,f"Coil distance: {args.coil_distance} mm")
    logger.log(logging.INFO,f"Rotation step: {args.rotation_step} degrees")
    
    try:
        # Set up simulation context directly to have more control
        context = SimulationContext(
    dependencies={"simnibs": "3.6"},
    config={
        "search_radius": args.search_radius,
        "spatial_resolution": args.spatial_res,
        "distance": args.coil_distance,
        "rotation_step": args.rotation_step
    },
    pipeline_mode="mri_efield",
    experiment_phase="preprocessing",
    debug_mode=True,
    subject_id=args.subject,
    data_root_path=data_root_path,
    coil_file_path=paths["coil"],
    output_path=exp_path,
    tensor_nifti_path=tensor_path,
    anisotropy_type=simnibs_anisotropy  # Use the mapped value for SimNIBS
)
            
        # Create SimulationRunner configuration
        from tms_efield_prediction.simulation.runner import SimulationRunnerConfig
        
        config = SimulationRunnerConfig(
            workspace=data_dir,
            subject_id=args.subject,
            experiment_type=args.exp_type,
            output_dir=exp_path,
            n_cpus=args.n_cpus,
            n_batches=args.n_batches,
            batch_index=args.batch_index,
            max_positions=args.max_positions,
            clean_temp=args.clean_temp,
            coil_config=coil_config,
            field_config=field_config
        )
        
        # Create CustomSimulationRunner with separate meshes
        # IMPORTANT: Pass both mesh paths separately
        runner = CustomSimulationRunner(
            context=context,
            config=config,
            verified_mesh_path=paths["mesh"],  # Full mesh for positioning
            verified_roi_mesh_path=paths["mesh_roi"],  # ROI mesh for simulation
            verified_roi_path=paths["roi_center"],
            debug_hook=None,
            resource_monitor=None
        )
        
        # Run the simulation
        results = runner.run()
        
        # Save results summary
        results_file = os.path.join(exp_path, f"simulation_results_{args.subject}")
        if args.n_batches is not None and args.batch_index is not None:
            results_file += f"_batch{args.batch_index}"
        results_file += ".json"
        
        # Save results
        with open(results_file, 'w') as f:
            # Convert NumPy arrays to lists for JSON serialization
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            json.dump(results, f, cls=NumpyEncoder, indent=2)
        
        logger.log(logging.INFO,f"Simulation completed successfully")
        logger.log(logging.INFO,f"Results saved to {results_file}")
        
        return 0            
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())