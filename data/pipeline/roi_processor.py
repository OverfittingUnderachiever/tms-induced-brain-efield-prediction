# tms_efield_prediction/data/pipeline/roi_processor.py
"""
ROI mesh processing module.

This module provides functionality for generating a region of interest mesh
as a preprocessing step before TMS E-field simulations.
"""

import os
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from simnibs import mesh_io

# Project imports
from ...utils.state.context import PipelineContext
from ...utils.debug.hooks import DebugHook
from ...utils.resource.monitor import ResourceMonitor

# Configure logging
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger('roi_processor')

@dataclass
class ROIProcessingResult:
    """Result of ROI mesh processing."""
    success: bool
    roi_mesh_path: str
    execution_time: float
    node_reduction: float  # Percentage of nodes reduced from original mesh
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class ROIProcessor:
    """Processor for generating region of interest (ROI) meshes."""
    
    def __init__(
        self,
        context: PipelineContext,
        debug_hook: Optional[DebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """Initialize the ROI processor."""
        self.context = context
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Register with resource monitor if provided
        if resource_monitor:
            resource_monitor.register_component(
                "ROIProcessor",
                self._reduce_memory
            )
    
    def _reduce_memory(self, target_reduction: float) -> None:
        """Reduce memory usage when requested by resource monitor."""
        import gc
        gc.collect()
    
    def check_roi_mesh_exists(self, subject_id: str, data_root_path: str) -> Tuple[bool, Optional[str]]:
        """
        Check if an ROI mesh already exists for the given subject.
        
        Args:
            subject_id: Subject ID
            data_root_path: Root path to subject data
            
        Returns:
            tuple: (exists, mesh_path) - boolean if mesh exists and path to mesh if found
        """
        # Define potential paths for ROI mesh
        potential_locations = []
        
        # First priority: Check in experiment/all directory (new structure)
        experiment_all_dir = os.path.join(data_root_path, "experiment", "all")
        if os.path.exists(experiment_all_dir):
            roi_path = os.path.join(experiment_all_dir, f"sub-{subject_id}_middle_gray_matter_roi.msh")
            potential_locations.append(roi_path)
            logger.log(logging.INFO, f"Checking for ROI mesh in experiment/all: {roi_path}")
        
        # Second priority: Check in headmodel directory (old structure)
        headmodel_dir = os.path.join(data_root_path, "headmodel")
        if os.path.exists(headmodel_dir):
            roi_path = os.path.join(headmodel_dir, f"sub-{subject_id}_middle_gray_matter_roi.msh")
            potential_locations.append(roi_path)
            logger.log(logging.INFO, f"Checking for ROI mesh in headmodel: {roi_path}")
        
        # Third priority: Check in experiment directory (another possible location)
        experiment_dir = os.path.join(data_root_path, "experiment")
        if os.path.exists(experiment_dir):
            roi_path = os.path.join(experiment_dir, f"sub-{subject_id}_middle_gray_matter_roi.msh")
            potential_locations.append(roi_path)
            logger.log(logging.INFO, f"Checking for ROI mesh in experiment: {roi_path}")
        
        # Check each potential location
        for roi_path in potential_locations:
            if os.path.exists(roi_path):
                logger.log(logging.INFO, f"Found existing ROI mesh: {roi_path}")
                return True, roi_path
        
        logger.log(logging.INFO, f"No existing ROI mesh found for subject {subject_id}")
        return False, None
    
    def verify_input_files(self, mesh_path: str, roi_center_path: str) -> bool:
        """
        Verify that input files exist and are accessible.
        
        Args:
            mesh_path: Path to the mesh file
            roi_center_path: Path to the ROI center file
            
        Returns:
            bool: True if files exist and are accessible
        """
        # Check if mesh file exists and is accessible
        if not os.path.exists(mesh_path):
            logger.error(f"Mesh file does not exist: {mesh_path}")
            return False
            
        if not os.path.exists(roi_center_path):
            logger.error(f"ROI center file does not exist: {roi_center_path}")
            return False
        
        # Check if files are not empty
        if os.path.getsize(mesh_path) == 0:
            logger.error(f"Mesh file is empty: {mesh_path}")
            return False
            
        if os.path.getsize(roi_center_path) == 0:
            logger.error(f"ROI center file is empty: {roi_center_path}")
            return False
        
        logger.log(logging.INFO, f"Mesh file verified: {mesh_path} ({os.path.getsize(mesh_path)} bytes)")
        logger.log(logging.INFO, f"ROI center file verified: {roi_center_path} ({os.path.getsize(roi_center_path)} bytes)")
        return True
    
    def get_skin_average_normal_vector(self, mesh, roi_center, roi_radius):
        """Calculate average normal vector at skin within ROI."""
        # Extract skin region
        skin_region_id = 1005  # Assuming region_idx 1005 corresponds to skin
        skin_cells = mesh.crop_mesh(tags=skin_region_id)
        
        # Get skin triangle centers and normals
        skin_centers = skin_cells.elements_baricenters()[:]
        skin_normals = skin_cells.triangle_normals()[:]
        
        # Compute skin ROI
        roi_center_skin = roi_center.skin
        distances = np.linalg.norm(skin_centers - roi_center_skin, axis=1)
        skin_roi = distances < roi_radius
        
        # Average normal vector in the ROI
        skin_normal_avg = np.mean(skin_normals[skin_roi], axis=0)
        
        return skin_normal_avg
    
    def compute_cylindrical_roi(self, mesh, roi_center_gm, skin_normal_avg, roi_radius):
        """Compute a cylindrical ROI mask."""
        top = roi_center_gm + (skin_normal_avg * 10)
        base = roi_center_gm - (skin_normal_avg * 30)

        e = base - top
        m = np.cross(top, base)

        # Initialize the mask array
        nodes = mesh.nodes[:]

        # Compute distances and projections
        cross_e_rP = np.cross(e, nodes)
        d = np.linalg.norm(m + cross_e_rP, axis=1) / np.linalg.norm(e)

        # Compute rQ
        rQ = nodes + np.cross(e, m + cross_e_rP) / np.linalg.norm(e)**2

        # Compute weights wA and wB
        wA = np.linalg.norm(np.cross(rQ, base), axis=1) / np.linalg.norm(m)
        wB = np.linalg.norm(np.cross(rQ, top), axis=1) / np.linalg.norm(m)

        cylindrical_roi = (d <= roi_radius) & (wA >= 0) & (wA <= 1) & (wB >= 0) & (wB <= 1)
        return cylindrical_roi

    def crop_mesh_nodes(self, mesh, nodes_bool):
        """Crop mesh to keep only elements with all nodes in the selection."""
        node_keep_indexes = np.append(np.where(nodes_bool)[0] + 1, -1)
        elements_to_keep = np.where(np.all(np.isin(mesh.elm.node_number_list, node_keep_indexes).reshape(-1, 4),axis=1))[0]
        return mesh.crop_mesh(elements=elements_to_keep+1)

    def remove_islands(self, cropped, roi_center):
        """Remove disconnected components from the mesh."""
        _,center_id = cropped.find_closest_element(roi_center.gm, return_index=True)
        comps = cropped.elm.connected_components()
        valid_elms = [c for c in comps if np.isin(center_id, c)][0]
        return cropped.crop_mesh(elements=valid_elms)
    
    def generate_roi_mesh(self, 
                          mesh_path: str, 
                          roi_center_path: str,
                          output_mesh_path: str,
                          roi_radius: float = 20.0) -> ROIProcessingResult:
        """Generate a region of interest mesh for TMS simulations."""
        start_time = time.time()
        
        try:
            # First check if ROI mesh already exists
            roi_exists, existing_path = self.check_roi_mesh_exists(
                self.context.subject_id, 
                self.context.data_root_path
            )
            
            if roi_exists:
                # Use existing ROI mesh
                logger.log(logging.INFO, f"Using existing ROI mesh: {existing_path}")
                return ROIProcessingResult(
                    success=True,
                    roi_mesh_path=existing_path,
                    execution_time=time.time() - start_time,
                    node_reduction=0.0,  # No reduction to calculate as we're using existing mesh
                    metadata={"existing_mesh": True},
                    error_message=None
                )
            
            # Verify input files exist and are accessible
            if not self.verify_input_files(mesh_path, roi_center_path):
                return ROIProcessingResult(
                    success=False,
                    roi_mesh_path="",
                    execution_time=time.time() - start_time,
                    node_reduction=0.0,
                    metadata={},
                    error_message="Input file verification failed"
                )
            
            # Ensure directories exist
            os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
            
            logger.log(logging.INFO,f"Generating ROI mesh:")
            logger.log(logging.INFO,f"- Input mesh: {mesh_path}")
            logger.log(logging.INFO,f"- ROI center: {roi_center_path}")
            logger.log(logging.INFO,f"- Output mesh: {output_mesh_path}")
            logger.log(logging.INFO,f"- ROI radius: {roi_radius}mm")
            
            # Debug: Check file existence
            logger.log(logging.INFO,f"Mesh file exists: {os.path.exists(mesh_path)}")
            logger.log(logging.INFO,f"ROI center file exists: {os.path.exists(roi_center_path)}")
            
            # Try to use supervisor's utilities if available
            try:
                import sys
                import Code.utils as u
                logger.log(logging.INFO,"Using Code.utils module for loading")
                
                # This exactly matches the supervisor's code
                sub = self.context.subject_id
                sub_path = os.path.join(self.context.data_root_path)
                
                # Log the actual path for debugging
                actual_path = os.path.realpath(mesh_path)
                logger.log(logging.INFO,f"Actual mesh path: {actual_path}")
                
                # 1. Load the mesh like the supervisor does
                logger.log(logging.INFO,"Loading mesh...")
                msh = mesh_io.read_msh(mesh_path)
                logger.log(logging.INFO,f"Mesh loaded of type: {type(msh)}")
                
                # 2. Load ROI center like the supervisor does
                logger.log(logging.INFO,"Loading ROI center...")
                roi_center = u.loadmat(roi_center_path)['roi_center']
                logger.log(logging.INFO,f"ROI center loaded of type: {type(roi_center)}")
                
                # Print original node count for reference
                try:
                    original_nodes = msh.nodes[:]
                    original_node_count = len(original_nodes)
                    logger.log(logging.INFO,f"Mesh has {original_node_count} nodes")
                except Exception as e:
                    logger.error(f"Couldn't access nodes: {str(e)}")
                    # Let's debug the mesh object
                    logger.log(logging.INFO,f"Mesh attributes: {dir(msh)}")
                    logger.log(logging.INFO,f"Nodes attributes: {dir(msh.nodes)}")
                    raise ValueError(f"Cannot access mesh nodes: {str(e)}")
                
                # 3. Get skin normal vector
                logger.log(logging.INFO,"Computing skin normal vector...")
                skin_normal_avg = self.get_skin_average_normal_vector(msh, roi_center, roi_radius)
                logger.log(logging.INFO,f"Computed skin normal vector: {skin_normal_avg}")
                
                # 4. Compute cylindrical ROI
                logger.log(logging.INFO,"Computing cylindrical ROI...")
                cylindrical_roi = self.compute_cylindrical_roi(msh, roi_center.gm, skin_normal_avg, roi_radius)
                roi_node_count = np.sum(cylindrical_roi)
                logger.log(logging.INFO,f"Cylindrical ROI contains {roi_node_count} nodes")
                
                # 5. Crop mesh to ROI
                logger.log(logging.INFO,"Cropping mesh to ROI...")
                cropped = self.crop_mesh_nodes(msh, cylindrical_roi)
                
                # 6. Remove islands
                logger.log(logging.INFO,"Removing islands...")
                final_roi = self.remove_islands(cropped, roi_center)
                
                # 7. Save final ROI mesh
                logger.log(logging.INFO,f"Saving ROI mesh to {output_mesh_path}...")
                final_roi.write(output_mesh_path)
                logger.log(logging.INFO,"ROI mesh saved successfully")
                
                # 8. Calculate node reduction
                final_node_count = len(final_roi.nodes[:])
                node_reduction = (original_node_count - final_node_count) / original_node_count * 100
                
                # 9. Create result
                result = ROIProcessingResult(
                    success=True,
                    roi_mesh_path=output_mesh_path,
                    execution_time=time.time() - start_time,
                    node_reduction=node_reduction,
                    metadata={
                        "original_node_count": original_node_count,
                        "roi_node_count": final_node_count,
                        "roi_radius": roi_radius
                    }
                )
                
                return result
                
            except ImportError:
                # If Code.utils is not available, we'll use a different approach
                logger.error("Code.utils module not available. Using fallback method.")
                
                # Define a simple class to match the supervisor's code
                class ROICenter:
                    def __init__(self, gm, skin, skin_vec):
                        self.gm = gm
                        self.skin = skin
                        self.skin_vec = skin_vec
                
                # Load mesh
                msh = mesh_io.read_msh(mesh_path)
                logger.log(logging.INFO,f"Mesh loaded with type: {type(msh)}")
                
                # Load ROI center using scipy instead
                import scipy.io as sio
                mat_data = sio.loadmat(roi_center_path)
                logger.log(logging.INFO,f"MAT file keys: {list(mat_data.keys())}")
                
                if 'roi_center' in mat_data:
                    roi_data = mat_data['roi_center']
                    logger.log(logging.INFO,f"ROI center shape: {roi_data.shape}")
                    
                    if roi_data.shape == (1, 1) and hasattr(roi_data[0,0], "__len__"):
                        # Nested array - extract coordinates
                        nested = roi_data[0,0]
                        logger.log(logging.INFO,f"Nested data type: {type(nested)}, length: {len(nested)}")
                        
                        gm = np.array(nested[0]).flatten()
                        skin = np.array(nested[1]).flatten()
                        skin_vec = np.array(nested[2]).flatten()
                    else:
                        # Fallback - create from mesh geometry
                        raise ValueError("Could not extract ROI center data, falling back to mesh geometry")
                else:
                    raise ValueError("No 'roi_center' field in MAT file")
                
                # Create ROI center object
                roi_center = ROICenter(gm, skin, skin_vec)
                logger.log(logging.INFO,f"Created ROI center: gm={roi_center.gm}, skin={roi_center.skin}, skin_vec={roi_center.skin_vec}")
                
                # Continue with the same logic as above...
                # 3. Get skin normal vector
                skin_normal_avg = self.get_skin_average_normal_vector(msh, roi_center, roi_radius)
                
                # 4. Compute cylindrical ROI
                cylindrical_roi = self.compute_cylindrical_roi(msh, roi_center.gm, skin_normal_avg, roi_radius)
                
                # 5. Crop mesh to ROI
                cropped = self.crop_mesh_nodes(msh, cylindrical_roi)
                
                # 6. Remove islands
                final_roi = self.remove_islands(cropped, roi_center)
                
                # 7. Save final ROI mesh
                final_roi.write(output_mesh_path)
                
                # 8. Calculate stats
                original_node_count = len(msh.nodes[:])
                final_node_count = len(final_roi.nodes[:])
                node_reduction = (original_node_count - final_node_count) / original_node_count * 100
                
                # Return result
                return ROIProcessingResult(
                    success=True,
                    roi_mesh_path=output_mesh_path,
                    execution_time=time.time() - start_time,
                    node_reduction=node_reduction,
                    metadata={
                        "original_node_count": original_node_count,
                        "roi_node_count": final_node_count,
                        "roi_radius": roi_radius
                    }
                )
        except Exception as e:
            logger.error(f"Error generating ROI mesh: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return ROIProcessingResult(
                success=False,
                roi_mesh_path="",
                execution_time=time.time() - start_time,
                node_reduction=0.0,
                metadata={},
                error_message=str(e)
            )
