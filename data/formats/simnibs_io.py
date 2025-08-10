"""
SimNIBS data loading utilities.

This module provides functions for loading and processing SimNIBS mesh files
and related data formats used in TMS E-field prediction.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import os
import h5py
from simnibs import mesh_io
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tms_efield_prediction.utils.debug.hooks import DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor


@dataclass
class MeshData:
    """Container for mesh data with relevant metadata."""
    nodes: np.ndarray
    elements: np.ndarray
    node_data: Dict[str, np.ndarray]
    element_data: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


def load_mesh(
    file_path: str,
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
) -> MeshData:
    """
    Load a SimNIBS mesh file with explicit resource tracking.
    
    Args:
        file_path: Path to the .msh file
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        
    Returns:
        MeshData object containing mesh information
    
    Raises:
        FileNotFoundError: If mesh file doesn't exist
        ValueError: If mesh file is invalid
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simnibs_io.load_mesh", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("load_mesh_start", {"path": file_path})
    
    try:
        # Load mesh using SimNIBS utilities
        mesh = mesh_io.read_msh(file_path)
        
        # Extract relevant data
        nodes = mesh.nodes[:]
        elements = {
            'tetra': mesh.elm.tetrahedra,
            'triangles': mesh.elm.triangles if hasattr(mesh.elm, 'triangles') else None
        }
        
        # Extract node and element data
        node_data = {}
        for i, field in enumerate(mesh.nodedata):
            node_data[field.name] = field.value
            
        element_data = {}
        for i, field in enumerate(mesh.elmdata):
            element_data[field.name] = field.value
        
        # Extract metadata
        metadata = {
            'node_count': nodes.shape[0],
            'tetra_count': elements['tetra'].shape[0] if elements['tetra'] is not None else 0,
            'triangle_count': elements['triangles'].shape[0] if elements['triangles'] is not None else 0
        }
        
        mesh_data = MeshData(
            nodes=nodes,
            elements=elements,
            node_data=node_data,
            element_data=element_data,
            metadata=metadata
        )
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "load_mesh_complete", 
                {
                    "path": file_path,
                    "node_count": nodes.shape[0],
                    "metadata": metadata
                }
            )
        
        return mesh_data
        
    except Exception as e:
        if debug_hook:
            debug_hook.record_error("load_mesh_error", {
                "path": file_path,
                "error": str(e)
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simnibs_io.load_mesh", "end")


def load_dadt_data(
    file_path: str,
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
) -> np.ndarray:
    """
    Load dA/dt data from an HDF5 or numpy file.
    
    Args:
        file_path: Path to the dA/dt data file (.h5 or .npy)
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        
    Returns:
        numpy array containing dA/dt data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simnibs_io.load_dadt_data", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("load_dadt_start", {"path": file_path})
    
    try:
        # Handle different file formats
        if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            with h5py.File(file_path, 'r') as f:
                if 'dAdt' in f:
                    dadt = f['dAdt'][:]
                else:
                    # Try to find the first dataset in the file
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            dadt = f[key][:]
                            break
        elif file_path.endswith('.npy'):
            dadt = np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format for dA/dt data: {file_path}")
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "load_dadt_complete", 
                {
                    "path": file_path,
                    "shape": dadt.shape,
                    "dtype": str(dadt.dtype)
                }
            )
        
        return dadt
    
    except Exception as e:
        if debug_hook:
            debug_hook.record_error("load_dadt_error", {
                "path": file_path,
                "error": str(e)
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simnibs_io.load_dadt_data", "end")

def load_matsimnibs(
    file_path: str,
    debug_hook: Optional[DebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None
) -> np.ndarray:
    """
    Load matsimnibs coil position data.
    
    Args:
        file_path: Path to the matsimnibs file (.mat, .npy)
        debug_hook: Optional debug hook for tracking
        resource_monitor: Optional resource monitor for memory tracking
        
    Returns:
        numpy array containing coil position matrices
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or matrices can't be found
    """
    if resource_monitor:
        resource_monitor.update_component_usage("simnibs_io.load_matsimnibs", "start")
    
    if debug_hook and debug_hook.should_sample():
        debug_hook.record_event("load_matsimnibs_start", {"path": file_path})
    
    try:
        # Handle different file formats
        if file_path.endswith('.mat'):
            with h5py.File(file_path, 'r') as f:
                # Log file structure for debugging
                if debug_hook and debug_hook.should_sample():
                    debug_hook.record_event("load_matsimnibs_structure", {
                        "keys": list(f.keys()),
                        "has_matsimnibs": '/matsimnibs' in f
                    })
                
                # Method 1: Use the prototype's approach for .mat files with nested structure
                if '/matsimnibs' in f:
                    try:
                        pos_matrices = []
                        ref = f['/matsimnibs'][0,0]
                        obj = f[ref][0]
                        for r in obj:
                            pos_matrices.append(np.array(f[r]).T)
                        
                        if pos_matrices:
                            matsimnibs = np.stack(pos_matrices)
                            
                            if debug_hook and debug_hook.should_sample():
                                debug_hook.record_event("load_matsimnibs_method", {
                                    "method": "nested_structure",
                                    "matrices_found": len(pos_matrices)
                                })
                        else:
                            raise ValueError("No matrices found in the expected structure")
                    except Exception as e:
                        if debug_hook:
                            debug_hook.record_error("load_matsimnibs_nested_error", {
                                "error": str(e),
                                "structure_info": "Attempted to extract using nested reference structure"
                            })
                        
                        # Fallback: Try to find suitable datasets directly
                        found = False
                        for key in f.keys():
                            if isinstance(f[key], h5py.Dataset) and len(f[key].shape) >= 3:
                                data = f[key][:]
                                if data.shape[-1] == 4 and data.shape[-2] == 4:
                                    matsimnibs = data
                                    found = True
                                    
                                    if debug_hook and debug_hook.should_sample():
                                        debug_hook.record_event("load_matsimnibs_method", {
                                            "method": "fallback_dataset",
                                            "key": key
                                        })
                                    break
                        
                        if not found:
                            raise ValueError(f"Could not find transformation matrices: {str(e)}")
                else:
                    # No '/matsimnibs' key found - try generic approach to find 4x4 matrices
                    found = False
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            try:
                                data = f[key][:]
                                if len(data.shape) >= 3 and data.shape[-1] == 4 and data.shape[-2] == 4:
                                    matsimnibs = data
                                    found = True
                                    
                                    if debug_hook and debug_hook.should_sample():
                                        debug_hook.record_event("load_matsimnibs_method", {
                                            "method": "generic_dataset",
                                            "key": key
                                        })
                                    break
                            except Exception as dataset_e:
                                if debug_hook:
                                    debug_hook.record_error("load_matsimnibs_dataset_error", {
                                        "key": key,
                                        "error": str(dataset_e)
                                    })
                                continue
                    
                    if not found:
                        raise ValueError(f"No suitable transformation matrices found in file: {file_path}")
        
        elif file_path.endswith('.npy'):
            # Direct load for .npy files
            matsimnibs = np.load(file_path)
            
            if debug_hook and debug_hook.should_sample():
                debug_hook.record_event("load_matsimnibs_method", {"method": "npy_direct_load"})
        else:
            raise ValueError(f"Unsupported file format for matsimnibs data: {file_path}")
        
        # Validate the shape
        if len(matsimnibs.shape) < 3 or matsimnibs.shape[-1] != 4 or matsimnibs.shape[-2] != 4:
            error_msg = f"Invalid matsimnibs data shape: {matsimnibs.shape}, expected last two dimensions to be (4, 4)"
            if debug_hook:
                debug_hook.record_error("load_matsimnibs_shape_error", {"error": error_msg})
            raise ValueError(error_msg)
        
        if debug_hook and debug_hook.should_sample():
            debug_hook.record_event(
                "load_matsimnibs_complete", 
                {
                    "path": file_path,
                    "shape": matsimnibs.shape
                }
            )
        
        return matsimnibs
    
    except Exception as e:
        if debug_hook:
            debug_hook.record_error("load_matsimnibs_error", {
                "path": file_path,
                "error": str(e)
            })
        raise
    finally:
        if resource_monitor:
            resource_monitor.update_component_usage("simnibs_io.load_matsimnibs", "end")