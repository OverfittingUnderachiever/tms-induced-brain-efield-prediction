# /home/freyhe/MA_Henry/tms_efield_prediction/data/transformations/element_to_node.py
"""
Utility for interpolating element-based field values to node-based values.
Specifically designed to handle the case where dA/dt data is available
only for triangular elements in a mesh that may contain multiple element types.
"""

import numpy as np
from typing import Optional, List, Tuple
import time

def interpolate_element_to_node(
    element_data: np.ndarray,
    node_count: int,
    element_nodes: np.ndarray,
    element_types: Optional[np.ndarray] = None,
    triangle_type: int = 2,  # SimNIBS uses type 2 for triangles
    one_indexed: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Interpolate element-based field values to node-based values using simple averaging.
    Handles the case where element_data only contains values for triangular elements.
    
    Args:
        element_data: Element-based field data with shape (n_simulations, n_triangles, n_components)
        node_count: Total number of nodes in the mesh
        element_nodes: List of node indices for each element, shape (n_elements, nodes_per_element)
        element_types: Types of each element (if None, assumes all elements are triangles)
        triangle_type: Element type code for triangles (default: 2 for SimNIBS)
        one_indexed: Whether node indices are 1-indexed (SimNIBS convention) or 0-indexed
        verbose: Whether to print progress information
        
    Returns:
        Node-based field data with shape (n_simulations, n_nodes, n_components)
    """
    start_time = time.time()
    if verbose:
        print(f"Starting element-to-node interpolation...")
    
    # Get dimensions
    n_simulations, n_triangles, n_components = element_data.shape
    
    # Filter for triangle elements if element_types is provided
    triangle_indices = None
    if element_types is not None:
        triangle_indices = np.where(element_types == triangle_type)[0]
        if verbose:
            print(f"Found {len(triangle_indices)} triangular elements out of {len(element_types)} total elements.")
        
        if len(triangle_indices) != n_triangles:
            print(f"WARNING: Number of triangles in mesh ({len(triangle_indices)}) doesn't match data shape ({n_triangles}).")
            if len(triangle_indices) > n_triangles:
                print(f"Assuming data covers only the first {n_triangles} triangles.")
                triangle_indices = triangle_indices[:n_triangles]
            else:
                print(f"Not enough triangles in mesh. Some data points will be unused.")
    
    # Initialize output array for node-based values
    node_data = np.zeros((n_simulations, node_count, n_components), dtype=element_data.dtype)
    
    # Initialize arrays to track contribution to each node
    node_values_sum = np.zeros((n_simulations, node_count, n_components), dtype=element_data.dtype)
    node_contribution_count = np.zeros((n_simulations, node_count), dtype=int)
    
    if verbose:
        print(f"Processing {n_triangles} triangular elements for {node_count} nodes across {n_simulations} simulations...")
    
    # Process each simulation
    for sim_idx in range(n_simulations):
        if verbose and sim_idx % max(1, n_simulations // 10) == 0:
            print(f"Processing simulation {sim_idx+1}/{n_simulations}...")
        
        # If using filtered triangles, process them specifically
        if triangle_indices is not None:
            for i, elm_idx in enumerate(triangle_indices):
                if i >= n_triangles:  # Safety check
                    break
                    
                # Get the nodes for this triangle element
                elm_nodes = element_nodes[elm_idx]
                
                # Get field value from the data (using index i since element_data is only for triangles)
                field_value = element_data[sim_idx, i]
                
                # Contribute this element's value to each of its nodes
                for node_idx in elm_nodes:
                    # Adjust for 1-based indexing if needed
                    if one_indexed:
                        node_idx = node_idx - 1
                    
                    # Skip invalid indices
                    if node_idx < 0 or node_idx >= node_count:
                        continue
                    
                    # Add the element's field value to this node's sum
                    node_values_sum[sim_idx, node_idx] += field_value
                    node_contribution_count[sim_idx, node_idx] += 1
        else:
            # Process elements sequentially, assuming they all match the data order
            for i in range(n_triangles):
                # Get the nodes for this element
                elm_nodes = element_nodes[i]
                
                # Get field value from the data
                field_value = element_data[sim_idx, i]
                
                # Contribute this element's value to each of its nodes
                for node_idx in elm_nodes:
                    # Adjust for 1-based indexing if needed
                    if one_indexed:
                        node_idx = node_idx - 1
                    
                    # Skip invalid indices
                    if node_idx < 0 or node_idx >= node_count:
                        continue
                    
                    # Add the element's field value to this node's sum
                    node_values_sum[sim_idx, node_idx] += field_value
                    node_contribution_count[sim_idx, node_idx] += 1
    
    # Compute averages, avoiding division by zero
    for sim_idx in range(n_simulations):
        for node_idx in range(node_count):
            count = node_contribution_count[sim_idx, node_idx]
            if count > 0:
                node_data[sim_idx, node_idx] = node_values_sum[sim_idx, node_idx] / count
    
    # Check for nodes with no contributions
    no_contrib_count = np.sum(node_contribution_count[0] == 0)
    if no_contrib_count > 0 and verbose:
        no_contrib_percent = no_contrib_count / node_count * 100
        print(f"Warning: {no_contrib_count} nodes ({no_contrib_percent:.1f}%) have no element contributions.")
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"Interpolation completed in {elapsed:.2f} seconds.")
        print(f"Output shape: {node_data.shape}")
        
        # Print some statistics about the interpolated data
        for sim_idx in [0]:  # Just show stats for the first simulation
            sim_data = node_data[sim_idx]
            nonzero_nodes = np.sum(np.linalg.norm(sim_data, axis=1) > 0)
            nonzero_percent = nonzero_nodes / node_count * 100
            print(f"Simulation {sim_idx} statistics:")
            print(f"  Total non-zero nodes: {nonzero_nodes} ({nonzero_percent:.1f}%)")
            
            # Only compute unique values if there are non-zero nodes
            if nonzero_nodes > 0:
                magnitudes = np.linalg.norm(sim_data, axis=1)
                nonzero_magnitudes = magnitudes[magnitudes > 0]
                unique_count = len(np.unique(nonzero_magnitudes.round(decimals=6)))
                print(f"  Number of unique magnitude values: {unique_count}")
                print(f"  Min magnitude: {np.min(nonzero_magnitudes)}")
                print(f"  Max magnitude: {np.max(nonzero_magnitudes)}")
                print(f"  Mean magnitude: {np.mean(nonzero_magnitudes)}")
    
    return node_data