#!/usr/bin/env python
"""
Unit tests for the updated load_matsimnibs function in simnibs_io.py.

This script verifies that the updated function correctly loads .mat files
with the new structure and produces results identical to the prototype implementation.
"""

import os
import sys
import unittest
import numpy as np
import h5py
import logging
from typing import Tuple, List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the module containing the updated function
from tms_efield_prediction.data.formats.simnibs_io import load_matsimnibs, MeshData
from tms_efield_prediction.utils.debug.hooks import DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestMatrixLoader:
    """Helper class to load matrices using the prototype's approach for comparison."""
    
    @staticmethod
    def load_matsimnibs_prototype(file_path: str) -> np.ndarray:
        """
        Loads matsimnibs data using the prototype's approach.
        
        Args:
            file_path: Path to the .mat file
            
        Returns:
            numpy array containing coil position matrices
        """
        if not file_path.endswith('.mat'):
            raise ValueError(f"Expected .mat file, got: {file_path}")
            
        with h5py.File(file_path, 'r') as f:
            pos_matrices = []
            # Extract using the prototype's approach
            if '/matsimnibs' in f:
                ref = f['/matsimnibs'][0,0]
                obj = f[ref][0]
                for r in obj:
                    pos_matrices.append(np.array(f[r]).T)
                matsimnibs = np.stack(pos_matrices)
                return matsimnibs
            else:
                raise ValueError(f"File does not contain '/matsimnibs' key: {file_path}")


class SimpleDebugHook(DebugHook):
    """Simple implementation of DebugHook for testing."""
    
    def __init__(self):
        self.events = []
        self.errors = []
        
    def should_sample(self) -> bool:
        return True
        
    def record_event(self, name: str, data: Dict[str, Any]) -> None:
        self.events.append({"name": name, "data": data})
        logger.info(f"Event: {name} - {data}")
        
    def record_error(self, name: str, data: Dict[str, Any]) -> None:
        self.errors.append({"name": name, "data": data})
        logger.error(f"Error: {name} - {data}")
        
    def get_events(self) -> List[Dict[str, Any]]:
        return self.events
        
    def get_errors(self) -> List[Dict[str, Any]]:
        return self.errors


class TestMatSimNIBSLoader(unittest.TestCase):
    """Test cases for the updated load_matsimnibs function."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data paths."""
        # Look in the actual data directory first
        base_dirs = [
            # Typical location in the project structure
            '/home/freyhe/MA_Henry/data',
            # Alternative locations to try
            os.path.expanduser('~/MA_Henry/data'),
            os.path.expanduser('~/data'),
            # Fall back to test directory if needed
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../../test_data'))
        ]
        
        # Try subject IDs in order
        subject_ids = ['001', '000', '002', '003']
        # Try experiment names in order
        experiment_names = ['all', 'test', 'high_resolution', 'experiment']
        
        cls.mat_file_path = None
        
        # Search for existing matsimnibs files
        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                for subject_id in subject_ids:
                    subject_dir = os.path.join(base_dir, f'sub-{subject_id}')
                    if os.path.exists(subject_dir):
                        for exp_name in experiment_names:
                            exp_dir = os.path.join(subject_dir, 'experiment', exp_name)
                            if os.path.exists(exp_dir):
                                # Check for matsimnibs file
                                mat_path = os.path.join(exp_dir, f'sub-{subject_id}_matsimnibs.mat')
                                if os.path.exists(mat_path):
                                    cls.mat_file_path = mat_path
                                    cls.test_data_dir = base_dir
                                    cls.test_subject_dir = subject_dir
                                    cls.test_experiment_dir = exp_dir
                                    break
                    if cls.mat_file_path:
                        break
            if cls.mat_file_path:
                break
        
        # Log the file path that was found (or not)
        if cls.mat_file_path:
            logger.info(f"Found matsimnibs file at: {cls.mat_file_path}")
        else:
            logger.warning("No matsimnibs .mat file found in any of the searched locations.")
            logger.warning("Tests will be skipped if the file doesn't exist at runtime.")
    
    def setUp(self):
        """Set up test environment before each test."""
        self.debug_hook = SimpleDebugHook()
        self.resource_monitor = None  # We'll skip resource monitoring for tests
    
    def test_file_exists(self):
        """Verify that the test file exists."""
        self.assertTrue(
            os.path.exists(self.mat_file_path),
            f"Test .mat file not found at: {self.mat_file_path}"
        )
    
    def test_load_matsimnibs(self):
        """Test loading of .mat file using the updated function."""
        logger.info(f"Testing load_matsimnibs with file: {self.mat_file_path}")
        
        # Load using the updated function
        matrices = load_matsimnibs(
            self.mat_file_path,
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        # Validate basic properties
        self.assertIsNotNone(matrices, "Loaded matrices should not be None")
        self.assertIsInstance(matrices, np.ndarray, "Loaded data should be a numpy array")
        
        # Verify shape
        self.assertGreaterEqual(len(matrices.shape), 3, "Should have at least 3 dimensions")
        self.assertEqual(matrices.shape[-1], 4, "Last dimension should be 4")
        self.assertEqual(matrices.shape[-2], 4, "Second-to-last dimension should be 4")
        
        # Log detailed information
        logger.info(f"Loaded matrix shape: {matrices.shape}")
        logger.info(f"First matrix:\n{matrices[0]}")
        
        # Verify that each matrix is a valid transformation matrix
        self._validate_transformation_matrices(matrices)
        
        return matrices
    
    def test_compare_with_prototype(self):
        """Compare results with the prototype's implementation."""
        # Skip if the file doesn't exist
        if not os.path.exists(self.mat_file_path):
            self.skipTest(f"Test .mat file not found: {self.mat_file_path}")
        
        logger.info("Comparing with prototype implementation")
        
        # Load using both approaches
        matrices_updated = load_matsimnibs(
            self.mat_file_path,
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        matrices_prototype = TestMatrixLoader.load_matsimnibs_prototype(self.mat_file_path)
        
        # Compare shapes
        self.assertEqual(
            matrices_updated.shape, 
            matrices_prototype.shape,
            "Shapes of matrices should match"
        )
        
        # Compare values
        np.testing.assert_allclose(
            matrices_updated, 
            matrices_prototype,
            rtol=1e-5, 
            atol=1e-8,
            err_msg="Matrix values from updated function don't match prototype"
        )
        
        logger.info("Matrix loading implementations produce identical results")
    
    def test_debug_hook_integration(self):
        """Test that the debug hook is properly utilized."""
        # Skip if the file doesn't exist
        if not os.path.exists(self.mat_file_path):
            self.skipTest(f"Test .mat file not found: {self.mat_file_path}")
            
        # Load with debug hook
        _ = load_matsimnibs(
            self.mat_file_path,
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        # Verify events were recorded
        events = self.debug_hook.get_events()
        event_names = [e["name"] for e in events]
        
        self.assertIn("load_matsimnibs_start", event_names, "Should record start event")
        self.assertIn("load_matsimnibs_complete", event_names, "Should record completion event")
        
        # Verify no errors
        errors = self.debug_hook.get_errors()
        self.assertEqual(len(errors), 0, f"Should not have errors, but got: {errors}")
        
        # Log debug events
        logger.info(f"Debug events recorded: {event_names}")
        
    def _validate_transformation_matrices(self, matrices: np.ndarray) -> None:
        """
        Validate that the matrices are proper transformation matrices.
        
        Args:
            matrices: Array of transformation matrices to validate
        """
        # Reshape to handle different dimensions
        if len(matrices.shape) > 3:
            matrices_flat = matrices.reshape(-1, 4, 4)
        else:
            matrices_flat = matrices
            
        for i, matrix in enumerate(matrices_flat):
            # Check that the matrix has the right shape
            self.assertEqual(matrix.shape, (4, 4), f"Matrix {i} has incorrect shape: {matrix.shape}")
            
            # Check that the bottom row is [0, 0, 0, 1]
            np.testing.assert_allclose(
                matrix[3, :], 
                [0, 0, 0, 1],
                rtol=1e-5, 
                atol=1e-8,
                err_msg=f"Matrix {i} has incorrect bottom row: {matrix[3, :]}"
            )
            
            # Check that the rotation part (top-left 3x3) is orthogonal
            rot = matrix[:3, :3]
            ident = np.dot(rot, rot.T)
            np.testing.assert_allclose(
                ident, 
                np.eye(3),
                rtol=1e-5, 
                atol=1e-8,
                err_msg=f"Matrix {i} rotation part is not orthogonal"
            )
        
        logger.info(f"All {len(matrices_flat)} matrices are valid transformation matrices")


def print_file_structure(file_path: str) -> None:
    """
    Print the structure of an HDF5 file for debugging.
    
    Args:
        file_path: Path to the HDF5 (.mat) file
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info(f"Examining file structure: {file_path}")
    
    def print_attrs(name, obj):
        logger.info(f"Object: {name}")
        if isinstance(obj, h5py.Dataset):
            logger.info(f"  Type: Dataset")
            logger.info(f"  Shape: {obj.shape}")
            logger.info(f"  Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            logger.info(f"  Type: Group")
            logger.info(f"  Keys: {list(obj.keys())}")
    
    with h5py.File(file_path, 'r') as f:
        # Print top-level keys
        logger.info(f"Top-level keys: {list(f.keys())}")
        
        # Examine matsimnibs structure if it exists
        if '/matsimnibs' in f:
            logger.info("Examining '/matsimnibs' structure:")
            matsimnibs = f['/matsimnibs']
            logger.info(f"  Type: {type(matsimnibs)}")
            logger.info(f"  Shape: {matsimnibs.shape}")
            
            if len(matsimnibs.shape) > 0:
                ref = matsimnibs[0,0]
                logger.info(f"  Reference: {ref}")
                
                obj = f[ref]
                logger.info(f"  Referenced object type: {type(obj)}")
                logger.info(f"  Referenced object shape: {obj.shape}")
                
                if len(obj.shape) > 0:
                    obj_refs = obj[0]
                    logger.info(f"  Number of matrices: {len(obj_refs)}")
                    
                    if len(obj_refs) > 0:
                        sample_matrix = np.array(f[obj_refs[0]]).T
                        logger.info(f"  Sample matrix shape: {sample_matrix.shape}")
                        logger.info(f"  Sample matrix:\n{sample_matrix}")
        
        # Visit all items in the file
        f.visititems(print_attrs)


if __name__ == "__main__":
    # Setup the tests to find the actual file paths
    TestMatSimNIBSLoader.setUpClass()
    
    # Print the structure of the test file if it exists
    test_file = TestMatSimNIBSLoader.mat_file_path
    
    if test_file and os.path.exists(test_file):
        logger.info(f"Analyzing structure of: {test_file}")
        print_file_structure(test_file)
    else:
        logger.warning("No matsimnibs file found to analyze")
    
    # Run the tests
    unittest.main()