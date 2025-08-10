#!/usr/bin/env python3
"""
Test script to verify the TMS E-field prediction pipeline functionality.

This script:
1. Sets up a minimal test environment
2. Runs the transformation pipeline on a small test dataset
3. Verifies basic functionality
4. Tests error handling for SimNIBS-specific formats
"""

import os
import sys
import numpy as np
import tempfile
import shutil
import logging
import time
from pathlib import Path
import unittest
import importlib.util
from unittest.mock import Mock, patch




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tms_tester')

# Constants for testing
TEST_SUBJECT_ID = "test001"
TEST_GRID_SIZE = 32

class TMSRunnerTest(unittest.TestCase):
    """Test case for TMS runner script."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directories
        cls.test_dir = tempfile.mkdtemp(prefix="tms_test_")
        cls.data_dir = os.path.join(cls.test_dir, "data")
        cls.output_dir = os.path.join(cls.test_dir, "output")
        cls.cache_dir = os.path.join(cls.test_dir, "cache")
        
        # Create directory structure
        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        
        # Import the tms_runner script
        cls.runner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tms_runner.py")
        
        # Mock needed classes and functions
        cls.setup_mocks()
        
        # Create test data
        cls.create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove temporary directories
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def setup_mocks(cls):
        """Set up mock objects for testing."""
        # Mock SimNIBS classes
        cls.mesh_io_mock = Mock()
        cls.resource_monitor_mock = Mock()
        cls.debug_hook_mock = Mock()
        
        # Mock mesh data
        cls.mesh_data_mock = Mock()
        cls.mesh_data_mock.nodes = np.random.rand(1000, 3)
        cls.mesh_data_mock.metadata = {
            'node_count': 1000,
            'tetra_count': 500,
            'triangle_count': 1000
        }
        cls.mesh_data_mock.elements = {
            'tetra': np.random.randint(1, 1000, (500, 4)),
            'triangles': np.random.randint(1, 1000, (1000, 3))
        }
        cls.mesh_data_mock.node_data = {
            'E': np.random.rand(1000, 3)  # Vector E-field
        }
        cls.mesh_data_mock.element_data = {}
    
    @classmethod
    def create_test_data(cls):
        """Create test data for pipeline."""
        # Create subject directories
        subject_dir = os.path.join(cls.data_dir, f"sub-{TEST_SUBJECT_ID}")
        headmodel_dir = os.path.join(subject_dir, "headmodel")
        experiment_dir = os.path.join(subject_dir, "experiment", "nn")
        
        os.makedirs(headmodel_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create dummy MRI data (3D array with random values)
        mri_data = np.random.rand(100, 100, 100)
        
        # Create dummy dA/dt data (4D array with 3 vector components)
        dadt_data = np.random.rand(10, 1000, 3)  # 10 coil positions, 1000 triangles, 3 components
        
        # Create dummy E-field data
        efield_data = np.random.rand(10, 1000, 3)  # Same shape as dA/dt for simplicity
        
        # Save test data
        np.save(os.path.join(headmodel_dir, f"sub-{TEST_SUBJECT_ID}_mri.npy"), mri_data)
        np.save(os.path.join(experiment_dir, "dAdts.npy"), dadt_data)
        np.save(os.path.join(experiment_dir, f"sub-{TEST_SUBJECT_ID}_efields.npy"), efield_data)
        
        # Create dummy coil positions
        coil_positions = np.random.rand(10, 4, 4)  # 10 positions, 4x4 matrices
        np.save(os.path.join(experiment_dir, f"sub-{TEST_SUBJECT_ID}_matsimnibs.npy"), coil_positions)
        
        # Create dummy ROI center
        roi_center = {
            'gm': np.random.rand(3),
            'skin': np.random.rand(3),
            'skin_vec': np.array([0, 0, 1])
        }
        
        # Create a simple JSON file since we don't need actual .mat file for testing
        with open(os.path.join(subject_dir, "experiment", f"sub-{TEST_SUBJECT_ID}_roi_center.json"), 'w') as f:
            f.write('{"roi_center": [0, 0, 0]}')
        
        # Store test data for assertions
        cls.test_data = {
            'mri': mri_data,
            'dadt': dadt_data,
            'efield': efield_data,
            'coil_positions': coil_positions,
            'roi_center': roi_center
        }
    
    def test_mesh_to_grid_transformer_basic(self):
        """Test basic functionality of MeshToGridTransformer."""
        try:
            # Import the transformer class
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from tms_efield_prediction.data.transformations.mesh_to_grid import MeshToGridTransformer
            
            # Mock context for testing
            context_mock = Mock()
            context_mock.config = {"n_bins": TEST_GRID_SIZE}
            context_mock.output_shape = (TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE)
            
            # Create transformer
            transformer = MeshToGridTransformer(context_mock)
            
            # Test transform with simple data
            node_centers = np.random.rand(1000, 3)
            data_values = np.random.rand(1000)
            
            # Run transform
            grid_data, mask, metadata = transformer.transform(data_values, node_centers, TEST_GRID_SIZE)
            
            # Verify outputs
            self.assertEqual(grid_data.shape, (TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE))
            self.assertEqual(mask.shape, (TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE))
            self.assertIn('grid_shape', metadata)
            self.assertIn('is_vector', metadata)
            
            logger.info("MeshToGridTransformer basic test passed")
            
        except ImportError:
            self.skipTest("MeshToGridTransformer not available")
    
    def test_channel_stacking_pipeline_basic(self):
        """Test basic functionality of ChannelStackingPipeline."""
        try:
            # Import the stacker class
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from tms_efield_prediction.data.transformations.stack_pipeline import ChannelStackingPipeline, StackingConfig
            
            # Mock context for testing
            context_mock = Mock()
            context_mock.config = {"n_bins": TEST_GRID_SIZE}
            context_mock.output_shape = (TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE)
            context_mock.normalization_method = "minmax"
            context_mock.dadt_scaling_factor = 1.0
            
            # Create stacker
            stacking_config = StackingConfig(
                normalization_method="minmax",
                dadt_scaling_factor=1.0,
                channel_order=["mri", "dadt"],
                output_shape=(TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE)
            )
            
            stacker = ChannelStackingPipeline(context_mock, config=stacking_config)
            
            # Create test data
            mri_data = np.random.rand(TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE)
            dadt_data = np.random.rand(TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE, 3)  # Vector dA/dt
            
            # Test stack_channels
            stacked_data = stacker.channel_stacker({
                'mri': mri_data,
                'dadt': dadt_data,
                'channel_order': ["mri", "dadt"]
            }).output
            
            # Verify output - should have 4 channels (MRI + 3 dA/dt components)
            self.assertEqual(stacked_data.shape, (TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE, 4))
            
            logger.info("ChannelStackingPipeline basic test passed")
            
        except ImportError:
            self.skipTest("ChannelStackingPipeline not available")
    
    def test_simnibs_io_error_handling(self):
        """Test error handling for SimNIBS IO functions."""
        try:
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

            # Import the IO module
            from tms_efield_prediction.data.formats.simnibs_io import load_mesh, load_dadt_data, load_matsimnibs
            
            # Test with invalid paths to trigger error handling
            with self.assertRaises(Exception):
                load_mesh("nonexistent_path.msh")
            
            with self.assertRaises(Exception):
                load_dadt_data("nonexistent_path.h5")
            
            with self.assertRaises(Exception):
                load_matsimnibs("nonexistent_path.mat")
            
            logger.info("SimNIBS IO error handling test passed")
            
        except ImportError:
            self.skipTest("SimNIBS IO module not available")
    
    # Update the test_cache_manager method in test_tms_pipeline.py

    # Replace the test_cache_manager method in test_tms_pipeline.py with this:

    # Replace the test_cache_manager method in test_tms_pipeline.py with this:

    # Replace the test_cache_manager method in test_tms_pipeline.py with this:

    # Replace the test_cache_manager method in test_tms_pipeline.py with this:

    # Replace the test_cache_manager method in test_tms_pipeline.py with this:

# Replace the test_cache_manager method in test_tms_pipeline.py with this:

    def test_cache_manager(self):
        """Test CacheManager functionality."""
        import sys
        import os
        import tempfile
        import shutil
        import importlib.util
        import logging
        
        # Get a logger
        logger = logging.getLogger('tms_tester')
        
        # Create a temporary directory for this test
        temp_dir = tempfile.mkdtemp(prefix="tms_test_")
        
        # Store original sys.path to restore it later
        original_path = sys.path.copy()
        
        try:
            # Add project root to path
            # This assumes tests/integration is two levels below project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # Try to find the module using standard import
            try:
                from tms_efield_prediction.tms_runner import CacheManager
            except ImportError:
                # If standard import fails, try to locate the file
                module_path = os.path.join(project_root, 'tms_efield_prediction', 'tms_runner.py')
                
                if not os.path.exists(module_path):
                    self.fail(f"Could not find tms_runner.py in expected location: {module_path}")
                
                # Import directly from file path
                spec = importlib.util.spec_from_file_location("tms_runner", module_path)
                tms_runner = importlib.util.module_from_spec(spec)
                sys.modules["tms_runner"] = tms_runner
                spec.loader.exec_module(tms_runner)
                CacheManager = tms_runner.CacheManager
            
            # Create cache manager
            cache_dir = os.path.join(temp_dir, "cache")
            cache_manager = CacheManager(cache_dir)
            
            # Create test data
            test_data = np.random.random((10, 10, 10))
            test_metadata = {"test_key": "test_value"}
            
            # Test cache save/load
            self.assertTrue(cache_manager.save_to_cache("test_key", test_data, test_metadata))
            self.assertTrue(cache_manager.has_cache("test_key"))
            
            # Load from cache
            loaded_data = cache_manager.load_from_cache("test_key")
            self.assertIsNotNone(loaded_data)
            
            # Compare data
            self.assertEqual(loaded_data.shape, test_data.shape)
            np.testing.assert_array_equal(loaded_data, test_data)
            
            # Test metadata validation
            self.assertTrue(cache_manager.has_cache("test_key", {"test_key": "test_value"}))
            self.assertFalse(cache_manager.has_cache("test_key", {"test_key": "wrong_value"}))
            
            # Test clear cache
            cache_manager.clear_cache("test_key")
            self.assertFalse(cache_manager.has_cache("test_key"))
            
            logger.info("Cache manager test passed")
            
        finally:
            # Restore original sys.path
            sys.path = original_path
            
            # Clean up the temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run tests."""
    unittest.main()


if __name__ == "__main__":
    main()