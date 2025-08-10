#!/usr/bin/env python3
"""
Test script for TMS E-field prediction pipeline.

This script tests the runner with a sample subject.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import tms_runner module
from tms_efield_prediction import tms_runner
from tms_efield_prediction.tms_runner import RunnerConfig
from tms_efield_prediction.utils.state.context import TMSPipelineContext
from tms_efield_prediction.utils.debug.hooks import PipelineDebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.data.transformations.mesh_to_grid import MeshToGridTransformer


class TMSRunnerTest(unittest.TestCase):
    """Test cases for TMS runner script."""
    
    def setUp(self):
        """Set up test environment."""
        # Define test config
        self.config = tms_runner.RunnerConfig(
            subject_id="005",
            data_root_path="/data_lu",
            output_path="/tmp/tms_output",
            n_bins=32,  # Small grid for tests
            use_cache=True,
            cache_dir="/tmp/tms_cache",
            visualize=True,
            benchmark=True
        )
        
        # Create test directories
        os.makedirs(self.config.output_path, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Create resource monitor and debug hook
        self.resource_monitor = tms_runner.create_resource_monitor()
        self.debug_hook = tms_runner.create_debug_hook()
        
        # Create context
        self.context = tms_runner.create_pipeline_context(
            self.config, self.resource_monitor
        )
    
    def test_mesh_to_grid_transformer_basic(self):
        """Test basic functionality of MeshToGridTransformer."""
        # Create test data
        TEST_GRID_SIZE = 16
        node_centers = np.random.random((100, 3)) * 100
        data_values = np.random.random(100)
        
        # Create transformer
        transformer = MeshToGridTransformer(
            context=self.context,
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        # Test transform method
        grid_data, mask, metadata = transformer.transform(data_values, node_centers, TEST_GRID_SIZE)
        
        # Check results
        self.assertEqual(grid_data.shape, (TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE))
        self.assertEqual(mask.shape, (TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE))
        self.assertTrue('grid_shape' in metadata)
        self.assertEqual(metadata['grid_shape'], (TEST_GRID_SIZE, TEST_GRID_SIZE, TEST_GRID_SIZE))
    
    def test_cache_manager(self):
        """Test CacheManager functionality."""
        # Create cache manager
        cache_manager = tms_runner.CacheManager(self.config.cache_dir)
        
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
    
    @unittest.skip("Skip full pipeline test for now")
    def test_run_pipeline(self):
        """Test the full pipeline."""
        # Run pipeline
        results = tms_runner.run_pipeline(self.config)
        
        # Check results
        self.assertEqual(results["status"], "success")
        self.assertEqual(results["subject_id"], self.config.subject_id)
        self.assertTrue("benchmarks" in results)
        self.assertTrue("visualizations" in results)
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop resource monitor
        self.resource_monitor.stop_monitoring()
        
        # Clean up test directories
        import shutil
        shutil.rmtree(self.config.output_path, ignore_errors=True)
        shutil.rmtree(self.config.cache_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()