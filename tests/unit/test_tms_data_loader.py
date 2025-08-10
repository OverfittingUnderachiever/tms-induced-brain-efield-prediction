#!/usr/bin/env python3
"""
Unit tests for TMSDataLoader path resolution.

This tests the updated _derive_paths method in TMSDataLoader to ensure it 
correctly resolves paths in both new and old directory structures.
"""

import sys
import os
import unittest
import pytest
import numpy as np

from unittest import mock
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import from tms_efield_prediction
from tms_efield_prediction.data.pipeline.loader import TMSDataLoader
from tms_efield_prediction.utils.state.context import TMSPipelineContext
from tms_efield_prediction.data.pipeline.tms_data_types import TMSRawData


# Mock classes for testing
class MockDebugHook:
    """Mock DebugHook for testing"""
    
    def __init__(self, should_sample_result=True):
        self._should_sample_result = should_sample_result
    
    def should_sample(self):
        return self._should_sample_result
    
    def record_state(self, name, data):
        pass
    
    def record_event(self, name, data):
        pass
    
    def record_error(self, name, data):
        pass


class MockResourceMonitor:
    """Mock ResourceMonitor for testing"""
    
    def update_component_usage(self, component_name, action):
        pass


class MockMeshData:
    """Mock MeshData for testing"""
    
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.node_data = {}
        self.element_data = {}
        self.metadata = {}


class MockFileOpener:
    """Mock file opener that returns an h5py-like object"""
    
    def __init__(self, contents=None):
        self.contents = contents or {"roi_center": np.zeros((3,))}
        self.closed = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True
        return False
    
    def __getitem__(self, key):
        if key in self.contents:
            return self.contents[key]
        raise KeyError(f"Key {key} not found")


class TestTMSDataLoader(unittest.TestCase):
    """Test the TMSDataLoader path resolution logic"""
    
    def setUp(self):
        """Set up common test fixtures"""
        # Create a context for testing
        self.context = TMSPipelineContext(
            dependencies={},
            config={},
            pipeline_mode="dual_modal",
            experiment_phase="preprocessing",
            debug_mode=True,
            subject_id="001",
            data_root_path="/data",
            coil_file_path="/coil/coil.ccd",
            stacking_mode="simple",
            normalization_method="zero_one",
            output_shape=(128, 128, 128),
            dadt_scaling_factor=1.0
        )
        
        # Create debug hook and resource monitor
        self.debug_hook = MockDebugHook()
        self.resource_monitor = MockResourceMonitor()

    def test_derive_paths_new_structure(self):
        """Test _derive_paths with new directory structure"""
        # Mock file existence for new structure
        with unittest.mock.patch("os.path.exists") as mock_exists:
            # Configure which paths exist
            def exists_side_effect(path):
                path = os.path.normpath(path)
                # Directories
                if path in ["/data/sub-001", "/data/sub-001/experiment", 
                            "/data/sub-001/experiment/all", "/data/sub-001/headmodel"]:
                    return True
                # Files in new structure
                if path in ["/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh",
                            "/data/sub-001/experiment/all/sub-001_matsimnibs.mat",
                            "/data/sub-001/experiment/all/sub-001_middle_gray_matter_efields.mat",
                            "/data/sub-001/experiment/all/dAdts.h5",
                            "/data/sub-001/experiment/sub-001_roi_center.mat"]:
                    return True
                return False
            
            mock_exists.side_effect = exists_side_effect
            
            # Create loader
            loader = TMSDataLoader(
                context=self.context,
                debug_hook=self.debug_hook,
                resource_monitor=self.resource_monitor
            )
            
            # Check paths
            self.assertEqual(
                os.path.normpath(loader.mesh_file), 
                os.path.normpath("/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh")
            )
            self.assertEqual(
                os.path.normpath(loader.roi_center_file), 
                os.path.normpath("/data/sub-001/experiment/sub-001_roi_center.mat")
            )
            self.assertEqual(
                os.path.normpath(loader.matsimnibs_file), 
                os.path.normpath("/data/sub-001/experiment/all/sub-001_matsimnibs.mat")
            )
            self.assertEqual(
                os.path.normpath(loader.efields_file), 
                os.path.normpath("/data/sub-001/experiment/all/sub-001_middle_gray_matter_efields.mat")
            )
            self.assertEqual(
                os.path.normpath(loader.dadt_file), 
                os.path.normpath("/data/sub-001/experiment/all/dAdts.h5")
            )

    def test_derive_paths_old_structure(self):
        """Test _derive_paths with old directory structure"""
        # Mock file existence for old structure
        with unittest.mock.patch("os.path.exists") as mock_exists:
            # Configure which paths exist
            def exists_side_effect(path):
                path = os.path.normpath(path)
                # Directories
                if path in ["/data/sub-001", "/data/sub-001/experiment", "/data/sub-001/headmodel"]:
                    return True
                # "all" directory doesn't exist in old structure
                if path == "/data/sub-001/experiment/all":
                    return False
                # Files in old structure
                if path in ["/data/sub-001/headmodel/sub-001.msh",
                            "/data/sub-001/experiment/sub-001_roi_center.mat",
                            "/data/sub-001/experiment/sub-001_matsimnibs.npy",
                            "/data/sub-001/experiment/sub-001_efields.npy",
                            "/data/sub-001/experiment/dAdts.h5"]:
                    return True
                return False
            
            mock_exists.side_effect = exists_side_effect
            
            # Create loader
            loader = TMSDataLoader(
                context=self.context,
                debug_hook=self.debug_hook,
                resource_monitor=self.resource_monitor
            )
            
            # Check paths
            self.assertEqual(
                os.path.normpath(loader.mesh_file), 
                os.path.normpath("/data/sub-001/headmodel/sub-001.msh")
            )
            self.assertEqual(
                os.path.normpath(loader.roi_center_file), 
                os.path.normpath("/data/sub-001/experiment/sub-001_roi_center.mat")
            )
            self.assertEqual(
                os.path.normpath(loader.matsimnibs_file), 
                os.path.normpath("/data/sub-001/experiment/sub-001_matsimnibs.npy")
            )
            self.assertEqual(
                os.path.normpath(loader.efields_file), 
                os.path.normpath("/data/sub-001/experiment/sub-001_efields.npy")
            )
            self.assertEqual(
                os.path.normpath(loader.dadt_file), 
                os.path.normpath("/data/sub-001/experiment/dAdts.h5")
            )

    @unittest.mock.patch("h5py.File")
    @unittest.mock.patch("tms_efield_prediction.data.formats.simnibs_io.load_dadt_data")
    @unittest.mock.patch("tms_efield_prediction.data.formats.simnibs_io.load_matsimnibs")
    @unittest.mock.patch("tms_efield_prediction.data.formats.simnibs_io.load_mesh")
    @unittest.mock.patch("simnibs.msh.mesh_io.read_msh")
    @unittest.mock.patch("os.path.exists")
    def test_load_raw_data_new_structure(self, mock_exists, mock_read_msh, mock_load_mesh, 
                                        mock_load_matsimnibs, mock_load_dadt, mock_h5py_file):
        """Test load_raw_data with new directory structure"""
        # Configure which paths exist
        def exists_side_effect(path):
            path = os.path.normpath(path)
            # Directories
            if path in ["/data/sub-001", "/data/sub-001/experiment", 
                        "/data/sub-001/experiment/all", "/data/sub-001/headmodel"]:
                return True
            # Files in new structure
            if path in ["/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh",
                        "/data/sub-001/experiment/all/sub-001_matsimnibs.mat",
                        "/data/sub-001/experiment/all/sub-001_middle_gray_matter_efields.mat",
                        "/data/sub-001/experiment/all/dAdts.h5",
                        "/data/sub-001/experiment/sub-001_roi_center.mat"]:
                return True
            return False
        
        mock_exists.side_effect = exists_side_effect
        
        # Set up mocks
        mock_read_msh.return_value = unittest.mock.MagicMock()
        mock_load_mesh.return_value = MockMeshData()
        mock_load_matsimnibs.return_value = np.zeros((5, 4, 4))
        mock_load_dadt.return_value = np.zeros((5, 128, 128, 128, 3))
        mock_h5py_file.return_value = MockFileOpener()
        
        # Create loader
        loader = TMSDataLoader(
            context=self.context,
            debug_hook=self.debug_hook,
            resource_monitor=self.resource_monitor
        )
        
        # Load raw data
        raw_data = loader.load_raw_data()
        
        # Verify raw data
        self.assertIsInstance(raw_data, TMSRawData)
        self.assertEqual(raw_data.subject_id, "001")
        self.assertIsNotNone(raw_data.mri_mesh)
        self.assertIsNotNone(raw_data.coil_positions)
        
        # Verify that load_mesh was called with correct path
        mock_load_mesh.assert_called_once()
        mesh_path_arg = mock_load_mesh.call_args[0][0]
        self.assertEqual(
            os.path.normpath(mesh_path_arg), 
            os.path.normpath("/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh")
        )
        
        # Verify that load_matsimnibs was called with correct path
        mock_load_matsimnibs.assert_called_once()
        matsimnibs_path_arg = mock_load_matsimnibs.call_args[0][0]
        self.assertEqual(
            os.path.normpath(matsimnibs_path_arg), 
            os.path.normpath("/data/sub-001/experiment/all/sub-001_matsimnibs.mat")
        )


if __name__ == "__main__":
    unittest.main()