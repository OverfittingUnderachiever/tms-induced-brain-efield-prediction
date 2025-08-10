"""
Unit tests for the TMSDataLoader path resolution logic.

Tests the updated _derive_paths method in TMSDataLoader to ensure it
correctly resolves paths in both new flatter directory structure and
the old nested structure.
"""

import sys
import os
import unittest
from unittest import mock
import pytest
import h5py
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the TMSDataLoader class
from tms_efield_prediction.utils.state.context import TMSPipelineContext, PipelineState
from tms_efield_prediction.utils.debug.hooks import DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.data.pipeline.loader import TMSDataLoader
from tms_efield_prediction.data.pipeline.tms_data_types import TMSRawData, TMSProcessedData


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


@pytest.fixture
def mock_new_structure_filesystem(monkeypatch):
    """Set up mock filesystem with new structure"""
    # Dictionary of files that should exist
    files_exist = {
        # New structure files
        "/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh": True,
        "/data/sub-001/experiment/all/sub-001_matsimnibs.mat": True,
        "/data/sub-001/experiment/all/sub-001_middle_gray_matter_efields.mat": True,
        "/data/sub-001/experiment/all/dAdts.h5": True,
        "/data/sub-001/experiment/sub-001_roi_center.mat": True,
        
        # These old structure files don't exist in new structure
        "/data/sub-001/headmodel/sub-001.msh": False,
        "/data/sub-001/experiment/sub-001_matsimnibs.npy": False,
        "/data/sub-001/experiment/sub-001_efields.npy": False,
        "/data/sub-001/experiment/dAdts.h5": False,
    }
    
    # Dictionary of directories that should exist
    dirs_exist = {
        "/data": True,
        "/data/sub-001": True,
        "/data/sub-001/experiment": True,
        "/data/sub-001/experiment/all": True,
        "/data/sub-001/headmodel": True,
    }
    
    # Mock os.path.exists
    def mock_exists(path):
        path = os.path.normpath(path)
        if path in files_exist:
            return files_exist[path]
        if path in dirs_exist:
            return dirs_exist[path]
        return False
    
    monkeypatch.setattr(os.path, "exists", mock_exists)
    
    # Mock os.path.isdir
    def mock_isdir(path):
        path = os.path.normpath(path)
        return path in dirs_exist and dirs_exist[path]
    
    monkeypatch.setattr(os.path, "isdir", mock_isdir)
    
    # Mock h5py.File
    monkeypatch.setattr(h5py, "File", MockFileOpener)
    
    # Mock np.load
    monkeypatch.setattr(np, "load", lambda path: np.zeros((10, 10)))
    
    return (files_exist, dirs_exist)


@pytest.fixture
def mock_old_structure_filesystem(monkeypatch):
    """Set up mock filesystem with old structure"""
    # Dictionary of files that should exist
    files_exist = {
        # Old structure files
        "/data/sub-001/headmodel/sub-001.msh": True,
        "/data/sub-001/experiment/sub-001_roi_center.mat": True,
        "/data/sub-001/experiment/sub-001_matsimnibs.npy": True,
        "/data/sub-001/experiment/sub-001_efields.npy": True,
        "/data/sub-001/experiment/dAdts.h5": True,
        
        # These new structure files don't exist in old structure
        "/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh": False,
        "/data/sub-001/experiment/all/sub-001_matsimnibs.mat": False,
        "/data/sub-001/experiment/all/sub-001_middle_gray_matter_efields.mat": False,
        "/data/sub-001/experiment/all/dAdts.h5": False,
    }
    
    # Dictionary of directories that should exist
    dirs_exist = {
        "/data": True,
        "/data/sub-001": True,
        "/data/sub-001/experiment": True,
        "/data/sub-001/experiment/all": False,  # This is the key difference!
        "/data/sub-001/headmodel": True,
    }
    
    # Mock os.path.exists
    def mock_exists(path):
        path = os.path.normpath(path)
        if path in files_exist:
            return files_exist[path]
        if path in dirs_exist:
            return dirs_exist[path]
        return False
    
    monkeypatch.setattr(os.path, "exists", mock_exists)
    
    # Mock os.path.isdir
    def mock_isdir(path):
        path = os.path.normpath(path)
        return path in dirs_exist and dirs_exist[path]
    
    monkeypatch.setattr(os.path, "isdir", mock_isdir)
    
    # Mock h5py.File
    monkeypatch.setattr(h5py, "File", MockFileOpener)
    
    # Mock np.load
    monkeypatch.setattr(np, "load", lambda path: np.zeros((10, 10)))
    
    return (files_exist, dirs_exist)


@pytest.fixture
def mock_debug_hook():
    """Create a mock debug hook"""
    return MockDebugHook()


@pytest.fixture
def mock_resource_monitor():
    """Create a mock resource monitor"""
    return MockResourceMonitor()


@pytest.fixture
def tms_pipeline_context():
    """Create a TMSPipelineContext for testing"""
    return TMSPipelineContext(
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


class TestTMSDataLoaderPaths:
    """Test the TMSDataLoader path resolution logic"""
    
    def test_derive_paths_new_structure(self, mock_new_structure_filesystem, 
                                        tms_pipeline_context, mock_debug_hook, 
                                        mock_resource_monitor):
        """Test _derive_paths with new directory structure"""
        loader = TMSDataLoader(
            context=tms_pipeline_context,
            debug_hook=mock_debug_hook,
            resource_monitor=mock_resource_monitor
        )
        
        # Check that paths were correctly derived for new structure
        assert os.path.normpath(loader.experiment_path) == os.path.normpath("/data/sub-001/experiment")
        assert os.path.normpath(loader.experiment_all_path) == os.path.normpath("/data/sub-001/experiment/all")
        assert os.path.normpath(loader.headmodel_path) == os.path.normpath("/data/sub-001/headmodel")
        
        # Check specific file paths are correctly resolved
        assert os.path.normpath(loader.mesh_file) == os.path.normpath("/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh")
        assert os.path.normpath(loader.roi_center_file) == os.path.normpath("/data/sub-001/experiment/sub-001_roi_center.mat")
        assert os.path.normpath(loader.matsimnibs_file) == os.path.normpath("/data/sub-001/experiment/all/sub-001_matsimnibs.mat")
        assert os.path.normpath(loader.efields_file) == os.path.normpath("/data/sub-001/experiment/all/sub-001_middle_gray_matter_efields.mat")
        assert os.path.normpath(loader.dadt_file) == os.path.normpath("/data/sub-001/experiment/all/dAdts.h5")
    
    def test_derive_paths_old_structure(self, mock_old_structure_filesystem, 
                                       tms_pipeline_context, mock_debug_hook, 
                                       mock_resource_monitor):
        """Test _derive_paths with old directory structure"""
        loader = TMSDataLoader(
            context=tms_pipeline_context,
            debug_hook=mock_debug_hook,
            resource_monitor=mock_resource_monitor
        )
        
        # Check that paths were correctly derived for old structure
        assert os.path.normpath(loader.experiment_path) == os.path.normpath("/data/sub-001/experiment")
        # experiment_all_path should be set but the directory doesn't exist
        assert os.path.normpath(loader.experiment_all_path) == os.path.normpath("/data/sub-001/experiment/all")
        assert os.path.normpath(loader.headmodel_path) == os.path.normpath("/data/sub-001/headmodel")
        
        # Check specific file paths are correctly resolved - should fall back to old structure
        assert os.path.normpath(loader.mesh_file) == os.path.normpath("/data/sub-001/headmodel/sub-001.msh")
        assert os.path.normpath(loader.roi_center_file) == os.path.normpath("/data/sub-001/experiment/sub-001_roi_center.mat")
        assert os.path.normpath(loader.matsimnibs_file) == os.path.normpath("/data/sub-001/experiment/sub-001_matsimnibs.npy")
        assert os.path.normpath(loader.efields_file) == os.path.normpath("/data/sub-001/experiment/sub-001_efields.npy")
        assert os.path.normpath(loader.dadt_file) == os.path.normpath("/data/sub-001/experiment/dAdts.h5")
    
    @mock.patch("logging.Logger.info")
    @mock.patch("logging.Logger.warning")
    def test_load_raw_data_new_structure(self, mock_warning, mock_info, 
                                        mock_new_structure_filesystem, 
                                        tms_pipeline_context, mock_debug_hook, 
                                        mock_resource_monitor):
        """Test load_raw_data with new directory structure"""
        # Mock the file loading functions
        with mock.patch("tms_efield_prediction.data.formats.simnibs_io.load_mesh") as mock_load_mesh, \
             mock.patch("tms_efield_prediction.data.formats.simnibs_io.load_matsimnibs") as mock_load_matsimnibs, \
             mock.patch("tms_efield_prediction.data.formats.simnibs_io.load_dadt_data") as mock_load_dadt:
            
            # Set up return values
            mock_load_mesh.return_value = MockMeshData()
            mock_load_matsimnibs.return_value = np.zeros((5, 4, 4))
            mock_load_dadt.return_value = np.zeros((5, 128, 128, 128, 3))
            
            # Create loader
            loader = TMSDataLoader(
                context=tms_pipeline_context,
                debug_hook=mock_debug_hook,
                resource_monitor=mock_resource_monitor
            )
            
            # Load raw data
            raw_data = loader.load_raw_data()
            
            # Verify raw data
            assert isinstance(raw_data, TMSRawData)
            assert raw_data.subject_id == "001"
            assert raw_data.mri_mesh is not None
            assert raw_data.coil_positions is not None
            assert raw_data.roi_center is not None
            
            # Verify that load_mesh was called with correct path
            mock_load_mesh.assert_called_once()
            mesh_path_arg = mock_load_mesh.call_args[0][0]
            assert os.path.normpath(mesh_path_arg) == os.path.normpath("/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh")
            
            # Verify that load_matsimnibs was called with correct path
            mock_load_matsimnibs.assert_called_once()
            matsimnibs_path_arg = mock_load_matsimnibs.call_args[0][0]
            assert os.path.normpath(matsimnibs_path_arg) == os.path.normpath("/data/sub-001/experiment/all/sub-001_matsimnibs.mat")
    
    @mock.patch("logging.Logger.info")
    @mock.patch("logging.Logger.warning")
    def test_load_raw_data_old_structure(self, mock_warning, mock_info, 
                                       mock_old_structure_filesystem, 
                                       tms_pipeline_context, mock_debug_hook, 
                                       mock_resource_monitor):
        """Test load_raw_data with old directory structure"""
        # Mock the file loading functions
        with mock.patch("tms_efield_prediction.data.formats.simnibs_io.load_mesh") as mock_load_mesh, \
             mock.patch("tms_efield_prediction.data.formats.simnibs_io.load_matsimnibs") as mock_load_matsimnibs, \
             mock.patch("tms_efield_prediction.data.formats.simnibs_io.load_dadt_data") as mock_load_dadt:
            
            # Set up return values
            mock_load_mesh.return_value = MockMeshData()
            mock_load_matsimnibs.return_value = np.zeros((5, 4, 4))
            mock_load_dadt.return_value = np.zeros((5, 128, 128, 128, 3))
            
            # Create loader
            loader = TMSDataLoader(
                context=tms_pipeline_context,
                debug_hook=mock_debug_hook,
                resource_monitor=mock_resource_monitor
            )
            
            # Load raw data
            raw_data = loader.load_raw_data()
            
            # Verify raw data
            assert isinstance(raw_data, TMSRawData)
            assert raw_data.subject_id == "001"
            assert raw_data.mri_mesh is not None
            assert raw_data.coil_positions is not None
            assert raw_data.roi_center is not None
            
            # Verify that load_mesh was called with correct path
            mock_load_mesh.assert_called_once()
            mesh_path_arg = mock_load_mesh.call_args[0][0]
            assert os.path.normpath(mesh_path_arg) == os.path.normpath("/data/sub-001/headmodel/sub-001.msh")
            
            # Verify that load_matsimnibs was called with correct path
            mock_load_matsimnibs.assert_called_once()
            matsimnibs_path_arg = mock_load_matsimnibs.call_args[0][0]
            assert os.path.normpath(matsimnibs_path_arg) == os.path.normpath("/data/sub-001/experiment/sub-001_matsimnibs.npy")


if __name__ == "__main__":
    # Can also run with pytest directly
    pytest.main(["-xvs", __file__])