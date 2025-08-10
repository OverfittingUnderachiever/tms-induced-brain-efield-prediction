"""
Unit tests for the verify_data_structure function in run_tms_simulation.py
and the _derive_paths method in TMSDataLoader.

These tests verify the updated path resolution logic for the new flatter directory structure.
"""

import sys
import os
import unittest
from unittest import mock
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the verify_data_structure function directly from run_tms_simulation.py
# This assumes run_tms_simulation.py is at the project root
from tms_efield_prediction.run_tms_simulation import verify_data_structure
# Import TMSDataLoader from the appropriate package
from tms_efield_prediction.data.pipeline.loader import TMSDataLoader


class MockFileSystem:
    """Mock file system for testing path resolution."""
    
    def __init__(self, structure_type="new"):
        """
        Initialize mock file system with either new or old structure.
        
        Args:
            structure_type: "new" for flat structure, "old" for nested structure
        """
        self.structure_type = structure_type
        self.files = {}
        self.directories = {}
        
        # Set up common directories
        self.directories.update({
            "/home/freyhe/MA_Henry": True,
            "/home/freyhe/MA_Henry/data": True,
            "/home/freyhe/MA_Henry/data/sub-001": True,
            "/home/freyhe/MA_Henry/data/sub-001/experiment": True,
            "/home/freyhe/MA_Henry/data/sub-001/headmodel": True,
        })
        
        # Set up structure-specific directories and files
        if structure_type == "new":
            self._setup_new_structure()
        else:
            self._setup_old_structure()
    
    def _setup_new_structure(self):
        """Set up files and directories for new flatter structure."""
        # Add experiment/all directory
        self.directories["/home/freyhe/MA_Henry/data/sub-001/experiment/all"] = True
        
        # Add files in new structure
        self.files.update({
            # Key files in experiment/all
            "/home/freyhe/MA_Henry/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh": True,
            "/home/freyhe/MA_Henry/data/sub-001/experiment/all/sub-001_middle_gray_matter_roi.msh": True,
            "/home/freyhe/MA_Henry/data/sub-001/experiment/all/sub-001_matsimnibs.mat": True,
            "/home/freyhe/MA_Henry/data/sub-001/experiment/all/sub-001_middle_gray_matter_efields.mat": True,
            
            # ROI center in experiment directory
            "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_roi_center.mat": True,
            
            # Alternative location for ROI init
            "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_roi_init.mat": True,
        })
    
    def _setup_old_structure(self):
        """Set up files and directories for old nested structure."""
        # Add nested directories in old structure
        self.directories.update({
            "/home/freyhe/MA_Henry/data/sub-001/experiment/high_resolution": True,
            "/home/freyhe/MA_Henry/data/sub-001/experiment/high_resolution/efield_results": True,
            "/home/freyhe/MA_Henry/data/sub-001/experiment/high_resolution/simulations": True,
        })
        
        # Add files in old structure
        self.files.update({
            # Mesh in headmodel
            "/home/freyhe/MA_Henry/data/sub-001/headmodel/sub-001.msh": True,
            "/home/freyhe/MA_Henry/data/sub-001/headmodel/sub-001_middle_gray_matter_roi.msh": True,
            
            # ROI center and matsimnibs in experiment directory
            "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_roi_center.mat": True,
            "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_matsimnibs.npy": True,
            "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_efields.npy": True,
        })
    
    def exists(self, path):
        """Mock os.path.exists function"""
        norm_path = self._normalize_path(path)
        return self.files.get(norm_path, False) or self.directories.get(norm_path, False)
    
    def isdir(self, path):
        """Mock os.path.isdir function"""
        norm_path = self._normalize_path(path)
        return self.directories.get(norm_path, False)
    
    def isfile(self, path):
        """Mock os.path.isfile function"""
        norm_path = self._normalize_path(path)
        return self.files.get(norm_path, False)
    
    def listdir(self, path):
        """Mock os.listdir function"""
        norm_path = self._normalize_path(path)
        if not self.directories.get(norm_path, False):
            raise FileNotFoundError(f"Directory not found: {path}")
        
        # Get all files and directories that start with this path
        children = []
        for item_path in list(self.files.keys()) + list(self.directories.keys()):
            if item_path.startswith(norm_path + "/") and "/" not in item_path[len(norm_path)+1:]:
                children.append(os.path.basename(item_path))
        
        return children
    
    def _normalize_path(self, path):
        """Normalize path to handle both absolute and relative paths"""
        # Convert to absolute path if relative
        if not path.startswith("/"):
            path = os.path.join("/home/freyhe/MA_Henry", path)
        
        # Remove double slashes and trailing slash
        path = path.replace("//", "/")
        if path.endswith("/") and path != "/":
            path = path[:-1]
            
        return path


@pytest.fixture
def mock_new_structure(monkeypatch):
    """Set up mock filesystem with new structure"""
    fs = MockFileSystem(structure_type="new")
    
    # Mock os.path.exists, os.path.isdir, os.path.isfile, and os.listdir
    monkeypatch.setattr(os.path, "exists", fs.exists)
    monkeypatch.setattr(os.path, "isdir", fs.isdir)
    monkeypatch.setattr(os.path, "isfile", fs.isfile)
    monkeypatch.setattr(os, "listdir", fs.listdir)
    
    # Mock open for testing file writes
    mock_open = mock.mock_open()
    monkeypatch.setattr("builtins.open", mock_open)
    
    # Mock os.makedirs
    monkeypatch.setattr(os, "makedirs", lambda path, exist_ok=True: None)
    
    # Mock os.remove
    monkeypatch.setattr(os, "remove", lambda path: None)
    
    return fs


@pytest.fixture
def mock_old_structure(monkeypatch):
    """Set up mock filesystem with old structure"""
    fs = MockFileSystem(structure_type="old")
    
    # Mock os.path.exists, os.path.isdir, os.path.isfile, and os.listdir
    monkeypatch.setattr(os.path, "exists", fs.exists)
    monkeypatch.setattr(os.path, "isdir", fs.isdir)
    monkeypatch.setattr(os.path, "isfile", fs.isfile)
    monkeypatch.setattr(os, "listdir", fs.listdir)
    
    # Mock open for testing file writes
    mock_open = mock.mock_open()
    monkeypatch.setattr("builtins.open", mock_open)
    
    # Mock os.makedirs
    monkeypatch.setattr(os, "makedirs", lambda path, exist_ok=True: None)
    
    # Mock os.remove
    monkeypatch.setattr(os, "remove", lambda path: None)
    
    return fs


# Mock TMSPipelineContext for testing the loader
class MockTMSPipelineContext:
    def __init__(self, subject_id, data_root_path):
        self.subject_id = subject_id
        self.data_root_path = data_root_path
        self.debug_mode = False


@mock.patch("logging.Logger.log")  # Mock logger to avoid log output in tests
@mock.patch("logging.Logger.info")
@mock.patch("logging.Logger.warning")
@mock.patch("logging.Logger.error")
class TestVerifyDataStructureWithNewFormat:
    """Test verify_data_structure with the new directory structure."""
    
    def test_verify_data_structure_new(self, mock_error, mock_warning, mock_info, 
                                       mock_log, mock_new_structure):
        """
        Test verify_data_structure function with new structure.
        
        This test verifies that the function correctly identifies files in the
        new flatter directory structure with files in experiment/all.
        """
        # Run verify_data_structure with the new structure
        paths = verify_data_structure("/home/freyhe/MA_Henry", "001")
        
        # Check that paths were correctly identified
        assert paths is not None, "verify_data_structure should return paths dict"
        
        # Check specific paths in the new structure
        assert paths["mesh"] == "/home/freyhe/MA_Henry/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh"
        assert paths["roi_center"] == "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_roi_center.mat"
        assert paths["mesh_roi"] == "/home/freyhe/MA_Henry/data/sub-001/experiment/all/sub-001_middle_gray_matter_roi.msh"
        
        # Check that output directory is correctly set
        assert paths["output_dir"] == "/home/freyhe/MA_Henry/data/sub-001/experiment/simulation_results"
        
        # No warnings or errors should be logged
        assert mock_error.call_count == 0, "No errors should be logged"
    
    def test_tms_data_loader_derive_paths_new(self, mock_error, mock_warning, mock_info, 
                                             mock_log, mock_new_structure):
        """
        Test TMSDataLoader._derive_paths with new structure.
        
        This test verifies that the method correctly identifies files in the
        new flatter directory structure with files in experiment/all.
        """
        # Create TMSDataLoader with mock context
        context = MockTMSPipelineContext("001", "/home/freyhe/MA_Henry/data")
        loader = TMSDataLoader(context=context)
        
        # Check that paths were correctly derived
        assert loader.mesh_file == "/home/freyhe/MA_Henry/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh"
        assert loader.roi_center_file == "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_roi_center.mat"
        assert loader.matsimnibs_file == "/home/freyhe/MA_Henry/data/sub-001/experiment/all/sub-001_matsimnibs.mat"
        assert loader.efields_file == "/home/freyhe/MA_Henry/data/sub-001/experiment/all/sub-001_middle_gray_matter_efields.mat"


@mock.patch("logging.Logger.log")  # Mock logger to avoid log output in tests
@mock.patch("logging.Logger.info")
@mock.patch("logging.Logger.warning")
@mock.patch("logging.Logger.error")
class TestVerifyDataStructureWithOldFormat:
    """Test verify_data_structure with the old directory structure."""
    
    def test_verify_data_structure_old(self, mock_error, mock_warning, mock_info, 
                                      mock_log, mock_old_structure):
        """
        Test verify_data_structure function with old structure.
        
        This test verifies that the function correctly falls back to the old
        nested directory structure when experiment/all is not present.
        """
        # Run verify_data_structure with the old structure
        paths = verify_data_structure("/home/freyhe/MA_Henry", "001")
        
        # Check that paths were correctly identified
        assert paths is not None, "verify_data_structure should return paths dict"
        
        # Check specific paths in the old structure
        assert paths["mesh"] == "/home/freyhe/MA_Henry/data/sub-001/headmodel/sub-001.msh"
        assert paths["roi_center"] == "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_roi_center.mat"
        assert paths["mesh_roi"] == "/home/freyhe/MA_Henry/data/sub-001/headmodel/sub-001_middle_gray_matter_roi.msh"
        
        # Check that output directory is correctly set
        assert paths["output_dir"] == "/home/freyhe/MA_Henry/data/sub-001/experiment/simulation_results"
        
        # Warning should be logged about using old structure
        assert mock_warning.call_count > 0, "Should warn about using old directory structure"
    
    def test_tms_data_loader_derive_paths_old(self, mock_error, mock_warning, mock_info, 
                                            mock_log, mock_old_structure):
        """
        Test TMSDataLoader._derive_paths with old structure.
        
        This test verifies that the method correctly falls back to the old
        nested directory structure when experiment/all is not present.
        """
        # Create TMSDataLoader with mock context
        context = MockTMSPipelineContext("001", "/home/freyhe/MA_Henry/data")
        loader = TMSDataLoader(context=context)
        
        # Check that paths were correctly derived
        assert loader.mesh_file == "/home/freyhe/MA_Henry/data/sub-001/headmodel/sub-001.msh"
        assert loader.roi_center_file == "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_roi_center.mat"
        assert loader.matsimnibs_file == "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_matsimnibs.npy"
        assert loader.efields_file == "/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_efields.npy"


class TestMissingFiles:
    """Test handling of missing files in both structures."""
    
    @pytest.fixture
    def mock_missing_files(self, monkeypatch):
        """Set up mock filesystem with missing critical files"""
        fs = MockFileSystem(structure_type="new")
        
        # Remove critical files
        fs.files["/home/freyhe/MA_Henry/data/sub-001/experiment/all/sub-001_middle_gray_matter.msh"] = False
        fs.files["/home/freyhe/MA_Henry/data/sub-001/experiment/sub-001_roi_center.mat"] = False
        
        # Mock os.path.exists
        monkeypatch.setattr(os.path, "exists", fs.exists)
        
        # Mock os.path.isdir and os.path.isfile
        monkeypatch.setattr(os.path, "isdir", fs.isdir)
        monkeypatch.setattr(os.path, "isfile", fs.isfile)
        
        # Mock os.listdir
        monkeypatch.setattr(os, "listdir", fs.listdir)
        
        return fs
    
    @mock.patch("logging.Logger.error")
    def test_missing_critical_files(self, mock_error, mock_missing_files):
        """Test that verify_data_structure handles missing critical files correctly."""
        # Run verify_data_structure with missing files
        paths = verify_data_structure("/home/freyhe/MA_Henry", "001")
        
        # Should return None when critical files are missing
        assert paths is None, "verify_data_structure should return None when critical files are missing"
        
        # Error should be logged
        assert mock_error.call_count > 0, "Error should be logged when files are missing"


if __name__ == "__main__":
    # Can also run with pytest directly
    pytest.main(["-xvs", __file__])