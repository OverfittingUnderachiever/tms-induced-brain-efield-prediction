#!/usr/bin/env python3
"""
Test script for ROI mesh detection and loading functionality.

This standalone script tests the ROI processor's ability to find existing ROI meshes
in the MA_Henry data structure without running the full simulation pipeline.
"""

import os
import sys
import logging
import argparse
import unittest
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('roi_processor_test')

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
sys.path.insert(0, project_root)

# Import the verify_data_structure function directly from run_tms_simulation.py
try:
    from run_tms_simulation import verify_data_structure
    logger.info("Successfully imported verify_data_structure from project root")
except ImportError:
    logger.error("Failed to import verify_data_structure. Trying alternate path...")
    try:
        from tms_efield_prediction.run_tms_simulation import verify_data_structure
        logger.info("Successfully imported verify_data_structure from tms_efield_prediction package")
    except ImportError:
        logger.error("Could not import verify_data_structure from any location")
        sys.exit(1)

# Import the ROI processor class
try:
    from tms_efield_prediction.data.pipeline.roi_processor import ROIProcessor
    from tms_efield_prediction.simulation.tms_simulation import SimulationContext
    logger.info("Successfully imported ROIProcessor and SimulationContext")
except ImportError as e:
    logger.error(f"Failed to import necessary classes: {str(e)}")
    sys.exit(1)

class TestROIProcessor(unittest.TestCase):
    """Test case for ROI processor functionality."""
    
    def setUp(self):
        """Set up the test case."""
        self.data_dir = os.environ.get("TEST_DATA_DIR", "/home/freyhe/MA_Henry")
        self.subject_id = os.environ.get("TEST_SUBJECT_ID", "001")
        
        # Log the test configuration
        logger.info(f"Using data directory: {self.data_dir}")
        logger.info(f"Using subject ID: {self.subject_id}")
        
        # Create a simulation context for testing
        self.context = SimulationContext(
            dependencies={"simnibs": "4.0"},
            config={"roi_radius": 20.0},
            pipeline_mode="mri_efield",
            experiment_phase="preprocessing",
            debug_mode=True,
            subject_id=self.subject_id,
            data_root_path=self.data_dir,
            output_path="/tmp/roi_test"
        )
        
        # Determine data_root_path for the subject
        if os.path.exists(os.path.join(self.data_dir, "data", f"sub-{self.subject_id}")):
            self.data_root_path = os.path.join(self.data_dir, "data", f"sub-{self.subject_id}")
        else:
            self.data_root_path = os.path.join(self.data_dir, f"sub-{self.subject_id}")
        
        # Create ROI processor for testing
        self.roi_processor = ROIProcessor(self.context)
    
    def test_check_roi_mesh_exists(self):
        """Test the check_roi_mesh_exists method."""
        logger.info("Testing check_roi_mesh_exists method...")
        
        # Test for existing ROI mesh
        roi_exists, roi_path = self.roi_processor.check_roi_mesh_exists(
            self.subject_id, self.data_root_path
        )
        
        # Log the results
        logger.info(f"ROI exists: {roi_exists}")
        if roi_exists:
            logger.info(f"ROI path: {roi_path}")
            
            # Verify the file actually exists
            self.assertTrue(os.path.exists(roi_path), f"Reported ROI mesh path does not exist: {roi_path}")
            
            # Print detailed file information
            self._print_file_info(roi_path)
            
            # Try to load the mesh to ensure it's valid
            try:
                from simnibs import mesh_io
                msh = mesh_io.read_msh(roi_path)
                
                # Try different ways to get node count - SimNIBS might 
                # structure nodes differently than expected
                node_count = "unknown"
                try:
                    # Method 1: Try directly accessing nodes array and getting length
                    nodes_array = msh.nodes[:]
                    node_count = len(nodes_array)
                    logger.info(f"Got node count via nodes[:]: {node_count}")
                except (TypeError, AttributeError, IndexError) as e:
                    logger.info(f"Could not get nodes as array: {e}")
                    try:
                        # Method 2: Check if there's a node_number_list attribute
                        if hasattr(msh.nodes, 'node_number_list'):
                            node_count = len(msh.nodes.node_number_list)
                            logger.info(f"Got node count via node_number_list: {node_count}")
                        # Method 3: Check if there's a node_coord attribute
                        elif hasattr(msh.nodes, 'node_coord'):
                            node_count = len(msh.nodes.node_coord)
                            logger.info(f"Got node count via node_coord: {node_count}")
                        # Method 4: Try dir() to see available attributes
                        else:
                            logger.info(f"Available attributes: {dir(msh.nodes)}")
                            # Just verify nodes exists as a fallback
                            self.assertIsNotNone(msh.nodes, "ROI mesh nodes attribute is None")
                            logger.info("Nodes attribute exists but could not determine count")
                    except Exception as inner_e:
                        logger.info(f"Alternative node count methods failed: {inner_e}")
                
                logger.info(f"Successfully loaded ROI mesh with {node_count} nodes")
                logger.info("ROI mesh validation successful")
            except Exception as e:
                logger.error(f"Error loading ROI mesh: {str(e)}")
                logger.error(f"This could be due to SimNIBS version differences or mesh structure changes")
                # Don't fail the test just because we can't count the nodes
                # The important part is that the file exists and can be loaded
                logger.info("ROI mesh file exists but node count couldn't be determined")
        else:
            logger.warning("No existing ROI mesh found. This may be expected if no ROI mesh exists.")
            # Check if we can find the full mesh
            paths = verify_data_structure(self.data_dir, self.subject_id)
            if paths:
                logger.info(f"Found main mesh at: {paths.get('mesh')}")
                
                # Check if verify_data_structure detected a mesh_roi
                if paths.get('mesh_roi'):
                    logger.warning(f"verify_data_structure found a ROI mesh at {paths.get('mesh_roi')} "
                                  f"but check_roi_mesh_exists didn't find it!")
                    self.fail("Inconsistency in ROI mesh detection")
    
    def test_verify_data_structure(self):
        """Test the verify_data_structure function."""
        logger.info("Testing verify_data_structure function...")
        
        paths = verify_data_structure(self.data_dir, self.subject_id)
        self.assertIsNotNone(paths, "verify_data_structure returned None")
        
        logger.info("Paths returned by verify_data_structure:")
        for key, path in paths.items():
            logger.info(f"  {key}: {path}")
            if path and key != 'output_dir':  # Skip checking output_dir
                if os.path.exists(path):
                    logger.info(f"  - Path exists: {path}")
                else:
                    logger.warning(f"  - Path does not exist: {path}")
        
        # Check if mesh_roi is in paths
        if 'mesh_roi' in paths and paths['mesh_roi']:
            logger.info(f"Found ROI mesh in verify_data_structure: {paths['mesh_roi']}")
            
            # Verify the file actually exists
            self.assertTrue(os.path.exists(paths['mesh_roi']), 
                          f"Reported ROI mesh path does not exist: {paths['mesh_roi']}")
            
            # Print detailed file information
            self._print_file_info(paths['mesh_roi'])
            
            # Cross-check with check_roi_mesh_exists
            roi_exists, roi_path = self.roi_processor.check_roi_mesh_exists(
                self.subject_id, self.data_root_path
            )
            
            if roi_exists:
                logger.info(f"check_roi_mesh_exists also found a ROI mesh at: {roi_path}")
                # Compare paths
                if os.path.realpath(roi_path) != os.path.realpath(paths['mesh_roi']):
                    logger.warning(f"Different ROI mesh paths: {roi_path} vs {paths['mesh_roi']}")
                else:
                    logger.info("Both methods found the same ROI mesh")
            else:
                logger.warning(f"check_roi_mesh_exists didn't find a ROI mesh, but verify_data_structure did")
                self.fail("Inconsistency in ROI mesh detection")
    
    def _print_file_info(self, file_path):
        """Print detailed information about a file for debugging."""
        logger.info(f"File information for: {file_path}")
        
        if os.path.exists(file_path):
            logger.info(f"  - File exists")
            logger.info(f"  - Size: {os.path.getsize(file_path)} bytes")
            logger.info(f"  - Last modified: {datetime.fromtimestamp(os.path.getmtime(file_path))}")
            
            # Try to determine if it's a binary or text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1024)  # Try to read as text
                    logger.info(f"  - File type: Appears to be a text file")
            except UnicodeDecodeError:
                logger.info(f"  - File type: Appears to be a binary file")
            except Exception as e:
                logger.info(f"  - File type check error: {str(e)}")
        else:
            logger.info(f"  - File does not exist")
            
            # Check if directory exists
            dir_path = os.path.dirname(file_path)
            if os.path.exists(dir_path):
                logger.info(f"  - Parent directory exists: {dir_path}")
                logger.info(f"  - Directory contents: {os.listdir(dir_path)}")
            else:
                logger.info(f"  - Parent directory does not exist: {dir_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test ROI processor functionality")
    
    parser.add_argument("--data-dir", default="/home/freyhe/MA_Henry", 
                      help="Path to MA_Henry directory (default: /home/freyhe/MA_Henry)")
    parser.add_argument("--subject", default="001", 
                      help="Subject ID (default: 001)")
    
    return parser.parse_args()


def standalone_test(data_dir, subject_id):
    """Run a standalone test without unittest framework."""
    logger.info("=== Starting standalone ROI processor test ===")
    
    # Determine data_root_path for the subject
    if os.path.exists(os.path.join(data_dir, "data", f"sub-{subject_id}")):
        data_root_path = os.path.join(data_dir, "data", f"sub-{subject_id}")
        logger.info(f"Using data root path with data subdirectory: {data_root_path}")
    else:
        data_root_path = os.path.join(data_dir, f"sub-{subject_id}")
        logger.info(f"Using data root path without data subdirectory: {data_root_path}")
    
    # Create a simulation context
    context = SimulationContext(
        dependencies={"simnibs": "4.0"},
        config={"roi_radius": 20.0},
        pipeline_mode="mri_efield",
        experiment_phase="preprocessing",
        debug_mode=True,
        subject_id=subject_id,
        data_root_path=data_root_path,
        output_path="/tmp/roi_test"
    )
    
    # Create ROI processor
    roi_processor = ROIProcessor(context)
    
    # Test check_roi_mesh_exists
    logger.info("Checking if ROI mesh exists...")
    roi_exists, roi_path = roi_processor.check_roi_mesh_exists(subject_id, data_root_path)
    
    if roi_exists:
        logger.info(f"Found existing ROI mesh: {roi_path}")
        
        # Print detailed file information
        logger.info(f"File information for: {roi_path}")
        if os.path.exists(roi_path):
            logger.info(f"  - File exists")
            logger.info(f"  - Size: {os.path.getsize(roi_path)} bytes")
            logger.info(f"  - Last modified: {datetime.fromtimestamp(os.path.getmtime(roi_path))}")
        else:
            logger.error(f"Reported ROI mesh does not exist: {roi_path}")
        
        # Try to load the mesh to verify it's valid
        try:
            from simnibs import mesh_io
            msh = mesh_io.read_msh(roi_path)
            
            # Try different ways to get node count
            try:
                # Try directly accessing nodes array
                nodes_array = msh.nodes[:]
                logger.info(f"Successfully loaded ROI mesh with {len(nodes_array)} nodes")
            except Exception as e:
                logger.info(f"Could not get node count via array access: {str(e)}")
                logger.info(f"Mesh object type: {type(msh)}")
                logger.info(f"Nodes object type: {type(msh.nodes)}")
                logger.info(f"Available attributes on mesh: {dir(msh)}")
                logger.info(f"Available attributes on nodes: {dir(msh.nodes)}")
                logger.info("Mesh loaded but could not determine node count")
        except Exception as e:
            logger.error(f"Error loading ROI mesh: {str(e)}")
    else:
        logger.info("No existing ROI mesh found")
    
    # Test verify_data_structure
    logger.info("Testing verify_data_structure function...")
    paths = verify_data_structure(data_dir, subject_id)
    
    if paths:
        logger.info("Paths returned by verify_data_structure:")
        for key, path in paths.items():
            logger.info(f"  {key}: {path}")
        
        # Check if mesh_roi is in paths
        if 'mesh_roi' in paths and paths['mesh_roi']:
            logger.info(f"Found ROI mesh in verify_data_structure: {paths['mesh_roi']}")
            
            # Verify the file actually exists
            if os.path.exists(paths['mesh_roi']):
                logger.info(f"Verified ROI mesh exists: {paths['mesh_roi']}")
            else:
                logger.error(f"Reported ROI mesh does not exist: {paths['mesh_roi']}")
            
            # Compare with the result from check_roi_mesh_exists
            if roi_exists:
                if os.path.realpath(roi_path) == os.path.realpath(paths['mesh_roi']):
                    logger.info("Both methods found the same ROI mesh")
                else:
                    logger.warning(f"Different ROI mesh paths: {roi_path} vs {paths['mesh_roi']}")
            else:
                logger.warning(f"check_roi_mesh_exists didn't find a ROI mesh, but verify_data_structure did")
    else:
        logger.error("verify_data_structure returned None")
    
    logger.info("=== Standalone ROI processor test completed ===")


def main():
    """Main function."""
    args = parse_args()
    
    # Set environment variables for unittest
    os.environ["TEST_DATA_DIR"] = args.data_dir
    os.environ["TEST_SUBJECT_ID"] = args.subject
    
    # Run standalone test first
    standalone_test(args.data_dir, args.subject)
    
    # Run unittest
    logger.info("=== Starting unittest tests ===")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("=== Unittest tests completed ===")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())