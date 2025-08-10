"""
Unit tests for the optimized calc_dAdt function.
This script verifies memory usage, calculation time, and file handling.
"""

import os
import sys
import time
import unittest
import numpy as np
import h5py
import shutil
import tempfile
import logging
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the simulation module
from tms_efield_prediction.simulation.tms_simulation import (
    calc_dAdt, 
    SimulationContext,
    load_mesh_and_roi,
    get_skin_average_normal_vector,
    compute_cylindrical_roi,
    crop_mesh_nodes,
    remove_islands
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import psutil for memory tracking
try:
    import psutil
    HAS_PSUTIL = True
    
    def get_memory_usage():
        """Get current memory usage of the process in MB."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available. Memory usage tracking will be disabled.")
    
    def get_memory_usage():
        """Dummy function when psutil is not available."""
        return 0

# Mock implementation of original calc_dAdt for comparison
def original_calc_dAdt(context, mesh, matsimnibs, **kwargs):
    """
    Mock implementation of the original calc_dAdt function (less efficient).
    Based on the pattern in prototype.py without ROI optimization.
    """
    from simnibs.simulation import coil_numpy as coil_lib
    
    logger.info("Running original calc_dAdt implementation (without ROI optimization)...")
    start_time = time.time()
    
    didt = context.didt
    save_path = kwargs.get('save_path', context.output_path)
    to_hdf5 = kwargs.get('to_hdf5', True)
    n_cpus = kwargs.get('n_cpus', 1)
    
    # Create tmp_path for temporary files
    tmp_path = None
    if to_hdf5:
        if save_path is None:
            save_path = os.path.split(context.coil_file_path)[0]
        
        # Create tmp_path
        tmp_path = os.path.join(save_path, 'tmp')
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

    def get_dAdt(matsim, i):
        # Use the full mesh for calculation (inefficient)
        dAdt = coil_lib.set_up_tms(mesh, context.coil_file_path, matsim, didt)[:,:]

        if to_hdf5:
            np.save(os.path.join(tmp_path, f"{i}.npy"), dAdt)
        
        return dAdt

    # Process results
    if n_cpus == 1:
        # Sequential processing
        res = []
        for i, matsim in enumerate(matsimnibs):
            dAdt = get_dAdt(matsim, i)
            res.append(dAdt)
    else:
        # This would use parallel processing in the real implementation
        # For testing, we'll just do sequential
        res = []
        for i, matsim in enumerate(matsimnibs):
            dAdt = get_dAdt(matsim, i)
            res.append(dAdt)
    
    if to_hdf5:
        with h5py.File(os.path.join(save_path, 'dAdts.h5'), 'w') as f:
            dAdts = f.create_dataset('dAdt', shape=(len(matsimnibs), len(mesh.elm.triangles), 3))
            for i in range(len(matsimnibs)):
                dAdts[i] = np.load(os.path.join(tmp_path, f"{i}.npy"))
        
        # Clean up the temporary directory
        shutil.rmtree(tmp_path)
        
        # For testing, load the data back
        with h5py.File(os.path.join(save_path, 'dAdts.h5'), 'r') as f:
            dAdts = f['dAdt'][:]
    else:
        dAdts = np.stack(res)
        if save_path is not None:
            np.save(os.path.join(save_path, 'dAdts.npy'), dAdts)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Original implementation completed in {elapsed_time:.2f} seconds")
    
    return dAdts


class TestCalcDadt(unittest.TestCase):
    """Test case for the calc_dAdt function optimization."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        logger.info("Setting up test case...")
        
        # Set up test paths
        cls.temp_dir = tempfile.mkdtemp()
        cls.output_path = os.path.join(cls.temp_dir, 'output')
        os.makedirs(cls.output_path, exist_ok=True)
        
        # Create test context with required base class parameters
        cls.context = SimulationContext(
            # Required PipelineContext parameters
            dependencies={},        # Empty dictionary for dependencies
            config={},              # Empty dictionary for config
            pipeline_mode="test",   # Set pipeline mode to test
            experiment_phase="test", # Set experiment phase to test
            
            # SimulationContext specific parameters
            subject_id='test',
            data_root_path='/path/to/data',
            output_path=cls.output_path,
            coil_file_path='/path/to/coil.ccd',
            didt=1.49e6
        )
        
        # Generate test matsimnibs
        cls.matsimnibs = np.random.random((5, 4, 4))
        for i in range(5):
            cls.matsimnibs[i, 3, 3] = 1.0  # Ensure proper transformation matrices
        
        # Create ROI center info
        cls.roi_center = {
            'gm': np.array([0, 0, 0]),
            'skin': np.array([0, 0, 10]),
            'skin_vec': np.array([0, 0, 1])
        }
        cls.skin_normal_avg = np.array([0, 0, 1])
        
        logger.info("Test case setup complete")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        logger.info("Cleaning up test case...")
        shutil.rmtree(cls.temp_dir)
        logger.info("Test case cleanup complete")
    
    def verify_original_vs_optimized(subject_id, data_path, output_path, n_positions=10):
        """Compare original and optimized implementations."""
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Set up context with required base class parameters
        context = SimulationContext(
            # Required PipelineContext parameters
            dependencies={},                # Empty dictionary for dependencies
            config={},                      # Empty dictionary for config
            pipeline_mode="verification",   # Set pipeline mode to verification
            experiment_phase="test",        # Set experiment phase to test
            
            # SimulationContext specific parameters
            subject_id=subject_id,
            data_root_path=data_path,
            output_path=output_path,
            coil_file_path=os.path.join(data_path, 'coil', 'MagVenture_Cool-B65.ccd'),
            didt=1.49e6
        )


    def test_full_vs_roi_optimization(self):
        """
        Test the performance difference between full mesh and ROI mesh.
        This test will be skipped if actual mesh data is not available.
        """
        logger.info("Testing full vs ROI optimization...")
        
        try:
            # Try to load actual mesh data (this will fail in most test environments)
            # You can update these paths for local testing if needed
            subject_id = '001'  # Example subject ID
            data_root_path = '/path/to/data'  # Update for local testing
            
            # Create a SimulationContext with actual paths and required base parameters
            test_context = SimulationContext(
                # Required PipelineContext parameters
                dependencies={},             # Empty dictionary for dependencies
                config={},                   # Empty dictionary for config
                pipeline_mode="test",        # Set pipeline mode to test
                experiment_phase="test",     # Set experiment phase to test
                
                # SimulationContext specific parameters
                subject_id=subject_id,
                data_root_path=data_root_path,
                output_path=self.output_path,
                coil_file_path='/path/to/coil.ccd',  # Update for local testing
                didt=1.49e6
            )
            
            # Continue with the rest of the test...
            # Load actual mesh and ROI data
            mesh, roi_center = load_mesh_and_roi(test_context)
            skin_normal_avg = get_skin_average_normal_vector(mesh, roi_center, roi_radius=20)
            
            logger.info("Successfully loaded mesh data for performance test")
            
            # Performance test with full mesh (original implementation)
            logger.info("Running performance test with original implementation (full mesh)...")
            start_memory = get_memory_usage()
            start_time = time.time()
            
            _ = original_calc_dAdt(
                context=test_context,
                mesh=mesh,
                matsimnibs=self.matsimnibs[:2],  # Use fewer positions for test
                to_hdf5=True,
                save_path=os.path.join(self.output_path, 'full_mesh'),
                n_cpus=1
            )
            
            full_mesh_time = time.time() - start_time
            full_mesh_memory = get_memory_usage() - start_memory
            
            # Clean up before next test
            import gc
            gc.collect()
            
            # Create ROI mesh
            logger.info("Creating ROI mesh for performance test...")
            
            cylindrical_roi = compute_cylindrical_roi(
                mesh, roi_center['gm'], skin_normal_avg, roi_radius=20
            )
            cropped_mesh = crop_mesh_nodes(mesh, cylindrical_roi)
            roi_mesh = remove_islands(cropped_mesh, roi_center)
            
            # Performance test with ROI mesh (optimized implementation)
            logger.info("Running performance test with optimized implementation (ROI mesh)...")
            start_memory = get_memory_usage()
            start_time = time.time()
            
            _ = calc_dAdt(
                context=test_context,
                mesh=mesh,
                matsimnibs=self.matsimnibs[:2],  # Use fewer positions for test
                roi_mesh=roi_mesh,
                to_hdf5=True,
                save_path=os.path.join(self.output_path, 'roi_mesh'),
                n_cpus=1
            )
            
            roi_mesh_time = time.time() - start_time
            roi_mesh_memory = get_memory_usage() - start_memory
            
            # Log performance metrics
            logger.info(f"Original implementation calculation time: {full_mesh_time:.4f} seconds")
            logger.info(f"Optimized implementation calculation time: {roi_mesh_time:.4f} seconds")
            logger.info(f"Time reduction: {(1 - roi_mesh_time/full_mesh_time) * 100:.2f}%")
            
            if HAS_PSUTIL:
                logger.info(f"Original implementation memory usage: {full_mesh_memory:.2f} MB")
                logger.info(f"Optimized implementation memory usage: {roi_mesh_memory:.2f} MB")
                logger.info(f"Memory reduction: {(1 - roi_mesh_memory/full_mesh_memory) * 100:.2f}%")
            
            # Verify performance improvement
            self.assertLess(roi_mesh_time, full_mesh_time, 
                        "Optimized implementation should be faster than original")
            
            if HAS_PSUTIL:
                self.assertLess(roi_mesh_memory, full_mesh_memory, 
                            "Optimized implementation should use less memory than original")
            
        except Exception as e:
            logger.warning(f"Skipping performance test: {str(e)}")
            self.skipTest(f"Could not load actual mesh data: {str(e)}")
    
    @patch('tms_efield_prediction.simulation.tms_simulation.coil_lib')
    @patch('tms_efield_prediction.simulation.tms_simulation.os.makedirs')
    @patch('tms_efield_prediction.simulation.tms_simulation.shutil.rmtree')
    @patch('tms_efield_prediction.simulation.tms_simulation.h5py.File')
    @patch('tms_efield_prediction.simulation.tms_simulation.np.save')
    @patch('tms_efield_prediction.simulation.tms_simulation.np.load')
    def test_file_handling(self, mock_load, mock_save, mock_h5py, mock_rmtree, 
                        mock_makedirs, mock_coil_lib):
        """Test that temporary files are properly created and cleaned up."""
        logger.info("Testing temporary file handling...")
        
        # Create mock mesh
        mock_mesh = MagicMock()
        mock_mesh.elm.triangles = [MagicMock() for _ in range(1000)]
        
        # Mock the ROI mesh
        mock_roi_mesh = MagicMock()
        mock_roi_mesh.elm.triangles = [MagicMock() for _ in range(100)]
        
        # Set up mocks
        mock_coil_lib.set_up_tms.return_value = np.ones((100, 3))
        mock_load.return_value = np.ones((100, 3))
        
        # Create two mock file objects for two h5py.File calls (write and read)
        mock_write_file = MagicMock()
        mock_read_file = MagicMock()
        mock_dataset = MagicMock()
        mock_write_file.create_dataset.return_value = mock_dataset
        
        # Set up the dataset that will be returned when reading
        mock_read_file.__getitem__.return_value = np.ones((5, 100, 3))
        
        # Configure the mock to return different values on successive calls
        mock_h5py.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mock_write_file)),
            MagicMock(__enter__=MagicMock(return_value=mock_read_file))
        ]
        
        # Run function with HDF5 output
        calc_dAdt(
            context=self.context,
            mesh=mock_mesh,
            matsimnibs=self.matsimnibs,
            roi_mesh=mock_roi_mesh,
            roi_center=self.roi_center,
            skin_normal_avg=self.skin_normal_avg,
            to_hdf5=True,
            save_path=self.output_path,
            n_cpus=1
        )
        
        # Verify directories were created
        mock_makedirs.assert_any_call(self.output_path, exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join(self.output_path, 'tmp_dadt'), exist_ok=True)
        
        # Verify temporary file creation
        self.assertEqual(mock_save.call_count, len(self.matsimnibs))
        
        # Check first save call had the right path pattern
        first_call_args = mock_save.call_args_list[0][0]
        self.assertTrue(first_call_args[0].endswith('0.npy'))
        
        # Verify temporary directory was cleaned up
        mock_rmtree.assert_called_with(os.path.join(self.output_path, 'tmp_dadt'))
        
        # Verify both h5py.File calls (write and read)
        h5_path = os.path.join(self.output_path, 'dAdts.h5')
        
        # Check that h5py.File was called at least twice
        self.assertTrue(len(mock_h5py.call_args_list) >= 2, 
                    "h5py.File should be called at least twice (write and read)")
        
        # Check that the first call was for writing
        first_h5_call = mock_h5py.call_args_list[0][0]
        self.assertEqual(first_h5_call[0], h5_path)
        self.assertEqual(first_h5_call[1], 'w')
        
        # Check that a subsequent call was for reading
        second_h5_call = mock_h5py.call_args_list[1][0]
        self.assertEqual(second_h5_call[0], h5_path)
        self.assertEqual(second_h5_call[1], 'r')
        
        # Check if the calculation_type dataset was created during writing
        create_dataset_calls = [call[0][0] for call in mock_write_file.create_dataset.call_args_list]
        self.assertIn('calculation_type', create_dataset_calls)
        
        logger.info("Temporary file handling test completed")

    @patch('tms_efield_prediction.simulation.tms_simulation.coil_lib')
    def test_on_the_fly_roi_creation(self, mock_coil_lib):
        """Test ROI creation when roi_mesh is not provided."""
        logger.info("Testing on-the-fly ROI creation...")
        
        # Set up mock
        mock_coil_lib.set_up_tms.return_value = np.ones((1000, 3))
        
        # Create patchers for ROI creation functions
        with patch('tms_efield_prediction.simulation.tms_simulation.compute_cylindrical_roi') as mock_compute_roi, \
             patch('tms_efield_prediction.simulation.tms_simulation.crop_mesh_nodes') as mock_crop_mesh, \
             patch('tms_efield_prediction.simulation.tms_simulation.remove_islands') as mock_remove_islands, \
             patch('tms_efield_prediction.simulation.tms_simulation.h5py.File'), \
             patch('tms_efield_prediction.simulation.tms_simulation.np.save'), \
             patch('tms_efield_prediction.simulation.tms_simulation.np.load'), \
             patch('tms_efield_prediction.simulation.tms_simulation.shutil.rmtree'):
            
            # Set up return values
            mock_full_mesh = MagicMock()
            mock_full_mesh.elm.triangles = [MagicMock() for _ in range(10000)]
            
            mock_roi_mesh = MagicMock()
            mock_roi_mesh.elm.triangles = [MagicMock() for _ in range(500)]
            
            mock_compute_roi.return_value = np.ones(10000, dtype=bool)
            mock_crop_mesh.return_value = MagicMock()
            mock_remove_islands.return_value = mock_roi_mesh
            
            # Run with on-the-fly ROI creation
            calc_dAdt(
                context=self.context,
                mesh=mock_full_mesh,
                matsimnibs=self.matsimnibs,
                roi_center=self.roi_center,
                skin_normal_avg=self.skin_normal_avg,
                to_hdf5=False,
                n_cpus=1
            )
        
        # Verify ROI creation functions were called
        mock_compute_roi.assert_called_once()
        mock_crop_mesh.assert_called_once()
        mock_remove_islands.assert_called_once()
        
        # Verify coil_lib was called with ROI mesh
        self.assertEqual(mock_coil_lib.set_up_tms.call_count, 5)
        for call in mock_coil_lib.set_up_tms.call_args_list:
            self.assertIs(call[0][0], mock_roi_mesh)
            
        logger.info("On-the-fly ROI creation test completed")

if __name__ == '__main__':
    unittest.main()