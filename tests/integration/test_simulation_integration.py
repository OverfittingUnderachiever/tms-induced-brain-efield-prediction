"""
Integration tests for TMS simulation components.

This module contains integration tests that verify the proper
interaction between simulation components and the pipeline.
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import simulation components
from tms_efield_prediction.simulation.tms_simulation import (
    SimulationContext, 
    SimulationState,
    load_mesh_and_roi,
    get_skin_average_normal_vector
)
from tms_efield_prediction.simulation.coil_position import (
    CoilPositioningConfig,
    CoilPositionGenerator
)
from tms_efield_prediction.simulation.field_calculation import (
    FieldCalculationConfig,
    FieldCalculator
)
from tms_efield_prediction.simulation.runner import (
    SimulationRunnerConfig,
    SimulationRunner,
    run_simulation
)
from tms_efield_prediction.simulation.pipeline_integration import (
    SimulationPipelineConfig,
    SimulationPipelineAdapter
)

# Import pipeline components
from tms_efield_prediction.utils.state.context import TMSPipelineContext
from tms_efield_prediction.utils.debug.hooks import DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.data.pipeline.tms_data_types import (
    TMSRawData,
    TMSProcessedData,
    TMSSample
)
from tms_efield_prediction.data.formats.simnibs_io import MeshData


@patch('simnibs.mesh_io.read_msh')
@patch('simnibs.simulation.coil_numpy.set_up_tms')
@patch('simnibs.run_simnibs')
class TestEndToEndWorkflow(unittest.TestCase):
    """Tests for the end-to-end simulation workflow."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create subdirectories
        os.makedirs(os.path.join(self.test_dir, 'data', 'sub-001', 'headmodel'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'data', 'sub-001', 'experiment', 'nn'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'data', 'coil'), exist_ok=True)
        
        # Create dummy mesh file
        with open(os.path.join(self.test_dir, 'data', 'sub-001', 'headmodel', '001.msh'), 'w') as f:
            f.write('dummy mesh')
        
        # Create dummy coil file
        with open(os.path.join(self.test_dir, 'data', 'coil', 'MagVenture_Cool-B65.ccd'), 'w') as f:
            f.write('dummy coil')
        
        # Create dummy roi_center file
        with open(os.path.join(self.test_dir, 'data', 'sub-001', 'experiment', '001_roi_center.mat'), 'wb') as f:
            f.write(b'dummy roi')
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_runner_state_transitions(self, mock_run_simnibs, mock_setup_tms, mock_read_msh):
        """Test that SimulationRunner goes through proper state transitions."""
        # Configure mocks
        mock_mesh = MagicMock()
        mock_mesh.crop_mesh.return_value = mock_mesh
        mock_mesh.elements_baricenters.return_value = np.array([[0, 0, 0]])
        mock_mesh.triangle_normals.return_value = np.array([[0, 0, 1]])
        mock_mesh.nodes = np.array([[0, 0, 0]])
        mock_mesh.find_closest_element.return_value = (
            np.array([[0, 0, 0]]),  # centers
            np.array([0])           # indices
        )
        
        # Mock h5py
        with patch('h5py.File') as mock_h5py:
            # Configure mock h5py
            mock_h5py_file = MagicMock()
            mock_h5py.return_value.__enter__.return_value = mock_h5py_file
            mock_h5py_file.__getitem__.return_value = {
                'gm': np.array([0, 0, 0]),
                'skin': np.array([0, 0, 10]),
                'skin_vec': np.array([0, 0, 1])
            }
            
            # Return our mock mesh when read_msh is called
            mock_read_msh.return_value = mock_mesh
            
            # Return mock dA/dt values
            mock_setup_tms.return_value = np.zeros((100, 3))
            
            # Create context
            context = SimulationContext(
                dependencies={"simnibs": "4.0"},
                config={"test": "config"},
                pipeline_mode="mri_efield",
                experiment_phase="preprocessing",
                subject_id="001",
                data_root_path=os.path.join(self.test_dir, 'data', 'sub-001'),
                coil_file_path=os.path.join(self.test_dir, 'data', 'coil', 'MagVenture_Cool-B65.ccd'),
                output_path=os.path.join(self.test_dir, 'data', 'sub-001', 'experiment', 'nn')
            )
            
            # Create simplified config for testing
            config = SimulationRunnerConfig(
                workspace=self.test_dir,
                subject_id="001",
                experiment_type="nn",
                n_cpus=1,
                coil_config=CoilPositioningConfig(
                    search_radius=10.0,
                    spatial_resolution=5.0,
                    rotation_angles=np.array([0])  # Just one angle for testing
                )
            )
            
            # Create runner
            runner = SimulationRunner(
                context, config, MagicMock(), MagicMock()
            )
            
            # Initial state
            self.assertEqual(runner.state.simulation_phase, "initialization")
            
            # Run the first phases
            paths = runner.prepare_paths()
            
            # Should have transitioned to mesh_loading in load_data
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                
                # Mock numpy.load for later
                with patch('numpy.load') as mock_np_load, \
                     patch('numpy.save') as mock_np_save:
                    
                    mock_np_load.return_value = np.ones((5, 10, 3))
                    
                    # Run through all phases
                    msh, roi_center, normal = runner.load_data(paths)
                    matsimnibs, grid = runner.generate_coil_positions(msh, roi_center)
                    dadt = runner.calculate_dadt(msh, matsimnibs)
                    efield = runner.run_efield_simulations(matsimnibs, roi_center, normal, paths)
                    outputs = runner.save_results(efield, paths)
                    
                    # Should have gone through all states
                    self.assertEqual(runner.state.simulation_phase, "data_extraction")
            
            # Try the full run() method
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                
                # Mock numpy functions
                with patch('numpy.load') as mock_np_load, \
                     patch('numpy.save') as mock_np_save, \
                     patch('shutil.rmtree') as mock_rmtree:
                    
                    mock_np_load.return_value = np.ones((5, 10, 3))
                    
                    # Reset state
                    runner.state = SimulationState()
                    
                    # Run full pipeline
                    results = runner.run()
                    
                    # Should complete successfully
                    self.assertEqual(results['status'], 'completed')
                    self.assertEqual(runner.state.simulation_phase, "completed")
    
    @patch('tms_efield_prediction.simulation.runner.run_simulation')
    def test_pipeline_integration(self, mock_run_simulation, *args):
        """Test integration with the existing pipeline."""
        # Configure mock to return results
        mock_run_simulation.return_value = {
            "status": "completed",
            "output_paths": {
                "efield": os.path.join(self.test_dir, 'efields.npy')
            }
        }
        
        # Create pipeline context
        pipeline_context = TMSPipelineContext(
            dependencies={"simnibs": "4.0"},
            config={"test": "config"},
            pipeline_mode="mri_efield",
            experiment_phase="preprocessing",
            subject_id="001",
            data_root_path=os.path.join(self.test_dir, 'data', 'sub-001')
        )
        
        # Create simulation config
        sim_config = SimulationPipelineConfig(
            run_simulations=True
        )
        
        # Create adapter
        adapter = SimulationPipelineAdapter(
            pipeline_context, sim_config, MagicMock(), MagicMock()
        )
        
        # Mock necessary functions
        with patch('numpy.load') as mock_np_load, \
             patch('numpy.save') as mock_np_save, \
             patch('h5py.File') as mock_h5py, \
             patch('os.path.exists') as mock_exists:
            
            # Configure mocks
            mock_exists.return_value = True
            mock_np_load.return_value = np.ones((5, 10, 3))
            
            # Mock h5py
            mock_h5py_file = MagicMock()
            mock_h5py.return_value.__enter__.return_value = mock_h5py_file
            mock_h5py_file.__getitem__.return_value = np.ones((5, 10, 3))
            mock_h5py_file.keys.return_value = ["dAdt"]
            
            # Create raw data
            raw_data = TMSRawData(
                subject_id="001",
                mri_mesh=MeshData(
                    nodes=np.array([[0, 0, 0]]),
                    elements={"tetra": np.array([[0, 0, 0, 0]])},
                    node_data={"T1": np.array([1.0])},
                    element_data={},
                    metadata={}
                )
            )
            
            # Test with run_simulations=True
            processed = adapter.preprocess_raw_data(raw_data)
            
            # Check that simulation was run
            mock_run_simulation.assert_called_once()
            
            # Check that data was updated
            self.assertIsNotNone(processed.efield_data)
            
            # Reset mock
            mock_run_simulation.reset_mock()
            
            # Test with run_simulations=False
            adapter.config.run_simulations = False
            processed = adapter.preprocess_raw_data(raw_data)
            
            # Check that simulation was not run
            mock_run_simulation.assert_not_called()
            
            # Create samples
            samples = adapter.create_samples_from_simulations(processed)
            
            # Should have samples
            self.assertGreater(len(samples), 0)
    
    def test_resource_monitoring_integration(self, *args):
        """Test resource monitoring integration."""
        # Create context
        context = SimulationContext(
            dependencies={"simnibs": "4.0"},
            config={"test": "config"},
            pipeline_mode="mri_efield",
            experiment_phase="preprocessing",
            subject_id="001",
            data_root_path=os.path.join(self.test_dir, 'data', 'sub-001'),
            coil_file_path=os.path.join(self.test_dir, 'data', 'coil', 'MagVenture_Cool-B65.ccd'),
            output_path=os.path.join(self.test_dir, 'data', 'sub-001', 'experiment', 'nn')
        )
        
        # Create config
        config = SimulationRunnerConfig(
            workspace=self.test_dir,
            subject_id="001",
            experiment_type="nn",
            n_cpus=1
        )
        
        # Create real resource monitor
        resource_monitor = ResourceMonitor(max_memory_gb=1)
        
        # Create runner
        runner = SimulationRunner(
            context, config, MagicMock(), resource_monitor
        )
        
        # Start monitoring
        resource_monitor.start_monitoring()
        
        # Should register with resource monitor
        self.assertIn("SimulationRunner", resource_monitor.components)
        
        # Check that reduce_memory callback works
        runner._reduce_memory(0.5)
        
        # Stop monitoring
        resource_monitor.stop_monitoring()
        
        # Test CoilPositionGenerator
        config = CoilPositioningConfig()
        generator = CoilPositionGenerator(context, config, MagicMock(), resource_monitor)
        
        # Should register with resource monitor
        self.assertIn("CoilPositionGenerator", resource_monitor.components)
        
        # Check that reduce_memory callback works
        generator._reduce_memory(0.5)
        
        # Test FieldCalculator
        config = FieldCalculationConfig()
        calculator = FieldCalculator(context, config, MagicMock(), resource_monitor)
        
        # Should register with resource monitor
        self.assertIn("FieldCalculator", resource_monitor.components)
        
        # Check that reduce_memory callback works
        calculator._reduce_memory(0.5)


@patch('tms_efield_prediction.simulation.runner.run_simulation')
class TestPipelinePhaseIsolation(unittest.TestCase):
    """Tests to verify that simulation maintains pipeline phase isolation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create subdirectories
        os.makedirs(os.path.join(self.test_dir, 'data', 'sub-001', 'headmodel'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'data', 'sub-001', 'experiment', 'nn'), exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_phase_transitions(self, mock_run_simulation):
        """Test that pipeline phases are maintained during simulation."""
        # Configure mock to return results
        mock_run_simulation.return_value = {
            "status": "completed",
            "output_paths": {
                "efield": os.path.join(self.test_dir, 'efields.npy')
            }
        }
        
        # Create pipeline context in preprocessing phase
        preprocessing_context = TMSPipelineContext(
            dependencies={"simnibs": "4.0"},
            config={"test": "config"},
            pipeline_mode="mri_efield",
            experiment_phase="preprocessing",
            subject_id="001",
            data_root_path=os.path.join(self.test_dir, 'data', 'sub-001')
        )
        
        # Create simulation config
        sim_config = SimulationPipelineConfig(
            run_simulations=True
        )
        
        # Create adapter for preprocessing
        preprocessing_adapter = SimulationPipelineAdapter(
            preprocessing_context, sim_config, MagicMock(), MagicMock()
        )
        
        # Create pipeline state
        pipeline_state = preprocessing_context.experiment_phase
        
        # Check that adapter preserves phase
        self.assertEqual(preprocessing_adapter.context.experiment_phase, "preprocessing")
        
        # Create adapter for training phase
        training_context = TMSPipelineContext(
            dependencies={"simnibs": "4.0"},
            config={"test": "config"},
            pipeline_mode="mri_efield",
            experiment_phase="training",
            subject_id="001",
            data_root_path=os.path.join(self.test_dir, 'data', 'sub-001')
        )
        
        training_adapter = SimulationPipelineAdapter(
            training_context, sim_config, MagicMock(), MagicMock()
        )
        
        # Check that second adapter has correct phase
        self.assertEqual(training_adapter.context.experiment_phase, "training")
        
        # First adapter should still be in preprocessing
        self.assertEqual(preprocessing_adapter.context.experiment_phase, "preprocessing")
    
    def test_caching_behavior(self, mock_run_simulation):
        """Test that simulation results are properly cached."""
        # Configure mock to return results
        mock_run_simulation.return_value = {
            "status": "completed",
            "output_paths": {
                "efield": os.path.join(self.test_dir, 'efields.npy')
            }
        }
        
        # Create pipeline context
        pipeline_context = TMSPipelineContext(
            dependencies={"simnibs": "4.0"},
            config={"test": "config"},
            pipeline_mode="mri_efield",
            experiment_phase="preprocessing",
            subject_id="001",
            data_root_path=os.path.join(self.test_dir, 'data', 'sub-001')
        )
        
        # Create simulation config with save_intermediate=True
        sim_config = SimulationPipelineConfig(
            run_simulations=True,
            save_intermediate=True
        )
        
        # Create adapter
        adapter = SimulationPipelineAdapter(
            pipeline_context, sim_config, MagicMock(), MagicMock()
        )
        
        # Mock necessary functions
        with patch('numpy.load') as mock_np_load, \
             patch('numpy.save') as mock_np_save, \
             patch('h5py.File') as mock_h5py, \
             patch('os.path.exists') as mock_exists:
            
            # Configure mocks
            mock_exists.return_value = True
            mock_np_load.return_value = np.ones((5, 10, 3))
            
            # Mock h5py
            mock_h5py_file = MagicMock()
            mock_h5py.return_value.__enter__.return_value = mock_h5py_file
            mock_h5py_file.__getitem__.return_value = np.ones((5, 10, 3))
            mock_h5py_file.keys.return_value = ["dAdt"]
            
            # Create raw data with existing fields
            raw_data = TMSRawData(
                subject_id="001",
                mri_mesh=MeshData(
                    nodes=np.array([[0, 0, 0]]),
                    elements={"tetra": np.array([[0, 0, 0, 0]])},
                    node_data={"T1": np.array([1.0])},
                    element_data={},
                    metadata={}
                ),
                efield_data=np.ones((5, 10, 3)),
                dadt_data=np.ones((5, 10, 3)),
                coil_positions=np.ones((5, 4, 4))
            )
            
            # Process raw data - should not run simulations since data exists
            processed = adapter.preprocess_raw_data(raw_data)
            
            # Should not run simulation
            mock_run_simulation.assert_not_called()
            
            # Now create raw data with missing fields
            raw_data_missing = TMSRawData(
                subject_id="001",
                mri_mesh=MeshData(
                    nodes=np.array([[0, 0, 0]]),
                    elements={"tetra": np.array([[0, 0, 0, 0]])},
                    node_data={"T1": np.array([1.0])},
                    element_data={},
                    metadata={}
                )
            )
            
            # Process raw data - should run simulations
            processed = adapter.preprocess_raw_data(raw_data_missing)
            
            # Should run simulation
            mock_run_simulation.assert_called_once()


if __name__ == '__main__':
    unittest.main()