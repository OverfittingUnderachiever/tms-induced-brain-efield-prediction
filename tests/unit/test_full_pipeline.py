# tests/integration/pipeline_tests/test_full_pipeline.py
import unittest
import os
import shutil
import sys
import numpy as np
import tempfile
import pickle

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from tms_efield_prediction.data.pipeline.controller import PipelineController
from tms_efield_prediction.data.pipeline.loader import DataLoader, Dataset
from tms_efield_prediction.data.pipeline.preprocessor import Preprocessor
from tms_efield_prediction.data.pipeline.validator import DataValidator

from tms_efield_prediction.utils.state.context import PipelineContext
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor


class TestFullPipeline(unittest.TestCase):
    """Test the full data pipeline."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.output_dir = os.path.join(self.test_dir, "output")
        
        # Create directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test data
        self._create_test_data()
        
        # Basic pipeline configuration
        self.config = {
            "batch_size": 2,
            "normalization_method": "z_score",
            "orientation_encoding": "vector",
            "orientation_normalization": True,
            "output_shape": (128, 128, 64),  # Smaller for tests
            "clip_values": (-3.0, 3.0)
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def _create_test_data(self):
        """Create test data for pipeline."""
        # Create MRI data (small for testing)
        for i in range(5):
            mri_data = np.random.randn(64, 64, 32).astype(np.float32)
            mri_file = os.path.join(self.raw_dir, f"mri_{i}.npy")
            np.save(mri_file, mri_data)
        
        # Create orientation data
        for i in range(5):
            orientation = np.random.randn(3).astype(np.float32)
            orientation = orientation / np.linalg.norm(orientation)
            orientation_file = os.path.join(self.raw_dir, f"orientation_{i}.npy")
            np.save(orientation_file, orientation)
    
    def test_data_loader(self):
        """Test data loader component."""
        # Create pipeline context
        context = PipelineContext(
            dependencies={"numpy": "1.20.0"},
            config=self.config,
            pipeline_mode="dual_modal",
            experiment_phase="preprocessing"
        )
        
        # Create data loader
        loader = DataLoader(
            data_dir=self.test_dir,
            pipeline_context=context
        )
        
        # Generate dummy data
        dataset = loader._generate_dummy_data(5)
        
        # Check dataset properties
        self.assertEqual(len(dataset.mri_data), 5)
        self.assertEqual(len(dataset.orientation_data), 5)
        self.assertEqual(dataset.metadata["pipeline_mode"], "dual_modal")
        
        # Check data shapes
        self.assertEqual(dataset.mri_data[0].shape, (255, 255, 191))
        self.assertEqual(dataset.orientation_data[0].shape, (3,))
        
        # Check normalization
        for orientation in dataset.orientation_data:
            norm = np.linalg.norm(orientation)
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_preprocessor(self):
        """Test preprocessor component."""
        # Create pipeline context
        context = PipelineContext(
            dependencies={"numpy": "1.20.0"},
            config=self.config,
            pipeline_mode="dual_modal",
            experiment_phase="preprocessing"
        )
        
        # Create preprocessor
        preprocessor = Preprocessor(
            pipeline_context=context,
            output_dir=os.path.join(self.output_dir, "preprocessed")
        )
        
        # Create test data
        mri_data = [np.random.randn(64, 64, 32).astype(np.float32) for _ in range(3)]
        orientation_data = [np.random.randn(3).astype(np.float32) for _ in range(3)]
        
        # Normalize orientations
        for i in range(len(orientation_data)):
            orientation_data[i] = orientation_data[i] / np.linalg.norm(orientation_data[i])
        
        # Process data
        result = preprocessor.preprocess_sample(mri_data[0], orientation_data[0])
        
        # Check output
        self.assertIn("mri", result)
        self.assertIn("orientation", result)
        self.assertEqual(result["mri"].shape, self.config["output_shape"])
        self.assertEqual(result["orientation"].shape, (3,))
        
        # Test batch processing
        batch_result = preprocessor.preprocess_batch(mri_data, orientation_data)
        
        # Check batch output
        self.assertIn("mri", batch_result)
        self.assertIn("orientation", batch_result)
        self.assertEqual(batch_result["mri"].shape, (3,) + self.config["output_shape"])
        self.assertEqual(batch_result["orientation"].shape, (3, 3))
    
    def test_validator(self):
        """Test data validator component."""
        # Create pipeline context
        context = PipelineContext(
            dependencies={"numpy": "1.20.0"},
            config=self.config,
            pipeline_mode="dual_modal",
            experiment_phase="preprocessing"
        )
        
        # Create validator
        validator = DataValidator(
            pipeline_context=context
        )
        
        # Create test processed data directory
        processed_dir = os.path.join(self.output_dir, "validation_test")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Create valid test data
        mri_data = np.random.randn(3, 128, 128, 64).astype(np.float32)
        orientation_data = np.random.randn(3, 3).astype(np.float32)
        
        # Normalize orientations
        for i in range(orientation_data.shape[0]):
            orientation_data[i] = orientation_data[i] / np.linalg.norm(orientation_data[i])
        
        # Save test data
        np.save(os.path.join(processed_dir, "mri_processed.npy"), mri_data)
        np.save(os.path.join(processed_dir, "orientation_processed.npy"), orientation_data)
        
        # Create metadata
        metadata = {
            "pipeline_mode": "dual_modal",
            "num_samples": 3,
            "mri_shape": mri_data.shape,
            "orientation_shape": orientation_data.shape,
            "config": context.config,
            "timestamp": 12345.67
        }
        
        with open(os.path.join(processed_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
        
        # Validate
        result = validator.validate_processed_data(processed_dir)
        
        # Check validation result
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        
        # Test validation with missing file
        os.remove(os.path.join(processed_dir, "metadata.pkl"))
        
        # Validate again
        result = validator.validate_processed_data(processed_dir)
        
        # Check validation result
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].error_code, "FILE_MISSING")
    
    def test_full_pipeline_controller(self):
        """Test the full pipeline controller."""
        # Create pipeline controller
        controller = PipelineController(
            raw_data_dir=self.raw_dir,
            output_base_dir=self.output_dir,
            pipeline_mode="dual_modal",
            config=self.config,
            max_memory_gb=1,  # Small for testing
            debug_mode=True
        )
        
        # Execute preprocessing
        result = controller.execute_preprocessing()
        
        # Check execution result
        # Note: This might fail if test data is not properly set up
        # Focus on verifying the controller executes without errors
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.execution_time)
        self.assertEqual(result.pipeline_mode, "dual_modal")


if __name__ == "__main__":
    unittest.main()