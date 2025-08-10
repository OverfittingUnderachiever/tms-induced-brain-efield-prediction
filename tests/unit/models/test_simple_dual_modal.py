# tests/unit/models/test_simple_dual_modal.py
import unittest
import torch
import numpy as np
import os
import tempfile
from typing import Dict, Any

from tms_efield_prediction.models.architectures.simple_dual_modal import SimpleDualModalModel
from tms_efield_prediction.data.pipeline.tms_data_types import TMSProcessedData
from tms_efield_prediction.utils.state.context import ModelContext


class TestSimpleDualModalModel(unittest.TestCase):
    """Unit tests for SimpleDualModalModel architecture."""
    
    def setUp(self):
        """Set up test environment."""
        # Define test configuration to match actual data format
        self.config = {
            'model_type': 'simple_dual_modal',
            'input_shape': [4, 25, 25, 25],  # [C, D, H, W] - 4 channels, 25x25x25 volume
            'output_shape': [3, 25, 25, 25],  # 3 channels for E-field vector output
            'mri_channels': 1,
            'dadt_channels': 3,  # 3 channels for dA/dt vector field
            'feature_maps': 16,
            'levels': 3,  # 3 levels since 25x25x25 is smaller than 64x64x64
            'dropout_rate': 0.0,  # No dropout for deterministic testing
            'normalization': 'batch',
            'activation': 'relu',
            'use_attention': True,
            'use_residual': True
        }
        
        # Create model context
        self.context = ModelContext(
            dependencies={'torch': torch.__version__},
            config=self.config,
            debug_mode=False
        )
        
        # Create model
        self.model = SimpleDualModalModel(self.config, self.context)
        
        # Set to eval mode to disable dropout and batch norm statistics
        self.model.eval()
    
    def test_model_initialization(self):
        """Test model initialization."""
        # Check if model is an instance of SimpleDualModalModel
        self.assertIsInstance(self.model, SimpleDualModalModel)
        
        # Check if model has expected attributes
        self.assertEqual(self.model.mri_channels, 1)
        self.assertEqual(self.model.dadt_channels, 3)
        self.assertEqual(self.model.base_filters, 16)
        self.assertEqual(self.model.levels, 3)
        
        # Check if model has expected modules
        self.assertTrue(hasattr(self.model, 'mri_encoders'))
        self.assertTrue(hasattr(self.model, 'dadt_encoders'))
        self.assertTrue(hasattr(self.model, 'decoders'))
        self.assertTrue(hasattr(self.model, 'fusion_conv'))
    
    def test_forward_pass(self):
        """Test forward pass with random input."""
        # Create random input tensor in the format similar to actual data
        batch_size = 2
        
        # Create tensor in [B, C, D, H, W] format
        input_tensor = torch.rand(batch_size, 4, 25, 25, 25)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, 3, 25, 25, 25)
        self.assertEqual(output.shape, expected_shape)
        
        # Also test with data in [D, H, W, C] format (like actual data)
        input_tensor_dhwc = torch.rand(25, 25, 25, 4)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor_dhwc)
        
        # Check output shape - should be [B, C, D, H, W]
        expected_shape = (1, 3, 25, 25, 25)
        self.assertEqual(output.shape, expected_shape)
    
    def test_process_data(self):
        """Test processing TMSProcessedData."""
        # Create dummy TMSProcessedData in the format [D, H, W, C]
        # This matches the format where channels are last
        input_features = np.random.rand(25, 25, 25, 4).astype(np.float32)
        target_efield = np.random.rand(25, 25, 25, 3).astype(np.float32)
        
        data = TMSProcessedData(
            subject_id='test_subject',
            input_features=input_features,
            target_efield=target_efield,
            mask=None,
            metadata={}
        )
        
        # Process data
        processed = self.model.process_data(data)
        
        # Check if processed data is a tensor
        self.assertIsInstance(processed, torch.Tensor)
        
        # Check if shape is correct - should be [B, C, D, H, W]
        self.assertEqual(processed.shape, (1, 4, 25, 25, 25))
    
    def test_predict(self):
        """Test prediction with TMSProcessedData."""
        # Create dummy TMSProcessedData in [D, H, W, C] format
        input_features = np.random.rand(25, 25, 25, 4).astype(np.float32)
        target_efield = np.random.rand(25, 25, 25, 3).astype(np.float32)
        
        data = TMSProcessedData(
            subject_id='test_subject',
            input_features=input_features,
            target_efield=target_efield,
            mask=None,
            metadata={}
        )
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model.predict(data)
        
        # Check if prediction is a numpy array
        self.assertIsInstance(prediction, np.ndarray)
        
        # Check if shape is correct - should be [B, C, D, H, W]
        self.assertEqual(prediction.shape, (1, 3, 25, 25, 25))
    
    def test_save_load(self):
        """Test saving and loading model."""
        # Set torch to deterministic mode for consistent results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create a fixed input tensor for consistency
            torch.manual_seed(42)  # Set seed for reproducibility
            input_tensor = torch.rand(1, 4, 25, 25, 25)
            
            # Set model to eval mode
            self.model.eval()
            
            # Generate output with original model
            with torch.no_grad():
                output_original = self.model(input_tensor)
            
            # Save model
            self.model.save(tmp_path)
            
            # Load model
            loaded_model = SimpleDualModalModel.load(tmp_path)
            loaded_model.eval()
            
            # Generate output with loaded model
            with torch.no_grad():
                output_loaded = loaded_model(input_tensor)
            
            # Print shapes and sample values for debugging
            print(f"Original output shape: {output_original.shape}")
            print(f"Loaded output shape: {output_loaded.shape}")
            print(f"Original output first values: {output_original[0, 0, 0, 0, :5]}")
            print(f"Loaded output first values: {output_loaded[0, 0, 0, 0, :5]}")
            
            # Check if outputs are equal (use lower precision tolerance)
            # Interpolation can cause small numerical differences
            torch.testing.assert_close(
                output_original, 
                output_loaded,
                rtol=1e-4,  # Increased relative tolerance
                atol=1e-4   # Increased absolute tolerance
            )
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            # Reset torch settings
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    unittest.main()