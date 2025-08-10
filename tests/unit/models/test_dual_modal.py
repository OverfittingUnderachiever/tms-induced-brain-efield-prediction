# tests/unit/models/test_dual_modal.py
import unittest
import torch
import numpy as np
import os
import tempfile
from typing import Dict, Any

from tms_efield_prediction.models.architectures.dual_modal import DualModalModel
from tms_efield_prediction.data.pipeline.tms_data_types import TMSProcessedData
from tms_efield_prediction.utils.state.context import ModelContext


class TestDualModalModel(unittest.TestCase):
    """Unit tests for DualModalModel architecture."""
    
    def setUp(self):
        """Set up test environment."""
        # Define test configuration to match your actual data format
        self.config = {
            'model_type': 'dual_modal',
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
            'use_residual': True,
            'fusion_type': 'concat',
            'fusion_level': 1,  # Fusion at level 1
            'shared_decoder': True
        }
        
        # Create model context
        self.context = ModelContext(
            dependencies={'torch': torch.__version__},
            config=self.config,
            debug_mode=False
        )
        
        # Create model
        self.model = DualModalModel(self.config, self.context)
        
        # Set to eval mode to disable dropout and batch norm statistics
        self.model.eval()
    
    def test_model_initialization(self):
        """Test model initialization."""
        # Check if model is an instance of DualModalModel
        self.assertIsInstance(self.model, DualModalModel)
        
        # Check if model has expected attributes
        self.assertEqual(self.model.mri_channels, 1)
        self.assertEqual(self.model.dadt_channels, 3)  # Updated to expect 3 channels
        self.assertEqual(self.model.base_filters, 16)
        self.assertEqual(self.model.levels, 3)
        
        # Check if model has expected modules
        self.assertTrue(hasattr(self.model, 'mri_encoders'))
        self.assertTrue(hasattr(self.model, 'dadt_encoders'))
        self.assertTrue(hasattr(self.model, 'decoders'))
        self.assertTrue(hasattr(self.model, 'fusion'))
    
    def test_forward_pass(self):
        """Test forward pass with random input."""
        # Create random input tensor in the format similar to your actual data
        batch_size = 2
        
        # Create tensor in [B, C, D, H, W] format
        input_tensor = torch.rand(batch_size, 4, 25, 25, 25)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, 3, 25, 25, 25)
        self.assertEqual(output.shape, expected_shape)
        
        # Also test with data in [D, H, W, C] format (like your actual data)
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
        # This matches the format from stack_pipeline.py where channels are last
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
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            self.model.save(tmp_path)
            
            # Modify parameters to check if loading works
            for param in self.model.parameters():
                if param.requires_grad:
                    param.data.fill_(0.0)
            
            # Load model
            loaded_model = DualModalModel.load(tmp_path)
            
            # Check if loaded model has the same configuration
            loaded_config = loaded_model.config
            self.assertEqual(loaded_config['mri_channels'], self.config['mri_channels'])
            self.assertEqual(loaded_config['dadt_channels'], self.config['dadt_channels'])
            self.assertEqual(loaded_config['feature_maps'], self.config['feature_maps'])
            
            # Create input tensor matching expected format
            input_tensor = torch.rand(1, 4, 25, 25, 25)
            
            # Compare outputs
            self.model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                output_original = self.model(input_tensor)
                output_loaded = loaded_model(input_tensor)
            
            # Check if outputs are close
            torch.testing.assert_close(output_original, output_loaded)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_hyperparameters(self):
        """Test getting and setting hyperparameters."""
        # Get current hyperparameters
        hyperparams = self.model.get_hyperparameters()
        
        # Check if hyperparameters match configuration
        self.assertEqual(hyperparams['base_filters'], self.config['feature_maps'])
        self.assertEqual(hyperparams['levels'], self.config['levels'])
        self.assertEqual(hyperparams['fusion_type'], self.config['fusion_type'])
        
        # Create modified hyperparameters
        modified_hyperparams = hyperparams.copy()
        modified_hyperparams['base_filters'] = 32
        modified_hyperparams['levels'] = 4
        modified_hyperparams['fusion_type'] = 'attention'
        
        # Set hyperparameters
        self.model.set_hyperparameters(modified_hyperparams)
        
        # Check if model was updated
        self.assertEqual(self.model.base_filters, 32)
        self.assertEqual(self.model.levels, 4)
        self.assertEqual(self.model.fusion_type, 'attention')
    
    def test_input_validation(self):
        """Test input validation."""
        # Test with missing required parameter
        invalid_config = self.config.copy()
        del invalid_config['input_shape']
        
        with self.assertRaises(ValueError):
            DualModalModel(invalid_config)
        
        # Test with invalid fusion level
        invalid_config = self.config.copy()
        invalid_config['fusion_level'] = 5  # Greater than levels
        
        with self.assertRaises(ValueError):
            DualModalModel(invalid_config)
    
    def test_memory_usage(self):
        """Test memory usage reporting."""
        # Get memory usage
        memory_info = self.model.get_memory_usage()
        
        # Check if memory info contains expected keys
        expected_keys = ['param_count', 'param_size_bytes', 'buffer_size_bytes', 
                        'total_size_bytes', 'total_size_mb']
        
        for key in expected_keys:
            self.assertIn(key, memory_info)
        
        # Check if parameter count is positive
        self.assertGreater(memory_info['param_count'], 0)
        
        # Check if memory size is consistent
        self.assertEqual(memory_info['param_size_bytes'], memory_info['param_count'] * 4)  # 4 bytes per float32
        self.assertEqual(memory_info['total_size_bytes'], 
                        memory_info['param_size_bytes'] + memory_info['buffer_size_bytes'])


if __name__ == '__main__':
    unittest.main()