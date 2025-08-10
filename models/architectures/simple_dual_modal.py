# tms_efield_prediction/models/architectures/simple_dual_modal.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np

from ..components.layers import Conv3DBlock, UpsampleBlock3D
from ..components.blocks import EncoderBlock3D, DecoderBlock3D
from .base import BaseModel
from ...data.pipeline.tms_data_types import TMSProcessedData, TMSDataType


class SimpleDualModalModel(BaseModel):
    """Simplified Dual-modal architecture for TMS E-field prediction.
    
    This model uses separate encoding paths for MRI and dA/dt data,
    fuses them at a configurable level, and processes through a
    decoder path to predict the E-field.
    """
    
    def __init__(self, config: Dict[str, Any], context: Optional[Any] = None):
        """Initialize the simplified dual-modal model.
        
        Args:
            config: Model configuration dictionary
            context: Optional model context for state tracking
        """
        # Add debug output to help diagnose issues
        print("\n=== Initializing SimpleDualModalModel ===")
        super().__init__(config, context)
        
        # Extract configuration parameters
        self.input_shape = config.get('input_shape', [4, 30, 30, 30])  # Default to your data format
        self.output_shape = config.get('output_shape', [3, 30, 30, 30])
        self.base_filters = config.get('feature_maps', 16)  # Reduced default filters for smaller data
        self.levels = config.get('levels', 3)  # Reduced levels for 25x25x25 data
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.norm_type = config.get('normalization', 'batch')
        self.activation = config.get('activation', 'leaky_relu')
        self.use_attention = config.get('use_attention', True)
        self.use_residual = config.get('use_residual', True)
        
        # Dual-modal specific parameters
        self.mri_channels = config.get('mri_channels', 1)
        self.dadt_channels = config.get('dadt_channels', 3)  # Default to 3 for vector field
        
        # Debug print to confirm channel setup
        print(f"Model config: mri_channels={self.mri_channels}, dadt_channels={self.dadt_channels}")
        print(f"Input shape: {self.input_shape}, Output shape: {self.output_shape}")
        
        # Check if input channels match expected
        if self.input_shape[0] != (self.mri_channels + self.dadt_channels):
            print(f"Warning: Input channels ({self.input_shape[0]}) don't match expected MRI+dA/dt channels ({self.mri_channels + self.dadt_channels})")
            # Auto-adjust to match
            self.mri_channels = 1
            self.dadt_channels = self.input_shape[0] - 1
            print(f"Adjusted to: mri_channels={self.mri_channels}, dadt_channels={self.dadt_channels}")
        
        # Build the model architecture
        self._build_model()
    
    def _build_model(self):
        """Build the simplified dual-modal model architecture."""
        print("Building simplified dual-modal architecture...")
        
        # Initial convolutions
        self.mri_initial = Conv3DBlock(
            self.mri_channels, self.base_filters,
            kernel_size=3, padding=1,
            norm_type=self.norm_type,
            activation=self.activation
        )
        
        self.dadt_initial = Conv3DBlock(
            self.dadt_channels, self.base_filters,
            kernel_size=3, padding=1,
            norm_type=self.norm_type,
            activation=self.activation
        )
        
        # MRI encoder path
        self.mri_encoders = nn.ModuleList()
        mri_in_channels = self.base_filters
        
        for level in range(self.levels):
            out_channels = self.base_filters * (2 ** level)
            
            encoder = EncoderBlock3D(
                mri_in_channels, out_channels,
                use_residual=self.use_residual,
                norm_type=self.norm_type,
                activation=self.activation,
                attention=self.use_attention
            )
            
            self.mri_encoders.append(encoder)
            mri_in_channels = out_channels
        
        # dA/dt encoder path
        self.dadt_encoders = nn.ModuleList()
        dadt_in_channels = self.base_filters
        
        for level in range(self.levels):
            out_channels = self.base_filters * (2 ** level)
            
            encoder = EncoderBlock3D(
                dadt_in_channels, out_channels,
                use_residual=self.use_residual,
                norm_type=self.norm_type,
                activation=self.activation,
                attention=self.use_attention
            )
            
            self.dadt_encoders.append(encoder)
            dadt_in_channels = out_channels
        
        # Calculate fusion channels based on deepest level
        deepest_level_channels = self.base_filters * (2 ** (self.levels - 1))
        
        # Create simple manual fusion components
        print(f"Creating fusion components with {deepest_level_channels*2} -> {deepest_level_channels} channels")
        self.fusion_conv = nn.Conv3d(
            deepest_level_channels * 2,  # Concatenated mri + dadt
            deepest_level_channels,      # Output channels
            kernel_size=1                # 1x1 conv
        )
        self.fusion_norm = nn.BatchNorm3d(deepest_level_channels)
        self.fusion_act = nn.ReLU(inplace=True)
        
        # Decoder path
        self.decoders = nn.ModuleList()
        in_channels = deepest_level_channels
        
        for level in range(self.levels - 1, 0, -1):
            skip_channels = self.base_filters * (2 ** (level - 1))
            out_channels = self.base_filters * (2 ** (level - 1))
            
            decoder = DecoderBlock3D(
                in_channels, skip_channels, out_channels,
                use_residual=self.use_residual,
                norm_type=self.norm_type,
                activation=self.activation,
                attention=self.use_attention
            )
            
            self.decoders.append(decoder)
            in_channels = out_channels
        
        # Final convolution to output channels
        self.final_conv = nn.Conv3d(
            self.base_filters, self.output_shape[0],
            kernel_size=1
        )
        
        # Save the expected output shape for final interpolation if needed
        self.target_shape = self.output_shape[1:]  # [D, H, W]
        
        print("Model building complete")
    
    def validate_config(self) -> bool:
        """Validate model configuration.
        
        Returns:
            bool: True if configuration is valid, raises exception otherwise
        """
        # Call parent validation
        super().validate_config()
        
        # Additional validation for DualModalModel
        if 'input_shape' not in self.config:
            raise ValueError("input_shape must be specified in config")
        
        if 'output_shape' not in self.config:
            raise ValueError("output_shape must be specified in config")
        
        return True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, C, D, H, W] where C = mri_channels + dadt_channels
            
        Returns:
            torch.Tensor: Output tensor of shape [B, output_channels, D, H, W]
        """
        # Debug print to track execution
        print(f"\nForward pass input shape: {x.shape}")
        
        # Ensure input is in the correct format
        if x.dim() == 4:
            # Add batch dimension if missing
            x = x.unsqueeze(0)
            print(f"Added batch dimension, new shape: {x.shape}")
        
        # Handle case where channels might be in the last dimension
        if x.shape[1] > x.shape[-1] and x.shape[-1] <= 4:
            # Likely in [B, D, H, W, C] format - permute to [B, C, D, H, W]
            x = x.permute(0, 4, 1, 2, 3)
            print(f"Permuted dimensions, new shape: {x.shape}")
        
        # Now split input into MRI and dA/dt components
        mri_data = x[:, :self.mri_channels]
        dadt_data = x[:, self.mri_channels:self.mri_channels + self.dadt_channels]
        
        print(f"Split into MRI ({mri_data.shape}) and dA/dt ({dadt_data.shape})")
        
        # Initial convolutions
        mri_features = self.mri_initial(mri_data)
        dadt_features = self.dadt_initial(dadt_data)
        
        # Store features at each level for skip connections
        mri_enc_features = [mri_features]
        dadt_enc_features = [dadt_features]
        
        # Encoder paths
        for level in range(self.levels):
            # MRI encoder
            mri_features = self.mri_encoders[level](mri_features)
            mri_enc_features.append(mri_features)
            
            # dA/dt encoder
            dadt_features = self.dadt_encoders[level](dadt_features)
            dadt_enc_features.append(dadt_features)
        
        print(f"Encoded features - MRI: {mri_enc_features[-1].shape}, dA/dt: {dadt_enc_features[-1].shape}")
        
        # Apply our custom fusion at the deepest level
        # Concatenate the features along channel dimension
        fused_features = torch.cat([mri_enc_features[-1], dadt_enc_features[-1]], dim=1)
        print(f"Concatenated features shape: {fused_features.shape}")
        
        # Apply fusion convolution to reduce channels
        x = self.fusion_conv(fused_features)
        x = self.fusion_norm(x)
        x = self.fusion_act(x)
        print(f"After fusion shape: {x.shape}")
        
        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoders):
            # Calculate level and corresponding skip connection
            level = self.levels - i - 2
            
            # Use MRI features for skip connections
            skip = mri_enc_features[level + 1]
            
            # Apply decoder
            x = decoder(x, skip)
            print(f"After decoder {i} shape: {x.shape}")
        
        # Final convolution
        x = self.final_conv(x)
        print(f"After final conv shape: {x.shape}")
        
        # Check if we need to interpolate to match expected output shape
        if tuple(x.shape[2:]) != tuple(self.target_shape):
            print(f"Interpolating output from {tuple(x.shape[2:])} to {tuple(self.target_shape)}")
            x = nn.functional.interpolate(
                x, 
                size=tuple(self.target_shape),
                mode='trilinear', 
                align_corners=False
            )
        
        print(f"Final output shape: {x.shape}")
        return x
    
    def process_data(self, data: Union[TMSProcessedData, torch.Tensor]) -> torch.Tensor:
        """Process TMSProcessedData into model-ready input.
        
        This method handles the specific preprocessing required for the dual-modal architecture.
        
        Args:
            data: Processed TMS data
            
        Returns:
            torch.Tensor: Prepared input tensor
        """
        # Convert to tensor if not already
        if isinstance(data, TMSProcessedData):
            # We expect input_features to have channels for MRI and dA/dt
            features = torch.from_numpy(data.input_features).float()
            
            # Handle data format: input_features is [D, H, W, C]
            # Need to convert to [B, C, D, H, W] for PyTorch
            if len(features.shape) == 4 and features.shape[-1] <= 4:
                # Input is [D, H, W, C] - move channels to second dimension
                features = features.permute(3, 0, 1, 2).unsqueeze(0)  # Result: [1, C, D, H, W]
            elif len(features.shape) == 4:
                # Input is already [C, D, H, W]
                features = features.unsqueeze(0)  # Add batch dimension
            
            return features
            
        elif isinstance(data, torch.Tensor):
            # Handle different tensor layouts
            if len(data.shape) == 4 and data.shape[-1] <= 4:
                # Input is [D, H, W, C] - move channels to second dimension
                data = data.permute(3, 0, 1, 2).unsqueeze(0)  # Result: [1, C, D, H, W]
            elif len(data.shape) == 4:
                # Input is already [C, D, H, W]
                data = data.unsqueeze(0)  # Add batch dimension
            elif len(data.shape) == 5:
                # Input might already be [B, C, D, H, W] or [B, D, H, W, C]
                if data.shape[-1] <= 4:
                    # Input is [B, D, H, W, C] - move channels to second dimension
                    data = data.permute(0, 4, 1, 2, 3)  # Result: [B, C, D, H, W]
            
            return data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")