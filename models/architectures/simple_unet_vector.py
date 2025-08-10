# tms_efield_prediction/models/architectures/simple_unet_vector.py
"""
Simple UNet architecture for TMS E-field magnitude prediction with vector input.

This model accepts vector (3-channel) MRI and dA/dt input data
but still predicts only the magnitude of the E-field.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import logging
import torch.nn.functional as F

from ..components.layers import Conv3DBlock
from ..components.blocks import EncoderBlock3D, DecoderBlock3D
from .base import BaseModel
from ...data.pipeline.tms_data_types import TMSProcessedData

logger = logging.getLogger(__name__)

class SimpleUNetVectorModel(BaseModel):
    """UNet model for predicting E-field magnitude from vector input data.
    
    This model takes vector (3-channel) MRI and vector dA/dt data as input
    and predicts the magnitude of the resulting E-field using a 3D UNet architecture.
    """
    
    def __init__(self, config: Dict[str, Any], context: Optional[Any] = None):
        """Initialize the UNet vector-to-magnitude model.
        
        Args:
            config: Model configuration dictionary
            context: Optional model context for state tracking
        """
        # Initialize the base model
        super().__init__(config, context)
        
        # Extract configuration parameters with dynamic defaults
        self.input_channels = config.get('input_channels', 6)  # Default: 3 (MRI vector) + 3 (dA/dt vector)
        self.output_channels = config.get('output_channels', 1)  # For magnitude only
        
        # Extract spatial dimensions dynamically (avoiding hardcoded values)
        input_shape = config.get('input_shape', [self.input_channels])
        if len(input_shape) == 1:
            # If only channels provided, use default spatial dimensions
            self.spatial_dims = [30, 30, 30]  # Default but only as fallback
            logger.warning(f"Input shape only specified channels ({input_shape[0]}). "
                          f"Using default spatial dimensions {self.spatial_dims}")
        else:
            # Extract spatial dimensions from input_shape
            self.spatial_dims = input_shape[1:]
            
        # Similarly for output shape
        output_shape = config.get('output_shape', [self.output_channels])
        if len(output_shape) == 1:
            # If only channels provided, use input spatial dimensions
            self.output_dims = self.spatial_dims
            logger.warning(f"Output shape only specified channels ({output_shape[0]}). "
                          f"Using input spatial dimensions {self.output_dims}")
        else:
            # Extract output dimensions
            self.output_dims = output_shape[1:]
        
        # Other UNet parameters
        self.feature_maps = config.get('feature_maps', 16)  # Base filters
        self.levels = config.get('levels', 3)  # Auto-derive based on input size?
        self.norm_type = config.get('norm_type', 'batch')
        self.activation = config.get('activation', 'relu')
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.use_residual = config.get('use_residual', True)
        self.use_attention = config.get('use_attention', False)
        # Feature multiplier parameter
        self.feature_multiplier = config.get('feature_multiplier', 2.0)
        
        # Validate levels based on spatial dimensions
        # Each level halves the spatial dimensions, so we can't have more levels
        # than the number of times we can halve the smallest dimension
        min_dim = min(self.spatial_dims)
        max_possible_levels = 1
        while min_dim > 1:
            min_dim = min_dim // 2
            max_possible_levels += 1
            
        if self.levels > max_possible_levels:
            logger.warning(f"Requested levels ({self.levels}) exceeds maximum possible "
                          f"({max_possible_levels}) for spatial dimensions {self.spatial_dims}. "
                          f"Reducing to {max_possible_levels}.")
            self.levels = max_possible_levels
        
        # Log configuration
        logger.info(f"Vector UNet Config: Input Channels={self.input_channels}, Output Channels={self.output_channels}, "
                   f"Base Filters={self.feature_maps}, Levels={self.levels}, Feature Multiplier={self.feature_multiplier}")
        logger.info(f"Input Shape: {[self.input_channels, *self.spatial_dims]}, Output Shape: {[self.output_channels, *self.output_dims]}")
        
        # Build the model
        logger.info("Building Vector UNet architecture...")
        self._build_model()
        logger.info("Vector UNet architecture building complete.")
        
    def validate_config(self) -> bool:
        """Validate model configuration.
        
        Returns:
            bool: True if configuration is valid, raises exception otherwise
        """
        # Call parent validation
        super().validate_config()
        
        # Additional validation for SimpleUNetVectorModel
        if 'input_shape' not in self.config:
            raise ValueError("input_shape must be specified in config")
        
        if 'output_shape' not in self.config:
            raise ValueError("output_shape must be specified in config")
        
        # Check vector-specific configuration
        if self.input_channels < 6:
            logger.warning(f"Input channels is {self.input_channels}, which may be too few "
                          f"for vector MRI (3) + vector dA/dt (3)")
        
        return True
        
    def _build_model(self):
        """Build the UNet model architecture."""
        # Initial convolutional block to increase channels
        self.initial_conv = Conv3DBlock(
            self.input_channels, 
            self.feature_maps,
            kernel_size=3, 
            padding=1,
            norm_type=self.norm_type,
            activation=self.activation
        )
            
        # Encoder blocks (downsampling)
        self.encoders = nn.ModuleList()
        in_channels = self.feature_maps
        
        for level in range(self.levels):
            # Use feature_multiplier instead of fixed 2.0
            out_channels = int(self.feature_maps * (self.feature_multiplier ** level))
            
            encoder = EncoderBlock3D(
                in_channels, 
                out_channels,
                use_residual=self.use_residual,
                norm_type=self.norm_type,
                activation=self.activation,
                attention=self.use_attention
            )
            
            self.encoders.append(encoder)
            in_channels = out_channels
        
        # Decoder blocks (upsampling)
        self.decoders = nn.ModuleList()
        
        for level in range(self.levels - 1, -1, -1):  # Now goes all the way to 0
            if level == 0:
                # Special case for the final level (level 0)
                out_channels = self.feature_maps
                skip_channels = self.feature_maps  # Skip connection from initial conv
            else:
                # Use feature_multiplier instead of fixed 2.0
                out_channels = int(self.feature_maps * (self.feature_multiplier ** (level - 1)))
                skip_channels = out_channels
            
            decoder = DecoderBlock3D(
                in_channels, 
                skip_channels, 
                out_channels,
                use_residual=self.use_residual,
                norm_type=self.norm_type,
                activation=self.activation,
                attention=self.use_attention
            )
            
            self.decoders.append(decoder)
            in_channels = out_channels
                
        # Final convolutional layer to output magnitude
        self.final_conv = nn.Conv3d(
            in_channels, 
            self.output_channels,
            kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape [B, C, D, H, W] where C=6 for vector MRI (3) + vector dA/dt (3)

        Returns:
            torch.Tensor: Output tensor of shape [B, 1, D, H, W] representing magnitude
        """
        # Check input dimensions and type
        if x.ndim == 4:
            x = x.unsqueeze(0) # Add batch dimension if missing
        if not x.is_floating_point():
            x = x.float() # Ensure input is float
            
        # Validate input channels - now expecting 6 channels by default
        if x.shape[1] != self.input_channels:
            logger.warning(f"Expected {self.input_channels} input channels, got {x.shape[1]}. "
                           f"This may cause issues if the model was configured for vector data.")

        # Initial convolution
        initial_features = self.initial_conv(x) # Use a different variable name for clarity

        # Store encoder outputs for skip connections
        encoder_outputs = [initial_features]
        current_features = initial_features

        # Encoder path
        for encoder in self.encoders:
            current_features = encoder(current_features)
            encoder_outputs.append(current_features)

        # Decoder path with skip connections
        # Note: current_features is now the bottleneck tensor
        for i, decoder in enumerate(self.decoders):
            # Get the corresponding encoder output from the end of the list backwards
            # Skip index: -1 is bottleneck, -2 is last encoder output, etc.
            # Skip connection for decoder 'i' corresponds to encoder output at index 'len(encoder_outputs) - 2 - i'
            skip_idx = len(encoder_outputs) - 2 - i
            if skip_idx < 0: # Should not happen with correct level setup
                raise IndexError(f"Decoder {i}: Invalid skip connection index {skip_idx}")
            skip = encoder_outputs[skip_idx]

            # Apply decoder
            current_features = decoder(current_features, skip)

        # Final convolution to get magnitude (output has 1 channel)
        magnitude_output = self.final_conv(current_features)

        # Apply activation function (e.g., Softplus for non-negativity)
        magnitude_output = F.softplus(magnitude_output)

        # Check if we need to interpolate to match expected output shape
        output_spatial_dims = tuple(magnitude_output.shape[2:])
        expected_dims = tuple(self.output_dims)

        if output_spatial_dims != expected_dims:
            # Use functional interpolate
            magnitude_output = F.interpolate(
                magnitude_output,
                size=expected_dims,
                mode='trilinear',    # Suitable for 3D volume data
                align_corners=False  # Usually recommended for non-image data or when not critical
            )
            # Log only if sizes were different before interpolation
            logger.debug(f"Interpolated output from {output_spatial_dims} to {expected_dims}")

        return magnitude_output
    
    def process_data(self, data: Union[TMSProcessedData, torch.Tensor]) -> torch.Tensor:
        """Process data into model-ready input.
        
        Args:
            data: Raw data to process
            
        Returns:
            torch.Tensor: Processed input tensor
        """
        # Handle different input types
        if isinstance(data, TMSProcessedData):
            # Extract features from TMSProcessedData
            features = torch.from_numpy(data.input_features).float()
            
            # Handle channel layout if needed
            if len(features.shape) == 4 and features.shape[-1] <= self.input_channels:
                # Input seems to be in [D, H, W, C] format - rearrange to [C, D, H, W]
                features = features.permute(3, 0, 1, 2)
            
            # Add batch dimension if needed
            if len(features.shape) == 4:
                features = features.unsqueeze(0)  # [1, C, D, H, W]
                
            return features
            
        elif isinstance(data, torch.Tensor):
            # Handle tensor input with various shapes
            if len(data.shape) == 3:
                # Assume [D, H, W] - add channel and batch dimensions
                return data.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                
            elif len(data.shape) == 4:
                # Check if this is [D, H, W, C] or [C, D, H, W]
                if data.shape[-1] <= self.input_channels:
                    # Likely [D, H, W, C] - permute and add batch dimension
                    return data.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, D, H, W]
                else:
                    # Likely [C, D, H, W] - add batch dimension
                    return data.unsqueeze(0)  # [1, C, D, H, W]
                    
            elif len(data.shape) == 5:
                # Already in [B, C, D, H, W] format
                return data
                
            else:
                raise ValueError(f"Unsupported tensor shape: {data.shape}")
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")