# tms_efield_prediction/models/components/blocks.py
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, List

from .layers import Conv3DBlock, SpatialAttention3D, ChannelAttention3D


class ResidualBlock3D(nn.Module):
    """3D Residual Block.
    
    A block with skip connection that performs:
    x -> Conv -> Norm -> Activation -> Conv -> Norm -> Add(input) -> Activation
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """Initialize the 3D residual block.
        
        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            norm_type: Normalization type
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        # First conv block
        self.conv1 = Conv3DBlock(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout
        )
        
        # Second conv block without activation
        self.conv2 = Conv3DBlock(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            norm_type=norm_type,
            activation="none",
            dropout=0.0
        )
        
        # Final activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Store residual
        residual = x
        
        # Forward pass through conv blocks
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Add residual and apply activation
        x = x + residual
        x = self.activation(x)
        
        return x


class CBAMBlock3D(nn.Module):
    """Convolutional Block Attention Module (CBAM) for 3D data.
    
    Applies sequential channel and spatial attention to input feature maps.
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        """Initialize the CBAM module.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Channel reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention3D(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention3D(channels, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CBAM module.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor with attention applied
        """
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        return x


class EncoderBlock3D(nn.Module):
    """3D Encoder block for downsampling paths.
    
    Combines downsampling with convolution blocks and optional residual connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        use_residual: bool = True,
        norm_type: str = "batch",
        activation: str = "relu",
        attention: bool = False
    ):
        """Initialize the 3D encoder block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Downsampling stride
            use_residual: Whether to use residual connections
            norm_type: Normalization type
            activation: Activation function
            attention: Whether to use attention mechanism
        """
        super().__init__()
        
        # Downsampling convolution
        self.downsample = Conv3DBlock(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            norm_type=norm_type,
            activation=activation
        )
        
        # Additional processing
        if use_residual:
            self.process = ResidualBlock3D(
                out_channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                activation=activation
            )
        else:
            self.process = Conv3DBlock(
                out_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                norm_type=norm_type,
                activation=activation
            )
        
        # Optional attention module
        if attention:
            self.attention = CBAMBlock3D(out_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder block.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.downsample(x)
        x = self.process(x)
        x = self.attention(x)
        return x


class DecoderBlock3D(nn.Module):
    """3D Decoder block for upsampling paths with skip connections.
    
    Combines upsampling with convolution blocks and skip connection handling.
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_mode: str = "transpose",
        use_residual: bool = True,
        norm_type: str = "batch",
        activation: str = "relu",
        attention: bool = False
    ):
        """Initialize the 3D decoder block.
        
        Args:
            in_channels: Number of input channels from previous layer
            skip_channels: Number of channels from skip connection
            out_channels: Number of output channels
            upsample_mode: Upsampling mode
            use_residual: Whether to use residual connections
            norm_type: Normalization type
            activation: Activation function
            attention: Whether to use attention mechanism
        """
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )
        
        # Fusion convolution after concatenating with skip connection
        self.fusion = Conv3DBlock(
            in_channels // 2 + skip_channels, out_channels,
            kernel_size=3, padding=1,
            norm_type=norm_type,
            activation=activation
        )
        
        # Additional processing
        if use_residual:
            self.process = ResidualBlock3D(
                out_channels,
                kernel_size=3,
                norm_type=norm_type,
                activation=activation
            )
        else:
            self.process = Conv3DBlock(
                out_channels, out_channels,
                kernel_size=3, padding=1,
                norm_type=norm_type,
                activation=activation
            )
        
        # Optional attention module
        if attention:
            self.attention = CBAMBlock3D(out_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder block.
        
        Args:
            x: Input tensor from previous decoder layer
            skip: Skip connection tensor from encoder
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.upsample(x)
        
        # Handle different spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode='trilinear', align_corners=False
            )
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Process combined features
        x = self.fusion(x)
        x = self.process(x)
        x = self.attention(x)
        
        return x


class DualEncoderFusion(nn.Module):
    """Fusion block for dual encoder architectures.
    
    Used to combine features from MRI and dA/dt encoders in the dual-modal approach.
    """
    
    def __init__(
        self,
        mri_channels: int,
        dadt_channels: int,
        output_channels: int,
        fusion_type: str = "concat",
        norm_type: str = "batch",
        activation: str = "relu",
        use_attention: bool = True
    ):
        """Initialize the dual encoder fusion block.
        
        Args:
            mri_channels: Number of channels from MRI encoder
            dadt_channels: Number of channels from dA/dt encoder
            output_channels: Number of output channels
            fusion_type: Fusion type ("concat", "add", "attention")
            norm_type: Normalization type
            activation: Activation function
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.fusion = Conv3DBlock(
                mri_channels + dadt_channels, output_channels,
                kernel_size=3, padding=1,
                norm_type=norm_type,
                activation=activation
            )
        elif fusion_type == "add":
            # Match channel dimensions before adding
            self.mri_proj = Conv3DBlock(
                mri_channels, output_channels,
                kernel_size=1, padding=0,
                norm_type=norm_type,
                activation=activation
            )
            self.dadt_proj = Conv3DBlock(
                dadt_channels, output_channels,
                kernel_size=1, padding=0,
                norm_type=norm_type,
                activation=activation
            )
            self.post_add = ResidualBlock3D(
                output_channels,
                norm_type=norm_type,
                activation=activation
            )
        elif fusion_type == "attention":
            # Cross-attention mechanism
            self.mri_proj = Conv3DBlock(
                mri_channels, output_channels,
                kernel_size=1, padding=0,
                norm_type=norm_type,
                activation=activation
            )
            self.dadt_proj = Conv3DBlock(
                dadt_channels, output_channels,
                kernel_size=1, padding=0,
                norm_type=norm_type,
                activation=activation
            )
            self.attention = CBAMBlock3D(output_channels)
            self.fusion = Conv3DBlock(
                output_channels * 2, output_channels,
                kernel_size=3, padding=1,
                norm_type=norm_type,
                activation=activation
            )
        
        # Optional attention after fusion
        if use_attention and fusion_type != "attention":
            self.attention = CBAMBlock3D(output_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, mri_features: torch.Tensor, dadt_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the fusion block.
        
        Args:
            mri_features: Features from MRI encoder
            dadt_features: Features from dA/dt encoder
            
        Returns:
            torch.Tensor: Fused features
        """
        # Handle different spatial dimensions
        if mri_features.shape[2:] != dadt_features.shape[2:]:
            dadt_features = nn.functional.interpolate(
                dadt_features, size=mri_features.shape[2:],
                mode='trilinear', align_corners=False
            )
        
        # Fusion based on type
        if self.fusion_type == "concat":
            x = torch.cat([mri_features, dadt_features], dim=1)
            x = self.fusion(x)
        elif self.fusion_type == "add":
            x_mri = self.mri_proj(mri_features)
            x_dadt = self.dadt_proj(dadt_features)
            x = x_mri + x_dadt
            x = self.post_add(x)
        elif self.fusion_type == "attention":
            x_mri = self.mri_proj(mri_features)
            x_dadt = self.dadt_proj(dadt_features)
            # Apply attention to each branch
            x_mri_att = self.attention(x_mri)
            x_dadt_att = self.attention(x_dadt)
            # Concatenate and fuse
            x = torch.cat([x_mri_att, x_dadt_att], dim=1)
            x = self.fusion(x)
        
        # Apply attention if needed
        x = self.attention(x)
        
        return x