# tms_efield_prediction/models/components/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List


class Conv3DBlock(nn.Module):
    """3D Convolution block with normalization and activation.
    
    This block combines Conv3D, normalization (BatchNorm3D or InstanceNorm3D),
    and activation into a single reusable component.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 1,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """Initialize the 3D convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Convolution stride
            padding: Convolution padding
            norm_type: Normalization type ("batch", "instance", or "none")
            activation: Activation function ("relu", "leaky_relu", or "none")
            dropout: Dropout probability
        """
        super().__init__()
        
        # Convolution layer
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm_type == "none")  # No bias if using normalization
        )
        
        # Normalization layer
        if norm_type == "batch":
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm3d(out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Identity()
        
        # Dropout layer
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolution block.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SpatialAttention3D(nn.Module):
    """3D Spatial Attention module.
    
    Applies attention across spatial dimensions of 3D data.
    """
    
    def __init__(self, channels: int, kernel_size: int = 7):
        """Initialize the 3D spatial attention module.
        
        Args:
            channels: Number of input channels
            kernel_size: Size of attention kernel
        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention module.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Output tensor with attention applied
        """
        # Calculate attention mask
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention mask
        return x * attention


class ChannelAttention3D(nn.Module):
    """3D Channel Attention module.
    
    Applies attention across channel dimension of 3D data.
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        """Initialize the 3D channel attention module.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Channel reduction ratio for MLP
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP
        reduced_channels = max(1, channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention module.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Output tensor with attention applied
        """
        # Calculate attention weights
        avg_pool = self.mlp(self.avg_pool(x))
        max_pool = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_pool + max_pool)
        
        # Apply attention weights
        return x * attention


class UpsampleBlock3D(nn.Module):
    """3D Upsampling block for decoder paths.
    
    Supports different upsampling methods including transposed convolution
    and interpolation followed by convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "transpose",
        scale_factor: int = 2,
        norm_type: str = "batch",
        activation: str = "relu"
    ):
        """Initialize the 3D upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mode: Upsampling mode ("transpose", "nearest", "trilinear")
            scale_factor: Scale factor for upsampling
            norm_type: Normalization type
            activation: Activation function
        """
        super().__init__()
        
        if mode == "transpose":
            self.upsample = nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=scale_factor, stride=scale_factor
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            )
        
        self.conv_block = Conv3DBlock(
            out_channels, out_channels,
            norm_type=norm_type,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling block.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Upsampled tensor
        """
        x = self.upsample(x)
        x = self.conv_block(x)
        return x