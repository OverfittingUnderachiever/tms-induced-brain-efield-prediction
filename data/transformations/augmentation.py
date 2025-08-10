import torch
import numpy as np
import random
from typing import Union, Optional, Tuple, List, Dict, Any
from ..pipeline.tms_data_types import TMSSample, TMSProcessedData
import torch.nn.functional as F


class BatchAugmentation:
    """
    Implements efficient batch-level augmentation for TMS data.
    Core batch transformation functions used by both standard augmentation and TrivialAugment.
    """
    
    @staticmethod
    def batch_spatial_shift(
        batch_tensor: torch.Tensor, 
        shifts: torch.Tensor,
        dims_first: bool = True
    ) -> torch.Tensor:
        """Vectorized implementation that processes all samples at once"""
        
        # Handle channels-last format if needed
        if not dims_first:
            batch_tensor = batch_tensor.permute(0, 4, 1, 2, 3)
            
        batch_size = batch_tensor.shape[0]
        _, channels, x_dim, y_dim, z_dim = batch_tensor.shape
        device = batch_tensor.device
        
        # Create output tensor
        output = torch.zeros_like(batch_tensor)
        
        # Create a batch mask for samples with non-zero shifts
        has_shift = (shifts != 0).any(dim=1)
        
        # Copy samples with no shift directly
        output[~has_shift] = batch_tensor[~has_shift]
        
        # Process only samples that have shifts
        if has_shift.any():
            # Process all samples with shifts using a grid_sample approach
            # This is more efficient than per-sample slicing
            grid_x, grid_y, grid_z = torch.meshgrid(
                torch.arange(x_dim, device=device),
                torch.arange(y_dim, device=device),
                torch.arange(z_dim, device=device),
                indexing="ij"
            )
            
            # Expand to batch dimension [B, X, Y, Z]
            grid_x = grid_x.expand(batch_size, -1, -1, -1)
            grid_y = grid_y.expand(batch_size, -1, -1, -1)
            grid_z = grid_z.expand(batch_size, -1, -1, -1)
            
            # Apply shifts to grid for each sample in batch (vectorized)
            # Only modify grid for samples with shifts
            shifted_x = grid_x.clone()
            shifted_y = grid_y.clone()
            shifted_z = grid_z.clone()
            
            # Apply shifts using broadcasting
            shifted_x[has_shift] = shifted_x[has_shift] - shifts[has_shift, 0].view(-1, 1, 1, 1)
            shifted_y[has_shift] = shifted_y[has_shift] - shifts[has_shift, 1].view(-1, 1, 1, 1)
            shifted_z[has_shift] = shifted_z[has_shift] - shifts[has_shift, 2].view(-1, 1, 1, 1)
            
            # Normalize to [-1, 1] for grid_sample
            shifted_x = 2.0 * shifted_x / (x_dim - 1) - 1.0
            shifted_y = 2.0 * shifted_y / (y_dim - 1) - 1.0
            shifted_z = 2.0 * shifted_z / (z_dim - 1) - 1.0
            
            # Create sampling grid [B, X, Y, Z, 3]
            grid = torch.stack([shifted_z, shifted_y, shifted_x], dim=-1)
            
            # Apply grid_sample once for all samples with shifts
            for c in range(channels):
                output[has_shift, c:c+1] = F.grid_sample(
                    batch_tensor[has_shift, c:c+1], 
                    grid[has_shift], 
                    mode='nearest',
                    padding_mode='zeros',
                    align_corners=True
                )
        
        # Convert back to original format if needed
        if not dims_first:
            output = output.permute(0, 2, 3, 4, 1)
            
        return output
    
    @staticmethod
    def batch_rotation(
        batch_tensor: torch.Tensor,
        angles: torch.Tensor,
        center: Tuple[int, int, int] = None,
        dims_first: bool = True
    ) -> torch.Tensor:
        """
        Efficient batch rotation implementation focusing on correct center-of-rotation behavior.
        
        Args:
            batch_tensor: Tensor of shape [B, C, D, H, W] or [B, D, H, W, C]
            angles: Rotation angles in radians with shape [B, 3]
            center: Center of rotation as (d, h, w) or None for center of volume
            dims_first: Whether dimensions come first (BCDHW) or last (BDHWC)
        
        Returns:
            Rotated tensor with same shape as input
        """
        if not dims_first:
            batch_tensor = batch_tensor.permute(0, 4, 1, 2, 3)
        
        batch_size, channels, depth, height, width = batch_tensor.shape
        device = batch_tensor.device
        
        # Skip if no rotation needed for any sample
        if torch.all(angles == 0):
            return batch_tensor.clone()
        
        # Use center of volume if not specified
        if center is None:
            center = (depth / 2, height / 2, width / 2)
        
        # Normalized center coordinates [-1, 1]
        center_d = 2.0 * center[0] / (depth - 1) - 1.0
        center_h = 2.0 * center[1] / (height - 1) - 1.0
        center_w = 2.0 * center[2] / (width - 1) - 1.0
        
        # Identify samples with non-zero rotation
        has_rotation = (angles != 0).any(dim=1)
        
        # Create output tensor - start with copies of the input
        output = batch_tensor.clone()
        
        if not has_rotation.any():
            if not dims_first:
                output = output.permute(0, 2, 3, 4, 1)
            return output
        
        # Group by unique rotation angles to avoid redundant computation
        unique_angles, inverse_indices = torch.unique(angles[has_rotation], dim=0, return_inverse=True)
        num_unique = unique_angles.shape[0]
        
        # Create base sampling grid (normalized coordinates)
        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, depth, device=device),
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing='ij'
        )
        
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)  # [D, H, W, 3]
        
        # Process each unique rotation
        for i in range(num_unique):
            angle_x, angle_y, angle_z = unique_angles[i]
            
            # Find which samples use this rotation
            curr_samples = torch.where(has_rotation)[0][inverse_indices == i]
            
            # No need to process if empty
            if len(curr_samples) == 0:
                continue
            
            # 1. Create a copy of the base grid for this rotation
            grid = base_grid.clone()
            
            # 2. Translate points to make center the origin
            grid[..., 0] -= center_w  # X coordinate (width)
            grid[..., 1] -= center_h  # Y coordinate (height)
            grid[..., 2] -= center_d  # Z coordinate (depth)
            
            # 3. Apply rotation (Z, Y, X order - extrinsic rotations)
            # X-axis rotation
            if angle_x != 0:
                cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
                y, z = grid[..., 1].clone(), grid[..., 2].clone()
                grid[..., 1] = y * cos_x - z * sin_x
                grid[..., 2] = y * sin_x + z * cos_x
            
            # Y-axis rotation
            if angle_y != 0:
                cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
                x, z = grid[..., 0].clone(), grid[..., 2].clone()
                grid[..., 0] = x * cos_y + z * sin_y
                grid[..., 2] = -x * sin_y + z * cos_y
            
            # Z-axis rotation
            if angle_z != 0:
                cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)
                x, y = grid[..., 0].clone(), grid[..., 1].clone()
                grid[..., 0] = x * cos_z - y * sin_z
                grid[..., 1] = x * sin_z + y * cos_z
            
            # 4. Translate back to original center
            grid[..., 0] += center_w
            grid[..., 1] += center_h  
            grid[..., 2] += center_d
            
            # 5. Expand grid for all samples with this rotation and apply
            grid_batch = grid.unsqueeze(0).expand(len(curr_samples), -1, -1, -1, -1)
            
            # Apply the grid to all affected samples at once
            output[curr_samples] = F.grid_sample(
                batch_tensor[curr_samples],
                grid_batch,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
        
        if not dims_first:
            output = output.permute(0, 2, 3, 4, 1)
        
        return output

    @staticmethod
    def batch_elastic_deformation(
        batch_tensor: torch.Tensor,
        deformation_strengths: torch.Tensor,
        sigma: float = 4.0,
        dims_first: bool = True
    ) -> torch.Tensor:
        """Vectorized elastic deformation processing all samples at once"""

        if not dims_first:
            batch_tensor = batch_tensor.permute(0, 4, 1, 2, 3)
        
        batch_size, channels, x_dim, y_dim, z_dim = batch_tensor.shape
        device = batch_tensor.device
        
        # Create output tensor
        output = batch_tensor.clone()
        
        # Create mask for samples with non-zero deformation
        has_deform = deformation_strengths > 0
        
        if not has_deform.any():
            # No samples need deformation
            if not dims_first:
                output = output.permute(0, 2, 3, 4, 1)
            return output
        
        # Process only samples with deformation
        deform_batch_size = has_deform.sum().item()
        
        # Create random displacement field for all samples at once
        random_x = torch.randn((deform_batch_size, 1, x_dim, y_dim, z_dim), device=device)
        random_y = torch.randn((deform_batch_size, 1, x_dim, y_dim, z_dim), device=device)
        random_z = torch.randn((deform_batch_size, 1, x_dim, y_dim, z_dim), device=device)
        
        # Apply Gaussian smoothing with properly sized kernel
        kernel_size = min(7, x_dim//4, y_dim//4, z_dim//4)
        if kernel_size % 2 == 0:  # Ensure kernel size is odd
            kernel_size += 1
        padding = kernel_size // 2
        
        # Create Gaussian kernel with explicit size control
        grid = torch.linspace(-padding, padding, kernel_size, device=device)
        kernel_x = torch.exp(-grid**2 / (2*sigma**2))
        kernel_x = kernel_x / kernel_x.sum()
        
        kernel_y = kernel_x.clone()
        kernel_z = kernel_x.clone()
        
        # Reshape kernels for 3D convolution
        kernel_x = kernel_x.view(1, 1, kernel_size, 1, 1)
        kernel_y = kernel_y.view(1, 1, 1, kernel_size, 1)
        kernel_z = kernel_z.view(1, 1, 1, 1, kernel_size)
        
        # Apply separable Gaussian convolution
        dx = F.conv3d(random_x, kernel_x.expand(deform_batch_size, 1, -1, 1, 1),
                    padding=(padding, 0, 0), groups=deform_batch_size)
        dy = F.conv3d(random_y, kernel_y.expand(deform_batch_size, 1, 1, -1, 1),
                    padding=(0, padding, 0), groups=deform_batch_size)
        dz = F.conv3d(random_z, kernel_z.expand(deform_batch_size, 1, 1, 1, -1),
                    padding=(0, 0, padding), groups=deform_batch_size)
        
        # Apply scaling - vectorized across all samples
        strengths = deformation_strengths[has_deform].view(-1, 1, 1, 1, 1)
        dx = dx / (dx.std(dim=(2, 3, 4), keepdim=True) + 1e-8) * strengths
        dy = dy / (dy.std(dim=(2, 3, 4), keepdim=True) + 1e-8) * strengths
        dz = dz / (dz.std(dim=(2, 3, 4), keepdim=True) + 1e-8) * strengths
        
        # Create coordinate grid for all samples at once
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.arange(x_dim, device=device),
            torch.arange(y_dim, device=device),
            torch.arange(z_dim, device=device),
            indexing="ij"
        )
        
        # Expand for batch dimension
        grid_x = grid_x.expand(deform_batch_size, -1, -1, -1)
        grid_y = grid_y.expand(deform_batch_size, -1, -1, -1)
        grid_z = grid_z.expand(deform_batch_size, -1, -1, -1)
        
        # Add displacement
        sample_x = grid_x.float() + dx.squeeze(1)
        sample_y = grid_y.float() + dy.squeeze(1)  
        sample_z = grid_z.float() + dz.squeeze(1)
        
        # Normalize to [-1, 1] for grid_sample
        sample_x = 2.0 * sample_x / (x_dim - 1) - 1.0
        sample_y = 2.0 * sample_y / (y_dim - 1) - 1.0
        sample_z = 2.0 * sample_z / (z_dim - 1) - 1.0
        
        # Create sampling grid
        grid = torch.stack([sample_z, sample_y, sample_x], dim=-1)
        
        # Apply deformation to all samples with deformation
        deform_indices = torch.where(has_deform)[0]
        for c in range(channels):
            output[deform_indices, c:c+1] = F.grid_sample(
                batch_tensor[deform_indices, c:c+1],
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
        
        if not dims_first:
            output = output.permute(0, 2, 3, 4, 1)
            
        return output
    
    @staticmethod
    def _gaussian_smoothing(tensor, sigma):
        """
        Apply gaussian smoothing to a 3D tensor.
        Simple implementation using sequential 1D convolutions.
        """
        # Get device and size
        device = tensor.device
        size = int(sigma * 4) + 1
        if size % 2 == 0:
            size += 1  # Ensure odd size for symmetric kernel
        
        # Create 1D gaussian kernel
        grid = torch.arange(size, device=device) - size // 2
        kernel_1d = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply 1D convolutions sequentially along each dimension
        padded = F.pad(tensor, [size//2]*6, mode='replicate')
        
        # X dimension
        weight = kernel_1d.view(1, 1, size, 1, 1).to(device)
        smoothed = F.conv3d(padded.unsqueeze(0).unsqueeze(0), weight, 
                            padding=0).squeeze(0).squeeze(0)
        
        # Y dimension
        weight = kernel_1d.view(1, 1, 1, size, 1).to(device)
        smoothed = F.conv3d(smoothed.unsqueeze(0).unsqueeze(0), weight, 
                           padding=0).squeeze(0).squeeze(0)
        
        # Z dimension
        weight = kernel_1d.view(1, 1, 1, 1, size).to(device)
        smoothed = F.conv3d(smoothed.unsqueeze(0).unsqueeze(0), weight, 
                           padding=0).squeeze(0).squeeze(0)
        
        return smoothed
    
    @staticmethod
    def batch_intensity_scaling(
        batch_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        dims_first: bool = True
    ) -> torch.Tensor:
        """Fully vectorized intensity scaling"""
    
        if not dims_first:
            batch_tensor = batch_tensor.permute(0, 4, 1, 2, 3)
        
        # Check if scale_factors is per-sample or per-channel
        per_channel = (scale_factors.dim() > 1)
        
        # Apply scaling to entire batch at once
        if per_channel:
            # Reshape for broadcasting: [B, C, 1, 1, 1]
            scale_factors = scale_factors.view(scale_factors.shape[0], scale_factors.shape[1], 1, 1, 1)
        else:
            # Reshape for broadcasting: [B, 1, 1, 1, 1]
            scale_factors = scale_factors.view(-1, 1, 1, 1, 1)
        
        # Vectorized scaling
        output = batch_tensor * scale_factors
        
        if not dims_first:
            output = output.permute(0, 2, 3, 4, 1)
            
        return output
    
    @staticmethod
    def batch_gaussian_noise(
        batch_tensor: torch.Tensor,
        noise_std: torch.Tensor,
        dims_first: bool = True
    ) -> torch.Tensor:
        """Fully vectorized Gaussian noise addition"""
    
        if not dims_first:
            batch_tensor = batch_tensor.permute(0, 4, 1, 2, 3)
        
        batch_size, channels, x_dim, y_dim, z_dim = batch_tensor.shape
        
        # Check if noise_std is per-sample or per-channel
        per_channel = (noise_std.dim() > 1)
        
        # Create mask for samples with non-zero noise
        if per_channel:
            has_noise = (noise_std > 0).any(dim=1)
            # Reshape for broadcasting: [B, C, 1, 1, 1]
            noise_std = noise_std.view(noise_std.shape[0], noise_std.shape[1], 1, 1, 1)
        else:
            has_noise = (noise_std > 0)
            # Reshape for broadcasting: [B, 1, 1, 1, 1]
            noise_std = noise_std.view(-1, 1, 1, 1, 1)
        
        # Start with original data
        output = batch_tensor.clone()
        
        # Only generate noise for samples that need it
        if has_noise.any():
            if per_channel:
                # Generate noise for each channel separately
                noise_shape = (has_noise.sum().item(), channels, x_dim, y_dim, z_dim)
                noise_mask = noise_std[has_noise] > 0
                
                # Create noise tensor
                noise = torch.zeros(noise_shape, device=batch_tensor.device)
                noise[noise_mask] = torch.randn(noise_mask.sum().item(), device=batch_tensor.device)
                
                # Scale by standard deviation
                noise = noise * noise_std[has_noise]
            else:
                # Generate the same noise for all channels
                noise = torch.randn(
                    (has_noise.sum().item(), channels, x_dim, y_dim, z_dim),
                    device=batch_tensor.device
                ) * noise_std[has_noise]
            
            # Add noise only to samples that need it
            output[has_noise] = output[has_noise] + noise
        
        if not dims_first:
            output = output.permute(0, 2, 3, 4, 1)
            
        return output


class TrivialAugment:
    """
    Implementation of the TrivialAugment strategy based on the paper.
    
    For each sample in a batch:
    1. Randomly select ONE augmentation type
    2. Apply that augmentation with a random strength sampled uniformly from [0, max_strength]
    3. Never stack different augmentations (keeping it simple)
    """
    
    def __init__(self, 
                 max_rotation_degrees: float = 30.0,
                 max_shift: int = 5,
                 max_elastic_strength: float = 3.0,
                 max_intensity_factor: float = 0.1,  # As delta from 1.0
                 max_noise_std: float = 0.05,
                 center: tuple = (23, 23, 23)):
        """
        Initialize TrivialAugment with max strength parameters for each augmentation type.
        
        Args:
            max_rotation_degrees: Maximum rotation angle in degrees
            max_shift: Maximum spatial shift in voxels
            max_elastic_strength: Maximum elastic deformation strength
            max_intensity_factor: Maximum intensity factor as delta from 1.0
            max_noise_std: Maximum noise standard deviation
            center: Center of rotation as (d, h, w)
        """
        self.max_params = {
            'rotation': max_rotation_degrees,
            'shift': max_shift,
            'elastic': max_elastic_strength,
            'intensity': max_intensity_factor,
            'noise': max_noise_std
        }
        self.center = center
        
    def apply_to_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply TrivialAugment to a batch of samples.
        
        Args:
            batch: Dictionary with keys 'input_features' and 'target_efield'
            
        Returns:
            Batch with augmented samples
        """
        from ..pipeline.tms_data_types import TMSProcessedData  # Import here to avoid circular imports
        
        # Get batch size and device
        batch_size = batch['input_features'].shape[0]
        device = batch['input_features'].device
        
        # Determine feature format (channels-first or channels-last)
        if len(batch['input_features'].shape) == 5:
            dims_first = batch['input_features'].shape[1] <= 10
        else:
            dims_first = True
        
        # Get number of channels
        if dims_first:
            num_channels = batch['input_features'].shape[1]
        else:
            num_channels = batch['input_features'].shape[-1]
        
        # List of available augmentation types
        aug_types = list(self.max_params.keys())
        
        # For each sample, select a random augmentation type
        selected_augs = [random.choice(aug_types) for _ in range(batch_size)]
        
        # Create tensors for each augmentation type's parameters
        rotation_angles = torch.zeros((batch_size, 3), device=device)
        shifts = torch.zeros((batch_size, 3), device=device)
        elastic_strengths = torch.zeros(batch_size, device=device)
        intensity_factors = torch.ones(batch_size, device=device)
        noise_stds = torch.zeros(batch_size, device=device)
        
        # For each sample, set parameters for the selected augmentation type
        for i, aug_type in enumerate(selected_augs):
            if aug_type == 'rotation':
                # Random angle in [0, max_rotation_degrees] in radians
                angle_degrees = torch.rand(1, device=device) * self.max_params['rotation']
                angle_radians = angle_degrees * (np.pi / 180.0)
                rotation_angles[i, 1] = angle_radians  # Y-axis rotation
            
            elif aug_type == 'shift':
                # Random shift in [-max_shift, max_shift]
                max_shift = self.max_params['shift']
                for j in range(3):
                    shifts[i, j] = torch.randint(-max_shift, max_shift+1, (1,), device=device)
            
            elif aug_type == 'elastic':
                # Random strength in [0, max_elastic_strength]
                elastic_strengths[i] = torch.rand(1, device=device) * self.max_params['elastic']
            
            elif aug_type == 'intensity':
                # Random factor around 1.0 with max delta
                max_delta = self.max_params['intensity']
                intensity_factors[i] = 1.0 + (torch.rand(1, device=device) * 2 * max_delta - max_delta)
            
            elif aug_type == 'noise':
                # Random std in [0, max_noise_std]
                noise_stds[i] = torch.rand(1, device=device) * self.max_params['noise']
        
        # Apply the selected augmentations (one per sample)
        
        # 1. Apply rotation if any samples need it
        if (rotation_angles != 0).any():
            batch['input_features'] = BatchAugmentation.batch_rotation(
                batch['input_features'],
                rotation_angles,
                center=self.center,
                dims_first=dims_first
            )
            # Apply same rotation to target E-fields
            batch['target_efield'] = BatchAugmentation.batch_rotation(
                batch['target_efield'],
                rotation_angles,
                center=self.center,
                dims_first=dims_first
            )
        
        # 2. Apply elastic deformation if any samples need it
        if (elastic_strengths > 0).any():
            batch['input_features'] = BatchAugmentation.batch_elastic_deformation(
                batch['input_features'],
                elastic_strengths,
                sigma=4.0,  # Fixed sigma
                dims_first=dims_first
            )
            # Apply same deformation to target E-fields
            batch['target_efield'] = BatchAugmentation.batch_elastic_deformation(
                batch['target_efield'],
                elastic_strengths,
                sigma=4.0,
                dims_first=dims_first
            )
        
        # 3. Apply intensity scaling if any samples need it (input features only)
        if (intensity_factors != 1.0).any():
            batch['input_features'] = BatchAugmentation.batch_intensity_scaling(
                batch['input_features'],
                intensity_factors,
                dims_first=dims_first
            )
        
        # 4. Apply Gaussian noise if any samples need it (input features only)
        if (noise_stds > 0).any():
            batch['input_features'] = BatchAugmentation.batch_gaussian_noise(
                batch['input_features'],
                noise_stds,
                dims_first=dims_first
            )
        
        # 5. Apply spatial shift if any samples need it
        if (shifts != 0).any():
            batch['input_features'] = BatchAugmentation.batch_spatial_shift(
                batch['input_features'],
                shifts,
                dims_first=dims_first
            )
            # Apply same shifts to target E-fields
            batch['target_efield'] = BatchAugmentation.batch_spatial_shift(
                batch['target_efield'],
                shifts,
                dims_first=dims_first
            )
        
        # Update metadata for augmented samples
        if 'metadata' in batch and isinstance(batch['metadata'], list):
            for i, aug_type in enumerate(selected_augs):
                if i < len(batch['metadata']) and batch['metadata'][i]:
                    batch['metadata'][i]['augmented'] = True
                    batch['metadata'][i]['trivial_augment_type'] = aug_type
                    
                    # Add specific augmentation parameters
                    if aug_type == 'rotation':
                        batch['metadata'][i]['augmentation_rotation'] = rotation_angles[i].cpu().tolist()
                    elif aug_type == 'shift':
                        batch['metadata'][i]['augmentation_shift'] = shifts[i].cpu().tolist()
                    elif aug_type == 'elastic':
                        batch['metadata'][i]['augmentation_elastic'] = float(elastic_strengths[i].cpu())
                    elif aug_type == 'intensity':
                        batch['metadata'][i]['augmentation_intensity'] = float(intensity_factors[i].cpu())
                    elif aug_type == 'noise':
                        batch['metadata'][i]['augmentation_noise'] = float(noise_stds[i].cpu())
        
        return batch


# Simplified DataLoader with TrivialAugment support
class TMSDataLoader(torch.utils.data.DataLoader):
    """DataLoader with TrivialAugment support."""
    
    def __init__(self, 
                 dataset,
                 batch_size=8,
                 shuffle=False,
                 trivial_augment=None,
                 **kwargs):
        """
        Initialize DataLoader with TrivialAugment.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            trivial_augment: TrivialAugment instance or configuration dict
            **kwargs: Additional arguments for DataLoader
        """
        # Initialize TrivialAugment if provided
        if isinstance(trivial_augment, dict):
            self.trivial_augment = TrivialAugment(**trivial_augment)
        else:
            self.trivial_augment = trivial_augment
        
        # Use custom collate function for batch processing
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = self._collate_fn
            
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def _collate_fn(self, batch):
        """Custom collate function that applies TrivialAugment at batch level."""
        # Default collate to create tensors
        elem = batch[0]
        if isinstance(elem, TMSSample):
            # Convert samples to dictionaries for batch processing
            batch_dict = {
                'input_features': torch.stack([sample.input_features for sample in batch]),
                'target_efield': torch.stack([sample.target_efield for sample in batch]),
            }
            
            # Include metadata if available
            if any(sample.metadata for sample in batch):
                batch_dict['metadata'] = [sample.metadata for sample in batch]
                
            # Apply TrivialAugment if enabled
            if self.trivial_augment is not None:
                batch_dict = self.trivial_augment.apply_to_batch(batch_dict)
            
            return batch_dict
        else:
            # For other types, use default collation
            return torch.utils.data.dataloader.default_collate(batch)


def create_dataloader_with_trivial_augment(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    trivial_augment_params=None
):
    """
    Create efficient DataLoader with TrivialAugment.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        trivial_augment_params: Parameters for TrivialAugment
        
    Returns:
        DataLoader with TrivialAugment
    """
    return TMSDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        trivial_augment=trivial_augment_params
    )