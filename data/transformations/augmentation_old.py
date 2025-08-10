import torch
import numpy as np
from typing import Union, Optional, Tuple, List, Dict, Any
from ..pipeline.tms_data_types import TMSSample, TMSProcessedData


class BatchAugmentation:
    """
    Implements efficient batch-level augmentation for TMS data.
    Optimized for speed with fully vectorized operations.
    """
    
    @staticmethod
    def batch_spatial_shift(
        batch_tensor: torch.Tensor, 
        shifts: torch.Tensor,
        dims_first: bool = True
    ) -> torch.Tensor:
        """
        Apply different spatial shifts to each sample in a batch efficiently.
        
        Args:
            batch_tensor: Batch of tensors with shape [B, C, X, Y, Z] or [B, X, Y, Z, C]
            shifts: Tensor of shifts with shape [B, 3] containing (dx, dy, dz) for each sample
            dims_first: Whether spatial dimensions come before channels (True) or after (False)
            
        Returns:
            Shifted batch tensor with same shape as input
        """
        batch_size = batch_tensor.shape[0]
        device = batch_tensor.device
        
        # Handle channels-last format
        if not dims_first:
            # Convert to dimensions-first for processing
            batch_tensor = batch_tensor.permute(0, 4, 1, 2, 3)
        
        # Create empty output tensor
        output = torch.zeros_like(batch_tensor)
        
        # Get tensor dimensions
        if dims_first:
            _, channels, x_dim, y_dim, z_dim = batch_tensor.shape
        else:
            _, x_dim, y_dim, z_dim, channels = batch_tensor.shape
            
        # Process each sample in batch with vectorized operations
        for b in range(batch_size):
            # Get shifts for this sample
            dx, dy, dz = shifts[b]
            
            # Skip if no shift
            if dx == 0 and dy == 0 and dz == 0:
                output[b] = batch_tensor[b]
                continue
                
            # Calculate source and destination slices for this shift
            # Source = where to take data from original tensor
            # Dest = where to put data in output tensor
            x_src_start = max(0, -dx)
            x_src_end = min(x_dim, x_dim - dx)
            x_dst_start = max(0, dx)
            x_dst_end = min(x_dim, x_dim + dx)
            
            y_src_start = max(0, -dy)
            y_src_end = min(y_dim, y_dim - dy)
            y_dst_start = max(0, dy)
            y_dst_end = min(y_dim, y_dim + dy)
            
            z_src_start = max(0, -dz)
            z_src_end = min(z_dim, z_dim - dz)
            z_dst_start = max(0, dz)
            z_dst_end = min(z_dim, z_dim + dz)
            
            # Apply shift using slicing (very efficient)
            output[b, :, 
                   x_dst_start:x_dst_end, 
                   y_dst_start:y_dst_end, 
                   z_dst_start:z_dst_end] = batch_tensor[b, :,
                                                         x_src_start:x_src_end,
                                                         y_src_start:y_src_end,
                                                         z_src_start:z_src_end]
        
        # Convert back to original format if needed
        if not dims_first:
            output = output.permute(0, 2, 3, 4, 1)
            
        return output
    
    @staticmethod
    def generate_random_shifts(
        batch_size: int,
        max_shift: int = 3,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate random integer shifts for batch augmentation.
        
        Args:
            batch_size: Number of samples in batch
            max_shift: Maximum absolute shift distance in each dimension
            device: Target device for the tensor
            
        Returns:
            Tensor of shifts with shape [batch_size, 3]
        """
        # Generate random integer shifts in range [-max_shift, max_shift] for each dimension
        shifts = torch.randint(
            -max_shift, 
            max_shift + 1,  # +1 because randint upper bound is exclusive
            (batch_size, 3),
            device=device
        )
        
        return shifts
    
    @staticmethod
    def augment_batch_samples(
        batch: Dict[str, torch.Tensor],
        max_shift: int = 3
    ) -> Dict[str, torch.Tensor]:
        """
        Apply consistent spatial shifts to both inputs and targets in a batch.
        
        Args:
            batch: Dictionary with keys 'input_features' and 'target_efield'
            max_shift: Maximum absolute shift distance in each dimension
            
        Returns:
            Batch with augmented samples
        """
        batch_size = batch['input_features'].shape[0]
        device = batch['input_features'].device
        
        # Generate same random shifts for all samples in the batch
        shifts = BatchAugmentation.generate_random_shifts(
            batch_size, 
            max_shift=max_shift,
            device=device
        )
        
        # Determine feature format (channels-first or channels-last)
        if len(batch['input_features'].shape) == 5:
            # Check if last dimension is small (likely channels)
            dims_first = batch['input_features'].shape[1] <= 10
        else:
            # Default to dimensions-first if not 5D
            dims_first = True
            
        # Apply shifts to input features
        batch['input_features'] = BatchAugmentation.batch_spatial_shift(
            batch['input_features'],
            shifts,
            dims_first=dims_first
        )
        
        # Determine target format
        if len(batch['target_efield'].shape) == 5:
            # Check if last dimension is small (likely channels)
            dims_first = batch['target_efield'].shape[1] <= 10
        else:
            # Default to dimensions-first if not 5D
            dims_first = True
            
        # Apply SAME shifts to target E-fields
        batch['target_efield'] = BatchAugmentation.batch_spatial_shift(
            batch['target_efield'],
            shifts,
            dims_first=dims_first
        )
        
        # Update metadata if present
        if 'metadata' in batch:
            for i in range(batch_size):
                if isinstance(batch['metadata'], list):
                    if batch['metadata'][i]:
                        batch['metadata'][i]['augmented'] = True
                        batch['metadata'][i]['augmentation_type'] = 'spatial_shift'
                        batch['metadata'][i]['applied_shifts'] = shifts[i].cpu().tolist()
        
        return batch


# Efficient DataLoader with batch augmentation
class TMSDataLoader(torch.utils.data.DataLoader):
    """DataLoader with efficient batch-level augmentation."""
    
    def __init__(self, 
                 dataset,
                 batch_size=8,
                 shuffle=False,
                 augmentation_config=None,
                 **kwargs):
        """
        Initialize DataLoader with augmentation.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augmentation_config: Configuration for augmentation
            **kwargs: Additional arguments for DataLoader
        """
        self.augmentation_config = augmentation_config or {'enabled': False}
        
        # Use custom collate function for batch processing
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = self._collate_fn
            
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def _collate_fn(self, batch):
        """
        Custom collate function that applies augmentation at batch level.
        
        Args:
            batch: List of samples
            
        Returns:
            Collated batch with augmentation applied
        """
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
                
            # Apply augmentation if enabled
            if self.augmentation_config.get('enabled', False):
                shift_config = self.augmentation_config.get('spatial_shift', {})
                if shift_config.get('enabled', False):
                    max_shift = shift_config.get('max_shift', 3)
                    
                    # Apply augmentation to entire batch at once
                    batch_dict = BatchAugmentation.augment_batch_samples(
                        batch_dict,
                        max_shift=max_shift
                    )
            
            # Convert back to samples if needed, or return as dictionary
            return batch_dict
        else:
            # For other types, use default collation
            return torch.utils.data.dataloader.default_collate(batch)


# Example usage
def create_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    augmentation_config=None
):
    """
    Create efficient DataLoader with batch augmentation.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        augmentation_config: Augmentation configuration
        
    Returns:
        DataLoader with batch augmentation
    """
    return TMSDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        augmentation_config=augmentation_config
    )

def augmentation_collate_fn(batch, augmentation_config=None):
    """
    Standalone collate function that applies joint augmentation to batches.
    Can be used with standard PyTorch DataLoader.
    
    Args:
        batch: List of (feature, target) tuples from DataLoader
        augmentation_config: Configuration for augmentation
        
    Returns:
        Tuple of (features_batch, targets_batch) with augmentation applied
    """
    # Default collate to create tensors
    features = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    
    # Apply augmentation if enabled
    if augmentation_config and augmentation_config.get('enabled', False):
        # Create a batch dictionary
        batch_dict = {
            'input_features': features,
            'target_efield': targets
        }
        
        # Apply augmentation
        shift_config = augmentation_config.get('spatial_shift', {})
        if shift_config.get('enabled', False):
            max_shift = shift_config.get('max_shift', 3)
            
            # Apply augmentation to entire batch at once
            batch_dict = BatchAugmentation.augment_batch_samples(
                batch_dict,
                max_shift=max_shift
            )
            
        # Extract augmented tensors
        features = batch_dict['input_features']
        targets = batch_dict['target_efield']
    
    return features, targets