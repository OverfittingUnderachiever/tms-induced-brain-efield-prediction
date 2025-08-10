# tms_efield_prediction/data/transformations/enhanced_stack_pipeline.py
"""
Enhanced stacking pipeline for TMS E-field prediction.

This module provides functionality for stacking MRI and field data
using the improved voxel mapping and field processing components.
"""

import numpy as np
import time
import os
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import torch
import logging
import multiprocessing as mp

from utils.debug.hooks import PipelineDebugHook
from utils.resource.monitor import ResourceMonitor
from utils.state.context import TMSPipelineContext
from utils.pipeline.implementation_unit import ImplementationUnit, UnitResult, UnitPipeline
from ..pipeline.tms_data_types import TMSRawData, TMSProcessedData, TMSSample
from ..pipeline.field_processor import FieldProcessor, FieldProcessingConfig

logger = logging.getLogger(__name__)

# Reduce output verbosity
logger.setLevel(logging.INFO)  # Change to INFO from DEBUG to reduce verbosity


@dataclass
class EnhancedStackingConfig:
    """Configuration for enhanced stacking pipeline."""
    normalization_method: str = "standard"  # Options: "standard", "minmax", "robust"
    dadt_scaling_factor: float = 1.0e-6  # Scaling factor for dA/dt fields
    output_shape: Tuple[int, int, int] = (64, 64, 64)  # Output shape for voxelized data
    bin_size: int = 64  # Bin size for voxelization
    channel_stacking_mode: str = "mri_efield"  # Options: "mri_efield", "mri_dadt", "mri_efield_dadt"
    use_parallel_processing: bool = True  # Whether to use parallel processing
    use_cache: bool = True  # Whether to use caching
    cache_dir: Optional[str] = None  # Directory for cache files


class EnhancedStackingPipeline:
    """Enhanced pipeline for stacking MRI and field data."""

    def __init__(
        self,
        context: TMSPipelineContext,
        config: Optional[EnhancedStackingConfig] = None,
        debug_hook: Optional[PipelineDebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        use_cache: bool = False,  # Add caching parameter
        cache_dir: Optional[str] = None  # Add cache directory parameter
    ):
        """Initialize the enhanced stacking pipeline.

        Args:
            context: TMS-specific pipeline context
            config: Optional configuration for stacking
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
            use_cache: Whether to use caching (default: True)
            cache_dir: Directory for cache files (default: None, auto-generated)
        """
        self.context = context
        self.config = config or EnhancedStackingConfig(
            normalization_method=context.normalization_method,
            dadt_scaling_factor=getattr(context, 'dadt_scaling_factor', 1.0e-6),
            output_shape=context.output_shape,
            bin_size=context.output_shape[0],
            channel_stacking_mode=context.pipeline_mode,
            use_cache=use_cache,
            cache_dir=cache_dir
        )
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor

        # Set up caching
        self.use_cache = self.config.use_cache if hasattr(self.config, 'use_cache') else use_cache
        
        # Set up cache directory
        if hasattr(self.config, 'cache_dir') and self.config.cache_dir:
            self.cache_dir = self.config.cache_dir
        elif cache_dir:
            self.cache_dir = cache_dir
        else:
            # Default to a cache directory in the data root path
            self.cache_dir = os.path.join(self.context.data_root_path, "cache")
        
        # Create cache directory if it doesn't exist and caching is enabled
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Created cache directory: {self.cache_dir}")

        # Instantiate voxel mapper and field processor
        self.voxel_mapper = None  # Lazy initialization

        # Create field processor with custom configuration
        field_config = FieldProcessingConfig(
            bin_size=self.config.bin_size,
            n_processes=mp.cpu_count() if self.config.use_parallel_processing else 1,
            save_torch=False,  # Don't save to disk by default
            save_numpy=False,
            save_pickle=False
        )

        self.field_processor = FieldProcessor(
            context=context,
            config=field_config,
            debug_hook=debug_hook,
            resource_monitor=resource_monitor
        )

        # Register with resource monitor if provided
        if self.resource_monitor:
            self.resource_monitor.register_component(
                "enhanced_stacking_pipeline",
                self._reduce_memory,
                priority=15  # Medium-high priority
            )

        # Memory tracking
        self._memory_usage = 0
        self._intermediate_data = {}
        
        logger.info(f"Initialized EnhancedStackingPipeline with caching {'enabled' if self.use_cache else 'disabled'}")
        
    def _generate_cache_key(self, sample: TMSSample) -> str:
        """
        Generate a unique cache key for the sample based on its properties.
        
        Args:
            sample: The TMSSample to generate a key for
            
        Returns:
            A unique string key for the sample
        """
        # Create a string that uniquely identifies this sample and processing configuration
        key_components = [
            f"subject:{sample.subject_id}",
            f"coil_pos:{sample.coil_position_idx}",
            f"mode:{self.context.pipeline_mode}",
            f"shape:{'-'.join(str(x) for x in self.context.output_shape)}",
            f"norm:{self.context.normalization_method}"
        ]
        key_str = "_".join(key_components)
        
        # Create a hash of the key string
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, sample: TMSSample) -> str:
        """
        Get the cache file path for a sample.
        
        Args:
            sample: The TMSSample to get the cache path for
            
        Returns:
            The path to the cache file
        """
        key = self._generate_cache_key(sample)
        return os.path.join(self.cache_dir, f"{key}.pt")
    
    def _check_cache(self, sample: TMSSample) -> Optional[TMSProcessedData]:
        """
        Check if processed data for this sample is in the cache.
        
        Args:
            sample: The TMSSample to check cache for
            
        Returns:
            TMSProcessedData if found in cache, None otherwise
        """
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(sample)
        
        if os.path.exists(cache_path):
            try:
                cached_data = torch.load(cache_path, map_location='cpu')
                if isinstance(cached_data, TMSProcessedData):
                    logger.debug(f"Cache hit for sample {sample.sample_id}")
                    return cached_data
                else:
                    logger.warning(f"Cache file exists but contains invalid data for sample {sample.sample_id}")
            except Exception as e:
                logger.warning(f"Failed to load cached data for {sample.sample_id}: {e}")
        
        return None
    
    def _save_to_cache(self, sample: TMSSample, processed_data: TMSProcessedData) -> bool:
        """
        Save processed data to cache.
        
        Args:
            sample: The TMSSample that was processed
            processed_data: The processed data to cache
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.use_cache:
            return False
        
        cache_path = self._get_cache_path(sample)
        
        try:
            torch.save(processed_data, cache_path)
            logger.debug(f"Saved processed data to cache for sample {sample.sample_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save cache for {sample.sample_id}: {e}")
            return False

    # Modified section for EnhancedStackingPipeline to handle stacked arrays

    def process_sample(self, sample: TMSSample) -> TMSProcessedData:
        """
        Process a single TMS sample.
        
        Args:
            sample: TMS sample to process
            
        Returns:
            Processed data
        """
        start_time = time.time()
        
        # Log start if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "process_sample_started",
                {
                    'sample_id': sample.sample_id,
                    'using_stacked_array': sample.metadata.get('using_stacked_array', False)
                }
            )
        
        try:
            # Check if using stacked array
            if sample.metadata.get('using_stacked_array', False) and 'stacked_path' in sample.metadata:
                # Load directly from stacked array
                stacked_path = sample.metadata['stacked_path']
                
                # Load stacked array data
                stacked_data = torch.load(stacked_path, map_location='cpu')
                
                # Extract input features and target
                input_features = stacked_data['input_features']
                target_efield = stacked_data['target_efield']
                
                # Get metadata if available
                metadata = stacked_data.get('metadata', {}).copy()
                
                # Ensure input_features and target_efield are PyTorch tensors
                if isinstance(input_features, np.ndarray):
                    input_features = torch.from_numpy(input_features)
                
                if isinstance(target_efield, np.ndarray):
                    target_efield = torch.from_numpy(target_efield)
                
                # Convert from [depth, height, width, channels] to [channels, depth, height, width]
                if len(input_features.shape) == 4 and input_features.shape[-1] <= 32:  
                    # Permute from [depth, height, width, channels] to [channels, depth, height, width]
                    input_features = input_features.permute(3, 0, 1, 2)
                    
                # For target, add channel dimension if it's 3D
                if len(target_efield.shape) == 3:
                    # Add channel dimension [depth, height, width] -> [1, depth, height, width]
                    target_efield = target_efield.unsqueeze(0)
                
                # Add sample id to metadata
                metadata['sample_id'] = sample.sample_id
                metadata['source'] = 'stacked_array'
                metadata['stacked_path'] = stacked_path
                
                # Create processed data
                processed_data = TMSProcessedData(
                    subject_id=sample.subject_id,
                    input_features=input_features,  # Now in channels-first format
                    target_efield=target_efield,    # Now with channel dimension
                    mask=None,  # No explicit mask in stacked data
                    metadata=metadata
                )
            else:
                # Original processing flow for separate files
                # Load MRI data directly from context
                mri_tensor = self.context.config.get('mri_tensor')
                if mri_tensor is None:
                    raise ValueError("MRI tensor not available in context")
                
                # Load E-field and dA/dt data
                efield_data = torch.load(sample.efield_data, map_location='cpu')
                dadt_data = torch.load(sample.dadt_data, map_location='cpu')
                
                # Convert to numpy for processing
                if isinstance(mri_tensor, torch.Tensor):
                    mri_data = mri_tensor.numpy()
                else:
                    mri_data = mri_tensor
                    
                if isinstance(efield_data, torch.Tensor):
                    efield_data = efield_data.numpy()
                if isinstance(dadt_data, torch.Tensor):
                    dadt_data = dadt_data.numpy()
                
                # Process MRI data
                normalized_mri = self._normalize_mri(mri_data)
                
                # Stack with dA/dt data
                stacked_data = self._stack_mri_field(normalized_mri, dadt_data)
                
                # Convert back to PyTorch tensors
                stacked_tensor = torch.from_numpy(stacked_data)
                efield_tensor = torch.from_numpy(efield_data)
                
                # Convert stacked_tensor from [depth, height, width, channels] to [channels, depth, height, width]
                if len(stacked_tensor.shape) == 4 and stacked_tensor.shape[-1] in [3, 4, 6]:
                    stacked_tensor = stacked_tensor.permute(3, 0, 1, 2)
                    
                # For efield_tensor, add channel dimension if it's 3D
                if len(efield_tensor.shape) == 3:
                    efield_tensor = efield_tensor.unsqueeze(0)
                
                # Create processed data
                processed_data = TMSProcessedData(
                    subject_id=sample.subject_id,
                    input_features=stacked_tensor,
                    target_efield=efield_tensor,
                    mask=None,
                    metadata={
                        'sample_id': sample.sample_id,
                        'source': 'separate_files'
                    }
                )
            
            # Log completion if debug enabled
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "process_sample_completed",
                    {
                        'sample_id': sample.sample_id,
                        'execution_time': time.time() - start_time,
                        'input_shape': processed_data.input_features.shape,
                        'target_shape': processed_data.target_efield.shape if processed_data.target_efield is not None else None,
                        'using_stacked_array': sample.metadata.get('using_stacked_array', False)
                    }
                )
            
            return processed_data
        
        except Exception as e:
            error_msg = f"Error processing sample {sample.sample_id}: {e}"
            logger.error(error_msg, exc_info=True)
            
            if self.debug_hook:
                self.debug_hook.record_error(
                    "process_sample_error",
                    {
                        'sample_id': sample.sample_id,
                        'error': str(e),
                        'using_stacked_array': sample.metadata.get('using_stacked_array', False)
                    }
                )
            
            raise ValueError(error_msg)

    def process_batch(self, samples: List[TMSSample]) -> List[TMSProcessedData]:
        """Process a batch of TMS samples.

        Args:
            samples: List of TMS samples

        Returns:
            List of processed TMS data
        """
        start_time = time.time()

        # Log start if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "process_batch_started",
                {'batch_size': len(samples)}
            )

        # Process each sample
        processed_data = []
        cache_hits = 0
        
        for sample in samples:
            try:
                # First check cache
                cached_data = self._check_cache(sample)
                if cached_data is not None:
                    processed_data.append(cached_data)
                    cache_hits += 1
                    continue
                
                # Not in cache, process normally
                processed = self.process_sample(sample)
                processed_data.append(processed)
            except Exception as e:
                # Log error
                if self.debug_hook:
                    self.debug_hook.record_error(
                        e,
                        {
                            'sample_id': sample.sample_id,
                            'phase': 'enhanced_stacking_pipeline'
                        }
                    )
                # Re-raise
                raise

        # Log cache statistics
        if self.use_cache:
            logger.info(f"Processed {len(samples)} samples with {cache_hits} cache hits "
                       f"({cache_hits/len(samples)*100:.1f}% cache hit rate)")

        # Log completion if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "process_batch_completed",
                {
                    'batch_size': len(samples),
                    'execution_time': time.time() - start_time,
                    'cache_hits': cache_hits
                }
            )

        return processed_data

    def _stack_mri_field(self, mri_grid: torch.Tensor, field_grid: torch.Tensor) -> torch.Tensor:
        """Stack MRI and field data. Handles both scalar and vector MRI/field data.

        Args:
            mri_grid: MRI grid data with shape [D, H, W] or [3, D, H, W] or [D, H, W, 3]
            field_grid: Field grid data with shape [D, H, W] or [3, D, H, W] or [D, H, W, 3]

        Returns:
            Stacked data with shape [C, D, H, W] where C = num_mri_channels + num_field_channels
        """
        device = torch.device(self.context.config.get("device", "cpu"))
        mri_grid = mri_grid.to(device)
        field_grid = field_grid.to(device)

        mri_normalized = self._normalize_mri(mri_grid)

        # Ensure MRI has channel dimension
        if mri_normalized.ndim == 3:
            # Scalar MRI [D, H, W] -> [1, D, H, W]
            mri_expanded = mri_normalized.unsqueeze(0)
        elif mri_normalized.ndim == 4:
            if mri_normalized.shape[0] == 3 or mri_normalized.shape[0] == 1:
                # Already [C, D, H, W] format
                mri_expanded = mri_normalized
            elif mri_normalized.shape[-1] == 3:
                # [D, H, W, 3] -> [3, D, H, W]
                mri_expanded = mri_normalized.permute(3, 0, 1, 2)
            else:
                raise ValueError(f"Unexpected MRI grid shape: {mri_normalized.shape}")
        else:
            raise ValueError(f"Unexpected MRI grid dimensions: {mri_normalized.ndim}")

        # Ensure field has channel dimension
        if field_grid.ndim == 3:
            # Scalar field [D, H, W] -> [1, D, H, W]
            field_expanded = field_grid.unsqueeze(0)
        elif field_grid.ndim == 4:
            if field_grid.shape[0] == 3 or field_grid.shape[0] == 1:
                # Already [C, D, H, W] format
                field_expanded = field_grid
            elif field_grid.shape[-1] == 3:
                # [D, H, W, 3] -> [3, D, H, W]
                field_expanded = field_grid.permute(3, 0, 1, 2)
            else:
                raise ValueError(f"Unexpected field grid shape: {field_grid.shape}")
        else:
            raise ValueError(f"Unexpected field grid dimensions: {field_grid.ndim}")

        # Stack along channel dimension (dim=0)
        stacked_data = torch.cat([mri_expanded, field_expanded], dim=0)
        logger.debug(f"Stacked MRI ({mri_expanded.shape}) and Field ({field_expanded.shape}) -> {stacked_data.shape}")

        return stacked_data
        
    def _stack_mri_fields(self, mri_grid: torch.Tensor, field_grids: List[torch.Tensor]) -> torch.Tensor:
        """Stack MRI and multiple field data arrays.

        Args:
            mri_grid: MRI grid data
            field_grids: List of field grid data

        Returns:
            Stacked data
        """
        # Get the device from the context
        device = torch.device(self.context.config.get("device", "cpu"))

        # Move tensors to the correct device
        mri_grid = mri_grid.to(device)
        field_grids = [field.to(device) for field in field_grids]

        # Normalize MRI data
        mri_normalized = self._normalize_mri(mri_grid)

        # Prepare data for stacking
        data_to_stack = []

        # Convert MRI to proper shape
        if len(mri_normalized.shape) == 3:
            # Convert to [bin, bin, bin, 1]
            data_to_stack.append(mri_normalized.unsqueeze(0)) # Corrected unsqueeze dim
        else:
            data_to_stack.append(mri_normalized)

        # Add each field
        for field_grid in field_grids:
            if len(field_grid.shape) == 3:
                # Convert to [bin, bin, bin, 1]
                data_to_stack.append(field_grid.unsqueeze(0))# Corrected unsqueeze dim
            else:
                data_to_stack.append(field_grid)

        # Stack along first dimension (channel dimension)
        stacked_data = torch.cat(data_to_stack, dim=0) # correct concat dim

        return stacked_data

    def _normalize_mri(self, mri_data: torch.Tensor) -> torch.Tensor:
        """Normalize MRI data. Handles both scalar and vector MRI data.

        Args:
            mri_data: MRI data with shape [D, H, W] or [3, D, H, W] or [D, H, W, 3]

        Returns:
            Normalized MRI data with the same shape
        """
        # Get the device from the context
        device = torch.device(self.context.config.get("device", "cpu"))

        # Move MRI data to the correct device
        mri_data = mri_data.to(device)

        # Check if MRI data is vector (has channels)
        is_vector = False
        channels_last = False
        
        if mri_data.ndim == 4:
            if mri_data.shape[0] == 3:
                # Vector data with channels first [3, D, H, W]
                is_vector = True
            elif mri_data.shape[-1] == 3:
                # Vector data with channels last [D, H, W, 3]
                is_vector = True
                channels_last = True
        
        # Normalize based on channels
        if is_vector:
            # Ensure channels are first for normalization
            if channels_last:
                # Convert [D, H, W, C] to [C, D, H, W]
                mri_data = mri_data.permute(3, 0, 1, 2)
            
            # Normalize each channel separately
            channels = []
            for c in range(mri_data.shape[0]):
                channel_data = mri_data[c]
                
                if self.config.normalization_method == "standard":
                    # Z-score normalization
                    mean = torch.mean(channel_data)
                    std = torch.std(channel_data)
                    if std > 0:
                        norm_channel = (channel_data - mean) / std
                    else:
                        norm_channel = channel_data - mean
                elif self.config.normalization_method == "minmax":
                    # Min-max normalization
                    min_val = torch.min(channel_data)
                    max_val = torch.max(channel_data)
                    if max_val > min_val:
                        norm_channel = (channel_data - min_val) / (max_val - min_val)
                    else:
                        norm_channel = torch.zeros_like(channel_data)
                elif self.config.normalization_method == "robust":
                    # Robust normalization using percentiles
                    p10 = torch.quantile(channel_data.float(), 0.10)
                    p90 = torch.quantile(channel_data.float(), 0.90)
                    if p90 > p10:
                        norm_channel = (channel_data - p10) / (p90 - p10)
                    else:
                        norm_channel = torch.zeros_like(channel_data)
                else:
                    # No normalization
                    norm_channel = channel_data
                    
                channels.append(norm_channel)
            
            # Stack channels back together
            normalized_data = torch.stack(channels)
            
            # Convert back to original shape if needed
            if channels_last:
                normalized_data = normalized_data.permute(1, 2, 3, 0)
            
            return normalized_data
        else:
            # Original normalization for scalar data
            if self.config.normalization_method == "standard":
                # Z-score normalization
                mean = torch.mean(mri_data)
                std = torch.std(mri_data)
                if std > 0:
                    return (mri_data - mean) / std
                else:
                    return mri_data - mean
            elif self.config.normalization_method == "minmax":
                # Min-max normalization
                min_val = torch.min(mri_data)
                max_val = torch.max(mri_data)
                if max_val > min_val:
                    return (mri_data - min_val) / (max_val - min_val)
                else:
                    return torch.zeros_like(mri_data)
            elif self.config.normalization_method == "robust":
                # Robust normalization using percentiles
                p10 = torch.quantile(mri_data.float(), 0.10)
                p90 = torch.quantile(mri_data.float(), 0.90)
                if p90 > p10:
                    return (mri_data - p10) / (p90 - p10)
                else:
                    return torch.zeros_like(mri_data)
            else:
                # No normalization
                return mri_data

    def _update_memory_usage(self, key: str, data: Any) -> None:
        """Update memory usage tracking.

        Args:
            key: Identifier for the data object
            data: Data to track
        """
        # Calculate memory usage (approximate)
        if isinstance(data, np.ndarray):
            mem_bytes = data.nbytes
        elif isinstance(data, torch.Tensor):
            mem_bytes = data.element_size() * data.nelement() # Correct calculation for tensors
        elif isinstance(data, list) and all(isinstance(x, (np.ndarray, torch.Tensor)) for x in data):
            mem_bytes = sum(x.nbytes if isinstance(x, np.ndarray) else (x.element_size() * x.nelement()) for x in data)
        else:
            # For other types
            import sys
            mem_bytes = sys.getsizeof(data)

        # Store in intermediate data dictionary
        if key in self._intermediate_data:
            # Subtract old usage
            old_usage = self._intermediate_data[key]['bytes']
            self._memory_usage -= old_usage

        # Add new usage
        if hasattr(data, 'shape'):
            shape = data.shape
        elif isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'shape'):
            shape = f"list[{len(data)}] of {data[0].shape}"
        else:
            shape = str(type(data))

        if hasattr(data, 'dtype'):
            dtype = str(data.dtype)
        elif isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'dtype'):
            dtype = str(data[0].dtype)
        else:
            dtype = str(type(data))

        self._intermediate_data[key] = {
            'shape': shape,
            'dtype': dtype,
            'bytes': mem_bytes
        }
        self._memory_usage += mem_bytes

        # Update resource monitor if available
        if self.resource_monitor:
            self.resource_monitor.update_component_usage(
                "enhanced_stacking_pipeline",
                self._memory_usage
            )

    def _reduce_memory(self, target_reduction: float) -> None:
        """Callback for memory reduction requests.

        Args:
            target_reduction: Fraction of memory to reduce (0.0-1.0)
        """
        # Log memory reduction request
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "memory_reduction_requested",
                {
                    'target_reduction': target_reduction,
                    'current_memory': self._memory_usage
                }
            )

        if hasattr(self.field_processor, '_reduce_memory'):
            self.field_processor._reduce_memory(target_reduction)

        # Then, clear own intermediate data
        keys_to_clear = []
        for key, info in self._intermediate_data.items():
            keys_to_clear.append(key)
            self._memory_usage -= info['bytes']

            # Check if we've met the target
            if self._memory_usage / (1.0 - target_reduction) <= self._memory_usage:
                break

        # Actually clear the data
        for key in keys_to_clear:
            del self._intermediate_data[key]

        # Update resource monitor
        if self.resource_monitor:
            self.resource_monitor.update_component_usage(
                "enhanced_stacking_pipeline",
                self._memory_usage
            )

        # Log results
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "memory_reduction_complete",
                {
                    'cleared_keys': keys_to_clear,
                    'new_memory': self._memory_usage
                }
            )



class DADTMagnitudeStackingPipeline(EnhancedStackingPipeline):
    """Pipeline for TMS E-field prediction using dA/dt magnitude instead of vector."""
    
    def __init__(
        self,
        context: TMSPipelineContext,
        debug_hook: Optional[PipelineDebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """Initialize the dA/dt magnitude stacking pipeline.
        
        Args:
            context: TMS-specific pipeline context
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
            use_cache: Whether to use caching
            cache_dir: Directory for cache files
        """
        # Add a custom pipeline mode for magnitude if not already set
        if context.pipeline_mode == "mri_dadt":
            logger.info("Setting pipeline mode to 'mri_dadt_magnitude' for dA/dt magnitude processing")
            context.pipeline_mode = "mri_dadt_magnitude"
        
        # Initialize base class
        super().__init__(
            context=context,
            debug_hook=debug_hook,
            resource_monitor=resource_monitor,
            use_cache=use_cache,
            cache_dir=cache_dir
        )
        
        # Log initialization
        logger.info(f"Initialized dA/dt Magnitude Pipeline with mode '{context.pipeline_mode}'")
    
    def _compute_magnitude(self, vector_field: torch.Tensor) -> torch.Tensor:
        """Compute magnitude of a vector field.
        
        Args:
            vector_field: Vector field with shape [..., 3]
            
        Returns:
            Magnitude with shape [...]
        """
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "compute_magnitude_start",
                {'field_shape': vector_field.shape}
            )
        
        # Check if the tensor has the right shape
        if vector_field.shape[-1] != 3:
            raise ValueError(f"Expected vector field with last dimension 3, got shape {vector_field.shape}")
        
        # Compute magnitude
        magnitude = torch.sqrt(torch.sum(vector_field**2, dim=-1))
        
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "compute_magnitude_complete",
                {
                    'field_shape': vector_field.shape,
                    'magnitude_shape': magnitude.shape,
                    'min': float(torch.min(magnitude).item()),
                    'max': float(torch.max(magnitude).item()),
                    'mean': float(torch.mean(magnitude).item())
                }
            )
        
        return magnitude
    
    def process_sample(self, sample: TMSSample) -> TMSProcessedData:
        """Process a single TMS sample, converting dA/dt vector to magnitude.
        
        Args:
            sample: TMS sample to process
            
        Returns:
            TMSProcessedData with dA/dt magnitude
        """
        # First check if this sample is already in cache
        cached_data = self._check_cache(sample)
        if cached_data is not None:
            return cached_data
        
        # Get device from context
        device = torch.device(self.context.config.get("device", "cpu"))
        
        # Load MRI data
        mri_grid = self.context.config.get("mri_tensor")
        if mri_grid is None:
            raise ValueError("MRI tensor not found in context.config['mri_tensor']")
        mri_grid = mri_grid.to(device)
        
        # Load E-field magnitude and dA/dt vector
        try:
            # Load the SCALAR E-field magnitude (target) and convert to float32
            target_efield_magnitude = torch.load(sample.efield_data, map_location=device).float()
            
            # Load the dA/dt VECTOR data and convert to float32
            dadt_vector = torch.load(sample.dadt_data, map_location=device).float()
        except Exception as e:
            logger.error(f"Error loading tensors for sample {sample.sample_id}: {e}")
            raise
        
        # Ensure target E-field has shape [1, D, H, W]
        if target_efield_magnitude.ndim == 3:
            target_efield_magnitude = target_efield_magnitude.unsqueeze(0)
        
        # Compute dA/dt magnitude
        if dadt_vector.ndim == 4 and dadt_vector.shape[-1] == 3:
            # Vector format [D, H, W, 3]
            dadt_magnitude = self._compute_magnitude(dadt_vector)
        else:
            raise ValueError(f"Unexpected dA/dt shape: {dadt_vector.shape}, expected [D, H, W, 3]")
        
        # Normalize dA/dt magnitude
        dadt_max = torch.max(dadt_magnitude)
        if dadt_max > 0:
            dadt_magnitude = dadt_magnitude / dadt_max
        
        # Scale dA/dt with configured factor
        dadt_magnitude = dadt_magnitude * self.config.dadt_scaling_factor
        
        # Ensure MRI is float32
        if mri_grid.dtype != torch.float32:
            mri_grid = mri_grid.float()
            
        # Normalize MRI
        mri_normalized = self._normalize_mri(mri_grid)
        
        # Ensure MRI has 3D shape [D, H, W]
        if mri_normalized.ndim == 4 and mri_normalized.shape[0] == 1:
            mri_normalized = mri_normalized.squeeze(0)
        
        # Stack MRI and dA/dt magnitude
        # First make both tensors [1, D, H, W]
        mri_channel = mri_normalized.unsqueeze(0)
        dadt_magnitude_channel = dadt_magnitude.unsqueeze(0)
        
        # Stack along channel dimension and ensure float32
        stacked_data = torch.cat([mri_channel, dadt_magnitude_channel], dim=0).float()
        
        # Create processed data - FIXED: Removed 'sample_id' parameter and ensured float32
        processed_data = TMSProcessedData(
            subject_id=sample.subject_id,
            input_features=stacked_data,  # [2, D, H, W] - MRI + dA/dt magnitude
            target_efield=target_efield_magnitude,  # [1, D, H, W] - E-field magnitude
            mask=None,  # TODO: Handle masks if needed
            metadata={
                'coil_position_idx': sample.coil_position_idx,
                'stacking_mode': 'mri_dadt_magnitude',
                'input_shape': stacked_data.shape,
                'target_shape': target_efield_magnitude.shape
            }
        )
        
        # Save to cache
        self._save_to_cache(sample, processed_data)
        
        return processed_data