# tms_efield_prediction/data/pipeline/field_processor.py
"""
Field processor for TMS E-field and dA/dt data.

This module provides functionality for processing and stacking fields,
extracted from the generate_training_data.py script.
"""

import os
import numpy as np
import time
import pickle
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import multiprocessing as mp

from utils.debug.hooks import PipelineDebugHook
from utils.resource.monitor import ResourceMonitor
from utils.state.context import TMSPipelineContext


@dataclass
class FieldProcessingConfig:
    """Configuration for field processing."""
    bin_size: int = 25
    n_processes: Optional[int] = None
    save_torch: bool = True
    save_numpy: bool = False
    save_pickle: bool = False
    clean_intermediate: bool = False
    memory_limit: int = 8000  # MB


class FieldProcessor:
    """Processor for TMS-induced E-fields and dA/dt fields."""
    
    def __init__(
        self,
        context: TMSPipelineContext,
        config: Optional[FieldProcessingConfig] = None,
        debug_hook: Optional[PipelineDebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """Initialize the field processor.
        
        Args:
            context: TMS-specific pipeline context
            config: Configuration for field processing
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
        """
        self.context = context
        self.config = config or FieldProcessingConfig()
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Memory tracking
        self._memory_usage = 0
        self._intermediate_data = {}
        
        # Register with resource monitor if provided
        if self.resource_monitor:
            self.resource_monitor.register_component(
                "field_processor",
                self._reduce_memory,
                priority=15  # Medium-high priority
            )
    
    def process_single_field(
        self,
        field_data: np.ndarray,
        is_vector: bool = True,
        field_type: str = 'efield'
    ) -> np.ndarray:
        """Process a single field array.
        
        Args:
            field_data: Field data array
            is_vector: Whether the field is vector (True) or scalar (False)
            field_type: Type of field ('efield' or 'dadt')
            
        Returns:
            Processed field data
        """
        start_time = time.time()
        
        # Log start if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "process_single_field_started",
                {
                    'field_type': field_type,
                    'is_vector': is_vector,
                    'field_shape': field_data.shape
                }
            )
        
        # Apply normalization based on field type
        if field_type == 'efield':
            # Normalize E-field (typically using magnitude normalization)
            if is_vector:
                # Calculate magnitude for vector field
                magnitudes = np.sqrt(np.sum(field_data**2, axis=-1, keepdims=True))
                max_mag = np.max(magnitudes)
                if max_mag > 0:
                    normalized_field = field_data / max_mag
                else:
                    normalized_field = field_data
            else:
                # Normalize scalar field
                max_val = np.max(np.abs(field_data))
                if max_val > 0:
                    normalized_field = field_data / max_val
                else:
                    normalized_field = field_data
        elif field_type == 'dadt':
            # Apply scaling factor from context
            scaling_factor = self.context.dadt_scaling_factor
            normalized_field = field_data * scaling_factor
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        # Update memory tracking
        self._update_memory_usage(f"{field_type}_normalized", normalized_field)
        
        # Log completion if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "process_single_field_completed",
                {
                    'field_type': field_type,
                    'execution_time': time.time() - start_time,
                    'output_shape': normalized_field.shape
                }
            )
        
        return normalized_field
    
    def stack_fields(
        self,
        efield_data: np.ndarray,
        dadt_data: np.ndarray
    ) -> np.ndarray:
        """Stack E-field and dA/dt field data.
        
        Args:
            efield_data: E-field data
            dadt_data: dA/dt field data
            
        Returns:
            Stacked field data
        """
        start_time = time.time()
        
        # Log start if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "stack_fields_started",
                {
                    'efield_shape': efield_data.shape,
                    'dadt_shape': dadt_data.shape
                }
            )
        
        # Determine shapes to handle different combinations
        efield_shape = efield_data.shape
        dadt_shape = dadt_data.shape
        
        # Create stacked data based on shapes
        if len(efield_shape) == 3 and len(dadt_shape) == 4:
            # E-field is scalar [bin, bin, bin], dA/dt is vector [bin, bin, bin, 3]
            # Convert E-field to [bin, bin, bin, 1] for consistency
            efield_expanded = np.expand_dims(efield_data, axis=-1)
            # Stack into [bin, bin, bin, 4]
            stacked_data = np.concatenate([efield_expanded, dadt_data], axis=-1)
            
            # Log channel info
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "stack_fields_info",
                    {
                        'shape': 'scalar_efield_vector_dadt',
                        'format': '[bin, bin, bin, 4]',
                        'channel_info': ['efield_magnitude', 'dadt_x', 'dadt_y', 'dadt_z']
                    }
                )
                
        elif len(efield_shape) == 4 and len(dadt_shape) == 4:
            # Both are vector fields [bin, bin, bin, 3]
            # Stack into [bin, bin, bin, 6]
            stacked_data = np.concatenate([efield_data, dadt_data], axis=-1)
            
            # Log channel info
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "stack_fields_info",
                    {
                        'shape': 'vector_efield_vector_dadt',
                        'format': '[bin, bin, bin, 6]',
                        'channel_info': ['efield_x', 'efield_y', 'efield_z', 'dadt_x', 'dadt_y', 'dadt_z']
                    }
                )
                
        else:
            raise ValueError(
                f"Incompatible shapes for stacking: E-field {efield_shape}, dA/dt {dadt_shape}"
            )
        
        # Update memory tracking
        self._update_memory_usage("stacked_data", stacked_data)
        
        # Log completion if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "stack_fields_completed",
                {
                    'execution_time': time.time() - start_time,
                    'output_shape': stacked_data.shape
                }
            )
        
        return stacked_data
    
    def process_and_stack(
        self,
        efield_data: np.ndarray,
        dadt_data: np.ndarray,
        is_efield_vector: bool = True,
        is_dadt_vector: bool = True
    ) -> np.ndarray:
        """Process and stack E-field and dA/dt data.
        
        Args:
            efield_data: E-field data
            dadt_data: dA/dt field data
            is_efield_vector: Whether E-field is vector (True) or scalar (False)
            is_dadt_vector: Whether dA/dt is vector (True) or scalar (False)
            
        Returns:
            Processed and stacked field data
        """
        # Process individual fields
        processed_efield = self.process_single_field(
            efield_data, is_vector=is_efield_vector, field_type='efield'
        )
        
        processed_dadt = self.process_single_field(
            dadt_data, is_vector=is_dadt_vector, field_type='dadt'
        )
        
        # Stack processed fields
        stacked_data = self.stack_fields(processed_efield, processed_dadt)
        
        return stacked_data
    
    def batch_process(
        self,
        efield_data_list: List[np.ndarray],
        dadt_data_list: List[np.ndarray],
        is_efield_vector: bool = True,
        is_dadt_vector: bool = True,
        parallel: bool = True
    ) -> List[np.ndarray]:
        """Process and stack multiple field pairs in batch.
        
        Args:
            efield_data_list: List of E-field data arrays
            dadt_data_list: List of dA/dt field data arrays
            is_efield_vector: Whether E-field is vector (True) or scalar (False)
            is_dadt_vector: Whether dA/dt is vector (True) or scalar (False)
            parallel: Whether to process in parallel
            
        Returns:
            List of processed and stacked field data
        """
        start_time = time.time()
        
        # Validate inputs
        if len(efield_data_list) != len(dadt_data_list):
            raise ValueError(
                f"Number of E-fields ({len(efield_data_list)}) does not match "
                f"number of dA/dt fields ({len(dadt_data_list)})"
            )
        
        num_fields = len(efield_data_list)
        
        # Log start if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "batch_process_started",
                {
                    'num_fields': num_fields,
                    'parallel': parallel
                }
            )
        
        if parallel and num_fields > 1:
            # Process in parallel
            n_processes = self.config.n_processes or mp.cpu_count()
            n_processes = min(n_processes, num_fields)
            
            # Create worker arguments
            worker_args = [
                (efield_data_list[i], dadt_data_list[i], is_efield_vector, is_dadt_vector)
                for i in range(num_fields)
            ]
            
            # Use multiprocessing
            with mp.Pool(processes=n_processes) as pool:
                results = pool.starmap(
                    self._process_single_pair_worker,
                    worker_args
                )
        else:
            # Process sequentially
            results = []
            for i in range(num_fields):
                stacked_data = self.process_and_stack(
                    efield_data_list[i],
                    dadt_data_list[i],
                    is_efield_vector,
                    is_dadt_vector
                )
                results.append(stacked_data)
        
        # Log completion if debug enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "batch_process_completed",
                {
                    'num_fields': num_fields,
                    'execution_time': time.time() - start_time
                }
            )
        
        return results
    
    def _process_single_pair_worker(
        self,
        efield_data: np.ndarray,
        dadt_data: np.ndarray,
        is_efield_vector: bool,
        is_dadt_vector: bool
    ) -> np.ndarray:
        """Worker function for parallel processing of field pairs.
        
        Args:
            efield_data: E-field data array
            dadt_data: dA/dt field data array
            is_efield_vector: Whether E-field is vector
            is_dadt_vector: Whether dA/dt is vector
            
        Returns:
            Processed and stacked field data
        """
        # Process E-field
        processed_efield = self._process_field_worker(efield_data, is_efield_vector, 'efield')
        
        # Process dA/dt
        processed_dadt = self._process_field_worker(dadt_data, is_dadt_vector, 'dadt')
        
        # Stack fields
        if is_efield_vector:
            efield_dim = 3
        else:
            efield_dim = 1
            processed_efield = np.expand_dims(processed_efield, axis=-1)
        
        if is_dadt_vector:
            dadt_dim = 3
        else:
            dadt_dim = 1
            processed_dadt = np.expand_dims(processed_dadt, axis=-1)
        
        # Concatenate along last dimension
        stacked_data = np.concatenate([processed_efield, processed_dadt], axis=-1)
        
        return stacked_data
    
    def _process_field_worker(
        self,
        field_data: np.ndarray,
        is_vector: bool,
        field_type: str
    ) -> np.ndarray:
        """Worker function for parallel field processing.
        
        Args:
            field_data: Field data array
            is_vector: Whether the field is vector
            field_type: Type of field ('efield' or 'dadt')
            
        Returns:
            Processed field data
        """
        if field_type == 'efield':
            # Normalize E-field
            if is_vector:
                # Calculate magnitude for vector field
                magnitudes = np.sqrt(np.sum(field_data**2, axis=-1, keepdims=True))
                max_mag = np.max(magnitudes)
                if max_mag > 0:
                    normalized_field = field_data / max_mag
                else:
                    normalized_field = field_data
            else:
                # Normalize scalar field
                max_val = np.max(np.abs(field_data))
                if max_val > 0:
                    normalized_field = field_data / max_val
                else:
                    normalized_field = field_data
        elif field_type == 'dadt':
            # Apply scaling factor (hardcoded for worker)
            normalized_field = field_data * 1.0e-6  # Default scaling
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        return normalized_field
    
    def _update_memory_usage(self, key: str, data: Any) -> None:
        """Update memory usage tracking.
        
        Args:
            key: Identifier for the data object
            data: Data to track
        """
        # Calculate memory usage (approximate)
        if isinstance(data, np.ndarray):
            mem_bytes = data.nbytes
        elif isinstance(data, list) and all(isinstance(x, np.ndarray) for x in data):
            mem_bytes = sum(x.nbytes for x in data)
        else:
            # For other types
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
                "field_processor", 
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
        
        # Clear non-essential intermediate data
        keys_to_clear = []
        for key, info in self._intermediate_data.items():
            # Skip the most recently processed data
            if key == "stacked_data":
                continue
                
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
                "field_processor", 
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


