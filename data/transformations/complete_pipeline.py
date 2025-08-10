# tms_efield_prediction/data/transformations/complete_pipeline.py
"""
Complete TMS E-field prediction pipeline.

This module integrates mesh-to-grid transformation and channel stacking
into a complete pipeline for TMS E-field prediction.
"""

import numpy as np
import time
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field

from utils.debug.hooks import PipelineDebugHook
from utils.resource.monitor import ResourceMonitor
from utils.pipeline.implementation_unit import ImplementationUnit, UnitResult, UnitPipeline
from utils.state.context import TMSPipelineContext, PipelineState
from utils.debug.context import PipelineDebugState
from ..pipeline.tms_data_types import TMSRawData, TMSProcessedData, TMSSample, TMSSplit
from ..formats.simnibs_io import MeshData

# Import the new VoxelMapper instead of MeshToGridTransformer
from .voxel_mapping import VoxelMapper, create_transform_matrix
from .stack_pipeline import ChannelStackingPipeline, StackingConfig


class CompletePreprocessingPipeline:
    """Complete preprocessing pipeline for TMS E-field prediction."""
    
    def __init__(
        self, 
        context: TMSPipelineContext,
        debug_hook: Optional[PipelineDebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """Initialize the complete preprocessing pipeline.
        
        Args:
            context: TMS-specific pipeline context
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
        """
        self.context = context
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Create VoxelMapper (lazy initialization, will be created when needed)
        self.voxel_mapper = None
        
        # Create stacking configuration
        stacking_config = StackingConfig(
            normalization_method=context.normalization_method,
            dadt_scaling_factor=context.dadt_scaling_factor,
            output_shape=context.output_shape
        )
        
        self.channel_stacking = ChannelStackingPipeline(
            context=context,
            debug_hook=debug_hook,
            resource_monitor=resource_monitor,
            config=stacking_config
        )
        
        # Register with resource monitor if provided
        if self.resource_monitor:
            self.resource_monitor.register_component(
                "complete_preprocessing_pipeline",
                self._reduce_memory,
                priority=20  # High priority for this component
            )
        
        # State tracking
        self.pipeline_state = PipelineState()
        
        # Memory tracking
        self._memory_usage = 0
    
    def process_raw_data(self, raw_data: TMSRawData) -> List[TMSProcessedData]:
        """
        Process raw TMS data to create processed data ready for model input.
        
        Args:
            raw_data: Raw TMS data
            
        Returns:
            List of processed TMS data objects
        """
        start_time = time.time()
        
        # Create samples from raw data
        samples = self._create_samples(raw_data)
        
        # Process each sample
        processed_data = []
        for sample in samples:
            try:
                # Process the sample
                processed = self._process_sample(sample)
                processed_data.append(processed)
                
                # Update progress and log
                if self.debug_hook and self.debug_hook.should_sample():
                    self.debug_hook.record_event(
                        "sample_processed",
                        {
                            'sample_id': sample.sample_id,
                            'progress': f"{len(processed_data)}/{len(samples)}"
                        }
                    )
            except Exception as e:
                # Log error
                if self.debug_hook:
                    self.debug_hook.record_error(
                        e,
                        {
                            'sample_id': sample.sample_id,
                            'phase': 'complete_pipeline'
                        }
                    )
                # Re-raise
                raise
        
        # Update state
        self.pipeline_state.processed_data = {
            'subject_id': raw_data.subject_id,
            'n_samples': len(processed_data),
            'execution_time': time.time() - start_time
        }
        
        # Log completion
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "raw_data_processing_complete",
                {
                    'subject_id': raw_data.subject_id,
                    'n_samples': len(processed_data),
                    'execution_time': time.time() - start_time
                }
            )
        
        return processed_data
    
    def process_and_save(
        self, 
        raw_data: TMSRawData, 
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Process raw data and save results to disk.
        
        Args:
            raw_data: Raw TMS data
            output_dir: Directory to save processed data
            
        Returns:
            Dictionary with processing metadata
        """
        start_time = time.time()
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Process raw data
        processed_data = self.process_raw_data(raw_data)
        
        # Save processed data
        metadata = {}
        
        # Save each sample
        for i, sample in enumerate(processed_data):
            sample_path = os.path.join(
                output_dir, 
                f"{raw_data.subject_id}_sample_{sample.metadata['coil_position_idx']}.npz"
            )
            
            # Create save data
            save_data = {
                'input_features': sample.input_features,
                'subject_id': sample.subject_id,
                'metadata': sample.metadata
            }
            
            # Add target if available
            if sample.target_efield is not None:
                save_data['target_efield'] = sample.target_efield
            
            # Add mask if available
            if sample.mask is not None:
                save_data['mask'] = sample.mask
            
            # Save to disk
            np.savez_compressed(sample_path, **save_data)
            
            # Add to metadata
            metadata[f"sample_{i}"] = {
                'path': sample_path,
                'shape': sample.input_features.shape
            }
        
        # Save summary metadata
        metadata_path = os.path.join(output_dir, f"{raw_data.subject_id}_metadata.npz")
        np.savez(
            metadata_path,
            subject_id=raw_data.subject_id,
            n_samples=len(processed_data),
            processing_time=time.time() - start_time,
            sample_list=[p.metadata['sample_id'] for p in processed_data],
            normalization_method=self.context.normalization_method,
            dadt_scaling_factor=self.context.dadt_scaling_factor,
            output_shape=self.context.output_shape
        )
        
        # Add paths to metadata
        metadata['metadata_path'] = metadata_path
        metadata['output_dir'] = output_dir
        metadata['processing_time'] = time.time() - start_time
        
        # Log completion
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "processing_and_saving_complete",
                {
                    'subject_id': raw_data.subject_id,
                    'n_samples': len(processed_data),
                    'output_dir': output_dir,
                    'execution_time': metadata['processing_time']
                }
            )
        
        return metadata
    
    def create_splits(
        self, 
        processed_data: List[TMSProcessedData],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> TMSSplit:
        """
        Create train/validation/test splits from processed data.
        
        Args:
            processed_data: List of processed TMS data
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            TMSSplit object with train/val/test splits
        """
        # Check ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        
        # Convert to samples for consistent interface
        samples = []
        for data in processed_data:
            sample = TMSSample(
                sample_id=data.metadata['sample_id'],
                subject_id=data.subject_id,
                coil_position_idx=data.metadata['coil_position_idx'],
                mri_data=None,  # Already processed
                dadt_data=None,  # Already processed
                efield_data=data.target_efield,
                coil_position=None,  # Not needed for processed data
                metadata=data.metadata
            )
            samples.append(sample)
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Shuffle indices
        indices = np.arange(len(samples))
        np.random.shuffle(indices)
        
        # Calculate split sizes
        n_train = int(train_ratio * len(samples))
        n_val = int(val_ratio * len(samples))
        
        # Create splits
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Create split data
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        test_samples = [samples[i] for i in test_idx]
        
        # Create TMSSplit object
        split = TMSSplit(
            training=train_samples,
            validation=val_samples,
            testing=test_samples
        )
        
        # Log split creation
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "splits_created",
                {
                    'n_samples': len(samples),
                    'n_train': len(train_samples),
                    'n_val': len(val_samples),
                    'n_test': len(test_samples),
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio
                }
            )
        
        return split
    
    def _create_samples(self, raw_data: TMSRawData) -> List[TMSSample]:
        """
        Create individual samples from raw data.
        
        Args:
            raw_data: Raw TMS data
            
        Returns:
            List of TMS samples
        """
        samples = []
        
        # Extract MRI data from mesh
        if raw_data.mri_mesh is None:
            raise ValueError("MRI mesh is required for processing")
        
        # Extract coil positions
        if raw_data.coil_positions is None:
            raise ValueError("Coil positions are required for processing")
        
        n_positions = len(raw_data.coil_positions)
        
        # Create a sample for each coil position
        for i in range(n_positions):
            # Create sample ID
            sample_id = f"{raw_data.subject_id}_position_{i}"
            
            # Extract dA/dt data for this position
            if raw_data.dadt_data is not None:
                if len(raw_data.dadt_data) != n_positions:
                    raise ValueError(f"dA/dt data length ({len(raw_data.dadt_data)}) does not match coil positions ({n_positions})")
                dadt_data = raw_data.dadt_data[i]
            else:
                dadt_data = None
            
            # Extract E-field data for this position
            if raw_data.efield_data is not None:
                if len(raw_data.efield_data) != n_positions:
                    raise ValueError(f"E-field data length ({len(raw_data.efield_data)}) does not match coil positions ({n_positions})")
                efield_data = raw_data.efield_data[i]
            else:
                efield_data = None
            
            # Create sample
            sample = TMSSample(
                sample_id=sample_id,
                subject_id=raw_data.subject_id,
                coil_position_idx=i,
                mri_data=raw_data.mri_mesh,  # Pass mesh for now, will be converted later
                dadt_data=dadt_data,
                efield_data=efield_data,
                coil_position=raw_data.coil_positions[i],
                metadata={
                    'roi_center': raw_data.roi_center
                }
            )
            
            # Add additional metadata
            if raw_data.metadata:
                sample.metadata.update(raw_data.metadata)
            
            samples.append(sample)
        
        return samples
    
    def _process_sample(self, sample: TMSSample) -> TMSProcessedData:
        """
        Process a single TMS sample through the complete pipeline.
        
        Args:
            sample: TMS sample
            
        Returns:
            Processed TMS data
        """
        # Extract mesh data
        mesh_data = sample.mri_data
        
        # Check if it's already a grid (processed)
        if isinstance(mesh_data, np.ndarray):
            # Already in grid format, just normalize and stack
            mri_grid = mesh_data
            mask = None
        else:
            # Convert mesh to grid using VoxelMapper
            mesh_nodes = mesh_data.nodes
            node_centers = mesh_data.elements['tetra'].mean(axis=1) if isinstance(mesh_data, MeshData) else None
            
            if node_centers is None:
                raise ValueError("Cannot determine node centers from mesh data")
            
            # Extract scalar data from mesh if available
            scalar_data = None
            if isinstance(mesh_data, MeshData) and mesh_data.node_data:
                # Use first scalar field by default
                scalar_data = next(iter(mesh_data.node_data.values()))
            
            # Using VoxelMapper for transformation
            n_bins = self.context.output_shape[0]  # Assuming cubic grid
            
            # Create or get VoxelMapper
            if not hasattr(self, 'voxel_mapper') or self.voxel_mapper is None:
                self.voxel_mapper = VoxelMapper(
                    context=self.context,
                    bin_size=n_bins,
                    debug_hook=self.debug_hook,
                    resource_monitor=self.resource_monitor
                )
                
                # Preprocess using node centers data
                if self.debug_hook and self.debug_hook.should_sample():
                    self.debug_hook.record_event(
                        "voxel_mapper_initialization",
                        {'bin_size': n_bins}
                    )
                
            # Transform scalar data to grid
            mri_grid, mask, _ = self.voxel_mapper.transform(scalar_data, node_centers)
        
        # Create a modified sample with grid data
        grid_sample = TMSSample(
            sample_id=sample.sample_id,
            subject_id=sample.subject_id,
            coil_position_idx=sample.coil_position_idx,
            mri_data=mri_grid,
            dadt_data=sample.dadt_data,
            efield_data=sample.efield_data,
            coil_position=sample.coil_position,
            metadata=sample.metadata
        )
        
        # Process through stacking pipeline
        processed_data = self.channel_stacking.process_sample(grid_sample)
        
        # Add mask if available
        if mask is not None:
            processed_data.mask = mask
        
        return processed_data
    
    def _reduce_memory(self, target_reduction: float) -> None:
        """
        Callback for memory reduction requests.
        
        Args:
            target_reduction: Fraction of memory to reduce (0.0-1.0)
        """
        # Log memory reduction request
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "memory_reduction_requested",
                {
                    'target_reduction': target_reduction,
                    'component': "complete_preprocessing_pipeline"
                }
            )
        
        # Propagate request to sub-components
        # First reduce memory in VoxelMapper if available
        if hasattr(self, 'voxel_mapper') and self.voxel_mapper is not None:
            self.voxel_mapper._reduce_memory(target_reduction)
        
        # Then reduce memory in channel stacking pipeline
        if hasattr(self.channel_stacking, '_reduce_memory'):
            self.channel_stacking._reduce_memory(target_reduction)
        
        # Log completion
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "memory_reduction_complete",
                {
                    'target_reduction': target_reduction,
                    'component': "complete_preprocessing_pipeline"
                }
            )