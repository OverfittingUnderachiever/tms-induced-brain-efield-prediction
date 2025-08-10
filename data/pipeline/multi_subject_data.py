# tms_efield_prediction/data/pipeline/multi_subject_data.py

import torch
import logging
import os
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from ..transformations.augmentation_old import augmentation_collate_fn
from ..pipeline.loader import TMSDataLoader
from ..transformations.stack_pipeline import EnhancedStackingPipeline
from ...utils.state.context import TMSPipelineContext

logger = logging.getLogger(__name__)

class MultiSubjectDataManager:
    """
    Manages data loading and preprocessing across multiple subjects.
    Ensures memory-efficient processing by keeping data on CPU.
    """
    
    def __init__(
        self,
        data_root_path: str,
        output_shape: Tuple[int, int, int] = (25, 25, 25),
        normalization_method: str = "standard",
        mri_type: str = "dti"  # Add mri_type parameter with default
    ):
        """
        Initialize the multi-subject data manager.
        
        Args:
            data_root_path: Root directory for data
            output_shape: Shape of output volumes
            normalization_method: Method for normalization
            mri_type: Type of MRI data to use ("dti" or "conductivity")
        """
        self.data_root_path = data_root_path
        self.output_shape = output_shape
        self.normalization_method = normalization_method
        self.mri_type = mri_type  # Store mri_type
        self.subject_cache = {}  # Optional cache to avoid reloading
        
        # Create a context object that can be used by child objects
        self.context = TMSPipelineContext(
            dependencies={},
            config={"mri_type": mri_type},  # Include mri_type in config
            pipeline_mode="mri_dadt",
            experiment_phase="initialization",
            debug_mode=False,
            subject_id=None,  # Will be set per subject
            data_root_path=data_root_path,
            output_shape=output_shape,
            normalization_method=normalization_method,
            device="cpu"
        )

    def calculate_target_percentiles(
        self,
        subjects: List[str],
        percentiles: List[float] = [10.0, 90.0],
        mask_threshold: float = 0.01,
        force_reload: bool = False
    ) -> Dict[float, float]:
        """
        Calculate specified percentiles of target values across multiple subjects.
        
        Args:
            subjects: List of subject IDs
            percentiles: List of percentile values to calculate (e.g., [10.0, 90.0])
            mask_threshold: Threshold for masking low-intensity regions
            force_reload: Whether to force reload data even if cached
            
        Returns:
            Dict mapping percentile values to their corresponding thresholds
        """
        if not subjects:
            logger.warning("No subjects provided for percentile calculation")
            return {p: 0.0 for p in percentiles}
        
        # Format subject IDs
        subjects = [str(s).zfill(3) for s in subjects]
        
        logger.info(f"Calculating {percentiles} percentiles across {len(subjects)} subjects")
        
        # Store percentile values from each subject
        subject_percentiles = {p: [] for p in percentiles}
        
        # Process each subject
        for subject_id in subjects:
            logger.info(f"Loading data for subject {subject_id} for percentile calculation")
            _, targets = self.load_subject_data(subject_id, force_reload=force_reload)
            
            if targets is None:
                logger.warning(f"Skipping subject {subject_id}: No targets available")
                continue
            
            # Get valid values (above mask threshold)
            # targets could be a tensor of shape [N, 1, D, H, W] where N is number of samples
            # We want to flatten it to calculate percentiles on all values
            targets_flat = targets.view(-1)
            valid_values = targets_flat[targets_flat > mask_threshold].float()
            
            if valid_values.numel() == 0:
                logger.warning(f"No valid target values found for subject {subject_id}")
                continue
            
            # Calculate percentiles for this subject
            for p in percentiles:
                try:
                    p_value = torch.quantile(valid_values, p / 100.0)
                    subject_percentiles[p].append(p_value.item())
                except Exception as e:
                    logger.error(f"Error calculating {p}th percentile for subject {subject_id}: {e}")
        
        # Average percentiles across subjects
        result_percentiles = {}
        for p in percentiles:
            if subject_percentiles[p]:
                result_percentiles[p] = sum(subject_percentiles[p]) / len(subject_percentiles[p])
            else:
                # Default values if no subjects had valid percentiles
                if p <= 50.0:
                    result_percentiles[p] = 0.01  # Default low percentile
                else:
                    result_percentiles[p] = 1.0   # Default high percentile
        
        logger.info(f"Calculated percentiles: {result_percentiles}")
        return result_percentiles


    def load_subject_data(self, subject_id: str, force_reload: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess data for a single subject, keeping data on CPU.
        
        Args:
            subject_id: Subject ID to load
            force_reload: Whether to force reload even if data is cached
            
        Returns:
            Tuple of (features, targets) tensors, both on CPU
        """
        # Format subject_id with leading zeros to 3 digits
        subject_id = str(subject_id).zfill(3)
        
        logger.info(f"===== LOADING DATA FOR SUBJECT: {subject_id} =====")
        
        # Check cache first if not forcing reload
        if not force_reload and subject_id in self.subject_cache:
            logger.info(f"Using cached data for subject {subject_id}")
            return self.subject_cache[subject_id]
        
        # Try different directory formats
        subject_dir = None
        
        # First try with 'sub-' prefix (BIDS format)
        bids_dir = os.path.join(self.data_root_path, f"sub-{subject_id}")
        if os.path.exists(bids_dir):
            subject_dir = bids_dir
            logger.info(f"Found BIDS subject directory: {subject_dir}")
        else:
            # Then try without prefix
            plain_dir = os.path.join(self.data_root_path, subject_id)
            if os.path.exists(plain_dir):
                subject_dir = plain_dir
                logger.info(f"Found plain subject directory: {subject_dir}")
            else:
                logger.error(f"Could not find subject directory for {subject_id}")
                # List all directories to help debug
                try:
                    all_dirs = sorted([d for d in os.listdir(self.data_root_path) 
                                    if os.path.isdir(os.path.join(self.data_root_path, d))])
                    logger.error(f"Available directories: {all_dirs}")
                except Exception as e:
                    logger.error(f"Error listing directories: {e}")
                return None, None
        
        # Check if experiment directory exists
        experiment_dir = os.path.join(subject_dir, "experiment")
        if not os.path.exists(experiment_dir):
            logger.error(f"Experiment directory not found: {experiment_dir}")
            try:
                subj_contents = os.listdir(subject_dir)
                logger.error(f"Contents of {subject_dir}: {subj_contents}")
            except Exception as e:
                logger.error(f"Error listing contents: {e}")
            return None, None
        
        # Check for MRI directory
        mri_dir = os.path.join(experiment_dir, "MRI_arrays", "torch")
        if not os.path.exists(mri_dir):
            logger.error(f"MRI directory not found: {mri_dir}")
            return None, None
        
        # Get the actual directory name to extract subject_id format for context
        dir_basename = os.path.basename(subject_dir)
        context_subject_id = subject_id
        if dir_basename.startswith('sub-'):
            # Extract the actual ID used in the directory
            context_subject_id = dir_basename[4:]
        
        logger.info(f"Using context_subject_id: {context_subject_id}")
        
        # Always use CPU for data loading to avoid CUDA memory issues
        cpu_device = torch.device("cpu")
        
        # Set up pipeline context
        tms_config = {
            "mri_tensor": None, 
            "device": cpu_device,
            "mri_type": self.context.config.get("mri_type", "dti")  # Pass mri_type from context
        }
        pipeline_context = TMSPipelineContext(
            dependencies={},
            config=tms_config,
            pipeline_mode="mri_dadt",
            experiment_phase="training",
            debug_mode=True,
            subject_id=context_subject_id,  # Pass the extracted subject ID
            data_root_path=self.data_root_path,
            output_shape=self.output_shape,
            normalization_method=self.normalization_method,
            device=cpu_device  # Force CPU
        )
        
        try:
            # Load raw data with extra debug information
            data_loader = TMSDataLoader(context=pipeline_context)
            logger.info(f"Created data loader for subject {subject_id}, checking paths...")
            
            # Debug the paths
            subject_path = os.path.join(self.data_root_path, f"sub-{subject_id}")
            experiment_path = os.path.join(subject_path, "experiment")
            logger.info(f"Full subject path: {subject_path}")
            logger.info(f"Experiment path: {experiment_path}")
            
            # Try to load the data
            raw_data = data_loader.load_raw_data()
            if raw_data is None:
                logger.error(f"Raw data is None for subject {subject_id}")
                return None, None
            
            # Update context with MRI tensor
            pipeline_context.config["mri_tensor"] = raw_data.mri_tensor
            
            # Create sample list and process data
            samples = data_loader.create_sample_list(raw_data)
            
            # Skip if empty
            if not samples:
                logger.warning(f"No samples found for subject {subject_id}")
                return None, None
            
            stacking_pipeline = EnhancedStackingPipeline(context=pipeline_context)
            processed_data_list = stacking_pipeline.process_batch(samples)
            
            # Extract features and targets
            features_list = [processed_data.input_features for processed_data in processed_data_list]
            targets_list = [processed_data.target_efield for processed_data in processed_data_list]
            
            # Stack into tensors
            features_stacked = torch.stack(features_list)
            targets_vectors = torch.stack(targets_list)
            
            # Ensure tensors are on CPU
            features_stacked = features_stacked.cpu()
            targets_vectors = targets_vectors.cpu()
            
            logger.info(f"Successfully loaded {len(features_list)} samples for subject {subject_id}")
            logger.info(f"Features shape: {features_stacked.shape}, Targets shape: {targets_vectors.shape}")
            
            # Store in cache
            self.subject_cache[subject_id] = (features_stacked, targets_vectors)
            
            return features_stacked, targets_vectors
            
        except Exception as e:
            logger.error(f"Error loading data for subject {subject_id}: {e}", exc_info=True)
            return None, None
    
    def prepare_data_loaders(
        self,
        train_subjects: List[str],
        val_subjects: List[str],
        test_subjects: List[str],
        batch_size: int = 8,
        augmentation_config: Optional[Dict] = None,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Prepare data loaders for train, validation, and test sets.
        
        Args:
            train_subjects: List of subject IDs for training
            val_subjects: List of subject IDs for validation
            test_subjects: List of subject IDs for testing
            batch_size: Batch size for data loaders
            augmentation_config: Configuration for data augmentation
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory (useful for faster GPU transfer)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.debug(f"--- prepare_data_loaders invoked ---")
        logger.debug(f"    Received train_subjects: {train_subjects}")
        logger.debug(f"    Received val_subjects: {val_subjects}")
        logger.debug(f"    Received test_subjects: {test_subjects}")
        # Format subject IDs
        train_subjects = [s.zfill(3) for s in train_subjects]
        val_subjects = [s.zfill(3) for s in val_subjects]
        test_subjects = [s.zfill(3) for s in test_subjects]
        
        logger.info(f"Preparing data loaders for:")
        logger.info(f"  Training: {len(train_subjects)} subjects: {', '.join(train_subjects)}")
        logger.info(f"  Validation: {len(val_subjects)} subjects: {', '.join(val_subjects)}")
        logger.info(f"  Testing: {len(test_subjects)} subjects: {', '.join(test_subjects)}")
        
        # Load training data
        train_features_list = []
        train_targets_list = []
        for subject_id in train_subjects:
            features, targets = self.load_subject_data(subject_id)
            if features is not None and targets is not None:
                train_features_list.append(features)
                train_targets_list.append(targets)
        
        # Concatenate training data
        if train_features_list and train_targets_list:
            train_features = torch.cat(train_features_list, dim=0)
            train_targets = torch.cat(train_targets_list, dim=0)
            logger.info(f"Combined training data: {train_features.shape[0]} samples from {len(train_subjects)} subjects")
        else:
            raise ValueError("No training data could be loaded from any subject")
        
        # Load validation data
        val_features_list = []
        val_targets_list = []
        for subject_id in val_subjects:
            features, targets = self.load_subject_data(subject_id)
            if features is not None and targets is not None:
                val_features_list.append(features)
                val_targets_list.append(targets)
        
        # Concatenate validation data
        val_loader = None
        if val_features_list and val_targets_list:
            val_features = torch.cat(val_features_list, dim=0)
            val_targets = torch.cat(val_targets_list, dim=0)
            logger.info(f"Combined validation data: {val_features.shape[0]} samples from {len(val_subjects)} subjects")
            
            val_dataset = TensorDataset(val_features, val_targets)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        else:
            logger.warning("No validation data could be loaded. Validation will be skipped.")
        
        # Load test data
        test_features_list = []
        test_targets_list = []
        for subject_id in test_subjects:
            features, targets = self.load_subject_data(subject_id)
            if features is not None and targets is not None:
                test_features_list.append(features)
                test_targets_list.append(targets)
        
        # Concatenate test data
        test_loader = None
        if test_features_list and test_targets_list:
            test_features = torch.cat(test_features_list, dim=0)
            test_targets = torch.cat(test_targets_list, dim=0)
            logger.info(f"Combined test data: {test_features.shape[0]} samples from {len(test_subjects)} subjects")
            
            test_dataset = TensorDataset(test_features, test_targets)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        else:
            logger.warning("No test data could be loaded. Testing will be skipped.")
        
        # Create training dataset and loader
        train_dataset = TensorDataset(train_features, train_targets)
        
        # Set up augmentation collate function if needed
        collate_fn = None
        if augmentation_config and augmentation_config.get('enabled', False):
            collate_fn = lambda batch: augmentation_collate_fn(batch, augmentation_config)
            logger.info(f"Data augmentation enabled with config: {augmentation_config}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader, test_loader
    
    def clear_cache(self):
        """Clear the subject data cache to free memory."""
        self.subject_cache.clear()
        logger.info("Subject data cache cleared")


# Convenience function for simpler usage
def prepare_multi_subject_data(
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    data_root_path: str,
    output_shape: Tuple[int, int, int] = (25, 25, 25),
    normalization_method: str = "standard",
    batch_size: int = 8,
    augmentation_config: Optional[Dict] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    mri_type: str = "dti"  # Add mri_type parameter with default
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Convenience function to prepare data loaders for multiple subjects.
    
    Args:
        train_subjects: List of subject IDs for training
        val_subjects: List of subject IDs for validation
        test_subjects: List of subject IDs for testing
        data_root_path: Root directory for data
        output_shape: Shape of output volumes
        normalization_method: Method for normalization
        batch_size: Batch size for data loaders
        augmentation_config: Configuration for data augmentation
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory
        mri_type: Type of MRI data to use ("dti" or "conductivity")
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_manager = MultiSubjectDataManager(
        data_root_path=data_root_path,
        output_shape=output_shape,
        normalization_method=normalization_method,
        mri_type=mri_type  # Pass mri_type to data manager
    )
    
    return data_manager.prepare_data_loaders(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        batch_size=batch_size,
        augmentation_config=augmentation_config,
        num_workers=num_workers,
        pin_memory=pin_memory
    )