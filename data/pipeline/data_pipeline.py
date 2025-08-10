# tms_efield_prediction/data/pipeline/data_pipeline.py

import torch
import logging
from typing import Dict, Tuple, Optional, List, Union
from torch.utils.data import DataLoader, TensorDataset

from ..transformations.augmentation import augmentation_collate_fn
from ..data_splitter import DataSplitter
from ..pipeline.loader import TMSDataLoader
from ..transformations.stack_pipeline import EnhancedStackingPipeline
from ...utils.state.context import TMSPipelineContext

logger = logging.getLogger(__name__)

def prepare_tms_data(
    pipeline_context: TMSPipelineContext,
    train_pct: float = 0.6,
    val_pct: float = 0.2,
    test_pct: float = 0.2,
    batch_size: int = 8,
    random_seed: int = 42,
    augmentation_config: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepares TMS data for model training by loading, processing, and splitting data into DataLoaders.
    
    Args:
        pipeline_context: Configured TMSPipelineContext
        train_pct: Percentage of data for training
        val_pct: Percentage of data for validation
        test_pct: Percentage of data for testing
        batch_size: Batch size for DataLoaders
        random_seed: Random seed for reproducibility
        augmentation_config: Optional configuration for data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # 1. Load raw data
    logger.info("Loading raw data...")
    data_loader = TMSDataLoader(context=pipeline_context)
    raw_data = data_loader.load_raw_data()
    if raw_data is None:
        raise ValueError("Raw Data is None. Please check DataLoader configuration.")
    
    # Update context with MRI tensor
    pipeline_context.config["mri_tensor"] = raw_data.mri_tensor
    
    # 2. Create sample list and process data
    logger.info("Processing data samples...")
    samples = data_loader.create_sample_list(raw_data)
    stacking_pipeline = EnhancedStackingPipeline(context=pipeline_context)
    processed_data_list = stacking_pipeline.process_batch(samples)
    
    # 3. Extract features and targets
    features_list = [processed_data.input_features for processed_data in processed_data_list]
    targets_list = [processed_data.target_efield for processed_data in processed_data_list]
    features_stacked = torch.stack(features_list)
    targets_vectors = torch.stack(targets_list)
    
    logger.info(f"Features shape: {features_stacked.shape}")
    logger.info(f"Targets shape: {targets_vectors.shape}")
    
    # 4. Split data using DataSplitter
    logger.info(f"Splitting data: train={train_pct}, val={val_pct}, test={test_pct}")
    splitter = DataSplitter(
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        random_seed=random_seed
    )
    train_dataset, val_dataset, test_dataset = splitter.split_data(features_stacked, targets_vectors)
    
    # 5. Create DataLoaders with optional augmentation
    logger.info("Creating DataLoaders...")
    # Define collate function if augmentation is enabled
    if augmentation_config and augmentation_config.get('enabled', False):
        collate_fn = lambda batch: augmentation_collate_fn(batch, augmentation_config)
        logger.info(f"Data augmentation enabled with config: {augmentation_config}")
    else:
        collate_fn = None
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        prefetch_factor=2
    )
    
    # No augmentation for validation and test sets
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader