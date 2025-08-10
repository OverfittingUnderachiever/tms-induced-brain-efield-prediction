#!/usr/bin/env python3
"""
Training script for TMS E-field magnitude prediction using dA/dt magnitude instead of vector.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom components
from tms_efield_prediction.data.pipeline.multi_subject_data import MultiSubjectDataManager
from tms_efield_prediction.experiments.experiment_runner import MagnitudeExperimentRunner
from tms_efield_prediction.data.transformations.stack_pipeline import DADTMagnitudeStackingPipeline
from tms_efield_prediction.utils.state.context import TMSPipelineContext
from tms_efield_prediction.data.pipeline.loader import TMSDataLoader
from tms_efield_prediction.constants import EFIELD_MASK_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_dadt_magnitude')
def visualize_batch_slices(batch_data, batch_targets, slice_positions, output_dir, prefix, epoch=None, subject_info=None):
    """
    Visualize slices of a batch of data along the y-axis with enhanced information.
    
    Args:
        batch_data: Input tensor of shape [B, C, D, H, W]
        batch_targets: Target tensor of shape [B, 1, D, H, W]
        slice_positions: List of y positions to visualize
        output_dir: Directory to save visualizations
        prefix: Prefix for the saved files (e.g., 'train' or 'val')
        epoch: Current epoch number (None for validation)
        subject_info: Information about subjects (e.g., "Subjects: 4,6,7,8")
    """
    # Create output directory if it doesn't exist
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(exist_ok=True, parents=True)
    
    # Only visualize the first sample in the batch
    sample_data = batch_data[0].detach().cpu().numpy()  # Shape: [C, D, H, W]
    sample_target = batch_targets[0].detach().cpu().numpy()  # Shape: [1, D, H, W]
    
    # Get number of input channels
    num_channels = sample_data.shape[0]
    
    # Create a figure for each slice position
    for y_pos in slice_positions:
        fig, axes = plt.subplots(1, num_channels + 1, figsize=(5*(num_channels+1), 5))
        
        # Add a detailed title with subject information if available
        subject_str = f"{subject_info}" if subject_info else ""
        epoch_info = f"Epoch: {epoch}" if epoch is not None else "Pre-training"
        data_type = "Training data" if prefix.lower() == 'train' else "Validation data"
        
        # Combine all information in the title
        title_parts = [part for part in [data_type, subject_str, epoch_info] if part]
        title = " | ".join(title_parts)
        fig.suptitle(f"Y-Slice at position {y_pos}\n{title}", fontsize=14)
        
        # Plot input channels
        for c in range(num_channels):
            channel_name = "MRI" if c == 0 else f"dA/dt Magnitude"
            ax = axes[c]
            slice_data = sample_data[c, :, y_pos, :]
            
            # Calculate data statistics for title
            data_min = slice_data.min()
            data_max = slice_data.max()
            data_mean = slice_data.mean()
            
            # Normalize the data for better visualization
            norm = Normalize(vmin=data_min, vmax=data_max)
            im = ax.imshow(slice_data, cmap='viridis', norm=norm)
            ax.set_title(f"Input: {channel_name}\nRange: [{data_min:.2f}, {data_max:.2f}], Mean: {data_mean:.2f}")
            fig.colorbar(im, ax=ax)
            ax.set_axis_off()
            
            # Add slice position indicators
            ax.axhline(y=slice_data.shape[0]//2, color='r', linestyle='--', alpha=0.3)
            ax.axvline(x=slice_data.shape[1]//2, color='r', linestyle='--', alpha=0.3)
        
        # Plot target E-field magnitude
        ax = axes[-1]
        slice_target = sample_target[0, :, y_pos, :]
        
        # Calculate target statistics
        target_min = slice_target.min()
        target_max = slice_target.max()
        target_mean = slice_target.mean()
        
        norm = Normalize(vmin=target_min, vmax=target_max)
        im = ax.imshow(slice_target, cmap='plasma', norm=norm)
        ax.set_title(f"Target: E-field Magnitude\nRange: [{target_min:.2f}, {target_max:.2f}], Mean: {target_mean:.2f}")
        fig.colorbar(im, ax=ax)
        ax.set_axis_off()
        
        # Add slice position indicators
        ax.axhline(y=slice_target.shape[0]//2, color='r', linestyle='--', alpha=0.3)
        ax.axvline(x=slice_target.shape[1]//2, color='r', linestyle='--', alpha=0.3)
        
        # Add grid for better visual reference
        for a in axes:
            a.grid(False)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)  # Make room for the title
        
        # Save the figure with more descriptive filename
        subject_str = f"_{prefix}" if not subject_info else f"_{prefix}"
        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
        save_path = vis_dir / f"{subject_str}_y{y_pos}{epoch_str}.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
    logger.info(f"Saved {len(slice_positions)} visualization slices to {vis_dir}")



class MagnitudeDataManager(MultiSubjectDataManager):
    """Extended data manager that uses dA/dt magnitude."""
    
    def load_subject_data(self, subject_id: str, force_reload: bool = True) -> tuple:
        """Load and preprocess subject data using dA/dt magnitude.
        
        Args:
            subject_id: Subject ID to load
            force_reload: Whether to force reload even if cached
            
        Returns:
            Tuple of (features, targets) tensors
        """
        # Format subject_id with leading zeros to 3 digits
        subject_id = subject_id.zfill(3)
        

        
        logger.info(f"Loading magnitude data for subject {subject_id}")
        
        # Always use CPU for data loading to avoid CUDA memory issues
        cpu_device = torch.device("cpu")
        
        # Set up pipeline context
        tms_config = {"mri_tensor": None, "device": cpu_device}
        pipeline_context = TMSPipelineContext(
            dependencies={},
            config=tms_config,
            pipeline_mode="mri_dadt_magnitude",  # Use magnitude mode
            experiment_phase="training",
            debug_mode=True,
            subject_id=subject_id,
            data_root_path=self.data_root_path,
            output_shape=self.output_shape,
            normalization_method=self.normalization_method,
            device=cpu_device,
            
        )
        
        try:
            # Load raw data
            data_loader = TMSDataLoader(context=pipeline_context)
            raw_data = data_loader.load_raw_data()
            if raw_data is None:
                logger.error(f"Raw data is None for subject {subject_id}")
                return None, None
            
            # Update context with MRI tensor
            pipeline_context.config["mri_tensor"] = raw_data.mri_tensor
            
            # Create sample list
            samples = data_loader.create_sample_list(raw_data)
            if not samples:
                logger.warning(f"No samples found for subject {subject_id}")
                return None, None
            
            # Use our specialized magnitude stacking pipeline
            stacking_pipeline = DADTMagnitudeStackingPipeline(context=pipeline_context)
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
            
            logger.info(f"Loaded {len(features_list)} magnitude samples for subject {subject_id}")
            logger.info(f"Features shape: {features_stacked.shape}, Targets shape: {targets_vectors.shape}")
            
            # Store in cache
            self.subject_cache[subject_id] = (features_stacked, targets_vectors)
            
            return features_stacked, targets_vectors
            
        except Exception as e:
            logger.error(f"Error loading magnitude data for subject {subject_id}: {e}")
            return None, None


def modify_trainer_config_for_dtype_fix(trainer_config):
    """
    Modify trainer configuration to handle data type issues.
    
    Args:
        trainer_config: Original trainer configuration
        
    Returns:
        Modified trainer configuration
    """
    # Disable mixed precision as it's causing data type conflicts
    trainer_config["mixed_precision"] = False
    
    # Add explicit dtype handling
    trainer_config["dtype"] = "float32"
    
    return trainer_config


def prepare_magnitude_data(
    train_subjects: list,
    val_subjects: list,
    test_subjects: list,
    data_root_path: str,
    output_shape: tuple = (25, 25, 25),
    normalization_method: str = "standard",
    batch_size: int = 8,
    augmentation_config: dict = None,
    num_workers: int = 0,
    pin_memory: bool = True
):
    """Prepare data loaders using dA/dt magnitude with correct data types."""
    # Create our specialized data manager
    data_manager = MagnitudeDataManager(
        data_root_path=data_root_path,
        output_shape=output_shape,
        normalization_method=normalization_method
    )
    
    # Format subject IDs to ensure 3 digits
    train_subjects = [s.zfill(3) for s in train_subjects]
    val_subjects = [s.zfill(3) for s in val_subjects]
    test_subjects = [s.zfill(3) for s in test_subjects]
    
    logger.info(f"Preparing data loaders with dA/dt magnitude:")
    logger.info(f"  Training: {len(train_subjects)} subjects: {', '.join(train_subjects)}")
    logger.info(f"  Validation: {len(val_subjects)} subjects: {', '.join(val_subjects)}")
    logger.info(f"  Testing: {len(test_subjects)} subjects: {', '.join(test_subjects)}")
    
    # Get data from manager
    train_loaders, val_loaders, test_loaders = data_manager.prepare_data_loaders(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        batch_size=batch_size,
        augmentation_config=augmentation_config,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Apply data type conversion using custom collate function
    def float32_collate_fn(batch):
        features, targets = zip(*batch)
        features = torch.stack(features).float()  # Convert to float32
        targets = torch.stack(targets).float()    # Convert to float32
        return features, targets
    
    # Create new dataloaders with explicit dtype conversion
    train_dataset = train_loaders.dataset
    val_dataset = val_loaders.dataset if val_loaders else None
    test_dataset = test_loaders.dataset if test_loaders else None
    
    # Create modified loaders with float32 collate function
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=float32_collate_fn
    )
    
    val_loader = None
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=float32_collate_fn
        )
    
    test_loader = None
    if test_dataset:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=float32_collate_fn
        )
    
    logger.info("Created data loaders with explicit float32 type conversion")
    return train_loader, val_loader, test_loader
def visualize_individual_subjects(subjects, data_root_path, output_shape, slice_positions, experiment_dir, 
                          prefix, epoch, batch_size=1, normalization_method="standard"):
    """
    Create visualizations for individual subjects.
    
    Args:
        subjects: List of subject IDs to visualize
        data_root_path: Path to the data directory
        output_shape: Shape of the output grid
        slice_positions: List of y positions to visualize
        experiment_dir: Directory to save visualizations
        prefix: Prefix for filenames (e.g., 'train', 'val', 'test')
        epoch: Current epoch number or 0 for pre-training
        batch_size: Batch size for the data loader
        normalization_method: Normalization method for the data
    """
    logger.info(f"Creating individual {prefix} subject visualizations (epoch {epoch})...")

    # Limit to 3 subjects maximum to avoid too many visualizations
    if len(subjects) > 3:
        selected_subjects = subjects[:3]
        logger.info(f"Limiting visualization to first 3 subjects: {', '.join(selected_subjects)}")
    else:
        selected_subjects = subjects
    
    for subject_id in selected_subjects:
        # Create a data manager for a single subject
        data_manager = MagnitudeDataManager(
            data_root_path=data_root_path,
            output_shape=output_shape,
            normalization_method=normalization_method
        )
        
        # Format subject ID
        subject_id_padded = subject_id.zfill(3)
        
        logger.info(f"Creating visualization for {prefix} subject: {subject_id}")
        
        # Load the subject data directly from the data manager
        features, targets = data_manager.load_subject_data(subject_id, force_reload=False)
        
        if features is None or targets is None:
            logger.warning(f"No data available for subject {subject_id}")
            continue
            
        # Take just the first few samples (up to batch_size)
        if features.shape[0] > batch_size:
            features = features[:batch_size]
            targets = targets[:batch_size]
        
        # Create visualization
        visualize_batch_slices(
            features,  # Already a tensor with the right shape
            targets,   # Already a tensor with the right shape
            slice_positions,
            experiment_dir,
            prefix=f"{prefix}_subject_{subject_id}",
            epoch=epoch,
            subject_info=f"Individual Subject: {subject_id}"
        )


def main():
    """Run training using dA/dt magnitude instead of vector."""
    # Parse command line arguments - similar to test_multi_subject.py
    parser = argparse.ArgumentParser(description='Train TMS E-field prediction model using dA/dt magnitude')
    parser.add_argument('--data-dir', type=str, default='/home/freyhe/MA_Henry/data', help='Data directory path')
    parser.add_argument('--output-dir', type=str, default='dadt_magnitude_output', help='Output directory for results')
    
    # Specific subject assignments
    parser.add_argument('--train-subjects', type=str, default='4,6,7,8', help='Comma-separated list of training subject IDs')
    parser.add_argument('--val-subjects', type=str, default='3', help='Comma-separated list of validation subject IDs')
    parser.add_argument('--test-subjects', type=str, default='9', help='Comma-separated list of test subject IDs')
    
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    
    # Augmentation parameters
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--shift-enabled', action='store_true', help='Enable spatial shift augmentation')
    parser.add_argument('--shift-max', type=int, default=10, help='Max shift distance (1-5)')
    parser.add_argument('--shift-prob', type=float, default=0.0, help='Probability of applying shift')
    parser.add_argument('--rotation-enabled', action='store_true', help='Enable rotation augmentation')
    parser.add_argument('--rotation-max-degrees', type=float, default=45.0, help='Max rotation angle in degrees')
    parser.add_argument('--rotation-prob', type=float, default=0.0, help='Probability of applying rotation')
    parser.add_argument('--rotation-center', type=str, default='23,23,23', help='Center of rotation (x,y,z)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Parse subject lists
    train_subjects = [s.strip() for s in args.train_subjects.split(',')]
    val_subjects = [s.strip() for s in args.val_subjects.split(',')]
    test_subjects = [s.strip() for s in args.test_subjects.split(',')]
    
    # Parse rotation center
    rotation_center = tuple(int(x) for x in args.rotation_center.split(',')) if args.rotation_center else (23, 23, 23)
    
    # Configure augmentation
    augmentation_config = None
    if args.augment:
        augmentation_config = {'enabled': True}
        
        # Add shift augmentation if enabled
        if args.shift_enabled:
            augmentation_config['spatial_shift'] = {
                'enabled': True,
                'max_shift': args.shift_max,
                'probability': args.shift_prob
            }
            logger.info(f"Spatial shift augmentation enabled: max_shift={args.shift_max}, probability={args.shift_prob}")
        
        # Add rotation augmentation if enabled
        if args.rotation_enabled:
            # Convert degrees to radians
            max_angle_rad = np.radians(args.rotation_max_degrees)
            augmentation_config['rotation'] = {
                'enabled': True,
                'max_angle_y': max_angle_rad,
                'probability': args.rotation_prob,
                'center': rotation_center,
                'y_only': True  # Only rotate around y-axis
            }
            logger.info(f"Rotation augmentation enabled: max_angle={args.rotation_max_degrees}°, probability={args.rotation_prob}, center={rotation_center}")
    
    # Prepare multi-subject data with dA/dt magnitude
    output_shape = (25, 25, 25)  # Cubic grid with dimensions 25x25x25
    
    # Use our custom function for magnitude data
    train_loader, val_loader, test_loader = prepare_magnitude_data(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        data_root_path=args.data_dir,
        output_shape=output_shape,
        normalization_method="standard",
        batch_size=args.batch_size,
        augmentation_config=augmentation_config,
        pin_memory=torch.cuda.is_available()
    )
    
    # Configure and initialize the experiment runner
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"dadt_magnitude_experiment_{timestamp}")
    
    runner = MagnitudeExperimentRunner(
        experiment_dir=experiment_dir,
        architecture_name="simple_unet_magnitude",
        description=f"TMS E-field Magnitude Prediction with dA/dt Magnitude",
        resource_monitor_max_gb=8,
        device=device
    )
    
    # Configure model - IMPORTANT: Changed input_channels from 4 to 2
    model_config = {
        "model_type": "simple_unet_magnitude",
        "input_shape": [2, 25, 25, 25],     # [C_stacked=2, D, H, W] - MRI + dA/dt magnitude
        "output_shape": [1, 25, 25, 25],    # [1=Magnitude, D, H, W]
        "input_channels": 2,                # MRI (1) + dA/dt magnitude (1)
        "output_channels": 1,               # Predicting E-field magnitude only
        "feature_maps": 16,                 # Base filters for UNet
        "levels": 3,                        # UNet levels
        "norm_type": "batch",
        "activation": "relu",
        "dropout_rate": 0.2,
        "use_residual": True,
        "use_attention": False,
    }
    runner.configure_model(model_config)
    
    # Configure trainer
    trainer_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "optimizer_type": "adam",
        "scheduler_type": "reduce_on_plateau",
        "scheduler_patience": 3,
        "mask_threshold": EFIELD_MASK_THRESHOLD  ,
        "device": str(device),
        "mixed_precision": False,
        "validation_frequency": 1,
        "early_stopping": True,
        "early_stopping_patience": 5,
        "loss_type": "magnitude_mse",
    }
    runner.configure_trainer(trainer_config)
    
    # Set up visualization parameters
    slice_positions = [10, 15, 20]  # Y-axis positions to visualize
    
    # Create initial visualizations before training
    logger.info("Creating initial visualizations before training...")
    
    # Create visualizations for individual subjects (pre-training)
    visualize_individual_subjects(
        subjects=train_subjects,
        data_root_path=args.data_dir,
        output_shape=output_shape,
        slice_positions=slice_positions,
        experiment_dir=experiment_dir,
        prefix="train",
        epoch=0,
        batch_size=1
    )
    
    visualize_individual_subjects(
        subjects=val_subjects,
        data_root_path=args.data_dir,
        output_shape=output_shape,
        slice_positions=slice_positions,
        experiment_dir=experiment_dir,
        prefix="val",
        epoch=0,
        batch_size=1
    )
    
    visualize_individual_subjects(
        subjects=test_subjects,
        data_root_path=args.data_dir,
        output_shape=output_shape,
        slice_positions=slice_positions,
        experiment_dir=experiment_dir,
        prefix="test",
        epoch=0,
        batch_size=1
    )
    
    # Run the experiment
    logger.info("Starting the dA/dt magnitude experiment...")
    results = runner.train_and_evaluate(train_loader, val_loader, test_loader)
    
    # Create final visualizations after training
    logger.info("Creating final visualizations after training...")
    
    # Create visualizations for individual subjects (post-training)
    visualize_individual_subjects(
        subjects=train_subjects,
        data_root_path=args.data_dir,
        output_shape=output_shape,
        slice_positions=slice_positions,
        experiment_dir=experiment_dir,
        prefix="train",
        epoch=args.epochs,
        batch_size=1
    )
    
    visualize_individual_subjects(
        subjects=val_subjects,
        data_root_path=args.data_dir,
        output_shape=output_shape,
        slice_positions=slice_positions,
        experiment_dir=experiment_dir,
        prefix="val",
        epoch=args.epochs,
        batch_size=1
    )
    
    visualize_individual_subjects(
        subjects=test_subjects,
        data_root_path=args.data_dir,
        output_shape=output_shape,
        slice_positions=slice_positions,
        experiment_dir=experiment_dir,
        prefix="test",
        epoch=args.epochs,
        batch_size=1
    )
    
    # Report results
    if results["test_metrics"]:
        logger.info("=== Final Test Metrics ===")
        for metric_name, value in results["test_metrics"].items():
            logger.info(f"  {metric_name}: {value:.6f}")
    
    logger.info(f"Experiment completed. Results saved to {experiment_dir}")
    logger.info(f"Training time: {results['training_time']:.2f} seconds")
    
    # Log subject assignments and settings
    with open(os.path.join(experiment_dir, "experiment_settings.txt"), "w") as f:
        f.write("======= dA/dt Magnitude Experiment Settings =======\n\n")
        f.write("Input configuration: MRI + dA/dt Magnitude (2 channels)\n\n")
        f.write(f"Training subjects: {','.join(train_subjects)}\n")
        f.write(f"Validation subjects: {','.join(val_subjects)}\n")
        f.write(f"Testing subjects: {','.join(test_subjects)}\n\n")
        
        f.write("Augmentation settings:\n")
        if augmentation_config:
            if 'spatial_shift' in augmentation_config:
                shift = augmentation_config['spatial_shift']
                f.write(f"  Spatial shift: enabled={shift['enabled']}, max_shift={shift['max_shift']}, probability={shift['probability']}\n")
            else:
                f.write("  Spatial shift: disabled\n")
                
            if 'rotation' in augmentation_config:
                rot = augmentation_config['rotation']
                f.write(f"  Rotation: enabled={rot['enabled']}, max_angle_deg={np.degrees(rot['max_angle_y']):.1f}, probability={rot['probability']}, center={rot['center']}\n")
            else:
                f.write("  Rotation: disabled\n")
        else:
            f.write("  All augmentations disabled\n")
        
        # Also add visualization information
        f.write("\nVisualization settings:\n")
        f.write(f"  Slice positions: {slice_positions}\n")
        f.write(f"  Images saved in: {os.path.join(experiment_dir, 'visualizations')}\n")
        f.write(f"  Individual subject visualizations created for each dataset type\n")


if __name__ == "__main__":
    main()