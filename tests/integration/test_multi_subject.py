#!/usr/bin/env python3
# train_multi_subject.py
"""
Training script for TMS E-field magnitude prediction using multiple subjects.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import torch
import numpy as np

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the new multi-subject data module
from tms_efield_prediction.data.pipeline.multi_subject_data import prepare_multi_subject_data
from tms_efield_prediction.experiments.experiment_runner import MagnitudeExperimentRunner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_multi_subject')

def main():
    """Run training using multiple subjects with specific assignments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train TMS E-field magnitude prediction model with multiple subjects')
    parser.add_argument('--data-dir', type=str, default='/home/freyhe/MA_Henry/data', help='Data directory path')
    parser.add_argument('--output-dir', type=str, default='multi_subject_output', help='Output directory for results')
    
    # Specific subject assignments
    parser.add_argument('--train-subjects', type=str, default='1,3,4,6,7,8', help='Comma-separated list of training subject IDs')
    parser.add_argument('--val-subjects', type=str, default='9', help='Comma-separated list of validation subject IDs')
    parser.add_argument('--test-subjects', type=str, default='9', help='Comma-separated list of test subject IDs')
    
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    
    # Augmentation parameters - separate for shift and rotation
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    
    # Shift augmentation parameters
    parser.add_argument('--shift-enabled', action='store_true', help='Enable spatial shift augmentation')
    parser.add_argument('--shift-max', type=int, default=10, help='Max shift distance (1-5)')
    parser.add_argument('--shift-prob', type=float, default=0.8, help='Probability of applying shift')
    
    # Rotation augmentation parameters
    parser.add_argument('--rotation-enabled', action='store_true', help='Enable rotation augmentation')
    parser.add_argument('--rotation-max-degrees', type=float, default=45.0, help='Max rotation angle in degrees')
    parser.add_argument('--rotation-prob', type=float, default=0.5, help='Probability of applying rotation')
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
    
    # Ensure 3-digit subject IDs (pad with zeros if needed)
    train_subjects = [s.zfill(3) for s in train_subjects]
    val_subjects = [s.zfill(3) for s in val_subjects]
    test_subjects = [s.zfill(3) for s in test_subjects]
    
    logger.info(f"Using {len(train_subjects)} subjects for training: {train_subjects}")
    logger.info(f"Using {len(val_subjects)} subjects for validation: {val_subjects}")
    logger.info(f"Using {len(test_subjects)} subjects for testing: {test_subjects}")
    
    # Parse rotation center
    rotation_center = tuple(int(x) for x in args.rotation_center.split(',')) if args.rotation_center else (23, 23, 23)
    
    # Configure augmentation with separate shift and rotation settings
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
    
    # Prepare multi-subject data
    output_shape = (25, 25, 25)  # Cubic grid with dimensions 25x25x25
    
    train_loader, val_loader, test_loader = prepare_multi_subject_data(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        data_root_path=args.data_dir,
        output_shape=output_shape,
        normalization_method="standard",
        batch_size=args.batch_size,
        augmentation_config=augmentation_config,
        pin_memory=torch.cuda.is_available()  # Use pin_memory for faster GPU transfer
    )
    
    # Configure and initialize the experiment runner
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"multi_subject_experiment_{timestamp}")
    
    runner = MagnitudeExperimentRunner(
        experiment_dir=experiment_dir,
        architecture_name="simple_unet_magnitude",
        description=f"TMS E-field Magnitude Prediction - Multi-Subject Training",
        resource_monitor_max_gb=8,
        device=device
    )
    
    # Configure model
    model_config = {
        "model_type": "simple_unet_magnitude",
        "input_shape": [4, 25, 25, 25],     # [C_stacked, D, H, W]
        "output_shape": [1, 25, 25, 25],    # [1=Magnitude, D, H, W]
        "input_channels": 4,                # Channels from stacked input
        "output_channels": 1,               # Predicting magnitude only
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
        "mask_threshold": 1e-8,
        "device": str(device),
        "mixed_precision": True,            # Start simple
        "validation_frequency": 1,           # Validate every epoch
        "early_stopping": True,
        "early_stopping_patience": 5,        # Stop if no improvement for 5 epochs
        "loss_type": "magnitude_mse",
    }
    runner.configure_trainer(trainer_config)
    
    # Run the experiment
    logger.info("Starting the multi-subject experiment...")
    results = runner.train_and_evaluate(train_loader, val_loader, test_loader)
    
    # Report results
    if results["test_metrics"]:
        logger.info("=== Final Test Metrics ===")
        for metric_name, value in results["test_metrics"].items():
            logger.info(f"  {metric_name}: {value:.6f}")
    
    logger.info(f"Experiment completed. Results saved to {experiment_dir}")
    logger.info(f"Training time: {results['training_time']:.2f} seconds")
    
    # Log subject assignments and augmentation settings
    with open(os.path.join(experiment_dir, "experiment_settings.txt"), "w") as f:
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

if __name__ == "__main__":
    main()