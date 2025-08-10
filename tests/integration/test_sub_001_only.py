# test_single_subject.py
"""
Integration test for training and evaluation workflow with a single subject.
Subject: 001
Split: 70% training, 20% validation, 10% testing
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Reset root logger first (Python 3.7 workaround for missing 'force' parameter)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Now configure the root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_single_subject')

# Reduce verbosity for noisy loggers
logging.getLogger('tms_efield_prediction.data.transformations.stack_pipeline').setLevel(logging.WARNING)
logging.getLogger('ResourceMonitor').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Import the Model ---
from tms_efield_prediction.models.architectures.simple_unet_magnitude import SimpleUNetMagnitudeModel
from tms_efield_prediction.models.training.trainer import ModelTrainer, TrainerConfig
from tms_efield_prediction.models.evaluation.metrics import calculate_magnitude_metrics
from tms_efield_prediction.models.evaluation.visualization import (
    create_metrics_summary_plot,
    generate_standard_visualizations
)
from tms_efield_prediction.experiments.tracking import ExperimentTracker
from tms_efield_prediction.utils.state.context import ModelContext, TMSPipelineContext
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.data.pipeline.loader import TMSDataLoader
from tms_efield_prediction.data.transformations.stack_pipeline import EnhancedStackingPipeline


def load_subject_data(subject_id, data_root_path, output_shape=(25, 25, 25), normalization_method="standard", use_cache=True):
    """
    Load and preprocess data for a specific subject.
    
    Args:
        subject_id (str): Subject ID (e.g., "001", "003", etc.)
        data_root_path (str): Path to the data directory
        output_shape (tuple): Shape for output tensors
        normalization_method (str): Method for normalization
        use_cache (bool): Whether to use caching
        
    Returns:
        tuple: (features_tensor, targets_tensor) for the subject
    """
    # Format subject ID with leading zeros if needed
    subject_id = subject_id.zfill(3)
    
    logger.info(f"Loading data for subject {subject_id}...")
    
    # Create pipeline context for this subject
    # Force CPU for data loading to avoid CUDA memory issues during preprocessing
    cpu_device = torch.device("cpu")
    tms_config = {"mri_tensor": None, "device": cpu_device}
    pipeline_context = TMSPipelineContext(
        dependencies={},
        config=tms_config,
        pipeline_mode="mri_dadt",
        experiment_phase="training",
        debug_mode=True,
        subject_id=subject_id,
        data_root_path=data_root_path,
        output_shape=output_shape,
        normalization_method=normalization_method,
        device=cpu_device  # Use CPU for data loading and preprocessing
    )
    
    # Load raw data
    data_loader = TMSDataLoader(context=pipeline_context)
    raw_data = data_loader.load_raw_data()
    if raw_data is None:
        logger.error(f"Failed to load raw data for subject {subject_id}")
        return None, None
    
    # Update context with MRI tensor
    tms_config["mri_tensor"] = raw_data.mri_tensor
    pipeline_context.config = tms_config
    
    # Create sample list and process data
    samples = data_loader.create_sample_list(raw_data)
    
    # Use cache_dir specific to this test
    cache_dir = os.path.join(data_root_path, "cache", f"sub-{subject_id}")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create stacking pipeline with caching
    stacking_pipeline = EnhancedStackingPipeline(
        context=pipeline_context,
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    processed_data_list = stacking_pipeline.process_batch(samples)
    
    # Extract features and targets
    features_list = [processed_data.input_features for processed_data in processed_data_list]
    targets_list = [processed_data.target_efield for processed_data in processed_data_list]
    
    # Stack into tensors
    features_tensor = torch.stack(features_list)
    targets_tensor = torch.stack(targets_list)
    
    logger.info(f"Subject {subject_id} data loaded. Features shape: {features_tensor.shape}, Targets shape: {targets_tensor.shape}")
    
    return features_tensor, targets_tensor


def prepare_single_subject_datasets(subject_id, data_root_path, output_shape=(25, 25, 25), use_cache=True, 
                                    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Prepare datasets for training, validation, and testing from a single subject with specified split ratios.
    
    Args:
        subject_id (str): Subject ID (e.g., "001")
        data_root_path (str): Path to the data directory
        output_shape (tuple): Shape for output tensors
        use_cache (bool): Whether to use caching
        train_ratio (float): Ratio of data to use for training
        val_ratio (float): Ratio of data to use for validation
        test_ratio (float): Ratio of data to use for testing
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load data for the single subject
    features, targets = load_subject_data(
        subject_id, 
        data_root_path, 
        output_shape, 
        use_cache=use_cache
    )
    
    if features is None or targets is None:
        raise ValueError(f"Failed to load data for subject {subject_id}")
    
    # Transform features from [N, D, H, W, C] (channels-last) to [N, C, D, H, W] (channels-first) if needed
    if features.shape[-1] < features.shape[1]:  # Basic check if channels seem last
        logger.info(f"Subject {subject_id}: Permuting features from Channels Last -> Channels First")
        features = features.permute(0, 4, 1, 2, 3)
    
    # Log the total number of samples
    total_samples = features.shape[0]
    logger.info(f"Subject {subject_id} has {total_samples} total samples")
    
    # Calculate the number of samples for each split
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size  # Ensure all samples are used
    
    # Create a TensorDataset
    dataset = TensorDataset(features, targets)
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Log the split sizes
    logger.info(f"Dataset split for subject {subject_id}:")
    logger.info(f"  Training: {len(train_dataset)} samples ({train_ratio*100:.1f}%)")
    logger.info(f"  Validation: {len(val_dataset)} samples ({val_ratio*100:.1f}%)")
    logger.info(f"  Testing: {len(test_dataset)} samples ({test_ratio*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset


def train_and_evaluate(train_loader, val_loader, test_loader, output_dir='test_output_single_subject', split_info=None):
    """Train a UNet magnitude model and evaluate its performance.

    Args:
        train_loader: Training data loader. Yields (features_channels_first, target_vectors).
        val_loader: Validation data loader. Yields (features_channels_first, target_vectors).
        test_loader: Test data loader. Yields (features_channels_first, target_vectors).
        output_dir: Directory for outputs.
        split_info: Dictionary with split information for logging

    Returns:
        Tuple of (trained model, test metrics dictionary).
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log memory info
    if torch.cuda.is_available():
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

    # --- Configuration for the SimpleUNetMagnitudeModel ---
    unet_config = {
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
        # --- Include base config keys if needed by BaseModel ---
        "learning_rate": 0.001, # Example, will be overridden by TrainerConfig
        "optimizer": "adam",    # Example
        "loss_function": "mse", # Example
    }

    # --- Instantiate the Model ---
    model = SimpleUNetMagnitudeModel(config=unet_config)
    model.to(device)
    logger.info("SimpleUNetMagnitudeModel instantiated.")

    # Create model context (basic example)
    model_context = ModelContext(
        dependencies={},
        config={"architecture": "simple_unet_magnitude"}
    )

    # Create resource monitor
    resource_monitor = ResourceMonitor(
        max_memory_gb=8, # Adjust as needed
        check_interval=10.0
    )
    resource_monitor.start_monitoring()

    # Create experiment tracker
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracker = ExperimentTracker(
        experiment_dir=output_dir,
        architecture_name="simple_unet_magnitude",
        create_subdirs=True
    )
    tracker.set_description("Single-Subject Magnitude Prediction Test - Subject: 001 (70/20/10 split)")
    tracker.log_hyperparameters(unet_config) # Log model config

    # --- Create simple text file with split information ---
    if split_info:
        with open(os.path.join(tracker.experiment_dir, "split_info.txt"), "w") as f:
            f.write(f"Subject: {split_info['subject']}\n")
            f.write(f"Training: {split_info['train_size']} samples ({split_info['train_ratio']*100:.1f}%)\n")
            f.write(f"Validation: {split_info['val_size']} samples ({split_info['val_ratio']*100:.1f}%)\n")
            f.write(f"Testing: {split_info['test_size']} samples ({split_info['test_ratio']*100:.1f}%)\n")

    # --- Configuration for the Trainer ---
    trainer_config = TrainerConfig(
        batch_size=train_loader.batch_size, # Use batch size from loader
        epochs=15,  # Short training run for testing (increase for real training)
        learning_rate=0.001,
        mask_threshold=1e-8,
        optimizer_type="adam",
        scheduler_type="reduce_on_plateau", # Optional: reduce LR if val loss plateaus
        scheduler_patience=3,
        scheduler_factor=0.5,
        device=str(device),
        mixed_precision=False, # Start with False for simpler debugging
        checkpoint_dir=tracker.checkpoint_dir,
        validation_frequency=1, # Validate every epoch
        early_stopping=True,
        early_stopping_patience=5, # Stop if val loss doesn't improve for 5 epochs
        early_stopping_min_delta=0.0001,
        loss_type="magnitude_mse", # Custom identifier, actual logic is in _compute_loss
    )

    # Log trainer hyperparameters
    tracker.log_hyperparameters(trainer_config.__dict__)

    # Create trainer
    trainer = ModelTrainer(
        model,
        trainer_config,
        model_context,
        resource_monitor=resource_monitor,
        # debug_hook=None # Add if needed
    )

    # --- Prepare data for trainer (expects lists of tensors) ---
    # The trainer's TMSDataset will handle batching internally
    train_features_list = []
    train_targets_list = [] # List of target VECTORS
    for features_batch, targets_batch in train_loader:
        # Detach and move to CPU if needed, or keep on device if TMSDataset handles it
        train_features_list.extend([f.cpu() for f in features_batch])
        train_targets_list.extend([t.cpu() for t in targets_batch]) # Keep vectors

    val_features_list = []
    val_targets_list = [] # List of target VECTORS
    for features_batch, targets_batch in val_loader:
        val_features_list.extend([f.cpu() for f in features_batch])
        val_targets_list.extend([t.cpu() for t in targets_batch]) # Keep vectors

    logger.info(f"Prepared data lists: Train samples={len(train_features_list)}, Val samples={len(val_features_list)}")
    
    # Add detailed logging about the sizes
    logger.info(f"Preprocessed TRAIN data - Samples: {len(train_features_list)}")
    logger.info(f"Preprocessed VAL data - Samples: {len(val_features_list)}")
    
    if len(train_features_list) == 0 or len(val_features_list) == 0:
         logger.error("Data preparation resulted in empty lists. Stopping.")
         resource_monitor.stop_monitoring()
         tracker.finalize(status="FAILED")
         return None, {}


    # --- Train the model ---
    logger.info("Starting training...")
    start_time = time.time()
    history = trainer.train(train_features_list, train_targets_list, val_features_list, val_targets_list)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # --- Process and Plot Training History ---
    # Extract metrics from history (list of dicts)
    full_metrics_history = {"train": {}, "val": {}}
    if history and history.get("train"):
        for epoch_metrics in history["train"]:
            for key, value in epoch_metrics.items():
                if key not in full_metrics_history["train"]: full_metrics_history["train"][key] = []
                if value is not None: full_metrics_history["train"][key].append(value)
    if history and history.get("val"):
         for epoch_metrics in history["val"]:
            for key, value in epoch_metrics.items():
                if key not in full_metrics_history["val"]: full_metrics_history["val"][key] = []
                # Validation runs less often, might have None entries
                full_metrics_history["val"][key].append(value) # Keep None placeholders


    # Define metrics to plot in the summary (use new magnitude metric names)
    metrics_to_plot_in_summary = [
        'loss',
        'magnitude_mae',
        'magnitude_rmse',
        'magnitude_correlation',
        'hotspot_iou'
        # Add 'magnitude_rel_error' if desired and stable
    ]

    # Create the filtered metrics history dictionary for the plot
    filtered_train_metrics_history = {}
    for key in metrics_to_plot_in_summary:
        if key in full_metrics_history["train"] and full_metrics_history["train"][key]:
            filtered_train_metrics_history[key] = full_metrics_history["train"][key]
        else:
            logger.warning(f"Train metric '{key}' requested for plotting not found or empty in history.")

    # Create metrics summary plot using the filtered training data
    if filtered_train_metrics_history:
        metrics_plot_path = os.path.join(tracker.visualization_dir, "training_metrics_summary.png")
        try:
            fig_metrics = create_metrics_summary_plot(
                filtered_train_metrics_history,
                save_path=metrics_plot_path,
                title="Training Metrics Summary (Single-Subject Magnitude Prediction)"
            )
            if fig_metrics:
                 tracker.log_artifact("training_metrics_summary", metrics_plot_path)
                 logger.info(f"Saved training metrics summary plot to {metrics_plot_path}")
                 plt.close(fig_metrics) # Close plot
        except Exception as e:
            logger.error(f"Failed to create or save training metrics summary plot: {e}", exc_info=True)
    else:
        logger.warning("No valid training metrics found to generate the summary plot.")


    # --- Plot Train vs Validation Loss ---
    loss_plot_path = os.path.join(tracker.visualization_dir, "train_val_loss.png")
    try:
        fig_loss = plt.figure(figsize=(10, 6))
        train_losses = full_metrics_history["train"].get("loss", [])
        # Filter out None values from validation loss before plotting
        val_losses = [m for m in full_metrics_history["val"].get("loss", []) if m is not None]
        epochs_train = np.arange(1, len(train_losses) + 1)
        # Align validation epochs correctly (validation happens every N epochs)
        epochs_val = np.arange(1, len(history["val"]) + 1) * trainer_config.validation_frequency

        if train_losses: plt.plot(epochs_train, train_losses, 'bo-', label='Training Loss')
        if val_losses: plt.plot(epochs_val[:len(val_losses)], val_losses, 'ro-', label='Validation Loss') # Plot only valid points

        plt.title('Training and Validation Loss (Single-Subject Magnitude Prediction)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE Magnitude)')
        if train_losses or val_losses: plt.legend()
        plt.grid(True)
        plt.savefig(loss_plot_path, dpi=300)
        plt.close(fig_loss)
        tracker.log_artifact("train_val_loss", loss_plot_path)
        logger.info(f"Saved train/val loss plot to {loss_plot_path}")
    except Exception as e:
         logger.error(f"Failed to create train/val loss plot: {e}", exc_info=True)


    # --- Save Final Model ---
    if history["val"] and history["train"]: # Check if history exists
         final_epoch_metrics = history["val"][-1] if history["val"][-1] is not None else history["train"][-1]
         final_checkpoint_path = tracker.save_checkpoint(
             model=model,
             epoch=trainer.current_epoch, # Use actual last epoch run
             metrics=final_epoch_metrics
         )
         logger.info(f"Final model checkpoint saved: {final_checkpoint_path}")
    else:
         logger.warning("Could not save final model checkpoint due to missing history.")


    logger.info("Evaluating on test set...")
    model.eval()

    all_pred_magnitudes = []
    all_target_magnitudes = []

    with torch.no_grad():
        for features, targets_vec in test_loader: # targets_vec are [B, 3, D, H, W]
            features = features.to(device)
            
            # Forward pass -> outputs are predicted magnitudes [B, 1, D, H, W]
            outputs_mag = model(features)

            # Calculate target magnitude on CPU (as metrics are done on CPU anyway)
            targets_mag = torch.sqrt(torch.sum(targets_vec**2, dim=1, keepdim=True)) # [B, 1, D, H, W]

            # Store magnitudes for metrics calculation
            all_pred_magnitudes.append(outputs_mag.cpu())
            all_target_magnitudes.append(targets_mag.cpu()) # Already on CPU

    if not all_pred_magnitudes or not all_target_magnitudes:
        logger.error("Test loop yielded no predictions or targets. Cannot evaluate.")
        resource_monitor.stop_monitoring()
        tracker.finalize(status="FAILED")
        return model, {}

    try:
        # Concatenate lists of tensors into single tensors
        all_preds_mag_tensor = torch.cat(all_pred_magnitudes, dim=0)   # [N_test, 1, D, H, W]
        all_targets_mag_tensor = torch.cat(all_target_magnitudes, dim=0) # [N_test, 1, D, H, W]
        logger.info(f"Concatenated test predicted magnitudes shape: {all_preds_mag_tensor.shape}")
        logger.info(f"Concatenated test target magnitudes shape: {all_targets_mag_tensor.shape}")

        # Squeeze the channel dimension for metrics/visualization
        all_preds_mag_np = all_preds_mag_tensor.squeeze(1).numpy()   # [N_test, D, H, W]
        all_targets_mag_np = all_targets_mag_tensor.squeeze(1).numpy() # [N_test, D, H, W]

        # --- Create Mask for Test Set ---
        test_mask_np = (all_targets_mag_np > 1e-8) # Boolean mask [N_test, D, H, W]
        logger.info(f"Created test mask shape: {test_mask_np.shape}, "
                    f"Valid elements: {np.sum(test_mask_np)}/{test_mask_np.size}")

    except Exception as e:
        logger.error(f"Error concatenating/processing test results: {e}", exc_info=True)
        resource_monitor.stop_monitoring()
        tracker.finalize(status="FAILED")
        return model, {}


    # --- Calculate Metrics for Magnitude Prediction (with Mask) ---
    logger.info("Calculating metrics using mask...")
    test_metrics = calculate_magnitude_metrics(all_preds_mag_np, all_targets_mag_np, mask=test_mask_np)
    logger.info(f"Test metrics (Masked Magnitude): {test_metrics}")

    # Log test metrics
    tracker.log_metrics(test_metrics) # Log the masked metrics


    # --- Generate Standard Visualizations for Magnitude (with Mask) ---
    if all_preds_mag_np.shape[0] > 0: # Check if there are samples
        # Select the first sample for detailed visualization
        pred_mag_sample = all_preds_mag_np[0]     # [D, H, W]
        target_mag_sample = all_targets_mag_np[0] # [D, H, W]
        mask_sample = test_mask_np[0]             # [D, H, W]

        logger.info(f"DEBUG: Preparing to call generate_standard_visualizations (Magnitude with Mask).")
        logger.info(f"DEBUG: pred_mag_sample shape: {pred_mag_sample.shape}")
        logger.info(f"DEBUG: target_mag_sample shape: {target_mag_sample.shape}")
        logger.info(f"DEBUG: mask_sample shape: {mask_sample.shape}, Valid elements: {np.sum(mask_sample)}")

        try:
            # Generate visualizations
            vis_paths = generate_standard_visualizations(
                model_name=model_context.config.get("architecture", "simple_unet_magnitude"),
                pred_mag=pred_mag_sample,
                target_mag=target_mag_sample,
                mask=mask_sample,
                metrics_dict=test_metrics,
                output_dir=tracker.visualization_dir
            )
            
            # Log generated artifacts
            if isinstance(vis_paths, dict):
                for name, path in vis_paths.items():
                    if os.path.exists(path):
                        tracker.log_artifact(name, path)
                logger.info(f"Generated standard magnitude visualizations in {tracker.visualization_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate standard magnitude visualizations: {e}", exc_info=True)
            
            # Create a fallback visualization
            try:
                fallback_path = os.path.join(tracker.visualization_dir, "fallback_visualization.png")
                plt.figure(figsize=(12, 4))
                plt.subplot(131)
                plt.imshow(pred_mag_sample[:, 12, :], cmap='jet')
                plt.title('Predicted Magnitude')
                plt.subplot(132)
                plt.imshow(target_mag_sample[:, 12, :], cmap='jet')
                plt.title('Target Magnitude')
                plt.subplot(133)
                plt.imshow(mask_sample[:, 12, :], cmap='gray')
                plt.title('Mask')
                plt.savefig(fallback_path)
                plt.close()
                tracker.log_artifact("fallback_visualization", fallback_path)
                logger.info(f"Created fallback visualization at {fallback_path}")
            except Exception as e2:
                logger.error(f"Failed to create fallback visualization: {e2}", exc_info=True)
    
    # Finalize the experiment
    tracker.finalize(status="COMPLETED") # Set status
    # Stop resource monitoring
    resource_monitor.stop_monitoring()
    return model, test_metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train and evaluate a TMS E-field MAGNITUDE prediction model using a single subject')
    parser.add_argument('--data-dir', type=str, default='/home/freyhe/MA_Henry/data', help='Data directory containing subject folders')
    parser.add_argument('--output-dir', type=str, default='test_output_single_subject', help='Output directory for this test run')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training and evaluation')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of processed data')
    args = parser.parse_args()

    # Setup for single subject approach
    subject_id = '001'  # Single subject
    train_ratio = 0.7   # 70% for training
    val_ratio = 0.2     # 20% for validation
    test_ratio = 0.1    # 10% for testing
    output_shape = (25, 25, 25)
    
    logger.info(f"Preparing single-subject datasets. Subject: {subject_id}, "
               f"Split: {train_ratio*100:.1f}%/{val_ratio*100:.1f}%/{test_ratio*100:.1f}% (train/val/test)")
    
    # Prepare datasets
    use_cache = not args.no_cache
    logger.info(f"Data caching is {'disabled' if args.no_cache else 'enabled'}")
    
    train_dataset, val_dataset, test_dataset = prepare_single_subject_datasets(
        subject_id=subject_id,
        data_root_path=args.data_dir,
        output_shape=output_shape,
        use_cache=use_cache,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Create DataLoaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # Log detailed dataset counts
    logger.info(f"DATASET SIZES - Train: {len(train_dataset)} samples, "
               f"Validation: {len(val_dataset)} samples, Test: {len(test_dataset)} samples")
    
    # Create split info dictionary for logging
    split_info = {
        'subject': subject_id,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio
    }
    
    # Train and evaluate
    logger.info("Training and evaluating UNet magnitude model with single-subject data...")
    model, test_metrics = train_and_evaluate(
        train_loader, 
        val_loader, 
        test_loader, 
        args.output_dir,
        split_info
    )
    
    # Print final results
    if test_metrics:
        logger.info("--- Final Test Metrics (Single-Subject Magnitude Prediction) ---")
        logger.info(f"Subject: {subject_id}")
        for key, value in test_metrics.items():
            # Format based on type (avoid formatting None)
            if isinstance(value, (float, np.float32, np.float64)):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("-" * 40)
    else:
        logger.warning("Test evaluation did not produce metrics.")
    
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("Script finished.")


if __name__ == "__main__":
    main()