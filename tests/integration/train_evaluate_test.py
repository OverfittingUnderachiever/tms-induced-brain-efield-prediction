# train_evaluate_test.py
"""
Integration test for training and evaluation workflow (Magnitude Prediction).
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

# --- Import the NEW Model ---
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
from tms_efield_prediction.data.data_splitter import DataSplitter # ADDED DataSplitter import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger('train_evaluate_magnitude_test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate(train_loader, val_loader, test_loader, output_dir='test_output_magnitude'):
    """Train a UNet magnitude model and evaluate its performance.

    Args:
        train_loader: Training data loader. Yields (features_channels_first, target_vectors).
        val_loader: Validation data loader. Yields (features_channels_first, target_vectors).
        test_loader: Test data loader. Yields (features_channels_first, target_vectors).
        output_dir: Directory for outputs.

    Returns:
        Tuple of (trained model, test metrics dictionary).
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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

    # --- Instantiate the NEW Model ---
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
    tracker.set_description("Magnitude Prediction Test Run with Simple UNet")
    tracker.log_hyperparameters(unet_config) # Log model config

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
        # loss_weights removed
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
    if not train_features_list or not val_features_list:
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
                title="Training Metrics Summary (Magnitude Prediction)"
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

        plt.title('Training and Validation Loss (Magnitude Prediction)')
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
    # all_target_vectors = [] # Optional: Keep if needed elsewhere

    with torch.no_grad():
        for features, targets_vec in test_loader: # targets_vec are [B, 3, D, H, W]
            features = features.to(device)
            # No need to move targets_vec to device if only used for magnitude calc on CPU later
            # targets_vec = targets_vec.to(device)

            # Forward pass -> outputs are predicted magnitudes [B, 1, D, H, W]
            outputs_mag = model(features)

            # Calculate target magnitude on CPU (as metrics are done on CPU anyway)
            targets_mag = torch.sqrt(torch.sum(targets_vec**2, dim=1, keepdim=True)) # [B, 1, D, H, W]

            # Store magnitudes for metrics calculation
            all_pred_magnitudes.append(outputs_mag.cpu())
            all_target_magnitudes.append(targets_mag.cpu()) # Already on CPU
            # all_target_vectors.append(targets_vec.cpu())

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
            # Pass scalar fields AND the mask to the visualization function
            standard_vis_paths = generate_standard_visualizations(
                model_name=model_context.config.get("architecture", "simple_unet_magnitude"),
                pred_mag=pred_mag_sample,      # Pass 3D NumPy array
                target_mag=target_mag_sample,  # Pass 3D NumPy array
                mask=mask_sample,              # Pass the 3D boolean mask for this sample
                metrics_dict=test_metrics,     # Pass the calculated metrics dictionary
                output_dir=tracker.visualization_dir
            )
            # Log generated artifacts
            for name, path in standard_vis_paths.items():
                 tracker.log_artifact(name, path)
            logger.info(f"Generated standard magnitude visualizations (with mask) in {tracker.visualization_dir}")

        except Exception as e:
            logger.error(f"Failed to generate standard magnitude visualizations (with mask): {e}", exc_info=True)
    else:
        logger.warning("No test samples found to generate visualizations.")

    # --- Rest of the code remains the same ---
    # Finalize the experiment
    tracker.finalize(status="COMPLETED") # Set status
    # Stop resource monitoring
    resource_monitor.stop_monitoring()
    return model, test_metrics


def get_datasets(features, targets, train_pct=0.6, val_pct=0.2, test_pct=0.2, random_seed = 42):
    """Split data into training, validation, and test sets.

    Args:
        features: Features tensor [N, D, H, W, C_stacked] (channels LAST expected from loader)
        targets: Targets tensor [N, 3, D, H, W] (E-field vectors, channels FIRST expected from loader)
         train_pct (float): Percentage of data for training (default: 0.6)
        val_pct (float): Percentage of data for validation (default: 0.2)
        test_pct (float): Percentage of data for testing (default: 0.2)
        random_seed (int): Random seed for reproducibility (default: 42)

    Returns:
        Tuple of DataLoader objects for training, validation, and test sets.
        DataLoaders yield (features_channels_first, targets_vectors).
    """

    # Move to CPU before creating the dataset
    features = features.cpu()
    targets = targets.cpu()

    # Transform features from [N, D, H, W, C] (channels-last) to [N, C, D, H, W] (channels-first) for PyTorch Conv3D
    if features.shape[-1] < features.shape[1] : # Basic check if channels seem last
        logger.info("Permuting features from Channels Last -> Channels First")
        features_transformed = features.permute(0, 4, 1, 2, 3)
    else:
        logger.warning(f"Input features shape {features.shape} does not look like channels-last. Assuming channels-first.")
        features_transformed = features # Assume already channels-first

    # Targets are expected as [N, 3, D, H, W] (vectors, channels-first) - no permute needed
    targets_vectors = targets

    logger.info(f"Features shape for Dataset: {features_transformed.shape}")
    logger.info(f"Targets shape for Dataset: {targets_vectors.shape}")

    # Create dataset (Features are Channels First, Targets are Vectors)
    dataset = TensorDataset(features_transformed, targets_vectors)

    # Calculate split sizes
    n_samples = len(dataset)
    n_train = int(n_samples * train_pct)
    n_val = int(n_samples * val_pct)
    # Adjust test size to ensure total is n_samples
    n_test = n_samples - n_train - n_val
    if n_train + n_val + n_test != n_samples:
        logger.warning(f"Split sizes don't sum perfectly ({n_train}+{n_val}+{n_test}!={n_samples}). Adjusting train size.")
        n_train = n_samples - n_val - n_test # Adjust train size


    logger.info(f"Calculated split sizes: Train={n_train}, Val={n_val}, Test={n_test} (Total={n_samples})")

    # Split dataset
    if n_train <= 0 or n_val < 0 or n_test <= 0:
        raise ValueError(f"Invalid split sizes: Train={n_train}, Val={n_val}, Test={n_test}. Check percentages and data size.")

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(random_seed)  # For reproducibility
    )

    return train_dataset, val_dataset, test_dataset


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train and evaluate a TMS E-field MAGNITUDE prediction model')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory containing subject5/features.npy and subject5/targets.npy')
    parser.add_argument('--output-dir', type=str, default='test_output_magnitude', help='Output directory for this test run')
    parser.add_argument('--train-pct', type=float, default=0.6, help='Percentage of data for training')
    parser.add_argument('--val-pct', type=float, default=0.2, help='Percentage of data for validation')
    parser.add_argument('--test-pct', type=float, default=0.2, help='Percentage of data for testing')
    args = parser.parse_args()

    # 1. Define the TMSPipelineContext
    data_root_path = "/home/freyhe/MA_Henry/data"  # example. Adjust this to where your data is
    subject_id = "004"  # example
    output_shape = (25, 25, 25)
    normalization_method = "standard"  # example

    data_root_path = "/home/freyhe/MA_Henry/data"  # example. Adjust this to where your data is
    subject_id = "004"  # example
    output_shape = (25, 25, 25)
    normalization_method = "standard"  # example

    tms_config = {"mri_tensor": None, "device": device} # ADD device to config
    pipeline_context = TMSPipelineContext(
        dependencies={},
        config=tms_config,
        pipeline_mode="mri_dadt",  # CHANGE THIS LINE
        experiment_phase="training",
        debug_mode=True,
        subject_id=subject_id,
        data_root_path=data_root_path,
        output_shape=output_shape,
        normalization_method=normalization_method,
        device=device
    )

    # 2. Instantiate TMSDataLoader and load data
    data_loader = TMSDataLoader(context=pipeline_context)
    raw_data = data_loader.load_raw_data()
    if raw_data is None:
        raise ValueError("Raw Data is None Please Check DataLoader")

    # Update tms_config
    tms_config["mri_tensor"] = raw_data.mri_tensor
    pipeline_context.config = tms_config

    # 3. Create sample list and process data
    samples = data_loader.create_sample_list(raw_data)
    stacking_pipeline = EnhancedStackingPipeline(context=pipeline_context)
    processed_data_list = stacking_pipeline.process_batch(samples)

    # 4. Extract features and targets
    features_list = [processed_data.input_features for processed_data in processed_data_list]
    targets_list = [processed_data.target_efield for processed_data in processed_data_list]
    features_stacked = torch.stack(features_list)
    targets_vectors = torch.stack(targets_list)

    # Log the shapes
    logger.info(f"Features Stacked Shape: {features_stacked.shape}")
    logger.info(f"Targets Vectors Shape: {targets_vectors.shape}")

    # 5. Split data using DataSplitter
    train_dataset, val_dataset, test_dataset = get_datasets(features_stacked, targets_vectors,  train_pct=args.train_pct, val_pct=args.val_pct, test_pct=args.test_pct)
    # 6. Create DataLoaders
    batch_size = 8  # Define batch size here or get from config
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Train and evaluate
    logger.info("Training and evaluating UNet magnitude model...")
    model, test_metrics = train_and_evaluate(train_loader, val_loader, test_loader, args.output_dir)

    # Print final results
    if test_metrics:
        logger.info("--- Final Test Metrics (Magnitude Prediction) ---")
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