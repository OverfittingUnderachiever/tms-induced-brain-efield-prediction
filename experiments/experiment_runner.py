# tms_efield_prediction/experiments/experiment_runner.py

import os
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from datetime import datetime

from ..models.architectures.simple_unet_magnitude import SimpleUNetMagnitudeModel
from ..models.training.trainer import ModelTrainer, TrainerConfig
from ..models.evaluation.metrics import calculate_magnitude_metrics
from ..models.evaluation.visualization import generate_standard_visualizations, create_metrics_summary_plot
from ..utils.state.context import ModelContext
from ..utils.resource.monitor import ResourceMonitor
from .tracking import ExperimentTracker

logger = logging.getLogger(__name__)

class MagnitudeExperimentRunner:
    """
    High-level experiment runner for TMS E-field magnitude prediction models.
    Handles the complete workflow of experiment setup, model training, evaluation, and results visualization.
    """
    
    def __init__(
        self, 
        experiment_dir: str = 'experiment_output',
        architecture_name: str = 'simple_unet_magnitude',
        description: str = 'TMS E-field Magnitude Prediction Experiment',
        resource_monitor_max_gb: int = 8,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the experiment runner.
        
        Args:
            experiment_dir: Directory to save experiment outputs
            architecture_name: Name of the model architecture
            description: Experiment description
            resource_monitor_max_gb: Maximum GB for resource monitor
            device: Device to run on (defaults to CUDA if available)
        """
        self.experiment_dir = experiment_dir
        self.architecture_name = architecture_name
        self.description = description
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create timestamp for this experiment
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment tracker
        self.tracker = ExperimentTracker(
            experiment_dir=experiment_dir,
            architecture_name=architecture_name,
            create_subdirs=True
        )
        self.tracker.set_description(description)
        
        # Create resource monitor
        self.resource_monitor = ResourceMonitor(
            max_memory_gb=resource_monitor_max_gb,
            check_interval=10.0
        )
        self.resource_monitor.start_monitoring()
        
        # Initialize other attributes that will be set later
        self.model = None
        self.trainer = None
        self.model_config = None
        self.trainer_config = None
        self.training_history = None
        self.test_metrics = None
    
    def visualize_epoch_data(self, epoch, train_loader, val_loader):
        """
        Create visualizations for the current epoch's data.
        
        Args:
            epoch: Current epoch number
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        from tms_efield_prediction.models.evaluation.visualization import (
            visualize_epoch_samples,
            visualize_data_statistics
        )
        
        # Log this action
        logging.info(f"Creating data visualizations for epoch {epoch}...")
        
        # Generate sample visualizations
        visualize_epoch_samples(
            self.experiment_dir, 
            epoch, 
            train_loader, 
            val_loader,
            y_slices=[10, 15, 20],
            num_samples=5,
            num_batches=5
        )
        
        # Generate statistics visualizations
        visualize_data_statistics(
            self.experiment_dir,
            epoch,
            train_loader,
            val_loader
        )
        
        logging.info(f"Visualizations for epoch {epoch} completed")



    def configure_model(self, config: Dict[str, Any]) -> None:
        """
        Configure the model for the experiment.
        
        Args:
            config: Model configuration dictionary
        """
        self.model_config = config
        
        # Create model
        self.model = SimpleUNetMagnitudeModel(config=config)
        self.model.to(self.device)
        
        # Create model context
        self.model_context = ModelContext(
            dependencies={},
            config={"architecture": self.architecture_name}
        )
        
        # Log configuration
        self.tracker.log_hyperparameters(config)
        logger.info(f"Model configured: {self.architecture_name}")
    
    def configure_trainer(self, config: Dict[str, Any]) -> None:
        """
        Configure the trainer for the experiment.
        
        Args:
            config: Trainer configuration dictionary
        """
        # Create trainer config
        self.trainer_config = TrainerConfig(**config)
        
        # Update config if needed
        if not hasattr(self.trainer_config, 'checkpoint_dir') or not self.trainer_config.checkpoint_dir:
            self.trainer_config.checkpoint_dir = self.tracker.checkpoint_dir
        
        if not hasattr(self.trainer_config, 'device') or not self.trainer_config.device:
            self.trainer_config.device = str(self.device)
        
        # Log configuration
        self.tracker.log_hyperparameters(self.trainer_config.__dict__)
        logger.info("Trainer configured")
    
    def train_and_evaluate(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Run the complete experiment: training, evaluation, and visualization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            
        Returns:
            Dict with experiment results
        """
        if self.model is None or self.trainer_config is None:
            raise ValueError("Model and trainer must be configured before running the experiment")
        
        # Create trainer
        self.trainer = ModelTrainer(
            self.model,
            self.trainer_config,
            self.model_context,
            resource_monitor=self.resource_monitor
        )
        
        # Extract data for trainer (expects lists of tensors)
        train_features_list = []
        train_targets_list = []
        for features_batch, targets_batch in train_loader:
            train_features_list.extend([f.cpu() for f in features_batch])
            train_targets_list.extend([t.cpu() for t in targets_batch])
        
        val_features_list = []
        val_targets_list = []
        for features_batch, targets_batch in val_loader:
            val_features_list.extend([f.cpu() for f in features_batch])
            val_targets_list.extend([t.cpu() for t in targets_batch])
        
        logger.info(f"Prepared data lists: Train samples={len(train_features_list)}, Val samples={len(val_features_list)}")
        
        # Train the model
        logger.info("Starting training...")
        start_time = time.time()
        self.training_history = self.trainer.train(train_features_list, train_targets_list, val_features_list, val_targets_list)
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Process training history for visualization
        self._visualize_training_history()
        
        # Evaluate on test set
        self._evaluate_on_test_set(test_loader)
        
        # Save final model
        if self.training_history["val"] and self.training_history["train"]:
            final_epoch_metrics = self.training_history["val"][-1] if self.training_history["val"][-1] is not None else self.training_history["train"][-1]
            final_checkpoint_path = self.tracker.save_checkpoint(
                model=self.model,
                epoch=self.trainer.current_epoch,
                metrics=final_epoch_metrics
            )
            logger.info(f"Final model checkpoint saved: {final_checkpoint_path}")
        
        # Finalize experiment
        self.tracker.finalize(status="COMPLETED")
        self.resource_monitor.stop_monitoring()
        
        # Return results
        return {
            "model": self.model,
            "test_metrics": self.test_metrics,
            "training_history": self.training_history,
            "training_time": training_time
        }
    
    def _visualize_training_history(self) -> None:
        """Visualize the training history."""
        if not self.training_history:
            logger.warning("No training history to visualize")
            return
        
        # Extract metrics from history
        full_metrics_history = {"train": {}, "val": {}}
        if self.training_history.get("train"):
            for epoch_metrics in self.training_history["train"]:
                for key, value in epoch_metrics.items():
                    if key not in full_metrics_history["train"]: full_metrics_history["train"][key] = []
                    if value is not None: full_metrics_history["train"][key].append(value)
        
        if self.training_history.get("val"):
            for epoch_metrics in self.training_history["val"]:
                for key, value in epoch_metrics.items():
                    if key not in full_metrics_history["val"]: full_metrics_history["val"][key] = []
                    full_metrics_history["val"][key].append(value)
        
        # Define metrics to plot
        metrics_to_plot = [
            'loss',
            'magnitude_mae',
            'magnitude_rmse',
            'magnitude_correlation',
            'hotspot_iou'
        ]
        
        # Filter metrics for plotting
        filtered_train_metrics_history = {}
        for key in metrics_to_plot:
            if key in full_metrics_history["train"] and full_metrics_history["train"][key]:
                filtered_train_metrics_history[key] = full_metrics_history["train"][key]
        
        # Create metrics summary plot
        if filtered_train_metrics_history:
            metrics_plot_path = os.path.join(self.tracker.visualization_dir, "training_metrics_summary.png")
            try:
                fig_metrics = create_metrics_summary_plot(
                    filtered_train_metrics_history,
                    save_path=metrics_plot_path,
                    title="Training Metrics Summary (Magnitude Prediction)"
                )
                if fig_metrics:
                    self.tracker.log_artifact("training_metrics_summary", metrics_plot_path)
                    plt.close(fig_metrics)
            except Exception as e:
                logger.error(f"Failed to create metrics summary plot: {e}", exc_info=True)
        
        # Plot train vs validation loss
        loss_plot_path = os.path.join(self.tracker.visualization_dir, "train_val_loss.png")
        try:
            fig_loss = plt.figure(figsize=(10, 6))
            train_losses = full_metrics_history["train"].get("loss", [])
            # Filter out None values from validation loss
            val_losses = [m for m in full_metrics_history["val"].get("loss", []) if m is not None]
            epochs_train = np.arange(1, len(train_losses) + 1)
            # Align validation epochs correctly
            epochs_val = np.arange(1, len(self.training_history["val"]) + 1) * self.trainer_config.validation_frequency
            
            if train_losses: plt.plot(epochs_train, train_losses, 'bo-', label='Training Loss')
            if val_losses: plt.plot(epochs_val[:len(val_losses)], val_losses, 'ro-', label='Validation Loss')
            
            plt.title('Training and Validation Loss (Magnitude Prediction)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (MSE Magnitude)')
            if train_losses or val_losses: plt.legend()
            plt.grid(True)
            plt.savefig(loss_plot_path, dpi=300)
            plt.close(fig_loss)
            self.tracker.log_artifact("train_val_loss", loss_plot_path)
        except Exception as e:
            logger.error(f"Failed to create train/val loss plot: {e}", exc_info=True)
    
    def _evaluate_on_test_set(self, test_loader: torch.utils.data.DataLoader) -> None:
        """
        Evaluate the model on the test set.
        
        Args:
            test_loader: Test data loader
        """
        logger.info("Evaluating on test set...")
        self.model.eval()
        
        all_pred_magnitudes = []
        all_target_magnitudes = []
        
        with torch.no_grad():
            for features, targets_vec in test_loader:
                features = features.to(self.device)
                # Predict magnitudes
                outputs_mag = self.model(features)
                # Calculate target magnitude
                targets_mag = torch.sqrt(torch.sum(targets_vec**2, dim=1, keepdim=True))
                
                # Store magnitudes for metrics calculation
                all_pred_magnitudes.append(outputs_mag.cpu())
                all_target_magnitudes.append(targets_mag.cpu())
        
        if not all_pred_magnitudes or not all_target_magnitudes:
            logger.error("Test loop yielded no predictions or targets. Cannot evaluate.")
            return
        
        try:
            # Concatenate lists of tensors into single tensors
            all_preds_mag_tensor = torch.cat(all_pred_magnitudes, dim=0)
            all_targets_mag_tensor = torch.cat(all_target_magnitudes, dim=0)
            
            # Remove channel dimension for metrics calculation
            all_preds_mag_np = all_preds_mag_tensor.squeeze(1).numpy()
            all_targets_mag_np = all_targets_mag_tensor.squeeze(1).numpy()
            
            # Create mask for test set
            test_mask_np = (all_targets_mag_np > 1e-8)
            
            # Calculate metrics
            self.test_metrics = calculate_magnitude_metrics(all_preds_mag_np, all_targets_mag_np, mask=test_mask_np)
            logger.info(f"Test metrics: {self.test_metrics}")
            
            # Log metrics
            self.tracker.log_metrics(self.test_metrics)
            
            # Generate visualizations
            if all_preds_mag_np.shape[0] > 0:
                # Select the first sample for visualization
                pred_mag_sample = all_preds_mag_np[0]
                target_mag_sample = all_targets_mag_np[0]
                mask_sample = test_mask_np[0]
                
                # Generate visualizations
                standard_vis_paths = generate_standard_visualizations(
                    model_name=self.architecture_name,
                    pred_mag=pred_mag_sample,
                    target_mag=target_mag_sample,
                    mask=mask_sample,
                    metrics_dict=self.test_metrics,
                    output_dir=self.tracker.visualization_dir
                )
                
                # Log artifacts
                for name, path in standard_vis_paths.items():
                    self.tracker.log_artifact(name, path)
        
        except Exception as e:
            logger.error(f"Error during test evaluation: {e}", exc_info=True)