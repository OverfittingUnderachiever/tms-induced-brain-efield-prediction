"""
TMS E-field Prediction Model Trainer Implementation

This module provides a configurable, memory-aware training infrastructure for TMS E-field prediction models.
It implements a ModelTrainer class with support for callbacks, metrics tracking, and resource monitoring.
"""

import os
import time
import gc
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from ...utils.state.context import ModelContext
from ...utils.debug.hooks import DebugHook
from ...utils.resource.monitor import ResourceMonitor
from ...models.evaluation.metrics import calculate_magnitude_metrics
from ..evaluation.metrics import calculate_metrics
from .callbacks import TrainingCallback, EarlyStoppingCallback, ModelCheckpointCallback
from .schedulers import get_scheduler
from tms_efield_prediction.constants import EFIELD_MASK_THRESHOLD
import logging
from .losses import LossFactory

logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    """Configuration for model training."""

    # Basic training parameters
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer_type: str = "adam"
    scheduler_type: Optional[str] = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    loss_type: str = "magnitude_mse"
    mask_threshold: float = EFIELD_MASK_THRESHOLD
    hotspot_percentile: float = 90.0
    hotspot_weight: float = 5.0
    gradient_lambda: float = 0.5
    weighted_mse_factor: float = 2.0
    overlap_alpha: float = 0.5
    # Loss function configuration

    loss_weights: Dict[str, float] = field(default_factory=lambda: {"magnitude": 0.7, "direction": 0.3})
    
    loss_type: str = "magnitude_mse"
    mask_threshold: float = EFIELD_MASK_THRESHOLD

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    # ADD GRADIENT CHECKPOINTING HERE:
    gradient_checkpointing: bool = True
    
    # Checkpoint configuration
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    save_frequency: int = 5
    max_models_to_keep: int = 20
    
    # Validation configuration
    validation_frequency: int = 1
    validation_split: float = 0.2
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Resource management
    memory_cleanup_frequency: int = 1
    gradient_accumulation_steps: int = 1
    
    # Debug configuration
    debug_print_frequency: int = 10
    
    def validate(self) -> bool:
        """Validate the configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Basic validation
        if self.batch_size <= 0:
            return False
        if self.epochs <= 0:
            return False
        if self.learning_rate <= 0:
            return False
            
        # Ensure we have a valid device
        if self.device not in ["cuda", "cpu"]:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        
        # Don't use mixed precision on CPU
        if self.device == "cpu":
            self.mixed_precision = False
            
        return True

class TMSDataset(Dataset):
    """Simple dataset for TMS E-field prediction training."""
    
    def __init__(self, 
            features: List[torch.Tensor], 
            targets: List[torch.Tensor],
            debug_hook: Optional[DebugHook] = None,
            config: Optional[TrainerConfig] = None):  # Add config parameter
        """Initialize the TMS dataset.
        
        Args:
            features: List of input features (MRI and dA/dt data)
            targets: List of target E-fields
            debug_hook: Optional debug hook
            config: Trainer configuration
        """
        self.features = features
        self.targets = targets
        self.debug_hook = debug_hook
        self.config = config  # Store config
        
        # Only create loss_fn if config is provided
        if self.config is not None:
            self.loss_fn = LossFactory.create_loss(
                loss_type=self.config.loss_type,
                mask_threshold=self.config.mask_threshold,
                hotspot_percentile=getattr(self.config, 'hotspot_percentile', 90.0),
                hotspot_weight=getattr(self.config, 'hotspot_weight', 5.0),
                lambda_grad=getattr(self.config, 'gradient_lambda', 0.5),
                weight_factor=getattr(self.config, 'weighted_mse_factor', 2.0),
                alpha=getattr(self.config, 'overlap_alpha', 0.5)
            )
            # Validate
        if len(features) != len(targets):
            raise ValueError(f"Features and targets must have same length. Got {len(features)} vs {len(targets)}")
        
        if debug_hook:
            debug_hook.record_event("dataset_init", {
                "samples": len(features),
                "feature_shape": features[0].shape if features else None,
                "target_shape": targets[0].shape if targets else None
            })
    
    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            int: Number of samples
        """
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Features and target
        """
        feature = self.features[idx]
        target = self.targets[idx]
        
        return feature, target


class ModelTrainer:
    """Trainer for TMS E-field prediction models with configurable training loops."""
    
    def __init__(self, 
                model: torch.nn.Module, 
                config: TrainerConfig,
                model_context: ModelContext,
                resource_monitor: Optional[ResourceMonitor] = None,
                debug_hook: Optional[DebugHook] = None):
        """Initialize the model trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            model_context: Model context
            resource_monitor: Optional resource monitor
            debug_hook: Optional debug hook
        """
        self.model = model
        self.config = config
        self.model_context = model_context
        self.resource_monitor = resource_monitor
        self.debug_hook = debug_hook
        
        # Validate config
        if not config.validate():
            raise ValueError("Invalid trainer configuration")
        
        # Initialize device
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # ADD GRADIENT CHECKPOINTING HERE:
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = get_scheduler(self.optimizer, config)
        
        # Initialize metrics tracking
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Initialize callbacks
        self.callbacks: List[TrainingCallback] = []
        self._setup_default_callbacks()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Set up resource monitoring
        if self.resource_monitor:
            self.resource_monitor.register_component(
                "model_trainer", 
                self._reduce_memory,
                priority=1  # High priority
            )
    

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        import torch.utils.checkpoint as checkpoint
        
        # Log that we're enabling checkpointing
        logger.info("Enabling gradient checkpointing for memory efficiency")
        
        # Your models have .encoders and .decoders ModuleLists
        if hasattr(self.model, 'encoders'):
            for i, encoder in enumerate(self.model.encoders):
                # Wrap the forward method with checkpointing
                original_forward = encoder.forward
                def make_checkpoint_wrapper(original):
                    def checkpoint_wrapper(x):
                        return checkpoint.checkpoint(original, x, use_reentrant=False)
                    return checkpoint_wrapper
                encoder.forward = make_checkpoint_wrapper(original_forward)
                logger.debug(f"Enabled checkpointing for encoder {i}")
        
        if hasattr(self.model, 'decoders'):
            for i, decoder in enumerate(self.model.decoders):
                # Decoders take two inputs (x, skip), so we need a different wrapper
                original_forward = decoder.forward
                def make_checkpoint_wrapper_dual(original):
                    def checkpoint_wrapper(x, skip):
                        return checkpoint.checkpoint(original, x, skip, use_reentrant=False)
                    return checkpoint_wrapper
                decoder.forward = make_checkpoint_wrapper_dual(original_forward)
                logger.debug(f"Enabled checkpointing for decoder {i}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration.
        
        Returns:
            torch.optim.Optimizer: PyTorch optimizer
        """
        params = self.model.parameters()
        
        if self.config.optimizer_type.lower() == "adam":
            return torch.optim.Adam(params, lr=self.config.learning_rate)
        elif self.config.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(params, lr=self.config.learning_rate, momentum=0.9)
        elif self.config.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(params, lr=self.config.learning_rate)
        else:
            # Default to Adam
            return torch.optim.Adam(params, lr=self.config.learning_rate)
    
    def _setup_default_callbacks(self):
        """Set up default callbacks based on configuration."""
        # Early stopping callback
        if self.config.early_stopping:
            early_stopping = EarlyStoppingCallback(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                monitor='val_loss'
            )
            self.callbacks.append(early_stopping)
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpointCallback(
            checkpoint_dir=self.config.checkpoint_dir,
            save_best_only=self.config.save_best_only,
            save_frequency=self.config.save_frequency,
            monitor='val_loss',
            max_models_to_keep=self.config.max_models_to_keep  # ADD THIS LINE
        )
        self.callbacks.append(checkpoint_callback)
    
    def _reduce_memory(self, target_reduction: float) -> None:
        """Reduce memory usage when requested by resource monitor.
        
        Args:
            target_reduction: Target reduction percentage (0.0-1.0)
        """
        if self.debug_hook:
            self.debug_hook.record_event("memory_reduction", {
                "target_reduction": target_reduction,
                "current_epoch": self.current_epoch
            })
        
        # Clear GPU cache if using CUDA
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
    
    def train(self, 
             train_features: List[torch.Tensor], 
             train_targets: List[torch.Tensor],
             val_features: Optional[List[torch.Tensor]] = None,
             val_targets: Optional[List[torch.Tensor]] = None) -> Dict[str, List[Dict[str, float]]]:
        """Train the model.
        
        Args:
            train_features: Training features
            train_targets: Training targets
            val_features: Optional validation features
            val_targets: Optional validation targets
            
        Returns:
            Dict[str, List[Dict[str, float]]]: Training history
        """
        # Create datasets

        train_dataset = TMSDataset(train_features, train_targets, self.debug_hook, self.config)
        
        # Create validation set if not provided
        if val_features is None or val_targets is None:
            train_size = int(len(train_dataset) * (1 - self.config.validation_split))
            val_size = len(train_dataset) - train_size
            
            if self.debug_hook:
                self.debug_hook.record_event("validation_split", {
                    "train_size": train_size,
                    "val_size": val_size,
                    "total_size": len(train_dataset)
                })
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
        else:

            val_dataset = TMSDataset(val_features, val_targets, self.debug_hook, self.config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Single process for compatibility
            pin_memory=self.device.type == "cuda"
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # Single process for compatibility
            pin_memory=self.device.type == "cuda"
        )
        
        # Initialize training
        self.model.train()
        
        # Call on_train_begin for callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        # Main training loop
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Call on_epoch_begin for callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            self.train_metrics.append(train_metrics)
            
            # Validation phase
            if epoch % self.config.validation_frequency == 0:
                val_metrics = self._validate_epoch(val_loader)
                self.val_metrics.append(val_metrics)
                
                # Update best validation loss
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
            else:
                # Placeholder for consistency
                self.val_metrics.append({"loss": None})
            
            # Call on_epoch_end for callbacks
            stop_training = False
            for callback in self.callbacks:
                if callback.on_epoch_end(self, epoch, 
                                         self.train_metrics[-1], 
                                         self.val_metrics[-1]):
                    stop_training = True
            
            # Print progress
            if epoch % self.config.debug_print_frequency == 0:
                val_loss_str = f"{self.val_metrics[-1]['loss']:.4f}" if self.val_metrics[-1]['loss'] is not None else "N/A"
                print(f"Epoch {epoch}/{self.config.epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_loss_str}")
            
            # Memory cleanup
            if epoch % self.config.memory_cleanup_frequency == 0:
                self._reduce_memory(0.0)  # Just trigger cleanup
            
            # Check if we should stop
            if stop_training:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Call on_train_end for callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        # Return training history
        return {
            "train": self.train_metrics,
            "val": self.val_metrics
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        total_loss = 0.0
        batch_metrics = []
        
        # Set up progress tracking
        start_time = time.time()
        total_batches = len(train_loader)
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            # Call on_batch_begin for callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)
            
            # Move tensors to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision if enabled
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(features)
                    loss = self._compute_loss(outputs, targets)
            else:
                outputs = self.model(features)
                loss = self._compute_loss(outputs, targets)
            
            # Normalize loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass with mixed precision if enabled
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Step optimizer if needed
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                # Step optimizer if needed
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Compute metrics
            metrics = self._compute_metrics(outputs, targets)
            metrics["loss"] = loss.item() * self.config.gradient_accumulation_steps  # Rescale for reporting
            batch_metrics.append(metrics)
            
            total_loss += metrics["loss"]
            
            # Call on_batch_end for callbacks
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, metrics)
        
        # Average metrics across batches
        avg_metrics = {
            "loss": total_loss / total_batches
        }
        
        # Add other metrics
        for key in batch_metrics[0].keys():
            if key != "loss":
                avg_metrics[key] = sum(m[key] for m in batch_metrics) / len(batch_metrics)
        
        # Update learning rate scheduler if needed
        if self.scheduler is not None and self.config.scheduler_type == "step":
            self.scheduler.step()
        
        # Record timing
        avg_metrics["epoch_time"] = time.time() - start_time
        
        return avg_metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        batch_metrics = []
        
        # Set up progress tracking
        start_time = time.time()
        total_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(val_loader):
                # Move tensors to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = self._compute_loss(outputs, targets)
                
                # Compute metrics
                metrics = self._compute_metrics(outputs, targets)
                metrics["loss"] = loss.item()
                batch_metrics.append(metrics)
                
                total_loss += metrics["loss"]
        
        # Average metrics across batches
        avg_metrics = {
            "loss": total_loss / total_batches
        }
        
        # Add other metrics
        for key in batch_metrics[0].keys():
            if key != "loss":
                avg_metrics[key] = sum(m[key] for m in batch_metrics) / len(batch_metrics)
        
        # Update learning rate scheduler if needed
        if self.scheduler is not None and self.config.scheduler_type == "reduce_on_plateau":
            self.scheduler.step(avg_metrics["loss"])
        
        # Record timing
        avg_metrics["epoch_time"] = time.time() - start_time
        
        return avg_metrics
    
    def _compute_loss(self, outputs, targets):
        """Computes the loss using the selected loss function."""
        # Create loss function using factory pattern
        loss_fn = LossFactory.create_loss(
            loss_type=self.config.loss_type,
            mask_threshold=self.config.mask_threshold,
            reduction='mean',
            hotspot_percentile=getattr(self.config, 'hotspot_percentile', 90.0),
            hotspot_weight=getattr(self.config, 'hotspot_weight', 5.0),
            lambda_grad=getattr(self.config, 'gradient_lambda', 0.5),
            weight_factor=getattr(self.config, 'weighted_mse_factor', 2.0),
            alpha=getattr(self.config, 'overlap_alpha', 0.5)
        )
        
        # Apply the loss function
        return loss_fn(outputs, targets)
    
    def _compute_metrics(self, outputs, targets):
        """Computes evaluation metrics. Handles SCALAR targets."""
        # outputs shape: [B, 1, D, H, W] (predicted magnitude)
        # targets shape: [B, 1, D, H, W] (ground truth magnitude)

        target_magnitude = targets # Targets are already the magnitude [B, 1, D, H, W]
        predicted_magnitude = outputs

        # --- Masking (using target magnitude) ---
        mask = (target_magnitude > self.config.mask_threshold) # Boolean mask [B, 1, D, H, W]

        # Check for spatial dimension mismatch and interpolate if necessary
        if predicted_magnitude.shape[2:] != target_magnitude.shape[2:]:
            logger.warning(f"Spatial dimension mismatch between output {predicted_magnitude.shape[2:]} "
                      f"and target {target_magnitude.shape[2:]} in metrics. Interpolating target and mask.")
            try:
                output_size = predicted_magnitude.shape[2:]
                # Interpolate target and mask
                target_magnitude = F.interpolate(target_magnitude, size=output_size, mode='trilinear', align_corners=False)
                mask = F.interpolate(mask.float(), size=output_size, mode='trilinear', align_corners=False) > 0.5 # Re-binarize
            except ValueError as e:
                logger.error(f"Interpolation failed in _compute_metrics: {e}")
                logger.error(f"Predicted shape: {predicted_magnitude.shape}, Target shape: {target_magnitude.shape}")
                # Decide how to handle: return empty metrics, NaNs, or raise error
                return {} # Return empty metrics on interpolation failure

        # Convert to numpy, remove channel dimension for metric functions
        predicted_np = predicted_magnitude.squeeze(1).detach().cpu().numpy() # [B, D, H, W]
        target_np = target_magnitude.squeeze(1).detach().cpu().numpy()       # [B, D, H, W]
        mask_np = mask.squeeze(1).detach().cpu().numpy()                     # [B, D, H, W]
        try:
             # Calculate metrics using the functions that accept scalar inputs and mask
             metrics = calculate_magnitude_metrics(predicted_np, target_np, mask=mask_np)
        except Exception as e:
             logger.error(f"Error calculating metrics: {e}", exc_info=True)
             metrics = {} # Return empty metrics on error

        return metrics
    
    def save_checkpoint(self, path: str, metrics: Dict[str, float] = None):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            metrics: Optional metrics to save with checkpoint
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'config': self.config.__dict__,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
        if self.debug_hook:
            self.debug_hook.record_event("checkpoint_saved", {
                "path": path,
                "epoch": self.current_epoch,
                "metrics": metrics
            })
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
            
        Returns:
            Dict[str, Any]: Checkpoint data
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load other state
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.debug_hook:
            self.debug_hook.record_event("checkpoint_loaded", {
                "path": path,
                "epoch": checkpoint['epoch'],
                "metrics": checkpoint.get('metrics')
            })
        
        return checkpoint
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Generate predictions for input features.
        
        Args:
            features: Input features
            
        Returns:
            torch.Tensor: Predicted outputs
        """
        self.model.eval()
        
        # Ensure features is on the correct device
        features = features.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
        
        return outputs