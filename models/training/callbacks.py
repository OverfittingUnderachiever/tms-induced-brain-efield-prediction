"""
TMS E-field Prediction Training Callbacks

This module provides callback infrastructure for monitoring and intervention during model training.
Callbacks include early stopping, model checkpointing, and performance tracking.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
import torch
import numpy as np


class TrainingCallback(ABC):
    """Base class for training callbacks."""
    
    def on_train_begin(self, trainer: 'ModelTrainer') -> None:
        """Called at the beginning of training.
        
        Args:
            trainer: Model trainer instance
        """
        pass
    
    def on_train_end(self, trainer: 'ModelTrainer') -> None:
        """Called at the end of training.
        
        Args:
            trainer: Model trainer instance
        """
        pass
    
    def on_epoch_begin(self, trainer: 'ModelTrainer', epoch: int) -> None:
        """Called at the beginning of an epoch.
        
        Args:
            trainer: Model trainer instance
            epoch: Current epoch number
        """
        pass
    
    def on_epoch_end(self, trainer: 'ModelTrainer', epoch: int, 
                    train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float]) -> bool:
        """Called at the end of an epoch.
        
        Args:
            trainer: Model trainer instance
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        return False
    
    def on_batch_begin(self, trainer: 'ModelTrainer', batch: int) -> None:
        """Called at the beginning of a batch.
        
        Args:
            trainer: Model trainer instance
            batch: Current batch number
        """
        pass
    
    def on_batch_end(self, trainer: 'ModelTrainer', batch: int, 
                    metrics: Dict[str, float]) -> None:
        """Called at the end of a batch.
        
        Args:
            trainer: Model trainer instance
            batch: Current batch number
            metrics: Batch metrics
        """
        pass


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                monitor: str = 'val_loss'):
        """Initialize early stopping callback.
        
        Args:
            patience: Number of epochs with no improvement after which training will stop
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('inf')  # For loss minimization
        self.wait_count = 0
        self.stopped_epoch = 0
    
    def on_train_begin(self, trainer: 'ModelTrainer') -> None:
        """Reset early stopping state.
        
        Args:
            trainer: Model trainer instance
        """
        self.wait_count = 0
        self.best_value = float('inf')
        self.stopped_epoch = 0
    
    def on_epoch_end(self, trainer: 'ModelTrainer', epoch: int, 
                    train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float]) -> bool:
        """Check if training should stop.
        
        Args:
            trainer: Model trainer instance
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Get current metric value
        current_value = None
        if self.monitor == 'val_loss' and val_metrics.get('loss') is not None:
            current_value = val_metrics['loss']
        elif self.monitor == 'train_loss':
            current_value = train_metrics['loss']
        elif self.monitor in val_metrics:
            current_value = val_metrics[self.monitor]
        elif self.monitor in train_metrics:
            current_value = train_metrics[self.monitor]
        
        # Skip if metric is not available
        if current_value is None:
            return False
        
        # Check for improvement
        if current_value < self.best_value - self.min_delta:
            # Improvement found, reset counter
            self.best_value = current_value
            self.wait_count = 0
        else:
            # No improvement, increment counter
            self.wait_count += 1
            
            # Check if we should stop
            if self.wait_count >= self.patience:
                self.stopped_epoch = epoch
                print(f"Early stopping triggered. Best {self.monitor}: {self.best_value:.6f}")
                return True
        
        return False


class ModelCheckpointCallback(TrainingCallback):
    """Callback to save model checkpoints during training."""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints', 
                save_best_only: bool = True, save_frequency: int = 1,
                monitor: str = 'val_loss', max_models_to_keep: int = 20):
        """Initialize model checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best model
            save_frequency: Frequency of saving checkpoints (in epochs)
            monitor: Metric to monitor for best model
            max_models_to_keep: Maximum number of models to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.monitor = monitor
        self.best_value = float('inf')  # For loss minimization
        self.max_models_to_keep = max_models_to_keep
        self.saved_models = []  # List of (score, path) tuples
    
    def on_train_begin(self, trainer: 'ModelTrainer') -> None:
        """Ensure checkpoint directory exists.
        
        Args:
            trainer: Model trainer instance
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Make model-specific directory
        model_name = trainer.model.__class__.__name__
        self.model_checkpoint_dir = os.path.join(self.checkpoint_dir, model_name)
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
    
    def _manage_model_files(self, current_value: float, new_checkpoint_path: str):
        """Manage model files to keep only the best N models.
        
        Args:
            current_value: Current metric value
            new_checkpoint_path: Path to the new checkpoint
        """
        # Add the new model to our tracking list
        self.saved_models.append((current_value, new_checkpoint_path))
        
        # Sort by score (ascending for loss)
        self.saved_models.sort(key=lambda x: x[0])
        
        # If we exceed the limit, remove the worst models
        while len(self.saved_models) > self.max_models_to_keep:
            worst_score, worst_path = self.saved_models.pop()  # Remove worst (highest loss)
            
            # Delete the file if it exists
            if os.path.exists(worst_path):
                try:
                    os.remove(worst_path)
                except Exception as e:
                    print(f"Warning: Could not delete checkpoint file {worst_path}: {e}")
    
    def on_epoch_end(self, trainer: 'ModelTrainer', epoch: int, 
                    train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float]) -> bool:
        """Save model checkpoint if needed.
        
        Args:
            trainer: Model trainer instance
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            
        Returns:
            bool: Always False (don't stop training)
        """
        # Get current metric value
        current_value = None
        if self.monitor == 'val_loss' and val_metrics.get('loss') is not None:
            current_value = val_metrics['loss']
        elif self.monitor == 'train_loss':
            current_value = train_metrics['loss']
        elif self.monitor in val_metrics:
            current_value = val_metrics[self.monitor]
        elif self.monitor in train_metrics:
            current_value = train_metrics[self.monitor]
        
        # Skip if metric is not available
        if current_value is None:
            return False
        
        # Determine if we should save this model
        should_save = False
        
        # Check if this is the best model ever
        is_best = current_value < self.best_value
        if is_best:
            self.best_value = current_value
            should_save = True
            
            # Always save the best model
            best_path = os.path.join(self.model_checkpoint_dir, 'best_model.pt')
            trainer.save_checkpoint(best_path, {
                'train': train_metrics,
                'val': val_metrics,
                'epoch': epoch,
                'best': True
            })
        
        # Check if we should save regular checkpoint
        if not self.save_best_only and epoch % self.save_frequency == 0:
            # Only save if it's in the top N models
            if len(self.saved_models) < self.max_models_to_keep or \
               current_value < self.saved_models[-1][0]:  # Better than worst saved model
                should_save = True
        
        # Save model at regular intervals if needed
        if should_save and not self.save_best_only and epoch % self.save_frequency == 0:
            checkpoint_path = os.path.join(
                self.model_checkpoint_dir, 
                f'checkpoint_epoch_{epoch}.pt'
            )
            
            # Only save if we have room or if it's better than the worst model
            if len(self.saved_models) < self.max_models_to_keep or \
               current_value < self.saved_models[-1][0]:
                trainer.save_checkpoint(checkpoint_path, {
                    'train': train_metrics,
                    'val': val_metrics,
                    'epoch': epoch,
                    'best': is_best
                })
                
                # Manage saved model files
                self._manage_model_files(current_value, checkpoint_path)
        
        return False


class TensorBoardCallback(TrainingCallback):
    """Callback to log metrics to TensorBoard."""
    
    def __init__(self, log_dir: str = 'logs'):
        """Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        self.log_dir = log_dir
        self.writer = None
    
    def on_train_begin(self, trainer: 'ModelTrainer') -> None:
        """Initialize TensorBoard writer.
        
        Args:
            trainer: Model trainer instance
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            # Create log directory
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Create model-specific log directory
            model_name = trainer.model.__class__.__name__
            model_log_dir = os.path.join(self.log_dir, model_name)
            os.makedirs(model_log_dir, exist_ok=True)
            
            # Create writer
            self.writer = SummaryWriter(log_dir=model_log_dir)
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
    
    def on_epoch_end(self, trainer: 'ModelTrainer', epoch: int, 
                    train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float]) -> bool:
        """Log metrics to TensorBoard.
        
        Args:
            trainer: Model trainer instance
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            
        Returns:
            bool: Always False (don't stop training)
        """
        if self.writer is None:
            return False
        
        # Log training metrics
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)) and key != 'epoch':
                self.writer.add_scalar(f'train/{key}', value, epoch)
        
        # Log validation metrics
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)) and value is not None and key != 'epoch':
                self.writer.add_scalar(f'val/{key}', value, epoch)
        
        # Log learning rate
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            self.writer.add_scalar(f'train/lr_{i}', param_group['lr'], epoch)
        
        return False
    
    def on_train_end(self, trainer: 'ModelTrainer') -> None:
        """Close TensorBoard writer.
        
        Args:
            trainer: Model trainer instance
        """
        if self.writer is not None:
            self.writer.close()


class ResourceMonitorCallback(TrainingCallback):
    """Callback to monitor and log resource usage during training."""
    
    def __init__(self, 
                resource_monitor: 'ResourceMonitor', 
                log_frequency: int = 1,
                memory_limit_percentage: float = 0.9):
        """Initialize resource monitor callback.
        
        Args:
            resource_monitor: Resource monitor instance
            log_frequency: Frequency of logging resource usage (in epochs)
            memory_limit_percentage: Memory limit as percentage of available memory
        """
        self.resource_monitor = resource_monitor
        self.log_frequency = log_frequency
        self.memory_limit_percentage = memory_limit_percentage
        self.epoch_start_time = 0
        self.start_memory = 0
    
    def on_epoch_begin(self, trainer: 'ModelTrainer', epoch: int) -> None:
        """Record epoch start time and memory usage.
        
        Args:
            trainer: Model trainer instance
            epoch: Current epoch number
        """
        self.epoch_start_time = time.time()
        metrics = self.resource_monitor.get_current_metrics()
        self.start_memory = metrics.memory_used
    
    def on_epoch_end(self, trainer: 'ModelTrainer', epoch: int, 
                    train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float]) -> bool:
        """Log resource usage.
        
        Args:
            trainer: Model trainer instance
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            
        Returns:
            bool: True if memory limit is exceeded, False otherwise
        """
        if epoch % self.log_frequency != 0:
            return False
        
        # Get current metrics
        metrics = self.resource_monitor.get_current_metrics()
        
        # Calculate epoch duration
        epoch_duration = time.time() - self.epoch_start_time
        
        # Calculate memory change
        memory_change = metrics.memory_used - self.start_memory
        memory_change_mb = memory_change / (1024 * 1024)
        
        # Log metrics
        print(f"Resource usage - Epoch {epoch}: "
              f"Memory: {metrics.memory_used / (1024*1024*1024):.2f} GB "
              f"({metrics.memory_percentage*100:.1f}%), "
              f"Change: {memory_change_mb:.2f} MB, "
              f"CPU: {metrics.cpu_usage:.1f}%, "
              f"Time: {epoch_duration:.2f}s")
        
        # Check if memory limit is exceeded
        if metrics.memory_percentage > self.memory_limit_percentage:
            print(f"Memory limit exceeded: {metrics.memory_percentage*100:.1f}% > "
                  f"{self.memory_limit_percentage*100:.1f}%")
            
            # Trigger memory reduction
            self.resource_monitor.trigger_memory_reduction(target_percentage=0.7)
            
            # Only stop training if we're critically low
            if metrics.memory_percentage > 0.95:
                print("Critical memory shortage. Stopping training.")
                return True
        
        return False


class ProgressCallback(TrainingCallback):
    """Callback to display training progress."""
    
    def __init__(self, print_frequency: int = 1):
        """Initialize progress callback.
        
        Args:
            print_frequency: Frequency of progress updates (in epochs)
        """
        self.print_frequency = print_frequency
        self.start_time = 0
    
    def on_train_begin(self, trainer: 'ModelTrainer') -> None:
        """Record training start time.
        
        Args:
            trainer: Model trainer instance
        """
        self.start_time = time.time()
        print(f"Starting training for {trainer.config.epochs} epochs")
    
    def on_epoch_end(self, trainer: 'ModelTrainer', epoch: int, 
                    train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float]) -> bool:
        """Print training progress.
        
        Args:
            trainer: Model trainer instance
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            
        Returns:
            bool: Always False (don't stop training)
        """
        if epoch % self.print_frequency != 0:
            return False
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        remaining = elapsed / (epoch + 1) * (trainer.config.epochs - epoch - 1)
        
        # Format metrics
        train_loss = f"{train_metrics['loss']:.6f}"
        val_loss = f"{val_metrics['loss']:.6f}" if val_metrics.get('loss') is not None else "N/A"
        
        # Print progress
        print(f"Epoch {epoch}/{trainer.config.epochs} - "
              f"Train Loss: {train_loss}, Val Loss: {val_loss}, "
              f"Elapsed: {elapsed/60:.1f}m, Remaining: {remaining/60:.1f}m")
        
        # Print additional metrics if available
        additional_metrics = []
        for key, value in val_metrics.items():
            if key != 'loss' and value is not None:
                additional_metrics.append(f"{key}: {value:.4f}")
        
        if additional_metrics:
            print(f"Metrics: {', '.join(additional_metrics)}")
        
        return False
    
    def on_train_end(self, trainer: 'ModelTrainer') -> None:
        """Print training summary.
        
        Args:
            trainer: Model trainer instance
        """
        elapsed = time.time() - self.start_time
        print(f"Training completed in {elapsed/60:.1f} minutes. "
              f"Best validation loss: {trainer.best_val_loss:.6f}")