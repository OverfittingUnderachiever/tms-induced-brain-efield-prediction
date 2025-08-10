"""
Unit tests for TMS E-field prediction training components.
"""

import os
import shutil
import pytest
import tempfile
import numpy as np
import torch
from torch import nn
import torch.optim as optim

from tms_efield_prediction.models.training.trainer import ModelTrainer, TrainerConfig, TMSDataset
from tms_efield_prediction.models.training.callbacks import (
    TrainingCallback, EarlyStoppingCallback, ModelCheckpointCallback, ProgressCallback
)
from tms_efield_prediction.models.training.schedulers import get_scheduler, WarmupLRScheduler
from tms_efield_prediction.models.evaluation.metrics import calculate_metrics
from tms_efield_prediction.utils.state.context import ModelContext
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor


# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, output_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Handle both channel-first and channel-last formats
        if x.ndim == 5 and x.shape[1] == 4:  # [B, C, D, H, W]
            pass  # Already in the correct format
        elif x.ndim == 5 and x.shape[4] == 4:  # [B, D, H, W, C]
            x = x.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        elif x.ndim == 4 and x.shape[3] == 4:  # [D, H, W, C]
            x = x.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, D, H, W]
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x


# Fixture for temporary directory
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


# Fixture for model and config
@pytest.fixture
def model_and_config():
    """Create a model and configuration for testing."""
    model = SimpleModel()
    config = TrainerConfig(
        batch_size=2,
        epochs=2,
        learning_rate=0.001,
        device="cpu",  # Use CPU for testing
        mixed_precision=False,  # No mixed precision on CPU
        checkpoint_dir="test_checkpoints"
    )
    yield model, config


# Fixture for sample data
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample features and targets
    batch_size = 4
    features = []
    targets = []
    
    for _ in range(batch_size):
        # Random MRI-like data [25, 25, 25, 4]
        feature = torch.randn(25, 25, 25, 4)
        features.append(feature)
        
        # Random E-field data [3, 25, 25, 25]
        target = torch.randn(3, 25, 25, 25)
        targets.append(target)
    
    yield features, targets


# Test TrainerConfig validation
def test_trainer_config_validation():
    """Test TrainerConfig validation."""
    # Valid config
    config = TrainerConfig(batch_size=8, epochs=10, learning_rate=0.001)
    assert config.validate() is True
    
    # Invalid batch size
    config = TrainerConfig(batch_size=0, epochs=10, learning_rate=0.001)
    assert config.validate() is False
    
    # Invalid epochs
    config = TrainerConfig(batch_size=8, epochs=0, learning_rate=0.001)
    assert config.validate() is False
    
    # Invalid learning rate
    config = TrainerConfig(batch_size=8, epochs=10, learning_rate=0)
    assert config.validate() is False
    
    # Device setting on CPU-only machine
    config = TrainerConfig(batch_size=8, epochs=10, learning_rate=0.001, device="cuda")
    if not torch.cuda.is_available():
        config.validate()
        assert config.device == "cpu"
        assert config.mixed_precision is False


# Test TMSDataset
def test_tms_dataset(sample_data):
    """Test TMSDataset functionality."""
    features, targets = sample_data
    
    # Create dataset
    dataset = TMSDataset(features, targets)
    
    # Test length
    assert len(dataset) == len(features)
    
    # Test getitem
    feature, target = dataset[0]
    assert feature.shape == (25, 25, 25, 4)
    assert target.shape == (3, 25, 25, 25)
    
    # Test validation
    with pytest.raises(ValueError):
        # Mismatched lengths
        TMSDataset(features[:-1], targets)


# Test model initialization
def test_model_trainer_init(model_and_config, temp_dir):
    """Test ModelTrainer initialization."""
    model, config = model_and_config
    model_context = ModelContext(dependencies={}, config={})
    
    # Set checkpoint directory to temp_dir
    config.checkpoint_dir = temp_dir
    
    # Initialize trainer
    trainer = ModelTrainer(model, config, model_context)
    
    # Test initialization
    assert trainer.model is model
    assert trainer.config is config
    assert trainer.device == torch.device("cpu")
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)
    assert len(trainer.callbacks) > 0
    
    # Test optimizer creation
    assert isinstance(trainer._create_optimizer(), torch.optim.Optimizer)


# Test training loop
def test_training_loop(model_and_config, sample_data, temp_dir):
    """Test training loop execution."""
    model, config = model_and_config
    features, targets = sample_data
    model_context = ModelContext(dependencies={}, config={})
    
    # Set checkpoint directory to temp_dir
    config.checkpoint_dir = temp_dir
    config.epochs = 1  # Short training for test
    
    # Initialize trainer
    trainer = ModelTrainer(model, config, model_context)
    
    # Run training
    history = trainer.train(features, targets)
    
    # Check history
    assert "train" in history
    assert "val" in history
    assert len(history["train"]) == config.epochs
    assert len(history["val"]) == config.epochs
    
    # Check metrics
    assert "loss" in history["train"][0]
    
    # Check model state
    assert trainer.current_epoch == config.epochs - 1
    
    # Check predictions
    pred = trainer.predict(features[0])
    assert pred.shape[1:] == (3, 25, 25, 25)  # [B, 3, 25, 25, 25]


# Test callbacks
def test_callbacks(model_and_config, sample_data, temp_dir):
    """Test callback functionality."""
    model, config = model_and_config
    features, targets = sample_data
    model_context = ModelContext(dependencies={}, config={})
    
    # Set checkpoint directory to temp_dir
    config.checkpoint_dir = temp_dir
    config.epochs = 2  # Short training for test
    
    # Record calls to callback methods
    class TestCallback(TrainingCallback):
        def __init__(self):
            self.calls = {
                "on_train_begin": 0,
                "on_train_end": 0,
                "on_epoch_begin": 0,
                "on_epoch_end": 0,
                "on_batch_begin": 0,
                "on_batch_end": 0
            }
        
        def on_train_begin(self, trainer):
            self.calls["on_train_begin"] += 1
        
        def on_train_end(self, trainer):
            self.calls["on_train_end"] += 1
        
        def on_epoch_begin(self, trainer, epoch):
            self.calls["on_epoch_begin"] += 1
        
        def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
            self.calls["on_epoch_end"] += 1
            return False
        
        def on_batch_begin(self, trainer, batch):
            self.calls["on_batch_begin"] += 1
        
        def on_batch_end(self, trainer, batch, metrics):
            self.calls["on_batch_end"] += 1
    
    # Create test callback
    test_callback = TestCallback()
    
    # Initialize trainer
    trainer = ModelTrainer(model, config, model_context)
    trainer.callbacks = [test_callback]  # Replace with test callback
    
    # Run training
    trainer.train(features, targets)
    
    # Check callback calls
    assert test_callback.calls["on_train_begin"] == 1
    assert test_callback.calls["on_train_end"] == 1
    assert test_callback.calls["on_epoch_begin"] == config.epochs
    assert test_callback.calls["on_epoch_end"] == config.epochs
    assert test_callback.calls["on_batch_begin"] > 0
    assert test_callback.calls["on_batch_end"] > 0


# Test early stopping
def test_early_stopping(model_and_config, sample_data, temp_dir):
    """Test early stopping callback."""
    model, config = model_and_config
    features, targets = sample_data
    model_context = ModelContext(dependencies={}, config={})
    
    # Set checkpoint directory to temp_dir
    config.checkpoint_dir = temp_dir
    config.epochs = 10  # Long training to test early stopping
    
    # Create early stopping callback with strict settings
    early_stopping = EarlyStoppingCallback(patience=1, min_delta=0.0)
    
    # Initialize trainer
    trainer = ModelTrainer(model, config, model_context)
    trainer.callbacks = [early_stopping]  # Replace with early stopping only
    
    # Run training
    history = trainer.train(features, targets)
    
    # Check that training stopped early
    assert len(history["train"]) < config.epochs


# Test model checkpointing
def test_model_checkpointing(model_and_config, sample_data, temp_dir):
    """Test model checkpoint callback."""
    model, config = model_and_config
    features, targets = sample_data
    model_context = ModelContext(dependencies={}, config={})
    
    # Set checkpoint directory to temp_dir
    config.checkpoint_dir = temp_dir
    config.epochs = 2  # Short training for test
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpointCallback(
        checkpoint_dir=temp_dir,
        save_best_only=False,
        save_frequency=1
    )
    
    # Initialize trainer
    trainer = ModelTrainer(model, config, model_context)
    trainer.callbacks = [checkpoint_callback]  # Replace with checkpoint only
    
    # Run training
    trainer.train(features, targets)
    
    # Check if checkpoints were saved
    model_dir = os.path.join(temp_dir, model.__class__.__name__)
    assert os.path.exists(model_dir)
    assert any(f.startswith("checkpoint_epoch_") for f in os.listdir(model_dir))
    
    # Test loading checkpoint
    checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith("checkpoint_epoch_")]
    if checkpoint_files:
        checkpoint_path = os.path.join(model_dir, checkpoint_files[0])
        checkpoint = trainer.load_checkpoint(checkpoint_path)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "epoch" in checkpoint


# Test schedulers
def test_schedulers():
    """Test learning rate schedulers."""
    # Create model and optimizer
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Test various scheduler types
    scheduler_types = [
        "reduce_on_plateau",
        "cosine",
        "step",
        "exponential",
        "cyclic",
        "one_cycle",
        "none"
    ]
    
    for scheduler_type in scheduler_types:
        # Create config with scheduler type
        config = TrainerConfig(
            scheduler_type=scheduler_type,
            epochs=10
        )
        
        # Get scheduler
        scheduler = get_scheduler(optimizer, config)
        
        # Check scheduler type
        if scheduler_type.lower() == "none":
            assert scheduler is None
        else:
            assert scheduler is not None
            
            # Test step method
            if scheduler_type.lower() == "reduce_on_plateau":
                scheduler.step(0.1)
            else:
                scheduler.step()


# Test WarmupLRScheduler
def test_warmup_scheduler():
    """Test WarmupLRScheduler."""
    # Create model and optimizer
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create base scheduler
    base_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Create warmup scheduler
    warmup_scheduler = WarmupLRScheduler(
        optimizer=optimizer,
        scheduler=base_scheduler,
        warmup_epochs=2,
        warmup_method="linear",
        warmup_factor=0.1
    )
    
    # Test warmup phase
    for epoch in range(2):
        warmup_scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            # In warmup phase, LR should be less than base LR for epoch 0
            if epoch == 0:
                assert param_group["lr"] < 0.001
    
    # Test regular phase
    # Step through epochs 2-7
    for epoch in range(2, 8):
        warmup_scheduler.step(epoch)
        
        # After epoch 5 (3 epochs after warmup), LR should decrease
        if epoch >= 5:
            for param_group in optimizer.param_groups:
                # Should be reduced by gamma (0.1)
                assert param_group["lr"] < 0.0005, f"LR should be reduced at epoch {epoch}, but was {param_group['lr']}"


# Test metrics calculation
def test_metrics_calculation():
    """Test metrics calculation."""
    # Create sample predictions and targets
    pred = np.random.randn(2, 3, 10, 10, 10)  # [B, C, D, H, W]
    target = np.random.randn(2, 3, 10, 10, 10)  # [B, C, D, H, W]
    
    # Calculate metrics
    metrics = calculate_metrics(pred, target)
    
    # Check metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "rel_error" in metrics
    assert "mean_angular_error" in metrics
    assert "median_angular_error" in metrics
    assert "p95_angular_error" in metrics
    assert "correlation" in metrics
    
    # Check metric values
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["rel_error"] >= 0
    assert 0 <= metrics["correlation"] <= 1 or -1 <= metrics["correlation"] <= 0


# Test resource monitor integration
def test_resource_monitor_integration(model_and_config, sample_data, temp_dir):
    """Test resource monitor integration."""
    model, config = model_and_config
    features, targets = sample_data
    model_context = ModelContext(dependencies={}, config={})
    
    # Set checkpoint directory to temp_dir
    config.checkpoint_dir = temp_dir
    config.epochs = 1  # Short training for test
    
    # Create resource monitor
    resource_monitor = ResourceMonitor(
        max_memory_gb=1,  # Small limit for testing
        check_interval=0.1
    )
    
    # Initialize trainer with resource monitor
    trainer = ModelTrainer(
        model, config, model_context, 
        resource_monitor=resource_monitor
    )
    
    # Check registration
    assert "model_trainer" in resource_monitor.components
    
    # Start monitoring
    resource_monitor.start_monitoring()
    
    try:
        # Run training
        trainer.train(features, targets)
        
        # Test memory reduction
        trainer._reduce_memory(0.3)
        
        # Trigger memory reduction
        resource_monitor.trigger_memory_reduction(0.5)
    finally:
        # Stop monitoring
        resource_monitor.stop_monitoring()


# Test exception handling
def test_exception_handling(model_and_config, temp_dir):
    """Test exception handling in training loop."""
    model, config = model_and_config
    model_context = ModelContext(dependencies={}, config={})
    
    # Set checkpoint directory to temp_dir
    config.checkpoint_dir = temp_dir
    
    # Initialize trainer
    trainer = ModelTrainer(model, config, model_context)
    
    # Test with invalid data
    with pytest.raises(ValueError):
        trainer.train([], [])  # Empty data
    
    with pytest.raises(ValueError):
        trainer.train([torch.randn(10, 10, 10, 10)], [])  # Mismatched lengths


# Test compute_loss function
def test_compute_loss(model_and_config):
    """Test loss computation."""
    model, config = model_and_config
    model_context = ModelContext(dependencies={}, config={})
    
    # Initialize trainer
    trainer = ModelTrainer(model, config, model_context)
    
    # Test E-field loss
    outputs = torch.randn(2, 3, 25, 25, 25)  # [B, C, D, H, W]
    targets = torch.randn(2, 3, 25, 25, 25)  # [B, C, D, H, W]
    
    loss = trainer._compute_loss(outputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss.item() >= 0
    
    # Test with different loss type
    config.loss_type = "mse"
    trainer = ModelTrainer(model, config, model_context)
    loss = trainer._compute_loss(outputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


# Test validation loop
def test_validation_loop(model_and_config, sample_data, temp_dir):
    """Test validation loop."""
    model, config = model_and_config
    features, targets = sample_data
    model_context = ModelContext(dependencies={}, config={})
    
    # Set checkpoint directory to temp_dir
    config.checkpoint_dir = temp_dir
    
    # Initialize trainer
    trainer = ModelTrainer(model, config, model_context)
    
    # Create dataset and dataloader
    dataset = TMSDataset(features, targets)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False
    )
    
    # Run validation
    val_metrics = trainer._validate_epoch(dataloader)
    
    # Check metrics
    assert "loss" in val_metrics
    assert val_metrics["loss"] >= 0


# Test batch processing
def test_batch_processing(model_and_config, sample_data, temp_dir):
    """Test batch processing."""
    model, config = model_and_config
    features, targets = sample_data
    model_context = ModelContext(dependencies={}, config={})
    
    # Set checkpoint directory to temp_dir
    config.checkpoint_dir = temp_dir
    
    # Configure gradient accumulation
    config.gradient_accumulation_steps = 2
    
    # Initialize trainer
    trainer = ModelTrainer(model, config, model_context)
    
    # Create dataset and dataloader
    dataset = TMSDataset(features, targets)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False
    )
    
    # Run training epoch
    train_metrics = trainer._train_epoch(dataloader)
    
    # Check metrics
    assert "loss" in train_metrics
    assert train_metrics["loss"] >= 0


# Run tests with pytest -xvs tests/unit/models/test_training.py
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])