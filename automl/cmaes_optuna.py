"""
TMS E-field Prediction CMA-ES Integration with Optuna

This module provides CMA-ES optimization for TMS E-field prediction models
through Ray Tune's Optuna integration.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Try to import Optuna (required for CMA-ES)
try:
    import optuna
    from ray.tune.search.optuna import OptunaSearch
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.getLogger('tms_automl').warning(
        "Optuna not available. Cannot use CMA-ES. Install with: pip install optuna")

from ..utils.resource.monitor import ResourceMonitor
from ..utils.state.context import ModelContext
from ..experiments.tracking import ExperimentTracker
from .integration.tune_wrapper import TuneWrapper, AutoMLConfig, create_search_space

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tms_automl')


def run_cmaes_optimization(
    output_dir: str = "tune_results",
    num_samples: int = 10,
    max_concurrent_trials: int = 2,
    architecture_name: str = "simple_unet_magnitude",
    model_factory=None,
    data_loader=None,
    sigma0: float = 0.5,
    population_size: Optional[int] = None,
    param_space: Optional[Dict[str, Dict[str, Any]]] = None,
    base_config: Optional[Dict[str, Any]] = None
):
    """Run CMA-ES Optimization for TMS E-field prediction model.
    
    Args:
        output_dir: Directory to save results
        num_samples: Number of trials to run
        max_concurrent_trials: Maximum number of concurrent trials
        architecture_name: Name of the architecture to optimize
        model_factory: Function to create model from configuration
        data_loader: Function to load data from configuration
        sigma0: Initial standard deviation for CMA-ES
        population_size: Population size for CMA-ES (None for auto)
        param_space: Parameter space definition
        base_config: Base configuration with default values
        
    Returns:
        ray.tune.ExperimentAnalysis: Analysis object with results
    """
    # Check if Optuna is available
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna is not available. Cannot use CMA-ES.")
        logger.error("Install Optuna with: pip install optuna")
        return None
    
    # Create resource monitor
    resource_monitor = ResourceMonitor(max_memory_gb=8, check_interval=10.0)
    resource_monitor.start_monitoring()
    
    # Create experiment tracker
    tracker = ExperimentTracker(
        experiment_dir="experiments",
        architecture_name=architecture_name,
        create_subdirs=True
    )
    tracker.set_description(f"CMA-ES Optimization for {architecture_name}")
    
    # Set default base configuration and parameter space if not provided
    if base_config is None:
        base_config = {
            'batch_size': 4,
            'epochs': 20,
            'input_channels': 4,
            'output_channels': 1,
            'optimizer_type': 'adamw',
            'scheduler_type': 'reduce_on_plateau',
        }
    
    if param_space is None:
        param_space = {
            'learning_rate': {
                'type': 'float',
                'min': 1e-4,
                'max': 1e-2
            },
            'feature_maps': {
                'type': 'int',
                'min': 16,
                'max': 64
            },
            'levels': {
                'type': 'int',
                'min': 3,
                'max': 5
            },
            'dropout_rate': {
                'type': 'float',
                'min': 0.1,
                'max': 0.5
            }
        }
    
    # Create search space
    search_space = create_search_space(base_config, param_space)
    
    # Configure AutoML
    automl_config = AutoMLConfig(
        cpu_per_trial=1,
        gpu_per_trial=0.5 if torch.cuda.is_available() else 0,
        memory_per_trial_gb=4,
        search_space=search_space,
        search_algorithm="cmaes",  # Use CMA-ES
        scheduler="asha",
        max_concurrent_trials=max_concurrent_trials,
        max_resource_attr="training_iteration",
        max_resource_value=20,
        num_samples=num_samples,
        metric="val_loss",
        mode="min",
        checkpoint_frequency=5,
        output_dir=output_dir,
        architecture_name=architecture_name,
        experiment_tracker=tracker,
        debug_mode=True,
        cmaes_sigma0=sigma0,
        cmaes_population_size=population_size
    )
    
    # Create TuneWrapper
    tune_wrapper = TuneWrapper(
        config=automl_config,
        model_factory=model_factory,
        data_loader=data_loader,
        resource_monitor=resource_monitor
    )
    
    # Run trials
    analysis = tune_wrapper.run_trials()
    
    # Stop resource monitor
    resource_monitor.stop_monitoring()
    
    return analysis


if __name__ == "__main__":
    # Simple example function that creates a dummy model
    def create_dummy_model(config):
        # This would be your actual model creation function
        return torch.nn.Sequential(
            torch.nn.Conv3d(config.get('input_channels', 4), 
                           config.get('feature_maps', 32), 
                           kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(config.get('feature_maps', 32), 
                           config.get('output_channels', 1), 
                           kernel_size=3, padding=1)
        )
    
    # Simple example function that returns dummy data loaders
    def load_dummy_data(config):
        batch_size = config.get('batch_size', 4)
        # Create random data
        train_features = torch.randn(20, 4, 25, 25, 25)
        train_targets = torch.randn(20, 1, 25, 25, 25)
        val_features = torch.randn(5, 4, 25, 25, 25)
        val_targets = torch.randn(5, 1, 25, 25, 25)
        
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(train_features, train_targets)
        val_dataset = TensorDataset(val_features, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    # Run a simple example
    analysis = run_cmaes_optimization(
        output_dir="tune_results_cmaes",
        num_samples=5,  # Small number for testing
        max_concurrent_trials=1,
        architecture_name="simple_unet_magnitude",
        model_factory=create_dummy_model,
        data_loader=load_dummy_data,
        sigma0=0.5,
        population_size=None  # Auto-determine based on dimensions
    )
    
    if analysis:
        print("Best trial config:", analysis.best_config)
        print("Best trial result:", analysis.best_result)