"""
TMS E-field Prediction AutoML Integration

This module provides integration with AutoML frameworks for TMS E-field prediction models.
It implements wrappers for Ray Tune and other AutoML functions.
"""

import os
import json
import time
import pickle
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from functools import partial
import gc
from sklearn.model_selection import KFold

from ...utils.resource.monitor import ResourceMonitor
from ...utils.state.context import ModelContext
from ...models.evaluation.metrics import calculate_metrics
from ...experiments.tracking import ExperimentTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tms_automl')

# Try to import Ray Tune (optional dependency)
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, MedianStoppingRule
    from ray.tune.search.basic_variant import BasicVariantGenerator
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.bayesopt import BayesOptSearch
    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray Tune not available. Install with: pip install ray[tune]")
    RAY_AVAILABLE = False



@dataclass
class AutoMLConfig:
    """Configuration for AutoML."""
    
    # Trial resources
    cpu_per_trial: int = 1
    gpu_per_trial: float = 0.5
    memory_per_trial_gb: int = 8
    
    # Search space specification
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # Search algorithm
    search_algorithm: str = "bayesopt"  # bayesopt, random, grid, cmaes
    
    # Trial scheduling
    scheduler: str = "asha"  # asha, median, fifo
    max_concurrent_trials: int = 2
    
    # Resource allocation
    max_resource_attr: str = "training_iteration"
    max_resource_value: int = 40
    
    # Number of trials
    num_samples: int = 20
    
    # Performance metrics
    metric: str = "val_loss"
    mode: str = "min"  # min, max
    
    # Checkpointing
    checkpoint_frequency: int = 5
    checkpoint_keep_attr: List[str] = field(default_factory=lambda: ["best_model", "best_metrics"])
    
    # Early stopping
    grace_period: int = 5
    reduction_factor: float = 3
    
    # Output directory
    output_dir: str = "tune_results"
    
    # Architecture to optimize
    architecture_name: str = "dual_modal"
    
    # Experiment tracking
    track_experiments: bool = True
    experiment_tracker: Optional[ExperimentTracker] = None
    
    # Debug mode
    debug_mode: bool = False
    
    # CMA-ES specific parameters
    cmaes_sigma0: float = 0.5  # Initial standard deviation
    cmaes_population_size: Optional[int] = None  # Population size for CMA-ES, None for auto

    def validate(self) -> bool:
        """Validate the configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Check if Ray is available
        if not RAY_AVAILABLE:
            logger.error("Ray Tune is not available. AutoML features will not work.")
            return False
        
        # Check required fields
        if not self.search_space:
            logger.error("Search space must be defined.")
            return False
        
        # Check resource allocation
        if self.gpu_per_trial > 0 and not torch.cuda.is_available():
            logger.warning("CUDA not available but GPU requested. Setting gpu_per_trial to 0.")
            self.gpu_per_trial = 0
        
        # Check Optuna availability for CMA-ES
        if self.search_algorithm == "cmaes":
            try:
                import optuna
            except ImportError:
                logger.error("Optuna is not available. Cannot use CMA-ES.")
                logger.error("Install Optuna with: pip install optuna")
                return False
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        return True


class TuneWrapper:
    """Wrapper for Ray Tune integration."""
    
    def __init__(self, config: AutoMLConfig, model_factory: Callable, 
                data_loader: Callable, resource_monitor: Optional[ResourceMonitor] = None):
        """Initialize Ray Tune wrapper.
        
        Args:
            config: AutoML configuration
            model_factory: Function to create models from hyperparameters
            data_loader: Function to load data
            resource_monitor: Optional resource monitor
        """
        self.config = config
        self.model_factory = model_factory
        self.data_loader = data_loader
        self.resource_monitor = resource_monitor
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid AutoML configuration")
        
        # Initialize experiment tracker if tracking is enabled
        if self.config.track_experiments and self.config.experiment_tracker is None:
            self.config.experiment_tracker = ExperimentTracker(
                experiment_dir="experiments",
                architecture_name=self.config.architecture_name,
                debug_mode=self.config.debug_mode
            )
            self.config.experiment_tracker.set_description(f"AutoML tuning for {self.config.architecture_name}")
        
        logger.info(f"Initialized TuneWrapper for {self.config.architecture_name}")
    
    def run_trials(self):
        """Run AutoML trials.
        
        Returns:
            ray.tune.ExperimentAnalysis: Analysis object with trial results
        """
        if not RAY_AVAILABLE:
            logger.error("Ray Tune is not available. Cannot run trials.")
            return None
        
        # Set up Ray if it's not initialized
        if not ray.is_initialized():
            ray.init(
                log_to_driver=self.config.debug_mode,
                num_cpus=max(4, self.config.cpu_per_trial * self.config.max_concurrent_trials),
                num_gpus=max(1, self.config.gpu_per_trial * self.config.max_concurrent_trials)
            )
        
        # Create training function
        train_fn = self._create_training_function()
        
        # Configure search algorithm
        search_alg = self._configure_search_algorithm()
        
        # Configure scheduler
        scheduler = self._configure_scheduler()
        
        # Set up experiment name
        experiment_name = f"{self.config.architecture_name}_automl_{int(time.time())}"
        
        # Set up storage directory
        storage_dir = os.path.join(self.config.output_dir, experiment_name)
        
        # Log experiment setup
        logger.info(f"Starting AutoML experiment: {experiment_name}")
        logger.info(f"Search space: {self.config.search_space}")
        logger.info(f"Metric: {self.config.metric} ({self.config.mode})")
        logger.info(f"Trials: {self.config.num_samples}, Concurrent: {self.config.max_concurrent_trials}")
        
        # Set up resources per trial
        resources_per_trial = {
            "cpu": self.config.cpu_per_trial,
            "gpu": self.config.gpu_per_trial
        }
        
        # Run trials
        analysis = tune.run(
            train_fn,
            config=self.config.search_space,
            num_samples=self.config.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial=resources_per_trial,
            local_dir=storage_dir,
            checkpoint_freq=self.config.checkpoint_frequency,
            checkpoint_at_end=True,
            keep_checkpoints_num=1,
            checkpoint_score_attr=f"{self.config.mode}-{self.config.metric}",
            name=experiment_name,
            max_concurrent_trials=self.config.max_concurrent_trials,
            verbose=1 if self.config.debug_mode else 0,
            metric=self.config.metric,
            mode=self.config.mode
        )
        
        # Save results
        self._save_trial_results(analysis, experiment_name)
        
        # Log best trial
        best_trial = analysis.best_trial
        best_config = analysis.best_config
        best_result = analysis.best_result
        
        logger.info(f"Best trial: {best_trial.trial_id}")
        logger.info(f"Best config: {best_config}")
        logger.info(f"Best result: {best_result}")
        
        # Track best model if enabled
        if self.config.track_experiments and self.config.experiment_tracker:
            # Log hyperparameters
            self.config.experiment_tracker.log_hyperparameters(best_config)
            
            # Log metrics
            self.config.experiment_tracker.log_metrics(
                {k: v for k, v in best_result.items() if isinstance(v, (int, float))}
            )
            
            # Save best trial as artifact
            best_trial_path = os.path.join(storage_dir, "best_trial.json")
            with open(best_trial_path, 'w') as f:
                json.dump({
                    "trial_id": best_trial.trial_id,
                    "config": best_config,
                    "result": best_result
                }, f, indent=4)
            
            self.config.experiment_tracker.log_artifact(
                name="best_trial",
                artifact_path=best_trial_path,
                metadata={
                    "experiment_name": experiment_name,
                    "metric": self.config.metric,
                    "value": best_result.get(self.config.metric)
                }
            )
            
            # Finalize experiment
            self.config.experiment_tracker.finalize()
        
        return analysis
    
    def _create_training_function(self):
        """Create the training function for Ray Tune.
        
        Returns:
            Callable: Training function
        """
        # Get class references to access in the training function
        model_factory = self.model_factory
        data_loader = self.data_loader
        experiment_tracker = self.config.experiment_tracker
        resource_monitor = self.resource_monitor
        architecture_name = self.config.architecture_name
        
        # Define the training function
        def train_model(config):
            """Training function for Ray Tune.
            
            Args:
                config: Hyperparameter configuration
            """
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create model context
            model_context = ModelContext(
                dependencies={},
                config=config,
                debug_mode=False
            )
            
            # Load data
            train_data, val_data = data_loader(config)
            
            # Track trial
            trial_tracker = None
            if experiment_tracker:
                # Create trial-specific tracker
                trial_id = tune.get_trial_id()
                trial_tracker = ExperimentTracker(
                    experiment_dir=os.path.join("experiments", architecture_name, "trials"),
                    architecture_name=f"{architecture_name}_trial_{trial_id}",
                    create_subdirs=True,
                    debug_mode=False
                )
                trial_tracker.log_hyperparameters(config)
            
            # Create model
            model = model_factory(config)
            model.to(device)
            
            # Register with resource monitor if available
            if resource_monitor:
                resource_monitor.register_component(
                    f"tune_trial_{tune.get_trial_id()}", 
                    self._reduce_memory,
                    priority=1
                )
            
            # Create optimizer
            lr = config.get("learning_rate", 0.001)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Training loop
            best_val_loss = float('inf')
            best_model_state = None
            best_metrics = None
            
            for epoch in range(config.get("epochs", 50)):
                # Training phase
                model.train()
                running_loss = 0.0
                train_metrics = {}
                
                for i, (features, targets) in enumerate(train_data):
                    # Move data to device
                    features = features.to(device)
                    targets = targets.to(device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(features)
                    
                    # Calculate loss
                    loss = self._compute_loss(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    optimizer.step()
                    
                    # Update running loss
                    running_loss += loss.item()
                
                # Calculate epoch loss
                epoch_loss = running_loss / len(train_data)
                train_metrics["train_loss"] = epoch_loss
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for features, targets in val_data:
                        # Move data to device
                        features = features.to(device)
                        targets = targets.to(device)
                        
                        # Forward pass
                        outputs = model(features)
                        
                        # Calculate loss
                        loss = self._compute_loss(outputs, targets)
                        
                        # Update validation loss
                        val_loss += loss.item()
                        
                        # Store predictions and targets for metrics
                        all_preds.append(outputs.cpu())
                        all_targets.append(targets.cpu())
                
                # Calculate validation loss
                val_loss = val_loss / len(val_data)
                
                # Calculate metrics
                val_metrics = {}
                val_metrics["val_loss"] = val_loss
                
                # Combine predictions and targets
                all_preds = torch.cat(all_preds, dim=0)
                all_targets = torch.cat(all_targets, dim=0)
                
                # Calculate additional metrics
                detailed_metrics = calculate_metrics(all_preds.numpy(), all_targets.numpy())
                for key, value in detailed_metrics.items():
                    val_metrics[f"val_{key}"] = value
                
                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    best_metrics = val_metrics
                
                # Log metrics to Trial Tracker
                if trial_tracker:
                    combined_metrics = {**train_metrics, **val_metrics}
                    trial_tracker.log_metrics(combined_metrics, epoch)
                
                # Report metrics to Ray Tune
                metrics_dict = {
                    "epoch": epoch,
                    "training_iteration": epoch,  # Required for ASHA scheduler
                    **train_metrics,
                    **val_metrics
                }
                
                # Save best model state for checkpointing
                metrics_dict["best_model"] = best_model_state
                metrics_dict["best_metrics"] = best_metrics
                
                # Report to Ray Tune
                tune.report(**metrics_dict)
                
                # Clean up memory
                del all_preds
                del all_targets
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            
            # Unregister from resource monitor
            if resource_monitor:
                resource_monitor.unregister_component(f"tune_trial_{tune.get_trial_id()}")
            
            # Finalize trial tracker
            if trial_tracker:
                trial_tracker.finalize()
        
        return train_model
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for E-field prediction.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            torch.Tensor: Loss value
        """
        # Split loss into magnitude and direction components
        magnitude_weight = 0.7
        direction_weight = 0.3
        
        # Calculate magnitude loss
        magnitude_outputs = torch.sqrt(torch.sum(outputs**2, dim=1))
        magnitude_targets = torch.sqrt(torch.sum(targets**2, dim=1))
        magnitude_loss = torch.nn.functional.mse_loss(magnitude_outputs, magnitude_targets)
        
        # Calculate direction loss
        # Normalize vectors to get direction
        outputs_normalized = outputs / (magnitude_outputs.unsqueeze(1) + 1e-8)
        targets_normalized = targets / (magnitude_targets.unsqueeze(1) + 1e-8)
        
        # Cosine similarity (1 - cosine_similarity for loss)
        direction_loss = 1 - torch.mean((outputs_normalized * targets_normalized).sum(dim=1))
        
        # Combine losses
        combined_loss = magnitude_weight * magnitude_loss + direction_weight * direction_loss
        
        return combined_loss
    
    def _configure_search_algorithm(self):
        """Configure search algorithm based on configuration.
        
        Returns:
            ray.tune.search.SearchAlgorithm: Search algorithm
        """
        if not RAY_AVAILABLE:
            return None
        
        if self.config.search_algorithm == "bayesopt":
            # Bayesian Optimization
            search_alg = BayesOptSearch(
                metric=self.config.metric,
                mode=self.config.mode,
                utility_kwargs={
                    "kind": "ucb",
                    "kappa": 2.5,
                    "xi": 0.0
                }
            )
        elif self.config.search_algorithm == "grid":
            # Grid Search
            search_alg = BasicVariantGenerator(random_state=42)
        elif self.config.search_algorithm == "cmaes":
            # CMA-ES using Optuna integration
            try:
                import optuna
                from ray.tune.search.optuna import OptunaSearch
                
                optuna_sampler = optuna.samplers.CmaEsSampler(
                    sigma0=self.config.cmaes_sigma0,
                    population_size=self.config.cmaes_population_size,
                    seed=42,
                    n_startup_trials=min(3, self.config.num_samples // 3)  # Small random exploration first
                )
                
                search_alg = OptunaSearch(
                    metric=self.config.metric,
                    mode=self.config.mode,
                    sampler=optuna_sampler
                )
            except ImportError:
                logger.error("Optuna not available. Cannot use CMA-ES. Install with: pip install optuna")
                logger.warning("Falling back to random search.")
                search_alg = None
        else:
            # Random Search (default)
            search_alg = None
        
        # Limit concurrency
        if search_alg is not None:
            search_alg = ConcurrencyLimiter(
                search_alg, max_concurrent=self.config.max_concurrent_trials
            )
        
        return search_alg
    
    def _configure_scheduler(self):
        """Configure scheduler based on configuration.
        
        Returns:
            ray.tune.schedulers.TrialScheduler: Trial scheduler
        """
        if not RAY_AVAILABLE:
            return None
        
        if self.config.scheduler == "asha":
            # Asynchronous Successive Halving Algorithm
            scheduler = ASHAScheduler(
                metric=self.config.metric,
                mode=self.config.mode,
                max_t=self.config.max_resource_value,
                grace_period=self.config.grace_period,
                reduction_factor=self.config.reduction_factor
            )
        elif self.config.scheduler == "median":
            # Median Stopping Rule
            scheduler = MedianStoppingRule(
                metric=self.config.metric,
                mode=self.config.mode,
                grace_period=self.config.grace_period
            )
        else:
            # FIFO (default)
            scheduler = None
        
        return scheduler
    
    def _save_trial_results(self, analysis, experiment_name):
        """Save trial results to file.
        
        Args:
            analysis: Ray Tune analysis object
            experiment_name: Name of the experiment
        """
        # Create results directory
        results_dir = os.path.join(self.config.output_dir, experiment_name, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save best trial
        best_trial = analysis.best_trial
        best_config = analysis.best_config
        best_result = analysis.best_result
        
        best_trial_info = {
            "trial_id": best_trial.trial_id,
            "config": best_config,
            "result": best_result
        }
        
        with open(os.path.join(results_dir, "best_trial.json"), 'w') as f:
            json.dump(best_trial_info, f, indent=4)
        
        # Save all trial results
        trial_dfs = analysis.trial_dataframes
        
        # Convert to DataFrame
        results_df = None
        for trial_id, df in trial_dfs.items():
            df["trial_id"] = trial_id
            if results_df is None:
                results_df = df
            else:
                results_df = pd.concat([results_df, df])
        
        if results_df is not None:
            results_df.to_csv(os.path.join(results_dir, "all_trials.csv"), index=False)
        
        # Save analysis object
        with open(os.path.join(results_dir, "analysis.pkl"), 'wb') as f:
            pickle.dump(analysis, f)
        
        logger.info(f"Saved trial results to {results_dir}")
    
    def _reduce_memory(self, target_reduction: float):
        """Reduce memory usage for a tune trial.
        
        Args:
            target_reduction: Target reduction percentage
        """
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_search_space(
    base_config: Dict[str, Any],
    param_space: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Create a search space for AutoML.
    
    Args:
        base_config: Base configuration with default values
        param_space: Parameter space specification
        
    Returns:
        Dict[str, Any]: Search space for AutoML
    """
    if not RAY_AVAILABLE:
        logger.warning("Ray Tune not available. Search space will be limited.")
        return base_config
    
    # Start with base config
    search_space = base_config.copy()
    
    # Add parameter space
    for param_name, param_config in param_space.items():
        param_type = param_config.get("type", "float")
        
        if param_type == "float":
            search_space[param_name] = tune.loguniform(
                param_config.get("min", 1e-4),
                param_config.get("max", 1.0)
            )
        elif param_type == "int":
            search_space[param_name] = tune.randint(
                param_config.get("min", 1),
                param_config.get("max", 10) + 1  # +1 because randint is exclusive
            )
        elif param_type == "choice":
            search_space[param_name] = tune.choice(
                param_config.get("values", [])
            )
        elif param_type == "grid":
            search_space[param_name] = tune.grid_search(
                param_config.get("values", [])
            )
        else:
            # Use default value
            search_space[param_name] = param_config.get("default")
    
    return search_space


def create_model_factory(model_class, fixed_params=None):
    """Create a model factory function.
    
    Args:
        model_class: Model class
        fixed_params: Fixed parameters for the model
        
    Returns:
        Callable: Model factory function
    """
    def model_factory(config):
        """Create a model instance from configuration.
        
        Args:
            config: Configuration with hyperparameters
            
        Returns:
            torch.nn.Module: Model instance
        """
        # Combine fixed params and config
        params = {}
        if fixed_params:
            params.update(fixed_params)
        params.update(config)
        
        # Create model
        model = model_class(**params)
        
        return model
    
    return model_factory


def analyze_hyperparameters(results_path: str):
    """Analyze hyperparameter importance from trial results.
    
    Args:
        results_path: Path to trial results (CSV file)
        
    Returns:
        pd.DataFrame: Hyperparameter importance dataframe
    """
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("Required packages not available for hyperparameter analysis.")
        logger.error("Install with: pip install pandas scikit-learn matplotlib")
        return None
    
    # Load results
    df = pd.read_csv(results_path)
    
    # Identify hyperparameters vs. metrics
    hp_prefix = "config/"
    metric_cols = [col for col in df.columns if col.startswith("result/")]
    primary_metric = [col for col in metric_cols if "loss" in col]
    if primary_metric:
        primary_metric = primary_metric[0]
    else:
        primary_metric = metric_cols[0] if metric_cols else None
    
    if primary_metric is None:
        logger.error("No metrics found in results.")
        return None
    
    # Extract hyperparameters
    hp_cols = [col for col in df.columns if col.startswith(hp_prefix)]
    
    if not hp_cols:
        logger.error("No hyperparameters found in results.")
        return None
    
    # Create feature matrix
    X = df[hp_cols].copy()
    
    # Clean data (remove non-numeric columns)
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except:
                logger.warning(f"Removing non-numeric column: {col}")
                X.drop(col, axis=1, inplace=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get target variable
    y = df[primary_metric]
    
    # Train random forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'].str.replace(hp_prefix, ''), importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Hyperparameter')
    plt.title('Hyperparameter Importance')
    plt.tight_layout()
    
    # Save visualization
    viz_path = results_path.replace('.csv', '_importance.png')
    plt.savefig(viz_path, dpi=300)
    
    logger.info(f"Saved hyperparameter importance visualization to {viz_path}")
    
    return importance_df