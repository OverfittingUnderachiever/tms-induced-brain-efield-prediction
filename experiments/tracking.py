"""
TMS E-field Prediction Experiment Tracking

This module provides experiment tracking infrastructure for TMS E-field prediction models.
It implements the ExperimentTracker class for logging, saving, and comparing experiment results.
"""

import os
import json
import time
import shutil
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import hashlib
import torch
import matplotlib.pyplot as plt
import yaml

from ..models.evaluation.metrics import calculate_metrics
from ..models.evaluation.visualization import (
    visualize_prediction_vs_ground_truth,
    generate_standard_visualizations,
    visualize_model_comparison
)
from ..utils.state.context import ModelContext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tms_experiment_tracker')


class ExperimentTracker:
    """Experiment tracking for TMS E-field prediction models."""
    
    def __init__(self, 
                experiment_dir: str,
                architecture_name: str,
                create_subdirs: bool = True,
                debug_mode: bool = False):
        """Initialize experiment tracker.
        
        Args:
            experiment_dir: Base directory for experiments
            architecture_name: Name of the architecture
            create_subdirs: Whether to create subdirectories for the experiment
            debug_mode: Whether to enable debug logging
        """
        self.base_dir = experiment_dir
        self.architecture_name = architecture_name
        self.debug_mode = debug_mode
        
        # Set up logging level
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{architecture_name}_{timestamp}"
        
        # Create experiment directory structure
        if create_subdirs:
            self.experiment_dir = os.path.join(self.base_dir, self.architecture_name, self.experiment_id)
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
            self.results_dir = os.path.join(self.experiment_dir, "results")
            self.logs_dir = os.path.join(self.experiment_dir, "logs")
            self.visualization_dir = os.path.join(self.experiment_dir, "visualizations")
            
            os.makedirs(self.experiment_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.logs_dir, exist_ok=True)
            os.makedirs(self.visualization_dir, exist_ok=True)
        else:
            self.experiment_dir = self.base_dir
            self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
            self.results_dir = os.path.join(self.base_dir, "results")
            self.logs_dir = os.path.join(self.base_dir, "logs")
            self.visualization_dir = os.path.join(self.base_dir, "visualizations")
        
        # Initialize experiment metadata
        self.metadata = {
            "experiment_id": self.experiment_id,
            "architecture_name": architecture_name,
            "timestamp": timestamp,
            "start_time": time.time(),
            "end_time": None,
            "status": "initialized",
            "description": "",
            "hyperparameters": {},
            "metrics": {},
            "dataset_info": {},
            "environment_info": self._get_environment_info(),
            "artifacts": {}
        }
        
        # Save initial metadata
        self._save_metadata()
        
        logger.info(f"Initialized experiment tracker: {self.experiment_id}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def set_description(self, description: str):
        """Set experiment description.
        
        Args:
            description: Experiment description
        """
        self.metadata["description"] = description
        self._save_metadata()
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log model hyperparameters.
        
        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        self.metadata["hyperparameters"] = hyperparameters
        
        # Save as separate YAML file for better readability
        hyperparams_path = os.path.join(self.logs_dir, "hyperparameters.yaml")
        with open(hyperparams_path, 'w') as f:
            yaml.dump(hyperparameters, f, default_flow_style=False)
        
        self._save_metadata()
        
        logger.debug(f"Logged hyperparameters: {len(hyperparameters)} parameters")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information.
        
        Args:
            dataset_info: Dictionary with dataset information
        """
        self.metadata["dataset_info"] = dataset_info
        self._save_metadata()
        
        logger.debug(f"Logged dataset info: {dataset_info}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step or epoch number
        """
        # Initialize metrics dict if needed
        if "metrics" not in self.metadata:
            self.metadata["metrics"] = {}
        
        # Add step information if provided
        metrics_with_step = metrics.copy()
        if step is not None:
            metrics_with_step["step"] = step
        
        # Add to metrics history
        timestamp = time.time()
        self.metadata["metrics"][str(timestamp)] = metrics_with_step
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(list(self.metadata["metrics"].values()))
        metrics_path = os.path.join(self.results_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        # Update metadata file
        self._save_metadata()
        
        logger.debug(f"Logged metrics: {metrics}")
    
    def log_artifact(self, name: str, artifact_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an artifact file.
        
        Args:
            name: Artifact name
            artifact_path: Path to artifact file
            metadata: Optional metadata for the artifact
        """
        # Copy artifact file to experiment directory if it's not already there
        if not artifact_path.startswith(self.experiment_dir):
            # Generate artifact directory based on type
            artifact_ext = os.path.splitext(artifact_path)[1].lower()
            if artifact_ext in ['.pt', '.pth']:
                dest_dir = self.checkpoint_dir
            elif artifact_ext in ['.png', '.jpg', '.jpeg', '.svg']:
                dest_dir = self.visualization_dir
            elif artifact_ext in ['.json', '.yaml', '.yml']:
                dest_dir = self.logs_dir
            else:
                dest_dir = os.path.join(self.experiment_dir, "artifacts")
                os.makedirs(dest_dir, exist_ok=True)
            
            # Generate unique filename if name conflict
            dest_filename = os.path.basename(artifact_path)
            dest_path = os.path.join(dest_dir, dest_filename)
            
            # Handle name collisions
            if os.path.exists(dest_path):
                base_name, ext = os.path.splitext(dest_filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_filename = f"{base_name}_{timestamp}{ext}"
                dest_path = os.path.join(dest_dir, dest_filename)
            
            # Copy the file
            shutil.copy2(artifact_path, dest_path)
            artifact_path = dest_path
        
        # Store artifact in metadata
        if "artifacts" not in self.metadata:
            self.metadata["artifacts"] = {}
        
        self.metadata["artifacts"][name] = {
            "path": artifact_path,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Update metadata file
        self._save_metadata()
        
        logger.debug(f"Logged artifact: {name} at {artifact_path}")
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
                      epoch: Optional[int] = None, metrics: Optional[Dict[str, float]] = None,
                      checkpoint_name: Optional[str] = None):
        """Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optional optimizer
            epoch: Optional epoch number
            metrics: Optional metrics to save with checkpoint
            checkpoint_name: Optional checkpoint name, defaults to epoch-based name
            
        Returns:
            str: Path to saved checkpoint
        """
        if checkpoint_name is None and epoch is not None:
            checkpoint_name = f"checkpoint_epoch_{epoch}"
        elif checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Log as artifact
        self.log_artifact(
            name=checkpoint_name,
            artifact_path=checkpoint_path,
            metadata={
                "epoch": epoch,
                "metrics": metrics
            }
        )
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dict[str, Any]: Loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def evaluate_and_log(self, model: torch.nn.Module, test_data: Tuple, 
                       visualize: bool = True):
        """Evaluate model on test data and log results.
        
        Args:
            model: PyTorch model
            test_data: Tuple of (features, targets)
            visualize: Whether to generate visualizations
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        features, targets = test_data
        device = next(model.parameters()).device
        
        # Prepare data
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        
        # Move to model device
        features = features.to(device)
        targets = targets.to(device)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            predictions = model(features)
        
        # Move to CPU for metric calculation
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, targets)
        
        # Log metrics
        self.log_metrics(metrics)
        
        # Generate visualizations if requested
        if visualize:
            # Use first sample for visualization
            if predictions.ndim == 5 and predictions.shape[0] > 0:
                pred_sample = predictions[0]
            else:
                pred_sample = predictions
            
            if targets.ndim == 5 and targets.shape[0] > 0:
                target_sample = targets[0]
            else:
                target_sample = targets
            
            viz_paths = generate_standard_visualizations(
                model_name=self.architecture_name,
                pred=pred_sample,
                target=target_sample,
                metrics=metrics,
                output_dir=self.visualization_dir
            )
            
            # Log visualizations as artifacts
            for name, path in viz_paths.items():
                self.log_artifact(f"visualization_{name}", path)
        
        logger.info(f"Evaluated model: {self.architecture_name}")
        logger.info(f"Metrics: {metrics}")
        
        return metrics
    
    def finalize(self, status: str = "completed"):
        """Finalize the experiment and save final results.
        
        Args:
            status: Final experiment status
        """
        self.metadata["end_time"] = time.time()
        self.metadata["status"] = status
        self.metadata["duration_seconds"] = self.metadata["end_time"] - self.metadata["start_time"]
        
        # Generate a final experiment summary
        summary = {
            "experiment_id": self.experiment_id,
            "architecture_name": self.architecture_name,
            "duration_hours": self.metadata["duration_seconds"] / 3600,
            "status": status
        }
        
        # Add latest metrics
        if self.metadata["metrics"]:
            latest_metrics = list(self.metadata["metrics"].values())[-1]
            if "step" in latest_metrics:
                del latest_metrics["step"]
            summary["final_metrics"] = latest_metrics
        
        # Save summary to a file
        summary_path = os.path.join(self.experiment_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Update metadata
        self._save_metadata()
        
        logger.info(f"Finalized experiment: {self.experiment_id}")
        logger.info(f"Status: {status}")
        logger.info(f"Duration: {self.metadata['duration_seconds'] / 3600:.2f} hours")
        
        return summary
    
    def _save_metadata(self):
        """Save experiment metadata to file."""
        metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get information about the execution environment.
        
        Returns:
            Dict[str, Any]: Environment information
        """
        env_info = {
            "python_version": '.'.join(map(str, os.sys.version_info[:3])),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add PyTorch information if available
        try:
            env_info["torch_version"] = torch.__version__
            env_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env_info["cuda_version"] = torch.version.cuda
                env_info["gpu_count"] = torch.cuda.device_count()
                env_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except:
            pass
        
        return env_info
    
    def get_results(self) -> Dict[str, Any]:
        """Get experiment results.
        
        Returns:
            Dict[str, Any]: Experiment results
        """
        return {
            "experiment_id": self.experiment_id,
            "architecture_name": self.architecture_name,
            "status": self.metadata["status"],
            "hyperparameters": self.metadata["hyperparameters"],
            "metrics": self.metadata["metrics"],
            "duration_seconds": (self.metadata["end_time"] or time.time()) - self.metadata["start_time"]
        }


class ExperimentManager:
    """Manager for multiple experiments with architecture isolation."""
    
    def __init__(self, experiments_base_dir: str):
        """Initialize experiment manager.
        
        Args:
            experiments_base_dir: Base directory for all experiments
        """
        self.base_dir = experiments_base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create comparison directory
        self.comparison_dir = os.path.join(self.base_dir, "comparison")
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        # Cache of experiment trackers
        self.trackers = {}
        
        logger.info(f"Initialized experiment manager at {self.base_dir}")
    
    def create_experiment(self, architecture_name: str, description: str = "",
                        debug_mode: bool = False) -> ExperimentTracker:
        """Create a new experiment tracker.
        
        Args:
            architecture_name: Name of the architecture
            description: Experiment description
            debug_mode: Whether to enable debug logging
            
        Returns:
            ExperimentTracker: Created experiment tracker
        """
        # Create architecture-specific directory
        arch_dir = os.path.join(self.base_dir, architecture_name)
        os.makedirs(arch_dir, exist_ok=True)
        
        # Create experiment tracker
        tracker = ExperimentTracker(
            experiment_dir=self.base_dir,
            architecture_name=architecture_name,
            create_subdirs=True,
            debug_mode=debug_mode
        )
        
        # Set description
        if description:
            tracker.set_description(description)
        
        # Store in cache
        self.trackers[tracker.experiment_id] = tracker
        
        logger.info(f"Created experiment: {tracker.experiment_id}")
        return tracker
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentTracker]:
        """Get experiment tracker by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Optional[ExperimentTracker]: Experiment tracker or None if not found
        """
        # Check if in cache
        if experiment_id in self.trackers:
            return self.trackers[experiment_id]
        
        # Try to find experiment
        for root, dirs, files in os.walk(self.base_dir):
            if "metadata.json" in files:
                metadata_path = os.path.join(root, "metadata.json")
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get("experiment_id") == experiment_id:
                        # Found experiment, recreate tracker
                        arch_name = metadata.get("architecture_name", "unknown")
                        tracker = ExperimentTracker(
                            experiment_dir=root,
                            architecture_name=arch_name,
                            create_subdirs=False
                        )
                        
                        # Update metadata
                        tracker.metadata = metadata
                        
                        # Add to cache
                        self.trackers[experiment_id] = tracker
                        return tracker
                except:
                    logger.warning(f"Failed to load metadata from {metadata_path}")
        
        logger.warning(f"Experiment not found: {experiment_id}")
        return None
    
    def list_experiments(self, architecture_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments.
        
        Args:
            architecture_name: Optional filter by architecture name
            
        Returns:
            List[Dict[str, Any]]: List of experiment summaries
        """
        experiments = []
        
        # Get architecture directories
        if architecture_name:
            arch_dirs = [os.path.join(self.base_dir, architecture_name)]
        else:
            arch_dirs = [os.path.join(self.base_dir, d) for d in os.listdir(self.base_dir) 
                      if os.path.isdir(os.path.join(self.base_dir, d)) and d != "comparison"]
        
        # Scan architecture directories
        for arch_dir in arch_dirs:
            if not os.path.exists(arch_dir):
                continue
                
            # Get experiment directories
            exp_dirs = [os.path.join(arch_dir, d) for d in os.listdir(arch_dir) 
                     if os.path.isdir(os.path.join(arch_dir, d))]
            
            # Scan experiment directories
            for exp_dir in exp_dirs:
                metadata_path = os.path.join(exp_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Create summary
                        summary = {
                            "experiment_id": metadata.get("experiment_id", "unknown"),
                            "architecture_name": metadata.get("architecture_name", "unknown"),
                            "status": metadata.get("status", "unknown"),
                            "start_time": metadata.get("start_time", 0),
                            "end_time": metadata.get("end_time", None),
                            "description": metadata.get("description", ""),
                            "directory": exp_dir
                        }
                        
                        # Add latest metrics
                        metrics = metadata.get("metrics", {})
                        if metrics:
                            latest_metrics = list(metrics.values())[-1]
                            if "step" in latest_metrics:
                                del latest_metrics["step"]
                            summary["latest_metrics"] = latest_metrics
                        
                        experiments.append(summary)
                    except:
                        logger.warning(f"Failed to load metadata from {metadata_path}")
        
        # Sort by start time (most recent first)
        experiments.sort(key=lambda x: x.get("start_time", 0), reverse=True)
        
        return experiments
    
    def compare_experiments(self, experiment_ids: List[str] = None, 
                          architecture_names: List[str] = None,
                          metrics: List[str] = None,
                          visualize: bool = True) -> pd.DataFrame:
        """Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            architecture_names: List of architecture names to compare (best experiment per architecture)
            metrics: List of metrics to include (if None, use all available)
            visualize: Whether to generate comparison visualizations
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        if experiment_ids is None and architecture_names is None:
            logger.error("Must provide either experiment_ids or architecture_names")
            return pd.DataFrame()
        
        # Get experiments to compare
        experiments_to_compare = []
        
        if experiment_ids:
            # Get specified experiments
            for exp_id in experiment_ids:
                tracker = self.get_experiment(exp_id)
                if tracker:
                    experiments_to_compare.append(tracker.get_results())
        
        if architecture_names:
            # Get best experiment for each architecture
            for arch_name in architecture_names:
                arch_experiments = self.list_experiments(arch_name)
                if arch_experiments:
                    # Find best experiment (lowest loss or highest metric)
                    best_exp = None
                    best_value = None
                    
                    for exp in arch_experiments:
                        if "latest_metrics" in exp and exp.get("status") == "completed":
                            metrics_dict = exp["latest_metrics"]
                            
                            # Try to find loss metric
                            for metric_name in metrics_dict:
                                if "loss" in metric_name.lower():
                                    value = metrics_dict[metric_name]
                                    if best_value is None or value < best_value:
                                        best_value = value
                                        best_exp = exp
                                    break
                            
                            # If no loss metric, try first available metric
                            if best_exp is None and metrics_dict:
                                value = next(iter(metrics_dict.values()))
                                if best_value is None or value > best_value:
                                    best_value = value
                                    best_exp = exp
                    
                    if best_exp:
                        tracker = self.get_experiment(best_exp["experiment_id"])
                        if tracker:
                            experiments_to_compare.append(tracker.get_results())
        
        if not experiments_to_compare:
            logger.warning("No experiments found to compare")
            return pd.DataFrame()
        
        # Create comparison dataframe
        comparison_data = []
        
        for exp in experiments_to_compare:
            exp_data = {
                "experiment_id": exp["experiment_id"],
                "architecture": exp["architecture_name"],
                "duration_hours": exp.get("duration_seconds", 0) / 3600
            }
            
            # Add hyperparameters
            hyperparams = exp.get("hyperparameters", {})
            for key, value in hyperparams.items():
                if isinstance(value, (int, float, str, bool)):
                    exp_data[f"hp_{key}"] = value
            
            # Add metrics
            latest_metrics = {}
            metrics_dict = exp.get("metrics", {})
            if metrics_dict:
                latest_metrics = list(metrics_dict.values())[-1]
                if "step" in latest_metrics:
                    del latest_metrics["step"]
            
            for key, value in latest_metrics.items():
                exp_data[key] = value
            
            comparison_data.append(exp_data)
        
        # Create dataframe
        df = pd.DataFrame(comparison_data)
        
        # Generate visualization if requested
        if visualize and len(comparison_data) > 0:
            # Create metrics comparison dict
            metrics_dict = {}
            for exp in comparison_data:
                arch_name = exp["architecture"]
                metrics_dict[arch_name] = {
                    k: v for k, v in exp.items() 
                    if k not in ["experiment_id", "architecture", "duration_hours"] 
                    and not k.startswith("hp_")
                }
            
            # Generate visualization
            viz_path = os.path.join(self.comparison_dir, "model_comparison.png")
            _ = visualize_model_comparison(
                metrics_dict, 
                metrics=metrics,
                save_path=viz_path,
                title="Model Architecture Comparison"
            )
            
            # Save comparison data
            comparison_path = os.path.join(self.comparison_dir, "comparison_data.csv")
            df.to_csv(comparison_path, index=False)
            
            logger.info(f"Generated model comparison visualization at {viz_path}")
            logger.info(f"Saved comparison data to {comparison_path}")
        
        return df
    
    def get_best_model(self, architecture_name: str, metric_name: str = "loss", 
                     higher_is_better: bool = False) -> Optional[Dict[str, Any]]:
        """Get the best model for a specific architecture.
        
        Args:
            architecture_name: Name of the architecture
            metric_name: Metric to use for comparison
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Optional[Dict[str, Any]]: Best experiment summary or None if not found
        """
        # Get all experiments for this architecture
        experiments = self.list_experiments(architecture_name)
        
        if not experiments:
            logger.warning(f"No experiments found for architecture: {architecture_name}")
            return None
        
        # Find best experiment
        best_exp = None
        best_value = None
        
        for exp in experiments:
            if "latest_metrics" in exp and exp.get("status") == "completed":
                metrics_dict = exp["latest_metrics"]
                
                if metric_name in metrics_dict:
                    value = metrics_dict[metric_name]
                    
                    if best_value is None or (higher_is_better and value > best_value) or (not higher_is_better and value < best_value):
                        best_value = value
                        best_exp = exp
        
        if best_exp:
            logger.info(f"Found best model for {architecture_name}: {best_exp['experiment_id']}")
            logger.info(f"Best {metric_name}: {best_value}")
            return best_exp
        
        logger.warning(f"No valid experiments found for {architecture_name} with metric {metric_name}")
        return None


def generate_experiment_report(experiment_id: str, output_dir: str = "reports"):
    """Generate a comprehensive report for an experiment.
    
    Args:
        experiment_id: Experiment ID
        output_dir: Directory to save the report
        
    Returns:
        str: Path to the report file
    """
    # Create report directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize experiment manager
    manager = ExperimentManager("experiments")
    
    # Get experiment
    tracker = manager.get_experiment(experiment_id)
    if not tracker:
        logger.error(f"Experiment not found: {experiment_id}")
        return None
    
    # Get experiment results
    results = tracker.get_results()
    
    # Create report data
    report_data = {
        "experiment_id": results["experiment_id"],
        "architecture_name": results["architecture_name"],
        "status": results["status"],
        "duration_hours": results["duration_seconds"] / 3600,
        "hyperparameters": results["hyperparameters"],
        "metrics_history": results["metrics"],
        "metadata": tracker.metadata
    }
    
    # Get artifacts
    artifacts = tracker.metadata.get("artifacts", {})
    visualization_artifacts = {k: v for k, v in artifacts.items() if k.startswith("visualization_")}
    
    # Generate report path
    report_path = os.path.join(output_dir, f"report_{experiment_id}.json")
    
    # Save report data
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    
    # Copy visualizations to report directory
    report_viz_dir = os.path.join(output_dir, f"viz_{experiment_id}")
    os.makedirs(report_viz_dir, exist_ok=True)
    
    for name, artifact in visualization_artifacts.items():
        src_path = artifact["path"]
        if os.path.exists(src_path):
            dest_path = os.path.join(report_viz_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
    
    logger.info(f"Generated experiment report: {report_path}")
    return report_path