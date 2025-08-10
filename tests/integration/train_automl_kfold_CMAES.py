#!/usr/bin/env python3
# train_automl_kfold_CMAES.py
"""
CMA-ES AutoML script for TMS E-field magnitude prediction using Ray Tune with Optuna.
Uses CMA-ES for hyperparameter optimization and supports K-Fold/LOO with fold limiting.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import torch
import numpy as np
import yaml
import ray
from ray import tune
import psutil
from ray.tune import Callback

from tms_efield_prediction.data.pipeline.multi_subject_data import MultiSubjectDataManager
from tms_efield_prediction.models.training.losses import LossFactory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import optuna
    from ray.tune.search.optuna import OptunaSearch
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.error("Optuna not available. Cannot use CMA-ES. Install with: pip install optuna")
    sys.exit(1)

from tms_efield_prediction.automl.integration.ray_trainable import train_model_tune
from tms_efield_prediction.automl.integration.tune_wrapper import create_search_space, AutoMLConfig
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.experiments.tracking import ExperimentTracker
from tms_efield_prediction.utils.resource.gpu_checker import configure_gpu_environment
from tms_efield_prediction.automl.integration.kfold_automl import run_automl_with_params # Import the main function
from tms_efield_prediction.data.transformations.augmentation import TrivialAugment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cmaes_automl')


import ray
import os

os.makedirs(os.path.expanduser("~/ray_tmp"), exist_ok=True)
ray.init(_temp_dir=os.path.expanduser("~/ray_tmp"))

class ModelManager(Callback):
    """
    Ray Tune callback to manage saved models across all trials.
    Ensures only the top N best-performing models are kept.
    """
    
    def __init__(self, max_models=5, base_dir=None):
        """
        Initialize the model manager.
        
        Args:
            max_models: Maximum number of models to keep
            base_dir: Base directory where experiment results are stored
        """
        self.max_models = max_models
        self.base_dir = base_dir
        self.best_models = []  # list of (val_loss, trial_id, model_path) tuples
        
    def on_trial_result(self, iteration, trials, trial, result, **info):
        """
        Called when a trial reports a result to Ray Tune.
        Tracks models with the best validation loss.
        """
        if 'val_loss' not in result:
            return
        
        val_loss = result['val_loss']
        trial_id = trial.trial_id
        
        # Get the model path for this trial
        trial_dir = trial.logdir
        if self.base_dir is None:
            # Extract base directory from trial_dir
            self.base_dir = os.path.dirname(os.path.dirname(trial_dir))
        
        model_path = os.path.join(trial_dir, "model.pt")
        
        # If the model file doesn't exist yet, we'll consider it later
        if not os.path.exists(model_path):
            return
        
        # Check if this trial is already in our list
        for i, (loss, tid, path) in enumerate(self.best_models):
            if tid == trial_id:
                # Update the entry if this result is better
                if val_loss < loss:
                    self.best_models[i] = (val_loss, trial_id, model_path)
                    # Re-sort the list since we updated a value
                    self.best_models.sort()
                return
        
        # This is a new trial - add it to our tracking list
        if len(self.best_models) < self.max_models:
            # We don't have max_models yet, so just add it
            heapq.heappush(self.best_models, (val_loss, trial_id, model_path))
        elif val_loss < self.best_models[-1][0]:
            # Better than our worst saved model, replace it
            # First, delete the old model file
            _, _, old_path = self.best_models[-1]
            try:
                if os.path.exists(old_path):
                    os.remove(old_path)
                    logger.info(f"Removed model: {old_path}")
            except Exception as e:
                logger.warning(f"Failed to remove model {old_path}: {e}")
            
            # Remove worst model from heap
            heapq.heappop(self.best_models)
            # Add new model to heap
            heapq.heappush(self.best_models, (val_loss, trial_id, model_path))
    
    def on_trial_complete(self, iteration, trials, trial, **info):
        """
        Called when a trial completes.
        Ensures model is properly tracked if it wasn't during on_trial_result.
        """
        trial_id = trial.trial_id
        
        # Check if this trial is already in our list
        for _, tid, _ in self.best_models:
            if tid == trial_id:
                return  # Already tracking this trial
        
        # If not in our list, need to check if it should be
        try:
            result_file = os.path.join(trial.logdir, "result.json")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    results = json.load(f)
                    if 'val_loss' in results:
                        val_loss = results['val_loss']
                        model_path = os.path.join(trial.logdir, "model.pt")
                        
                        if os.path.exists(model_path):
                            if len(self.best_models) < self.max_models:
                                heapq.heappush(self.best_models, (val_loss, trial_id, model_path))
                            elif val_loss < self.best_models[-1][0]:
                                # Better than worst, replace it
                                _, _, old_path = heapq.heappop(self.best_models)
                                try:
                                    if os.path.exists(old_path):
                                        os.remove(old_path)
                                        logger.info(f"Removed model: {old_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to remove model {old_path}: {e}")
                                
                                heapq.heappush(self.best_models, (val_loss, trial_id, model_path))
        except Exception as e:
            logger.warning(f"Error processing trial completion for {trial_id}: {e}")
    
    def on_experiment_end(self, trials, **info):
        """
        Called when the entire experiment ends.
        Creates a summary file of the best models.
        """
        if not self.best_models:
            logger.warning("No models were tracked during the experiment.")
            return
            
        # Create a summary file
        summary_path = os.path.join(self.base_dir, "best_models_summary.json")
        try:
            summary = {
                "num_models_kept": len(self.best_models),
                "max_models_limit": self.max_models,
                "models": [
                    {
                        "val_loss": loss,
                        "trial_id": tid,
                        "model_path": path,
                        "relative_path": os.path.relpath(path, self.base_dir)
                    }
                    for loss, tid, path in sorted(self.best_models)
                ]
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Saved summary of {len(self.best_models)} best models to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to create model summary: {e}")


def trainable_wrapper(config):
    """Wraps the main training function for Ray Tune, handling parameter types."""
    modified_config = config.copy()

    defaults = {
        'feature_maps': 16, 'levels': 3, 'dropout_rate': 0.2,
        'feature_multiplier': 2.0, 'norm_type': 'batch', 'activation': 'relu',
        'use_residual': True, 'use_attention': False, 'optimizer_type': 'adamw',
        'scheduler_type': 'reduce_on_plateau', 'scheduler_patience': 3,
        'early_stopping_patience': 10, 'learning_rate': 0.001,
        'output_shape': (20, 20, 20), 'loss_type': 'magnitude_mse', # This default is fine
        'mask_threshold': 0.001, 'use_trivial_augment': True,
        'trivial_augment_max_rotation': 30.0, 'trivial_augment_max_shift': 5,
        'trivial_augment_max_elastic': 3.0, 'trivial_augment_max_intensity': 0.1,
        'trivial_augment_max_noise': 0.05,
        'model_mode': 'vector', 
        }

    for key, default_value in defaults.items():
        if key not in modified_config or modified_config[key] is None:
            modified_config[key] = default_value

    # Round integer parameters
    for int_param in ['feature_maps', 'levels', 'scheduler_patience', 'early_stopping_patience', 'trivial_augment_max_shift']:
        if int_param in modified_config:
             modified_config[int_param] = int(round(modified_config[int_param]))

    # Process boolean parameters
    for bool_param in ['use_residual', 'use_attention', 'use_trivial_augment']:
         if bool_param in modified_config and isinstance(modified_config[bool_param], float):
             modified_config[bool_param] = modified_config[bool_param] > 0.5

    if isinstance(modified_config['output_shape'], list):
        modified_config['output_shape'] = tuple(modified_config['output_shape'])

    # Determine model type and output channels based on model_mode
    if modified_config.get('model_mode', 'magnitude') == 'vector':
        model_type = "simple_unet_vector"
        output_channels = 3  # 3 channels for vector output
        loss_type = modified_config.get('loss_type', 'vector_mse')
    else:
        model_type = "simple_unet_magnitude"
        output_channels = 1  # 1 channel for magnitude output
        loss_type = modified_config.get('loss_type', 'magnitude_mse')
    
    # Ensure loss_type is preserved if it's set to ushape_mse
    if modified_config.get('loss_type') == 'ushape_mse':
        loss_type = 'ushape_mse'
    
    modified_config['model_config'] = {
        "model_type": "simple_unet_vector",  # Changed from simple_unet_magnitude
        "input_channels": 6,  # Explicitly set for 3-channel MRI + 3-channel dA/dt
        "output_shape": [1, *modified_config['output_shape']],
        "output_channels": 1,
        "feature_maps": modified_config['feature_maps'],
        "levels": modified_config['levels'],
        "norm_type": modified_config['norm_type'],
        "activation": modified_config['activation'],
        "dropout_rate": modified_config['dropout_rate'],
        "use_residual": modified_config['use_residual'],
        "use_attention": modified_config['use_attention'],
    }
    
    # Update loss type based on model mode
    modified_config['loss_type'] = loss_type

    center = (modified_config['output_shape'][0]//2, modified_config['output_shape'][1]//2, modified_config['output_shape'][2]//2)

    if modified_config.get('use_trivial_augment', True):
        modified_config['trivial_augment_config'] = {
            'max_rotation_degrees': modified_config['trivial_augment_max_rotation'],
            'max_shift': modified_config['trivial_augment_max_shift'],
            'max_elastic_strength': modified_config['trivial_augment_max_elastic'],
            'max_intensity_factor': modified_config['trivial_augment_max_intensity'],
            'max_noise_std': modified_config['trivial_augment_max_noise'],
            'center': center
        }
        modified_config['augmentation_config'] = None # Ensure standard augmentation is off
        # Clean up individual TrivialAugment params from top level
        for param in ['trivial_augment_max_rotation', 'trivial_augment_max_shift',
                      'trivial_augment_max_elastic', 'trivial_augment_max_intensity',
                      'trivial_augment_max_noise']:
            if param in modified_config: del modified_config[param]
    else:
        modified_config['trivial_augment_config'] = None
        # Setup standard augmentation if TrivialAugment is off
        modified_config['augmentation_config'] = {
             'enabled': True, # Assuming standard aug is wanted if Trivial is off
             'rotation': {'enabled': True, 'max_angle_y': np.radians(30.0), 'probability': 0.5, 'center': center, 'y_only': True},
             'elastic_deformation': {'enabled': True, 'max_strength': 3.0, 'sigma': 4.0, 'probability': 0.5},
             'intensity_scaling': {'enabled': True, 'min_factor': 0.9, 'max_factor': 1.1, 'probability': 0.5, 'per_channel': False},
             'gaussian_noise': {'enabled': True, 'max_std': 0.03, 'probability': 0.5, 'per_channel': False},
             'spatial_shift': {'enabled': True, 'max_shift': 5, 'probability': 0.5}
         }

    return train_model_tune(modified_config)

def run_automl_optimization(train_subjects, val_subjects, test_subjects, args, output_dir, experiment_name=None):
    """Runs a single Ray Tune optimization process for given subjects and config."""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"cmaes_automl_{timestamp}"

    experiment_output_dir = os.path.abspath(os.path.join(output_dir, experiment_name))
    os.makedirs(experiment_output_dir, exist_ok=True)

    tracker = ExperimentTracker(
        experiment_dir=experiment_output_dir,
        architecture_name="simple_unet_magnitude_automl",
        create_subdirs=True
    )
    tracker.set_description(f"CMA-ES AutoML for TMS E-field magnitude prediction")

    resource_monitor = ResourceMonitor(max_memory_gb=8, check_interval=10.0)
    resource_monitor.start_monitoring()

    train_subjects = [str(s).strip().zfill(3) for s in train_subjects]
    val_subjects = [str(s).strip().zfill(3) for s in val_subjects]
    test_subjects = [str(s).strip().zfill(3) for s in test_subjects]

    logger.info(f"Running optimization. Train: {train_subjects}, Val: {val_subjects}, Test: {test_subjects}")
    use_stacked_arrays = args.use_stacked_arrays and not args.use_separate_files
    logger.info(f"Using {'stacked arrays' if use_stacked_arrays else 'separate files'}")

    
    base_config = {
        'train_subjects': train_subjects, 'val_subjects': val_subjects, 'test_subjects': test_subjects,
        'data_dir': args.data_dir, 'batch_size': 16, 'epochs': args.max_epochs,
        'augment': True, 'use_trivial_augment': True, 'norm_type': 'batch',
        'activation': 'relu', 'optimizer_type': 'adamw', 'scheduler_type': 'reduce_on_plateau',
        'use_stacked_arrays': use_stacked_arrays, 'use_residual': True, 'use_attention': False,
        'early_stopping_patience': 10, 'output_shape': (20, 20, 20),
        'loss_type': 'ushape_mse',  # Using our U-shape MSE loss
        'mask_threshold': 0.01,
        'mri_type': args.mri_type,
        'max_models_to_keep': args.max_models_to_keep,  # ADD THIS LINE
        
    }

    # Rest of the function remains the same
    # Search space for tunable hyperparameters
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "feature_maps": tune.uniform(64.0, 128.0),
        "levels": tune.uniform(3.0, 5.0),
        "dropout_rate": tune.uniform(0.1, 0.3),
        "feature_multiplier": tune.uniform(1.3, 3.5),
        "trivial_augment_max_rotation": tune.uniform(25.0, 60.0),
        "trivial_augment_max_shift": tune.uniform(1.0, 10.0),
        "trivial_augment_max_elastic": tune.uniform(1.0, 5.0),
        "trivial_augment_max_intensity": tune.uniform(0.05, 0.15),
        "trivial_augment_max_noise": tune.uniform(0.01, 0.05)
    }

    # GPU configuration
    gpu_config = configure_gpu_environment()
    num_gpus_available = len(gpu_config.get("gpu_ids", [])) if "gpu_ids" in gpu_config else (torch.cuda.device_count() if torch.cuda.is_available() else 0)

    if args.gpu_ids:
        specified_gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        num_gpus_available = len(specified_gpu_ids)
        logger.info(f"User specified GPU IDs: {specified_gpu_ids}. Setting CUDA_VISIBLE_DEVICES={args.gpu_ids}. Visible GPUs: {num_gpus_available}")
    else:
         logger.info(f"Using all {num_gpus_available} available GPUs.")

    max_concurrent = min(args.max_concurrent, num_gpus_available) if num_gpus_available > 0 else args.max_concurrent
    logger.info(f"Setting max concurrent trials to {max_concurrent}")

    # CMA-ES Sampler
    optuna_sampler = optuna.samplers.CmaEsSampler(
        sigma0=args.sigma0,
        n_startup_trials=min(5, args.num_samples // 3),
        seed=42,
        popsize=args.population_size
    )

    search_alg = OptunaSearch(
        space=search_space, metric="val_loss", mode="min", sampler=optuna_sampler
    )

    from ray.tune.search import ConcurrencyLimiter
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)

    from ray.tune.schedulers import ASHAScheduler
    scheduler = ASHAScheduler(
        metric="val_loss", mode="min", max_t=args.max_epochs,
        grace_period=5, reduction_factor=3
    )

    resources_per_trial = {"cpu": 4, "gpu": 1 if num_gpus_available > 0 else 0}

    logger.info(f"Starting CMA-ES Optimization: {args.num_samples} trials, {max_concurrent} concurrent, {args.max_epochs} max epochs/trial.")
    logger.info(f"CMA-ES params: sigma0={args.sigma0}, population_size={args.population_size or 'auto'}")
    
    model_manager = ModelManager(
        max_models=args.max_models_to_keep,
        base_dir=os.path.dirname(experiment_output_dir)
    )
    analysis = tune.run(
        trainable_wrapper,
        config=base_config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial=resources_per_trial,
        local_dir=os.path.dirname(experiment_output_dir),
        name=os.path.basename(experiment_output_dir),
        max_concurrent_trials=max_concurrent,
        fail_fast=False,
        max_failures=3,
        verbose=1,

        callbacks=[model_manager]  # Add our custom model manager
    )

    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    best_config = analysis.get_best_config("val_loss", "min")
    best_result = analysis.get_best_result("val_loss", "min")

    readable_config = best_config.copy() if best_config else {}
    for param in ['feature_maps', 'levels', 'trivial_augment_max_shift']:
        if param in readable_config:
            readable_config[param] = int(round(readable_config[param]))

    if best_trial:
        logger.info(f"Best trial ID: {best_trial.trial_id}")
        logger.info(f"Best config found: {readable_config}")
        logger.info(f"Best result metrics: {best_result}")
        


        with open(os.path.join(experiment_output_dir, 'best_config.yaml'), 'w') as f:
            yaml.dump(readable_config, f, default_flow_style=False)
    else:
         logger.warning("No best trial found. Optimization might have failed or yielded no results.")

    try:
        results_df = analysis.results_df
        trial_results_path = os.path.join(experiment_output_dir, 'trial_results.csv')
        results_df.to_csv(trial_results_path, index=False)
        logger.info(f"Saved all trial results to {trial_results_path}")
    except Exception as e:
        logger.error(f"Error saving trial results dataframe: {e}")

    resource_monitor.stop_monitoring()
    tracker.finalize(status="completed" if best_trial else "failed")
    logger.info(f"AutoML optimization run finished. Results saved to {experiment_output_dir}")

    return analysis, best_trial, readable_config, best_result

def keep_top_models(experiment_dir, max_models=5):
    """
    Keep only the top N models based on validation loss.
    Handles both Ray Tune checkpoint directories and the 'checkpoints' directory.
    
    Args:
        experiment_dir: Directory where experiment results are stored
        max_models: Maximum number of models to keep
    """
    import glob
    import os
    import json
    import shutil
    
    logger.info(f"Cleaning up model files, keeping only top {max_models}...")
    
    # Find all trial directories
    trial_dirs = glob.glob(os.path.join(experiment_dir, "checkpoint_*"))
    
    # Extract validation loss from each trial
    trial_scores = []
    for trial_dir in trial_dirs:
        try:
            # Check for result.json file
            result_file = os.path.join(trial_dir, "result.json")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if 'val_loss' in data:
                        val_loss = data['val_loss']
                        trial_scores.append((val_loss, trial_dir))
                        logger.info(f"Trial {os.path.basename(trial_dir)}: val_loss = {val_loss}")
                    else:
                        logger.warning(f"No val_loss found in {result_file}")
        except Exception as e:
            logger.warning(f"Error processing trial {trial_dir}: {e}")
    
    # Check for duplicated checkpoints directory
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    if os.path.exists(checkpoints_dir) and os.path.isdir(checkpoints_dir):
        logger.info(f"Found additional 'checkpoints' directory: {checkpoints_dir}")
        # If we're also disabling the internal checkpointing, we can just remove this directory
        try:
            shutil.rmtree(checkpoints_dir)
            logger.info(f"Removed duplicated 'checkpoints' directory to save space")
        except Exception as e:
            logger.error(f"Failed to remove 'checkpoints' directory: {e}")
    
    if len(trial_scores) <= max_models:
        logger.info(f"Only {len(trial_scores)} valid trials found, no cleanup needed.")
        return
    
    # Sort by validation loss (ascending)
    trial_scores.sort()
    
    # Keep only the top N models
    trials_to_keep = trial_scores[:max_models]
    trials_to_remove = trial_scores[max_models:]
    
    logger.info(f"Keeping top {len(trials_to_keep)} trials with val_loss range: "
               f"{trials_to_keep[0][0]:.6f} to {trials_to_keep[-1][0]:.6f}")
    
    # Remove the extra trials
    for val_loss, trial_dir in trials_to_remove:
        try:
            logger.info(f"Removing trial: {os.path.basename(trial_dir)} with val_loss={val_loss:.6f}")
            shutil.rmtree(trial_dir)
        except Exception as e:
            logger.error(f"Failed to remove trial {trial_dir}: {e}")
    
    # Create a summary file
    summary_file = os.path.join(experiment_dir, "top_models_summary.json")
    try:
        summary = {
            "top_models": [
                {
                    "val_loss": val_loss,
                    "directory": os.path.basename(trial_dir),
                    "full_path": trial_dir
                }
                for val_loss, trial_dir in trials_to_keep
            ]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved summary of top {len(trials_to_keep)} models to {summary_file}")
    except Exception as e:
        logger.error(f"Failed to create model summary: {e}")


def parse_arguments():
    """Parse command line arguments."""
    num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    default_max_concurrent = max(1, num_gpus_available)
    
    
    parser = argparse.ArgumentParser(description='CMA-ES AutoML for TMS E-field magnitude prediction with K-Fold/LOO.')
    # Paths
    parser.add_argument('--mri-type', type=str, default='dti', choices=['dti', 'conductivity'],
                   help='Type of MRI data to use: "dti" for DTI tensor or "conductivity" for tissue type')
    parser.add_argument('--data-dir', type=str, default='/home/freyhe/MA_Henry/data', help='Data directory path')
    parser.add_argument('--output-dir', type=str, default='cmaes_automl_output', help='Base output directory for results')
    # Resources
    parser.add_argument('--max-concurrent', type=int, default=default_max_concurrent,
                        help=f'Max concurrent Ray Tune trials (default: {default_max_concurrent}, based on available GPUs)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                        help='Comma-separated list of specific GPU IDs to use (e.g., "0,1"). Uses all if None.')
    # Subjects
    parser.add_argument('--train-subjects', type=str, required=True,
                        help='Comma-separated list of training subject IDs (or all subjects for LOO/K-Fold pool)')
    parser.add_argument('--val-subjects', type=str, default="",
                        help='Comma-separated list of validation subject IDs, or "loo" for Leave-One-Out CV.')
    parser.add_argument('--test-subjects', type=str, required=True,
                        help='Comma-separated list of fixed test subject IDs')
    # AutoML / Tune
    parser.add_argument('--num-samples', type=int, default=100, help='Number of hyperparameter trials per fold')
    parser.add_argument('--max-epochs', type=int, default=25, help='Maximum number of epochs per trial')
    # CMA-ES
    parser.add_argument('--sigma0', type=float, default=0.5, help='Initial standard deviation for CMA-ES')
    parser.add_argument('--population-size', type=int, default=None,
                        help='Population size for CMA-ES (default: None - auto determined)')
    # Data Loading
    parser.add_argument('--use-stacked-arrays', action='store_true', default=True, help='Use pre-stacked array files')
    parser.add_argument('--use-separate-files', action='store_true', help='Force use of separate E-field/dA/dt files')
    # K-Fold / LOO Control
    parser.add_argument('--fold-dir-name', type=str, default='kfold_results',
                        help='Directory name prefix for results when using K-Fold/LOO validation')
    parser.add_argument('--max-folds-to-run', type=int, default=None,
                        help='Limit execution to only the first N folds generated by LOO or K-Fold')
    parser.add_argument('--k-folds', type=int, default=None,  # <<< THIS LINE ADDED/ENSURED
                        help='Specify number of folds for standard K-Fold CV (if not using --val-subjects loo)')

    parser.add_argument('--max-models-to-keep', type=int, default=20,
                        help='Maximum number of model checkpoints to keep during training')
    args = parser.parse_args()

    if args.use_separate_files:
        args.use_stacked_arrays = False
    else:
        args.use_stacked_arrays = True

    return args

def configure_ray_environment(args):
    """Initialize Ray based on GPU availability and arguments."""
    gpu_ids_to_use = []
    num_gpus_for_ray = 0

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        gpu_ids_to_use = args.gpu_ids.split(',')
        num_gpus_for_ray = len(gpu_ids_to_use)
        logger.info(f"Configuring Ray to use specified GPUs: {args.gpu_ids}. Num GPUs for Ray: {num_gpus_for_ray}")
    elif torch.cuda.is_available():
        num_gpus_for_ray = torch.cuda.device_count()
        logger.info(f"Configuring Ray to use all available GPUs: {num_gpus_for_ray}")
    else:
         logger.info("Configuring Ray for CPU-only operation.")

    # Limit max concurrent trials based on available GPUs for Ray
    max_concurrent = min(args.max_concurrent, num_gpus_for_ray) if num_gpus_for_ray > 0 else args.max_concurrent

    if not ray.is_initialized():
        total_system_ram_gb = psutil.virtual_memory().total / (1024**3)
        # Allocate a fraction of memory to Ray object store, ensure it's reasonable
        object_store_mem = int(min(max(total_system_ram_gb * 0.2, 4), 30) * 1024 * 1024 * 1024) # Min 4GB, Max 30GB or 20%

        ray_init_kwargs = {
            "num_cpus": max(4, 4 * max_concurrent), # Allocate more CPUs based on concurrency
            "num_gpus": num_gpus_for_ray,
            "log_to_driver": True,
            "object_store_memory": object_store_mem
        }
        # Optional: Add dashboard host if running in specific environments
        # ray_init_kwargs["dashboard_host"] = "0.0.0.0"

        os.environ["RAY_memory_monitor_refresh_ms"] = "5000" # Check memory more often
        os.environ["RAY_memory_usage_threshold"] = "0.95" # Default threshold

        try:
            ray.init(**ray_init_kwargs)
        except Exception as e:
             logger.error(f"Failed to initialize Ray: {e}", exc_info=True)
             sys.exit(1)
    else:
        logger.info("Ray is already initialized.")

if __name__ == "__main__":
    args = parse_arguments()

    configure_ray_environment(args)

    train_subjects = [s.strip() for s in args.train_subjects.split(',') if s.strip()]

    if args.val_subjects.strip().lower() == 'loo':
        val_subjects = ['loo']
    else:
        val_subjects = [s.strip() for s in args.val_subjects.split(',') if s.strip()]
    test_subjects = [s.strip() for s in args.test_subjects.split(',') if s.strip()]

    if not train_subjects:
        logger.error("No training subjects provided. Exiting.")
        sys.exit(1)
    if not test_subjects:
        logger.error("No test subjects provided. Exiting.")
        sys.exit(1)
    if not val_subjects and args.val_subjects.strip().lower() != 'loo' and args.k_folds is None:
        logger.warning("No validation subjects provided and not using 'loo' or '--k-folds'. Running single optimization.")

    try:
        results = run_automl_with_params(
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            output_dir=args.output_dir,
            optimization_function=run_automl_optimization,
            main_args=args,
            fold_dir_name=args.fold_dir_name,
            max_folds_to_run=args.max_folds_to_run,
            k_folds=args.k_folds  # <<< THIS LINE ADDED/ENSURED
        )
        logger.info("AutoML run process completed.")
    except Exception as e:
        logger.error(f"An error occurred during the AutoML run: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown.")