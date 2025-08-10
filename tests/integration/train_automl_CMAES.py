#!/usr/bin/env python3
# train_automl_CMAES.py
"""
CMA-ES AutoML script for TMS E-field magnitude prediction using Ray Tune with Optuna.
Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for hyperparameter optimization.
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

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import Optuna (required for CMA-ES)
try:
    import optuna
    from ray.tune.search.optuna import OptunaSearch
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.error("Optuna not available. Cannot use CMA-ES. Install with: pip install optuna")
    sys.exit(1)

# Import from our codebase
from tms_efield_prediction.automl.integration.ray_trainable import train_model_tune
from tms_efield_prediction.automl.integration.tune_wrapper import create_search_space, AutoMLConfig
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.experiments.tracking import ExperimentTracker
from tms_efield_prediction.utils.resource.gpu_checker import configure_gpu_environment
# Import TrivialAugment - make sure this path matches your actual implementation
from tms_efield_prediction.data.transformations.augmentation import TrivialAugment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cmaes_automl')

def trainable_wrapper(config):
    """
    Convert numeric parameters back to categorical values and round integer params 
    before passing to the actual trainable.
    """
    # Make a copy of the config to avoid modifying the original
    modified_config = config.copy()
    
    # Set defaults for missing parameters
    defaults = {
        'feature_maps': 16,
        'levels': 3,
        'dropout_rate': 0.2,
        'feature_multiplier': 2.0,  # Default to original behavior (doubling)
        'norm_type': 'batch',
        'activation': 'relu',
        'use_residual': True,
        'use_attention': False,
        'optimizer_type': 'adamw',
        'scheduler_type': 'reduce_on_plateau',
        'scheduler_patience': 3,
        'early_stopping_patience': 7,
        'learning_rate': 0.001,
        'output_shape': (20, 20, 20),  # Add default output shape
        
        # Fixed loss function settings
        'loss_type': 'magnitude_mse',
        'mask_threshold': 0.001,
        
        # TrivialAugment parameters - only need max strength values
        'use_trivial_augment': True,  # Use TrivialAugment by default
        'trivial_augment_max_rotation': 30.0,  # Max rotation in degrees
        'trivial_augment_max_shift': 5,        # Max shift in voxels
        'trivial_augment_max_elastic': 3.0,    # Max elastic deformation strength
        'trivial_augment_max_intensity': 0.1,  # Max intensity change (delta from 1.0)
        'trivial_augment_max_noise': 0.05,     # Max noise standard deviation
    }
    
    # Apply defaults for missing values
    for key, default_value in defaults.items():
        if key not in modified_config or modified_config[key] is None:
            modified_config[key] = default_value
    
    # Round integer parameters
    modified_config['feature_maps'] = int(round(modified_config['feature_maps']))
    modified_config['levels'] = int(round(modified_config['levels']))
    modified_config['scheduler_patience'] = int(round(modified_config['scheduler_patience']))
    modified_config['early_stopping_patience'] = int(round(modified_config['early_stopping_patience']))
    modified_config['trivial_augment_max_shift'] = int(round(modified_config['trivial_augment_max_shift']))
    
    # Process boolean parameters
    if isinstance(modified_config['use_residual'], float):
        modified_config['use_residual'] = modified_config['use_residual'] > 0.5
    if isinstance(modified_config['use_attention'], float):
        modified_config['use_attention'] = modified_config['use_attention'] > 0.5
    if isinstance(modified_config['use_trivial_augment'], float):
        modified_config['use_trivial_augment'] = modified_config['use_trivial_augment'] > 0.5
    
    # Handle output shape - ensure it's a tuple if it's a list
    if isinstance(modified_config['output_shape'], list):
        modified_config['output_shape'] = tuple(modified_config['output_shape'])
    
    # Set up model configuration explicitly
    modified_config['model_config'] = {
        "model_type": "simple_unet_magnitude",
        "output_shape": [1, *modified_config['output_shape']],  # Use configured output shape
        "output_channels": 1,  # Always 1 for magnitude
        "feature_maps": modified_config['feature_maps'],
        "levels": modified_config['levels'],
        "norm_type": modified_config['norm_type'],
        "activation": modified_config['activation'],
        "dropout_rate": modified_config['dropout_rate'],
        "use_residual": modified_config['use_residual'],
        "use_attention": modified_config['use_attention'],
    }
    
    # Check if TrivialAugment is enabled
    if modified_config.get('use_trivial_augment', True):
        # Create TrivialAugment configuration
        modified_config['trivial_augment_config'] = {
            'max_rotation_degrees': modified_config['trivial_augment_max_rotation'],
            'max_shift': modified_config['trivial_augment_max_shift'],
            'max_elastic_strength': modified_config['trivial_augment_max_elastic'],
            'max_intensity_factor': modified_config['trivial_augment_max_intensity'],
            'max_noise_std': modified_config['trivial_augment_max_noise'],
            'center': (modified_config['output_shape'][0]//2,  # Dynamically calculate center
                      modified_config['output_shape'][1]//2,
                      modified_config['output_shape'][2]//2)
        }
        
        # Set augmentation_config to None to ensure TrivialAugment is used
        modified_config['augmentation_config'] = None
        
        # Clean up config by removing individual TrivialAugment parameters
        for param in [
            'trivial_augment_max_rotation', 'trivial_augment_max_shift', 
            'trivial_augment_max_elastic', 'trivial_augment_max_intensity', 
            'trivial_augment_max_noise'
        ]:
            if param in modified_config:
                del modified_config[param]
    else:
        # If TrivialAugment is disabled, use the standard augmentation approach
        modified_config['trivial_augment_config'] = None
        
        # Calculate center based on output shape
        center = (modified_config['output_shape'][0]//2,
                 modified_config['output_shape'][1]//2,
                 modified_config['output_shape'][2]//2)
        
        # Create standard augmentation config
        modified_config['augmentation_config'] = {
            'enabled': True,
            'rotation': {
                'enabled': True,
                'max_angle_y': np.radians(30.0),
                'probability': 0.5,
                'center': center,  # Use dynamic center
                'y_only': True
            },
            'elastic_deformation': {
                'enabled': True,
                'max_strength': 3.0,
                'sigma': 4.0,
                'probability': 0.5
            },
            'intensity_scaling': {
                'enabled': True,
                'min_factor': 0.9,
                'max_factor': 1.1,
                'probability': 0.5,
                'per_channel': False
            },
            'gaussian_noise': {
                'enabled': True,
                'max_std': 0.03,
                'probability': 0.5,
                'per_channel': False
            },
            'spatial_shift': {
                'enabled': True,
                'max_shift': 5,
                'probability': 0.5
            }
        }
    
    # Call the actual trainable
    return train_model_tune(modified_config)

if __name__ == "__main__":
    import psutil

    # Detect available resources
    num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    total_system_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB

    # Each trial needs at least 10GB RAM based on the error logs
    memory_per_trial = 10  # GB per trial
    # Use a conservative memory limit - 70% of total system RAM to leave room for other processes
    safe_memory_limit = int(total_system_ram * 0.7)
    # Calculate how many trials we can run based on memory
    memory_based_concurrent = max(1, int(safe_memory_limit / memory_per_trial))

    # Configure GPU environment
    gpu_config = configure_gpu_environment()
    default_max_concurrent = gpu_config["max_concurrent"]
    
    # Get number of available GPUs
    num_gpus_available = len(gpu_config["gpu_ids"]) if "gpu_ids" in gpu_config else (torch.cuda.device_count() if torch.cuda.is_available() else 0)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CMA-ES AutoML for TMS E-field magnitude prediction')
    parser.add_argument('--data-dir', type=str, default='/home/freyhe/MA_Henry/data', help='Data directory path')
    parser.add_argument('--output-dir', type=str, default='cmaes_automl_output', help='Output directory for results')
    parser.add_argument('--max-concurrent', type=int, default=default_max_concurrent, 
                   help=f'Maximum number of concurrent trials (default: {default_max_concurrent} - based on available GPUs)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                   help='Comma-separated list of specific GPU IDs to use (e.g., "0,1,3,7"). If not specified, all available GPUs will be used.')
    # Subject assignments
    parser.add_argument('--train-subjects', type=str, default='4,6,7,8', help='Comma-separated list of training subject IDs')
    parser.add_argument('--val-subjects', type=str, default='3', help='Comma-separated list of validation subject IDs')
    parser.add_argument('--test-subjects', type=str, default='9', help='Comma-separated list of test subject IDs')
    
    # AutoML parameters
    parser.add_argument('--num-samples', type=int, default=70, help='Number of trials to run')
    parser.add_argument('--max-epochs', type=int, default=15, help='Maximum number of epochs per trial')
    
    # CMA-ES specific parameters
    parser.add_argument('--sigma0', type=float, default=0.5, help='Initial standard deviation for CMA-ES')
    parser.add_argument('--population-size', type=int, default=None, 
                       help='Population size for CMA-ES (default: None - auto determined based on dimensions)')
    
    # Use stacked arrays flag
    parser.add_argument('--use-stacked-arrays', action='store_true', default=True, 
                      help='Use pre-stacked array files (default: True)')
    parser.add_argument('--use-separate-files', action='store_true',
                      help='Use separate E-field and dA/dt files instead of stacked arrays')
    


    
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Parse subject lists
    train_subjects = [s.strip().zfill(3) for s in args.train_subjects.split(',')]
    val_subjects = [s.strip().zfill(3) for s in args.val_subjects.split(',')]
    test_subjects = [s.strip().zfill(3) for s in args.test_subjects.split(',')]
    
    # Determine whether to use stacked arrays (if --use-separate-files is specified, override --use-stacked-arrays)
    use_stacked_arrays = args.use_stacked_arrays and not args.use_separate_files
    logger.info(f"Using {'stacked arrays' if use_stacked_arrays else 'separate files'} for data loading")
    
    logger.info(f"Using {len(train_subjects)} subjects for training: {train_subjects}")
    logger.info(f"Using {len(val_subjects)} subjects for validation: {val_subjects}")
    logger.info(f"Using {len(test_subjects)} subjects for testing: {test_subjects}")
    
    # Create timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"cmaes_automl_{timestamp}"
    output_dir = os.path.abspath(os.path.join(args.output_dir, experiment_name))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment tracker
    tracker = ExperimentTracker(
        experiment_dir=output_dir,
        architecture_name="simple_unet_magnitude_automl",
        create_subdirs=True
    )
    tracker.set_description(f"CMA-ES AutoML for TMS E-field magnitude prediction")
    
    # Create resource monitor
    resource_monitor = ResourceMonitor(max_memory_gb=8, check_interval=10.0)
    resource_monitor.start_monitoring()
    
    # Define base configuration (non-tunable parameters)
    base_config = {
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'data_dir': args.data_dir,
        'batch_size': 64,
        'epochs': args.max_epochs,
        'augment': True,
        'use_trivial_augment': True,  # Enable TrivialAugment
        'norm_type': 'batch',         # Fixed to batch normalization
        'activation': 'relu',         # Fixed to ReLU
        'optimizer_type': 'adamw',    # Fixed to AdamW
        'scheduler_type': 'reduce_on_plateau',  # Fixed to reduce_on_plateau
        'use_stacked_arrays': use_stacked_arrays,  
        'use_residual': True,         # Fixed to True based on previous runs
        'use_attention': False,       # Fixed to False based on previous runs
        'early_stopping_patience': 7, # Fixed at 7
        'output_shape': (20, 20, 20), # Specify output shape explicitly
        
        # ADD GRADIENT CHECKPOINTING HERE:
        'gradient_checkpointing': True,  # Enable memory-efficient training
        
        # Fixed loss function parameters
        'loss_type': 'magnitude_mse',  # Use original MSE loss
        'mask_threshold': 0.001,
    }
    
    # Process specific GPU IDs if provided
    if args.gpu_ids:
        # Parse comma-separated list
        specified_gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
        logger.info(f"User specified GPU IDs: {specified_gpu_ids}")
        
        # Override the automatically detected GPU IDs
        gpu_config["gpu_ids"] = specified_gpu_ids
        # Update num_gpus_available
        num_gpus_available = len(specified_gpu_ids)
        
        # Set environment variable for CUDA visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={args.gpu_ids}")
    
    logger.info(f"Number of GPUs that will be used: {num_gpus_available}")
    
    # Override the max_concurrent parameter if needed to use all GPUs
    # Only override if user didn't explicitly specify a lower value
    if args.max_concurrent == default_max_concurrent:  # Using the default value
        max_concurrent = num_gpus_available
        logger.info(f"Setting max_concurrent to match available GPU count: {max_concurrent}")
    else:
        # User specified a custom max_concurrent value
        if args.max_concurrent < num_gpus_available:
            logger.info(f"User specified max_concurrent ({args.max_concurrent}) is less than available GPUs ({num_gpus_available})")
            max_concurrent = args.max_concurrent
        else:
            max_concurrent = num_gpus_available
            logger.info(f"Setting max_concurrent to available GPU count: {max_concurrent}")
    
    # Calculate optimal concurrent trials based on GPU availability
    gpu_per_trial = 1  # Using 1 GPU per trial
    
    # Log detected GPUs
    logger.info(f"Detected {num_gpus_available} available GPUs with IDs: {gpu_config['gpu_ids']}")
    logger.info(f"GPU memory status: {gpu_config.get('gpu_memory_info', 'Not available')}")
    
    # Calculate max concurrent based on available GPUs
    if num_gpus_available > 0 and gpu_per_trial > 0:
        # Calculate how many trials can run concurrently based on GPU requirement
        concurrent_trials_by_gpu = num_gpus_available  # One trial per available GPU
        logger.info(f"Setting max concurrent trials to {concurrent_trials_by_gpu} to utilize all available GPUs")
        max_concurrent = concurrent_trials_by_gpu
    else:
        # If no GPUs, use the provided argument
        max_concurrent = args.max_concurrent
        logger.info(f"No GPUs detected, using specified max_concurrent: {max_concurrent}")
    
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    total_memory_gb = psutil.virtual_memory().total / (1024**3)

    logger.info(f"System memory: {available_memory_gb:.1f}GB available, {total_memory_gb:.1f}GB total")

    # Calculate safe memory allocation for Ray
    # Use much more conservative memory allocation
    safe_memory_gb = max(0.5, available_memory_gb * 0.3)  # Use only 30% of available memory
    object_store_memory_bytes = int(safe_memory_gb * 0.5 * 1024**3)  # Half of safe memory for object store

    # When initializing Ray, use only supported parameters
    ray_init_kwargs = {
        "num_cpus": max(2, 2 * max_concurrent),  # Reduce CPU allocation
        "num_gpus": num_gpus_available if num_gpus_available > 0 else None,
        "log_to_driver": True,
        "object_store_memory": object_store_memory_bytes,
        # REMOVED: "memory": int(safe_memory_gb * 1024**3),  # This parameter doesn't exist!
    }

    logger.info(f"Ray init kwargs: CPUs={ray_init_kwargs['num_cpus']}, "
            f"GPUs={ray_init_kwargs['num_gpus']}, "
            f"Memory limit removed (not supported), "
            f"Object store={object_store_memory_bytes/(1024**3):.1f}GB")

    # Set environment variables for memory management before initializing Ray
    os.environ["RAY_memory_usage_threshold"] = "0.98"  # Raise the threshold from 0.95 to 0.98

    if not ray.is_initialized():
        try:
            ray.init(**ray_init_kwargs)
            logger.info("Ray initialized successfully")
        except Exception as e:
            logger.error(f"Ray initialization failed: {e}")
            # Try with even more conservative settings - remove object_store_memory too
            minimal_ray_kwargs = {
                "num_cpus": 2,
                "num_gpus": num_gpus_available if num_gpus_available > 0 else None,
                "log_to_driver": True,
            }
            logger.info("Retrying Ray initialization with minimal settings...")
            ray.init(**minimal_ray_kwargs)
        
    # Configure CMA-ES with Optuna
    optuna_sampler = optuna.samplers.CmaEsSampler(
        sigma0=args.sigma0,
        n_startup_trials=min(3, args.num_samples // 3),  # Use a few random trials for initial exploration
        seed=42,
        popsize=args.population_size  # None means it will be automatically determined
    )
    
    # Define the search space for CMA-ES
    search_space = {
        # Model parameters
        "learning_rate": tune.loguniform(1e-5, 1e-2),           # Learning rate
        "feature_maps": tune.uniform(16.0, 64.0),            # Feature maps (rounded to int)
        "levels": tune.uniform(2.0, 5.0),                    # UNet levels (rounded to int)
        "dropout_rate": tune.uniform(0.0, 0.5),              # Dropout rate
        "feature_multiplier": tune.uniform(1.3, 4.0),         # Feature multiplier
        # TrivialAugment parameters
        "trivial_augment_max_rotation": tune.uniform(15.0, 60.0),    # Maximum rotation in degrees
        "trivial_augment_max_shift": tune.uniform(1.0, 10.0),        # Maximum shift in voxels
        "trivial_augment_max_elastic": tune.uniform(1.0, 5.0),       # Maximum elastic deformation strength
        "trivial_augment_max_intensity": tune.uniform(0.05, 0.15),   # Maximum intensity delta from 1.0
        "trivial_augment_max_noise": tune.uniform(0.01, 0.05)        # Maximum noise standard deviation
    }
    
    # Create OptunaSearch instance with CMA-ES
    search_alg = OptunaSearch(
        space=search_space,
        metric="val_loss",
        mode="min",
        sampler=optuna_sampler,
    )
    
    # Limit concurrency
    from ray.tune.search import ConcurrencyLimiter
    search_alg = ConcurrencyLimiter(
        search_alg, max_concurrent=max_concurrent
    )
    
    # Configure ASHA scheduler
    from ray.tune.schedulers import ASHAScheduler
    
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=args.max_epochs,
        grace_period=5,
        reduction_factor=3
    )
    
    # Run trials using our custom training function
    logger.info(f"Starting CMA-ES Optimization with {args.num_samples} trials...")
    logger.info(f"Max concurrent trials: {max_concurrent}")
    logger.info(f"Max epochs per trial: {args.max_epochs}")
    logger.info(f"CMA-ES parameters: sigma0={args.sigma0}, population_size={args.population_size or 'auto'}")
    logger.info(f"Using TrivialAugment: ONE random augmentation per sample (never stacked)")
    logger.info(f"Gradient checkpointing enabled for memory efficiency")
    
    # Set up resources per trial
    resources_per_trial = {
        "cpu": 4,
        "gpu": gpu_per_trial if torch.cuda.is_available() else 0
    }
    
    # Run trials
    analysis = tune.run(
        trainable_wrapper,
        config=base_config, 
        num_samples=args.num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial=resources_per_trial,
        local_dir=os.path.dirname(output_dir),
        name=os.path.basename(output_dir),
        max_concurrent_trials=max_concurrent,
        fail_fast=False,  # Don't abort everything on failure
        max_failures=3,   # Allow retries
        verbose=1
    )
    
    # Log best trial
    best_trial = analysis.best_trial
    best_config = analysis.best_config
    best_result = analysis.best_result
    
    # Convert numeric parameters back to readable format for reporting
    readable_config = best_config.copy()
    for param in ['feature_maps', 'levels', 'scheduler_patience', 'early_stopping_patience', 'trivial_augment_max_shift']:
        if param in readable_config:
            readable_config[param] = int(round(readable_config[param]))
    
    # Report TrivialAugment parameters in a clear format
    trivial_augment_params = {
        'max_rotation_degrees': readable_config.get('trivial_augment_max_rotation', 30.0),
        'max_shift': readable_config.get('trivial_augment_max_shift', 5),
        'max_elastic_strength': readable_config.get('trivial_augment_max_elastic', 3.0),
        'max_intensity_factor': readable_config.get('trivial_augment_max_intensity', 0.1),
        'max_noise_std': readable_config.get('trivial_augment_max_noise', 0.05)
    }
    
    logger.info(f"Best trial: {best_trial.trial_id}")
    logger.info(f"Best config: {readable_config}")
    logger.info(f"Best TrivialAugment params: {trivial_augment_params}")
    logger.info(f"Best result: {best_result}")
    
    # Save best configuration to a separate file
    with open(os.path.join(output_dir, 'best_config.yaml'), 'w') as f:
        yaml.dump(readable_config, f, default_flow_style=False)
    
    # Save TrivialAugment parameters separately for clarity
    with open(os.path.join(output_dir, 'best_trivial_augment_params.yaml'), 'w') as f:
        yaml.dump(trivial_augment_params, f, default_flow_style=False)
    
    # Save trial results
    try:
        trial_results_path = os.path.join(output_dir, 'trial_results.csv')
        analysis.results_df.to_csv(trial_results_path, index=False)
        logger.info(f"Saved trial results to {trial_results_path}")
    except Exception as e:
        logger.error(f"Error saving trial results: {e}")
    
    # Stop resource monitor
    resource_monitor.stop_monitoring()
    
    # Finalize experiment tracker
    tracker.finalize(status="completed")
    
    logger.info(f"CMA-ES AutoML experiment completed. Results saved to {output_dir}")