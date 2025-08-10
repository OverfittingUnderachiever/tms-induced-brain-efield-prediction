#!/usr/bin/env python3
# test_single_run_kfold.py
"""
Standalone script to test TMS E-field magnitude prediction training,
optionally incorporating K-Fold cross-validation without Ray Tune/AutoML.
Uses fixed hyperparameter configurations and default settings defined within the script.
"""

import os
import sys
import logging
from datetime import datetime
import torch
import numpy as np
import yaml
import psutil # Keep for resource info if desired
import json     # For potentially saving results
from collections import defaultdict

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from our codebase - Adjust paths if necessary!
try:
    from tms_efield_prediction.automl.integration.ray_trainable import train_model_tune
    from tms_efield_prediction.utils.resource.gpu_checker import configure_gpu_environment
    from tms_efield_prediction.data.transformations.augmentation import TrivialAugment
    from tms_efield_prediction.automl.integration.kfold_automl import KFoldAutoMLManager # Import KFold manager
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from a location where the 'tms_efield_prediction' package is accessible")
    print("or adjust the sys.path.append line accordingly.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger('standalone_test_kfold')

# --- DEFAULT SETTINGS (Copied from above) ---
DEFAULT_SETTINGS = {
    "data_dir": '/home/freyhe/MA_Henry/data',
    "output_dir": 'standalone_test_output_kfold',
    "gpu_ids": "0",
    "train_subjects": '3,4,6,7,8,9',
    "val_subjects": '',
    "test_subjects": '1,2',
    "max_epochs": 15,
    "batch_size": 16,
    "use_stacked_arrays": True,
    "use_separate_files": False,
    "debug_data": True,
    "k_fold": 5,                 # <--- Number of folds (set <= 1 for single run)
    "k_fold_shuffle": True,
    "k_fold_random_state": 42,
}

# --- Fixed Hyperparameters (Copied from original test_single.py) ---
FIXED_HYPERPARAMETERS = {
    'learning_rate': 1.5e-3,
    'feature_maps': 32,
    'levels': 4,
    'dropout_rate': 0.15,
    'feature_multiplier': 2.5,
    'scheduler_patience': 3,
    'trivial_augment_max_rotation': 25.0,
    'trivial_augment_max_shift': 4,
    'trivial_augment_max_elastic': 2.5,
    'trivial_augment_max_intensity': 0.12,
    'trivial_augment_max_noise': 0.04,
    'augment': True,
    'use_trivial_augment': True,
    'norm_type': 'batch',
    'activation': 'relu',
    'optimizer_type': 'adamw',
    'scheduler_type': 'reduce_on_plateau',
    'use_residual': True,
    'use_attention': False,
    'early_stopping_patience': 7,
    'output_shape': (20, 20, 20),
    'loss_type': 'magnitude_mse',
    'mask_threshold': 0.01,
    'num_workers': 4, # Added for data loaders
    'pin_memory': True # Added for data loaders
}
# --- END OF CONFIGURABLE SECTION ---


def debug_data_directory(data_root_path):
    """Debug function to thoroughly check subject directories (copied from original)"""
    logger.info(f"\n===== DEBUGGING DATA DIRECTORY: {data_root_path} =====")
    if not data_root_path or not os.path.exists(data_root_path):
        logger.error(f"Data directory does not exist or path is invalid: '{data_root_path}'")
        return
    try:
        all_items = sorted(os.listdir(data_root_path))
        logger.info(f"All items in {data_root_path}: {all_items}")
        dirs = [d for d in all_items if os.path.isdir(os.path.join(data_root_path, d))]
        logger.info(f"Directories in {data_root_path}: {dirs}")
        subject_dirs = [d for d in dirs if d.startswith('sub-')]
        logger.info(f"Subject directories (with 'sub-' prefix): {subject_dirs}")
        if subject_dirs:
            first_subj_dir = os.path.join(data_root_path, subject_dirs[0])
            logger.info(f"Exploring first subject directory: {first_subj_dir}")
            subj_contents = os.listdir(first_subj_dir)
            logger.info(f"Contents of {first_subj_dir}: {subj_contents}")
            exp_dir = os.path.join(first_subj_dir, "experiment")
            if os.path.exists(exp_dir):
                logger.info(f"Experiment directory exists: {exp_dir}")
                exp_contents = os.listdir(exp_dir)
                logger.info(f"Contents of experiment directory: {exp_contents}")
                mri_dir = os.path.join(exp_dir, "MRI_arrays", "torch")
                if os.path.exists(mri_dir):
                    logger.info(f"MRI directory exists: {mri_dir}")
                    mri_files = os.listdir(mri_dir)
                    logger.info(f"MRI files: {mri_files}")
                else:
                    logger.warning(f"No MRI directory found at {mri_dir}")
            else:
                logger.warning(f"No experiment directory found at {exp_dir}")
    except Exception as e:
        logger.error(f"Error examining data directory: {e}", exc_info=True)
    logger.info("===== DEBUG COMPLETE =====")


def prepare_run_config(base_settings, hyperparameters, fold_subjects=None, fold_output_dir=None):
    """
    Prepares the final configuration dictionary for a single training run (or fold).

    Args:
        base_settings: Dictionary of general settings (paths, k-fold info, etc.).
        hyperparameters: Dictionary of fixed hyperparameters.
        fold_subjects (Optional[tuple]): Tuple of (train_subjects, val_subjects, test_subjects) for a specific fold.
                                         If None, uses subjects from base_settings.
        fold_output_dir (Optional[str]): Specific output directory for this run/fold.
                                         If None, uses base_settings["output_dir"].

    Returns:
        dict: The final configuration dictionary ready for train_model_tune.
    """
    final_config = {}
    final_config.update(base_settings)        # Start with general settings
    final_config.update(hyperparameters) # Add/overwrite with specific hyperparameters

    # --- Handle Subjects ---
    if fold_subjects:
        final_config['train_subjects'] = [s.strip().zfill(3) for s in fold_subjects[0]]
        final_config['val_subjects'] = [s.strip().zfill(3) for s in fold_subjects[1]]
        final_config['test_subjects'] = [s.strip().zfill(3) for s in fold_subjects[2]]
        logger.info(f"Using fold subjects - Train: {final_config['train_subjects']}, Val: {final_config['val_subjects']}, Test: {final_config['test_subjects']}")
    else:
        # Use subjects directly from settings for a single run
        final_config['train_subjects'] = [s.strip().zfill(3) for s in base_settings['train_subjects'].split(',') if s]
        final_config['val_subjects'] = [s.strip().zfill(3) for s in base_settings['val_subjects'].split(',') if s]
        final_config['test_subjects'] = [s.strip().zfill(3) for s in base_settings['test_subjects'].split(',') if s]
        logger.info(f"Using settings subjects - Train: {final_config['train_subjects']}, Val: {final_config['val_subjects']}, Test: {final_config['test_subjects']}")


    # --- Handle Output Directory ---
    final_config['output_dir'] = fold_output_dir if fold_output_dir else base_settings["output_dir"]

    # --- Data Loading Strategy ---
    final_config['use_stacked_arrays'] = base_settings.get('use_stacked_arrays', True) and not base_settings.get('use_separate_files', False)
    # logger.info(f"Data loading strategy: {'stacked arrays' if final_config['use_stacked_arrays'] else 'separate files'}") # Logged once outside

    # --- Replicate Config Processing (like in trainable_wrapper/original script) ---
    for key in ['feature_maps', 'levels', 'scheduler_patience', 'trivial_augment_max_shift', 'early_stopping_patience']:
         if key in final_config:
             final_config[key] = int(round(final_config[key]))

    final_config['model_config'] = {
        "model_type": "simple_unet_magnitude",
        "output_shape": [1, *final_config['output_shape']], # Prepend channel dim
        "output_channels": 1,
        "feature_maps": final_config['feature_maps'],
        "levels": final_config['levels'],
        "norm_type": final_config['norm_type'],
        "activation": final_config['activation'],
        "dropout_rate": final_config['dropout_rate'],
        "use_residual": final_config['use_residual'],
        "use_attention": final_config['use_attention'],
    }

    # --- Augmentation Config ---
    center = (final_config['output_shape'][0] // 2,
              final_config['output_shape'][1] // 2,
              final_config['output_shape'][2] // 2)

    if final_config.get('use_trivial_augment', True):
        # logger.info("Using TrivialAugment configuration.") # Logged once outside
        final_config['trivial_augment_config'] = {
            'max_rotation_degrees': final_config['trivial_augment_max_rotation'],
            'max_shift': final_config['trivial_augment_max_shift'],
            'max_elastic_strength': final_config['trivial_augment_max_elastic'],
            'max_intensity_factor': final_config['trivial_augment_max_intensity'],
            'max_noise_std': final_config['trivial_augment_max_noise'],
            'center': center
        }
        final_config['augmentation_config'] = None
    else:
        # logger.info("Using standard augmentation configuration.") # Logged once outside
        final_config['trivial_augment_config'] = None
        final_config['augmentation_config'] = {
             'enabled': True,
             'rotation': {'enabled': True, 'max_angle_y': np.radians(30.0), 'probability': 0.5, 'center': center, 'y_only': True},
             'elastic_deformation': {'enabled': True, 'max_strength': 3.0, 'sigma': 4.0, 'probability': 0.5},
             'intensity_scaling': {'enabled': True, 'min_factor': 0.9, 'max_factor': 1.1, 'probability': 0.5, 'per_channel': False},
             'gaussian_noise': {'enabled': True, 'max_std': 0.03, 'probability': 0.5, 'per_channel': False},
             'spatial_shift': {'enabled': True, 'max_shift': 5, 'probability': 0.5}
        }

    # --- Clean up K-Fold specific keys not needed by train_model_tune ---
    for key in ['k_fold', 'k_fold_shuffle', 'k_fold_random_state']:
        if key in final_config:
            del final_config[key]

    return final_config

def run_training_instance(config):
    """
    Runs a single training instance using the provided config.
    Handles calling train_model_tune and basic logging.
    """
    logger.info(f"--- Starting Training Instance ---")
    logger.info(f"Output Directory: {config['output_dir']}")

    # --- Log Final Config for this Instance ---
    log_config = config.copy()
    if 'device' in log_config and not isinstance(log_config['device'], (str, int, float, bool, list, dict, type(None))):
         log_config['device'] = str(log_config['device'])
    try:
        config_str = yaml.dump(log_config, default_flow_style=False, indent=2, sort_keys=False)
        logger.info("Configuration for this run:\n" + config_str)
        # Save config for this specific run/fold
        os.makedirs(config['output_dir'], exist_ok=True)
        config_save_path = os.path.join(config['output_dir'], 'run_config.yaml')
        with open(config_save_path, 'w') as f:
            f.write(config_str)
        logger.info(f"Saved run configuration to {config_save_path}")
    except Exception as dump_error:
        logger.warning(f"Could not dump/save config as YAML: {dump_error}")


    # --- Run the Training ---
    logger.info("Calling train_model_tune...")
    results = None
    try:
        # Directly call the core training function
        results = train_model_tune(config)

        logger.info("Training function call finished successfully.")
        if results:
             logger.info(f"Results/metrics returned: {results}")
             # Save results if returned
             results_save_path = os.path.join(config['output_dir'], 'run_results.json')
             try:
                 with open(results_save_path, 'w') as f:
                    # Convert numpy types for JSON
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.integer): return int(obj)
                            elif isinstance(obj, np.floating): return float(obj)
                            elif isinstance(obj, np.ndarray): return obj.tolist()
                            return super(NumpyEncoder, self).default(obj)
                    json.dump(results, f, indent=2, cls=NumpyEncoder)
                 logger.info(f"Saved run results to {results_save_path}")
             except Exception as e:
                 logger.error(f"Failed to save run results: {e}")

        else:
             logger.warning("Training function did not return specific results.")

    except ImportError as e:
         logger.error(f"ImportError during training: {e}. Check ray_trainable dependencies.")
    except Exception as e:
        logger.exception("Critical error during training run!") # Logs traceback

    logger.info(f"--- Finished Training Instance ---")
    return results


def main():
    """Main execution function"""
    settings = DEFAULT_SETTINGS
    hyperparameters = FIXED_HYPERPARAMETERS

    logger.info("--- Starting Standalone Test Script ---")
    logger.info(f"Base Settings: {settings}")
    logger.info(f"Fixed Hyperparameters: {hyperparameters}")

    # --- Check System Info ---
    total_system_ram = psutil.virtual_memory().total / (1024**3) # GB
    logger.info(f"System RAM: {total_system_ram:.2f} GB")

    # --- Optional Data Debug ---
    if settings.get("debug_data", False):
        debug_data_directory(settings.get("data_dir"))

    # --- Setup Base Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_fold_str = f"{settings['k_fold']}fold" if settings['k_fold'] > 1 else "single"
    base_output_dir = os.path.abspath(os.path.join(settings["output_dir"], f"{k_fold_str}_run_{timestamp}"))
    os.makedirs(base_output_dir, exist_ok=True)
    logger.info(f"Base output directory for this run: {base_output_dir}")
    # Update settings with the actual base output dir for potential use by KFoldManager
    settings["output_dir"] = base_output_dir

    # --- Configure GPU Environment ---
    # (This setup applies to all folds if run sequentially on the same machine)
    gpu_config = configure_gpu_environment() # Detects initial state
    num_gpus_available = len(gpu_config["gpu_ids"]) if "gpu_ids" in gpu_config else 0

    if settings.get("gpu_ids"): # Use .get for safety
        try:
            specified_gpu_ids_list = [int(gpu_id.strip()) for gpu_id in settings["gpu_ids"].split(',')]
            logger.info(f"Using specified GPU IDs: {specified_gpu_ids_list}")
            os.environ["CUDA_VISIBLE_DEVICES"] = settings["gpu_ids"]
            logger.info(f"Setting CUDA_VISIBLE_DEVICES={settings['gpu_ids']}")
        except ValueError:
            logger.error(f"Invalid gpu_ids format: '{settings['gpu_ids']}'. Falling back to default.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_config.get("gpu_ids", [])))
    else:
        logger.info(f"No specific GPU IDs provided. Using available: {gpu_config.get('gpu_ids', 'None')}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_config.get("gpu_ids", [])))

    # Check device after setting env var
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Effective device for training: {device}")
    if device.type == "cuda": logger.info(f"Visible GPUs to PyTorch: {torch.cuda.device_count()}")

    # --- Decide Run Mode: Single vs K-Fold ---
    num_folds = settings.get('k_fold', 1)

    if num_folds <= 1:
        logger.info("Running in single mode (k_fold <= 1).")
        # Prepare config for the single run
        run_config = prepare_run_config(settings, hyperparameters, fold_output_dir=base_output_dir)
        # Execute the single training instance
        run_training_instance(run_config)

    else:
        logger.info(f"Running in K-Fold mode with {num_folds} folds.")
        all_fold_results = []

        # --- Prepare Subjects for K-Fold ---
        # Combine initial train and val subjects into the pool for splitting
        initial_train_subjects = [s.strip() for s in settings['train_subjects'].split(',') if s]
        initial_val_subjects = [s.strip() for s in settings['val_subjects'].split(',') if s]
        subjects_for_kfold = sorted(list(set(initial_train_subjects + initial_val_subjects)))
        initial_test_subjects = [s.strip() for s in settings['test_subjects'].split(',') if s]

        logger.info(f"Subjects pool for K-Fold splitting: {subjects_for_kfold}")
        logger.info(f"Held-out test subjects: {initial_test_subjects}")

        if len(subjects_for_kfold) < num_folds:
             logger.error(f"Number of subjects for K-Fold ({len(subjects_for_kfold)}) is less than k ({num_folds}). Cannot perform K-Fold CV.")
             return # Exit if not enough subjects

        # --- Initialize KFold Manager ---
        kfold_manager = KFoldAutoMLManager(
            k=num_folds,
            base_output_dir=base_output_dir, # Manager can use this to save fold assignments
            fold_dir_name="folds_info",      # Subdir for assignments file
            shuffle=settings.get('k_fold_shuffle', True),
            random_state=settings.get('k_fold_random_state', 42)
        )

        # --- Generate Folds ---
        # Note: KFoldAutoMLManager expects train, val, test. We provide the pool and the test set.
        # It will split the pool into k train/val combinations, keeping the test set separate for each.
        folds = kfold_manager.prepare_folds(
            train_subjects=subjects_for_kfold, # Pass the combined pool here
            val_subjects=[],                  # Pass empty as validation comes from the split
            test_subjects=initial_test_subjects
        )
        logger.info(f"Generated {len(folds)} folds. Assignments saved in {kfold_manager.kfold_base_dir}")

        # --- Loop Through Folds ---
        for fold_idx, fold_subjects in enumerate(folds):
            fold_num = fold_idx + 1
            logger.info(f"\n{'='*30} Starting Fold {fold_num}/{num_folds} {'='*30}")

            # Create fold-specific output directory
            fold_output_dir = os.path.join(base_output_dir, f"fold_{fold_num}")
            os.makedirs(fold_output_dir, exist_ok=True)

            # Prepare config for this specific fold
            fold_config = prepare_run_config(
                settings,
                hyperparameters,
                fold_subjects=fold_subjects, # Pass (train, val, test) tuple for the fold
                fold_output_dir=fold_output_dir
            )

            # Run training for this fold
            fold_result = run_training_instance(fold_config)
            all_fold_results.append(fold_result if fold_result else {}) # Store results, handle None

            logger.info(f"{'='*30} Finished Fold {fold_num}/{num_folds} {'='*30}\n")
            # Optional: Add garbage collection or cache clearing if memory is tight between folds
            # import gc
            # gc.collect()
            # if torch.cuda.is_available(): torch.cuda.empty_cache()


        # --- Aggregate and Report K-Fold Results ---
        logger.info(f"\n{'='*30} K-Fold Run Summary {'='*30}")
        aggregated_metrics = defaultdict(list)
        valid_fold_count = 0

        for i, res in enumerate(all_fold_results):
            if res and isinstance(res, dict): # Check if results were returned and are dicts
                valid_fold_count += 1
                logger.info(f"Fold {i+1} Results: {res}")
                for key, value in res.items():
                     # Only aggregate numeric metrics
                     if isinstance(value, (int, float)) and not isinstance(value, bool):
                          # Exclude epoch/iteration counts unless desired
                          if key not in ["epoch", "training_iteration"]:
                             aggregated_metrics[key].append(value)
            else:
                 logger.warning(f"Fold {i+1} did not return valid results.")

        if valid_fold_count > 0:
            logger.info("\nAggregated Metrics (Mean +/- Std):")
            summary_results = {}
            for key, values in aggregated_metrics.items():
                if len(values) > 0: # Ensure we have values to aggregate
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    logger.info(f"  {key}: {mean_val:.4f} +/- {std_val:.4f}  (from {len(values)} folds)")
                    summary_results[key] = {'mean': mean_val, 'std': std_val, 'values': values}
                else:
                     logger.warning(f" Metric '{key}' had no valid values across folds.")

            # Save aggregated results
            summary_save_path = os.path.join(base_output_dir, 'kfold_summary_results.json')
            try:
                 with open(summary_save_path, 'w') as f:
                     # Use the same NumpyEncoder
                     class NumpyEncoder(json.JSONEncoder):
                         def default(self, obj):
                             if isinstance(obj, np.integer): return int(obj)
                             elif isinstance(obj, np.floating): return float(obj)
                             elif isinstance(obj, np.ndarray): return obj.tolist()
                             return super(NumpyEncoder, self).default(obj)
                     json.dump(summary_results, f, indent=2, cls=NumpyEncoder)
                 logger.info(f"Saved K-Fold summary results to {summary_save_path}")
            except Exception as e:
                 logger.error(f"Failed to save K-Fold summary results: {e}")
        else:
             logger.error("No valid results obtained from any K-Fold run.")

    logger.info("--- Standalone test script finished ---")


if __name__ == "__main__":
    main()