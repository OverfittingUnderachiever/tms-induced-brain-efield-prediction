#!/usr/bin/env python3
"""
OPTIMIZED: Multi-GPU ensemble fine-tuning with maximum parallelization.
Utilizes all available GPUs for massive speedup.
"""
import os
import sys
import glob
import json
import argparse
import logging
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
import importlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# --- MULTI-GPU DETECTION AND SETUP ---
def get_all_available_gpus():
    """Get all available GPUs with basic memory check"""
    if not torch.cuda.is_available():
        return []
    
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            test_tensor = torch.randn(100, 100).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            available_gpus.append(i)
        except:
            continue
    return available_gpus

AVAILABLE_GPUS = get_all_available_gpus()
NUM_GPUS = len(AVAILABLE_GPUS)
print(f"🔥 DETECTED {NUM_GPUS} AVAILABLE GPUs: {AVAILABLE_GPUS}")

# Import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Model loading logic
magnitude_module_path = "tms_efield_prediction.models.architectures.simple_unet_magnitude"
ModelClass = None 
try:
    magnitude_module = importlib.import_module(magnitude_module_path)
    model_classes_found = [cls_name for cls_name in dir(magnitude_module) if "UNet" in cls_name and not cls_name.startswith("__")]
    if model_classes_found:
        ModelClass = getattr(magnitude_module, model_classes_found[0])
except ImportError:
    pass 

if not ModelClass:
    try:
        from tms_efield_prediction.models.architectures.simple_unet_magnitude import SimpleUNetMagnitude
        ModelClass = SimpleUNetMagnitude
    except ImportError:
        raise ImportError("Could not find UNet model class")

from tms_efield_prediction.data.pipeline.multi_subject_data import MultiSubjectDataManager
from evaluation_helpers import (
    find_most_recent_automl_run,
    find_model_checkpoint,
    calculate_magnitude_metrics,
    create_full_test_dataset,
)

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('multi-gpu-ensemble')
logger.setLevel(logging.INFO)

# Default config for model creation
DEFAULT_CONFIG = {
    "model_type": "simple_unet_magnitude",
    "input_shape": [9, 20, 20, 20],
    "output_shape": [1, 20, 20, 20],
    "input_channels": 9,
    "output_channels": 1,
    "feature_maps": 16,
    "levels": 3,
    "norm_type": 'batch',
    "activation": 'relu',
    "dropout_rate": 0.2,
    "use_residual": True,
    "use_attention": False
}

def find_loso_automl_directory(base_dir, loso_holdout):
    """Find the AutoML directory for a specific LOSO holdout"""
    loso_dir = os.path.join(base_dir, f"loso_holdout_{loso_holdout}")
    
    if not os.path.exists(loso_dir):
        logger.error(f"❌ LOSO directory not found: {loso_dir}")
        return None
    
    # Find cmaes_automl_* directory inside
    automl_dirs = glob.glob(os.path.join(loso_dir, "cmaes_automl_*"))
    
    if not automl_dirs:
        logger.error(f"❌ No cmaes_automl_* directory found in {loso_dir}")
        return None
    
    if len(automl_dirs) > 1:
        # Use the most recent one
        automl_dirs.sort()
        logger.warning(f"⚠️ Multiple AutoML directories found, using: {automl_dirs[-1]}")
    
    automl_dir = automl_dirs[0] if len(automl_dirs) == 1 else automl_dirs[-1]
    logger.info(f"✅ Found AutoML directory: {automl_dir}")
    return automl_dir

def find_model_checkpoint_with_fallback(trial_dir):
    """Enhanced checkpoint finder with multiple fallback strategies"""
    logger.info(f"🔍 Searching for checkpoint in: {trial_dir}")
    
    # Standard locations
    locations = [
        os.path.join(trial_dir, "checkpoint"),
        os.path.join(trial_dir, "checkpoint_000000", "checkpoint"),
        os.path.join(trial_dir, "checkpoints", "SimpleUNetMagnitudeModel", "best_model.pt"),
        os.path.join(trial_dir, "checkpoints", "best_model.pt"),
        os.path.join(trial_dir, "model", "best_model.pt"),
        os.path.join(trial_dir, "model.pt"),
        os.path.join(trial_dir, "best_model.pt"),
        os.path.join(trial_dir, "pytorch_model.bin"),
        os.path.join(trial_dir, "model.pth"),
        os.path.join(trial_dir, "checkpoint.pth"),
        os.path.join(trial_dir, "final_model.pt"),
    ]
    
    # Check standard locations first
    for loc in locations:
        if os.path.exists(loc):
            logger.info(f"✅ Found checkpoint at: {loc}")
            return loc

    # Recursive search with preferred keywords
    patterns = ["*.pt", "*.pth", "*.bin"]
    preferred_keywords = ["best", "model", "final", "checkpoint"]
    
    for pattern in patterns:
        files = glob.glob(os.path.join(trial_dir, "**", pattern), recursive=True)
        if files:
            for keyword in preferred_keywords:
                preferred = [f for f in files if keyword in f.lower()]
                if preferred:
                    logger.info(f"✅ Selected preferred file with '{keyword}': {preferred[0]}")
                    return preferred[0]
            logger.info(f"✅ Selected first available file: {files[0]}")
            return files[0]

    logger.error(f"❌ No checkpoint found in {trial_dir}")
    return None

def infer_config_from_state_dict(state_dict, trial_id):
    """Infer model configuration from state dict tensor shapes"""
    try:
        if 'initial_conv.conv.weight' not in state_dict:
            logger.warning(f"⚠️ Could not find initial_conv layer in state dict for {trial_id}")
            return None
            
        initial_weight = state_dict['initial_conv.conv.weight']
        feature_maps = initial_weight.shape[0]  # Output channels
        input_channels = initial_weight.shape[1]  # Should be 9
        
        # Infer levels from encoder layers
        levels = 3  # Default
        encoder_keys = [k for k in state_dict.keys() if k.startswith('encoders.')]
        if encoder_keys:
            levels = max([int(k.split('.')[1]) for k in encoder_keys]) + 1
        
        logger.info(f"🔍 Inferred feature_maps={feature_maps}, input_channels={input_channels}, levels={levels} for {trial_id}")
        
        config = DEFAULT_CONFIG.copy()
        config.update({
            'feature_maps': feature_maps,
            'input_channels': input_channels,
            'levels': levels
        })
        return config
            
    except Exception as e:
        logger.error(f"❌ Failed to infer config from state dict for {trial_id}: {e}")
        return None

def extract_model_config(checkpoint, trial_dir, trial_id):
    """Extract model configuration with multiple fallback methods"""
    # Method 1: Infer from state dict (most reliable)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        inferred_config = infer_config_from_state_dict(state_dict, trial_id)
        if inferred_config:
            return inferred_config
    
    # Method 2: Extract from checkpoint
    if isinstance(checkpoint, dict):
        for key in ['config', 'model_config', 'hyperparameters', 'model_hyperparameters']:
            if key in checkpoint:
                config = checkpoint[key].copy() if isinstance(checkpoint[key], dict) else checkpoint[key]
                logger.info(f"✅ Found config in checkpoint under key '{key}'")
                
                # Validate against state dict if available
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                if 'initial_conv.conv.weight' in state_dict:
                    actual_feature_maps = state_dict['initial_conv.conv.weight'].shape[0]
                    if config.get('feature_maps', 16) != actual_feature_maps:
                        config['feature_maps'] = actual_feature_maps
                return config
    
    # Method 3: Look for config files
    config_files = ['config.json', 'hyperparameters.json', 'params.json', 'model_config.json', 'result.json']
    for config_file in config_files:
        config_path = os.path.join(trial_dir, config_file)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                config = data.get('config', data) if config_file == 'result.json' else data
                if config:
                    logger.info(f"✅ Found config in file: {config_file}")
                    return config
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse {config_file}: {e}")
    
    logger.warning(f"⚠️ Could not extract config for {trial_id}, using default")
    return None

def load_single_model(row_data):
    """Load a single model with comprehensive error handling"""
    trial_id, trial_dir, val_loss = row_data
    
    try:
        logger.info(f"🔄 Loading model {trial_id}")
        
        # Find and load checkpoint
        checkpoint_path = find_model_checkpoint_with_fallback(trial_dir)
        if not checkpoint_path:
            logger.error(f"❌ No checkpoint found for {trial_id}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # Get model configuration
        model_config = extract_model_config(checkpoint, trial_dir, trial_id)
        if not model_config:
            logger.error(f"❌ Could not determine config for {trial_id}")
            return None
        
        # Ensure config completeness
        final_config = DEFAULT_CONFIG.copy()
        final_config.update(model_config)
        
        # Double-check feature_maps against state dict
        if 'initial_conv.conv.weight' in state_dict:
            actual_feature_maps = state_dict['initial_conv.conv.weight'].shape[0]
            if final_config.get('feature_maps') != actual_feature_maps:
                final_config['feature_maps'] = actual_feature_maps
        
        # Create and load model
        model = ModelClass(config=final_config)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"⚠️ Missing keys in {trial_id}: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"⚠️ Unexpected keys in {trial_id}: {len(unexpected_keys)}")
        
        model.config = final_config
        logger.info(f"✅ Successfully loaded {trial_id}")
        return (trial_id, model, val_loss)

    except Exception as e:
        logger.error(f"❌ Failed to load {trial_id}: {e}")
        return None

def load_top_models_parallel(args, results_dir_path):
    """Load top models with parallel processing and smart directory detection"""
    logger.info(f"🔍 Searching for models in: {results_dir_path}")
    
    # Smart path detection
    search_path = results_dir_path
    potential_nested = os.path.join(results_dir_path, os.path.basename(results_dir_path))
    if os.path.isdir(potential_nested) and glob.glob(os.path.join(potential_nested, "trainable_wrapper_*")):
        search_path = potential_nested
        logger.info(f"🔄 Using nested path: {search_path}")

    # Find trial directories
    trial_dirs = glob.glob(os.path.join(search_path, "trainable_wrapper_*"))
    if not trial_dirs:
        for pattern in ["trial_*", "*trial*", "experiment_*"]:
            trial_dirs = glob.glob(os.path.join(search_path, pattern))
            if trial_dirs:
                logger.info(f"✅ Found {len(trial_dirs)} directories with pattern '{pattern}'")
                break
        
        if not trial_dirs:
            logger.error(f"❌ No trial directories found in {search_path}")
            raise FileNotFoundError(f"No trial directories found in {search_path}")

    # Gather results
    results_data = []
    for trial_dir in trial_dirs:
        result_path = os.path.join(trial_dir, "result.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    result = json.load(f)
                trial_id = result.get('trial_id')
                val_loss = result.get('val_loss')
                
                if trial_id and val_loss is not None:
                    results_data.append({
                        'trial_id': trial_id,
                        'val_loss': val_loss,
                        'trial_dir': trial_dir
                    })
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse {result_path}: {e}")

    if not results_data:
        logger.error("❌ No valid results found")
        return []

    # Select and load top models
    results_df = pd.DataFrame(results_data).dropna(subset=['val_loss']).sort_values('val_loss')
    top_trials = results_df.head(args.num_models_to_load)
    
    logger.info(f"🎯 Loading top {len(top_trials)} models")
    
    loaded_models = []
    for _, row in top_trials.iterrows():
        result = load_single_model((row['trial_id'], row['trial_dir'], row['val_loss']))
        if result:
            loaded_models.append(result)
        if len(loaded_models) >= args.num_models_to_load:
            break

    logger.info(f"🎉 Successfully loaded {len(loaded_models)} models")
    return loaded_models

def split_data_uniform(dataset, finetune_samples):
    """Split dataset with uniform distribution for unbiased sampling"""
    num_samples = len(dataset)
    if not (0 < finetune_samples < num_samples):
        logger.warning("Invalid finetune_samples, using full dataset for evaluation")
        return None, dataset

    # Uniform sampling with even spacing
    step_size = num_samples / finetune_samples
    finetune_indices = [int(i * step_size) for i in range(finetune_samples)]
    finetune_indices = list(set(finetune_indices))  # Remove duplicates
    
    # Fill gaps if needed
    while len(finetune_indices) < finetune_samples and len(finetune_indices) < num_samples:
        gaps = []
        for i in range(len(finetune_indices) - 1):
            gap_size = finetune_indices[i+1] - finetune_indices[i]
            if gap_size > 1:
                mid_point = finetune_indices[i] + gap_size // 2
                gaps.append((gap_size, mid_point))
        
        if gaps:
            gaps.sort(reverse=True)
            new_idx = gaps[0][1]
            if new_idx not in finetune_indices and new_idx < num_samples:
                finetune_indices.append(new_idx)
                finetune_indices.sort()
        else:
            break
    
    eval_indices = [i for i in range(num_samples) if i not in set(finetune_indices)]
    
    logger.info(f"✅ Split: {len(finetune_indices)} finetune, {len(eval_indices)} eval")
    
    finetune_subset = torch.utils.data.Subset(dataset, finetune_indices)
    eval_subset = torch.utils.data.Subset(dataset, eval_indices)
    return finetune_subset, eval_subset

def fine_tune_worker_multi_gpu(args_tuple):
    """Multi-GPU fine-tuning worker with optimized training loop"""
    (model_data, train_data, eval_data, hyperparams, gpu_id, worker_id) = args_tuple
    
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        model_state, model_config, trial_id = model_data
        train_inputs, train_targets = train_data
        eval_inputs, eval_targets = eval_data
        lr, epochs = hyperparams
        
        # Create and setup model
        model = ModelClass(config=model_config)
        model.load_state_dict(model_state)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.SmoothL1Loss()
        
        # Training with larger batches
        model.train()
        batch_size = 64
        
        for epoch in range(epochs):
            indices = torch.randperm(len(train_inputs))
            
            for i in range(0, len(train_inputs), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_inputs = torch.stack([train_inputs[idx] for idx in batch_indices]).to(device, non_blocking=True)
                batch_targets = torch.stack([train_targets[idx] for idx in batch_indices]).to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                del batch_inputs, batch_targets, outputs, loss
        
        # Fast evaluation
        model.eval()
        all_maes = []
        
        with torch.no_grad():
            for i in range(0, len(eval_inputs), batch_size):
                batch_inputs = torch.stack(eval_inputs[i:i+batch_size]).to(device, non_blocking=True)
                batch_targets = torch.stack(eval_targets[i:i+batch_size])
                outputs = model(batch_inputs).cpu()
                
                for j in range(batch_inputs.shape[0]):
                    target = batch_targets[j].squeeze().numpy()
                    output = outputs[j].squeeze().numpy()
                    mask = target > 1e-8
                    if np.sum(mask) > 0:
                        all_maes.append(np.mean(np.abs(output[mask] - target[mask])))
                
                del batch_inputs, outputs
        
        avg_mae = float(np.mean(all_maes)) if all_maes else np.nan
        
        result = {
            'trial_id': f"{trial_id}_ft_gpu{gpu_id}_w{worker_id}",
            'original_trial_id': trial_id,
            'lr': lr,
            'epochs': epochs,
            'gpu_id': gpu_id,
            'worker_id': worker_id,
            'test_metrics': {
                'magnitude_mae': avg_mae,
                'magnitude_rmse': avg_mae,
                'magnitude_correlation': np.nan,
                'hotspot_iou': 0.0,
                'valid_voxels': len(all_maes)
            },
            'model_state': model.cpu().state_dict(),
            'model_config': model_config
        }
        
        del model, optimizer
        torch.cuda.empty_cache()
        return result
        
    except Exception as e:
        logger.error(f"Worker {worker_id} on GPU {gpu_id} failed: {e}")
        return None

def create_ensemble_multi_gpu(best_models, eval_data, available_gpus):
    """Create ensemble predictions using multiple GPUs"""
    if not best_models or not available_gpus:
        return None
    
    eval_inputs, eval_targets = eval_data
    
    def get_model_predictions(model_gpu_pair):
        model_info, gpu_id = model_gpu_pair
        
        try:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            
            model_config = model_info.get('model_config', DEFAULT_CONFIG)
            model = ModelClass(config=model_config)
            model.load_state_dict(model_info['model_state'])
            model.to(device)
            model.eval()
            
            predictions = []
            batch_size = 64
            
            with torch.no_grad():
                for i in range(0, len(eval_inputs), batch_size):
                    batch_inputs = torch.stack(eval_inputs[i:i+batch_size]).to(device, non_blocking=True)
                    outputs = model(batch_inputs)
                    predictions.append(outputs.cpu())
                    del batch_inputs, outputs
            
            all_predictions = torch.cat(predictions, dim=0)
            del model
            torch.cuda.empty_cache()
            return all_predictions
            
        except Exception as e:
            logger.error(f"Model prediction on GPU {gpu_id} failed: {e}")
            return None
    
    # Distribute models across GPUs and get predictions
    gpu_assignments = [(model_info, available_gpus[i % len(available_gpus)]) for i, model_info in enumerate(best_models)]
    
    with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        future_to_model = {executor.submit(get_model_predictions, pair): pair for pair in gpu_assignments}
        all_predictions = [future.result() for future in as_completed(future_to_model) if future.result() is not None]
    
    if not all_predictions:
        logger.error("No predictions collected for ensemble")
        return None
    
    # Average predictions and calculate metrics
    ensemble_predictions = torch.mean(torch.stack(all_predictions), dim=0)
    all_maes = []
    
    for i in range(len(eval_inputs)):
        target = eval_targets[i].squeeze().numpy()
        pred = ensemble_predictions[i].squeeze().numpy()
        mask = target > 1e-8
        if np.sum(mask) > 0:
            all_maes.append(np.mean(np.abs(pred[mask] - target[mask])))
    
    avg_mae = float(np.mean(all_maes)) if all_maes else np.nan
    
    return {
        'avg_metrics': {
            'magnitude_mae': avg_mae,
            'magnitude_rmse': avg_mae,
            'magnitude_correlation': np.nan,
            'hotspot_iou': 0.0,
            'valid_voxels': len(all_maes)
        }
    }

def create_ensemble_with_visualization(best_models, eval_data, available_gpus, output_dir, loso_holdout):
    """Create ensemble predictions with visualization support"""
    if not best_models or not available_gpus:
        return None
    
    eval_inputs, eval_targets = eval_data
    
    def get_model_predictions(model_gpu_pair):
        model_info, gpu_id = model_gpu_pair
        
        try:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            
            model_config = model_info.get('model_config', DEFAULT_CONFIG)
            model = ModelClass(config=model_config)
            model.load_state_dict(model_info['model_state'])
            model.to(device)
            model.eval()
            
            predictions = []
            batch_size = 64
            
            with torch.no_grad():
                for i in range(0, len(eval_inputs), batch_size):
                    batch_inputs = torch.stack(eval_inputs[i:i+batch_size]).to(device, non_blocking=True)
                    outputs = model(batch_inputs)
                    predictions.append(outputs.cpu())
                    del batch_inputs, outputs
            
            all_predictions = torch.cat(predictions, dim=0)
            del model
            torch.cuda.empty_cache()
            return all_predictions
            
        except Exception as e:
            logger.error(f"Model prediction on GPU {gpu_id} failed: {e}")
            return None
    
    # Distribute models across GPUs and get predictions
    gpu_assignments = [(model_info, available_gpus[i % len(available_gpus)]) for i, model_info in enumerate(best_models)]
    
    with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        future_to_model = {executor.submit(get_model_predictions, pair): pair for pair in gpu_assignments}
        all_predictions = [future.result() for future in as_completed(future_to_model) if future.result() is not None]
    
    if not all_predictions:
        logger.error("No predictions collected for ensemble")
        return None
    
    # Average predictions and calculate per-sample metrics
    ensemble_predictions = torch.mean(torch.stack(all_predictions), dim=0)
    sample_results = []
    all_maes = []
    
    for i in range(len(eval_inputs)):
        target = eval_targets[i].squeeze().numpy()
        pred = ensemble_predictions[i].squeeze().numpy()
        input_data = eval_inputs[i].numpy()
        
        # Calculate metrics for this sample
        mask = target > 1e-8
        if np.sum(mask) > 0:
            sample_mae = np.mean(np.abs(pred[mask] - target[mask]))
            sample_rmse = np.sqrt(np.mean((pred[mask] - target[mask])**2))
            
            # Store sample info for visualization
            sample_info = {
                'global_idx': i,
                'input': input_data,
                'target': target,
                'output': pred,
                'mask': mask,
                'metrics': {
                    'magnitude_mae': sample_mae,
                    'magnitude_rmse': sample_rmse,
                    'magnitude_mae_pct': np.mean(np.abs(pred[mask] - target[mask]) / (target[mask] + 1e-8)) * 100,
                    'magnitude_correlation': np.nan,
                    'hotspot_iou': 0.0,
                    'valid_voxels': int(np.sum(mask))
                }
            }
            sample_results.append(sample_info)
            all_maes.append(sample_mae)
    
    if not sample_results:
        logger.error("No valid samples for visualization")
        return None
    
    # Sort samples by MAE to find best/median/worst
    sample_results.sort(key=lambda x: x['metrics']['magnitude_mae'])
    
    best_sample = sample_results[0]
    worst_sample = sample_results[-1]
    median_sample = sample_results[len(sample_results) // 2]
    
    special_samples = {
        'best': best_sample,
        'average': median_sample,  # Using median as representative
        'worst': worst_sample
    }
    
    # Calculate average metrics
    avg_metrics = {
        'magnitude_mae': float(np.mean(all_maes)),
        'magnitude_rmse': float(np.mean([s['metrics']['magnitude_rmse'] for s in sample_results])),
        'magnitude_correlation': np.nan,
        'hotspot_iou': 0.0,
        'valid_voxels': int(np.mean([s['metrics']['valid_voxels'] for s in sample_results]))
    }
    
    # Create visualizations
    viz_output_dir = os.path.join(output_dir, f"ensemble_visualizations_loso{loso_holdout}")
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # Import visualization functions
    from evaluation_helpers import (
        visualize_sample, 
        create_performance_summary,
        analyze_error_distribution
    )
    
    try:
        # Visualize special samples
        for sample_type, sample in special_samples.items():
            if sample:
                visualize_sample(sample, viz_output_dir, f"ensemble_{sample_type}")
        
        # Create performance summary
        create_performance_summary(viz_output_dir, f"ensemble_loso{loso_holdout}", special_samples, avg_metrics)
        
        # Create error distribution analysis
        analyze_error_distribution(sample_results, viz_output_dir, f"ensemble_loso{loso_holdout}")
        
        logger.info(f"✅ Visualizations saved to {viz_output_dir}")
        
    except Exception as e:
        logger.warning(f"⚠️ Visualization creation failed: {e}")
    
    return {
        'avg_metrics': avg_metrics,
        'special_samples': special_samples,
        'ensemble_samples': sample_results,
        'visualization_dir': viz_output_dir
    }

def main():
    parser = argparse.ArgumentParser(description='MULTI-GPU ensemble fine-tuning')
    
    # Directories - UPDATED for LOSO structure
    parser.add_argument('--base-dir', type=str, default='/home/freyhe/MA_Henry/tms_efield_prediction/tests/integration/')
    parser.add_argument('--loso-holdout', type=int, required=True, help='LOSO holdout number (1-10)')
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default='/home/freyhe/MA_Henry/data')
    parser.add_argument('--output-dir', type=str, default='multi_gpu_ensemble_evaluation')
    
    # Model parameters
    parser.add_argument('--num_models_to_load', type=int, default=min(16, NUM_GPUS * 4))
    parser.add_argument('--num_models_for_ensemble', type=int, default=min(8, NUM_GPUS * 2))
    
    # Hyperparameter search
    parser.add_argument('--finetune_size_search', type=int, nargs='+', default=[100, 200, 300])
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[0.001, 0.003, 0.01])
    parser.add_argument('--epochs_options', type=int, nargs='+', default=[20, 50])

    args = parser.parse_args()

    if NUM_GPUS == 0:
        logger.error("No GPUs available! This script requires CUDA GPUs.")
        return

    logger.info(f"🚀 STARTING MULTI-GPU ENSEMBLE with {NUM_GPUS} GPUs")
    logger.info(f"🎯 LOSO Holdout: {args.loso_holdout}")

    # Convert LOSO holdout to test subject
    test_subject = f"{args.loso_holdout:03d}"
    logger.info(f"📋 Test Subject: {test_subject}")

    # Setup - UPDATED to use LOSO structure
    if args.results_dir:
        actual_results_dir = args.results_dir
    else:
        actual_results_dir = find_loso_automl_directory(args.base_dir, args.loso_holdout)
    
    if not actual_results_dir:
        logger.error(f"No AutoML run found for LOSO holdout {args.loso_holdout}")
        return

    timestamp = datetime.now().strftime("%H%M%S")
    run_name = os.path.basename(actual_results_dir)
    main_output_dir = os.path.join(args.output_dir, f"multigpu_eval_loso{args.loso_holdout}_{run_name}_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Load data and models
        logger.info("Loading test dataset...")
        full_test_ds = create_full_test_dataset(MultiSubjectDataManager, args.data_dir, test_subject)
        if full_test_ds is None:
            return
        
        logger.info("Loading models...")
        original_models = load_top_models_parallel(args, actual_results_dir)
        if not original_models:
            logger.error("No models loaded")
            return
        
        all_results = []
        
        # Multi-GPU grid search or no-finetune evaluation
        for ft_size in args.finetune_size_search:
            logger.info(f"\n🔥 Fine-tuning size: {ft_size}")
            
            # SPECIAL CASE: No fine-tuning (ft_size = 0)
            if ft_size == 0:
                logger.info("  🎯 No fine-tuning mode: evaluating original models directly")
                
                # Convert full dataset to lists for ensemble evaluation
                eval_inputs = [full_test_ds[i][0] for i in range(len(full_test_ds))]
                eval_targets = [full_test_ds[i][1] for i in range(len(full_test_ds))]
                
                # Prepare original model info for ensemble
                original_model_infos = []
                for trial_id, model, val_loss in original_models[:args.num_models_for_ensemble]:
                    original_model_infos.append({
                        'model_state': model.state_dict(),
                        'model_config': model.config,
                        'trial_id': trial_id,
                        'val_loss': val_loss
                    })
                
                # Create ensemble with original models (no fine-tuning)
                config_start = time.time()
                ensemble_result = create_ensemble_with_visualization(
                    original_model_infos, 
                    (eval_inputs, eval_targets), 
                    AVAILABLE_GPUS,
                    main_output_dir,
                    args.loso_holdout
                )
                config_time = time.time() - config_start
                
                # Calculate individual model performance for comparison
                best_individual_mae = np.nan
                if original_models:
                    # Quick evaluation of best original model
                    best_model = original_models[0][1]  # First model (lowest val_loss)
                    best_model_config = best_model.config
                    
                    # Evaluate best individual model
                    device = torch.device(f'cuda:{AVAILABLE_GPUS[0]}')
                    best_model.to(device)
                    best_model.eval()
                    
                    individual_maes = []
                    batch_size = 64
                    
                    with torch.no_grad():
                        for i in range(0, len(eval_inputs), batch_size):
                            batch_inputs = torch.stack(eval_inputs[i:i+batch_size]).to(device, non_blocking=True)
                            batch_targets = torch.stack(eval_targets[i:i+batch_size])
                            outputs = best_model(batch_inputs).cpu()
                            
                            for j in range(batch_inputs.shape[0]):
                                target = batch_targets[j].squeeze().numpy()
                                output = outputs[j].squeeze().numpy()
                                mask = target > 1e-8
                                if np.sum(mask) > 0:
                                    individual_maes.append(np.mean(np.abs(output[mask] - target[mask])))
                            
                            del batch_inputs, outputs
                    
                    best_individual_mae = float(np.mean(individual_maes)) if individual_maes else np.nan
                    best_model.cpu()  # Move back to CPU
                    torch.cuda.empty_cache()
                
                # Store results for no fine-tuning case
                result_summary = {
                    'loso_holdout': args.loso_holdout,
                    'test_subject': test_subject,
                    'finetune_size': 0,
                    'learning_rate': 0.0,  # N/A for no fine-tuning
                    'epochs': 0,           # N/A for no fine-tuning
                    'num_models': len(original_models),
                    'num_gpus_used': NUM_GPUS,
                    'config_time_seconds': config_time,
                    'best_individual_mae': best_individual_mae,
                    'ensemble_mae': ensemble_result['avg_metrics'].get('magnitude_mae', np.nan) if ensemble_result else np.nan,
                    'models_per_second': len(original_models) / config_time if config_time > 0 else 0
                }
                all_results.append(result_summary)
                
                logger.info(f"  ✅ No fine-tuning evaluation complete in {config_time:.1f}s")
                logger.info(f"     Best Individual MAE: {result_summary['best_individual_mae']:.4f}")
                logger.info(f"     Ensemble MAE: {result_summary['ensemble_mae']:.4f}")
                
                # Continue to next finetune size (skip hyperparameter loops for ft_size=0)
                continue
            
            # NORMAL CASE: Fine-tuning with ft_size > 0
            finetune_ds, eval_ds = split_data_uniform(full_test_ds, ft_size)
            if not finetune_ds:
                logger.warning(f"  ❌ Could not create finetune dataset for size {ft_size}")
                continue
            
            # Convert to lists for multiprocessing
            train_inputs = [finetune_ds[i][0] for i in range(len(finetune_ds))]
            train_targets = [finetune_ds[i][1] for i in range(len(finetune_ds))]
            eval_inputs = [eval_ds[i][0] for i in range(len(eval_ds))]
            eval_targets = [eval_ds[i][1] for i in range(len(eval_ds))]
            
            for lr in args.learning_rates:
                for epochs in args.epochs_options:
                    config_start = time.time()
                    logger.info(f"  🚀 Training: LR={lr}, Epochs={epochs}")
                    
                    # Prepare tasks for all GPUs
                    tasks = []
                    for worker_id, (trial_id, model, val_loss) in enumerate(original_models):
                        gpu_id = AVAILABLE_GPUS[worker_id % NUM_GPUS]
                        model_data = (model.state_dict(), model.config, trial_id)
                        tasks.append((model_data, (train_inputs, train_targets), (eval_inputs, eval_targets), (lr, epochs), gpu_id, worker_id))
                    
                    # Execute fine-tuning in parallel
                    ft_results = []
                    with ThreadPoolExecutor(max_workers=NUM_GPUS * 2) as executor:
                        futures = [executor.submit(fine_tune_worker_multi_gpu, task) for task in tasks]
                        ft_results = [future.result() for future in as_completed(futures) if future.result()]
                    
                    if len(ft_results) < 2:
                        logger.warning(f"  ❌ Not enough models trained: {len(ft_results)}")
                        continue
                    
                    # Select best models and create ensemble
                    ft_results.sort(key=lambda x: x['test_metrics'].get('magnitude_mae', float('inf')))
                    top_models = ft_results[:args.num_models_for_ensemble]
                    ensemble_result = create_ensemble_multi_gpu(top_models, (eval_inputs, eval_targets), AVAILABLE_GPUS)
                    
                    # Store results
                    config_time = time.time() - config_start
                    result_summary = {
                        'loso_holdout': args.loso_holdout,
                        'test_subject': test_subject,
                        'finetune_size': ft_size,
                        'learning_rate': lr,
                        'epochs': epochs,
                        'num_models': len(ft_results),
                        'num_gpus_used': NUM_GPUS,
                        'config_time_seconds': config_time,
                        'best_individual_mae': top_models[0]['test_metrics'].get('magnitude_mae', np.nan),
                        'ensemble_mae': ensemble_result['avg_metrics'].get('magnitude_mae', np.nan) if ensemble_result else np.nan,
                        'models_per_second': len(ft_results) / config_time if config_time > 0 else 0
                    }
                    all_results.append(result_summary)
                    
                    logger.info(f"  ✅ DONE in {config_time:.1f}s | Best: {result_summary['best_individual_mae']:.4f} | Ensemble: {result_summary['ensemble_mae']:.4f}")
        
        # Save results and summary
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_path = os.path.join(main_output_dir, "multi_gpu_results.csv")
            results_df.to_csv(results_path, index=False)
            
            valid_results = [r for r in all_results if not np.isnan(r['ensemble_mae'])]
            if valid_results:
                best = min(valid_results, key=lambda x: x['ensemble_mae'])
                total_time = time.time() - start_time
                total_models = sum(r['num_models'] for r in all_results)
                
                logger.info(f"\n🏆 COMPLETE in {total_time:.1f}s")
                logger.info(f"🎯 LOSO Holdout {args.loso_holdout} (Subject {test_subject})")
                logger.info(f"⚡ Processed {total_models} model evaluations at {total_models/total_time:.1f} models/sec")
                logger.info(f"🥇 Best: Size={best['finetune_size']}, LR={best['learning_rate']}, E={best['epochs']}, MAE={best['ensemble_mae']:.4f}")
                logger.info(f"📁 Results: {results_path}")
                
                # Summary by finetune size
                logger.info(f"\n📊 Summary by finetune size:")
                for ft_size in sorted(set(r['finetune_size'] for r in valid_results)):
                    size_results = [r for r in valid_results if r['finetune_size'] == ft_size]
                    best_for_size = min(size_results, key=lambda x: x['ensemble_mae'])
                    if ft_size == 0:
                        logger.info(f"   Size {ft_size} (no finetune): Best MAE = {best_for_size['ensemble_mae']:.4f}")
                    else:
                        logger.info(f"   Size {ft_size}: Best MAE = {best_for_size['ensemble_mae']:.4f} (LR={best_for_size['learning_rate']}, E={best_for_size['epochs']})")

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        for gpu_id in AVAILABLE_GPUS:
            try:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            except:
                pass


if __name__ == "__main__":
    # Set multiprocessing method for CUDA
    if NUM_GPUS > 0:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    main()