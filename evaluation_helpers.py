# evaluation_helpers.py
"""
Helper functions for model evaluation, including data processing,
metric calculation, and visualization.
"""
import os
import glob
import json
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

logger = logging.getLogger('evaluate-top-models.helpers')

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

def extract_model_config_from_checkpoint(checkpoint, trial_dir, trial_id):
    """Extract the original model configuration from checkpoint or trial directory"""
    
    # Method 1: ALWAYS try shape inference first (most reliable)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        inferred_config = infer_config_from_state_dict(state_dict, trial_id)
        if inferred_config is not None:
            logger.info(f"✅ Inferred config from state dict shapes")
            return inferred_config
    
    # Method 2: Look for config in checkpoint (but validate against shapes)
    if isinstance(checkpoint, dict):
        config_keys = ['config', 'model_config', 'hyperparameters', 'model_hyperparameters']
        for key in config_keys:
            if key in checkpoint:
                config = checkpoint[key]
                logger.info(f"✅ Found config in checkpoint under key '{key}'")
                
                # Validate config against actual state dict shapes
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                if 'initial_conv.conv.weight' in state_dict:
                    actual_feature_maps = state_dict['initial_conv.conv.weight'].shape[0]
                    config_feature_maps = config.get('feature_maps', 16)
                    
                    if actual_feature_maps != config_feature_maps:
                        logger.warning(f"⚠️ Config feature_maps={config_feature_maps} doesn't match actual={actual_feature_maps}")
                        logger.info(f"🔧 Correcting feature_maps to {actual_feature_maps}")
                        config = config.copy()
                        config['feature_maps'] = actual_feature_maps
                
                return config
    
    # Method 3: Look for config files in trial directory
    config_files = ['config.json', 'hyperparameters.json', 'params.json', 'model_config.json']
    for config_file in config_files:
        config_path = os.path.join(trial_dir, config_file)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"✅ Found config in file: {config_file}")
                return config
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse {config_file}: {e}")
    
    # Method 4: Look for result.json which might contain config
    result_path = os.path.join(trial_dir, "result.json")
    if os.path.exists(result_path):
        try:
            with open(result_path, 'r') as f:
                result = json.load(f)
            if 'config' in result:
                logger.info(f"✅ Found config in result.json")
                return result['config']
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse result.json for config: {e}")
    
    logger.warning(f"⚠️ Could not extract config for {trial_id}, using default")
    return None

def infer_config_from_state_dict(state_dict, trial_id):
    """Infer model configuration from state dict tensor shapes"""
    try:
        # Look for the initial conv layer to get feature_maps
        if 'initial_conv.conv.weight' in state_dict:
            # Shape is [out_channels, in_channels, d, h, w]
            initial_weight = state_dict['initial_conv.conv.weight']
            feature_maps = initial_weight.shape[0]  # Output channels
            input_channels = initial_weight.shape[1]  # Should be 9
            
            # Also infer levels from the number of encoder/decoder layers
            levels = 3  # Default
            encoder_keys = [k for k in state_dict.keys() if k.startswith('encoders.')]
            if encoder_keys:
                max_encoder_idx = max([int(k.split('.')[1]) for k in encoder_keys])
                levels = max_encoder_idx + 1
            
            logger.info(f"🔍 Inferred feature_maps={feature_maps}, input_channels={input_channels}, levels={levels} for {trial_id}")
            
            # Create config with inferred parameters
            inferred_config = DEFAULT_CONFIG.copy()
            inferred_config['feature_maps'] = feature_maps
            inferred_config['input_channels'] = input_channels
            inferred_config['levels'] = levels
            
            # Log the key architectural parameters
            logger.info(f"📐 Model architecture: feature_maps={feature_maps}, levels={levels}")
            
            return inferred_config
        else:
            logger.warning(f"⚠️ Could not find initial_conv layer in state dict for {trial_id}")
            # Try to find any conv layer to get some info
            conv_keys = [k for k in state_dict.keys() if 'conv.weight' in k]
            if conv_keys:
                logger.info(f"🔍 Available conv layers: {conv_keys[:5]}...")  # Show first 5
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to infer config from state dict for {trial_id}: {e}")
        return None

def find_model_checkpoint_debug(trial_dir):
    """Enhanced checkpoint finder with debugging"""
    logger.info(f"🔍 Searching for checkpoint in: {trial_dir}")
    
    possible_locations = [
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
    
    logger.info(f"📍 Checking {len(possible_locations)} standard locations...")
    for i, loc in enumerate(possible_locations):
        if os.path.exists(loc):
            logger.info(f"✅ Found checkpoint at standard location {i+1}: {loc}")
            return loc
        else:
            logger.debug(f"❌ Not found: {loc}")

    # Recursive search for .pt, .pth, .bin files
    logger.info(f"🔄 Performing recursive search for model files...")
    checkpoint_patterns = ["*.pt", "*.pth", "*.bin"]
    
    for pattern in checkpoint_patterns:
        checkpoint_files = glob.glob(os.path.join(trial_dir, "**", pattern), recursive=True)
        if checkpoint_files:
            logger.info(f"📦 Found {len(checkpoint_files)} files with pattern '{pattern}':")
            for f in checkpoint_files[:5]:  # Show first 5
                logger.info(f"   - {f}")
            
            # Prefer files with specific keywords
            preferred_keywords = ["best", "model", "final", "checkpoint"]
            for keyword in preferred_keywords:
                preferred_files = [f for f in checkpoint_files if keyword in f.lower()]
                if preferred_files:
                    selected_file = preferred_files[0]
                    logger.info(f"✅ Selected preferred file with '{keyword}': {selected_file}")
                    return selected_file
            
            # Return first file if no preferred found
            selected_file = checkpoint_files[0]
            logger.info(f"✅ Selected first available file: {selected_file}")
            return selected_file

    # If nothing found, list directory contents for debugging
    logger.error(f"❌ No checkpoint files found in {trial_dir}")
    try:
        if os.path.exists(trial_dir):
            contents = []
            for root, dirs, files in os.walk(trial_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, trial_dir)
                    contents.append(rel_path)
            
            logger.info(f"📂 Complete directory structure ({len(contents)} files):")
            for item in contents[:20]:  # Show first 20 items
                logger.info(f"   - {item}")
            if len(contents) > 20:
                logger.info(f"   ... and {len(contents) - 20} more files")
        else:
            logger.error(f"❌ Trial directory does not exist: {trial_dir}")
    except Exception as e:
        logger.error(f"❌ Error listing directory contents: {e}")
    
    return None

def load_single_model(row_data, ModelClass):
    """Load a single model in parallel with detailed error reporting"""
    trial_id, trial_dir, val_loss = row_data
    
    try:
        logger.info(f"🔄 Loading model {trial_id} from {trial_dir}")
        
        # Find checkpoint
        checkpoint_path = find_model_checkpoint_debug(trial_dir)
        if not checkpoint_path:
            logger.error(f"❌ No checkpoint found for {trial_id} in {trial_dir}")
            # List directory contents for debugging
            if os.path.exists(trial_dir):
                contents = os.listdir(trial_dir)
                logger.info(f"📂 Directory contents: {contents}")
            return None

        logger.info(f"📦 Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            logger.info(f"✅ Found model_state_dict in checkpoint")
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            logger.info(f"✅ Using entire checkpoint as state_dict")
        else:
            logger.error(f"❌ Invalid checkpoint format for {trial_id}")
            return None

        # Extract the original model configuration
        model_config = extract_model_config_from_checkpoint(checkpoint, trial_dir, trial_id)
        if model_config is None:
            logger.error(f"❌ Could not determine model configuration for {trial_id}")
            return None
        
        # Ensure all required config keys are present
        final_config = DEFAULT_CONFIG.copy()
        final_config.update(model_config)
        
        # Double-check that feature_maps matches the actual state dict
        if 'initial_conv.conv.weight' in state_dict:
            actual_feature_maps = state_dict['initial_conv.conv.weight'].shape[0]
            config_feature_maps = final_config.get('feature_maps', 16)
            
            if actual_feature_maps != config_feature_maps:
                logger.warning(f"⚠️ Final config mismatch: config={config_feature_maps}, actual={actual_feature_maps}")
                logger.info(f"🔧 Force-correcting feature_maps to {actual_feature_maps}")
                final_config['feature_maps'] = actual_feature_maps
        
        logger.info(f"🏗️ Creating model with feature_maps={final_config.get('feature_maps', 'unknown')}")
        
        # Create model with the correct configuration
        model = ModelClass(config=final_config)
        
        logger.info(f"📥 Loading state dict (strict=False)")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"⚠️ Missing keys in {trial_id}: {len(missing_keys)} keys")
            if len(missing_keys) <= 5:  # Show details for small numbers
                logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"⚠️ Unexpected keys in {trial_id}: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 5:  # Show details for small numbers  
                logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        # Check if there were any size mismatches (these would cause exceptions)
        logger.info(f"✅ Successfully loaded model {trial_id} with feature_maps={final_config.get('feature_maps')}")
        
        # Store the config in the model for later use
        model.config = final_config
        
        return (trial_id, model, val_loss)

    except Exception as e:
        logger.error(f"❌ Failed to load {trial_id}: {e}")
        import traceback
        logger.error(f"📚 Traceback: {traceback.format_exc()}")
        return None

def load_top_models_parallel(args, results_dir_path, ModelClass):
    """PARALLEL: Load models with threading and detailed debugging"""
    logger.info(f"🔍 Searching for models in: {results_dir_path}")
    
    search_path = results_dir_path
    base_name = os.path.basename(results_dir_path)
    potential_nested_path = os.path.join(results_dir_path, base_name)

    if os.path.isdir(potential_nested_path) and glob.glob(os.path.join(potential_nested_path, "trainable_wrapper_*")):
        search_path = potential_nested_path
        logger.info(f"🔄 Using nested path: {search_path}")

    trial_dirs = glob.glob(os.path.join(search_path, "trainable_wrapper_*"))
    logger.info(f"📁 Found {len(trial_dirs)} trial directories")
    
    if not trial_dirs:
        # Try alternative patterns
        alternative_patterns = ["trial_*", "*trial*", "experiment_*"]
        for pattern in alternative_patterns:
            trial_dirs = glob.glob(os.path.join(search_path, pattern))
            if trial_dirs:
                logger.info(f"✅ Found {len(trial_dirs)} directories with pattern '{pattern}'")
                break
        
        if not trial_dirs:
            logger.error(f"❌ No trial directories found in {search_path}")
            logger.info(f"📂 Directory contents: {os.listdir(search_path) if os.path.exists(search_path) else 'PATH DOES NOT EXIST'}")
            raise FileNotFoundError(f"No trial directories found in {search_path}")

    # Quick results gathering with better error handling
    results_data = []
    for trial_dir_path_iter in trial_dirs:
        result_path = os.path.join(trial_dir_path_iter, "result.json")
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
                        'trial_dir': trial_dir_path_iter
                    })
                    logger.info(f"✅ Found valid result: {trial_id} (val_loss: {val_loss:.6f})")
                else:
                    logger.warning(f"⚠️ Invalid result.json in {trial_dir_path_iter}: missing trial_id or val_loss")
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse {result_path}: {e}")
        else:
            logger.warning(f"⚠️ No result.json in {trial_dir_path_iter}")
    
    if not results_data:
        logger.error("❌ No valid results found in any trial directory")
        return []

    # Select top models
    logger.info(f"📊 Sorting {len(results_data)} models by validation loss")
    results_df = pd.DataFrame(results_data).dropna(subset=['val_loss']).sort_values('val_loss')
    top_trials = results_df.head(args.num_models_to_load)
    
    logger.info(f"🎯 Selected top {len(top_trials)} models:")
    for _, row in top_trials.iterrows():
        logger.info(f"   - {row['trial_id']}: val_loss={row['val_loss']:.6f}")

    # SEQUENTIAL MODEL LOADING for better debugging (instead of parallel)
    loaded_models = []
    for _, row in top_trials.iterrows():
        row_data = (row['trial_id'], row['trial_dir'], row['val_loss'])
        result = load_single_model(row_data, ModelClass)
        if result:
            loaded_models.append(result)
        
        # Stop early if we got enough models
        if len(loaded_models) >= args.num_models_to_load:
            break

    logger.info(f"🎉 Successfully loaded {len(loaded_models)} out of {len(top_trials)} attempted models")
    return loaded_models

def split_data_for_finetuning_uniform(full_dataset, finetune_num_samples):
    """
    Splits a dataset into a fine-tuning set (with uniformly distributed samples)
    and an evaluation set. Uses uniform spacing for unbiased selection.
    
    Example: 1000 total samples, 100 finetune samples
    -> selects indices [0, 10, 20, 30, ..., 990]
    -> evaluation gets the remaining 900 samples
    """
    num_samples = len(full_dataset)

    # Check for valid number of finetuning samples
    if not (0 < finetune_num_samples < num_samples):
        logger.error(f"finetune_num_samples ({finetune_num_samples}) must be between 1 and total samples-1 ({num_samples-1}).")
        logger.warning("No fine-tuning will be performed. Using full dataset for evaluation.")
        return None, full_dataset

    # UNIFORM SAMPLING: Select evenly spaced indices
    if finetune_num_samples >= num_samples:
        # If requesting more samples than available, use all
        finetune_indices = list(range(num_samples))
        eval_indices = []
    else:
        # Calculate step size for uniform distribution
        step_size = num_samples / finetune_num_samples
        finetune_indices = [int(i * step_size) for i in range(finetune_num_samples)]
        
        # Ensure no duplicates and within bounds
        finetune_indices = list(set(finetune_indices))
        finetune_indices = [idx for idx in finetune_indices if idx < num_samples]
        finetune_indices.sort()
        
        # If we lost samples due to rounding, add them back uniformly
        while len(finetune_indices) < finetune_num_samples and len(finetune_indices) < num_samples:
            # Find the largest gap and insert a sample there
            gaps = []
            for i in range(len(finetune_indices) - 1):
                gap_size = finetune_indices[i+1] - finetune_indices[i]
                if gap_size > 1:
                    mid_point = finetune_indices[i] + gap_size // 2
                    gaps.append((gap_size, mid_point))
            
            if gaps:
                # Add sample in the middle of the largest gap
                gaps.sort(reverse=True)
                new_idx = gaps[0][1]
                if new_idx not in finetune_indices and new_idx < num_samples:
                    finetune_indices.append(new_idx)
                    finetune_indices.sort()
            else:
                break
        
        # Create eval indices (all remaining samples)
        finetune_set = set(finetune_indices)
        eval_indices = [i for i in range(num_samples) if i not in finetune_set]

    if len(finetune_indices) == 0:
        logger.warning("No samples selected for fine-tuning. This should not happen with valid input.")
        return None, full_dataset

    logger.info(f"✅ Uniform splitting: {len(finetune_indices)} for fine-tuning (every ~{num_samples/len(finetune_indices):.1f} samples), {len(eval_indices)} for evaluation.")
    logger.info(f"📍 Fine-tune indices: {finetune_indices[:10]}{'...' if len(finetune_indices) > 10 else ''}")
    
    finetune_subset = torch.utils.data.Subset(full_dataset, finetune_indices)
    eval_subset = torch.utils.data.Subset(full_dataset, eval_indices)
    return finetune_subset, eval_subset


def find_most_recent_automl_run(base_dir):
    if not os.path.isdir(base_dir):
        logger.error(f"Base directory not found: {base_dir}")
        return None

    patterns_to_check = ["cmaes_automl*", "improved_automl_*"]
    run_dirs = []
    for pattern in patterns_to_check:
        run_dirs.extend(glob.glob(os.path.join(base_dir, pattern)))

    run_dirs = [d for d in run_dirs if os.path.isdir(d)]

    if not run_dirs:
        logger.error(f"No AutoML run directories found in {base_dir} matching patterns: {patterns_to_check}")
        return None

    run_dirs.sort(key=os.path.getmtime)
    most_recent = run_dirs[-1]
    logger.info(f"Found most recent AutoML run: {os.path.basename(most_recent)}")
    return most_recent

def find_model_checkpoint(trial_dir):
    possible_locations = [
        os.path.join(trial_dir, "checkpoint"),
        os.path.join(trial_dir, "checkpoint_000000", "checkpoint"),
        os.path.join(trial_dir, "checkpoints", "SimpleUNetMagnitudeModel", "best_model.pt"),
        os.path.join(trial_dir, "checkpoints", "best_model.pt"),
        os.path.join(trial_dir, "model", "best_model.pt"),
        os.path.join(trial_dir, "model.pt"),
        os.path.join(trial_dir, "best_model.pt")
    ]
    for loc in possible_locations:
        if os.path.exists(loc):
            logger.info(f"Found checkpoint at: {loc}")
            return loc

    checkpoint_files = glob.glob(os.path.join(trial_dir, "**", "*.pt"), recursive=True)
    if checkpoint_files:
        preferred_files = [f for f in checkpoint_files if "best" in f or "model" in f]
        if preferred_files:
            logger.info(f"Found preferred checkpoint through glob search: {preferred_files[0]}")
            return preferred_files[0]
        logger.info(f"Found checkpoint through glob search: {checkpoint_files[0]}")
        return checkpoint_files[0]
    return None

def calculate_magnitude_metrics(prediction, target, mask=None):
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if mask is None:
        mask = target > 1e-8
    elif torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = mask.astype(bool)
    masked_pred = prediction[mask]
    masked_target = target[mask]
    valid_voxels = np.sum(mask)

    if valid_voxels > 0:
        mae = np.mean(np.abs(masked_pred - masked_target))  # <--- THIS WAS MISSING
        mae_pct = np.mean(np.abs(masked_pred - masked_target) / (masked_target + 1e-8)) * 100
        mse = np.mean(np.square(masked_pred - masked_target))
        rmse = np.sqrt(mse)
        correlation = np.nan
        if valid_voxels > 1:
            correlation = np.corrcoef(masked_pred, masked_target)[0, 1]

        percentile = 95
        target_threshold = 0
        if np.any(masked_target[masked_target > 1e-8]):
            target_threshold = np.percentile(masked_target[masked_target > 1e-8], percentile)
        
        pred_threshold = 0
        if np.any(masked_pred[masked_pred > 1e-8]):
            pred_threshold = np.percentile(masked_pred[masked_pred > 1e-8], percentile)

        target_hotspot = (prediction >= target_threshold) & mask
        pred_hotspot = (prediction >= pred_threshold) & mask
        
        intersection = np.logical_and(target_hotspot, pred_hotspot).sum()
        union = np.logical_or(target_hotspot, pred_hotspot).sum()
        hotspot_iou = intersection / union if union > 0 else 0.0

    else:
        mae = np.nan
        mae_pct = np.nan
        rmse = np.nan
        correlation = np.nan
        hotspot_iou = np.nan

    return {
        'magnitude_mae': float(mae), 
        'magnitude_mae_pct': float(mae_pct),
        'magnitude_rmse': float(rmse),
        'magnitude_correlation': float(correlation),
        'hotspot_iou': float(hotspot_iou),
        'valid_voxels': int(valid_voxels)
    }


def create_full_test_dataset(data_manager_class, data_dir, test_subject):
    logger.info(f"Loading full test data for subject {test_subject}...")
    data_manager = data_manager_class(
        data_root_path=data_dir,
        output_shape=(20, 20, 20),
        normalization_method="standard"
    )
    features, targets = data_manager.load_subject_data(test_subject)

    if features is None or targets is None:
        logger.error(f"Failed to load data for test subject {test_subject}")
        return None

    logger.info(f"Loaded {features.shape[0]} samples for test subject {test_subject}")
    logger.info(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
    
    return torch.utils.data.TensorDataset(features, targets)

def split_data_for_finetuning(full_dataset, finetune_num_samples):
    """
    Splits a dataset into a fine-tuning set (with a fixed number of samples)
    and an evaluation set. Uses a random split for unbiased selection.
    """
    num_samples = len(full_dataset)

    # Check for valid number of finetuning samples
    if not (0 < finetune_num_samples < num_samples):
        logger.error(f"finetune_num_samples ({finetune_num_samples}) must be between 1 and total samples-1 ({num_samples-1}).")
        logger.warning("No fine-tuning will be performed. Using full dataset for evaluation.")
        return None, full_dataset

    # Generate a random, reproducible split
    all_indices = np.arange(num_samples)
    np.random.seed(42)  # Use a fixed seed for reproducibility of the split
    np.random.shuffle(all_indices)

    finetune_indices = all_indices[:finetune_num_samples]
    eval_indices = all_indices[finetune_num_samples:]

    if len(finetune_indices) == 0:
        logger.warning("No samples selected for fine-tuning. This should not happen with valid input.")
        return None, full_dataset

    logger.info(f"Splitting data: {len(finetune_indices)} for fine-tuning, {len(eval_indices)} for evaluation.")
    finetune_subset = torch.utils.data.Subset(full_dataset, finetune_indices)
    eval_subset = torch.utils.data.Subset(full_dataset, eval_indices)
    return finetune_subset, eval_subset

# ... (The rest of the helper functions are unchanged) ...
def visualize_sample(sample, output_dir, filename_prefix):
    if sample is None:
        logger.warning(f"Sample is None for {filename_prefix}, skipping visualization.")
        return
        
    input_data = sample['input']    
    target_data = sample['target'] 
    output_data = sample['output']  

    mask = target_data > 1e-8 

    if len(input_data.shape) == 4:
        channels, depth, height, width = input_data.shape
        mid_d, mid_h, mid_w = depth // 2, height // 2, width // 2
        input_slice_axial = input_data[:, mid_d, :, :] 
        input_slice_coronal = input_data[:, :, mid_h, :] 
        input_slice_sagittal = input_data[:, :, :, mid_w]
    else:
        logger.warning(f"Unexpected input shape: {input_data.shape} for {filename_prefix}, skipping.")
        return

    if len(target_data.shape) == 3:
        t_depth, t_height, t_width = target_data.shape
        t_mid_d, t_mid_h, t_mid_w = t_depth // 2, t_height // 2, t_width // 2
        target_slices_dict = {
            'axial': target_data[t_mid_d, :, :], 'coronal': target_data[:, t_mid_h, :], 'sagittal': target_data[:, :, t_mid_w]
        }
        output_slices_dict = {
            'axial': output_data[t_mid_d, :, :], 'coronal': output_data[:, t_mid_h, :], 'sagittal': output_data[:, :, t_mid_w]
        }
        mask_slices_dict = {
            'axial': mask[t_mid_d, :, :], 'coronal': mask[:, t_mid_h, :], 'sagittal': mask[:, :, t_mid_w]
        }
        input_slices_for_display_dict = {
             'axial': input_slice_axial, 'coronal': input_slice_coronal, 'sagittal': input_slice_sagittal
        }
    else:
        logger.warning(f"Unexpected target shape: {target_data.shape} for {filename_prefix}, skipping.")
        return

    orientations = ['axial', 'coronal', 'sagittal']
    fig_cols = 1 + 1 + 1 + 1 + 1 
    
    for orientation in orientations:
        i_slice_display = input_slices_for_display_dict[orientation]
        t_slice = target_slices_dict[orientation]
        o_slice = output_slices_dict[orientation]
        m_slice = mask_slices_dict[orientation]

        fig, axes = plt.subplots(1, fig_cols, figsize=(4 * fig_cols, 4))
        ax_idx = 0

        im = axes[ax_idx].imshow(i_slice_display[0], cmap='gray') 
        axes[ax_idx].set_title(f'Input Ch0 ({orientation})'); axes[ax_idx].axis('off')
        cax = make_axes_locatable(axes[ax_idx]).append_axes("right", size="5%", pad=0.05); plt.colorbar(im, cax=cax)
        ax_idx += 1
        
        im = axes[ax_idx].imshow(m_slice, cmap='binary_r'); axes[ax_idx].set_title('Mask'); axes[ax_idx].axis('off')
        cax = make_axes_locatable(axes[ax_idx]).append_axes("right", size="5%", pad=0.05); plt.colorbar(im, cax=cax, ticks=[0,1])
        ax_idx += 1

        valid_t = t_slice[m_slice]; valid_o = o_slice[m_slice]
        all_vals = np.concatenate([valid_t, valid_o]) if valid_t.size > 0 and valid_o.size > 0 else (valid_t if valid_t.size > 0 else valid_o)
        vmin = np.min(all_vals) if all_vals.size > 0 else 0
        vmax = np.max(all_vals) if all_vals.size > 0 else 0.1
        GLOBAL_VMIN = 0.0
        GLOBAL_VMAX = 1.0
        norm = Normalize(vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)


        im = axes[ax_idx].imshow(t_slice, cmap='viridis', norm=norm); axes[ax_idx].contour(m_slice, levels=[0.5], colors='w', linewidths=0.8, alpha=0.7)
        axes[ax_idx].set_title('Target'); axes[ax_idx].axis('off')
        cax = make_axes_locatable(axes[ax_idx]).append_axes("right", size="5%", pad=0.05); plt.colorbar(im, cax=cax)
        ax_idx += 1

        im = axes[ax_idx].imshow(o_slice, cmap='viridis', norm=norm); axes[ax_idx].contour(m_slice, levels=[0.5], colors='w', linewidths=0.8, alpha=0.7)
        axes[ax_idx].set_title('Prediction'); axes[ax_idx].axis('off')
        cax = make_axes_locatable(axes[ax_idx]).append_axes("right", size="5%", pad=0.05); plt.colorbar(im, cax=cax)
        ax_idx += 1
        
        err_map_pct = np.abs(t_slice - o_slice) / (t_slice + 1e-8) * 100
        masked_err_disp = np.where(m_slice, err_map_pct, np.nan)

        err_max = np.nanmax(masked_err_disp) if np.any(np.isfinite(masked_err_disp)) else 1.0
        if err_max == 0: err_max = 1.0
        im = axes[ax_idx].imshow(masked_err_disp, cmap='hot', vmin=0, vmax=err_max)
        mae_val = sample['metrics'].get('magnitude_mae_pct', np.nan)
        if ax_idx < len(axes):
            axes[ax_idx].set_title(f'Abs. Error (%): {mae_val:.2f}%')

        cax = make_axes_locatable(axes[ax_idx]).append_axes("right", size="5%", pad=0.05); plt.colorbar(im, cax=cax)

        global_idx = sample.get('global_idx', 'N/A')
        fig.suptitle(f"{filename_prefix.replace('_', ' ').title()} | Idx: {global_idx} | MAE: {mae_val:.4f} ({orientation})", fontsize=12)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_{orientation}.png"), dpi=100, bbox_inches='tight')
        plt.close()

def create_performance_summary(output_dir, model_id, special_samples, avg_metrics):
    if special_samples.get('best') is None and special_samples.get('average') is None and special_samples.get('worst') is None:
        logger.warning(f"No special samples for model {model_id}. Skipping performance summary.")
        return None

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"Model {model_id} Perf. Summary (Axial)", fontsize=16, y=0.98)
    gs = plt.GridSpec(4, 3, height_ratios=[0.5, 1, 1, 1], hspace=0.5, wspace=0.3)

    ax_metrics = fig.add_subplot(gs[0, :])
    avg_mae = avg_metrics.get('magnitude_mae', np.nan)
    metrics_text = (f"Avg Metrics: MAE: {avg_mae:.4f} | RMSE: {avg_metrics.get('magnitude_rmse', np.nan):.4f} | "
                    f"Corr: {avg_metrics.get('magnitude_correlation', np.nan):.4f} | IoU: {avg_metrics.get('hotspot_iou', np.nan):.4f}")
    ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=10, bbox=dict(fc='lightgray', alpha=0.5)); ax_metrics.axis('off')

    sample_types = ['best', 'average', 'worst']
    for row_idx, sample_type in enumerate(sample_types, 1):
        sample = special_samples.get(sample_type)
        if sample is None:
            for col_idx in range(3):
                ax = fig.add_subplot(gs[row_idx, col_idx]); ax.text(0.5,0.5, f"{sample_type.capitalize()} N/A"); ax.axis('off')
            continue

        target_data, output_data, mask_data = sample['target'], sample['output'], sample['mask']
        mid_d = target_data.shape[0] // 2
        t_slice, o_slice, m_slice = target_data[mid_d,:,:], output_data[mid_d,:,:], mask_data[mid_d,:,:]

        valid_t = t_slice[m_slice]; valid_o = o_slice[m_slice]
        all_vals = np.concatenate([valid_t, valid_o]) if valid_t.size > 0 and valid_o.size > 0 else (valid_t if valid_t.size > 0 else valid_o)
        vmin = np.min(all_vals) if all_vals.size > 0 else 0
        vmax = np.max(all_vals) if all_vals.size > 0 else 0.1
        if vmin == vmax: vmin -=0.01; vmax += 0.01
        if vmin == vmax: vmax = vmin + 1e-5
        norm = Normalize(vmin=vmin, vmax=vmax)

        titles = [f"{sample_type.capitalize()} Target", f"{sample_type.capitalize()} Pred.", f"Abs. Error (MAE {sample['metrics']['magnitude_mae']:.4f})"]
        data_to_plot = [t_slice, o_slice, np.where(m_slice, np.abs(t_slice - o_slice), np.nan)]
        cmaps = ['viridis', 'viridis', 'hot']
        norms_plot = [norm, norm, Normalize(vmin=0, vmax=np.nanmax(data_to_plot[2]) if np.any(np.isfinite(data_to_plot[2])) else 1.0)]
        if norms_plot[2].vmax == 0: norms_plot[2].vmax = 1.0


        for col_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = ax.imshow(data_to_plot[col_idx], cmap=cmaps[col_idx], norm=norms_plot[col_idx])
            if col_idx < 2: ax.contour(m_slice, levels=[0.5], colors='w', linewidths=0.8, alpha=0.7) # Fixed lw to linewidths
            ax.set_title(titles[col_idx], fontsize=10); ax.axis('off')
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05); plt.colorbar(im, cax=cax)
            
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "performance_summary.png")
    plt.savefig(summary_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved performance summary to {summary_path}")
    return summary_path

def create_cross_model_sample_comparison(results, output_dir):
    if len(results) < 1: return
    os.makedirs(output_dir, exist_ok=True)
    for sample_type in ['best', 'worst', 'average']:
        num_models = len(results)
        fig = plt.figure(figsize=(15, num_models * 3))
        fig.suptitle(f"Cross-Model {sample_type.capitalize()} Sample (Axial)", fontsize=16, y=0.99)
        gs = plt.GridSpec(num_models, 3, hspace=0.5, wspace=0.3)
        sorted_results = sorted(results, key=lambda x: x.get('val_loss', float('inf')))

        for i, result in enumerate(sorted_results):
            trial_id = result['trial_id']
            sample = result.get('special_samples', {}).get(sample_type)
            if sample is None:
                for j_col in range(3):
                    ax = fig.add_subplot(gs[i,j_col]); ax.text(0.5,0.5,f"{trial_id[:8]} ({sample_type}) N/A"); ax.axis('off')
                continue

            t_slice, o_slice, m_slice = sample['target'][sample['target'].shape[0]//2,:,:], \
                                        sample['output'][sample['output'].shape[0]//2,:,:], \
                                        sample['mask'][sample['mask'].shape[0]//2,:,:]
            
            valid_t = t_slice[m_slice]; valid_o = o_slice[m_slice]
            all_vals = np.concatenate([valid_t, valid_o]) if valid_t.size > 0 and valid_o.size > 0 else (valid_t if valid_t.size > 0 else valid_o)
            vmin = np.min(all_vals) if all_vals.size > 0 else 0
            vmax = np.max(all_vals) if all_vals.size > 0 else 0.1
            GLOBAL_VMIN = 0.0
            GLOBAL_VMAX = 1.0
            norm = Normalize(vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)


            titles = [f"{trial_id[:8]} Target", f"{trial_id[:8]} Pred.", f"Error (MAE {sample['metrics']['magnitude_mae']:.4f})"]
            data_to_plot = [t_slice, o_slice, np.where(m_slice, np.abs(t_slice-o_slice), np.nan)]
            cmaps = ['viridis','viridis','hot']
            norms_plot = [norm, norm, Normalize(vmin=0, vmax=np.nanmax(data_to_plot[2]) if np.any(np.isfinite(data_to_plot[2])) else 1.0)]
            if norms_plot[2].vmax == 0: norms_plot[2].vmax = 1.0

            for j_col in range(3):
                ax = fig.add_subplot(gs[i, j_col])
                im = ax.imshow(data_to_plot[j_col], cmap=cmaps[j_col], norm=norms_plot[j_col])
                if j_col < 2: ax.contour(m_slice, levels=[0.5], colors='w', linewidths=0.8, alpha=0.7) # Fixed lw to linewidths
                ax.set_title(titles[j_col], fontsize=9); ax.axis('off')
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05); plt.colorbar(im, cax=cax)

        plt.tight_layout(rect=[0,0,1,0.97])
        plt.savefig(os.path.join(output_dir, f"cross_model_{sample_type}_comparison.png"), dpi=100, bbox_inches='tight')
        plt.close()
    logger.info(f"Saved cross-model sample comparisons to {output_dir}")

def create_model_comparison_visualization(sample1, sample2, output_dir, filename_prefix, label1, label2):
    if sample1 is None or sample2 is None: return
    datas = [
        (sample1['target'][sample1['target'].shape[0]//2,:,:], 
         sample1['output'][sample1['output'].shape[0]//2,:,:],
         sample1['mask'][sample1['mask'].shape[0]//2,:,:],
         sample1['metrics']['magnitude_mae']),
        (sample2['target'][sample2['target'].shape[0]//2,:,:],
         sample2['output'][sample2['output'].shape[0]//2,:,:],
         sample2['mask'][sample2['mask'].shape[0]//2,:,:],
         sample2['metrics']['magnitude_mae'])
    ]
    labels_row = [label1, label2]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(f"Viz: {filename_prefix.replace('_', ' ').title()}", fontsize=14, y=0.99)

    for i_row in range(2):
        t_slice, p_slice, m_slice, mae_val = datas[i_row]
        valid_t = t_slice[m_slice]; valid_p = p_slice[m_slice]
        all_vals = np.concatenate([valid_t, valid_p]) if valid_t.size > 0 and valid_p.size > 0 else (valid_t if valid_t.size > 0 else valid_p)
        vmin = np.min(all_vals) if all_vals.size > 0 else 0
        vmax = np.max(all_vals) if all_vals.size > 0 else 0.1
        GLOBAL_VMIN = 0.0
        GLOBAL_VMAX = 1.0
        norm = Normalize(vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)


        ax_t = axes[i_row, 0]
        im_t = ax_t.imshow(t_slice, cmap='viridis', norm=norm)
        ax_t.contour(m_slice, levels=[0.5], colors='w', linewidths=0.8, alpha=0.7) # Fixed lw to linewidths
        ax_t.set_title(f"{labels_row[i_row]} - Target", fontsize=9); ax_t.axis('off')
        cax = make_axes_locatable(ax_t).append_axes("right",size="5%",pad=0.05); plt.colorbar(im_t, cax=cax)

        ax_p = axes[i_row, 1]
        im_p = ax_p.imshow(p_slice, cmap='viridis', norm=norm)
        ax_p.contour(m_slice, levels=[0.5], colors='w', linewidths=0.8, alpha=0.7) # Fixed lw to linewidths
        ax_p.set_title(f"{labels_row[i_row]} - Pred. (MAE {mae_val:.4f})", fontsize=9); ax_p.axis('off')
        cax = make_axes_locatable(ax_p).append_axes("right",size="5%",pad=0.05); plt.colorbar(im_p, cax=cax)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}.png"), dpi=100, bbox_inches='tight')
    plt.close()

def create_ensemble_vs_best_comparison(ensemble_results, individual_results, output_dir, ensemble_name="Ensemble"):
    if not individual_results: return
    valid_ind_res = [r for r in individual_results if 'test_metrics' in r and pd.notna(r['test_metrics'].get('magnitude_mae'))]
    if not valid_ind_res: return
        
    best_individual = min(valid_ind_res, key=lambda x: x['test_metrics']['magnitude_mae'])
    best_model_id = best_individual['trial_id']
    comparison_dir = os.path.join(output_dir, "ensemble_vs_best_model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    ens_metrics = ensemble_results['avg_metrics']
    best_m_metrics = best_individual['test_metrics']
    
    fig, ax = plt.subplots(figsize=(8, 4))
    m_keys = ['magnitude_mae', 'magnitude_rmse', 'magnitude_correlation', 'hotspot_iou']
    labels = ['MAE', 'RMSE', 'Corr', 'IoU']
    ens_vals = [ens_metrics.get(m, np.nan) for m in m_keys]
    best_m_vals = [best_m_metrics.get(m, np.nan) for m in m_keys]

    x = np.arange(len(labels)); width = 0.35
    ax.bar(x - width/2, ens_vals, width, label=ensemble_name, color='skyblue')
    ax.bar(x + width/2, best_m_vals, width, label=f'Best Ind. ({best_model_id[:8]})', color='salmon')
    ax.set_ylabel('Metric Value', fontsize=9); ax.set_title(f'{ensemble_name} vs Best Individual Metrics', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.legend(fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    fig.text(0.5, 0.01, "Note: Ind. model metrics on full test, ensemble on eval split.", ha='center', fontsize=7, style='italic')
    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(os.path.join(comparison_dir, "metrics_comparison.png"), dpi=100, bbox_inches='tight')
    plt.close()

    sample_type = 'average'
    if ensemble_results['special_samples'].get(sample_type) and best_individual.get('special_samples',{}).get(sample_type):
        create_model_comparison_visualization(
            ensemble_results['special_samples'][sample_type], best_individual['special_samples'][sample_type],
            comparison_dir, f"{sample_type}_sample_visualization",
            f"{ensemble_name} ({sample_type})", f"Best Ind. ({best_model_id[:8]}, {sample_type})"
        )

def analyze_error_distribution(sample_results, output_dir, model_id_or_name):
    if not sample_results: 
        logger.warning(f"No sample results for error distribution analysis of {model_id_or_name}. Skipping.")
        return None
    error_values = [s['metrics']['magnitude_mae'] for s in sample_results if pd.notna(s['metrics'].get('magnitude_mae'))]
    if not error_values: 
        logger.warning(f"No valid MAE values found for model {model_id_or_name}. Skipping error analysis.")
        return None

    stats = {'min':np.min(error_values),'max':np.max(error_values),'mean':np.mean(error_values),
             'median':np.median(error_values),'std':np.std(error_values),'p25':np.percentile(error_values,25),
             'p75':np.percentile(error_values,75),'p95':np.percentile(error_values,95)}

    fig = plt.figure(figsize=(10, 8)) 
    fig.suptitle(f"MAE Dist. for {model_id_or_name}", fontsize=12, y=0.99)
    stats_txt = (f"Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, Std: {stats['std']:.4f}\n"
                 f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, IQR: {stats['p75']-stats['p25']:.4f}")
    fig.text(0.5,0.93,stats_txt,ha='center',fontsize=9)
    gs = plt.GridSpec(2,2,wspace=0.3,hspace=0.4,top=0.88)

    ax1 = fig.add_subplot(gs[0,0]); 
    ax1.hist(error_values, bins=20, color='c', edgecolor='k', alpha=0.7)
    ax1.axvline(stats['mean'],color='r',linestyle='--',linewidth=1,label=f"Mean {stats['mean']:.3f}")
    ax1.axvline(stats['median'],color='g',linestyle=':',linewidth=1,label=f"Med {stats['median']:.3f}")
    ax1.set_xlabel('MAE',fontsize=8); ax1.set_ylabel('Freq',fontsize=8); ax1.set_title('Histogram',fontsize=9); ax1.legend(fontsize=7)
    ax1.tick_params(labelsize=7)

    ax2 = fig.add_subplot(gs[0,1]); 
    ax2.boxplot(error_values,vert=False,widths=0.6,patch_artist=True,
                boxprops=dict(facecolor='m', edgecolor='k'), 
                medianprops=dict(color='black',linewidth=1.5)) 
    ax2.set_xlabel('MAE',fontsize=8); ax2.set_yticklabels([]); ax2.set_title('Box Plot',fontsize=9)
    ax2.tick_params(labelsize=7)

    ax3 = fig.add_subplot(gs[1,0]); 
    sorted_err = np.sort(error_values); 
    cdf_y = np.arange(1,len(sorted_err)+1)/len(sorted_err)
    ax3.plot(sorted_err,cdf_y,'b.',alpha=0.3,ms=3); ax3.plot(sorted_err,cdf_y,'b-',linewidth=1)
    for p,lbl in [(stats['p25'],'25th'),(stats['median'],'Med'),(stats['p75'],'75th'),(stats['p95'],'95th')]:
        ax3.axvline(p,linestyle=':',linewidth=0.8,label=f"{lbl} ({p:.3f})")
    ax3.set_xlabel('MAE',fontsize=8); ax3.set_ylabel('Cum. Prob.',fontsize=8); ax3.set_title('CDF',fontsize=9); ax3.legend(fontsize=7)
    ax3.grid(True,linestyle=':',alpha=0.5); ax3.tick_params(labelsize=7)

    ax4 = fig.add_subplot(gs[1,1])
    try:
        from scipy.stats import gaussian_kde
        if len(np.unique(error_values)) > 1:
            density = gaussian_kde(error_values); x_kde = np.linspace(min(error_values),max(error_values),100)
            ax4.plot(x_kde,density(x_kde),'darkorange'); ax4.fill_between(x_kde,density(x_kde),alpha=0.3,color='sandybrown')
        else: ax4.hist(error_values,bins=10,density=True,color='sandybrown',alpha=0.7)
        ax4.axvline(stats['mean'],color='r',linestyle='--',linewidth=1,label=f"Mean {stats['mean']:.3f}")
        ax4.set_xlabel('MAE',fontsize=8); ax4.set_ylabel('Density',fontsize=8); ax4.set_title('KDE',fontsize=9); ax4.legend(fontsize=7)
    except ImportError: 
        logger.warning("Scipy not found, KDE plot skipped.")
        ax4.text(0.5,0.5,"KDE fail (scipy)",ha='center',va='center'); ax4.set_title('KDE (Error)',fontsize=9)
    except Exception as e_kde:
        logger.warning(f"KDE plot failed: {e_kde}. Fallback to hist.")
        ax4.hist(error_values,bins=10,density=True,color='sandybrown',alpha=0.7)
        ax4.set_title('KDE (Fallback Hist)',fontsize=9);
    ax4.tick_params(labelsize=7)

    os.makedirs(output_dir, exist_ok=True)
    plot_p = os.path.join(output_dir, f"error_dist_{model_id_or_name.replace(' ','_')}.png")
    plt.savefig(plot_p, dpi=100, bbox_inches='tight')
    plt.close(fig)
    txt_p = os.path.join(output_dir, f"error_stats_{model_id_or_name.replace(' ','_')}.txt")
    with open(txt_p, 'w') as f:
        f.write(f"Error Stats for {model_id_or_name}\n" + "="*30 + "\n")
        for k,v in stats.items(): f.write(f"{k.capitalize()}: {v:.6f}\n")
    logger.info(f"Saved error stats for {model_id_or_name} to {output_dir}")
    return stats

def create_combined_error_histogram(all_model_results, ensemble_results_dict, output_dir):
    num_ind_show = 3
    sorted_ind_res = sorted([r for r in all_model_results if r.get('test_metrics',{}).get('magnitude_mae') is not None],
                            key=lambda x: x['test_metrics']['magnitude_mae'])
    top_ind_res = sorted_ind_res[:num_ind_show]
    plot_items = [{'name':f"Ind: {r['trial_id'][:8]}",'samples':r['sample_images']} for r in top_ind_res]
    for name, ens_res in ensemble_results_dict.items():
        if ens_res: plot_items.append({'name':name,'samples':ens_res['ensemble_samples']})
    if not plot_items: return None

    num_plots = len(plot_items); cols = min(3,num_plots); rows = (num_plots+cols-1)//cols
    fig,axes = plt.subplots(rows,cols,figsize=(cols*4,rows*3),squeeze=False); axes=axes.flatten()
    colors = plt.cm.coolwarm(np.linspace(0,1,num_plots))

    for i,item in enumerate(plot_items):
        if i >= len(axes): break
        ax=axes[i]; name=item['name']; samples=item['samples']
        errors = [s['metrics']['magnitude_mae'] for s in samples if pd.notna(s['metrics'].get('magnitude_mae'))]
        if not errors: ax.text(0.5,0.5,"No MAE data"); ax.set_title(name,fontsize=9); continue
        
        mean_e,med_e = np.mean(errors),np.median(errors)
        ax.hist(errors,bins=15,alpha=0.7,color=colors[i],ec='k',lw=0.5)
        ax.axvline(mean_e,c='k',ls='--',lw=1,label=f'M {mean_e:.3f}')
        ax.axvline(med_e,c='dimgray',ls=':',lw=1,label=f'Md {med_e:.3f}')
        ax.set_xlabel('MAE',fontsize=8); ax.set_ylabel('Freq',fontsize=8); ax.tick_params(labelsize=7)
        ax.set_title(name,fontsize=9); ax.legend(fontsize=6, loc='upper right')

    for j in range(i+1,len(axes)): fig.delaxes(axes[j])
    fig.suptitle('MAE Distribution Comparison',fontsize=12,y=0.99)
    plt.tight_layout(rect=[0,0,1,0.95])
    os.makedirs(output_dir,exist_ok=True)
    out_p = os.path.join(output_dir,'combined_mae_dist.png')
    plt.savefig(out_p,dpi=100,bbox_inches='tight'); plt.close(fig)
    logger.info(f"Saved combined MAE dist plot to {out_p}")
    return out_p

def compare_models(results, configs, output_dir):
    comparison_dir = os.path.join(output_dir, "model_comparison"); os.makedirs(comparison_dir, exist_ok=True)
    model_data = []
    aug_map = {'trivial_augment_max_rotation':'Aug: Rot (deg)','trivial_augment_max_shift':'Aug: Shift (vox)',
               'rotation_enabled':'Aug: Rot On','shift_enabled':'Aug: Shift On'}
    all_raw_params = set()
    for _, (_, cfg) in enumerate(configs):
        if cfg: all_raw_params.update(cfg.keys())

    final_cols = set(['trial_id', 'val_loss'])
    for result, (tid_cfg, cfg_dict) in zip(results, configs):
        if not cfg_dict: cfg_dict = {}
        info = {'trial_id':result['trial_id'],'val_loss':result['val_loss']}
        proc_raw = set()
        for raw, nice in aug_map.items():
            if raw in cfg_dict: info[nice]=cfg_dict[raw]; proc_raw.add(raw); final_cols.add(nice)
        for raw in all_raw_params:
            if raw in proc_raw or raw not in cfg_dict: continue
            if 'augment' in raw.lower() or any(p in raw.lower() for p in ['shift','rot','elast','intens','noise']):
                parts = raw.replace('trivial_augment_','').replace('_enabled','').split('_')
                auto_nice = f"Aug: {' '.join(p.capitalize() for p in parts)}"
                if auto_nice not in info: info[auto_nice]=cfg_dict[raw]; final_cols.add(auto_nice)
            elif raw not in info: info[raw]=cfg_dict[raw]; final_cols.add(raw)
        for metric, value in result['test_metrics'].items():
            m_name=f'test_{metric}'; info[m_name]=value; final_cols.add(m_name)
        model_data.append(info)

    if not model_data: logger.warning("No model data for comparison. Skipping CSV/HTML/plots."); return None
    df = pd.DataFrame(model_data).reindex(columns=list(final_cols))
    df.to_csv(os.path.join(comparison_dir,"model_comparison.csv"),index=False)
    logger.info(f"Saved model comparison CSV to {comparison_dir}")

    metric_cols = [c for c in df.columns if c.startswith('test_')]
    aug_cols = [c for c in df.columns if c.startswith('Aug:')]
    hp_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in metric_cols+aug_cols+['trial_id','val_loss']]
    metrics_plot = [m for m in ['test_magnitude_mae','test_magnitude_rmse','test_magnitude_correlation','test_hotspot_iou'] if m in df.columns]

    if metrics_plot and not df.empty:
        plt.figure(figsize=(max(10, len(df) * 1.2 if len(df) > 0 else 10), 5)) 
        df_plot = df.sort_values('val_loss')
        num_metrics_to_plot = len(metrics_plot)

        for i, metric in enumerate(metrics_plot):
            plt.subplot(1, num_metrics_to_plot, i + 1)
            x_raw_labels = df_plot['trial_id'].astype(str)
            x_axis_positions = range(len(df_plot))
            
            bars = plt.bar(x_axis_positions, df_plot[metric].fillna(0))
            plt.title(metric.replace('test_magnitude_', '').replace('test_', '').upper(), fontsize=9)
            
            use_indices_as_ticks = False
            if len(df_plot) > 0 and len(x_raw_labels.iloc[0]) > 10: 
                use_indices_as_ticks = True

            if use_indices_as_ticks:
                plt.xticks(x_axis_positions, x_raw_labels, rotation=90, ha='center', fontsize=7)
            else:
                plt.xticks(x_axis_positions, x_raw_labels, rotation=45, ha='right', fontsize=7)
            
            plt.ylabel(metric, fontsize=8)
            plt.tick_params(axis='y', labelsize=7)
            plt.tight_layout()

            for bar_item in bars: 
                height = bar_item.get_height()
                plt.text(bar_item.get_x() + bar_item.get_width() / 2., height,
                         f'{height:.3f}',
                         ha='center', 
                         va='bottom', 
                         fontsize=6)
        plt.savefig(os.path.join(comparison_dir,"metrics_comparison.png"),dpi=100,bbox_inches='tight'); plt.close()

    if not df.empty and len(df.select_dtypes(include=[np.number]).columns) > 1:
        corr_full = df.select_dtypes(include=[np.number]).drop(columns=['trial_id'],errors='ignore').corr()
        if not corr_full.empty:
            plt.figure(figsize=(max(10,len(corr_full)*0.6),max(8,len(corr_full)*0.5))) 
            sns.heatmap(corr_full,annot=True,fmt=".2f",cmap='coolwarm',mask=np.triu(corr_full),annot_kws={"size":6})
            plt.title('Numeric Param & Metrics Correlation',fontsize=10); plt.xticks(fontsize=7,rotation=45,ha='right'); plt.yticks(fontsize=7)
            plt.tight_layout(); plt.savefig(os.path.join(comparison_dir,"full_correlation_heatmap.png"),dpi=100); plt.close()

    try:
        if not df.empty:
            table_cols = ['trial_id','val_loss'] + sorted([m for m in metric_cols if m in df.columns]) + \
                         sorted([a for a in aug_cols if a in df.columns]) + sorted([h for h in hp_cols if h in df.columns])
            remaining = sorted([c for c in df.columns if c not in table_cols]); table_cols.extend(remaining)
            df_sorted = df.sort_values('val_loss')
            fmt_dict = {'val_loss':'{:.5f}'}; [fmt_dict.update({m:'{:.5f}'}) for m in metric_cols]
            if 'learning_rate' in df.columns: fmt_dict['learning_rate'] = '{:.1e}'
            
            styled_df = df_sorted[table_cols].style.format(fmt_dict,na_rep='N/A').background_gradient(subset=['val_loss'],cmap='RdYlGn_r')
            html_path = os.path.join(comparison_dir,"model_comparison_details.html")
            with open(html_path,'w') as f:
                f.write("<h2>Model Comparison</h2>" + styled_df.to_html()) 
            logger.info(f"Saved HTML comparison to {html_path}")
    except Exception as e: logger.warning(f"Could not create HTML table: {e}", exc_info=False)
    return df