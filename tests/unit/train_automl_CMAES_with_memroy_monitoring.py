#!/usr/bin/env python3
# train_automl_CMAES_with_memory_monitoring.py
"""
CMA-ES AutoML script with integrated memory monitoring for TMS E-field magnitude prediction.
This version includes the DetailedMemoryMonitor from your memory experiments to track
GPU memory usage during AutoML trials.
"""

import os
import sys
import argparse
import logging
import time
import threading
from datetime import datetime
import psutil
import torch
import numpy as np
import yaml
import ray
from ray import tune
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server/SSH usage
import matplotlib.pyplot as plt
import pandas as pd
import pickle

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
logger = logging.getLogger('cmaes_automl_memory')

class DetailedMemoryMonitor:
    """Enhanced memory monitor that tracks GPU memory usage during AutoML trials."""
    
    def __init__(self, interval=0.2, log_to_file=True, output_dir="./memory_logs"):
        self.interval = interval
        self.log_to_file = log_to_file
        self.output_dir = output_dir
        self.monitoring = False
        self.monitor_thread = None
        
        # Data storage
        self.memory_data = []
        self.system_memory_data = []
        self.timestamps = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.gpu_available else 0
        
        if self.gpu_available:
            logger.info(f"GPU monitoring enabled for {self.num_gpus} GPU(s)")
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}, Total Memory: {props.total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("No GPU available for monitoring")
    
    def start_monitoring(self):
        """Start the memory monitoring thread."""
        if self.monitoring:
            logger.warning("Memory monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"AutoML memory monitoring started (interval: {self.interval}s)")
    
    def stop_monitoring(self):
        """Stop the memory monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("AutoML memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop for AutoML trials."""
        start_time = time.time()
        
        while self.monitoring:
            current_time = time.time()
            relative_time = current_time - start_time
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            system_data = {
                'time': relative_time,
                'used_gb': system_memory.used / 1024**3,
                'available_gb': system_memory.available / 1024**3,
                'percent': system_memory.percent
            }
            self.system_memory_data.append(system_data)
            
            # Get GPU memory info for all GPUs
            gpu_data = {'time': relative_time}
            if self.gpu_available:
                for gpu_id in range(self.num_gpus):
                    try:
                        torch.cuda.set_device(gpu_id)
                        memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        memory_cached = torch.cuda.memory_cached(gpu_id) / 1024**3 if hasattr(torch.cuda, 'memory_cached') else 0
                        
                        # Get max memory if available
                        try:
                            max_memory = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
                        except:
                            max_memory = memory_allocated
                        
                        gpu_data.update({
                            f'gpu_{gpu_id}_reserved_gb': memory_reserved,
                            f'gpu_{gpu_id}_allocated_gb': memory_allocated,
                            f'gpu_{gpu_id}_cached_gb': memory_cached,
                            f'gpu_{gpu_id}_max_allocated_gb': max_memory
                        })
                        
                        # Get total GPU memory
                        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                        gpu_data[f'gpu_{gpu_id}_total_gb'] = total_memory
                        # Use RESERVED memory for utilization - this is the REAL GPU usage!
                        gpu_data[f'gpu_{gpu_id}_utilization_percent'] = (memory_reserved / total_memory) * 100
                        
                    except Exception as e:
                        logger.warning(f"Error getting GPU {gpu_id} memory info: {e}")
            
            self.memory_data.append(gpu_data)
            self.timestamps.append(datetime.fromtimestamp(current_time))
            
            # Log to file if enabled
            if self.log_to_file and len(self.memory_data) % 20 == 0:  # Log every 20 samples
                self._save_current_data()
            
            time.sleep(self.interval)
    
    def _save_current_data(self):
        """Save current memory data to files."""
        try:
            # Save GPU memory data
            if self.memory_data:
                gpu_df = pd.DataFrame(self.memory_data)
                gpu_df.to_csv(os.path.join(self.output_dir, 'automl_gpu_memory_data.csv'), index=False)
            
            # Save system memory data
            if self.system_memory_data:
                system_df = pd.DataFrame(self.system_memory_data)
                system_df.to_csv(os.path.join(self.output_dir, 'automl_system_memory_data.csv'), index=False)
                
        except Exception as e:
            logger.error(f"Error saving memory data: {e}")
    
    def save_final_data(self):
        """Save all collected data to files."""
        self._save_current_data()
        
        # Also save as pickle for easy loading
        try:
            data_dict = {
                'gpu_memory': self.memory_data,
                'system_memory': self.system_memory_data,
                'timestamps': self.timestamps,
                'monitoring_interval': self.interval,
                'experiment_type': 'automl_cmaes'
            }
            with open(os.path.join(self.output_dir, 'automl_memory_data.pkl'), 'wb') as f:
                pickle.dump(data_dict, f)
        except Exception as e:
            logger.error(f"Error saving pickle data: {e}")
    
    def create_automl_memory_plots(self, save_path=None):
        """Create comprehensive AutoML memory usage plots showing multi-GPU patterns."""
        if not self.memory_data:
            logger.warning("No memory data to plot")
            return
        
        if save_path is None:
            save_path = self.output_dir
        
        # Convert data to DataFrames for easier plotting
        gpu_df = pd.DataFrame(self.memory_data)
        system_df = pd.DataFrame(self.system_memory_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('AutoML CMA-ES Memory Consumption During Trials', fontsize=16, fontweight='bold')
        
        # Plot 1: All GPU Memory Usage Over Time (Reserved = Real Usage)
        ax1 = axes[0, 0]
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        active_gpus = 0
        if self.gpu_available and self.num_gpus > 0:
            for gpu_id in range(self.num_gpus):
                # Use RESERVED memory as primary metric (real GPU usage)
                if f'gpu_{gpu_id}_reserved_gb' in gpu_df.columns:
                    reserved = gpu_df[f'gpu_{gpu_id}_reserved_gb']
                    if reserved.max() > 0.5:  # Only plot active GPUs
                        ax1.plot(gpu_df['time'], reserved, 
                                label=f'GPU {gpu_id} Reserved (Real)', linewidth=2, 
                                color=colors[gpu_id % len(colors)])
                        active_gpus += 1
                # Also plot allocated as secondary
                elif f'gpu_{gpu_id}_allocated_gb' in gpu_df.columns:
                    allocated = gpu_df[f'gpu_{gpu_id}_allocated_gb']
                    if allocated.max() > 0.5:
                        ax1.plot(gpu_df['time'], allocated, 
                                label=f'GPU {gpu_id} Allocated', linewidth=2, 
                                color=colors[gpu_id % len(colors)], linestyle='--')
                        active_gpus += 1
        
        ax1.set_title(f'GPU Memory Usage - {active_gpus} Active GPUs (CMA-ES Trials)')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory (GB)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line for GPU memory limit
        ax1.axhline(y=10.75, color='red', linestyle='--', alpha=0.7, label='GPU Limit (10.75 GB)')
        
        # Plot 2: GPU Memory Utilization
        ax2 = axes[0, 1]
        for gpu_id in range(self.num_gpus):
            util_col = f'gpu_{gpu_id}_utilization_percent'
            if util_col in gpu_df.columns:
                utilization = gpu_df[util_col]
                if utilization.max() > 5:  # Only plot if there's significant utilization
                    ax2.plot(gpu_df['time'], utilization,
                            label=f'GPU {gpu_id}', linewidth=2,
                            color=colors[gpu_id % len(colors)])
        
        ax2.set_title('GPU Utilization (REAL usage - Reserved Memory %)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Utilization (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: System Memory vs Total GPU Memory
        ax3 = axes[1, 0]
        ax3.plot(system_df['time'], system_df['used_gb'], label='System RAM Used', linewidth=2, color='red')
        
        # Calculate total GPU memory usage across all GPUs (using reserved memory)
        reserved_cols = [col for col in gpu_df.columns if 'reserved_gb' in col]
        if reserved_cols:
            total_gpu_memory = gpu_df[reserved_cols].sum(axis=1)
            ax3.plot(gpu_df['time'], total_gpu_memory, label='Total GPU Memory (Reserved)', linewidth=2, color='blue')
        else:
            # Fallback to allocated if reserved not available
            allocated_cols = [col for col in gpu_df.columns if 'allocated_gb' in col]
            if allocated_cols:
                total_gpu_memory = gpu_df[allocated_cols].sum(axis=1)
                ax3.plot(gpu_df['time'], total_gpu_memory, label='Total GPU Memory (Allocated)', linewidth=2, color='blue')
        
        ax3.set_title('System vs GPU Memory Usage')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Memory (GB)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: AutoML Trial Statistics
        ax4 = axes[1, 1]
        
        # Calculate statistics for each GPU
        gpu_stats = []
        gpu_names = []
        
        for gpu_id in range(self.num_gpus):
            # Use reserved memory for statistics (real usage)
            reserved_col = f'gpu_{gpu_id}_reserved_gb'
            allocated_col = f'gpu_{gpu_id}_allocated_gb'
            
            if reserved_col in gpu_df.columns:
                memory_data = gpu_df[reserved_col]
                memory_type = "Reserved"
            elif allocated_col in gpu_df.columns:
                memory_data = gpu_df[allocated_col]
                memory_type = "Allocated"
            else:
                continue
                
            if memory_data.max() > 0.5:  # Only include active GPUs
                gpu_stats.append([
                    memory_data.mean(),
                    memory_data.max(),
                    memory_data.std()
                ])
                gpu_names.append(f'GPU {gpu_id}')
        
        if gpu_stats:
            gpu_stats = np.array(gpu_stats)
            x = np.arange(len(gpu_names))
            width = 0.25
            
            bars1 = ax4.bar(x - width, gpu_stats[:, 0], width, label='Mean', alpha=0.8)
            bars2 = ax4.bar(x, gpu_stats[:, 1], width, label='Peak', alpha=0.8)
            bars3 = ax4.bar(x + width, gpu_stats[:, 2], width, label='Std Dev', alpha=0.8)
            
            ax4.set_xlabel('GPU')
            ax4.set_ylabel('Memory (GB)')
            ax4.set_title('CMA-ES Trial Memory Statistics')
            ax4.set_xticks(x)
            ax4.set_xticklabels(gpu_names)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plots
        plot_path = os.path.join(save_path, 'automl_cmaes_memory_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"AutoML CMA-ES memory plots saved to {plot_path}")
        
        # Create detailed multi-GPU timeline
        self._create_detailed_automl_timeline(gpu_df, save_path)
        
        plt.close(fig)
    
    def _create_detailed_automl_timeline(self, gpu_df, save_path):
        """Create a detailed timeline showing all GPU usage patterns."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        active_gpus = []
        
        for gpu_id in range(self.num_gpus):
            # Use reserved memory as primary (real usage)
            reserved_col = f'gpu_{gpu_id}_reserved_gb'
            allocated_col = f'gpu_{gpu_id}_allocated_gb'
            
            if reserved_col in gpu_df.columns:
                memory_data = gpu_df[reserved_col]
                memory_label = "Reserved (Real)"
            elif allocated_col in gpu_df.columns:
                memory_data = gpu_df[allocated_col]
                memory_label = "Allocated"
            else:
                continue
                
            if memory_data.max() > 0.5:  # Only plot active GPUs
                # Plot memory usage
                ax.plot(gpu_df['time'], memory_data, 
                       label=f'GPU {gpu_id} {memory_label}', linewidth=2, 
                       color=colors[gpu_id % len(colors)])
                active_gpus.append(gpu_id)
                
                # Add peak annotation
                max_idx = memory_data.idxmax()
                max_mem = memory_data.loc[max_idx]
                max_time = gpu_df.loc[max_idx, 'time']
                
                ax.annotate(f'GPU {gpu_id}\nPeak: {max_mem:.1f}GB', 
                           xy=(max_time, max_mem), 
                           xytext=(max_time + 10, max_mem + 0.5),
                           arrowprops=dict(arrowstyle='->', color=colors[gpu_id % len(colors)]),
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # Add memory limit lines
        ax.axhline(y=10.75, color='red', linestyle='--', alpha=0.7, 
                   label='GPU Memory Limit (10.75 GB)')
        ax.axhline(y=8.6, color='orange', linestyle=':', alpha=0.7, 
                   label='80% Memory (8.6 GB)')
        
        ax.set_title(f'CMA-ES AutoML - {len(active_gpus)} GPU Memory Timeline', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory (GB)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        detailed_plot_path = os.path.join(save_path, 'automl_cmaes_detailed_timeline.png')
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Detailed CMA-ES timeline saved to {detailed_plot_path}")
        plt.close(fig)


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
    parser = argparse.ArgumentParser(description='CMA-ES AutoML with Memory Monitoring for TMS E-field magnitude prediction')
    parser.add_argument('--data-dir', type=str, default='/home/freyhe/MA_Henry/data', help='Data directory path')
    parser.add_argument('--output-dir', type=str, default='cmaes_automl_memory_output', help='Output directory for results')
    parser.add_argument('--max-concurrent', type=int, default=default_max_concurrent, 
                   help=f'Maximum number of concurrent trials (default: {default_max_concurrent} - based on available GPUs)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                   help='Comma-separated list of specific GPU IDs to use (e.g., "0,1,3,7"). If not specified, all available GPUs will be used.')
    # Subject assignments
    parser.add_argument('--train-subjects', type=str, default='4,6,7,8', help='Comma-separated list of training subject IDs')
    parser.add_argument('--val-subjects', type=str, default='3', help='Comma-separated list of validation subject IDs')
    parser.add_argument('--test-subjects', type=str, default='9', help='Comma-separated list of test subject IDs')
    
    # AutoML parameters - DEFAULT TO 8 TRIALS FOR MEMORY EXPERIMENT
    parser.add_argument('--num-samples', type=int, default=8, help='Number of trials to run (default: 8 for memory monitoring)')
    parser.add_argument('--max-epochs', type=int, default=10, help='Maximum number of epochs per trial (reduced for memory experiment)')
    
    # CMA-ES specific parameters
    parser.add_argument('--sigma0', type=float, default=0.5, help='Initial standard deviation for CMA-ES')
    parser.add_argument('--population-size', type=int, default=None, 
                       help='Population size for CMA-ES (default: None - auto determined based on dimensions)')
    
    # Memory monitoring parameters
    parser.add_argument('--monitoring-interval', type=float, default=0.2, 
                       help='Memory monitoring interval in seconds (default: 0.2)')
    
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
    experiment_name = f"cmaes_automl_memory_{timestamp}"
    output_dir = os.path.abspath(os.path.join(args.output_dir, experiment_name))
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Experiment output directory: {output_dir}")
    logger.info(f"Running {args.num_samples} trials with memory monitoring")
    
    # Create experiment tracker
    tracker = ExperimentTracker(
        experiment_dir=output_dir,
        architecture_name="simple_unet_magnitude_automl_memory",
        create_subdirs=True
    )
    tracker.set_description(f"CMA-ES AutoML with Memory Monitoring - {args.num_samples} trials")
    
    # Create memory monitor
    memory_monitor = DetailedMemoryMonitor(
        interval=args.monitoring_interval,
        log_to_file=True,
        output_dir=os.path.join(output_dir, 'memory_logs')
    )
    
    # Create resource monitor
    resource_monitor = ResourceMonitor(max_memory_gb=8, check_interval=10.0)
    resource_monitor.start_monitoring()
    
    # Define base configuration (non-tunable parameters)
    base_config = {
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'data_dir': args.data_dir,
        'batch_size': 64,  # Conservative batch size for memory monitoring
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
        'early_stopping_patience': 5, # Reduced for faster trials in memory experiment
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
        n_startup_trials=min(2, args.num_samples // 4),  # Use fewer startup trials for small experiments
        seed=42,
        popsize=args.population_size  # None means it will be automatically determined
    )
    
    # Define a smaller search space for the memory experiment (8 trials)
    search_space = {
        # Model parameters - reduced ranges for faster convergence with fewer trials
        "learning_rate": tune.loguniform(5e-4, 5e-3),        # Narrower learning rate range
        "feature_maps": tune.uniform(24.0, 48.0),            # Smaller feature maps range
        "levels": tune.uniform(3.0, 4.0),                    # Smaller levels range
        "dropout_rate": tune.uniform(0.1, 0.3),              # Narrower dropout range
        "feature_multiplier": tune.uniform(1.5, 2.5),        # Narrower multiplier range
        # TrivialAugment parameters - reduced ranges
        "trivial_augment_max_rotation": tune.uniform(20.0, 40.0),    # Narrower rotation range
        "trivial_augment_max_shift": tune.uniform(3.0, 7.0),         # Narrower shift range
        "trivial_augment_max_elastic": tune.uniform(2.0, 4.0),       # Narrower elastic range
        "trivial_augment_max_intensity": tune.uniform(0.05, 0.12),   # Narrower intensity range
        "trivial_augment_max_noise": tune.uniform(0.02, 0.04)        # Narrower noise range
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
        grace_period=3,  # Reduced grace period for faster trials
        reduction_factor=2  # Less aggressive reduction for memory experiment
    )
    
    # Start memory monitoring
    logger.info("="*70)
    logger.info("STARTING CMA-ES AUTOML WITH MEMORY MONITORING")
    logger.info("="*70)
    logger.info(f"Number of trials: {args.num_samples}")
    logger.info(f"Max concurrent trials: {max_concurrent}")
    logger.info(f"Max epochs per trial: {args.max_epochs}")
    logger.info(f"Memory monitoring interval: {args.monitoring_interval}s")
    logger.info(f"CMA-ES parameters: sigma0={args.sigma0}, population_size={args.population_size or 'auto'}")
    logger.info(f"Using TrivialAugment: ONE random augmentation per sample")
    logger.info(f"Gradient checkpointing enabled for memory efficiency")
    logger.info("="*70)
    
    memory_monitor.start_monitoring()
    
    try:
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
            max_failures=2,   # Reduced max failures for small experiment
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
        
        logger.info("="*70)
        logger.info("CMA-ES AUTOML WITH MEMORY MONITORING COMPLETED")
        logger.info("="*70)
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
            
    except Exception as e:
        logger.error(f"AutoML experiment failed: {e}")
        raise
        
    finally:
        # Stop memory monitoring and create plots
        memory_monitor.stop_monitoring()
        memory_monitor.save_final_data()
        
        logger.info("Creating memory consumption plots...")
        memory_monitor.create_automl_memory_plots(output_dir)
        
        # Generate experiment summary with memory statistics
        if memory_monitor.memory_data:
            gpu_df = pd.DataFrame(memory_monitor.memory_data)
            
            summary = {
                'experiment_name': experiment_name,
                'experiment_type': 'automl_cmaes_memory',
                'num_trials': args.num_samples,
                'max_concurrent_trials': max_concurrent,
                'monitoring_interval': args.monitoring_interval,
                'total_monitoring_samples': len(memory_monitor.memory_data),
                'experiment_duration_seconds': memory_monitor.memory_data[-1]['time'] if memory_monitor.memory_data else 0,
                'num_gpus_used': num_gpus_available
            }
            
            # Add GPU statistics for all GPUs (using reserved memory for real usage)
            total_peak_memory = 0
            total_mean_memory = 0
            active_gpus = 0
            
            for gpu_id in range(num_gpus_available):
                # Use reserved memory for real GPU usage statistics
                reserved_col = f'gpu_{gpu_id}_reserved_gb'
                allocated_col = f'gpu_{gpu_id}_allocated_gb'
                
                if reserved_col in gpu_df.columns:
                    memory_data = gpu_df[reserved_col]
                    memory_type = "reserved"
                elif allocated_col in gpu_df.columns:
                    memory_data = gpu_df[allocated_col]
                    memory_type = "allocated"
                else:
                    continue
                    
                max_mem = memory_data.max()
                mean_mem = memory_data.mean()
                
                summary[f'gpu_{gpu_id}_max_memory_gb'] = max_mem
                summary[f'gpu_{gpu_id}_mean_memory_gb'] = mean_mem
                summary[f'gpu_{gpu_id}_memory_type'] = memory_type
                summary[f'gpu_{gpu_id}_peak_utilization_percent'] = (max_mem / 10.75) * 100
                
                if max_mem > 0.5:  # Consider GPU active
                    total_peak_memory += max_mem
                    total_mean_memory += mean_mem
                    active_gpus += 1
            
            summary['total_peak_memory_gb'] = total_peak_memory
            summary['total_mean_memory_gb'] = total_mean_memory  
            summary['active_gpus'] = active_gpus
            
            # Save summary
            summary_path = os.path.join(output_dir, 'automl_memory_experiment_summary.yaml')
            with open(summary_path, 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
            
            # Print final summary
            print("\n" + "="*70)
            print("CMA-ES AUTOML MEMORY EXPERIMENT SUMMARY")
            print("="*70)
            print(f"Experiment: {experiment_name}")
            print(f"Total trials: {args.num_samples}")
            print(f"Concurrent trials: {max_concurrent}")
            print(f"Total duration: {summary['experiment_duration_seconds']:.1f} seconds")
            print(f"Active GPUs: {active_gpus}/{num_gpus_available}")
            print()
            
            for gpu_id in range(num_gpus_available):
                max_mem = summary.get(f'gpu_{gpu_id}_max_memory_gb', 0)
                mean_mem = summary.get(f'gpu_{gpu_id}_mean_memory_gb', 0)
                memory_type = summary.get(f'gpu_{gpu_id}_memory_type', 'unknown')
                utilization = summary.get(f'gpu_{gpu_id}_peak_utilization_percent', 0)
                if max_mem > 0.5:
                    print(f"GPU {gpu_id} - Peak: {max_mem:.2f} GB ({utilization:.1f}%), Average: {mean_mem:.2f} GB ({memory_type})")
            
            print(f"\nTotal Peak Memory: {total_peak_memory:.2f} GB across {active_gpus} active GPUs")
            print(f"Results saved to: {output_dir}")
            print("="*70)
        
        # Stop resource monitor
        resource_monitor.stop_monitoring()
        
        # Finalize experiment tracker
        tracker.finalize(status="completed")
        
        # Shutdown Ray
        ray.shutdown()
        
    logger.info("CMA-ES AutoML Memory Experiment finished!")