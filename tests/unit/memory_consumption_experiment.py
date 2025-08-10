#!/usr/bin/env python3
# memory_consumption_experiment.py
"""
Experiment script to monitor and visualize GPU memory consumption during TMS E-field prediction training.
Creates detailed memory usage graphs for analysis.
UPDATED: Simplified GPU labeling and removed peak annotations.
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server/SSH usage
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from collections import defaultdict
import gc

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from your codebase
from tms_efield_prediction.automl.integration.ray_trainable import train_model_tune
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.experiments.tracking import ExperimentTracker
from tms_efield_prediction.utils.resource.gpu_checker import configure_gpu_environment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('memory_experiment')

class DetailedMemoryMonitor:
    """Enhanced memory monitor that tracks GPU memory usage in detail."""
    
    def __init__(self, interval=0.5, log_to_file=True, output_dir="./memory_logs"):
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
        logger.info(f"Memory monitoring started (interval: {self.interval}s)")
    
    def stop_monitoring(self):
        """Stop the memory monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
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
            
            # Get GPU memory info
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
            if self.log_to_file and len(self.memory_data) % 10 == 0:  # Log every 10 samples
                self._save_current_data()
            
            time.sleep(self.interval)
    
    def _save_current_data(self):
        """Save current memory data to files."""
        try:
            # Save GPU memory data
            if self.memory_data:
                gpu_df = pd.DataFrame(self.memory_data)
                gpu_df.to_csv(os.path.join(self.output_dir, 'gpu_memory_data.csv'), index=False)
            
            # Save system memory data
            if self.system_memory_data:
                system_df = pd.DataFrame(self.system_memory_data)
                system_df.to_csv(os.path.join(self.output_dir, 'system_memory_data.csv'), index=False)
                
        except Exception as e:
            logger.error(f"Error saving memory data: {e}")
    
    def save_final_data(self):
        """Save all collected data to files."""
        self._save_current_data()
        
        # Also save as pickle for easy loading
        try:
            import pickle
            data_dict = {
                'gpu_memory': self.memory_data,
                'system_memory': self.system_memory_data,
                'timestamps': self.timestamps,
                'monitoring_interval': self.interval
            }
            with open(os.path.join(self.output_dir, 'memory_data.pkl'), 'wb') as f:
                pickle.dump(data_dict, f)
        except Exception as e:
            logger.error(f"Error saving pickle data: {e}")
    
    def create_memory_plots(self, save_path=None):
        """Create comprehensive memory usage plots."""
        if not self.memory_data:
            logger.warning("No memory data to plot")
            return
        
        if save_path is None:
            save_path = self.output_dir
        
        # Convert data to DataFrames for easier plotting
        gpu_df = pd.DataFrame(self.memory_data)
        system_df = pd.DataFrame(self.system_memory_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Memory Consumption During Training', fontsize=16, fontweight='bold')
        
        # Plot 1: GPU Memory Usage Over Time - Only GPU 0
        ax1 = axes[0, 0]
        if self.gpu_available and 'gpu_0_reserved_gb' in gpu_df.columns:
            ax1.plot(gpu_df['time'], gpu_df['gpu_0_reserved_gb'], 
                    label='GPU', linewidth=2, color='blue')
        
        ax1.set_title('GPU Memory Usage')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory (GB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: GPU Memory Utilization Percentage - Only GPU 0
        ax2 = axes[0, 1]
        if self.gpu_available and 'gpu_0_utilization_percent' in gpu_df.columns:
            ax2.plot(gpu_df['time'], gpu_df['gpu_0_utilization_percent'], 
                    label='GPU', linewidth=2, color='blue')
        
        ax2.set_title('GPU Memory Utilization')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Utilization (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: System Memory Usage
        ax3 = axes[1, 0]
        ax3.plot(system_df['time'], system_df['used_gb'], label='Used RAM', linewidth=2, color='red')
        ax3.plot(system_df['time'], system_df['available_gb'], label='Available RAM', linewidth=2, color='green')
        ax3.set_title('System Memory Usage')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Memory (GB)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Memory Usage Summary Statistics - Only GPU 0
        ax4 = axes[1, 1]
        
        # Calculate statistics for GPU 0 only
        stats_data = []
        if self.gpu_available and 'gpu_0_reserved_gb' in gpu_df.columns:
            reserved = gpu_df['gpu_0_reserved_gb']
            stats_data.extend([
                'GPU Mean', reserved.mean(),
                'GPU Max', reserved.max(),
                'GPU Min', reserved.min()
            ])
        
        # Add system memory stats
        system_used = system_df['used_gb']
        stats_data.extend([
            'System Mean', system_used.mean(),
            'System Max', system_used.max(),
            'System Min', system_used.min()
        ])
        
        # Create bar plot of statistics
        if stats_data:
            labels = stats_data[::2]
            values = stats_data[1::2]
            colors = ['blue', 'blue', 'blue', 'red', 'red', 'red']
            bars = ax4.bar(range(len(labels)), values, color=colors[:len(labels)])
            ax4.set_title('Memory Usage Statistics (GB)')
            ax4.set_ylabel('Memory (GB)')
            ax4.set_xticks(range(len(labels)))
            ax4.set_xticklabels(labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plots
        plot_path = os.path.join(save_path, 'memory_consumption_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Memory plots saved to {plot_path}")
        
        # Create individual detailed GPU plot
        if self.gpu_available and 'gpu_0_reserved_gb' in gpu_df.columns:
            self._create_detailed_gpu_plot(gpu_df, save_path)
        
        plt.close(fig)
    
    def _create_detailed_gpu_plot(self, gpu_df, save_path):
        """Create a detailed GPU memory usage plot - GPU 0 only, simple legend."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if 'gpu_0_reserved_gb' in gpu_df.columns:
            # Plot reserved memory as the main line
            ax.plot(gpu_df['time'], gpu_df['gpu_0_reserved_gb'], 
                   label='Memory Used', linewidth=3, color='blue')
            
            # Add total memory line
            if 'gpu_0_total_gb' in gpu_df.columns:
                total_mem = gpu_df['gpu_0_total_gb'].iloc[0]
                ax.axhline(y=total_mem, color='red', linestyle='-', alpha=0.8, linewidth=2,
                          label='Max Memory of GPU')
                
                # Add 80% memory line
                eighty_percent_mem = total_mem * 0.8
                ax.axhline(y=eighty_percent_mem, color='orange', linestyle=':', alpha=0.8, linewidth=2,
                          label='80%')
        
        ax.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory (GB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        detailed_plot_path = os.path.join(save_path, 'detailed_gpu_memory_timeline.png')
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Detailed GPU plot saved to {detailed_plot_path}")
        plt.close(fig)

    def create_automl_memory_plots(self, save_path=None):
        """Create comprehensive AutoML memory usage plots - GPU 0 only."""
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
        fig.suptitle('GPU Memory Consumption During Training', fontsize=16, fontweight='bold')
        
        # Plot 1: GPU Memory Usage Over Time - GPU 0 only
        ax1 = axes[0, 0]
        if self.gpu_available and 'gpu_0_allocated_gb' in gpu_df.columns:
            allocated = gpu_df['gpu_0_allocated_gb']
            if allocated.max() > 0.5:  # Only plot if GPU 0 is active
                ax1.plot(gpu_df['time'], allocated, 
                        label='GPU', linewidth=2, color='blue')
        
        ax1.set_title('GPU Memory Usage')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory (GB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line for GPU memory limit
        ax1.axhline(y=10.75, color='red', linestyle='--', alpha=0.7, label='GPU Limit (10.75 GB)')
        
        # Plot 2: GPU Memory Utilization - GPU 0 only
        ax2 = axes[0, 1]
        if self.gpu_available and 'gpu_0_utilization_percent' in gpu_df.columns:
            ax2.plot(gpu_df['time'], gpu_df['gpu_0_utilization_percent'], 
                    label='GPU', linewidth=2, color='blue')
        
        ax2.set_title('GPU Utilization')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Utilization (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: System Memory vs GPU Memory
        ax3 = axes[1, 0]
        ax3.plot(system_df['time'], system_df['used_gb'], label='System RAM Used', linewidth=2, color='red')
        
        # Calculate GPU 0 memory usage
        if 'gpu_0_allocated_gb' in gpu_df.columns:
            ax3.plot(gpu_df['time'], gpu_df['gpu_0_allocated_gb'], label='GPU Memory', linewidth=2, color='blue')
        
        ax3.set_title('System vs GPU Memory Usage')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Memory (GB)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Memory Statistics - GPU 0 only
        ax4 = axes[1, 1]
        
        # Calculate statistics for GPU 0
        if 'gpu_0_allocated_gb' in gpu_df.columns:
            allocated = gpu_df['gpu_0_allocated_gb']
            if allocated.max() > 0.5:  # Only include if GPU 0 is active
                gpu_stats = [allocated.mean(), allocated.max(), allocated.std()]
                
                x = [0]
                width = 0.25
                
                bars1 = ax4.bar(x[0] - width, gpu_stats[0], width, label='Mean', alpha=0.8, color='blue')
                bars2 = ax4.bar(x[0], gpu_stats[1], width, label='Peak', alpha=0.8, color='red')
                bars3 = ax4.bar(x[0] + width, gpu_stats[2], width, label='Std Dev', alpha=0.8, color='green')
                
                ax4.set_xlabel('GPU')
                ax4.set_ylabel('Memory (GB)')
                ax4.set_title('Memory Statistics')
                ax4.set_xticks(x)
                ax4.set_xticklabels(['GPU'])
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
        plot_path = os.path.join(save_path, 'automl_memory_consumption_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"AutoML memory plots saved to {plot_path}")
        
        # Create detailed GPU timeline
        self._create_detailed_automl_timeline(gpu_df, save_path)
        
        plt.close(fig)
    
    def _create_detailed_automl_timeline(self, gpu_df, save_path):
        """Create a detailed timeline showing GPU 0 usage patterns only."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        if 'gpu_0_allocated_gb' in gpu_df.columns:
            allocated = gpu_df['gpu_0_allocated_gb']
            if allocated.max() > 0.5:  # Only plot if GPU 0 is active
                # Plot allocated memory
                ax.plot(gpu_df['time'], allocated, 
                       label='GPU', linewidth=2, color='blue')
        
        # Add memory limit lines
        ax.axhline(y=10.75, color='red', linestyle='--', alpha=0.7, 
                   label='GPU Memory Limit (10.75 GB)')
        ax.axhline(y=8.6, color='orange', linestyle=':', alpha=0.7, 
                   label='80% Memory (8.6 GB)')
        
        ax.set_title('GPU Memory Timeline', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory (GB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        detailed_plot_path = os.path.join(save_path, 'automl_detailed_gpu_timeline.png')
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Detailed AutoML timeline saved to {detailed_plot_path}")
        plt.close(fig)


def estimate_memory_usage(config, num_gpus):
    """Estimate GPU memory usage based on configuration - targets RESERVED memory."""
    
    batch_size_per_gpu = config['batch_size'] // num_gpus
    feature_maps = config['feature_maps']
    levels = config['levels']
    
    # More realistic estimation targeting RESERVED memory (the real usage)
    # Input: 20x20x20 = 8000 voxels, 9 channels, float32 = 4 bytes
    input_memory_per_sample = 20 * 20 * 20 * 9 * 4 / 1024**3  # GB
    input_memory_total = input_memory_per_sample * batch_size_per_gpu
    
    # Model parameters estimation
    total_feature_maps = feature_maps * (2 ** levels)
    model_params = total_feature_maps * 1000  
    model_memory = (model_params * 4) / 1024**3  
    
    # Activation memory - more conservative estimate
    # Based on empirical observation: reserved memory ≈ allocated * 1.15-1.25
    activation_multiplier = batch_size_per_gpu * feature_maps / 8000  # More conservative
    
    # Base memory calculation
    base_memory = input_memory_total + model_memory
    allocated_estimate = base_memory * (1.8 + activation_multiplier)
    
    # Reserved memory is typically 15-25% higher than allocated (PyTorch cache)
    reserved_estimate = allocated_estimate * 1.2  # 20% overhead for PyTorch cache
    
    # Add PyTorch overhead
    final_estimate = reserved_estimate + 1.0  # 1GB base overhead
    
    return min(final_estimate, 12.0)  # Cap at reasonable maximum


def create_memory_training_config(base_config, memory_stress_level="medium", num_gpus=1):
    """Create training configuration optimized for memory monitoring and RTX 2080 Ti."""
    
    # Adjusted for RTX 2080 Ti - targeting 80-90% RESERVED memory utilization
    # (Reserved memory is the REAL GPU usage, not allocated memory)
    stress_configs = {
        "light": {
            "batch_size_per_gpu": 64,    # Total: 64 * num_gpus
            "feature_maps": 32,
            "levels": 3,
            "epochs": 5
        },
        "medium": {
            "batch_size_per_gpu": 64,    # Reduced from 128 for better safety margin
            "feature_maps": 64,
            "levels": 4,
            "epochs": 10
        },
        "heavy": {
            "batch_size_per_gpu": 96,    # Further reduced - target 80-85% RESERVED memory
            "feature_maps": 128,         # Much larger model
            "levels": 5, 
            "epochs": 15
        },
        "extreme": {
            "batch_size_per_gpu": 128,   # Further reduced - target 85-90% RESERVED memory
            "feature_maps": 256,         # Very large model
            "levels": 6,
            "epochs": 20
        }
    }
    
    config = base_config.copy()
    stress_config = stress_configs.get(memory_stress_level, stress_configs["medium"])
    
    # Calculate total batch size based on number of GPUs
    total_batch_size = stress_config["batch_size_per_gpu"] * num_gpus
    config.update(stress_config)
    config['batch_size'] = total_batch_size
    config['num_gpus'] = num_gpus
    
    # Force multi-GPU usage if available
    config['use_data_parallel'] = num_gpus > 1
    config['gpu_ids'] = list(range(num_gpus)) if num_gpus > 1 else [0]
    
    # Enable gradient checkpointing for memory efficiency demonstration
    # Note: Disable for extreme stress testing to maximize memory usage
    config['gradient_checkpointing'] = memory_stress_level != "extreme"
    
    # Explicitly set model configuration for consistent memory usage
    config['model_config'] = {
        "model_type": "simple_unet_magnitude",
        "output_shape": [1, *config.get('output_shape', (20, 20, 20))],
        "output_channels": 1,
        "feature_maps": config['feature_maps'],
        "levels": config['levels'],
        "norm_type": config.get('norm_type', 'batch'),
        "activation": config.get('activation', 'relu'),
        "dropout_rate": config.get('dropout_rate', 0.2),
        "use_residual": config.get('use_residual', True),
        "use_attention": config.get('use_attention', False),
    }
    
    return config


def run_memory_experiment(config, monitor, experiment_tracker):
    """Run a single training experiment with memory monitoring."""
    logger.info("Starting memory consumption experiment...")
    
    # Start memory monitoring
    monitor.start_monitoring()
    
    # Record start time for inference calculation
    start_time = time.time()
    
    try:
        # Run training
        result = train_model_tune(config)
        
        # Calculate inference/training time
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Add inference time to result
        result['inference_time_seconds'] = inference_time
        result['inference_time_minutes'] = inference_time / 60
        
        logger.info(f"Training completed successfully. Final validation loss: {result.get('val_loss', 'N/A')}")
        logger.info(f"Total inference/training time: {inference_time:.2f} seconds ({inference_time/60:.2f} minutes)")
        return result
        
    except Exception as e:
        # Still calculate time even if failed
        end_time = time.time()
        inference_time = end_time - start_time
        logger.error(f"Training failed with error: {e}")
        return {
            "error": str(e), 
            "inference_time_seconds": inference_time,
            "inference_time_minutes": inference_time / 60
        }
        
    finally:
        # Stop monitoring and save data
        monitor.stop_monitoring()
        monitor.save_final_data()
        
        # Force garbage collection to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Memory consumption experiment for TMS E-field prediction')
    parser.add_argument('--data-dir', type=str, default='/home/freyhe/MA_Henry/data', 
                       help='Data directory path')
    parser.add_argument('--output-dir', type=str, default='memory_experiment_output', 
                       help='Output directory for results')
    parser.add_argument('--memory-stress', type=str, choices=['light', 'medium', 'heavy', 'extreme'], 
                       default='heavy', help='Memory stress level for the experiment')
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available GPUs)')
    parser.add_argument('--force-single-gpu', action='store_true',
                       help='Force single GPU usage even if multiple GPUs are available')
    parser.add_argument('--monitoring-interval', type=float, default=0.2, 
                       help='Memory monitoring interval in seconds')
    parser.add_argument('--train-subjects', type=str, default='4,6,7', 
                       help='Comma-separated list of training subject IDs')
    parser.add_argument('--val-subjects', type=str, default='3', 
                       help='Comma-separated list of validation subject IDs')
    parser.add_argument('--use-stacked-arrays', action='store_true', default=True,
                       help='Use pre-stacked array files')
    
    args = parser.parse_args()
    
    # Configure GPU environment
    gpu_config = configure_gpu_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine number of GPUs to use
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if args.force_single_gpu:
        num_gpus = 1
        logger.info("Forcing single GPU usage as requested")
    elif args.num_gpus is not None:
        num_gpus = min(args.num_gpus, available_gpus)
        logger.info(f"Using {num_gpus} GPUs as requested (available: {available_gpus})")
    else:
        # Use all available GPUs by default for maximum memory stress
        num_gpus = available_gpus
        logger.info(f"Using all available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        logger.error("No GPUs available! This experiment requires GPU(s).")
        exit(1)
    
    logger.info(f"GPU Configuration:")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / 1024**3
        logger.info(f"  GPU {i}: {props.name} ({total_memory_gb:.1f} GB)")
    
    # Calculate total GPU memory across all GPUs
    total_gpu_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                          for i in range(num_gpus)) / 1024**3
    logger.info(f"Total GPU Memory Available: {total_gpu_memory:.1f} GB across {num_gpus} GPU(s)")
    
    # Parse subject lists  
    train_subjects = [s.strip().zfill(3) for s in args.train_subjects.split(',')]
    val_subjects = [s.strip().zfill(3) for s in args.val_subjects.split(',')]
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"memory_experiment_{args.memory_stress}_{timestamp}"
    output_dir = os.path.abspath(os.path.join(args.output_dir, experiment_name))
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Experiment output directory: {output_dir}")
    logger.info(f"Memory stress level: {args.memory_stress}")
    logger.info(f"Training subjects: {train_subjects}")
    logger.info(f"Validation subjects: {val_subjects}")
    
    # Create experiment tracker
    tracker = ExperimentTracker(
        experiment_dir=output_dir,
        architecture_name=f"memory_experiment_{args.memory_stress}",
        create_subdirs=True
    )
    tracker.set_description(f"Memory consumption experiment - {args.memory_stress} stress level")
    
    # Create memory monitor
    memory_monitor = DetailedMemoryMonitor(
        interval=args.monitoring_interval,
        log_to_file=True,
        output_dir=os.path.join(output_dir, 'memory_logs')
    )
    
    # Create base configuration
    base_config = {
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': [],  # No test subjects for this experiment
        'data_dir': args.data_dir,
        'augment': True,
        'use_trivial_augment': True,
        'norm_type': 'batch',
        'activation': 'relu',
        'optimizer_type': 'adamw',
        'scheduler_type': 'reduce_on_plateau',
        'scheduler_patience': 3,
        'early_stopping_patience': 7,
        'learning_rate': 0.001,
        'dropout_rate': 0.2,
        'use_residual': True,
        'use_attention': False,
        'output_shape': (20, 20, 20),
        'use_stacked_arrays': args.use_stacked_arrays,
        'loss_type': 'magnitude_mse',
        'mask_threshold': 0.001,
    }
    
    # Create memory-optimized config
    experiment_config = create_memory_training_config(base_config, args.memory_stress, num_gpus)
    
    # Log the effective configuration
    logger.info(f"Experiment Configuration:")
    logger.info(f"  Memory Stress Level: {args.memory_stress}")
    logger.info(f"  Number of GPUs: {num_gpus}")
    logger.info(f"  Batch Size per GPU: {experiment_config['batch_size'] // num_gpus}")
    logger.info(f"  Total Batch Size: {experiment_config['batch_size']}")
    logger.info(f"  Feature Maps: {experiment_config['feature_maps']}")
    logger.info(f"  UNet Levels: {experiment_config['levels']}")
    logger.info(f"  Gradient Checkpointing: {experiment_config['gradient_checkpointing']}")
    logger.info(f"  Multi-GPU Training: {experiment_config.get('use_data_parallel', False)}")
    
    # Estimate memory usage
    estimated_memory_per_gpu = estimate_memory_usage(experiment_config, num_gpus)
    logger.info(f"Estimated Memory Usage per GPU: {estimated_memory_per_gpu:.1f} GB")
    
    # Warning if memory usage might be too high
    for i in range(num_gpus):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        usage_percent = (estimated_memory_per_gpu / gpu_memory) * 100
        if usage_percent > 90:
            logger.warning(f"GPU {i}: Estimated usage {usage_percent:.1f}% may cause OOM errors!")
        elif usage_percent > 80:
            logger.warning(f"GPU {i}: High estimated usage {usage_percent:.1f}%")
        else:
            logger.info(f"GPU {i}: Estimated usage {usage_percent:.1f}% looks good")
    
    # Save experiment configuration
    config_path = os.path.join(output_dir, 'experiment_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(experiment_config, f, default_flow_style=False)
    logger.info(f"Experiment configuration saved to {config_path}")
    
    # Run the memory experiment
    logger.info("="*50)
    logger.info(f"Starting memory consumption experiment")
    logger.info(f"Configuration: {experiment_config}")
    logger.info("="*50)
    
    try:
        result = run_memory_experiment(experiment_config, memory_monitor, tracker)
        
        # Create memory plots
        logger.info("Creating memory consumption plots...")
        memory_monitor.create_memory_plots(output_dir)
        
        # Save experiment results
        results_path = os.path.join(output_dir, 'experiment_results.yaml')
        with open(results_path, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
        
        # Generate summary report
        summary = {
            'experiment_name': experiment_name,
            'memory_stress_level': args.memory_stress,
            'monitoring_interval': args.monitoring_interval,
            'total_monitoring_samples': len(memory_monitor.memory_data),
            'experiment_duration_seconds': memory_monitor.memory_data[-1]['time'] if memory_monitor.memory_data else 0,
            'inference_time_seconds': result.get('inference_time_seconds', 0),
            'inference_time_minutes': result.get('inference_time_minutes', 0),
            'final_result': result,
            'gpu_available': torch.cuda.is_available(),
            'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available() and memory_monitor.memory_data:
            # Add GPU memory statistics for all GPUs
            gpu_df = pd.DataFrame(memory_monitor.memory_data)
            for gpu_id in range(num_gpus):
                # Use reserved memory for statistics
                if f'gpu_{gpu_id}_reserved_gb' in gpu_df.columns:
                    reserved = gpu_df[f'gpu_{gpu_id}_reserved_gb']
                    summary[f'gpu_{gpu_id}_max_memory_gb'] = reserved.max()
                    summary[f'gpu_{gpu_id}_mean_memory_gb'] = reserved.mean()
                    summary[f'gpu_{gpu_id}_final_memory_gb'] = reserved.iloc[-1]
        
        summary_path = os.path.join(output_dir, 'experiment_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info("="*50)
        logger.info("Memory consumption experiment completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Memory plots saved as PNG files")
        logger.info(f"Raw data available in CSV and pickle formats")
        logger.info("="*50)
        
        # Print summary for all GPUs
        print("\nEXPERIMENT SUMMARY:")
        print(f"Experiment: {experiment_name}")
        print(f"Memory stress level: {args.memory_stress}")
        print(f"Total duration: {summary.get('experiment_duration_seconds', 0):.1f} seconds")
        print(f"Inference/Training time: {summary.get('inference_time_seconds', 0):.1f} seconds ({summary.get('inference_time_minutes', 0):.2f} minutes)")
        print(f"Monitoring samples: {summary.get('total_monitoring_samples', 0)}")
        
        if torch.cuda.is_available() and num_gpus > 1:
            total_peak_memory = 0
            total_mean_memory = 0
            for gpu_id in range(num_gpus):
                max_mem = summary.get(f'gpu_{gpu_id}_max_memory_gb', 0)
                mean_mem = summary.get(f'gpu_{gpu_id}_mean_memory_gb', 0)
                if max_mem > 0:
                    gpu_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    utilization = (max_mem / gpu_total) * 100
                    print(f"GPU {gpu_id} - Peak: {max_mem:.2f} GB ({utilization:.1f}%), Average: {mean_mem:.2f} GB")
                    total_peak_memory += max_mem
                    total_mean_memory += mean_mem
            print(f"Total Peak Memory Across All GPUs: {total_peak_memory:.2f} GB")
            print(f"Total Average Memory Across All GPUs: {total_mean_memory:.2f} GB")
        elif torch.cuda.is_available():
            max_mem = summary.get('gpu_0_max_memory_gb', 0)
            mean_mem = summary.get('gpu_0_mean_memory_gb', 0)
            if max_mem > 0:
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                utilization = (max_mem / gpu_total) * 100
                print(f"GPU 0 - Peak: {max_mem:.2f} GB ({utilization:.1f}%), Average: {mean_mem:.2f} GB")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        tracker.finalize(status="failed")
        raise
    else:
        tracker.finalize(status="completed")
        
    logger.info("Experiment finished.")