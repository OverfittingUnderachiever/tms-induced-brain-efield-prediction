# tms_efield_prediction/utils/resource/gpu_checker.py

import subprocess
import re
import os
import psutil
import sys
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class InsufficientGPUError(Exception):
    """Exception raised when there aren't enough available GPUs."""
    pass

def check_gpu_availability() -> Tuple[List[int], float]:
    """
    Check GPU availability and return list of GPUs that are free enough to use.
    
    Returns:
        Tuple[List[int], float]: List of available GPU IDs and available memory in GB
    """
    # Run nvidia-smi to get GPU usage
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Error checking GPUs: {e}")
        return [], 0.0
    
    # Parse nvidia-smi output
    gpu_data = []
    for line in output.strip().split('\n'):
        try:
            index, utilization, memory_used, memory_total = line.strip().split(', ')
            gpu_data.append({
                'index': int(index),
                'utilization': float(utilization),
                'memory_used_mb': float(memory_used),
                'memory_total_mb': float(memory_total),
                'memory_free_mb': float(memory_total) - float(memory_used)
            })
        except ValueError:
            continue
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    available_memory_gb = system_memory.available / (1024**3)
    
    # Define thresholds for "free enough"
    MAX_UTIL = 15.0  # Maximum utilization % to consider a GPU "free"
    MIN_FREE_MEMORY_MB = 8000  # Minimum free memory in MB to consider a GPU "free"
    
    # Find free GPUs
    free_gpus = []
    for gpu in gpu_data:
        if gpu['utilization'] <= MAX_UTIL and gpu['memory_free_mb'] >= MIN_FREE_MEMORY_MB:
            free_gpus.append(gpu['index'])
    
    # Log GPU status
    logger.info(f"Found {len(gpu_data)} GPUs, {len(free_gpus)} are relatively free")
    for gpu in gpu_data:
        status = "FREE" if gpu['index'] in free_gpus else "BUSY"
        logger.info(f"GPU {gpu['index']}: {gpu['utilization']}% util, {gpu['memory_used_mb']/1024:.1f}GB used, {gpu['memory_free_mb']/1024:.1f}GB free - {status}")
    
    logger.info(f"System memory: {available_memory_gb:.1f}GB available")
    
    return free_gpus, available_memory_gb

def calculate_optimal_concurrent_trials(free_gpus: List[int], available_memory_gb: float, memory_per_trial: float = 10.0) -> int:
    """
    Calculate the optimal number of concurrent trials based on available resources.
    
    Args:
        free_gpus: List of GPU IDs that are free
        available_memory_gb: Available system memory in GB
        memory_per_trial: Estimated memory needed per trial in GB
        
    Returns:
        int: Optimal number of concurrent trials
    """
    # Calculate memory-based limit
    safety_factor = 0.7  # Use only 70% of available memory to leave buffer
    memory_buffer_gb = 10.0  # Additional buffer in GB
    safe_memory = (available_memory_gb * safety_factor) - memory_buffer_gb
    memory_based_limit = max(1, int(safe_memory / memory_per_trial))
    
    # Get the minimum of GPU count and memory-based limit
    max_concurrent = min(len(free_gpus), memory_based_limit)
    
    logger.info(f"Optimal concurrent trials: {max_concurrent} (based on {len(free_gpus)} free GPUs and {available_memory_gb:.1f}GB memory)")
    return max_concurrent

def get_visible_gpus() -> List[int]:
    """
    Get the list of GPUs that are visible to the current process.
    
    Returns:
        List[int]: List of visible GPU IDs
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # Parse the environment variable
        gpu_str = os.environ["CUDA_VISIBLE_DEVICES"]
        if gpu_str.strip():
            try:
                return [int(idx) for idx in gpu_str.split(",")]
            except ValueError:
                logger.warning(f"Invalid CUDA_VISIBLE_DEVICES: {gpu_str}")
    
    # If not set or invalid, check how many GPUs are available
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        return [int(line.strip()) for line in output.split('\n')]
    except:
        return []

def configure_gpu_environment(min_required_gpus: int = 4, max_concurrent: Optional[int] = None) -> Dict:
    """
    Configure GPU environment for optimal resource usage.
    Requires at least min_required_gpus to be available.
    
    Args:
        min_required_gpus: Minimum number of GPUs required (default: 4)
        max_concurrent: Optional override for maximum concurrent trials
        
    Returns:
        Dict: Configuration dictionary with gpu_ids, max_concurrent, etc.
        
    Raises:
        InsufficientGPUError: If fewer than min_required_gpus are available
    """
    # Check GPU availability
    free_gpus, available_memory = check_gpu_availability()
    
    # Check if we have enough free GPUs
    if len(free_gpus) < min_required_gpus:
        # Get hostname to suggest another cluster
        hostname = os.uname().nodename
        
        # Create message with alternative clusters
        if hostname == "nrgpu1":
            alternatives = "nrgpu2 or nrgpu3"
        elif hostname == "nrgpu2":
            alternatives = "nrgpu1 or nrgpu3"
        elif hostname == "nrgpu3":
            alternatives = "nrgpu1 or nrgpu2"
        else:
            alternatives = "another cluster"
            
        error_msg = (f"Insufficient GPUs available! Only {len(free_gpus)} free GPUs found, "
                     f"but {min_required_gpus} are required.\n"
                     f"Current host: {hostname}\n"
                     f"Please switch to {alternatives} for better GPU availability.")
        
        raise InsufficientGPUError(error_msg)
    
    # If no override provided, calculate optimal concurrent trials
    if max_concurrent is None:
        max_concurrent = calculate_optimal_concurrent_trials(free_gpus, available_memory)
    
    # Select GPUs to use (limit to max_concurrent)
    selected_gpus = free_gpus[:max_concurrent]
    
    # Set environment variable for selected GPUs
    if selected_gpus:
        gpu_list = ",".join(map(str, selected_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_list}")
    
    return {
        "gpu_ids": selected_gpus,
        "max_concurrent": max_concurrent,
        "available_memory_gb": available_memory
    }

# Example usage if run as script
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    
    # Run the configuration
    config = configure_gpu_environment(min_required_gpus=4)
    
    # Print the result
    print("\nGPU CONFIGURATION SUMMARY:")
    print(f"Selected GPUs: {config['gpu_ids']}")
    print(f"Concurrent trials: {config['max_concurrent']}")
    print(f"Available memory: {config['available_memory_gb']:.1f} GB")
    
    # Print command hint