#!/usr/bin/env python3
# test_system_gpu_monitor.py
"""
Test script to verify system-level GPU monitoring is working correctly.
This will help debug any issues before running the full AutoML experiment.
"""

import sys
import time
import torch
import numpy as np
from system_gpu_memory_monitor import SystemGPUMemoryMonitor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_gpu_load():
    """Create some GPU memory load for testing."""
    if not torch.cuda.is_available():
        print("No CUDA available for creating GPU load")
        return []
    
    tensors = []
    num_gpus = torch.cuda.device_count()
    
    print(f"Creating memory load on {num_gpus} GPU(s)...")
    
    for gpu_id in range(min(num_gpus, 4)):  # Test on up to 4 GPUs
        try:
            with torch.cuda.device(gpu_id):
                # Create some tensors to use memory
                tensor_size = (1000, 1000, 100)  # ~400MB per tensor
                tensor = torch.randn(tensor_size, device=f'cuda:{gpu_id}')
                tensors.append(tensor)
                print(f"  GPU {gpu_id}: Created tensor of shape {tensor_size}")
        except Exception as e:
            print(f"  GPU {gpu_id}: Error creating tensor - {e}")
    
    return tensors

def test_nvidia_smi():
    """Test if nvidia-smi is available and working."""
    import subprocess
    
    print("Testing nvidia-smi availability...")
    try:
        result = subprocess.run(['nvidia-smi', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ nvidia-smi available: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ nvidia-smi failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ nvidia-smi not available: {e}")
        return False

def test_gpu_query():
    """Test GPU memory query using nvidia-smi."""
    import subprocess
    
    print("\nTesting GPU memory query...")
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total,memory.free,memory.used',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ GPU query successful:")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 5:
                        gpu_id, name, total, free, used = parts[:5]
                        print(f"  GPU {gpu_id}: {name}")
                        print(f"    Total: {int(total)/1024:.1f} GB")
                        print(f"    Used:  {int(used)/1024:.1f} GB")
                        print(f"    Free:  {int(free)/1024:.1f} GB")
            return True
        else:
            print(f"❌ GPU query failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ GPU query error: {e}")
        return False

def main():
    print("="*60)
    print("SYSTEM GPU MONITOR TEST")
    print("="*60)
    
    # Test 1: Check nvidia-smi
    if not test_nvidia_smi():
        print("\n❌ Cannot proceed - nvidia-smi not available")
        print("Make sure NVIDIA drivers are installed and nvidia-smi is in PATH")
        return False
    
    # Test 2: Test GPU query
    if not test_gpu_query():
        print("\n❌ Cannot proceed - GPU query failed")
        return False
    
    # Test 3: Test the monitor class
    print("\n" + "="*40)
    print("TESTING SYSTEM GPU MONITOR CLASS")
    print("="*40)
    
    monitor = SystemGPUMemoryMonitor(
        interval=0.5,  # Faster sampling for test
        log_to_file=True,
        output_dir="./test_gpu_monitor"
    )
    
    if not monitor.gpu_available:
        print("❌ Monitor reports no GPUs available")
        return False
    
    print(f"✅ Monitor initialized with {monitor.num_gpus} GPU(s)")
    for gpu in monitor.gpu_info:
        print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']:.1f} GB)")
    
    # Test 4: Create some GPU load and monitor it
    print("\n" + "="*40)
    print("TESTING WITH GPU MEMORY LOAD")
    print("="*40)
    
    print("Starting monitoring...")
    monitor.start_monitoring()
    
    # Take baseline measurements
    time.sleep(2)
    print("Taking baseline measurements...")
    
    # Create GPU load
    tensors = create_gpu_load()
    print("Created GPU memory load...")
    
    # Monitor for a bit
    time.sleep(5)
    print("Monitoring with load...")
    
    # Clear GPU load
    del tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleared GPU memory load...")
    
    # Monitor for a bit more
    time.sleep(3)
    print("Monitoring after cleanup...")
    
    # Stop monitoring
    monitor.stop_monitoring()
    monitor.save_final_data()
    
    # Test 5: Analyze the results
    print("\n" + "="*40)
    print("ANALYZING RESULTS")
    print("="*40)
    
    if monitor.memory_data:
        import pandas as pd
        df = pd.DataFrame(monitor.memory_data)
        
        print(f"Collected {len(df)} samples over {df['time'].max():.1f} seconds")
        
        # Check each GPU
        for gpu_id in range(monitor.num_gpus):
            used_col = f'gpu_{gpu_id}_used_gb'
            if used_col in df.columns:
                used_memory = df[used_col]
                max_usage = used_memory.max()
                mean_usage = used_memory.mean()
                
                if max_usage > 0.1:  # If we saw significant usage
                    print(f"✅ GPU {gpu_id}: Peak {max_usage:.2f} GB, Mean {mean_usage:.2f} GB")
                else:
                    print(f"⚠️  GPU {gpu_id}: No significant memory usage detected")
        
        # Create test plots
        try:
            monitor.create_system_memory_plots("./test_gpu_monitor")
            print("✅ Plots created successfully")
        except Exception as e:
            print(f"❌ Plot creation failed: {e}")
        
        print(f"\n✅ Test completed! Check ./test_gpu_monitor/ for results")
        print("If you see memory usage data above, the monitor is working correctly!")
        
        return True
    else:
        print("❌ No memory data collected - monitor may not be working")
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "="*60)
    if success:
        print("✅ SYSTEM GPU MONITOR TEST PASSED")
        print("You can now run the full AutoML experiment with confidence!")
    else:
        print("❌ SYSTEM GPU MONITOR TEST FAILED")
        print("Please check the issues above before running the full experiment")
    print("="*60)
    
    sys.exit(0 if success else 1)