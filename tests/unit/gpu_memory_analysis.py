#!/usr/bin/env python3
# gpu_memory_analysis.py
"""
Script to analyze GPU memory usage from the experiment CSV data
to verify if multi-GPU training is actually happening.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_gpu_memory_data(csv_file='gpu_memory_data.csv'):
    """Analyze GPU memory usage patterns from experiment data."""
    
    try:
        # Load the data
        df = pd.read_csv(csv_file)
        print("=== GPU MEMORY USAGE ANALYSIS ===")
        print(f"Data shape: {df.shape}")
        print(f"Time range: {df['time'].min():.1f}s to {df['time'].max():.1f}s")
        print(f"Total duration: {df['time'].max() - df['time'].min():.1f} seconds")
        print(f"Samples collected: {len(df)}")
        print()
        
        # Analyze each GPU
        active_gpus = []
        gpu_stats = {}
        
        print("GPU-by-GPU Analysis:")
        print("-" * 80)
        
        for gpu_id in range(8):  # Check GPUs 0-7
            allocated_col = f'gpu_{gpu_id}_allocated_gb'
            reserved_col = f'gpu_{gpu_id}_reserved_gb'
            utilization_col = f'gpu_{gpu_id}_utilization_percent'
            
            if allocated_col in df.columns:
                allocated = df[allocated_col]
                reserved = df[reserved_col] if reserved_col in df.columns else None
                utilization = df[utilization_col] if utilization_col in df.columns else None
                
                max_allocated = allocated.max()
                mean_allocated = allocated.mean()
                final_allocated = allocated.iloc[-1]
                
                # Calculate peak utilization
                peak_utilization = (max_allocated / 10.75) * 100
                
                gpu_stats[gpu_id] = {
                    'max_allocated': max_allocated,
                    'mean_allocated': mean_allocated,
                    'final_allocated': final_allocated,
                    'peak_utilization': peak_utilization
                }
                
                print(f"GPU {gpu_id}:")
                print(f"  Max Allocated: {max_allocated:.2f} GB")
                print(f"  Mean Allocated: {mean_allocated:.2f} GB")
                print(f"  Final Allocated: {final_allocated:.2f} GB")
                print(f"  Peak Utilization: {peak_utilization:.1f}%")
                
                if reserved is not None:
                    max_reserved = reserved.max()
                    print(f"  Max Reserved: {max_reserved:.2f} GB")
                
                if utilization is not None:
                    max_util_reported = utilization.max()
                    print(f"  Max Util (reported): {max_util_reported:.1f}%")
                
                print()
                
                # Consider GPU active if it used more than 0.5 GB
                if max_allocated > 0.5:
                    active_gpus.append(gpu_id)
        
        # Summary
        print("=== SUMMARY ===")
        print(f"Active GPUs (>0.5GB usage): {active_gpus}")
        print(f"Number of active GPUs: {len(active_gpus)}")
        
        if len(active_gpus) == 1:
            print("⚠️  SINGLE GPU TRAINING DETECTED")
            print("   Only GPU 0 shows significant memory usage.")
            print("   Multi-GPU training is NOT working properly.")
        elif len(active_gpus) > 1:
            print("✅ MULTI-GPU TRAINING CONFIRMED")
            print(f"   {len(active_gpus)} GPUs are actively being used.")
            
            # Check if memory patterns are similar (indicating data parallel training)
            if len(active_gpus) >= 2:
                gpu0_max = gpu_stats[active_gpus[0]]['max_allocated']
                similar_usage = True
                for gpu_id in active_gpus[1:]:
                    gpu_max = gpu_stats[gpu_id]['max_allocated']
                    ratio = min(gpu0_max, gpu_max) / max(gpu0_max, gpu_max)
                    if ratio < 0.8:  # Less than 80% similar
                        similar_usage = False
                        break
                
                if similar_usage:
                    print("✅ Memory usage patterns are similar across GPUs (Data Parallel training)")
                else:
                    print("⚠️  Memory usage patterns differ significantly between GPUs")
        else:
            print("❌ NO SIGNIFICANT GPU USAGE DETECTED")
        
        # Calculate total memory usage across all active GPUs
        total_peak_memory = sum(gpu_stats[gpu_id]['max_allocated'] for gpu_id in active_gpus)
        total_mean_memory = sum(gpu_stats[gpu_id]['mean_allocated'] for gpu_id in active_gpus)
        
        print()
        print(f"Total Peak Memory (all active GPUs): {total_peak_memory:.2f} GB")
        print(f"Total Mean Memory (all active GPUs): {total_mean_memory:.2f} GB")
        print(f"Peak Memory per GPU (average): {total_peak_memory/max(1,len(active_gpus)):.2f} GB")
        
        return active_gpus, gpu_stats
        
    except Exception as e:
        print(f"Error analyzing GPU memory data: {e}")
        return [], {}

def plot_gpu_memory_timeline(csv_file='gpu_memory_data.csv', save_path='gpu_memory_analysis.png'):
    """Create a timeline plot of GPU memory usage."""
    
    try:
        df = pd.read_csv(csv_file)
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        
        # Plot allocated memory for each GPU
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for gpu_id in range(8):
            allocated_col = f'gpu_{gpu_id}_allocated_gb'
            if allocated_col in df.columns:
                allocated = df[allocated_col]
                if allocated.max() > 0.1:  # Only plot if GPU shows some usage
                    plt.plot(df['time'], allocated, 
                            label=f'GPU {gpu_id}', 
                            linewidth=2, 
                            color=colors[gpu_id % len(colors)])
        
        plt.title('GPU Memory Usage Timeline - Multi-GPU Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Allocated Memory (GB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line for GPU memory limit
        plt.axhline(y=10.75, color='red', linestyle='--', alpha=0.7, 
                   label='GPU Memory Limit (10.75 GB)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved to: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating timeline plot: {e}")

if __name__ == "__main__":
    # Analyze the GPU memory data
    active_gpus, gpu_stats = analyze_gpu_memory_data()
    
    # Create timeline plot
    plot_gpu_memory_timeline()
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)