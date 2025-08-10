#!/usr/bin/env python3
# analyze_memory_data.py
"""
Analysis script for memory consumption experiment data.
Can be used to re-analyze saved memory data and create custom visualizations.
UPDATED: Better handling for multi-GPU setups and RTX 2080 Ti optimizations.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server/SSH usage
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import yaml

def load_memory_data(experiment_dir):
    """Load memory data from experiment directory."""
    
    # Try to load pickle data first (most complete)
    pickle_path = os.path.join(experiment_dir, 'memory_logs', 'memory_data.pkl')
    if os.path.exists(pickle_path):
        print(f"Loading data from {pickle_path}")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    # Fall back to CSV files
    gpu_csv_path = os.path.join(experiment_dir, 'memory_logs', 'gpu_memory_data.csv')
    system_csv_path = os.path.join(experiment_dir, 'memory_logs', 'system_memory_data.csv')
    
    data = {}
    
    if os.path.exists(gpu_csv_path):
        print(f"Loading GPU data from {gpu_csv_path}")
        gpu_df = pd.read_csv(gpu_csv_path)
        data['gpu_memory'] = gpu_df.to_dict('records')
    
    if os.path.exists(system_csv_path):
        print(f"Loading system data from {system_csv_path}")
        system_df = pd.read_csv(system_csv_path)
        data['system_memory'] = system_df.to_dict('records')
    
    return data

def analyze_memory_patterns(data, output_dir):
    """Analyze memory usage patterns and create detailed visualizations."""
    
    # Convert to DataFrames
    gpu_df = pd.DataFrame(data.get('gpu_memory', []))
    system_df = pd.DataFrame(data.get('system_memory', []))
    
    if gpu_df.empty and system_df.empty:
        print("No data available for analysis")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create comprehensive analysis plots
    create_memory_timeline_analysis(gpu_df, system_df, output_dir)
    create_memory_distribution_analysis(gpu_df, system_df, output_dir)
    create_memory_efficiency_analysis(gpu_df, output_dir)
    create_memory_spike_analysis(gpu_df, output_dir)
    
    # Generate statistical report
    generate_memory_statistics_report(gpu_df, system_df, output_dir)

def create_memory_timeline_analysis(gpu_df, system_df, output_dir):
    """Create detailed timeline analysis of memory usage."""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Detailed Memory Timeline Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: GPU Memory Timeline with Multiple Metrics
    ax1 = axes[0]
    if not gpu_df.empty:
        gpu_cols = [col for col in gpu_df.columns if 'gpu_' in col and '_allocated_gb' in col]
        for col in gpu_cols:
            gpu_id = col.split('_')[1]
            ax1.plot(gpu_df['time'], gpu_df[col], label=f'GPU {gpu_id} Allocated', linewidth=2)
            
            # Add reserved memory if available
            reserved_col = f'gpu_{gpu_id}_reserved_gb'
            if reserved_col in gpu_df.columns:
                ax1.fill_between(gpu_df['time'], gpu_df[col], gpu_df[reserved_col], 
                               alpha=0.3, label=f'GPU {gpu_id} Reserved')
    
    ax1.set_title('GPU Memory Allocation Over Time')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Memory (GB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GPU Memory Utilization Percentage
    ax2 = axes[1]
    if not gpu_df.empty:
        util_cols = [col for col in gpu_df.columns if 'utilization_percent' in col]
        for col in util_cols:
            gpu_id = col.split('_')[1]
            ax2.plot(gpu_df['time'], gpu_df[col], label=f'GPU {gpu_id}', linewidth=2)
    
    ax2.set_title('GPU Memory Utilization Percentage')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Utilization (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Plot 3: System Memory Usage
    ax3 = axes[2]
    if not system_df.empty:
        ax3.plot(system_df['time'], system_df['used_gb'], label='Used RAM', linewidth=2, color='red')
        ax3.plot(system_df['time'], system_df['available_gb'], label='Available RAM', linewidth=2, color='green')
        ax3.fill_between(system_df['time'], 0, system_df['used_gb'], alpha=0.3, color='red')
    
    ax3.set_title('System Memory Usage')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Memory (GB)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_memory_timeline.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_memory_distribution_analysis(gpu_df, system_df, output_dir):
    """Create distribution analysis of memory usage."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Memory Usage Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: GPU Memory Distribution (Histogram)
    ax1 = axes[0, 0]
    if not gpu_df.empty:
        gpu_cols = [col for col in gpu_df.columns if 'gpu_' in col and '_allocated_gb' in col]
        for col in gpu_cols:
            gpu_id = col.split('_')[1]
            ax1.hist(gpu_df[col], bins=30, alpha=0.7, label=f'GPU {gpu_id}', density=True)
    
    ax1.set_title('GPU Memory Usage Distribution')
    ax1.set_xlabel('Memory (GB)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GPU Memory Box Plot
    ax2 = axes[0, 1]
    if not gpu_df.empty:
        gpu_data = []
        gpu_labels = []
        gpu_cols = [col for col in gpu_df.columns if 'gpu_' in col and '_allocated_gb' in col]
        for col in gpu_cols:
            gpu_id = col.split('_')[1]
            gpu_data.append(gpu_df[col].dropna())
            gpu_labels.append(f'GPU {gpu_id}')
        
        if gpu_data:
            ax2.boxplot(gpu_data, labels=gpu_labels)
    
    ax2.set_title('GPU Memory Usage Box Plot')
    ax2.set_ylabel('Memory (GB)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: System Memory Distribution
    ax3 = axes[1, 0]
    if not system_df.empty:
        ax3.hist(system_df['used_gb'], bins=30, alpha=0.7, label='Used RAM', color='red', density=True)
        ax3.hist(system_df['available_gb'], bins=30, alpha=0.7, label='Available RAM', color='green', density=True)
    
    ax3.set_title('System Memory Distribution')
    ax3.set_xlabel('Memory (GB)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Memory Usage Correlation (if multiple GPUs)
    ax4 = axes[1, 1]
    if not gpu_df.empty:
        gpu_cols = [col for col in gpu_df.columns if 'gpu_' in col and '_allocated_gb' in col]
        if len(gpu_cols) > 1:
            # Create correlation matrix
            corr_data = gpu_df[gpu_cols]
            corr_matrix = corr_data.corr()
            im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(gpu_cols)))
            ax4.set_yticks(range(len(gpu_cols)))
            ax4.set_xticklabels([f"GPU {col.split('_')[1]}" for col in gpu_cols], rotation=45)
            ax4.set_yticklabels([f"GPU {col.split('_')[1]}" for col in gpu_cols])
            
            # Add correlation values
            for i in range(len(gpu_cols)):
                for j in range(len(gpu_cols)):
                    ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                            ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
            
            plt.colorbar(im, ax=ax4, shrink=0.8)
        else:
            ax4.text(0.5, 0.5, 'Correlation analysis\nrequires multiple GPUs', 
                    ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_title('GPU Memory Usage Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_memory_efficiency_analysis(gpu_df, output_dir):
    """Analyze memory efficiency and usage patterns."""
    
    if gpu_df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Memory Efficiency Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Memory Allocation Rate (Change over time)
    ax1 = axes[0, 0]
    gpu_cols = [col for col in gpu_df.columns if 'gpu_' in col and '_allocated_gb' in col]
    for col in gpu_cols:
        gpu_id = col.split('_')[1]
        # Calculate memory change rate
        memory_diff = gpu_df[col].diff().fillna(0)
        ax1.plot(gpu_df['time'], memory_diff, label=f'GPU {gpu_id}', linewidth=1, alpha=0.7)
    
    ax1.set_title('Memory Allocation Rate (Change per Sample)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Memory Change (GB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 2: Memory Efficiency (Allocated vs Reserved)
    ax2 = axes[0, 1]
    for col in gpu_cols:
        gpu_id = col.split('_')[1]
        reserved_col = f'gpu_{gpu_id}_reserved_gb'
        if reserved_col in gpu_df.columns:
            # Calculate efficiency as allocated/reserved
            efficiency = gpu_df[col] / gpu_df[reserved_col].replace(0, np.nan)
            ax2.plot(gpu_df['time'], efficiency, label=f'GPU {gpu_id}', linewidth=2)
    
    ax2.set_title('Memory Efficiency (Allocated/Reserved Ratio)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Efficiency Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Plot 3: Memory Utilization Statistics
    ax3 = axes[1, 0]
    util_stats = []
    gpu_labels = []
    
    for col in gpu_cols:
        gpu_id = col.split('_')[1]
        util_col = f'gpu_{gpu_id}_utilization_percent'
        if util_col in gpu_df.columns:
            util_data = gpu_df[util_col].dropna()
            util_stats.append([
                util_data.mean(),
                util_data.std(),
                util_data.max(),
                util_data.min()
            ])
            gpu_labels.append(f'GPU {gpu_id}')
    
    if util_stats:
        util_stats = np.array(util_stats)
        x = np.arange(len(gpu_labels))
        width = 0.2
        
        ax3.bar(x - 1.5*width, util_stats[:, 0], width, label='Mean', alpha=0.8)
        ax3.bar(x - 0.5*width, util_stats[:, 1], width, label='Std Dev', alpha=0.8)
        ax3.bar(x + 0.5*width, util_stats[:, 2], width, label='Max', alpha=0.8)
        ax3.bar(x + 1.5*width, util_stats[:, 3], width, label='Min', alpha=0.8)
        
        ax3.set_xlabel('GPU')
        ax3.set_ylabel('Utilization (%)')
        ax3.set_title('Memory Utilization Statistics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(gpu_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Memory Pressure Timeline
    ax4 = axes[1, 1]
    for col in gpu_cols:
        gpu_id = col.split('_')[1]
        total_col = f'gpu_{gpu_id}_total_gb'
        if total_col in gpu_df.columns:
            # Calculate memory pressure as percentage of total
            total_memory = gpu_df[total_col].iloc[0]
            pressure = (gpu_df[col] / total_memory) * 100
            ax4.plot(gpu_df['time'], pressure, label=f'GPU {gpu_id}', linewidth=2)
            
            # Add warning lines
            ax4.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Warning (80%)' if gpu_id == '0' else '')
            ax4.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Critical (90%)' if gpu_id == '0' else '')
    
    ax4.set_title('Memory Pressure Timeline')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Memory Usage (% of Total)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_efficiency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_memory_spike_analysis(gpu_df, output_dir):
    """Analyze memory spikes and allocation patterns."""
    
    if gpu_df.empty:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Memory Spike Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Memory Spikes Detection
    ax1 = axes[0]
    gpu_cols = [col for col in gpu_df.columns if 'gpu_' in col and '_allocated_gb' in col]
    
    for col in gpu_cols:
        gpu_id = col.split('_')[1]
        memory_data = gpu_df[col]
        
        # Calculate rolling statistics for spike detection
        window_size = max(10, len(memory_data) // 50)  # Adaptive window size
        rolling_mean = memory_data.rolling(window=window_size, center=True).mean()
        rolling_std = memory_data.rolling(window=window_size, center=True).std()
        
        # Detect spikes (values above mean + 2*std)
        spike_threshold = rolling_mean + 2 * rolling_std
        spikes = memory_data > spike_threshold
        
        # Plot memory usage
        ax1.plot(gpu_df['time'], memory_data, label=f'GPU {gpu_id} Memory', linewidth=2)
        ax1.plot(gpu_df['time'], rolling_mean, '--', alpha=0.7, label=f'GPU {gpu_id} Rolling Mean')
        
        # Highlight spikes
        spike_times = gpu_df['time'][spikes]
        spike_values = memory_data[spikes]
        ax1.scatter(spike_times, spike_values, color='red', alpha=0.7, s=30, 
                   label=f'GPU {gpu_id} Spikes' if len(spike_times) > 0 else '')
    
    ax1.set_title('Memory Spike Detection')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Memory (GB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spike Statistics and Duration Analysis
    ax2 = axes[1]
    
    spike_stats = []
    spike_labels = []
    
    for col in gpu_cols:
        gpu_id = col.split('_')[1]
        memory_data = gpu_df[col]
        
        # Calculate spike statistics
        window_size = max(10, len(memory_data) // 50)
        rolling_mean = memory_data.rolling(window=window_size, center=True).mean()
        rolling_std = memory_data.rolling(window=window_size, center=True).std()
        spike_threshold = rolling_mean + 2 * rolling_std
        spikes = memory_data > spike_threshold
        
        # Calculate spike characteristics
        num_spikes = spikes.sum()
        if num_spikes > 0:
            spike_duration = spikes.groupby((~spikes).cumsum()).sum()
            avg_spike_duration = spike_duration[spike_duration > 0].mean()
            max_spike_value = memory_data[spikes].max()
            spike_intensity = (memory_data[spikes] - rolling_mean[spikes]).mean()
        else:
            avg_spike_duration = 0
            max_spike_value = 0
            spike_intensity = 0
        
        spike_stats.append([num_spikes, avg_spike_duration, max_spike_value, spike_intensity])
        spike_labels.append(f'GPU {gpu_id}')
    
    if spike_stats:
        spike_stats = np.array(spike_stats)
        x = np.arange(len(spike_labels))
        
        # Create subplot for spike statistics
        ax2_twin = ax2.twinx()
        
        bar1 = ax2.bar(x - 0.3, spike_stats[:, 0], 0.2, label='Spike Count', alpha=0.8, color='red')
        bar2 = ax2.bar(x - 0.1, spike_stats[:, 1], 0.2, label='Avg Duration', alpha=0.8, color='orange')
        bar3 = ax2_twin.bar(x + 0.1, spike_stats[:, 2], 0.2, label='Max Value (GB)', alpha=0.8, color='blue')
        bar4 = ax2_twin.bar(x + 0.3, spike_stats[:, 3], 0.2, label='Avg Intensity (GB)', alpha=0.8, color='green')
        
        ax2.set_xlabel('GPU')
        ax2.set_ylabel('Count / Duration', color='red')
        ax2_twin.set_ylabel('Memory (GB)', color='blue')
        ax2.set_title('Memory Spike Statistics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(spike_labels)
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_spike_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_memory_statistics_report(gpu_df, system_df, output_dir):
    """Generate a comprehensive statistical report of memory usage."""
    
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'data_summary': {},
        'gpu_statistics': {},
        'system_statistics': {},
        'efficiency_metrics': {},
        'recommendations': []
    }
    
    # Data summary
    if not gpu_df.empty:
        report['data_summary']['gpu_samples'] = len(gpu_df)
        report['data_summary']['monitoring_duration_seconds'] = gpu_df['time'].max() - gpu_df['time'].min()
        report['data_summary']['sampling_rate_hz'] = len(gpu_df) / (gpu_df['time'].max() - gpu_df['time'].min() + 1e-6)
    
    if not system_df.empty:
        report['data_summary']['system_samples'] = len(system_df)
    
    # GPU statistics
    if not gpu_df.empty:
        gpu_cols = [col for col in gpu_df.columns if 'gpu_' in col and '_allocated_gb' in col]
        
        for col in gpu_cols:
            gpu_id = col.split('_')[1]
            memory_data = gpu_df[col]
            
            gpu_stats = {
                'mean_memory_gb': memory_data.mean(),
                'max_memory_gb': memory_data.max(),
                'min_memory_gb': memory_data.min(),
                'std_memory_gb': memory_data.std(),
                'median_memory_gb': memory_data.median(),
                'memory_range_gb': memory_data.max() - memory_data.min()
            }
            
            # Add utilization statistics if available
            util_col = f'gpu_{gpu_id}_utilization_percent'
            if util_col in gpu_df.columns:
                util_data = gpu_df[util_col]
                gpu_stats.update({
                    'mean_utilization_percent': util_data.mean(),
                    'max_utilization_percent': util_data.max(),
                    'time_above_80_percent': (util_data > 80).sum() / len(util_data) * 100,
                    'time_above_90_percent': (util_data > 90).sum() / len(util_data) * 100
                })
            
            # Add total memory if available
            total_col = f'gpu_{gpu_id}_total_gb'
            if total_col in gpu_df.columns:
                total_memory = gpu_df[total_col].iloc[0]
                gpu_stats['total_memory_gb'] = total_memory
                gpu_stats['peak_utilization_percent'] = (memory_data.max() / total_memory) * 100
            
            report['gpu_statistics'][f'gpu_{gpu_id}'] = gpu_stats
    
    # System statistics
    if not system_df.empty:
        report['system_statistics'] = {
            'mean_used_memory_gb': system_df['used_gb'].mean(),
            'max_used_memory_gb': system_df['used_gb'].max(),
            'min_available_memory_gb': system_df['available_gb'].min(),
            'mean_memory_percent': system_df['percent'].mean(),
            'max_memory_percent': system_df['percent'].max()
        }
    
    # Efficiency metrics
    if not gpu_df.empty:
        gpu_cols = [col for col in gpu_df.columns if 'gpu_' in col and '_allocated_gb' in col]
        
        for col in gpu_cols:
            gpu_id = col.split('_')[1]
            reserved_col = f'gpu_{gpu_id}_reserved_gb'
            
            if reserved_col in gpu_df.columns:
                allocated = gpu_df[col]
                reserved = gpu_df[reserved_col]
                
                # Calculate efficiency metrics
                efficiency_ratio = allocated / reserved.replace(0, np.nan)
                efficiency_ratio = efficiency_ratio.dropna()
                
                if len(efficiency_ratio) > 0:
                    report['efficiency_metrics'][f'gpu_{gpu_id}'] = {
                        'mean_efficiency_ratio': efficiency_ratio.mean(),
                        'min_efficiency_ratio': efficiency_ratio.min(),
                        'memory_waste_gb': (reserved - allocated).mean()
                    }
    
    # Generate recommendations
    recommendations = []
    
    # Check for high memory usage
    for gpu_id, stats in report['gpu_statistics'].items():
        if stats.get('peak_utilization_percent', 0) > 90:
            recommendations.append(f"{gpu_id}: Peak utilization exceeded 90% ({stats['peak_utilization_percent']:.1f}%). Consider reducing batch size or model complexity.")
        
        if stats.get('time_above_80_percent', 0) > 50:
            recommendations.append(f"{gpu_id}: Memory usage above 80% for {stats['time_above_80_percent']:.1f}% of training time. Monitor for potential OOM errors.")
    
    # Check for memory efficiency
    for gpu_id, metrics in report['efficiency_metrics'].items():
        if metrics.get('mean_efficiency_ratio', 1) < 0.7:
            recommendations.append(f"{gpu_id}: Low memory efficiency (allocated/reserved = {metrics['mean_efficiency_ratio']:.2f}). Consider memory optimization techniques.")
    
    # Check system memory
    if report['system_statistics'].get('max_memory_percent', 0) > 85:
        recommendations.append("System memory usage exceeded 85%. Consider reducing concurrent processes or increasing system RAM.")
    
    if not recommendations:
        recommendations.append("Memory usage appears optimal. No specific recommendations.")
    
    report['recommendations'] = recommendations
    
    # Save report
    report_path = os.path.join(output_dir, 'memory_analysis_report.yaml')
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
    
    # Also create a human-readable summary
    summary_path = os.path.join(output_dir, 'memory_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MEMORY CONSUMPTION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {report['analysis_timestamp']}\n\n")
        
        # Data summary
        f.write("DATA SUMMARY:\n")
        f.write(f"  Monitoring Duration: {report['data_summary'].get('monitoring_duration_seconds', 0):.1f} seconds\n")
        f.write(f"  Total Samples: {report['data_summary'].get('gpu_samples', 0)}\n")
        f.write(f"  Sampling Rate: {report['data_summary'].get('sampling_rate_hz', 0):.2f} Hz\n\n")
        
        # GPU statistics
        f.write("GPU MEMORY STATISTICS:\n")
        for gpu_id, stats in report['gpu_statistics'].items():
            f.write(f"  {gpu_id.upper()}:\n")
            f.write(f"    Peak Memory: {stats['max_memory_gb']:.2f} GB\n")
            f.write(f"    Average Memory: {stats['mean_memory_gb']:.2f} GB\n")
            f.write(f"    Memory Range: {stats['memory_range_gb']:.2f} GB\n")
            if 'peak_utilization_percent' in stats:
                f.write(f"    Peak Utilization: {stats['peak_utilization_percent']:.1f}%\n")
            if 'time_above_80_percent' in stats:
                f.write(f"    Time Above 80%: {stats['time_above_80_percent']:.1f}%\n")
            f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"  {i}. {rec}\n")
    
    print(f"Memory analysis report saved to {report_path}")
    print(f"Human-readable summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze memory consumption experiment data')
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory for analysis (default: same as experiment_dir)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Experiment directory {args.experiment_dir} does not exist")
        exit(1)
    
    output_dir = args.output_dir or args.experiment_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing memory data from: {args.experiment_dir}")
    print(f"Saving analysis results to: {output_dir}")
    
    # Load and analyze data
    try:
        data = load_memory_data(args.experiment_dir)
        if not data:
            print("No memory data found in experiment directory")
            exit(1)
        
        analyze_memory_patterns(data, output_dir)
        print("\nMemory analysis completed successfully!")
        print(f"Analysis results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        exit(1)