#!/usr/bin/env python3
"""
Runner script for TMS E-field prediction pipeline.

This script:
1. Loads SimNIBS mesh files and simulation data
2. Transforms mesh data to grid format
3. Stacks MRI and dA/dt data
4. Visualizes and validates results
5. Benchmarks memory usage
6. Implements caching for large meshes
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import h5py
import logging
import psutil
import json
import pickle
from datetime import datetime
from dataclasses import dataclass, field
import sys
# Import SimNIBS
from simnibs import mesh_io

# Project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tms_efield_prediction.utils.state.context import TMSPipelineContext, RetentionPolicy
from tms_efield_prediction.utils.debug.hooks import PipelineDebugHook, DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.data.formats.simnibs_io import load_mesh, load_dadt_data, load_matsimnibs, MeshData
from tms_efield_prediction.data.transformations.mesh_to_grid import MeshToGridTransformer
from tms_efield_prediction.data.transformations.stack_pipeline import ChannelStackingPipeline, StackingConfig
from tms_efield_prediction.data.pipeline.tms_data_types import TMSRawData, TMSProcessedData, TMSSample

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tms_runner')


@dataclass
class RunnerConfig:
    """Configuration for runner script."""
    subject_id: str
    data_root_path: str
    output_path: str
    n_bins: int = 128
    cache_dir: str = "cache"
    use_cache: bool = True
    visualize: bool = True
    benchmark: bool = True
    normalization_method: str = "minmax"
    dadt_scaling_factor: float = 1.0
    stacking_mode: str = "channel_stack"
    channel_order: List[str] = field(default_factory=lambda: ["mri", "dadt"])
    debug_mode: bool = True


class CacheManager:
    """Manages caching of intermediate data for large meshes."""
    
    def __init__(self, cache_dir: str):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_manifest = {}
        self._load_manifest()
    
    def _load_manifest(self):
        """Load cache manifest from disk."""
        manifest_path = os.path.join(self.cache_dir, "cache_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    self.cache_manifest = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache manifest: {str(e)}")
                self.cache_manifest = {}
    
    def _save_manifest(self):
        """Save cache manifest to disk."""
        manifest_path = os.path.join(self.cache_dir, "cache_manifest.json")
        try:
            with open(manifest_path, 'w') as f:
                json.dump(self.cache_manifest, f)
        except Exception as e:
            logger.warning(f"Failed to save cache manifest: {str(e)}")
    
    def get_cache_path(self, key: str) -> str:
        """Get path for a cache file."""
        return os.path.join(self.cache_dir, f"{key}.npy")
    
    def has_cache(self, key: str, metadata: Dict = None) -> bool:
        """Check if cache exists and is valid.
        
        Args:
            key: Cache key
            metadata: Optional metadata to validate cache
            
        Returns:
            True if valid cache exists
        """
        # Check if key exists in manifest
        if key not in self.cache_manifest:
            return False
        
        # Check if cache file exists
        cache_path = self.get_cache_path(key)
        if not os.path.exists(cache_path):
            return False
        
        # Validate metadata if provided
        if metadata:
            cached_metadata = self.cache_manifest[key].get("metadata", {})
            for k, v in metadata.items():
                if k not in cached_metadata or cached_metadata[k] != v:
                    return False
        
        return True
    
    def save_to_cache(self, key: str, data: np.ndarray, metadata: Dict = None) -> bool:
        """Save data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
            metadata: Optional metadata
            
        Returns:
            True if cached successfully
        """
        try:
            cache_path = self.get_cache_path(key)
            np.save(cache_path, data)
            
            self.cache_manifest[key] = {
                "timestamp": datetime.now().isoformat(),
                "shape": data.shape,
                "dtype": str(data.dtype),
                "metadata": metadata or {}
            }
            
            self._save_manifest()
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {str(e)}")
            return False
    
    def load_from_cache(self, key: str) -> Optional[np.ndarray]:
        """Load data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        try:
            if not self.has_cache(key):
                return None
                
            cache_path = self.get_cache_path(key)
            data = np.load(cache_path)
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load {key} from cache: {str(e)}")
            return None
    
    def clear_cache(self, key: Optional[str] = None):
        """Clear cache files.
        
        Args:
            key: Optional specific key to clear
        """
        if key:
            if key in self.cache_manifest:
                cache_path = self.get_cache_path(key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                del self.cache_manifest[key]
                self._save_manifest()
        else:
            # Clear all cache
            for key in self.cache_manifest:
                cache_path = self.get_cache_path(key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            self.cache_manifest = {}
            self._save_manifest()


class VisualizationUtils:
    """Utilities for visualizing TMS data."""
    
    @staticmethod
    def plot_slice(data: np.ndarray, slice_idx: int = None, axis: int = 2, title: str = "", 
                  cmap: str = "viridis", figsize: Tuple[int, int] = (10, 8)):
        """
        Plot a slice of 3D data.
        
        Args:
            data: 3D array to visualize
            slice_idx: Index for slice (if None, middle slice is used)
            axis: Axis to slice (0=x, 1=y, 2=z)
            title: Plot title
            cmap: Colormap
            figsize: Figure size
        """
        if len(data.shape) > 3:
            # Handle vector data
            if data.shape[-1] == 3:
                # Vector field - use magnitude
                plot_data = np.linalg.norm(data, axis=-1)
            else:
                # Multi-channel data - use first channel
                plot_data = data[..., 0]
        else:
            plot_data = data
        
        # Get slice
        if slice_idx is None:
            slice_idx = plot_data.shape[axis] // 2
        
        if axis == 0:
            slice_data = plot_data[slice_idx, :, :]
        elif axis == 1:
            slice_data = plot_data[:, slice_idx, :]
        else:
            slice_data = plot_data[:, :, slice_idx]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.imshow(slice_data, cmap=cmap)
        plt.colorbar(label='Value')
        plt.title(f"{title} (Slice {slice_idx})")
        plt.axis('equal')
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_comparison(data1: np.ndarray, data2: np.ndarray, slice_idx: int = None, 
                       axis: int = 2, titles: Tuple[str, str] = ("Data 1", "Data 2"), 
                       figsize: Tuple[int, int] = (15, 8)):
        """
        Plot a comparison of two datasets.
        
        Args:
            data1: First 3D array
            data2: Second 3D array
            slice_idx: Index for slice (if None, middle slice is used)
            axis: Axis to slice (0=x, 1=y, 2=z)
            titles: Titles for the two plots
            figsize: Figure size
        """
        # Handle different data types
        if len(data1.shape) > 3:
            plot_data1 = np.linalg.norm(data1, axis=-1) if data1.shape[-1] == 3 else data1[..., 0]
        else:
            plot_data1 = data1
            
        if len(data2.shape) > 3:
            plot_data2 = np.linalg.norm(data2, axis=-1) if data2.shape[-1] == 3 else data2[..., 0]
        else:
            plot_data2 = data2
        
        # Get slice
        if slice_idx is None:
            slice_idx = min(plot_data1.shape[axis], plot_data2.shape[axis]) // 2
        
        if axis == 0:
            slice_data1 = plot_data1[slice_idx, :, :]
            slice_data2 = plot_data2[slice_idx, :, :]
        elif axis == 1:
            slice_data1 = plot_data1[:, slice_idx, :]
            slice_data2 = plot_data2[:, slice_idx, :]
        else:
            slice_data1 = plot_data1[:, :, slice_idx]
            slice_data2 = plot_data2[:, :, slice_idx]
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot first dataset
        im1 = axes[0].imshow(slice_data1)
        axes[0].set_title(f"{titles[0]} (Slice {slice_idx})")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot second dataset
        im2 = axes[1].imshow(slice_data2)
        axes[1].set_title(f"{titles[1]} (Slice {slice_idx})")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_vector_field(vector_data: np.ndarray, slice_idx: int = None, axis: int = 2, 
                         title: str = "", figsize: Tuple[int, int] = (10, 8), 
                         sample_rate: int = 5):
        """
        Plot a vector field slice.
        
        Args:
            vector_data: 3D vector field data with shape [..., 3]
            slice_idx: Index for slice (if None, middle slice is used)
            axis: Axis to slice (0=x, 1=y, 2=z)
            title: Plot title
            figsize: Figure size
            sample_rate: Sampling rate for quiver plot
        """
        if len(vector_data.shape) < 4 or vector_data.shape[-1] != 3:
            raise ValueError("Expected vector data with shape [..., 3]")
        
        # Get slice
        if slice_idx is None:
            slice_idx = vector_data.shape[axis] // 2
        
        if axis == 0:
            slice_data = vector_data[slice_idx, ::sample_rate, ::sample_rate, :]
            X, Y = np.meshgrid(
                np.arange(0, vector_data.shape[2], sample_rate),
                np.arange(0, vector_data.shape[1], sample_rate)
            )
            U, V = slice_data[:, :, 1], slice_data[:, :, 2]
            
        elif axis == 1:
            slice_data = vector_data[::sample_rate, slice_idx, ::sample_rate, :]
            X, Y = np.meshgrid(
                np.arange(0, vector_data.shape[2], sample_rate),
                np.arange(0, vector_data.shape[0], sample_rate)
            )
            U, V = slice_data[:, :, 0], slice_data[:, :, 2]
            
        else:
            slice_data = vector_data[::sample_rate, ::sample_rate, slice_idx, :]
            X, Y = np.meshgrid(
                np.arange(0, vector_data.shape[1], sample_rate),
                np.arange(0, vector_data.shape[0], sample_rate)
            )
            U, V = slice_data[:, :, 0], slice_data[:, :, 1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.quiver(X, Y, U, V, np.sqrt(U**2 + V**2), cmap='viridis')
        ax.set_title(f"{title} (Slice {slice_idx})")
        plt.colorbar(label='Magnitude')
        plt.axis('equal')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_plot(fig, filename: str):
        """Save plot to file."""
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)


class MemoryBenchmark:
    """Utilities for benchmarking memory usage."""
    
    def __init__(self, name: str, debug_hook: Optional[DebugHook] = None):
        """
        Initialize memory benchmark.
        
        Args:
            name: Benchmark name
            debug_hook: Optional debug hook for tracking
        """
        self.name = name
        self.debug_hook = debug_hook
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.start_memory = 0
        self.end_memory = 0
        self.measurements = []
    
    def __enter__(self):
        """Start benchmark."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End benchmark."""
        self.end()
    
    def start(self):
        """Start benchmark measurement."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory
        self.measurements = [(0.0, self.start_memory)]
        
        # Start background thread to measure memory
        import threading
        self.running = True
        self.thread = threading.Thread(target=self._monitor_memory)
        self.thread.daemon = True
        self.thread.start()
        
        # Log start
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "benchmark_start",
                {
                    "name": self.name,
                    "start_memory_mb": self.start_memory
                }
            )
        
        return self
    
    def end(self):
        """End benchmark measurement."""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(1.0)  # Wait for thread to finish
            
        self.end_time = time.time()
        self.end_memory = self._get_memory_usage()
        self.measurements.append((self.end_time - self.start_time, self.end_memory))
        
        # Log end
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event(
                "benchmark_end",
                {
                    "name": self.name,
                    "duration": self.end_time - self.start_time,
                    "start_memory_mb": self.start_memory,
                    "end_memory_mb": self.end_memory,
                    "peak_memory_mb": self.peak_memory,
                    "memory_diff_mb": self.end_memory - self.start_memory
                }
            )
        
        return self
    
    def _monitor_memory(self):
        """Background thread to monitor memory usage."""
        while self.running:
            current_memory = self._get_memory_usage()
            current_time = time.time() - self.start_time
            
            self.measurements.append((current_time, current_memory))
            self.peak_memory = max(self.peak_memory, current_memory)
            
            time.sleep(0.1)  # 100ms sampling rate
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # MB
    
    def get_results(self) -> Dict[str, Any]:
        """Get benchmark results."""
        return {
            "name": self.name,
            "duration": (self.end_time - self.start_time) if self.end_time else None,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": self.end_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_diff_mb": self.end_memory - self.start_memory if self.end_time else None,
            "measurements": self.measurements
        }
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage over time."""
        if not self.measurements:
            return None
            
        times, memories = zip(*self.measurements)
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, memories)
        plt.xlabel('Time (s)')
        plt.ylabel('Memory Usage (MB)')
        plt.title(f'Memory Usage - {self.name}')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return plt.gcf()


def create_resource_monitor() -> ResourceMonitor:
    """Create resource monitor for memory tracking."""
    resource_monitor = ResourceMonitor()
    resource_monitor.start_monitoring()
    return resource_monitor


def create_debug_hook() -> PipelineDebugHook:
    """Create debug hook for tracking."""
    # Create a PipelineDebugContext object first
    from utils.debug.context import PipelineDebugContext, RetentionPolicy
    
    debug_context = PipelineDebugContext(
        verbosity_level=2,
        memory_limit=64000,  # 64GB
        sampling_rate=0.5,
        retention_policy=RetentionPolicy(),
        history_buffer_size=1000
    )
    
    # Then create the PipelineDebugHook with the context
    return PipelineDebugHook(debug_context)


def create_pipeline_context(config: RunnerConfig, resource_monitor: ResourceMonitor) -> TMSPipelineContext:
    """Create appropriate pipeline context."""
    return TMSPipelineContext(
        dependencies={"simnibs": "4.0"},
        config={
            "n_bins": config.n_bins,
            "mask_dilation": True,
            "dilation_iterations": 1,
            "cache_enabled": config.use_cache
        },
        pipeline_mode="mri_efield",  # Using MRI + E-field approach
        experiment_phase="preprocessing",
        debug_mode=config.debug_mode,
        resource_monitor=resource_monitor,
        subject_id=config.subject_id,
        data_root_path=config.data_root_path,
        coil_file_path="",  # Will be set later if needed
        stacking_mode=config.stacking_mode,
        normalization_method=config.normalization_method,
        output_shape=(config.n_bins, config.n_bins, config.n_bins),
        dadt_scaling_factor=config.dadt_scaling_factor
    )


def get_file_paths(config: RunnerConfig) -> Dict[str, str]:
    """
    Derive file paths for a subject.
    
    Args:
        config: Runner configuration
        
    Returns:
        Dictionary of file paths
    """
    # Structure based on provided documentation
    base_path = os.path.join(config.data_root_path, f"sub-{config.subject_id}")
    
    # Define paths for different data types
    paths = {
        "mesh": os.path.join(base_path, "headmodel", f"sub-{config.subject_id}.msh"),
        "mesh_roi": os.path.join(base_path, "experiment", "nn", f"sub-{config.subject_id}_middle_gray_matter_roi.msh"),
        "coil_positions": os.path.join(base_path, "experiment", "nn", f"sub-{config.subject_id}_matsimnibs.npy"),
        "dadt": os.path.join(base_path, "experiment", "nn", f"dAdts.h5"),
        "efields": os.path.join(base_path, "experiment", "nn", f"sub-{config.subject_id}_efields.npy"),
        "roi_center": os.path.join(base_path, "experiment", f"sub-{config.subject_id}_roi_center.mat"),
        "output_dir": os.path.join(config.output_path, f"sub-{config.subject_id}")
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(paths["output_dir"], exist_ok=True)
    
    # Create cache directory if it doesn't exist
    if config.use_cache:
        cache_dir = os.path.join(config.cache_dir, f"sub-{config.subject_id}")
        os.makedirs(cache_dir, exist_ok=True)
        paths["cache_dir"] = cache_dir
    
    return paths


def load_raw_data(
    file_paths: Dict[str, str], 
    config: RunnerConfig,
    debug_hook: Optional[PipelineDebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None,
    cache_manager: Optional[CacheManager] = None
) -> TMSRawData:
    """
    Load raw TMS data files with error handling.
    
    Args:
        file_paths: Dictionary of file paths
        config: Runner configuration
        debug_hook: Optional debug hook
        resource_monitor: Optional resource monitor
        cache_manager: Optional cache manager
        
    Returns:
        TMSRawData object
    """
    logger.log(logging.INFO,f"Loading raw data for subject {config.subject_id}")
    
    # Track timing
    start_time = time.time()
    
    # Try to load mesh data
    try:
        # Check cache first if enabled
        mesh_data = None
        if config.use_cache and cache_manager and cache_manager.has_cache(f"mesh_{config.subject_id}"):
            logger.log(logging.INFO,"Loading mesh from cache")
            mesh_data = cache_manager.load_from_cache(f"mesh_{config.subject_id}")
        
        if mesh_data is None:
            logger.log(logging.INFO,f"Loading mesh from {file_paths['mesh']}")
            mesh_data = load_mesh(file_paths["mesh"], debug_hook, resource_monitor)
            
            # Cache result if enabled
            if config.use_cache and cache_manager:
                logger.log(logging.INFO,"Caching mesh data")
                cache_manager.save_to_cache(f"mesh_{config.subject_id}", mesh_data.nodes)
    except Exception as e:
        logger.error(f"Failed to load mesh: {str(e)}")
        raise
    
    # Try to load dA/dt data
    try:
        logger.log(logging.INFO,f"Loading dA/dt data from {file_paths['dadt']}")
        dadt_data = load_dadt_data(file_paths["dadt"], debug_hook, resource_monitor)
    except Exception as e:
        logger.error(f"Failed to load dA/dt data: {str(e)}")
        # Use empty array as fallback
        dadt_data = np.array([])
    
    # Try to load coil positions
    try:
        logger.log(logging.INFO,f"Loading coil positions from {file_paths['coil_positions']}")
        coil_positions = load_matsimnibs(file_paths["coil_positions"], debug_hook, resource_monitor)
    except Exception as e:
        logger.error(f"Failed to load coil positions: {str(e)}")
        # Use empty array as fallback
        coil_positions = np.array([])
    
    # Try to load E-field data (for ground truth)
    try:
        logger.log(logging.INFO,f"Loading E-field data from {file_paths['efields']}")
        efield_data = np.load(file_paths["efields"])
    except Exception as e:
        logger.error(f"Failed to load E-field data: {str(e)}")
        # Use empty array as fallback
        efield_data = np.array([])
    
    # Create raw data container
    raw_data = TMSRawData(
        subject_id=config.subject_id,
        mri_mesh=mesh_data,
        dadt_data=dadt_data,
        efield_data=efield_data,
        coil_positions=coil_positions,
        roi_center=None,  # Will be set if available
        metadata={
            "data_paths": file_paths,
            "loading_time": time.time() - start_time
        }
    )
    
    logger.log(logging.INFO,f"Raw data loaded in {raw_data.metadata['loading_time']:.2f}s")
    
    return raw_data


def process_sample(
    raw_data: TMSRawData,
    sample_idx: int,
    config: RunnerConfig,
    context: TMSPipelineContext,
    debug_hook: Optional[PipelineDebugHook] = None,
    resource_monitor: Optional[ResourceMonitor] = None,
    cache_manager: Optional[CacheManager] = None
) -> TMSProcessedData:
    """
    Process a single TMS sample through the pipeline.
    
    Args:
        raw_data: Raw TMS data
        sample_idx: Sample index (coil position)
        config: Runner configuration
        context: Pipeline context
        debug_hook: Optional debug hook
        resource_monitor: Optional resource monitor
        cache_manager: Optional cache manager
        
    Returns:
        Processed TMS data
    """
    logger.log(logging.INFO,f"Processing sample {sample_idx} for subject {config.subject_id}")
    
    # Create mesh-to-grid transformer
    transformer = MeshToGridTransformer(
        context=context,
        debug_hook=debug_hook,
        resource_monitor=resource_monitor
    )
    
    # Get node centers and transform MRI data to grid
    node_centers = raw_data.mri_mesh.nodes[:, :3]
    node_data = next(iter(raw_data.mri_mesh.node_data.values()))  # Use first node data field
    
    # Check cache for MRI grid data
    mri_grid = None
    if config.use_cache and cache_manager:
        cache_key = f"mri_grid_{config.subject_id}_{config.n_bins}"
        if cache_manager.has_cache(cache_key):
            logger.log(logging.INFO,"Loading MRI grid from cache")
            mri_grid = cache_manager.load_from_cache(cache_key)
    
    # Transform MRI data if not cached
    if mri_grid is None:
        logger.log(logging.INFO,"Transforming MRI data to grid")
        mri_grid, mri_mask, _ = transformer.transform(node_data, node_centers, config.n_bins)
        
        # Cache result if enabled
        if config.use_cache and cache_manager:
            logger.log(logging.INFO,"Caching MRI grid")
            cache_manager.save_to_cache(cache_key, mri_grid)
    
    # Transform dA/dt data to grid for specific sample
    if raw_data.dadt_data.size > 0 and sample_idx < raw_data.dadt_data.shape[0]:
        logger.log(logging.INFO,"Transforming dA/dt data to grid")
        dadt_sample = raw_data.dadt_data[sample_idx]
        dadt_grid, _, _ = transformer.transform(dadt_sample, node_centers, config.n_bins)
    else:
        logger.warning("No dA/dt data available, using zeros")
        dadt_grid = np.zeros((config.n_bins, config.n_bins, config.n_bins, 3), dtype=np.float32)
    
    # Transform E-field data to grid for ground truth
    if raw_data.efield_data.size > 0 and sample_idx < raw_data.efield_data.shape[0]:
        logger.log(logging.INFO,"Transforming E-field data to grid")
        efield_sample = raw_data.efield_data[sample_idx]
        efield_grid, mask, _ = transformer.transform(efield_sample, node_centers, config.n_bins)
    else:
        logger.warning("No E-field data available, using zeros")
        efield_grid = np.zeros((config.n_bins, config.n_bins, config.n_bins, 3), dtype=np.float32)
        mask = np.ones((config.n_bins, config.n_bins, config.n_bins), dtype=bool)
    
    # Create stacking pipeline
    stacking_config = StackingConfig(
        normalization_method=config.normalization_method,
        dadt_scaling_factor=config.dadt_scaling_factor,
        channel_order=config.channel_order
    )
    
    stacker = ChannelStackingPipeline(
        context=context,
        debug_hook=debug_hook,
        resource_monitor=resource_monitor,
        config=stacking_config
    )
    
    # Create sample
    sample = TMSSample(
        sample_id=f"{config.subject_id}_{sample_idx}",
        subject_id=config.subject_id,
        coil_position_idx=sample_idx,
        mri_data=mri_grid,
        dadt_data=dadt_grid,
        efield_data=efield_grid,
        coil_position=raw_data.coil_positions[sample_idx] if raw_data.coil_positions.size > 0 else None,
        metadata={
            "mask": mask,
            "grid_size": config.n_bins
        }
    )
    
    # Process through stacking pipeline
    logger.log(logging.INFO,"Stacking MRI and dA/dt data")
    processed_data = stacker.process_sample(sample)
    
    return processed_data


def run_pipeline(config: RunnerConfig) -> Dict[str, Any]:
    """
    Run the complete TMS E-field prediction pipeline.
    
    Args:
        config: Runner configuration
        
    Returns:
        Dictionary with results
    """
    # Set up resource monitoring and debugging
    logger.log(logging.INFO,f"Starting pipeline for subject {config.subject_id}")
    
    # Create resource monitor
    resource_monitor = create_resource_monitor()
    
    # Create debug hook
    debug_hook = create_debug_hook() if config.debug_mode else None
    
    # Create pipeline context
    context = create_pipeline_context(config, resource_monitor)
    
    # Get file paths
    file_paths = get_file_paths(config)
    
    # Create cache manager if enabled
    cache_manager = CacheManager(file_paths["cache_dir"]) if config.use_cache else None
    
    # Results dictionary
    results = {
        "subject_id": config.subject_id,
        "config": vars(config),
        "file_paths": file_paths,
        "benchmarks": {},
        "visualizations": {}
    }
    
    try:
        # Create ROI processor and check if ROI mesh exists
        from tms_efield_prediction.data.pipeline.roi_processor import ROIProcessor
        roi_processor = ROIProcessor(context, debug_hook, resource_monitor)
        
        # Check if ROI mesh already exists
        roi_exists, roi_path = roi_processor.check_roi_mesh_exists()
        
        # Generate ROI mesh if needed
        if not roi_exists:
            logger.log(logging.INFO,"ROI mesh not found. Generating new ROI mesh...")
            with MemoryBenchmark("generate_roi_mesh", debug_hook) as bench:
                roi_result = roi_processor.generate_roi_mesh(roi_radius=20.0)
            
            results["benchmarks"]["generate_roi_mesh"] = bench.get_results()
            results["roi_processor"] = {
                "success": roi_result.success,
                "roi_mesh_path": roi_result.roi_mesh_path,
                "node_reduction": f"{roi_result.node_reduction:.2f}%"
            }
            
            if config.benchmark:
                # Save memory usage plot
                benchmark_dir = os.path.join(file_paths["output_dir"], "benchmarks")
                os.makedirs(benchmark_dir, exist_ok=True)
                plot_path = os.path.join(benchmark_dir, "roi_mesh_memory.png")
                bench.plot_memory_usage(plot_path)
                results["benchmarks"]["generate_roi_mesh"]["plot"] = plot_path
            
            if not roi_result.success:
                logger.error(f"Failed to generate ROI mesh: {roi_result.error_message}")
                raise RuntimeError(f"ROI mesh generation failed: {roi_result.error_message}")
            
            # Update ROI path for simulation
            roi_path = roi_result.roi_mesh_path
        else:
            logger.log(logging.INFO,f"Using existing ROI mesh: {roi_path}")
            results["roi_processor"] = {
                "success": True,
                "roi_mesh_path": roi_path,
                "note": "Used existing ROI mesh"
            }
        
        # Update file paths with ROI mesh
        file_paths["mesh_roi"] = roi_path
        results["file_paths"]["mesh_roi"] = roi_path
    

        # Load raw data with benchmark
        with MemoryBenchmark("load_raw_data", debug_hook) as bench:
            raw_data = load_raw_data(
                file_paths, 
                config, 
                debug_hook, 
                resource_monitor,
                cache_manager
            )
        
        results["benchmarks"]["load_raw_data"] = bench.get_results()
        
        if config.benchmark:
            # Save memory usage plot
            benchmark_dir = os.path.join(file_paths["output_dir"], "benchmarks")
            os.makedirs(benchmark_dir, exist_ok=True)
            plot_path = os.path.join(benchmark_dir, "load_data_memory.png")
            bench.plot_memory_usage(plot_path)
            results["benchmarks"]["load_raw_data"]["plot"] = plot_path
        
        # Process sample with benchmark
        sample_idx = 0  # Process first sample
        with MemoryBenchmark("process_sample", debug_hook) as bench:
            processed_data = process_sample(
                raw_data, 
                sample_idx, 
                config, 
                context, 
                debug_hook, 
                resource_monitor,
                cache_manager
            )
        
        results["benchmarks"]["process_sample"] = bench.get_results()
        
        if config.benchmark:
            # Save memory usage plot
            plot_path = os.path.join(benchmark_dir, "process_sample_memory.png")
            bench.plot_memory_usage(plot_path)
            results["benchmarks"]["process_sample"]["plot"] = plot_path
        
        # Save results
        output_npy = os.path.join(file_paths["output_dir"], f"processed_sample_{sample_idx}.npy")
        np.save(output_npy, processed_data.input_features)
        results["output_path"] = output_npy
        
        # Visualize results if enabled
        if config.visualize:
            logger.log(logging.INFO,"Generating visualizations")
            
            # Create visualization directory
            viz_dir = os.path.join(file_paths["output_dir"], "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Extract data
            mri_channel = processed_data.input_features[..., 0]
            dadt_magnitude = np.linalg.norm(processed_data.input_features[..., 1:4], axis=-1) if processed_data.input_features.shape[-1] >= 4 else None
            efield_magnitude = np.linalg.norm(processed_data.target_efield, axis=-1) if processed_data.target_efield is not None else None
            
            # Visualize MRI
            mri_fig = VisualizationUtils.plot_slice(mri_channel, title="MRI")
            mri_path = os.path.join(viz_dir, "mri_slice.png")
            VisualizationUtils.save_plot(mri_fig, mri_path)
            results["visualizations"]["mri"] = mri_path
            
            # Visualize dA/dt if available
            if dadt_magnitude is not None:
                dadt_fig = VisualizationUtils.plot_slice(dadt_magnitude, title="dA/dt Magnitude")
                dadt_path = os.path.join(viz_dir, "dadt_magnitude.png")
                VisualizationUtils.save_plot(dadt_fig, dadt_path)
                results["visualizations"]["dadt"] = dadt_path
            
            # Visualize E-field if available
            if efield_magnitude is not None:
                efield_fig = VisualizationUtils.plot_slice(efield_magnitude, title="E-field Magnitude")
                efield_path = os.path.join(viz_dir, "efield_magnitude.png")
                VisualizationUtils.save_plot(efield_fig, efield_path)
                results["visualizations"]["efield"] = efield_path
            
            # Visualize stacked data
            stacked_fig = VisualizationUtils.plot_comparison(
                mri_channel, 
                dadt_magnitude if dadt_magnitude is not None else np.zeros_like(mri_channel),
                titles=("MRI", "dA/dt Magnitude")
            )
            stacked_path = os.path.join(viz_dir, "stacked_comparison.png")
            VisualizationUtils.save_plot(stacked_fig, stacked_path)
            results["visualizations"]["stacked"] = stacked_path
            
            # Visualize vector field if available
            if processed_data.target_efield is not None:
                vector_fig = VisualizationUtils.plot_vector_field(
                    processed_data.target_efield, 
                    title="E-field Vectors"
                )
                vector_path = os.path.join(viz_dir, "efield_vectors.png")
                VisualizationUtils.save_plot(vector_fig, vector_path)
                results["visualizations"]["vector"] = vector_path
        
        # Success
        results["status"] = "success"
        logger.log(logging.INFO,f"Pipeline completed successfully for subject {config.subject_id}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        results["status"] = "error"
        results["error"] = str(e)
    
    finally:
        # Stop resource monitoring
        resource_monitor.stop_monitoring()
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TMS E-field Prediction Pipeline Runner")
    
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--data-root", required=True, help="Root directory for data")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-bins", type=int, default=128, help="Grid size (n_bins)")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--no-benchmark", action="store_true", help="Disable benchmarking")
    parser.add_argument("--norm-method", default="minmax", help="Normalization method")
    parser.add_argument("--dadt-scale", type=float, default=1.0, help="dA/dt scaling factor")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create configuration from arguments
    config = RunnerConfig(
        subject_id=args.subject,
        data_root_path=args.data_root,
        output_path=args.output,
        n_bins=args.n_bins,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        visualize=not args.no_viz,
        benchmark=not args.no_benchmark,
        normalization_method=args.norm_method,
        dadt_scaling_factor=args.dadt_scale
    )
    
    # Run pipeline
    results = run_pipeline(config)
    
    # Save results
    output_json = os.path.join(config.output_path, f"results_{config.subject_id}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    import json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super(NumpyEncoder, self).default(obj)
    
    # Remove non-serializable elements from results
    results_copy = {}
    for k, v in results.items():
        if k != "benchmarks" or not isinstance(v, dict):
            results_copy[k] = v
        else:
            results_copy[k] = {}
            for bk, bv in v.items():
                if bk == "measurements":
                    # Skip detailed measurements
                    continue
                results_copy[k][bk] = bv
    
    with open(output_json, 'w') as f:
        json.dump(results_copy, f, cls=NumpyEncoder, indent=2)
    
    logger.log(logging.INFO,f"Results saved to {output_json}")
    
    # Print summary
    if results["status"] == "success":
        print(f"Pipeline completed successfully for subject {config.subject_id}")
        if "benchmarks" in results:
            for name, bench in results["benchmarks"].items():
                print(f"  {name}: {bench.get('duration', 0):.2f}s, "
                      f"Memory: {bench.get('peak_memory_mb', 0):.2f} MB peak")
    else:
        print(f"Pipeline failed for subject {config.subject_id}: {results.get('error', 'Unknown error')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())