"""
Simulation runner for TMS E-field prediction.

This module provides a runner implementation for TMS simulations
with explicit state management, resource monitoring, and phase tracking.
"""

import os
import time
import numpy as np
import h5py
import shutil
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from joblib import Parallel, delayed
from tqdm import tqdm

import logging

# Project imports
from tms_efield_prediction.utils.debug.hooks import DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.simulation.tms_simulation import (
    SimulationContext, 
    SimulationState,
    load_mesh_and_roi,
    get_skin_average_normal_vector,
    run_efield_sim,
    extract_save_efield
)
from tms_efield_prediction.simulation.coil_position import (
    CoilPositioningConfig,
    CoilPositionGenerator,
    batch_positions
)
from tms_efield_prediction.simulation.field_calculation import (
    FieldCalculationConfig,
    FieldCalculator
)

logging.basicConfig(level=logging.INFO)

# Create your logger
logger = logging.getLogger(__name__)

# Create default configuration factories
def default_coil_config():
    return CoilPositioningConfig()

def default_field_config():
    return FieldCalculationConfig()

@dataclass
class SimulationRunnerConfig:
    """Configuration for TMS simulation runner."""
    # Subject and paths
    workspace: str = ""
    subject_id: str = ""
    experiment_type: str = "nn"
    output_dir: str = ""
    
    # Simulation parameters
    n_cpus: int = 1
    n_batches: Optional[int] = None
    batch_index: Optional[int] = None
    max_positions: Optional[int] = None
    
    # ROI parameters
    roi_radius: float = 20.0
    
    # Flags
    save_mesh: bool = True
    clean_temp: bool = True
    
    # Complex configurations
    coil_config: CoilPositioningConfig = field(default_factory=default_coil_config)
    field_config: FieldCalculationConfig = field(default_factory=default_field_config)

class SimulationRunner:
    """Runner for TMS simulation with state management."""
    
    def __init__(
        self, 
        context: SimulationContext,
        config: SimulationRunnerConfig,
        debug_hook: Optional[DebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """
        Initialize the simulation runner.
        
        Args:
            context: Simulation context
            config: Runner configuration
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor for memory tracking
        """
        self.context = context
        self.config = config
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Initialize state
        self.state = SimulationState()
        
        # Register with resource monitor if provided
        if resource_monitor:
            resource_monitor.register_component(
                "SimulationRunner",
                self._reduce_memory
            )
    
    def _reduce_memory(self, target_reduction: float) -> None:
        """
        Reduce memory usage.
        
        Args:
            target_reduction: Target reduction percentage
        """
        # Clear any cached data that's not essential
        if self.state.matsimnibs is not None and target_reduction > 0.5:
            self.state.matsimnibs = None
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def prepare_paths(self) -> Dict[str, str]:
        """
        Prepare paths for simulation.
        
        Returns:
            Dictionary of paths
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("SimulationRunner.prepare_paths", "start")
        
        try:
            # Generate paths based on configuration
            workspace = self.config.workspace
            sub_id = self.config.subject_id
            exp_type = self.config.experiment_type
            
            # Create base paths
            sub_path = os.path.join(workspace, 'data', f'sub-{sub_id}')
            exp_path = os.path.join(sub_path, 'experiment', exp_type)
            
            # Create output paths
            if self.config.output_dir:
                output_path = self.config.output_dir
            else:
                output_path = exp_path
            
            # Create specific file paths
            paths = {
                'coil': os.path.join(workspace, 'data', 'coil', 'MagVenture_Cool-B65.ccd'),
                'tensor_nifti': os.path.join(sub_path, 'headmodel', f'd2c_sub-{sub_id}', 
                                           'dti_results_T1space', 'DTI_conf_tensor.nii.gz'),
                'mesh_roi': os.path.join(exp_path, f'sub-{sub_id}_middle_gray_matter_roi.msh'),
                'sim_tmp': os.path.join(exp_path, 'tmp'),
                'output': output_path,
                'efield_dir': os.path.join(output_path, 'efield_sims')
            }
            
            # Create necessary directories
            os.makedirs(exp_path, exist_ok=True)
            os.makedirs(paths['sim_tmp'], exist_ok=True)
            os.makedirs(paths['efield_dir'], exist_ok=True)
            
            # Update context with paths
            self.context.coil_file_path = paths['coil']
            self.context.tensor_nifti_path = paths['tensor_nifti']
            self.context.output_path = paths['output']
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "prepare_paths_complete", 
                    {
                        "subject_id": self.config.subject_id,
                        "paths": paths
                    }
                )
            
            return paths
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "SimulationRunner.prepare_paths",
                    "subject_id": self.config.subject_id
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("SimulationRunner.prepare_paths", "end")
    
    def load_data(self, paths: Dict[str, str]) -> Tuple[Any, Dict[str, np.ndarray], np.ndarray]:
        """
        Load mesh and ROI data.
        
        Args:
            paths: Dictionary of paths
            
        Returns:
            Tuple of (mesh, roi_center, skin_normal_avg)
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("SimulationRunner.load_data", "start")
        
        # Transition state
        self.state = self.state.transition_to("mesh_loading")
        
        try:
            # Load mesh and ROI
            msh, roi_center = load_mesh_and_roi(
                self.context,
                self.debug_hook,
                self.resource_monitor
            )
            
            # Get normal vector
            skin_normal_avg = get_skin_average_normal_vector(
                msh, 
                roi_center, 
                self.config.roi_radius,
                self.debug_hook,
                self.resource_monitor
            )
            
            # Store in state
            self.state.mesh_data = {
                'mesh': msh,
                'roi_center': roi_center,
                'skin_normal_avg': skin_normal_avg
            }
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "load_data_complete", 
                    {
                        "subject_id": self.config.subject_id,
                        "node_count": len(msh.nodes)
                    }
                )
            
            return msh, roi_center, skin_normal_avg
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "SimulationRunner.load_data",
                    "subject_id": self.config.subject_id
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("SimulationRunner.load_data", "end")
    
    def generate_coil_positions(self, msh, roi_center):
        """
        Generate coil positions around ROI using the full head mesh.
        
        Args:
            msh: SimNIBS mesh (this should be the full head mesh)
            roi_center: ROI center information
            
        Returns:
            Tuple of (matsimnibs, grid)
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("SimulationRunner.generate_coil_positions", "start")
        
        # Transition state
        self.state = self.state.transition_to("position_generation")
        
        try:
            # Verify that this is a full head mesh by checking for skin elements
            msh_surf = msh.crop_mesh(elm_type=2)  # Get surface elements
            skin_tags = [5, 1005]  # Standard tags for skin in SimNIBS
            
            # Check if mesh has skin elements
            has_skin = False
            for tag in skin_tags:
                try:
                    skin_elms = msh_surf.crop_mesh(tags=tag)
                    if hasattr(skin_elms.elm, 'triangles') and len(skin_elms.elm.triangles) > 0:
                        has_skin = True
                        break
                except Exception as e:
                    if self.debug_hook:
                        self.debug_hook.record_error(e, {
                            "component": "SimulationRunner.generate_coil_positions.skin_check",
                            "tag": tag
                        })
            
            if not has_skin:
                raise ValueError("The mesh does not contain skin elements (tags 5 or 1005). "
                                "Please use the full head mesh for coil positioning.")
            
            # Create position generator
            generator = CoilPositionGenerator(
                self.context,
                self.config.coil_config,
                self.debug_hook,
                self.resource_monitor
            )
            
            # Generate positions
            matsimnibs, grid = generator.generate_positions(msh, roi_center)
            
            # Store in state
            self.state.matsimnibs = matsimnibs
            self.state.coil_data = {'grid': grid}
            
            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "generate_coil_positions_complete", 
                    {
                        "matsimnibs_shape": matsimnibs.shape,
                        "grid_shape": grid.shape,
                        "subject_id": self.context.subject_id
                    }
                )
            
            return matsimnibs, grid
            
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "component": "SimulationRunner.generate_coil_positions",
                    "subject_id": self.context.subject_id
                })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("SimulationRunner.generate_coil_positions", "end")
    
    # Continue with the rest of the methods...
    # ...

def run_simulation(
    workspace: str,
    subject_id: str,
    experiment_type: str = "nn",
    output_dir: str = "",
    n_cpus: int = 1,
    n_batches: Optional[int] = None,
    batch_index: Optional[int] = None,
    max_positions: Optional[int] = None,
    debug_mode: bool = True
) -> Dict[str, Any]:
    """
    Run TMS simulation for a subject.
    
    Args:
        workspace: Workspace path
        subject_id: Subject ID
        experiment_type: Experiment type
        output_dir: Output directory
        n_cpus: Number of CPU cores to use
        n_batches: Number of batches to split simulation into
        batch_index: Index of batch to process
        max_positions: Maximum number of positions to process
        debug_mode: Whether to enable debug mode
        
    Returns:
        Dictionary with results
    """
    # Create context
    from tms_efield_prediction.utils.state.context import RetentionPolicy
    from tms_efield_prediction.utils.debug.context import PipelineDebugContext
    
    # Create debug context if in debug mode
    debug_hook = None
    if debug_mode:
        debug_context = PipelineDebugContext(
            verbosity_level=2,
            memory_limit=64000,  # 64GB
            sampling_rate=0.5,
            retention_policy=RetentionPolicy(),
            history_buffer_size=1000
        )
        debug_hook = DebugHook(debug_context)
    
    # Create resource monitor
    resource_monitor = ResourceMonitor(max_memory_gb=64)
    resource_monitor.start_monitoring()
    
    try:
        # Create simulation context
        context = SimulationContext(
            dependencies={"simnibs": "4.0"},
            config={
                "workspace": workspace,
                "subject_id": subject_id,
                "experiment_type": experiment_type,
                "output_dir": output_dir
            },
            pipeline_mode="mri_efield",
            experiment_phase="preprocessing",
            debug_mode=debug_mode,
            resource_monitor=resource_monitor,
            subject_id=subject_id,
            data_root_path=os.path.join(workspace, 'data', f'sub-{subject_id}'),
            output_path=output_dir if output_dir else os.path.join(workspace, 'data', f'sub-{subject_id}', 'experiment', experiment_type)
        )
        
        # Create configuration with separate factories
        config = SimulationRunnerConfig(
            workspace=workspace,
            subject_id=subject_id,
            experiment_type=experiment_type,
            output_dir=output_dir,
            n_cpus=n_cpus,
            n_batches=n_batches,
            batch_index=batch_index,
            max_positions=max_positions,
            coil_config=default_coil_config(),
            field_config=default_field_config()
        )
        
        # Create and run simulation
        runner = SimulationRunner(
            context,
            config,
            debug_hook,
            resource_monitor
        )
        
        return runner.run()
        
    finally:
        # Stop resource monitoring
        resource_monitor.stop_monitoring()