# tms_efield_prediction/data/pipeline/loader.py

"""
TMS data loading functionality.

This module extends the general data loading infrastructure
to handle TMS E-field prediction specific data formats.
"""

import os
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import h5py
import glob
import torch
import logging

from tms_efield_prediction.utils.state.context import TMSPipelineContext, PipelineState
from tms_efield_prediction.utils.debug.hooks import DebugHook
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor
from tms_efield_prediction.data.pipeline.tms_data_types import TMSRawData, TMSProcessedData, TMSSample, TMSSplit
from tms_efield_prediction.data.formats.simnibs_io import load_mesh, load_dadt_data, load_matsimnibs, MeshData

logger = logging.getLogger(__name__)


class TMSDataLoader:
    """
    Data loader for TMS E-field prediction.

    This class handles loading of MRI, dA/dt, and E-field data
    from the specific data structure at ~/MA_Henry/data/sub-XXX.
    """

    def __init__(
        self,
        context: TMSPipelineContext,
        debug_hook: Optional[DebugHook] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        use_stacked_arrays: bool = True  # Add flag to use stacked arrays
    ):
        """
        Initialize the TMS data loader.

        Args:
            context: TMS pipeline context with configuration
            debug_hook: Optional debug hook for tracking
            resource_monitor: Optional resource monitor
            use_stacked_arrays: Whether to use pre-stacked arrays (default: True)
        """
        self.context = context
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        self.use_stacked_arrays = use_stacked_arrays
        self.state = PipelineState(
            current_phase="initialization",
            processed_data={},
            experiment_history=[]
        )

        # Derive paths based on context
        self._derive_paths()

    def _derive_paths(self) -> None:
        """
        Derive file paths based on context.
        Now supports different MRI types based on configuration.
        """
        # Ensure correct subject directory format
        subject_path = os.path.join(self.context.data_root_path, f"sub-{self.context.subject_id}")
        self.experiment_path = os.path.join(subject_path, "experiment")

        # Debug: Print directory structure
        logger.info(f"Subject path: {subject_path}")
        logger.info(f"Experiment path: {self.experiment_path}")
        
        if os.path.exists(self.experiment_path):
            logger.info(f"Experiment directory exists. Contents: {os.listdir(self.experiment_path)}")
        else:
            logger.error(f"Experiment directory does not exist: {self.experiment_path}")
            return

        # Dynamically derive bin suffix from output_shape
        bin_size = self.context.output_shape[0]  # Assuming cubic grid (all dimensions equal)
        bin_suffix = f"b{bin_size}"
        
        logger.info(f"Using bin suffix '{bin_suffix}' derived from output_shape {self.context.output_shape}")
        
        # Get MRI type from context config
        mri_type = self.context.config.get("mri_type", "dti")  # Default to DTI
        logger.info(f"Using MRI type: {mri_type}")
        
        # MRI directory
        mri_dir = os.path.join(self.experiment_path, "MRI_arrays", "torch")
        if not os.path.exists(mri_dir):
            logger.error(f"MRI directory not found: {mri_dir}")
            self.mri_path = None
            return
            
        # Determine MRI path based on type
        if mri_type.lower() == "dti":
            self.mri_path = os.path.join(mri_dir, f"mri_dti_tensor_{bin_suffix}.pt")
            logger.info(f"Looking for DTI MRI file: {self.mri_path}")
        else:  # Default to conductivity
            self.mri_path = os.path.join(mri_dir, f"mri_conductivity_{bin_suffix}.pt")
            logger.info(f"Looking for conductivity MRI file: {self.mri_path}")
        
        # Check if MRI file exists
        if os.path.exists(self.mri_path):
            logger.info(f"MRI file exists: {self.mri_path}")
        else:
            logger.warning(f"Specified MRI file not found: {self.mri_path}")
            # Try to find any alternative MRI file with matching bin size
            try:
                available_files = os.listdir(mri_dir)
                logger.info(f"Available MRI files: {available_files}")
                matching_files = [f for f in available_files if f.endswith(f"_{bin_suffix}.pt")]
                if matching_files:
                    alternative_path = os.path.join(mri_dir, matching_files[0])
                    logger.warning(f"Using alternative MRI file: {alternative_path}")
                    self.mri_path = alternative_path
                else:
                    logger.error(f"No MRI files found with bin suffix '{bin_suffix}'")
                    self.mri_path = None
            except Exception as e:
                logger.error(f"Error listing MRI directory: {e}")
                self.mri_path = None
        # Initialize lists for different file types
        self.stacked_paths = []
        self.efield_scalar_paths = [] 
        self.dadt_paths = []

        if self.use_stacked_arrays:
            # Path to stacked arrays directory
            stacked_dir = os.path.join(self.experiment_path, "stacked_arrays", "torch")
            if os.path.exists(stacked_dir):
                stacked_pattern = os.path.join(stacked_dir, f"stacked_*_{bin_suffix}.pt")
                stacked_files = sorted(glob.glob(stacked_pattern))
                
                logger.info(f"Found {len(stacked_files)} stacked array files matching pattern for subject {self.context.subject_id}")
                if len(stacked_files) == 0:
                    logger.info(f"No stacked files with pattern {stacked_pattern}, checking available files")
                    try:
                        all_stacked_files = os.listdir(stacked_dir)
                        logger.info(f"Available stacked files: {all_stacked_files}")
                    except Exception as e:
                        logger.error(f"Error listing stacked directory: {e}")
                
                for stacked_path in stacked_files:
                    # Extract index from filename (e.g., "stacked_42_b25.pt" -> "42")
                    try:
                        filename = os.path.basename(stacked_path)
                        index = filename.replace("stacked_", "").replace(f"_{bin_suffix}.pt", "")
                        
                        # Check if stacked file exists
                        if not os.path.exists(stacked_path):
                            logger.warning(f"Stacked array file not found, skipping: {stacked_path}")
                            continue
                        
                        # Add to paths list
                        self.stacked_paths.append(stacked_path)
                        
                    except Exception as e:
                        logger.warning(f"Error processing stacked array file {stacked_path}: {e}")
                        continue
                        
                if not self.stacked_paths:
                    logger.warning(f"No stacked array files found for subject {self.context.subject_id}. "
                                f"Will fall back to separate E-field and dA/dt files.")
                    self.use_stacked_arrays = False
            else:
                logger.warning(f"Stacked arrays directory not found: {stacked_dir}")
                self.use_stacked_arrays = False
                    
        # If we're not using stacked arrays or none were found, fall back to separate files
        if not self.use_stacked_arrays:
            # Create paths to all SCALAR E-field and dA/dt files
            efield_dir = os.path.join(self.experiment_path, "E_arrays", "torch")
            dadt_dir = os.path.join(self.experiment_path, "dAdt_arrays", "torch")
            
            # Debug directories
            if os.path.exists(efield_dir):
                logger.info(f"E-field directory exists: {efield_dir}")
            else:
                logger.error(f"E-field directory not found: {efield_dir}")
                
            if os.path.exists(dadt_dir):
                logger.info(f"dA/dt directory exists: {dadt_dir}")
            else:
                logger.error(f"dA/dt directory not found: {dadt_dir}")
            
            # Find all available E-field files using glob pattern
            efield_pattern = os.path.join(efield_dir, f"efield_*_{bin_suffix}.pt")
            efield_files = sorted(glob.glob(efield_pattern))
            
            logger.info(f"Found {len(efield_files)} E-field files matching pattern for subject {self.context.subject_id}")
            
            # Process each E-field file and find corresponding dA/dt file
            for efield_path in efield_files:
                # Extract index from filename (e.g., "efield_42_b30.pt" -> "42")
                try:
                    # Extract the index between "efield_" and "_b{bin_size}.pt"
                    filename = os.path.basename(efield_path)
                    index = filename.replace("efield_", "").replace(f"_{bin_suffix}.pt", "")
                    
                    # Create corresponding dA/dt path
                    dadt_path = os.path.join(dadt_dir, f"dadt_{index}_{bin_suffix}.pt")
                    
                    # Check if dA/dt file exists
                    if not os.path.exists(efield_path):
                        logger.warning(f"Scalar E-field file not found, skipping: {efield_path}")
                        continue
                    if not os.path.exists(dadt_path):
                        logger.warning(f"dA/dt file not found, skipping: {dadt_path}")
                        continue
                    
                    # Add to paths lists
                    self.efield_scalar_paths.append(efield_path)
                    self.dadt_paths.append(dadt_path)
                    
                except Exception as e:
                    logger.warning(f"Error processing E-field file {efield_path}: {e}")
                    continue

    def load_raw_data(self) -> TMSRawData:
        """
        Load raw TMS data for the configured subject.
        Now supports loading from stacked arrays.
        """
        if self.resource_monitor:
            self.resource_monitor.update_component_usage("tms_dataloader.load_raw_data", "start")
        if self.debug_hook:
            try:
                if self.debug_hook.should_sample():
                    self.debug_hook.record_event("load_raw_data_start", {"subject": self.context.subject_id})
            except AttributeError:
                pass
        self.state.current_phase = "data_loading"

        try:
            # Load MRI tensor (still required for both approaches)
            mri_tensor = None
            if os.path.exists(self.mri_path):
                mri_tensor = torch.load(self.mri_path, map_location='cpu')
                logger.info(f"Loaded MRI tensor from {self.mri_path} with shape {mri_tensor.shape}")
            else:
                logger.error(f"MRI file not found: {self.mri_path}")
                raise FileNotFoundError(f"MRI file not found: {self.mri_path}")

            # Create a container for raw data
            raw_data = TMSRawData(
                subject_id=self.context.subject_id,
                mri_mesh=None,
                dadt_data=None,
                efield_data=None,
                coil_positions=None,
                roi_center=None,
                metadata={
                    "data_paths": {
                        "mri_path": self.mri_path,
                        "stacked_paths": self.stacked_paths if self.use_stacked_arrays else [],
                        "efield_scalar_paths": self.efield_scalar_paths,
                        "dadt_paths": self.dadt_paths,
                        "using_stacked_arrays": self.use_stacked_arrays
                    }
                },
                mri_tensor=mri_tensor
            )

            if self.debug_hook and self.debug_hook.should_sample():
                self.debug_hook.record_event(
                    "load_raw_data_complete",
                    {
                        "subject": self.context.subject_id,
                        "has_mri": mri_tensor is not None,
                        "using_stacked_arrays": self.use_stacked_arrays,
                        "num_stacked_paths": len(self.stacked_paths) if self.use_stacked_arrays else 0,
                        "num_efield_scalar_paths": len(self.efield_scalar_paths),
                        "num_dadt_paths": len(self.dadt_paths)
                    }
                )

            return raw_data

        except Exception as e:
            logger.error(f"Error during raw data loading: {e}", exc_info=True)
            if self.debug_hook:
                if hasattr(self.debug_hook, 'record_error'):
                    self.debug_hook.record_error("load_raw_data_error", {
                        "subject": self.context.subject_id,
                        "error": str(e)
                    })
            raise
        finally:
            if self.resource_monitor:
                self.resource_monitor.update_component_usage("tms_dataloader.load_raw_data", "end")

    def create_sample_list(self, raw_data: TMSRawData) -> List[TMSSample]:
        """
        Create list of individual samples from raw data.
        Now supports creating samples from stacked arrays.
        """
        samples = []
        
        if self.use_stacked_arrays and self.stacked_paths:
            # Create samples from stacked array paths
            for i, stacked_path in enumerate(self.stacked_paths):
                sample = TMSSample(
                    sample_id=f"{raw_data.subject_id}_{i:04d}",
                    subject_id=raw_data.subject_id,
                    coil_position_idx=i,
                    mri_data=None,  # MRI data is already in the stacked file
                    dadt_data=None,  # dA/dt is in the stacked file
                    efield_data=None,  # E-field is in the stacked file
                    coil_position=None,
                    metadata={
                        "stacked_path": stacked_path,
                        "using_stacked_array": True
                    }
                )
                samples.append(sample)
            logger.info(f"Created {len(samples)} TMSSamples from stacked arrays for subject {raw_data.subject_id}.")
        else:
            # Fall back to separate E-field and dA/dt paths
            num_positions = min(len(self.efield_scalar_paths), len(self.dadt_paths))
            
            if num_positions == 0:
                logger.error("No valid E-field or dA/dt paths found during sample creation.")
                return []

            if len(self.efield_scalar_paths) != len(self.dadt_paths):
                logger.warning(f"Mismatch between E-field ({len(self.efield_scalar_paths)}) and dA/dt ({len(self.dadt_paths)}) paths. Using first {num_positions} files.")

            for i in range(num_positions):
                efield_scalar_path = self.efield_scalar_paths[i]
                dadt_path = self.dadt_paths[i]

                sample = TMSSample(
                    sample_id=f"{raw_data.subject_id}_{i:04d}",
                    subject_id=raw_data.subject_id,
                    coil_position_idx=i,
                    mri_data=None,
                    dadt_data=dadt_path,
                    efield_data=efield_scalar_path,
                    coil_position=None,
                    metadata={
                        "using_stacked_array": False
                    }
                )
                samples.append(sample)
            logger.info(f"Created {len(samples)} TMSSamples from separate E-field and dA/dt files for subject {raw_data.subject_id}.")
        
        return samples