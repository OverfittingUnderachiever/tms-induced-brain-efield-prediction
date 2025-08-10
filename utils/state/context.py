# tms_efield_prediction/utils/state/context.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from typing_extensions import Literal
from typing import Dict, List, Tuple, Any, Optional, Union
import os  # Also adding this since it's used in the validate() method

@dataclass
class ModelContext:
    """Base context class for model and pipeline operations."""
    dependencies: Dict[str, str]  # Module name -> version
    config: Dict[str, Any]
    debug_mode: bool = False
    
    def validate(self) -> bool:
        """Validate context is properly configured.
        
        Returns:
            bool: True if valid, raises exception otherwise
        """
        # Basic validation - can be extended in subclasses
        if not self.dependencies:
            raise ValueError("Dependencies must be specified")
        if not self.config:
            raise ValueError("Configuration must be specified")
        return True


@dataclass
class RetentionPolicy:
    """Configuration for debug data retention."""
    max_history_items: int = 1000
    compression_threshold: int = 800  # Start compressing at 80% capacity
    retain_on_error: bool = True
    auto_cleanup_threshold: float = 0.9  # Cleanup when memory usage exceeds 90%


# Don't use inheritance for PipelineContext to avoid dataclass parameter ordering issues
@dataclass
class PipelineContext:
    """Context for pipeline operations with explicit resource monitoring."""
    dependencies: Dict[str, str]  # Module name -> version
    config: Dict[str, Any]
    pipeline_mode: Literal['dual_modal', 'oriented_mri', 'mri_efield']
    experiment_phase: Literal['preprocessing', 'training', 'evaluation']
    debug_mode: bool = False
    resource_monitor: Optional[Any] = None  # Will be ResourceMonitor type
    
    def validate(self) -> bool:
        """Validate pipeline context configuration.
        
        Returns:
            bool: True if valid, raises exception otherwise
        """
        # Basic validation - similar to ModelContext
        if not self.dependencies:
            raise ValueError("Dependencies must be specified")
        if not self.config:
            raise ValueError("Configuration must be specified")
        
        # Pipeline-specific validation
        if self.pipeline_mode not in ['dual_modal', 'oriented_mri', 'mri_efield']:
            raise ValueError(f"Invalid pipeline mode: {self.pipeline_mode}")
        
        if self.experiment_phase not in ['preprocessing', 'training', 'evaluation']:
            raise ValueError(f"Invalid experiment phase: {self.experiment_phase}")
            
        return True


@dataclass
class ModuleState:
    """Base state class for tracking module state."""
    version: int = 1
    data: Dict[str, Any] = field(default_factory=dict)
    
    def create_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current state.
        
        Returns:
            Dict[str, Any]: Snapshot of current state
        """
        return {
            'version': self.version,
            'data': self.data.copy()
        }

@dataclass

@dataclass
class TMSPipelineContext(PipelineContext):
    """Context specific to TMS E-field prediction pipeline."""
    dependencies: Dict[str, str]  # Module name -> version
    config: Dict[str, Any]
    pipeline_mode: Literal['dual_modal', 'oriented_mri', 'mri_efield', 'mri_dadt']  # Added 'mri_dadt'
    experiment_phase: Literal['preprocessing', 'training', 'evaluation']
    debug_mode: bool = False
    subject_id: str = ""
    data_root_path: str = ""
    coil_file_path: str = ""
    stacking_mode: str = "channel_stack"  # How MRI and dA/dt are combined
    normalization_method: str = "minmax"  # Normalization method for MRI
    output_shape: Tuple[int, int, int] = (30, 30, 30)  # Default, but should be specified explicitly  
    dadt_scaling_factor: float = 1.0  # Scaling factor for dA/dt values
    device: str = "cpu" 
    train_pct = 1.0,
    val_pct = 0.0,
    test_pct = 0.0
    def get_bin_suffix(self) -> str:
        """Derive bin suffix from output_shape parameter."""
        bin_size = self.output_shape[0]
        return f"b{bin_size}"

    def get_bin_suffix(self) -> str:
        """Derive bin suffix from output_shape parameter.
        
        Returns:
            str: Bin suffix (e.g., 'b30' for output_shape=(30,30,30))
        """
        # Assuming cubic grid (all dimensions equal)
        bin_size = self.output_shape[0]
        return f"b{bin_size}"
    def validate(self) -> bool:
        """Validate TMS-specific context."""
        basic_valid = super().validate()
        
        # Additional validation
        if not self.subject_id:
            return False
        
        if not os.path.exists(self.data_root_path):
            return False
            
        if self.coil_file_path and not os.path.exists(self.coil_file_path):
            return False
            
        return basic_valid


@dataclass
class PipelineState(ModuleState):
    """State for pipeline operations with phase tracking."""
    current_phase: str = field(default="preprocessing")
    processed_data: Dict[str, Any] = field(default_factory=dict)
    experiment_history: List[Dict] = field(default_factory=list)
    
    def transition_to(self, new_phase: str, validator_fn=None) -> 'PipelineState':
        """Transition to a new pipeline phase with validation.
        
        Args:
            new_phase: Target phase to transition to
            validator_fn: Optional function to validate the transition
            
        Returns:
            PipelineState: New pipeline state after transition
            
        Raises:
            ValueError: If transition validation fails
        """
        # Validate transition if validator provided
        if validator_fn and not validator_fn(self, new_phase):
            raise ValueError(f"Invalid transition from {self.current_phase} to {new_phase}")
        
        # Create history entry
        history_entry = {
            'from_phase': self.current_phase,
            'to_phase': new_phase,
            'state_version': self.version,
            'timestamp': None  # Will be added by tracking system
        }
        
        # Create new state
        new_state = PipelineState(
            version=self.version + 1,
            data=self.data.copy(),
            current_phase=new_phase,
            processed_data=self.processed_data.copy(),
            experiment_history=self.experiment_history + [history_entry]
        )
        
        return new_state