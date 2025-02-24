# tms_efield_prediction/utils/state/transitions.py
from typing import Callable, Dict, Any, Optional

from .context import PipelineState

class StateTransitionValidator:
    """Validator for pipeline state transitions."""
    
    def __init__(self):
        self.transition_validators: Dict[str, Dict[str, Callable]] = {
            'preprocessing': {
                'training': self._validate_preprocessing_to_training,
            },
            'training': {
                'evaluation': self._validate_training_to_evaluation,
            }
        }
    
    def validate_transition(self, 
                           state: PipelineState, 
                           target_phase: str) -> bool:
        """Validate a state transition.
        
        Args:
            state: Current pipeline state
            target_phase: Target phase to transition to
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        current_phase = state.current_phase
        
        # Check if transition is defined
        if current_phase not in self.transition_validators:
            return False
            
        if target_phase not in self.transition_validators[current_phase]:
            return False
            
        # Execute the validator
        validator = self.transition_validators[current_phase][target_phase]
        return validator(state)
    
    def _validate_preprocessing_to_training(self, state: PipelineState) -> bool:
        """Validate transition from preprocessing to training.
        
        Args:
            state: Current pipeline state
            
        Returns:
            bool: True if transition is valid
        """
        # Check that processed data exists and is valid
        if not state.processed_data:
            return False
            
        # Verify required data components based on pipeline mode
        # This would be expanded with specific checks for each pipeline mode
        return True
    
    def _validate_training_to_evaluation(self, state: PipelineState) -> bool:
        """Validate transition from training to evaluation.
        
        Args:
            state: Current pipeline state
            
        Returns:
            bool: True if transition is valid
        """
        # Check that model checkpoints exist
        # This would be expanded with specific model checkpoint validation
        return True