# tms_efield_prediction/data/pipeline/validator.py
import numpy as np
import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass

from ...utils.debug.hooks import PipelineDebugHook
from ...utils.state.context import PipelineContext


@dataclass
class ValidationError:
    """Error from data validation."""
    error_code: str
    message: str
    context: Dict[str, Any]


@dataclass
class ValidationResult:
    """Result of data validation operation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]


class DataValidator:
    """Validator for TMS data preprocessing outputs."""
    
    def __init__(self,
                pipeline_context: PipelineContext,
                debug_hook: Optional[PipelineDebugHook] = None):
        """Initialize the data validator.
        
        Args:
            pipeline_context: Pipeline context
            debug_hook: Optional debug hook for logging
        """
        self.context = pipeline_context
        self.debug_hook = debug_hook
        self.pipeline_mode = pipeline_context.pipeline_mode
    
    def validate_processed_data(self,
                               processed_dir: str) -> ValidationResult:
        """Validate processed data.
        
        Args:
            processed_dir: Directory with processed data
            
        Returns:
            ValidationResult: Validation result with errors and warnings
        """
        errors = []
        warnings = []
        
        # Check if directory exists
        if not os.path.isdir(processed_dir):
            errors.append(ValidationError(
                error_code="DIR_NOT_FOUND",
                message=f"Processed data directory not found: {processed_dir}",
                context={"directory": processed_dir}
            ))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Select validation method based on pipeline mode
        if self.pipeline_mode == "dual_modal":
            validation_errors, validation_warnings = self._validate_dual_modal(processed_dir)
            errors.extend(validation_errors)
            warnings.extend(validation_warnings)
        elif self.pipeline_mode == "oriented_mri":
            validation_errors, validation_warnings = self._validate_oriented_mri(processed_dir)
            errors.extend(validation_errors)
            warnings.extend(validation_warnings)
        elif self.pipeline_mode == "mri_efield":
            validation_errors, validation_warnings = self._validate_mri_efield(processed_dir)
            errors.extend(validation_errors)
            warnings.extend(validation_warnings)
        else:
            errors.append(ValidationError(
                error_code="INVALID_PIPELINE",
                message=f"Unknown pipeline mode: {self.pipeline_mode}",
                context={"pipeline_mode": self.pipeline_mode}
            ))
        
        # Result is valid if no errors
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    def _validate_dual_modal(self, processed_dir: str) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate dual modal processed data.
        
        Args:
            processed_dir: Directory with processed data
            
        Returns:
            Tuple containing lists of errors and warnings
        """
        errors = []
        warnings = []
        
        # Required files
        required_files = [
            "mri_processed.npy",
            "orientation_processed.npy",
            "metadata.pkl"
        ]
        
        # Check for required files
        for filename in required_files:
            filepath = os.path.join(processed_dir, filename)
            if not os.path.exists(filepath):
                errors.append(ValidationError(
                    error_code="FILE_MISSING",
                    message=f"Required file not found: {filename}",
                    context={"filepath": filepath}
                ))
        
        # If any files are missing, stop validation
        if errors:
            return errors, warnings
        
        # Load and validate data content
        try:
            mri_data = np.load(os.path.join(processed_dir, "mri_processed.npy"))
            orientation_data = np.load(os.path.join(processed_dir, "orientation_processed.npy"))
            
            # Basic shape validation
            if len(mri_data.shape) != 4:  # (samples, x, y, z)
                errors.append(ValidationError(
                    error_code="INVALID_SHAPE",
                    message=f"MRI data has invalid shape: {mri_data.shape}, expected 4D",
                    context={"shape": mri_data.shape}
                ))
            
            if len(orientation_data.shape) != 2:  # (samples, vector_components)
                errors.append(ValidationError(
                    error_code="INVALID_SHAPE",
                    message=f"Orientation data has invalid shape: {orientation_data.shape}, expected 2D",
                    context={"shape": orientation_data.shape}
                ))
            
            # Check sample count consistency
            if mri_data.shape[0] != orientation_data.shape[0]:
                errors.append(ValidationError(
                    error_code="SAMPLE_COUNT_MISMATCH",
                    message=f"Sample count mismatch: MRI={mri_data.shape[0]}, Orientation={orientation_data.shape[0]}",
                    context={
                        "mri_samples": mri_data.shape[0],
                        "orientation_samples": orientation_data.shape[0]
                    }
                ))
            
            # Check for NaN or Inf values
            if np.isnan(mri_data).any():
                errors.append(ValidationError(
                    error_code="NAN_VALUES",
                    message="MRI data contains NaN values",
                    context={"data_type": "mri"}
                ))
            
            if np.isinf(mri_data).any():
                errors.append(ValidationError(
                    error_code="INF_VALUES",
                    message="MRI data contains infinite values",
                    context={"data_type": "mri"}
                ))
            
            if np.isnan(orientation_data).any():
                errors.append(ValidationError(
                    error_code="NAN_VALUES",
                    message="Orientation data contains NaN values",
                    context={"data_type": "orientation"}
                ))
            
            if np.isinf(orientation_data).any():
                errors.append(ValidationError(
                    error_code="INF_VALUES",
                    message="Orientation data contains infinite values",
                    context={"data_type": "orientation"}
                ))
            
            # Normalization check for orientation vectors
            orientation_config = self.context.config.get('orientation_normalization', True)
            if orientation_config and len(orientation_data.shape) == 2 and orientation_data.shape[1] == 3:
                # Check if vectors are normalized
                norms = np.linalg.norm(orientation_data, axis=1)
                if not np.allclose(norms, 1.0, atol=1e-5):
                    non_unit_count = np.sum(~np.isclose(norms, 1.0, atol=1e-5))
                    warnings.append(ValidationError(
                        error_code="NON_UNIT_VECTORS",
                        message=f"Orientation vectors not normalized: {non_unit_count}/{len(norms)} vectors",
                        context={"non_unit_count": int(non_unit_count)}
                    ))
            
        except Exception as e:
            errors.append(ValidationError(
                error_code="VALIDATION_ERROR",
                message=f"Error during data validation: {str(e)}",
                context={"error": str(e)}
            ))
            
            # Log debug information if hook is available
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "validation_target": "dual_modal",
                    "processed_dir": processed_dir
                })
        
        return errors, warnings
    
    def _validate_oriented_mri(self, processed_dir: str) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate oriented MRI processed data.
        
        Args:
            processed_dir: Directory with processed data
            
        Returns:
            Tuple containing lists of errors and warnings
        """
        # This would implement validation for oriented_mri pipeline
        # Similar to dual_modal but with oriented_mri-specific checks
        # For now, return placeholder
        return [], []
    
    def _validate_mri_efield(self, processed_dir: str) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate MRI + E-field processed data.
        
        Args:
            processed_dir: Directory with processed data
            
        Returns:
            Tuple containing lists of errors and warnings
        """
        # This would implement validation for mri_efield pipeline
        # Similar to dual_modal but with mri_efield-specific checks
        # For now, return placeholder
        return [], []


class IntegrityValidator:
    """Validator for data integrity during pipeline execution."""
    
    def __init__(self,
                debug_hook: Optional[PipelineDebugHook] = None):
        """Initialize the integrity validator.
        
        Args:
            debug_hook: Optional debug hook for logging
        """
        self.debug_hook = debug_hook
        self.validation_functions = {}
    
    def register_validation(self, 
                           name: str, 
                           validation_fn: Callable[[Any], Tuple[bool, str]]):
        """Register a validation function.
        
        Args:
            name: Unique name for this validation
            validation_fn: Function that returns (is_valid, message)
        """
        self.validation_functions[name] = validation_fn
    
    def validate(self, 
                validation_name: str, 
                data: Any) -> Tuple[bool, str]:
        """Run a specific validation.
        
        Args:
            validation_name: Name of validation to run
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if validation_name not in self.validation_functions:
            return False, f"No validation named '{validation_name}' registered"
        
        try:
            result, message = self.validation_functions[validation_name](data)
            return result, message
        except Exception as e:
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "validation_name": validation_name,
                    "data_type": type(data).__name__
                })
            return False, f"Validation error: {str(e)}"
    
    def validate_all(self, data: Any) -> Dict[str, Tuple[bool, str]]:
        """Run all registered validations.
        
        Args:
            data: Data to validate
            
        Returns:
            Dict mapping validation names to (is_valid, message) tuples
        """
        results = {}
        for name in self.validation_functions:
            results[name] = self.validate(name, data)
        return results