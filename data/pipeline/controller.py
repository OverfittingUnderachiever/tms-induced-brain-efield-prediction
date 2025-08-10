# tms_efield_prediction/data/pipeline/controller.py
import os
import time
import gc
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ...utils.state.context import PipelineContext, PipelineState
from ...utils.debug.hooks import PipelineDebugHook
from ...utils.debug.context import PipelineDebugContext, RetentionPolicy
from ...utils.resource.monitor import ResourceMonitor, MemoryThresholds

from .loader import DataLoader, Dataset
from .preprocessor import Preprocessor, PreprocessingResult
from .validator import DataValidator, ValidationResult


@dataclass
class PipelineExecutionResult:
    """Result of pipeline execution."""
    success: bool
    execution_time: float
    pipeline_mode: str
    processed_dir: str
    validation_result: Optional[ValidationResult] = None
    error_message: Optional[str] = None


class PipelineController:
    """Controller for TMS E-field prediction pipeline execution."""
    
    def __init__(self,
                raw_data_dir: str,
                output_base_dir: str,
                pipeline_mode: str,
                config: Dict[str, Any],
                max_memory_gb: int = 64,
                debug_mode: bool = False):
        """Initialize the pipeline controller.
        
        Args:
            raw_data_dir: Directory with raw data
            output_base_dir: Base directory for outputs
            pipeline_mode: Pipeline mode to execute
            config: Pipeline configuration
            max_memory_gb: Maximum memory in GB
            debug_mode: Whether to enable debugging
        """
        self.raw_data_dir = raw_data_dir
        self.output_base_dir = output_base_dir
        self.pipeline_mode = pipeline_mode
        self.config = config
        self.max_memory_gb = max_memory_gb
        self.debug_mode = debug_mode
        
        # Create directory for this pipeline mode
        self.pipeline_dir = os.path.join(output_base_dir, pipeline_mode)
        os.makedirs(self.pipeline_dir, exist_ok=True)
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(
            max_memory_gb=max_memory_gb,
            check_interval=5.0,
            thresholds=MemoryThresholds()
        )
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        # Initialize debug infrastructure if enabled
        self.debug_hook = None
        if debug_mode:
            debug_context = PipelineDebugContext(
                verbosity_level=2,
                memory_limit=max_memory_gb // 4,  # 1/4 of max memory for debug
                sampling_rate=0.5,
                retention_policy=RetentionPolicy(),
                history_buffer_size=1000
            )
            self.debug_hook = PipelineDebugHook(debug_context)
        
        # Initialize pipeline context
        self.pipeline_context = PipelineContext(
            dependencies={"numpy": "1.20.0", "torch": "1.9.0"},
            config=config,
            pipeline_mode=pipeline_mode,
            experiment_phase="preprocessing",
            debug_mode=debug_mode,
            resource_monitor=self.resource_monitor
        )
        
        # Initialize pipeline state
        self.pipeline_state = PipelineState(
            current_phase="preprocessing"
        )
    
    def execute_preprocessing(self) -> PipelineExecutionResult:
        """Execute the preprocessing pipeline.
        
        Returns:
            PipelineExecutionResult: Result of pipeline execution
        """
        start_time = time.time()
        
        try:
            # Output directory for processed data
            processed_dir = os.path.join(self.pipeline_dir, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            # 1. Initialize components
            data_loader = DataLoader(
                data_dir=self.raw_data_dir,
                pipeline_context=self.pipeline_context,
                debug_hook=self.debug_hook
            )
            
            preprocessor = Preprocessor(
                pipeline_context=self.pipeline_context,
                output_dir=processed_dir,
                debug_hook=self.debug_hook
            )
            
            validator = DataValidator(
                pipeline_context=self.pipeline_context,
                debug_hook=self.debug_hook
            )
            
            # 2. Load data
            dataset = data_loader.load_raw_data()
            
            # 3. Preprocess data
            preprocessing_result = preprocessor.preprocess_dataset(
                mri_data_list=dataset.mri_data,
                orientation_list=dataset.orientation_data,
                batch_size=self.config.get("batch_size", 10)
            )
            
            # 4. Force memory cleanup
            del dataset
            preprocessor.cleanup()
            gc.collect()
            
            # 5. Validate processed data
            validation_result = validator.validate_processed_data(processed_dir)
            
            # 6. Create execution result
            execution_result = PipelineExecutionResult(
                success=validation_result.is_valid,
                execution_time=time.time() - start_time,
                pipeline_mode=self.pipeline_mode,
                processed_dir=processed_dir,
                validation_result=validation_result
            )
            
            # 7. Log result
            self._log_execution_result(execution_result)
            
            return execution_result
            
        except Exception as e:
            # Log the error
            if self.debug_hook:
                self.debug_hook.record_error(e, {
                    "pipeline_mode": self.pipeline_mode,
                    "phase": "preprocessing"
                })
            
            # Create error result
            error_result = PipelineExecutionResult(
                success=False,
                execution_time=time.time() - start_time,
                pipeline_mode=self.pipeline_mode,
                processed_dir="",
                error_message=str(e)
            )
            
            # Log result
            self._log_execution_result(error_result)
            
            return error_result
        
        finally:
            # Cleanup
            self.resource_monitor.stop_monitoring()
    
    def _log_execution_result(self, result: PipelineExecutionResult):
        """Log the execution result.
        
        Args:
            result: Execution result to log
        """
        # Create a simplified version for logging
        log_data = {
            "success": result.success,
            "execution_time": result.execution_time,
            "pipeline_mode": result.pipeline_mode,
            "processed_dir": result.processed_dir
        }
        
        if not result.success:
            if result.error_message:
                log_data["error"] = result.error_message
            elif result.validation_result:
                log_data["validation_errors"] = [
                    {"code": err.error_code, "message": err.message}
                    for err in result.validation_result.errors
                ]
        
        # Write to log file
        log_file = os.path.join(self.pipeline_dir, "execution_log.json")
        
        # Append to existing log or create new
        try:
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    logs = json.load(f)
            else:
                logs = []
                
            logs.append(log_data)
            
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"Error logging execution result: {e}")