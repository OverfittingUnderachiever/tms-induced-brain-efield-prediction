# tms_efield_prediction/utils/pipeline/implementation_unit.py
import time
from typing import Dict, Any, List, Optional, Callable, Generic, TypeVar, Union
from dataclasses import dataclass, field

from ..resource.monitor import ResourceMonitor
from ..debug.hooks import PipelineDebugHook
from ..debug.context import PipelineDebugState

# Type variables for generic input/output
T = TypeVar('T')
U = TypeVar('U')


@dataclass
class UnitResult(Generic[U]):
    """Result from an implementation unit with debug metadata."""
    output: U
    execution_time: float = 0.0
    debug_data: Dict[str, Any] = field(default_factory=dict)
    memory_usage: int = 0


class ImplementationUnit(Generic[T, U]):
    """Template for implementation units with built-in debugging and resource awareness."""
    
    def __init__(self, 
                transform_fn: Callable[[T], U], 
                name: str,
                debug_hook: Optional[PipelineDebugHook] = None,
                resource_monitor: Optional[ResourceMonitor] = None):
        """Initialize the implementation unit.
        
        Args:
            transform_fn: Function that transforms input to output
            name: Unique name for this unit
            debug_hook: Optional debug hook for logging
            resource_monitor: Optional resource monitor for memory management
        """
        self.transform = transform_fn
        self.name = name
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Register with resource monitor if provided
        if self.resource_monitor:
            self.resource_monitor.register_component(
                f"unit_{self.name}",
                self._reduce_memory
            )
        
        self._call_counter = 0
        self._memory_estimate = 0
    
    def __call__(self, input_data: T) -> UnitResult[U]:
        """Execute the transformation.
        
        Args:
            input_data: Input data to transform
            
        Returns:
            UnitResult containing the output and debug metadata
        """
        start_time = time.time()
        self._call_counter += 1
        
        # Prepare debug state if debug is enabled
        debug_state = None
        if self.debug_hook and self.debug_hook.should_sample():
            debug_state = PipelineDebugState(
                id=f"{self.name}_{self._call_counter}",
                context=self.debug_hook.context,
                metrics={}
            )
        
        # Execute transformation
        try:
            result = self.transform(input_data)
            execution_time = time.time() - start_time
            
            # Create result object
            unit_result = UnitResult(
                output=result,
                execution_time=execution_time,
                debug_data={},
                memory_usage=self._memory_estimate
            )
            
            # Record debug information if enabled
            if debug_state:
                debug_state.metrics = {
                    'execution_time': execution_time,
                    'call_count': self._call_counter
                }
                
                # Record the debug state
                self.debug_hook.record_state(debug_state)
                
                # Add to result
                unit_result.debug_data = debug_state.create_snapshot()
            
            # Update resource monitor if available
            if self.resource_monitor:
                # This is a very rough estimate - would be more accurate
                # in real implementation with actual memory tracking
                self._memory_estimate = 1024 * 1024  # 1MB placeholder
                self.resource_monitor.update_component_usage(
                    f"unit_{self.name}", 
                    self._memory_estimate
                )
            
            return unit_result
            
        except Exception as e:
            # Record error if debug is enabled
            if self.debug_hook:
                context = {
                    'unit_name': self.name,
                    'call_count': self._call_counter,
                    'execution_time': time.time() - start_time
                }
                self.debug_hook.record_error(e, context)
            
            # Re-raise the exception
            raise
    
    def _reduce_memory(self, target_reduction: float) -> None:
        """Callback for memory reduction requests.
        
        Args:
            target_reduction: Fraction of memory to reduce (0.0-1.0)
        """
        # In a real implementation, this would clear caches,
        # release resources, etc.
        self._memory_estimate = int(self._memory_estimate * (1.0 - target_reduction))
        
        # Update resource monitor
        if self.resource_monitor:
            self.resource_monitor.update_component_usage(
                f"unit_{self.name}", 
                self._memory_estimate
            )


class DebugAwareImplementationUnit(ImplementationUnit[T, U]):
    """Enhanced implementation unit with detailed debug tracking."""
    
    def __init__(self, 
                transform_fn: Callable[[T], U], 
                name: str,
                debug_hook: Optional[PipelineDebugHook] = None,
                resource_monitor: Optional[ResourceMonitor] = None,
                pre_validate_fn: Optional[Callable[[T], bool]] = None,
                post_validate_fn: Optional[Callable[[U], bool]] = None):
        """Initialize the debug-aware implementation unit.
        
        Args:
            transform_fn: Function that transforms input to output
            name: Unique name for this unit
            debug_hook: Optional debug hook for logging
            resource_monitor: Optional resource monitor for memory management
            pre_validate_fn: Optional function to validate input
            post_validate_fn: Optional function to validate output
        """
        super().__init__(transform_fn, name, debug_hook, resource_monitor)
        self.pre_validate = pre_validate_fn
        self.post_validate = post_validate_fn
    
    def __call__(self, input_data: T) -> UnitResult[U]:
        """Execute the transformation with validation and detailed debugging.
        
        Args:
            input_data: Input data to transform
            
        Returns:
            UnitResult containing the output and debug metadata
            
        Raises:
            ValueError: If pre or post validation fails
        """
        start_time = time.time()
        self._call_counter += 1
        
        # Prepare debug data
        debug_data = {
            'unit_name': self.name,
            'call_count': self._call_counter,
            'timestamp': start_time
        }
        
        # Pre-validation
        if self.pre_validate:
            pre_start = time.time()
            if not self.pre_validate(input_data):
                if self.debug_hook:
                    self.debug_hook.record_error(
                        ValueError(f"Pre-validation failed for {self.name}"),
                        {'phase': 'pre_validate', **debug_data}
                    )
                raise ValueError(f"Pre-validation failed for {self.name}")
                
            debug_data['pre_validate_time'] = time.time() - pre_start
        
        # Execute transformation
        try:
            transform_start = time.time()
            result = self.transform(input_data)
            transform_time = time.time() - transform_start
            debug_data['transform_time'] = transform_time
            
            # Post-validation
            if self.post_validate:
                post_start = time.time()
                if not self.post_validate(result):
                    if self.debug_hook:
                        self.debug_hook.record_error(
                            ValueError(f"Post-validation failed for {self.name}"),
                            {'phase': 'post_validate', **debug_data}
                        )
                    raise ValueError(f"Post-validation failed for {self.name}")
                    
                debug_data['post_validate_time'] = time.time() - post_start
            
            # Calculate total time
            total_time = time.time() - start_time
            debug_data['total_time'] = total_time
            
            # Record debug state if enabled
            if self.debug_hook and self.debug_hook.should_sample():
                debug_state = PipelineDebugState(
                    id=f"{self.name}_{self._call_counter}",
                    context=self.debug_hook.context,
                    metrics={
                        'execution_time': total_time,
                        'transform_time': transform_time,
                        'call_count': self._call_counter
                    }
                )
                self.debug_hook.record_state(debug_state)
            
            # Update resource monitor
            if self.resource_monitor:
                # Placeholder for actual memory tracking
                self._memory_estimate = 1024 * 1024  # 1MB placeholder
                self.resource_monitor.update_component_usage(
                    f"unit_{self.name}", 
                    self._memory_estimate
                )
            
            return UnitResult(
                output=result,
                execution_time=total_time,
                debug_data=debug_data,
                memory_usage=self._memory_estimate
            )
            
        except Exception as e:
            # Record error if debug is enabled
            if self.debug_hook:
                debug_data['error'] = str(e)
                debug_data['error_type'] = type(e).__name__
                self.debug_hook.record_error(e, debug_data)
            
            # Re-raise the exception
            raise


class UnitPipeline:
    """Chain of implementation units forming a pipeline."""
    
    def __init__(self, 
                units: List[ImplementationUnit],
                name: str,
                debug_hook: Optional[PipelineDebugHook] = None,
                resource_monitor: Optional[ResourceMonitor] = None):
        """Initialize the pipeline.
        
        Args:
            units: List of implementation units
            name: Unique name for this pipeline
            debug_hook: Optional debug hook for logging
            resource_monitor: Optional resource monitor for memory management
        """
        self.units = units
        self.name = name
        self.debug_hook = debug_hook
        self.resource_monitor = resource_monitor
        
        # Register with resource monitor if provided
        if self.resource_monitor:
            self.resource_monitor.register_component(
                f"pipeline_{self.name}",
                self._reduce_memory
            )
    
    def __call__(self, input_data: Any) -> UnitResult:
        """Execute the pipeline.
        
        Args:
            input_data: Input data for the first unit
            
        Returns:
            UnitResult from the last unit
        """
        start_time = time.time()
        
        # Debug data
        pipeline_debug = {
            'pipeline_name': self.name,
            'start_time': start_time,
            'unit_results': []
        }
        
        # Run through units
        current_data = input_data
        for i, unit in enumerate(self.units):
            unit_result = unit(current_data)
            current_data = unit_result.output
            
            # Store unit result
            pipeline_debug['unit_results'].append({
                'unit_name': unit.name,
                'execution_time': unit_result.execution_time,
                'memory_usage': unit_result.memory_usage
            })
        
        # Final timing
        total_time = time.time() - start_time
        pipeline_debug['total_time'] = total_time
        
        # Record debug event if enabled
        if self.debug_hook and self.debug_hook.should_sample():
            self.debug_hook.record_event('pipeline_execution', pipeline_debug)
        
        return UnitResult(
            output=current_data,
            execution_time=total_time,
            debug_data=pipeline_debug,
            memory_usage=sum(unit_result.get('memory_usage', 0) 
                             for unit_result in pipeline_debug['unit_results'])
        )
    
    def _reduce_memory(self, target_reduction: float) -> None:
        """Callback for memory reduction requests.
        
        Args:
            target_reduction: Fraction of memory to reduce (0.0-1.0)
        """
        # In a real implementation, this would propagate to units
        # based on their memory usage and priority
        for unit in self.units:
            if hasattr(unit, '_reduce_memory'):
                unit._reduce_memory(target_reduction)