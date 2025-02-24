# tms_efield_prediction/utils/debug/hooks.py
import time
import traceback
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

from .context import DebugContext, PipelineDebugContext, CircularBuffer, PipelineDebugState


class DebugHook:
    """Base class for debug hooks."""
    
    def __init__(self, debug_context: DebugContext):
        """Initialize the debug hook.
        
        Args:
            debug_context: Debug configuration
        """
        self.context = debug_context
        self.enabled = debug_context.verbosity_level > 0
        self.sampling_rate = debug_context.sampling_rate
        self._sample_counter = 0
    
    def should_sample(self) -> bool:
        """Check if this call should be sampled based on sampling rate.
        
        Returns:
            bool: True if should sample, False otherwise
        """
        if not self.enabled:
            return False
            
        # Always sample on sampling_rate=1.0
        if self.sampling_rate >= 1.0:
            return True
            
        # Otherwise sample based on counter
        self._sample_counter += 1
        threshold = int(1.0 / self.sampling_rate)
        result = (self._sample_counter % threshold) == 0
        
        # Reset counter periodically to avoid overflow
        if self._sample_counter > 10000:
            self._sample_counter = 0
            
        return result


class PipelineDebugHook(DebugHook):
    """Debug hook for pipeline operations."""
    
    def __init__(self, debug_context: PipelineDebugContext):
        """Initialize the pipeline debug hook.
        
        Args:
            debug_context: Pipeline debug configuration
        """
        super().__init__(debug_context)
        self.history = CircularBuffer(debug_context.history_buffer_size)
        self.pipeline_context = debug_context
        
    def record_state(self, state: PipelineDebugState):
        """Record a pipeline debug state.
        
        Args:
            state: Pipeline debug state to record
        """
        if not self.should_sample():
            return
            
        # Create snapshot and add to history
        snapshot = state.create_snapshot()
        self.history.append(snapshot)
    
    def record_event(self, event_type: str, data: Dict[str, Any]):
        """Record a pipeline event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if not self.should_sample():
            return
            
        # Create event record
        event = {
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        
        self.history.append(event)
    
    def record_error(self, error: Exception, context: Dict[str, Any]):
        """Record an error.
        
        Args:
            error: Exception that occurred
            context: Error context
        """
        # Always record errors regardless of sampling
        error_record = {
            'type': 'error',
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_msg': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        }
        
        self.history.append(error_record)
    
    def clear_history(self):
        """Clear the debug history."""
        self.history.clear()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the debug history.
        
        Returns:
            List[Dict[str, Any]]: Debug history
        """
        return self.history.get_all()