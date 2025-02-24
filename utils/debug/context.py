# tms_efield_prediction/utils/debug/context.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time
import threading
import queue
from collections import deque

from ..state.context import RetentionPolicy

@dataclass
class DebugContext:
    """Base class for debug configurations."""
    verbosity_level: int = 1  # 0=off, 1=basic, 2=detailed, 3=full
    memory_limit: int = 1024  # Max MB for debug data
    sampling_rate: float = 1.0  # Sampling rate for debug data 
    retention_policy: RetentionPolicy = field(default_factory=RetentionPolicy)
    
    def validate(self) -> bool:
        """Validate debug context configuration.
        
        Returns:
            bool: True if valid, raises exception otherwise
        """
        if not 0 <= self.verbosity_level <= 3:
            raise ValueError(f"Invalid verbosity level: {self.verbosity_level}")
            
        if not 0 < self.memory_limit <= 16000:  # Max 16GB for debug data
            raise ValueError(f"Invalid memory limit: {self.memory_limit}")
            
        if not 0.0 < self.sampling_rate <= 1.0:
            raise ValueError(f"Invalid sampling rate: {self.sampling_rate}")
            
        return True


@dataclass
class PipelineDebugContext(DebugContext):
    """Debug configuration for pipeline components."""
    history_buffer_size: int = 1000
    
    def validate(self) -> bool:
        """Validate pipeline debug context.
        
        Returns:
            bool: True if valid, raises exception otherwise
        """
        super().validate()
        
        if not 10 <= self.history_buffer_size <= 10000:
            raise ValueError(f"Invalid history buffer size: {self.history_buffer_size}")
            
        return True


@dataclass
class DebugState:
    """Base class for debug state tracking."""
    id: str
    context: DebugContext
    timestamp: float = field(default_factory=time.time)
    
    def create_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current debug state.
        
        Returns:
            Dict[str, Any]: Snapshot of current debug state
        """
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'context_level': self.context.verbosity_level
        }


@dataclass
class PipelineDebugState(DebugState):
    """Debug state for pipeline components."""
    metrics: Dict[str, float] = field(default_factory=dict)
    tensor_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    memory_usage: Dict[str, int] = field(default_factory=dict)
    
    def create_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current pipeline debug state.
        
        Returns:
            Dict[str, Any]: Snapshot of current pipeline debug state
        """
        base_snapshot = super().create_snapshot()
        
        # Add pipeline-specific fields based on verbosity
        if self.context.verbosity_level >= 1:
            base_snapshot['metrics'] = self.metrics.copy()
            
        if self.context.verbosity_level >= 2:
            base_snapshot['tensor_shapes'] = self.tensor_shapes.copy()
            
        if self.context.verbosity_level >= 3:
            base_snapshot['memory_usage'] = self.memory_usage.copy()
            
        return base_snapshot


class CircularBuffer:
    """Thread-safe circular buffer for debug history."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the circular buffer.
        
        Args:
            max_size: Maximum number of items to store
        """
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def append(self, item: Any):
        """Add an item to the buffer.
        
        Args:
            item: Item to add
        """
        with self.lock:
            self.buffer.append(item)
    
    def extend(self, items: List[Any]):
        """Add multiple items to the buffer.
        
        Args:
            items: Items to add
        """
        with self.lock:
            self.buffer.extend(items)
            # Truncate if needed
            if len(self.buffer) > self.buffer.maxlen:
                self.buffer = deque(
                    list(self.buffer)[-self.buffer.maxlen:],
                    maxlen=self.buffer.maxlen
                )
    
    def get_all(self) -> List[Any]:
        """Get all items in the buffer.
        
        Returns:
            List[Any]: All items in the buffer
        """
        with self.lock:
            return list(self.buffer)
    
    def clear(self):
        """Clear all items from the buffer."""
        with self.lock:
            self.buffer.clear()
    
    def __len__(self) -> int:
        """Get the number of items in the buffer.
        
        Returns:
            int: Number of items
        """
        with self.lock:
            return len(self.buffer)