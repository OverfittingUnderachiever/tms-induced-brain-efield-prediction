# tms_efield_prediction/utils/resource/monitor.py
import os
import gc
import time
import psutil
import threading
from typing import Dict, Callable, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class MemoryThresholds:
    """Memory usage thresholds for different actions."""
    sampling_reduction: float = 0.7  # 70% - reduce sampling rate
    compression_trigger: float = 0.8  # 80% - compress historical data
    trial_pause: float = 0.9  # 90% - pause lower priority components
    early_stopping: float = 0.95  # 95% - emergency early stopping


@dataclass
class ResourceMetrics:
    """Container for resource usage metrics."""
    memory_used: int = 0
    memory_total: int = 0
    memory_percentage: float = 0
    cpu_usage: float = 0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        return self.memory_percentage > 0.9  # 90%


class ResourceMonitor:
    """Memory and resource usage monitoring and management."""
    
    def __init__(self, 
                max_memory_gb: int = 64, 
                check_interval: float = 5.0,
                thresholds: Optional[MemoryThresholds] = None):
        """Initialize the resource monitor.
        
        Args:
            max_memory_gb: Maximum available memory in GB
            check_interval: Interval in seconds between resource checks
            thresholds: Memory usage thresholds, defaults if not provided
        """
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.check_interval = check_interval
        self.thresholds = thresholds or MemoryThresholds()
        
        # Component registration
        self.components: Dict[str, Callable[[float], None]] = {}
        self.component_usage: Dict[str, int] = {}
        self.component_priority: Dict[str, int] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.last_metrics = self._get_current_metrics()
        self.history: List[ResourceMetrics] = []
        
    def start_monitoring(self):
        """Start the background monitoring thread."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
    
    def register_component(self, 
                          component_name: str, 
                          reduction_callback: Callable[[float], None],
                          priority: int = 0):
        """Register a component for memory management.
        
        Args:
            component_name: Unique name for the component
            reduction_callback: Function to call to reduce memory usage
            priority: Priority level (higher is more important)
        """
        self.components[component_name] = reduction_callback
        self.component_priority[component_name] = priority
        self.component_usage[component_name] = 0
    
    def unregister_component(self, component_name: str):
        """Unregister a component.
        
        Args:
            component_name: Component name to unregister
        """
        if component_name in self.components:
            del self.components[component_name]
            del self.component_priority[component_name]
            del self.component_usage[component_name]
    
    def update_component_usage(self, component_name: str, memory_bytes: int):
        """Update the memory usage estimate for a component.
        
        Args:
            component_name: Component name
            memory_bytes: Current memory usage in bytes
        """
        if component_name in self.component_usage:
            self.component_usage[component_name] = memory_bytes
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get the current resource usage metrics.
        
        Returns:
            ResourceMetrics: Current resource metrics
        """
        return self._get_current_metrics()
    
    def trigger_memory_reduction(self, target_percentage: float = 0.8):
        """Trigger memory reduction across all components.
        
        Args:
            target_percentage: Target memory usage percentage
        """
        current = self._get_current_metrics()
        if current.memory_percentage <= target_percentage:
            return  # Already below target
            
        # Calculate how much memory to free
        target_bytes = int(self.max_memory_bytes * target_percentage)
        current_bytes = current.memory_used
        reduction_needed = current_bytes - target_bytes
        
        if reduction_needed <= 0:
            return  # No reduction needed
            
        # Sort components by priority (lowest first)
        sorted_components = sorted(
            self.components.keys(),
            key=lambda x: self.component_priority[x]
        )
        
        # Request memory reduction from each component
        for component in sorted_components:
            if reduction_needed <= 0:
                break
                
            # Calculate reduction request based on component's usage
            component_usage = self.component_usage.get(component, 0)
            reduction_request = min(
                reduction_needed,
                int(component_usage * 0.3)  # Ask for 30% reduction
            )
            
            if reduction_request > 0:
                # Convert to percentage of component's usage
                reduction_percentage = reduction_request / max(component_usage, 1)
                
                # Call the reduction callback
                try:
                    self.components[component](reduction_percentage)
                    reduction_needed -= reduction_request
                except Exception as e:
                    print(f"Error reducing memory for {component}: {e}")
    
    def _get_current_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics.
        
        Returns:
            ResourceMetrics: Current resource metrics
        """
        # Get process and system info
        process = psutil.Process(os.getpid())
        
        # Memory information
        memory_info = process.memory_info()
        memory_used = memory_info.rss  # Resident Set Size
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_total = system_memory.total
        
        # CPU usage
        cpu_usage = process.cpu_percent(interval=0.1) / psutil.cpu_count()
        
        return ResourceMetrics(
            memory_used=memory_used,
            memory_total=min(memory_total, self.max_memory_bytes),
            memory_percentage=memory_used / self.max_memory_bytes,
            cpu_usage=cpu_usage,
            timestamp=time.time()
        )
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._get_current_metrics()
                self.last_metrics = metrics
                
                # Add to history (with simple management)
                self.history.append(metrics)
                if len(self.history) > 1000:
                    self.history = self.history[-1000:]
                
                # Check thresholds and take action if needed
                self._check_thresholds(metrics)
                
                # Sleep until next check
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval * 2)  # Sleep longer after error
    
    def _check_thresholds(self, metrics: ResourceMetrics):
        """Check resource thresholds and take action if needed.
        
        Args:
            metrics: Current resource metrics
        """
        memory_pct = metrics.memory_percentage
        
        if memory_pct >= self.thresholds.early_stopping:
            # Emergency: trigger GC and aggressive memory reduction
            gc.collect()
            self.trigger_memory_reduction(target_percentage=0.7)
            
        elif memory_pct >= self.thresholds.trial_pause:
            # Critical: pause low priority components
            self.trigger_memory_reduction(target_percentage=0.8)
            
        elif memory_pct >= self.thresholds.compression_trigger:
            # High: compress historical data
            self.trigger_memory_reduction(target_percentage=0.75)
            
        elif memory_pct >= self.thresholds.sampling_reduction:
            # Warning: reduce sampling rate
            # (This would connect with the debug system)
            pass