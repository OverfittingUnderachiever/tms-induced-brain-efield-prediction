
# tms_efield_prediction/utils/debug/history.py
import time
import threading
import json
import pickle
import gzip
import os
import queue
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from .context import CircularBuffer


class DebugHistoryManager:
    """Manager for debug history with memory pressure handling."""
    
    def __init__(self, 
                max_items: int = 10000, 
                memory_limit_mb: int = 1024,
                output_dir: Optional[str] = None):
        """Initialize the debug history manager.
        
        Args:
            max_items: Maximum number of items to store in memory
            memory_limit_mb: Maximum memory usage in MB
            output_dir: Directory for writing history files
        """
        self.buffer = CircularBuffer(max_items)
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.output_dir = output_dir
        
        # Memory tracking
        self.estimated_memory = 0
        self.compress_threshold = 0.8  # 80% of memory limit
        
        # Async processing
        self.processing_queue = None
        self.processing_thread = None
        self.processing_active = False
        
    def start_async_processing(self):
        """Start asynchronous history processing."""
        if self.processing_active:
            return
            
        self.processing_queue = threading.Queue()
        self.processing_active = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
    
    def stop_async_processing(self):
        """Stop asynchronous history processing."""
        if not self.processing_active:
            return
            
        self.processing_active = False
        if self.processing_thread:
            # Send sentinel to quit
            if self.processing_queue:
                self.processing_queue.put(None)
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
    
    def add_history_item(self, item: Dict[str, Any]):
        """Add an item to the history.
        
        Args:
            item: History item to add
        """
        # Add to buffer
        self.buffer.append(item)
        
        # Update memory estimate (very rough)
        item_size = len(pickle.dumps(item))
        self.estimated_memory += item_size
        
        # Check memory pressure
        if self.estimated_memory > self.memory_limit * self.compress_threshold:
            if self.processing_queue:
                # Schedule compression if async processing is enabled
                self.processing_queue.put(('compress', None))
            else:
                # Immediate compression
                self._compress_history()
    
    def save_history(self, filename: Optional[str] = None):
        """Save history to a file.
        
        Args:
            filename: Optional filename, generated if not provided
        """
        if not self.output_dir:
            return
            
        # Generate filename if not provided
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"debug_history_{timestamp}.json.gz"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Get all items
        items = self.buffer.get_all()
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Write compressed JSON
            with gzip.open(filepath, 'wt') as f:
                json.dump(items, f)
                
            return filepath
        except Exception as e:
            print(f"Error saving history: {e}")
            return None
    
    def _compress_history(self):
        """Compress history to reduce memory usage."""
        items = self.buffer.get_all()
        
        # Reset memory estimate
        self.estimated_memory = 0
        
        if not items:
            return
            
        # Filter and simplify large items
        compressed_items = []
        for item in items:
            # Skip very large items, keep important ones
            if item.get('type') == 'error':
                # Keep all error records
                compressed_items.append(item)
            elif item.get('type') == 'state_transition':
                # Keep state transitions
                compressed_items.append(item)
            else:
                # For other items, check if they're small enough
                item_size = len(pickle.dumps(item))
                if item_size < 10 * 1024:  # 10 KB threshold
                    compressed_items.append(item)
                else:
                    # Create simplified version for large items
                    simple_item = {
                        'type': item.get('type', 'unknown'),
                        'timestamp': item.get('timestamp', time.time()),
                        'simplified': True,
                        'original_size': item_size
                    }
                    compressed_items.append(simple_item)
        
        # Replace buffer contents
        self.buffer.clear()
        self.buffer.extend(compressed_items)
        
        # Update memory estimate
        self.estimated_memory = len(pickle.dumps(compressed_items))
    
    def _processing_loop(self):
        """Background processing loop for async operations."""
        while self.processing_active:
            try:
                item = self.processing_queue.get(timeout=1.0)
                if item is None:
                    # Sentinel to quit
                    break
                    
                command, data = item
                
                if command == 'compress':
                    self._compress_history()
                elif command == 'save':
                    self.save_history(data)  # data is filename
                    
                self.processing_queue.task_done()
            except queue.Empty:
                # Timeout, no items in queue
                pass
            except Exception as e:
                print(f"Error in history processing loop: {e}")