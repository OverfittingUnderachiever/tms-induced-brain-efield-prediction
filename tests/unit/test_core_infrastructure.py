# tests/unit/test_core_infrastructure.py
import unittest
import time
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tms_efield_prediction.utils.state.context import (
    ModelContext, PipelineContext, RetentionPolicy, PipelineState
)
from tms_efield_prediction.utils.state.transitions import StateTransitionValidator
from tms_efield_prediction.utils.resource.monitor import ResourceMonitor, MemoryThresholds
from tms_efield_prediction.utils.debug.context import (
    DebugContext, PipelineDebugContext, CircularBuffer
)
from tms_efield_prediction.utils.debug.hooks import PipelineDebugHook
from tms_efield_prediction.utils.pipeline.implementation_unit import (
    ImplementationUnit, DebugAwareImplementationUnit, UnitPipeline, UnitResult
)


class TestCoreInfrastructure(unittest.TestCase):
    """Test the core infrastructure components."""

    def setUp(self):
        """Set up test environment."""
        # Basic configuration for tests
        self.dependencies = {'numpy': '1.20.0', 'pytorch': '1.9.0'}
        self.config = {'batch_size': 32, 'learning_rate': 0.001}
        
    def test_model_context_creation(self):
        """Test creating and validating ModelContext."""
        # Create context
        context = ModelContext(
            dependencies=self.dependencies,
            config=self.config,
            debug_mode=True
        )
        
        # Test validation
        self.assertTrue(context.validate())
        
        # Test invalid context
        with self.assertRaises(ValueError):
            invalid_context = ModelContext(
                dependencies={},  # Empty dependencies should fail
                config=self.config
            )
            invalid_context.validate()
    
    def test_pipeline_context_creation(self):
        """Test creating and validating PipelineContext."""
        # Create valid context
        context = PipelineContext(
            dependencies=self.dependencies,
            config=self.config,
            debug_mode=True,
            pipeline_mode='dual_modal',
            experiment_phase='preprocessing'
        )
        
        # Test validation
        self.assertTrue(context.validate())
        
        # Test invalid pipeline mode
        with self.assertRaises(ValueError):
            invalid_context = PipelineContext(
                dependencies=self.dependencies,
                config=self.config,
                pipeline_mode='invalid_mode',  # Invalid mode
                experiment_phase='preprocessing'
            )
            invalid_context.validate()
    
    def test_pipeline_state_transition(self):
        """Test pipeline state transitions."""
        # Create initial state
        state = PipelineState(
            version=1,
            current_phase="preprocessing",
            processed_data={"test_data": [1, 2, 3]}
        )
        
        # Test transition
        new_state = state.transition_to("training")
        
        # Verify state changes
        self.assertEqual(new_state.version, 2)
        self.assertEqual(new_state.current_phase, "training")
        self.assertEqual(len(new_state.experiment_history), 1)
        self.assertEqual(new_state.experiment_history[0]['from_phase'], "preprocessing")
        self.assertEqual(new_state.experiment_history[0]['to_phase'], "training")
        
        # Test with validator function
        validator = StateTransitionValidator()
        
        # Mock validation function for testing
        def mock_validator(state, target):
            return target == "evaluation"
        
        # Test valid transition
        valid_state = new_state.transition_to("evaluation", validator_fn=mock_validator)
        self.assertEqual(valid_state.current_phase, "evaluation")
        
        # Test invalid transition
        with self.assertRaises(ValueError):
            invalid_state = valid_state.transition_to("invalid", validator_fn=mock_validator)
    
    def test_resource_monitor(self):
        """Test resource monitor functionality."""
        # Create monitor
        monitor = ResourceMonitor(max_memory_gb=1, check_interval=0.1)
        
        # Test basic metrics
        metrics = monitor.get_current_metrics()
        self.assertGreater(metrics.memory_used, 0)
        self.assertEqual(metrics.memory_total, 1024 * 1024 * 1024)  # 1GB
        
        # Test component registration
        reduction_called = {'value': False}
        
        def test_reduction(target_percentage):
            reduction_called['value'] = True
        
        monitor.register_component("test_component", test_reduction)
        
        # Update component usage
        monitor.update_component_usage("test_component", 500 * 1024 * 1024)  # 500MB
        
        # Trigger reduction
        monitor.trigger_memory_reduction(target_percentage=0.4)  # Target 40%
        
        # Verify reduction was called
        self.assertTrue(reduction_called['value'])
        
        # Test component unregistration
        monitor.unregister_component("test_component")
        self.assertNotIn("test_component", monitor.components)
    
    def test_debug_context_and_hooks(self):
        """Test debug context and hooks."""
        # Create debug context
        debug_context = PipelineDebugContext(
            verbosity_level=2,
            memory_limit=512,
            sampling_rate=0.5,
            retention_policy=RetentionPolicy(max_history_items=100),
            history_buffer_size=100
        )
        
        # Test validation
        self.assertTrue(debug_context.validate())
        
        # Create debug hook
        debug_hook = PipelineDebugHook(debug_context)
        
        # Test sampling
        # With sampling_rate=0.5, approximately half the calls should be sampled
        sample_count = sum(debug_hook.should_sample() for _ in range(100))
        self.assertTrue(30 <= sample_count <= 70)  # Allow some variance
        
        # Test event recording
        for i in range(10):
            debug_hook.record_event(f"test_event_{i}", {"data": i})
        
        # Check history
        history = debug_hook.get_history()
        self.assertGreater(len(history), 0)
        self.assertLessEqual(len(history), 10)  # Due to sampling
    
    def test_circular_buffer(self):
        """Test circular buffer functionality."""
        # Create buffer with small size for testing
        buffer = CircularBuffer(max_size=5)
        
        # Add items
        for i in range(10):
            buffer.append(i)
        
        # Check size is limited
        self.assertEqual(len(buffer), 5)
        
        # Check content (should be the last 5 items)
        items = buffer.get_all()
        self.assertEqual(items, [5, 6, 7, 8, 9])
        
        # Test clear
        buffer.clear()
        self.assertEqual(len(buffer), 0)
    
    def test_implementation_unit(self):
        """Test basic implementation unit."""
        # Create a simple transformation function
        def square(x):
            return x * x
        
        # Create unit
        unit = ImplementationUnit(
            transform_fn=square,
            name="square_unit"
        )
        
        # Test execution
        result = unit(5)
        
        # Check result
        self.assertEqual(result.output, 25)
        self.assertGreater(result.execution_time, 0)
    
    def test_debug_aware_unit(self):
        """Test debug-aware implementation unit."""
        # Create debug context and hook
        debug_context = PipelineDebugContext(
            verbosity_level=2,
            memory_limit=512,
            sampling_rate=1.0,  # Always sample for testing
            retention_policy=RetentionPolicy(),
            history_buffer_size=100
        )
        debug_hook = PipelineDebugHook(debug_context)
        
        # Create a transformation function with validation
        def square(x):
            return x * x
        
        def pre_validate(x):
            return isinstance(x, (int, float))
        
        def post_validate(x):
            return x >= 0  # Square is always non-negative
        
        # Create unit
        unit = DebugAwareImplementationUnit(
            transform_fn=square,
            name="debug_square",
            debug_hook=debug_hook,
            pre_validate_fn=pre_validate,
            post_validate_fn=post_validate
        )
        
        # Test successful execution
        result = unit(5)
        self.assertEqual(result.output, 25)
        
        # Test pre-validation failure
        with self.assertRaises(ValueError):
            unit("not a number")
        
        # Check debug history
        history = debug_hook.get_history()
        self.assertGreater(len(history), 0)
        
        # Verify that error was recorded
        error_records = [item for item in history if item.get('type') == 'error']
        self.assertGreater(len(error_records), 0)
    
    def test_unit_pipeline(self):
        """Test pipeline of implementation units."""
        # Create a series of transformation units
        def square(x):
            return x * x
            
        def add_one(x):
            return x + 1
            
        def multiply_by_two(x):
            return x * 2
        
        # Create units
        square_unit = ImplementationUnit(square, "square")
        add_unit = ImplementationUnit(add_one, "add_one")
        mult_unit = ImplementationUnit(multiply_by_two, "multiply_by_two")
        
        # Create pipeline
        pipeline = UnitPipeline(
            units=[square_unit, add_unit, mult_unit],
            name="test_pipeline"
        )
        
        # Test execution: (5^2 + 1) * 2 = 52
        result = pipeline(5)
        self.assertEqual(result.output, 52)
        
        # Check debug data
        self.assertEqual(len(result.debug_data['unit_results']), 3)
        self.assertGreater(result.execution_time, 0)


if __name__ == "__main__":
    unittest.main()