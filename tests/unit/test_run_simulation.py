#!/usr/bin/env python3
"""
Simple test for TMS simulation components.
"""

import os
import sys
import logging



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_simulation')

# Try imports with the corrected path
try:
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


    from tms_efield_prediction.simulation.tms_simulation import SimulationContext, SimulationState
    from tms_efield_prediction.utils.debug.hooks import DebugHook
    from tms_efield_prediction.utils.debug.context import PipelineDebugContext, RetentionPolicy
    
    logger.info("Successfully imported simulation components")
    TEST_PASSED = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    TEST_PASSED = False

def main():
    """Run a simple test."""
    if not TEST_PASSED:
        logger.error("Test failed due to import errors")
        return 1
    
    # Create a simple state and test transitions
    state = SimulationState()
    logger.info(f"Initial state: {state.simulation_phase}")
    
    # Test state transitions
    state = state.transition_to("mesh_loading")
    logger.info(f"New state: {state.simulation_phase}")
    
    logger.info("Test passed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())