# test_stacked_loading.py

import os
import sys
import torch
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import TMS pipeline modules
from tms_efield_prediction.utils.state.context import TMSPipelineContext
from tms_efield_prediction.data.pipeline.loader import TMSDataLoader
from tms_efield_prediction.data.transformations.stack_pipeline import EnhancedStackingPipeline

def test_stacked_array_loading(
    subject_id: str = "003",
    data_root_path: str = "/home/freyhe/MA_Henry/data",
    output_shape: Tuple[int, int, int] = (25, 25, 25),
    use_stacked_arrays: bool = True
):
    """
    Test loading and processing samples from stacked arrays.
    
    Args:
        subject_id: Subject ID to load
        data_root_path: Path to data root directory
        output_shape: Output shape for processed data
        use_stacked_arrays: Whether to use stacked arrays
    """
    logger.info(f"Testing stacked array loading for subject {subject_id}")
    logger.info(f"Using {'stacked arrays' if use_stacked_arrays else 'separate files'}")
    
    # Set up pipeline context
    tms_config = {"mri_tensor": None, "device": torch.device("cpu")}
    pipeline_context = TMSPipelineContext(
        dependencies={},
        config=tms_config,
        pipeline_mode="mri_dadt",
        experiment_phase="training",
        debug_mode=True,
        subject_id=subject_id,
        data_root_path=data_root_path,
        output_shape=output_shape,
        normalization_method="standard",
        device=torch.device("cpu")
    )
    
    try:
        # Create data loader with stacked array flag
        data_loader = TMSDataLoader(
            context=pipeline_context,
            use_stacked_arrays=use_stacked_arrays
        )
        
        # Load raw data
        logger.info("Loading raw data...")
        raw_data = data_loader.load_raw_data()
        if raw_data is None:
            logger.error("Raw data is None")
            return
        
        # Update context with MRI tensor
        pipeline_context.config["mri_tensor"] = raw_data.mri_tensor
        
        # Create sample list
        logger.info("Creating sample list...")
        samples = data_loader.create_sample_list(raw_data)
        if not samples:
            logger.error("No samples found")
            return
        
        logger.info(f"Created {len(samples)} samples")
        
        # Process first sample to verify
        stacking_pipeline = EnhancedStackingPipeline(context=pipeline_context)
        logger.info(f"Processing first sample: {samples[0].sample_id}")
        
        # Check if using stacked array
        is_using_stacked = samples[0].metadata.get('using_stacked_array', False)
        logger.info(f"Sample is using stacked array: {is_using_stacked}")
        
        # Process the sample
        processed_data = stacking_pipeline.process_sample(samples[0])
        
        # Print information about processed data
        logger.info(f"Processed data subject ID: {processed_data.subject_id}")
        logger.info(f"Input features shape: {processed_data.input_features.shape}")
        logger.info(f"Target E-field shape: {processed_data.target_efield.shape}")
        
        # Check channel dimensions
        if len(processed_data.input_features.shape) == 4:
            num_channels = processed_data.input_features.shape[-1]
            logger.info(f"Number of input channels: {num_channels}")
            
            # Print channel information if available
            if 'channel_info' in processed_data.metadata:
                logger.info(f"Channel info: {processed_data.metadata['channel_info']}")
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test stacked array loading")
    parser.add_argument("--subject", type=str, default="003", help="Subject ID to load")
    parser.add_argument("--data-root", type=str, default="/home/freyhe/MA_Henry/data", 
                        help="Path to data root directory")
    parser.add_argument("--bin-size", type=int, default=25, help="Bin size for output shape")
    parser.add_argument("--use-stacked", action="store_true", help="Use stacked arrays")
    parser.add_argument("--use-separate", action="store_true", help="Use separate files")
    
    args = parser.parse_args()
    
    # Determine whether to use stacked arrays
    use_stacked = True  # Default
    if args.use_separate:
        use_stacked = False
    if args.use_stacked:
        use_stacked = True
    
    output_shape = (args.bin_size, args.bin_size, args.bin_size)
    
    test_stacked_array_loading(
        subject_id=args.subject,
        data_root_path=args.data_root,
        output_shape=output_shape,
        use_stacked_arrays=use_stacked
    )