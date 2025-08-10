# tms_efield_prediction/automl/integration/checkpoint_utils.py

import os
import tempfile
import logging
from typing import Dict, Any, Optional
from ray import train
from ray.train import Checkpoint

logger = logging.getLogger(__name__)

def load_checkpoint_if_exists(trainer) -> bool:
    """
    Loads a checkpoint if one exists using Ray Train's API.
    
    Args:
        trainer: The model trainer instance
        
    Returns:
        bool: True if a checkpoint was loaded, False otherwise
    """
    checkpoint = train.get_checkpoint()
    if checkpoint:
        try:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
                if os.path.exists(checkpoint_path):
                    trainer.load_checkpoint(checkpoint_path)
                    logger.info(f"Loaded checkpoint from {checkpoint_path}")
                    return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    
    return False

def save_and_report_checkpoint(
    trainer, 
    metrics: Dict[str, Any], 
    epoch: int, 
    checkpoint_frequency: int = 5,
    is_final: bool = False
) -> None:
    """
    Saves a checkpoint and reports metrics to Ray Tune.
    
    Args:
        trainer: The model trainer instance
        metrics: Dictionary of metrics to report
        epoch: Current epoch number
        checkpoint_frequency: How often to save checkpoints
        is_final: Whether this is the final checkpoint
    """
    # Determine if we should checkpoint
    should_checkpoint = is_final or (epoch % checkpoint_frequency == 0)
    
    if should_checkpoint:
        # Create a temporary directory for the checkpoint
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
        
        # Save the checkpoint
        trainer.save_checkpoint(checkpoint_path, metrics)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")
        
        # Report metrics with checkpoint
        train.report(
            metrics, 
            checkpoint=Checkpoint.from_directory(temp_dir)
        )
    else:
        # Report metrics without checkpoint
        train.report(metrics)

def load_checkpoint_if_exists(trainer) -> bool:
    """Loads a checkpoint if one exists using Ray Train's API."""
    checkpoint = train.get_checkpoint()
    if checkpoint:
        try:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
                if os.path.exists(checkpoint_path):
                    trainer.load_checkpoint(checkpoint_path)
                    logger.info(f"Loaded checkpoint from {checkpoint_path}")
                    return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    
    return False

def save_and_report_checkpoint(
    trainer, 
    metrics: Dict[str, Any], 
    epoch: int, 
    checkpoint_frequency: int = 5,
    is_final: bool = False
) -> None:
    """Saves a checkpoint and reports metrics to Ray Tune."""
    # Determine if we should checkpoint
    should_checkpoint = is_final or (epoch % checkpoint_frequency == 0)
    
    if should_checkpoint:
        # Create a temporary directory for the checkpoint
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
        
        # Save the checkpoint
        trainer.save_checkpoint(checkpoint_path, metrics)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")
        
        # Report metrics with checkpoint
        train.report(
            metrics, 
            checkpoint=Checkpoint.from_directory(temp_dir)
        )
    else:
        # Report metrics without checkpoint
        train.report(metrics)