# tms_efield_prediction/automl/integration/ray_trainable.py
import os
import torch
import logging
import numpy as np
from typing import Dict, Any

# Import necessary components from your project
from ...models.architectures.simple_unet_magnitude import SimpleUNetMagnitudeModel
from ...models.training.trainer import ModelTrainer, TrainerConfig
from ...utils.state.context import ModelContext
from ...models.evaluation.metrics import calculate_magnitude_metrics
from .checkpoint_utils import load_checkpoint_if_exists, save_and_report_checkpoint
# Import data loading function locally
# from ...data.pipeline.multi_subject_data import prepare_multi_subject_data

logger = logging.getLogger(__name__)


def disable_internal_checkpointing(trainer):
    """
    Disable the internal ModelCheckpointCallback during Ray Tune runs.
    This prevents duplicate checkpoint files.
    
    Args:
        trainer: ModelTrainer instance
    """
    try:
        # Find and modify the ModelCheckpointCallback
        for i, callback in enumerate(trainer.callbacks):
            if callback.__class__.__name__ == 'ModelCheckpointCallback':
                # Disable the callback completely
                trainer.callbacks.pop(i)
                logger.info("Disabled ModelCheckpointCallback to prevent duplicate checkpoints")
                break
    except Exception as e:
        logger.warning(f"Could not disable ModelCheckpointCallback: {e}")

def create_model(model_config, device='cuda'):
    """
    Create a model based on config parameters.
    
    Args:
        model_config: Dictionary with model configuration
        device: Device to place model on
        
    Returns:
        Instantiated model
    """
    model_type = model_config.get("model_type", "simple_unet_magnitude")
    
    if model_type == "simple_unet_magnitude":
        from tms_efield_prediction.models.architectures.simple_unet_magnitude import SimpleUNetMagnitudeModel
        model = SimpleUNetMagnitudeModel(config=model_config)
    
    elif model_type == "simple_unet_vector":
        from tms_efield_prediction.models.architectures.simple_unet_vector import SimpleUNetVectorModel
        model = SimpleUNetVectorModel(config=model_config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)

def train_model_tune(config: Dict[str, Any]):
    """
    Training function (potentially for Ray Tune) that trains a model with a given configuration.

    Args:
        config: Dictionary containing hyperparameters and settings for this run.
    """
    logger.info("--- Starting train_model_tune ---")

    # --- Configuration Extraction ---
    train_subjects = config['train_subjects']
    val_subjects = config['val_subjects']
    test_subjects = config['test_subjects']
    data_dir = config['data_dir']
    batch_size = config.get('batch_size', 16)
    output_shape = config.get('output_shape', (20, 20, 20))
    mri_type = config.get('mri_type', 'dti')  # Get mri_type with default
    num_workers = config.get('num_workers', 0)

    output_shape = tuple(config.get('output_shape', (20, 20, 20)))
    logger.info(f"Using output shape: {output_shape}")

    # Determine augmentation configuration
    trivial_augment_config = config.get('trivial_augment_config')
    augmentation_config = config.get('augmentation_config') # Standard config
    active_augmentation_config = None
    if trivial_augment_config:
        logger.info("TrivialAugment config found. Assuming it's handled by data loading/collate.")
        active_augmentation_config = None
    elif augmentation_config and augmentation_config.get('enabled', False):
        logger.info("Standard augmentation config found and enabled.")
        active_augmentation_config = augmentation_config
    else:
        logger.info("No specific augmentation configuration enabled.")
        active_augmentation_config = None

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Loading (Single Call) ---
    from ...data.pipeline.multi_subject_data import prepare_multi_subject_data

    logger.info(">>> Calling prepare_multi_subject_data (Single Call) for ALL data")
    try:
        train_loader, val_loader, test_loader = prepare_multi_subject_data(
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            data_root_path=data_dir,
            output_shape=output_shape,
            batch_size=batch_size,
            augmentation_config=config.get('augmentation_config'),
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            mri_type=mri_type  # Pass mri_type parameter
        )
    except ValueError as e:
         logger.error(f"ValueError during data preparation: {e}", exc_info=True)
         raise
    except Exception as e:
         logger.error(f"Unexpected error during data preparation: {e}", exc_info=True)
         raise

    if train_loader is None or len(train_loader) == 0:
        logger.error("Training data loader is empty or None after preparation!")
        raise ValueError("Failed to create a valid training data loader.")

    logger.info(f"Train loader created. Number of batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"Validation loader created. Number of batches: {len(val_loader)}")
    else:
        logger.warning("Validation loader is None.")
    if test_loader:
        logger.info(f"Test loader created. Number of batches: {len(test_loader)}")
    else:
        logger.warning("Test loader is None.")

    # --- Model Definition ---
    try:
        sample_features, _ = next(iter(train_loader))
        input_channels = sample_features.shape[1]
        logger.info(f"Derived input_channels: {input_channels} from training data batch shape: {sample_features.shape}")
    except StopIteration:
        logger.error("Training loader is empty, cannot derive input shape/channels.")
        raise ValueError("Training loader is empty, cannot proceed.")
    except Exception as e:
        logger.error(f"Error getting sample batch from train_loader: {e}", exc_info=True)
        logger.warning("Using default input_channels=4 due to error.")
        input_channels = 4

    if 'model_config' in config:
        model_config = config['model_config'].copy()
        model_config['input_channels'] = input_channels
        model_config['input_shape'] = [input_channels] + list(output_shape)
        model_config['output_shape'] = [1] + list(output_shape)
        for key in ['feature_maps', 'levels', 'norm_type', 'activation', 'dropout_rate', 'use_residual', 'use_attention']:
             if key in config: model_config[key] = config[key]
    else:
        logger.warning("Building model_config within train_model_tune as it was not found in main config.")
        model_config = {
            "model_type": config.get('model_type', "simple_unet_magnitude"),
            "input_shape": [input_channels] + list(output_shape),
            "output_shape": [1] + list(output_shape),
            "input_channels": input_channels,
            "output_channels": 1,
            "feature_maps": config['feature_maps'],
            "levels": config['levels'],
            "norm_type": config['norm_type'],
            "activation": config['activation'],
            "dropout_rate": config['dropout_rate'],
            "use_residual": config.get('use_residual', True),
            "use_attention": config.get('use_attention', False),
        }
    logger.info(f"Using model configuration: {model_config}")

    try:
        model = SimpleUNetMagnitudeModel(config=model_config)
        model.to(device)
        logger.info("Model created and moved to device.")
    except Exception as e:
        logger.error(f"Error creating SimpleUNetMagnitudeModel: {e}", exc_info=True)
        raise

    # --- Trainer Setup ---
    trainer_config = TrainerConfig(
        batch_size=batch_size,
        epochs=config.get('epochs', config.get('max_epochs', 15)),
        learning_rate=config['learning_rate'],
        optimizer_type=config.get('optimizer_type', 'adamw'),
        scheduler_type=config.get('scheduler_type', 'reduce_on_plateau'),
        scheduler_patience=config.get('scheduler_patience', 3),
        mask_threshold=config.get('mask_threshold', 1e-8),
        device=str(device),
        mixed_precision=config.get('mixed_precision', True),
        validation_frequency=config.get('validation_frequency', 1),
        early_stopping=config.get('early_stopping', True),
        early_stopping_patience=config.get('early_stopping_patience', 7),
        loss_type=config.get('loss_type', "magnitude_mse"),
        max_models_to_keep=config.get('max_models_to_keep', 20),  # This line gets the parameter
    )
    logger.info(f"Using trainer configuration: {trainer_config}")

    model_context = ModelContext(
        dependencies={},
        config={"architecture": model_config.get("model_type", "simple_unet_magnitude")}
    )

    try:
        trainer = ModelTrainer(
            model=model,
            config=trainer_config,
            model_context=model_context
        )
        logger.info("ModelTrainer created.")
        
        # Add this line to disable internal checkpointing
        disable_internal_checkpointing(trainer)
        
    except Exception as e:
        logger.error(f"Error creating ModelTrainer: {e}", exc_info=True)
        raise
    try:
        load_checkpoint_if_exists(trainer)
    except Exception as e:
        logger.warning(f"Could not load checkpoint (this might be expected on first run): {e}", exc_info=False)

    # --- Training ---
    # NOTE: Reverting to extracting data into lists as ModelTrainer.train expects positional arguments.
    # This can be memory-intensive for large datasets.
    logger.warning("Extracting data into lists for ModelTrainer.train. This may be memory intensive.")

    logger.info("Extracting training data...")
    train_features_list = []
    train_targets_list = []
    for i, (features_batch, targets_batch) in enumerate(train_loader):
        # Detach and move to CPU before extending list to free GPU memory quickly
        train_features_list.extend([f.detach().cpu() for f in features_batch])
        train_targets_list.extend([t.detach().cpu() for t in targets_batch])
        if (i + 1) % 50 == 0: # Log progress every 50 batches
            logger.info(f"  Processed {i+1}/{len(train_loader)} training batches.")
    logger.info(f"Finished extracting {len(train_features_list)} training samples.")

    logger.info("Extracting validation data...")
    val_features_list = []
    val_targets_list = []
    if val_loader:
        for i, (features_batch, targets_batch) in enumerate(val_loader):
            val_features_list.extend([f.detach().cpu() for f in features_batch])
            val_targets_list.extend([t.detach().cpu() for t in targets_batch])
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{len(val_loader)} validation batches.")
        logger.info(f"Finished extracting {len(val_features_list)} validation samples.")
    else:
        logger.info("No validation loader, skipping validation data extraction.")
        # Pass empty lists or None to trainer.train, depending on its signature
        val_features_list = None # Or [] if trainer expects empty list
        val_targets_list = None # Or []


    logger.info("Starting model training with extracted data lists.")
    training_history = None
    try:
        # Call trainer.train with positional arguments (lists of tensors)
        # Adjust arguments based on the exact signature of your ModelTrainer.train
        training_history = trainer.train(
            train_features_list,
            train_targets_list,
            val_features_list,  # Pass None or [] if val_loader was None
            val_targets_list    # Pass None or [] if val_loader was None
        )
        logger.info("Model training finished.")
    except Exception as e:
         logger.error(f"Error during model training: {e}", exc_info=True)
         raise # Re-raise the exception

    # Cleanup large lists to free memory
    del train_features_list, train_targets_list, val_features_list, val_targets_list
    import gc
    gc.collect()
    logger.info("Cleaned up extracted data lists.")


    # --- Evaluation ---
    test_metrics = {}
    if test_loader:
        logger.info("Evaluating model on test set.")
        model.eval()
        all_preds_list = []
        all_targets_list = []
        try:
            with torch.no_grad():
                for features, targets in test_loader:
                    features = features.to(device)
                    outputs = model(features)
                    all_preds_list.append(outputs.cpu())
                    all_targets_list.append(targets.cpu())

            if all_preds_list:
                all_preds_tensor = torch.cat(all_preds_list, dim=0)
                all_targets_tensor = torch.cat(all_targets_list, dim=0)

                if all_preds_tensor.dim() > 1 and all_preds_tensor.shape[1] == 1:
                     all_preds_tensor = all_preds_tensor.squeeze(1)
                if all_targets_tensor.dim() > 1 and all_targets_tensor.shape[1] == 1:
                     all_targets_tensor = all_targets_tensor.squeeze(1)

                all_preds_np = all_preds_tensor.numpy()
                all_targets_np = all_targets_tensor.numpy()
                test_mask_np = (all_targets_np > config.get('mask_threshold', 1e-8))

                test_metrics = calculate_magnitude_metrics(all_preds_np, all_targets_np, mask=test_mask_np)
                logger.info(f"Test metrics calculated: {test_metrics}")
            else:
                logger.warning("No predictions collected from test loader.")
        except Exception as e:
            logger.error(f"Error during test set evaluation: {e}", exc_info=True)
    else:
        logger.warning("No test loader available, skipping test set evaluation.")

    # --- Reporting ---
    last_val_loss = float('inf')
    current_epoch = 0
    if hasattr(trainer, 'current_epoch'):
         current_epoch = trainer.current_epoch

    if training_history and "val" in training_history and training_history["val"]:
         try:
             last_val_loss = training_history["val"][-1]["loss"]
         except (IndexError, KeyError, TypeError) as e:
              logger.warning(f"Could not extract last validation loss from history: {e}.")
              for entry in reversed(training_history.get("val", [])):
                  if isinstance(entry, dict) and 'loss' in entry:
                      last_val_loss = entry['loss']
                      logger.warning(f"Using validation loss from earlier epoch: {last_val_loss}")
                      break

    result_metrics = {
        "val_loss": last_val_loss,
        "training_iteration": current_epoch,
        "epoch": current_epoch
    }
    for key, value in test_metrics.items():
        result_metrics[f"test_{key}"] = value

    logger.info(f"Final result metrics: {result_metrics}")

    # Report metrics and save checkpoint (handle Ray Tune context)
    try:
        from ray import train as ray_train
        if ray_train.get_context():
             logger.info("Running under Ray Tune context. Saving checkpoint and reporting metrics.")
             save_and_report_checkpoint(
                 trainer=trainer,
                 metrics=result_metrics,
                 epoch=current_epoch,
                 is_final=True
             )
        else:
             logger.info("Not running under Ray Tune context. Skipping Tune reporting/checkpointing.")
             output_dir = config.get('output_dir', '.')
             if output_dir and os.path.exists(output_dir):
                 model_save_path = os.path.join(output_dir, 'final_model.pth')
                 torch.save(model.state_dict(), model_save_path)
                 logger.info(f"Standalone run: Saved final model to {model_save_path}")

    except ImportError:
        logger.info("Ray Tune not available. Skipping Tune reporting/checkpointing.")
        output_dir = config.get('output_dir', '.')
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
             model_save_path = os.path.join(output_dir, 'final_model.pth')
             torch.save(model.state_dict(), model_save_path)

    except Exception as e:
         logger.error(f"Error during final reporting/checkpointing: {e}", exc_info=True)

    # Add the new model saving code here
    # Always save the model in the trial directory for the ModelManager to track
    try:
        model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model_config,
            'val_loss': last_val_loss,
            'metrics': result_metrics
        }, model_save_path)
        logger.info(f"Saved model for trial: {model_save_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

    logger.info("--- Finished train_model_tune ---")
    return result_metrics