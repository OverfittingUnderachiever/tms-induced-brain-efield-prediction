# tms_efield_prediction/models/architectures/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import torch
import os
import yaml
import numpy as np
import logging

from ...utils.state.context import ModelContext
from ...data.pipeline.tms_data_types import TMSProcessedData

logger = logging.getLogger(__name__)

class BaseModel(ABC, torch.nn.Module):
    """Abstract base class for all TMS E-field prediction models."""

    def __init__(self, config: Dict[str, Any], context: Optional[ModelContext] = None):
        """Initialize the base model."""
        super().__init__()
        self.config = config
        self.context = context or self._create_default_context()

        try:
            self.validate_config()
        except Exception as e:
             logger.error(f"Configuration validation failed during __init__: {e}", exc_info=True)
             raise

        self.debug_hook = None
        if hasattr(self.context, 'debug_mode') and self.context.debug_mode:
             self._initialize_debug()


    def _create_default_context(self) -> ModelContext:
        """Create a default model context if none provided."""
        return ModelContext(
            dependencies={"torch": torch.__version__},
            config=self.config.copy() if self.config else {},
            debug_mode=False
        )

    def _initialize_debug(self):
        """Initialize debug hooks and monitoring."""
        from ...utils.debug.hooks import DebugHook
        from dataclasses import dataclass

        @dataclass
        class ModelDebugContext:
            verbosity_level: int = 1
            sampling_rate: float = 0.1

        self.debug_hook = DebugHook(ModelDebugContext())
        logger.info("Debug hook initialized.")


    # Removed @abstractmethod - Subclasses CAN override, but don't HAVE to.
    def validate_config(self) -> bool:
        """Validate BASE model configuration keys. Subclasses should extend."""
        logger.debug("Running BASE configuration validation...")
        required_keys = ["model_type"]
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
             raise ValueError(f"Missing required configuration key(s) in base validation: {', '.join(missing_keys)}")
        logger.debug("Base configuration validation passed.")
        return True

    @abstractmethod # Forward pass MUST be implemented by subclasses.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        raise NotImplementedError


    def process_input(self, data: Union[TMSProcessedData, torch.Tensor]) -> torch.Tensor:
        """Process input data into model-ready tensor."""
        if isinstance(data, TMSProcessedData):
            x = torch.from_numpy(data.input_features).float()
            if x.ndim == 3: # D, H, W -> B=1, C=1, D, H, W
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.ndim == 4: # C, D, H, W -> B=1, C, D, H, W or D, H, W, C -> B=1, C, D, H, W
                if x.shape[-1] <= 16: # Heuristic: Assume D,H,W,C
                    logger.debug("Assuming input shape [D, H, W, C], permuting to [B, C, D, H, W]")
                    x = x.permute(3, 0, 1, 2).unsqueeze(0)
                else: # Assume C, D, H, W
                    logger.debug("Assuming input shape [C, D, H, W], adding batch dim")
                    x = x.unsqueeze(0)
            elif x.ndim != 5: # B, C, D, H, W is ok
                 raise ValueError(f"Unsupported TMSProcessedData features shape: {x.shape}")
            return x.float()

        elif isinstance(data, torch.Tensor):
            x = data
            if not x.is_floating_point():
                x = x.float()
            if x.ndim == 3: # D, H, W -> 1, 1, D, H, W
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.ndim == 4: # C, D, H, W -> 1, C, D, H, W
                x = x.unsqueeze(0)
            elif x.ndim != 5: # B, C, D, H, W is ok
                raise ValueError(f"Unsupported input tensor shape: {x.shape}")
            return x
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")


    def predict(self, data: Union[TMSProcessedData, torch.Tensor]) -> np.ndarray:
        """Make a prediction using the model."""
        self.eval()
        x = self.process_input(data)
        device = next(self.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            output = self.forward(x)
        # Squeeze removes batch and channel dims if they are 1
        return output.squeeze().cpu().numpy()


    def save(self, path: str):
        """Save model state_dict and config to disk."""
        if not path:
             logger.error("Save path is empty, cannot save model.")
             return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_dict = {
                "model_state_dict": self.state_dict(),
                "config": self.config,
                "context": {
                    "dependencies": getattr(self.context, 'dependencies', {}),
                    "config": getattr(self.context, 'config', {}),
                    "debug_mode": getattr(self.context, 'debug_mode', False)
                } if self.context else None
            }
            torch.save(save_dict, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}", exc_info=True)


    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "BaseModel":
        """Load model from disk."""
        if not os.path.exists(path):
             raise FileNotFoundError(f"Model file not found at {path}")
        try:
            save_dict = torch.load(path, map_location=device)

            config = save_dict.get("config", {})
            config.update({k: v for k, v in kwargs.items() if k in config}) # Allow overrides

            context_data = save_dict.get("context")
            context = None
            if context_data:
                 context_config = context_data.get("config", {})
                 context_config.update({k: v for k, v in kwargs.items() if k in context_config})
                 context = ModelContext(
                     dependencies=context_data.get("dependencies", {}),
                     config=context_config,
                     debug_mode=kwargs.get("debug_mode", context_data.get("debug_mode", False))
                 )

            # Instantiate the specific subclass that called load
            model = cls(config=config, context=context)
            model.load_state_dict(save_dict["model_state_dict"])
            model.to(device)
            logger.info(f"Model loaded successfully from {path} to {device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}", exc_info=True)
            raise


    def get_memory_usage(self) -> Dict[str, Any]:
        """Get model memory usage statistics."""
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_size = param_count * 4 # float32
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        total_size_bytes = param_size + buffer_size
        return {
            "param_count": param_count,
            "param_size_bytes": param_size,
            "buffer_size_bytes": buffer_size,
            "total_size_bytes": total_size_bytes,
            "total_size_mb": total_size_bytes / (1024 * 1024)
        }


    def __str__(self) -> str:
        """String representation of the model."""
        try:
             mem_usage = self.get_memory_usage()
             param_str = f"{mem_usage['param_count']:,} ({mem_usage['total_size_mb']:.2f} MB)"
        except Exception:
             param_str = "N/A"

        # Get torch module summary, but handle potential errors
        try:
            module_summary = super().__str__()
        except Exception:
            module_summary = "(torch module summary unavailable)"

        return (
            f"{self.__class__.__name__} (\n"
            f"  Model Type: {self.config.get('model_type', 'N/A')}\n"
            f"  Trainable Parameters: {param_str}\n"
            f"{module_summary} \n)"
        )