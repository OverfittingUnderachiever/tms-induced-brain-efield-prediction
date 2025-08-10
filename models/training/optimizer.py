# tms_efield_prediction/models/training/optimizer.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional, Union
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR


class GradientScaler:
    """Gradient scaling for mixed precision training.
    
    Provides similar functionality to torch.cuda.amp.GradScaler but with
    additional debug and monitoring hooks.
    """
    
    def __init__(
        self,
        init_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
        debug_hook = None
    ):
        """Initialize the gradient scaler.
        
        Args:
            init_scale: Initial scale factor
            growth_factor: Factor by which to increase scale when no infs/NaNs
            backoff_factor: Factor by which to decrease scale when infs/NaNs
            growth_interval: Number of consecutive non-inf/NaN steps before growth
            enabled: Whether to use scaling (disable for full precision)
            debug_hook: Optional hook for debugging
        """
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.enabled = enabled
        self.debug_hook = debug_hook
        
        self._growth_tracker = 0
        self._has_overflow = False
        
        # Create native PyTorch scaler if available
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
            self._native_scaler = torch.cuda.amp.GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=enabled
            )
        else:
            self._native_scaler = None
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss value.
        
        Args:
            loss: PyTorch loss tensor
            
        Returns:
            torch.Tensor: Scaled loss
        """
        if not self.enabled:
            return loss
        
        if self._native_scaler is not None:
            return self._native_scaler.scale(loss)
        
        return loss * self.scale
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients to original magnitude.
        
        Args:
            optimizer: PyTorch optimizer
        """
        if not self.enabled:
            return
        
        if self._native_scaler is not None:
            self._native_scaler.unscale_(optimizer)
            return
        
        inv_scale = 1.0 / self.scale
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.mul_(inv_scale)
    
    def update(self, optimizer: torch.optim.Optimizer):
        """Update optimizer with scaled gradients.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            bool: True if gradients were updated, False if skipped due to inf/NaN
        """
        if not self.enabled:
            optimizer.step()
            return True
        
        if self._native_scaler is not None:
            return self._native_scaler.step(optimizer)
        
        # Check for inf/NaN
        self._has_overflow = self._check_overflow(optimizer)
        
        if self._has_overflow:
            # Skip update and adjust scale
            self.scale = max(1.0, self.scale * self.backoff_factor)
            self._growth_tracker = 0
            
            if self.debug_hook:
                self.debug_hook.record_event('gradient_overflow', {
                    'new_scale': self.scale
                })
            
            return False
        else:
            # Perform update
            optimizer.step()
            
            # Maybe grow scale
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
                
                if self.debug_hook:
                    self.debug_hook.record_event('gradient_scale_increased', {
                        'new_scale': self.scale
                    })
            
            return True
    
    def _check_overflow(self, optimizer: torch.optim.Optimizer) -> bool:
        """Check if gradients contain inf/NaN values.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            bool: True if overflow detected
        """
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        return True
        return False


class OptimizerFactory:
    """Factory for creating optimizers with standardized configurations."""
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optim_type: str,
        learning_rate: float,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> torch.optim.Optimizer:
        """Create an optimizer with the specified parameters.
        
        Args:
            model: PyTorch model
            optim_type: Optimizer type ('adam', 'adamw', 'sgd')
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient
            momentum: Momentum (for SGD)
            beta1: Beta1 parameter (for Adam/AdamW)
            beta2: Beta2 parameter (for Adam/AdamW)
            additional_params: Additional optimizer-specific parameters
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        if additional_params is None:
            additional_params = {}
        
        # Filter parameters that require gradients
        params = [p for p in model.parameters() if p.requires_grad]
        
        if optim_type.lower() == 'adam':
            return optim.Adam(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                **additional_params
            )
        
        elif optim_type.lower() == 'adamw':
            return optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                **additional_params
            )
        
        elif optim_type.lower() == 'sgd':
            return optim.SGD(
                params,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                **additional_params
            )
        
        elif optim_type.lower() == 'rmsprop':
            return optim.RMSprop(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                **additional_params
            )
        
        else:
            raise ValueError(f"Unsupported optimizer type: {optim_type}")
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        max_epochs: int,
        steps_per_epoch: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Union[_LRScheduler, ReduceLROnPlateau]:
        """Create a learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Scheduler type ('cosine', 'plateau', 'onecycle')
            max_epochs: Maximum number of epochs
            steps_per_epoch: Steps per epoch (required for some schedulers)
            additional_params: Additional scheduler-specific parameters
            
        Returns:
            Union[_LRScheduler, ReduceLROnPlateau]: Configured scheduler
        """
        if additional_params is None:
            additional_params = {}
        
        if scheduler_type.lower() == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                **additional_params
            )
        
        elif scheduler_type.lower() == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                **additional_params
            )
        
        elif scheduler_type.lower() == 'onecycle':
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch must be provided for OneCycleLR")
            
            return OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]['lr'],
                epochs=max_epochs,
                steps_per_epoch=steps_per_epoch,
                **additional_params
            )
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class LossFactory:
    """Factory for creating loss functions with standardized configurations."""
    
    @staticmethod
    def create_loss(
        loss_type: str,
        reduction: str = 'mean',
        weight: Optional[torch.Tensor] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """Create a loss function with the specified parameters.
        
        Args:
            loss_type: Loss function type ('mse', 'l1', 'huber', 'combined')
            reduction: Reduction method ('mean', 'sum', 'none')
            weight: Optional weight tensor for weighted losses
            additional_params: Additional loss-specific parameters
            
        Returns:
            nn.Module: Configured loss function
        """
        if additional_params is None:
            additional_params = {}
        
        if loss_type.lower() == 'mse':
            return nn.MSELoss(reduction=reduction)
        
        elif loss_type.lower() == 'l1':
            return nn.L1Loss(reduction=reduction)
        
        elif loss_type.lower() == 'huber':
            delta = additional_params.get('delta', 1.0)
            return nn.SmoothL1Loss(reduction=reduction, beta=delta)
        
        elif loss_type.lower() == 'combined':
            # Combined MSE and L1 loss
            mse_weight = additional_params.get('mse_weight', 0.5)
            l1_weight = additional_params.get('l1_weight', 0.5)
            
            mse_loss = nn.MSELoss(reduction=reduction)
            l1_loss = nn.L1Loss(reduction=reduction)
            
            class CombinedLoss(nn.Module):
                def __init__(self, mse_loss, l1_loss, mse_weight, l1_weight):
                    super().__init__()
                    self.mse_loss = mse_loss
                    self.l1_loss = l1_loss
                    self.mse_weight = mse_weight
                    self.l1_weight = l1_weight
                
                def forward(self, pred, target):
                    return (self.mse_weight * self.mse_loss(pred, target) + 
                            self.l1_weight * self.l1_loss(pred, target))
            
            return CombinedLoss(mse_loss, l1_loss, mse_weight, l1_weight)
        
        elif loss_type.lower() == 'efield_loss':
            # Specialized loss for E-field prediction
            magnitude_weight = additional_params.get('magnitude_weight', 0.5)
            direction_weight = additional_params.get('direction_weight', 0.5)
            eps = additional_params.get('eps', 1e-8)
            
            class EFieldLoss(nn.Module):
                def __init__(self, magnitude_weight, direction_weight, eps=1e-8):
                    super().__init__()
                    self.magnitude_weight = magnitude_weight
                    self.direction_weight = direction_weight
                    self.eps = eps
                    self.mse_loss = nn.MSELoss(reduction=reduction)
                
                def forward(self, pred, target):
                    # Calculate magnitudes
                    pred_mag = torch.sqrt(torch.sum(pred**2, dim=1) + self.eps)
                    target_mag = torch.sqrt(torch.sum(target**2, dim=1) + self.eps)
                    
                    # Magnitude loss
                    magnitude_loss = self.mse_loss(pred_mag, target_mag)
                    
                    # Normalize vectors for direction comparison
                    pred_norm = pred / (pred_mag.unsqueeze(1) + self.eps)
                    target_norm = target / (target_mag.unsqueeze(1) + self.eps)
                    
                    # Direction loss (cosine similarity)
                    direction_loss = 1.0 - torch.mean(torch.sum(pred_norm * target_norm, dim=1))
                    
                    # Combined loss
                    return (self.magnitude_weight * magnitude_loss + 
                            self.direction_weight * direction_loss)
            
            return EFieldLoss(magnitude_weight, direction_weight, eps)
        
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")