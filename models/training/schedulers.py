"""
TMS E-field Prediction Learning Rate Schedulers

This module provides learning rate schedulers for training optimization.
It includes various scheduling strategies optimized for TMS E-field prediction models.
"""

from typing import Dict, Any, Optional, Union
import torch
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, 
    CosineAnnealingLR, 
    StepLR, 
    ExponentialLR, 
    CyclicLR,
    OneCycleLR
)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Any,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Get learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration with scheduler settings
        
    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: Learning rate scheduler or None
    """
    scheduler_type = getattr(config, "scheduler_type", None)
    
    if scheduler_type is None or scheduler_type.lower() == "none":
        return None
    
    elif scheduler_type.lower() == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=getattr(config, "scheduler_factor", 0.5),
            patience=getattr(config, "scheduler_patience", 5),
            verbose=True,
            threshold=1e-4,
            min_lr=1e-7
        )
    
    elif scheduler_type.lower() == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=getattr(config, "epochs", 50),
            eta_min=getattr(config, "scheduler_min_lr", 1e-7)
        )
    
    elif scheduler_type.lower() == "step":
        return StepLR(
            optimizer,
            step_size=getattr(config, "scheduler_step_size", 10),
            gamma=getattr(config, "scheduler_gamma", 0.5)
        )
    
    elif scheduler_type.lower() == "exponential":
        return ExponentialLR(
            optimizer,
            gamma=getattr(config, "scheduler_gamma", 0.95)
        )
    
    elif scheduler_type.lower() == "cyclic":
        return CyclicLR(
            optimizer,
            base_lr=getattr(config, "scheduler_base_lr", 1e-5),
            max_lr=getattr(config, "scheduler_max_lr", 1e-3),
            step_size_up=getattr(config, "scheduler_step_size_up", 2000),
            cycle_momentum=False,
            mode=getattr(config, "scheduler_mode", "triangular2")
        )
    
    elif scheduler_type.lower() == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=getattr(config, "scheduler_max_lr", 1e-3),
            total_steps=getattr(config, "epochs", 50),
            pct_start=getattr(config, "scheduler_pct_start", 0.3),
            div_factor=getattr(config, "scheduler_div_factor", 25.0),
            final_div_factor=getattr(config, "scheduler_final_div_factor", 10000.0)
        )
    
    else:
        print(f"Warning: Unknown scheduler type '{scheduler_type}'. Using no scheduler.")
        return None


class CosineAnnealingWithWarmup:
    """Cosine annealing with warmup learning rate scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        min_lr: float = 1e-7,
        initial_lr_ratio: float = 0.1,
        verbose: bool = True
    ):
        """Initialize cosine annealing with warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
            min_lr: Minimum learning rate
            initial_lr_ratio: Initial learning rate ratio for warmup
            verbose: Whether to print LR changes
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.initial_lr_ratio = initial_lr_ratio
        self.verbose = verbose
        
        # Store base learning rates
        self.base_lrs = []
        for group in optimizer.param_groups:
            self.base_lrs.append(group['lr'])
    
    def step(self, epoch: int = None) -> None:
        """Update learning rate based on epoch.
        
        Args:
            epoch: Current epoch number
        """
        if epoch is None:
            raise ValueError("Epoch must be specified")
        
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            
            if epoch < self.warmup_epochs:
                # Linear warmup
                lr = base_lr * (self.initial_lr_ratio + (1 - self.initial_lr_ratio) * 
                                (epoch / self.warmup_epochs))
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            
            # Update learning rate
            group['lr'] = lr
            
            if self.verbose:
                print(f"Epoch {epoch}: LR set to {lr:.6f}")


class WarmupLRScheduler:
    """Generic warmup scheduler wrapper for any PyTorch scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        warmup_epochs: int = 5,
        warmup_method: str = 'linear',
        warmup_factor: float = 0.1,
        verbose: bool = False
    ):
        """Initialize warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler: Base scheduler to use after warmup
            warmup_epochs: Number of warmup epochs
            warmup_method: Warmup method ('linear' or 'constant')
            warmup_factor: Initial learning rate factor for warmup
            verbose: Whether to print LR changes
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.warmup_factor = warmup_factor
        self.verbose = verbose
        
        # Store base learning rates
        self.base_lrs = []
        for group in optimizer.param_groups:
            self.base_lrs.append(group['lr'])
    
    def step(self, epoch=None, metrics=None):
        """Update learning rate based on epoch or metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Metrics for adaptive schedulers (e.g., ReduceLROnPlateau)
        """
        if epoch < self.warmup_epochs:
            # In warmup phase - calculate and set learning rates directly
            if self.warmup_method == 'linear':
                # Linear warmup
                alpha = epoch / self.warmup_epochs
                factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                # Constant warmup
                factor = self.warmup_factor
                
            # Set learning rates
            for i, base_lr in enumerate(self.base_lrs):
                self.optimizer.param_groups[i]['lr'] = base_lr * factor
                
            if self.verbose:
                print(f"Epoch {epoch}: Warmup phase, LR set to {self.optimizer.param_groups[0]['lr']:.6f}")
        else:
            # In regular scheduling phase
            # First reset learning rates to base values if needed
            if epoch == self.warmup_epochs:  # First epoch after warmup
                for i, base_lr in enumerate(self.base_lrs):
                    self.optimizer.param_groups[i]['lr'] = base_lr
            
            # Then let the scheduler handle subsequent steps
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metrics is None:
                    raise ValueError("Metrics must be provided for ReduceLROnPlateau")
                self.scheduler.step(metrics)
            else:
                # Use the adjusted epoch relative to warmup end
                adjusted_epoch = epoch - self.warmup_epochs
                
                # Force the scheduler to use the adjusted epoch
                original_last_epoch = self.scheduler.last_epoch
                self.scheduler.last_epoch = adjusted_epoch - 1
                self.scheduler.step()
                # No need to restore as step() advances last_epoch
                
            if self.verbose:
                print(f"Epoch {epoch}: Regular phase, LR set to {self.optimizer.param_groups[0]['lr']:.6f}")