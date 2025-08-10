"""
TMS E-field Prediction Loss Functions

This module provides specialized loss functions for TMS E-field magnitude prediction,
with particular focus on hotspot prediction accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseLoss(nn.Module):
    """Base class for all TMS E-field prediction losses."""
    
    def __init__(self, mask_threshold: float = 0.01, reduction: str = 'mean'):
        """Initialize the base loss.
        
        Args:
            mask_threshold: Threshold for masking low-intensity regions
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.mask_threshold = mask_threshold
        self.reduction = reduction
    
    def _apply_mask(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply mask to focus on regions above threshold.
        
        Args:
            pred: Predicted magnitude field [B, 1, D, H, W]
            target: Target magnitude field [B, 1, D, H, W]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked prediction, masked target, and mask
        """
        # Create mask based on target values
        mask = (target > self.mask_threshold).float()
        
        # Handle spatial dimension mismatch if necessary
        if pred.shape[2:] != target.shape[2:]:
            logger.warning(f"Spatial dimension mismatch between prediction {pred.shape[2:]} "
                          f"and target {target.shape[2:]} in loss. Interpolating.")
            
            try:
                # Interpolate target and mask to match prediction shape
                target = F.interpolate(target, size=pred.shape[2:], mode='trilinear', align_corners=False)
                mask = F.interpolate(mask, size=pred.shape[2:], mode='trilinear', align_corners=False)
                # Re-binarize mask after interpolation
                mask = (mask > 0.5).float()
            except Exception as e:
                logger.error(f"Interpolation failed: {e}")
                # Return unmodified tensors if interpolation fails
                return pred, target, mask
        
        return pred, target, mask
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between prediction and target.
        
        Args:
            pred: Predicted magnitude field [B, 1, D, H, W]
            target: Target magnitude field [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: Loss value
        """
        raise NotImplementedError("Subclasses must implement forward method")


class MaskedMSELoss(BaseLoss):
    """Mean squared error loss with masking for E-field magnitude prediction."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute masked MSE loss.
        
        Args:
            pred: Predicted magnitude field [B, 1, D, H, W]
            target: Target magnitude field [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: MSE loss value
        """
        # Apply mask
        pred, target, mask = self._apply_mask(pred, target)
        
        # Compute masked MSE
        squared_error = (pred - target)**2
        masked_error = squared_error * mask
        
        # Normalize by number of valid elements
        num_valid = torch.sum(mask, dim=(1, 2, 3, 4)) + 1e-8  # Avoid division by zero
        
        if self.reduction == 'mean':
            # Average over valid elements, then over batch
            loss = torch.sum(masked_error, dim=(1, 2, 3, 4)) / num_valid
            return loss.mean()
        elif self.reduction == 'sum':
            # Sum over all elements
            return torch.sum(masked_error)
        else:  # 'none'
            # Return per-element loss
            return masked_error


class MaskedL1Loss(BaseLoss):
    """L1 loss with masking for E-field magnitude prediction."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute masked L1 loss.
        
        Args:
            pred: Predicted magnitude field [B, 1, D, H, W]
            target: Target magnitude field [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: L1 loss value
        """
        # Apply mask
        pred, target, mask = self._apply_mask(pred, target)
        
        # Compute masked L1
        abs_error = torch.abs(pred - target)
        masked_error = abs_error * mask
        
        # Normalize by number of valid elements
        num_valid = torch.sum(mask, dim=(1, 2, 3, 4)) + 1e-8
        
        if self.reduction == 'mean':
            loss = torch.sum(masked_error, dim=(1, 2, 3, 4)) / num_valid
            return loss.mean()
        elif self.reduction == 'sum':
            return torch.sum(masked_error)
        else:  # 'none'
            return masked_error


class WeightedMSELoss(BaseLoss):
    """MSE loss with intensity-based weighting for E-field magnitude prediction.
    
    This loss assigns higher weights to high-intensity regions, making the model
    focus more on accurately predicting hotspots.
    """
    
    def __init__(self, mask_threshold: float = 0.01, reduction: str = 'mean', 
                weight_factor: float = 2.0, hotspot_percentile: float = 90.0):
        """Initialize weighted MSE loss.
        
        Args:
            mask_threshold: Threshold for masking low-intensity regions
            reduction: Reduction method ('mean', 'sum', 'none')
            weight_factor: Factor to increase weight for high-intensity regions
            hotspot_percentile: Percentile threshold to define hotspots
        """
        super().__init__(mask_threshold, reduction)
        self.weight_factor = weight_factor
        self.hotspot_percentile = hotspot_percentile
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss.
        
        Args:
            pred: Predicted magnitude field [B, 1, D, H, W]
            target: Target magnitude field [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: Weighted MSE loss value
        """
        # Apply mask
        pred, target, mask = self._apply_mask(pred, target)
        
        # Calculate weights based on target intensity (higher intensity = higher weight)
        # Normalize target to [0, 1] range for weighting
        batch_size = target.shape[0]
        weights = torch.ones_like(target)
        
        for b in range(batch_size):
            # Get masked target values for this batch
            masked_target = target[b] * mask[b]
            max_val = torch.max(masked_target) + 1e-8
            
            # Normalize to [0, 1]
            normalized = masked_target / max_val
            
            # Calculate weight as 1 + weight_factor * normalized_intensity
            weights[b] = 1.0 + self.weight_factor * normalized
        
        # Apply weights to squared error
        squared_error = (pred - target)**2
        weighted_error = squared_error * weights * mask
        
        # Normalize by number of valid elements
        num_valid = torch.sum(mask, dim=(1, 2, 3, 4)) + 1e-8
        
        if self.reduction == 'mean':
            loss = torch.sum(weighted_error, dim=(1, 2, 3, 4)) / num_valid
            return loss.mean()
        elif self.reduction == 'sum':
            return torch.sum(weighted_error)
        else:  # 'none'
            return weighted_error


class HotspotMSELoss(BaseLoss):
    """MSE loss that focuses specifically on hotspot regions.
    
    This loss separately calculates error for hotspot and background regions,
    applying a higher weight to hotspot errors.
    """
    
    def __init__(self, mask_threshold: float = 0.01, reduction: str = 'mean',
                hotspot_percentile: float = 90.0, hotspot_weight: float = 5.0):
        """Initialize hotspot MSE loss.
        
        Args:
            mask_threshold: Threshold for masking low-intensity regions
            reduction: Reduction method ('mean', 'sum', 'none')
            hotspot_percentile: Percentile threshold to define hotspots
            hotspot_weight: Weight factor for hotspot regions
        """
        super().__init__(mask_threshold, reduction)
        self.hotspot_percentile = hotspot_percentile
        self.hotspot_weight = hotspot_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute hotspot-focused MSE loss.
        
        Args:
            pred: Predicted magnitude field [B, 1, D, H, W]
            target: Target magnitude field [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: Hotspot-focused MSE loss value
        """
        # Apply mask
        pred, target, mask = self._apply_mask(pred, target)
        
        # Calculate error for each sample in batch
        batch_size = target.shape[0]
        batch_losses = []
        
        for b in range(batch_size):
            # Get masked values for this batch
            masked_target = target[b] * mask[b]
            masked_pred = pred[b] * mask[b]
            
            # Find hotspot threshold (top percentile of non-zero values)
            valid_values = masked_target[masked_target > 0]
            if len(valid_values) > 0:
                hotspot_threshold = torch.quantile(valid_values, self.hotspot_percentile/100.0)
            else:
                # Fallback if no valid values
                hotspot_threshold = self.mask_threshold
            
            # Create hotspot mask
            hotspot_mask = (masked_target > hotspot_threshold).float()
            background_mask = mask[b] - hotspot_mask
            
            # Calculate hotspot and background errors
            hotspot_error = ((masked_pred - masked_target) * hotspot_mask)**2
            background_error = ((masked_pred - masked_target) * background_mask)**2
            
            # Calculate mean errors (avoid division by zero)
            num_hotspot = torch.sum(hotspot_mask) + 1e-8
            num_background = torch.sum(background_mask) + 1e-8
            
            hotspot_loss = torch.sum(hotspot_error) / num_hotspot
            background_loss = torch.sum(background_error) / num_background
            
            # Combine with weighting
            combined_loss = self.hotspot_weight * hotspot_loss + background_loss
            batch_losses.append(combined_loss)
        
        # Combine batch losses
        if self.reduction == 'mean':
            return torch.mean(torch.stack(batch_losses))
        elif self.reduction == 'sum':
            return torch.sum(torch.stack(batch_losses))
        else:  # 'none'
            return torch.stack(batch_losses)


class HotspotOverlapLoss(BaseLoss):
    """Combined loss that considers both intensity accuracy and spatial overlap of hotspots.
    
    This loss combines MSE with a Dice coefficient term that measures hotspot overlap,
    encouraging the model to correctly localize hotspot regions.
    """
    
    def __init__(self, mask_threshold: float = 0.01, reduction: str = 'mean',
                hotspot_percentile: float = 90.0, alpha: float = 0.5):
        """Initialize hotspot overlap loss.
        
        Args:
            mask_threshold: Threshold for masking low-intensity regions
            reduction: Reduction method ('mean', 'sum', 'none')
            hotspot_percentile: Percentile threshold to define hotspots
            alpha: Weight balance between MSE and Dice terms (0-1)
        """
        super().__init__(mask_threshold, reduction)
        self.hotspot_percentile = hotspot_percentile
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute hotspot overlap loss.
        
        Args:
            pred: Predicted magnitude field [B, 1, D, H, W]
            target: Target magnitude field [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: Hotspot overlap loss value
        """
        # Apply mask
        pred, target, mask = self._apply_mask(pred, target)
        
        # Calculate MSE component (intensity accuracy)
        squared_error = (pred - target)**2
        masked_error = squared_error * mask
        num_valid = torch.sum(mask, dim=(1, 2, 3, 4)) + 1e-8
        mse_loss = torch.sum(masked_error, dim=(1, 2, 3, 4)) / num_valid
        
        # Calculate Dice loss component (spatial accuracy of hotspots)
        batch_size = target.shape[0]
        dice_losses = []
        
        for b in range(batch_size):
            # Get masked values for this batch
            masked_target = target[b] * mask[b]
            masked_pred = pred[b] * mask[b]
            
            # Find hotspot thresholds (top percentile of non-zero values)
            # For target
            valid_target = masked_target[masked_target > 0]
            if len(valid_target) > 0:
                target_threshold = torch.quantile(valid_target, self.hotspot_percentile/100.0)
            else:
                target_threshold = self.mask_threshold
                
            # For prediction
            valid_pred = masked_pred[masked_pred > 0]
            if len(valid_pred) > 0:
                pred_threshold = torch.quantile(valid_pred, self.hotspot_percentile/100.0)
            else:
                pred_threshold = self.mask_threshold
            
            # Create binary hotspot masks
            target_hotspots = (masked_target > target_threshold).float()
            pred_hotspots = (masked_pred > pred_threshold).float()
            
            # Calculate Dice coefficient
            intersection = torch.sum(pred_hotspots * target_hotspots)
            union = torch.sum(pred_hotspots) + torch.sum(target_hotspots) + 1e-8
            dice = (2.0 * intersection) / union
            
            # Dice loss (1 - Dice coefficient)
            dice_loss = 1.0 - dice
            dice_losses.append(dice_loss)
        
        # Combine MSE and Dice losses with alpha weighting
        dice_loss_batch = torch.stack(dice_losses)
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * dice_loss_batch
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(combined_loss)
        elif self.reduction == 'sum':
            return torch.sum(combined_loss)
        else:  # 'none'
            return combined_loss


class GradientLoss(BaseLoss):
    """Loss that considers both intensity accuracy and spatial gradients.
    
    This loss emphasizes edges of hotspots by incorporating spatial gradients,
    helping the model to capture sharp transitions in the E-field.
    """
    
    def __init__(self, mask_threshold: float = 0.01, reduction: str = 'mean',
                lambda_grad: float = 0.5):
        """Initialize gradient loss.
        
        Args:
            mask_threshold: Threshold for masking low-intensity regions
            reduction: Reduction method ('mean', 'sum', 'none')
            lambda_grad: Weight for gradient component (0-1)
        """
        super().__init__(mask_threshold, reduction)
        self.lambda_grad = lambda_grad
    
    def _compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute spatial gradients.
        
        Args:
            x: Input tensor [B, 1, D, H, W]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Gradients in x, y, z directions
        """
        # Compute gradients along each spatial dimension
        dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        
        return dx, dy, dz
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient loss.
        
        Args:
            pred: Predicted magnitude field [B, 1, D, H, W]
            target: Target magnitude field [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: Gradient loss value
        """
        # Apply mask
        pred, target, mask = self._apply_mask(pred, target)
        
        # Calculate MSE component (intensity accuracy)
        squared_error = (pred - target)**2
        masked_error = squared_error * mask
        num_valid = torch.sum(mask, dim=(1, 2, 3, 4)) + 1e-8
        mse_loss = torch.sum(masked_error, dim=(1, 2, 3, 4)) / num_valid
        
        # Calculate gradient component
        # Compute gradients for prediction and target
        pred_dx, pred_dy, pred_dz = self._compute_gradients(pred)
        target_dx, target_dy, target_dz = self._compute_gradients(target)
        
        # Create masks for gradients (adjusted for gradient tensor shapes)
        mask_x = mask[:, :, :-1, :, :]
        mask_y = mask[:, :, :, :-1, :]
        mask_z = mask[:, :, :, :, :-1]
        
        # Calculate gradient MSE
        grad_error_x = ((pred_dx - target_dx) * mask_x)**2
        grad_error_y = ((pred_dy - target_dy) * mask_y)**2
        grad_error_z = ((pred_dz - target_dz) * mask_z)**2
        
        # Normalize by number of valid gradient elements
        num_valid_x = torch.sum(mask_x, dim=(1, 2, 3, 4)) + 1e-8
        num_valid_y = torch.sum(mask_y, dim=(1, 2, 3, 4)) + 1e-8
        num_valid_z = torch.sum(mask_z, dim=(1, 2, 3, 4)) + 1e-8
        
        grad_loss_x = torch.sum(grad_error_x, dim=(1, 2, 3, 4)) / num_valid_x
        grad_loss_y = torch.sum(grad_error_y, dim=(1, 2, 3, 4)) / num_valid_y
        grad_loss_z = torch.sum(grad_error_z, dim=(1, 2, 3, 4)) / num_valid_z
        
        # Combine gradient losses (average over dimensions)
        grad_loss = (grad_loss_x + grad_loss_y + grad_loss_z) / 3.0
        
        # Combine intensity and gradient losses
        combined_loss = (1.0 - self.lambda_grad) * mse_loss + self.lambda_grad * grad_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(combined_loss)
        elif self.reduction == 'sum':
            return torch.sum(combined_loss)
        else:  # 'none'
            return combined_loss

class UShapeMSELoss(BaseLoss):
    """MSE loss with U-shaped weighting that emphasizes errors at very low and very high true values.
    
    This loss applies higher weights to errors where the true value is either very low or very high,
    creating a U-shaped weighting curve based on parameters a, b, c, and max_val.
    """
    
    def __init__(
        self, 
        mask_threshold: float = 0.01, 
        reduction: str = 'mean',
        a: float = 4.0,
        b: float = 0.5,
        c: float = 1.0,
        max_val: float = 2.0
    ):
        """Initialize U-shape MSE loss.
        
        Args:
            mask_threshold: Threshold for masking low-intensity regions
            reduction: Reduction method ('mean', 'sum', 'none')
            a: Parameter controlling the width of the U-shape (higher = sharper U)
            b: Parameter controlling the center of the U-shape (typically 0.5)
            c: Parameter controlling the minimum weight (typically 1.0)
            max_val: Maximum weight value at the extremes
        """
        super().__init__(mask_threshold, reduction)
        self.a = a
        self.b = b
        self.c = c
        self.max_val = max_val
        
        logger.info(f"UShapeMSELoss initialized with: a={a}, b={b}, c={c}, max_val={max_val}")
    
    def _compute_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute U-shaped weights based on target values.
        
        For sharp threshold-based weighting with parameters mapped to percentiles:
        - For values <= low_percentile: weight = max_val
        - For values >= high_percentile: weight = max_val
        - For values in between: weight = c (typically 1.0)
        
        Args:
            target: Target tensor
            
        Returns:
            torch.Tensor: Weight tensor with same shape as target
        """
        # Get min and max values for normalization
        valid_targets = target[target > self.mask_threshold]
        if valid_targets.numel() == 0:
            return torch.ones_like(target)
            
        # Calculate percentile thresholds (10th and 90th percentiles by default)
        # We're converting b (typically 0.5) to determine thresholds
        # Lower b = wider middle range, higher b = narrower middle range
        low_percentile = self.b * 0.2  # 0.5 * 0.2 = 0.1 (10th percentile)
        high_percentile = 1.0 - low_percentile  # 1.0 - 0.1 = 0.9 (90th percentile)
        
        try:
            low_threshold = torch.quantile(valid_targets, low_percentile)
            high_threshold = torch.quantile(valid_targets, high_percentile)
        except:
            # Fallback if quantile calculation fails
            if valid_targets.numel() > 0:
                sorted_vals = torch.sort(valid_targets)[0]
                low_idx = max(0, int(low_percentile * valid_targets.numel()))
                high_idx = min(valid_targets.numel() - 1, int(high_percentile * valid_targets.numel()))
                low_threshold = sorted_vals[low_idx]
                high_threshold = sorted_vals[high_idx]
            else:
                # Default values if no valid targets
                low_threshold = self.mask_threshold
                high_threshold = 1.0
        
        # Create weight tensor (default weight = c)
        weights = torch.ones_like(target) * self.c
        
        # Set higher weights (max_val) for low and high values
        # Parameter a controls the sharpness of the transition
        # Here we use it to implement sharp thresholding
        low_mask = (target <= low_threshold) & (target > self.mask_threshold)
        high_mask = (target >= high_threshold)
        
        weights = torch.where(low_mask, torch.tensor(self.max_val, device=weights.device), weights)
        weights = torch.where(high_mask, torch.tensor(self.max_val, device=weights.device), weights)
        
        return weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute U-shape weighted MSE loss.
        
        Args:
            pred: Predicted magnitude field [B, 1, D, H, W]
            target: Target magnitude field [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: U-shape weighted MSE loss value
        """
        # Apply mask
        pred, target, mask = self._apply_mask(pred, target)
        
        # Calculate squared error
        squared_error = (pred - target)**2
        
        # Compute weights based on target values
        weights = self._compute_weights(target)
        
        # Apply weights and mask
        weighted_error = squared_error * weights * mask
        
        # Normalize by number of valid elements
        num_valid = torch.sum(mask, dim=(1, 2, 3, 4)) + 1e-8
        
        if self.reduction == 'mean':
            loss = torch.sum(weighted_error, dim=(1, 2, 3, 4)) / num_valid
            return loss.mean()
        elif self.reduction == 'sum':
            return torch.sum(weighted_error)
        else:  # 'none'
            return weighted_error


class LossFactory:
    """Factory for creating TMS E-field magnitude prediction losses."""
    
    @staticmethod
    def create_loss(
        loss_type: str,
        mask_threshold: float = 0.01,
        reduction: str = 'mean',
        **kwargs
    ) -> BaseLoss:
        """Create a loss function with the specified parameters.
        
        Args:
            loss_type: Loss function type ('mse', 'l1', 'weighted_mse', 'hotspot', etc.)
            mask_threshold: Threshold for masking low-intensity regions
            reduction: Reduction method ('mean', 'sum', 'none')
            **kwargs: Additional loss-specific parameters
            
        Returns:
            BaseLoss: Configured loss function
        """
        if loss_type.lower() == 'mse' or loss_type.lower() == 'magnitude_mse':
            return MaskedMSELoss(mask_threshold, reduction)
        
        elif loss_type.lower() == 'l1' or loss_type.lower() == 'mae':
            return MaskedL1Loss(mask_threshold, reduction)
        
        elif loss_type.lower() == 'weighted_mse':
            weight_factor = kwargs.get('weight_factor', 2.0)
            return WeightedMSELoss(
                mask_threshold=mask_threshold,
                reduction=reduction,
                weight_factor=weight_factor
            )
        
        elif loss_type.lower() == 'hotspot' or loss_type.lower() == 'hotspot_mse':
            hotspot_percentile = kwargs.get('hotspot_percentile', 90.0)
            hotspot_weight = kwargs.get('hotspot_weight', 5.0)
            return HotspotMSELoss(
                mask_threshold=mask_threshold,
                reduction=reduction,
                hotspot_percentile=hotspot_percentile,
                hotspot_weight=hotspot_weight
            )
        
        elif loss_type.lower() == 'hotspot_overlap' or loss_type.lower() == 'overlap':
            hotspot_percentile = kwargs.get('hotspot_percentile', 90.0)
            alpha = kwargs.get('alpha', 0.5)
            return HotspotOverlapLoss(
                mask_threshold=mask_threshold,
                reduction=reduction,
                hotspot_percentile=hotspot_percentile,
                alpha=alpha
            )
        elif loss_type.lower() == 'ushape' or loss_type.lower() == 'ushape_mse':
            a = kwargs.get('a', 4.0)
            b = kwargs.get('b', 0.5)
            c = kwargs.get('c', 1.0)
            max_val = kwargs.get('max_val', 2.0)
            return UShapeMSELoss(
                mask_threshold=mask_threshold,
                reduction=reduction,
                a=a,
                b=b,
                c=c,
                max_val=max_val
            )
        elif loss_type.lower() == 'gradient':
            lambda_grad = kwargs.get('lambda_grad', 0.5)
            return GradientLoss(
                mask_threshold=mask_threshold,
                reduction=reduction,
                lambda_grad=lambda_grad
            )
        

        elif loss_type.lower() == 'ushape' or loss_type.lower() == 'ushape_mse':
            low_threshold = kwargs.get('low_threshold', 0.1)
            high_threshold = kwargs.get('high_threshold', 0.9)
            extreme_weight = kwargs.get('extreme_weight', 1.5)
            middle_weight = kwargs.get('middle_weight', 1.0)
            
            return UShapeMSELoss(
                mask_threshold=mask_threshold,
                reduction=reduction,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                extreme_weight=extreme_weight,
                middle_weight=middle_weight
            )

        else:
            logger.warning(f"Unknown loss type '{loss_type}'. Defaulting to MSE.")
            return MaskedMSELoss(mask_threshold, reduction)
        

        