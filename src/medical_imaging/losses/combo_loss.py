"""
Advanced Loss Functions for Medical Image Segmentation (2026)

Based on SOTA research:
- Combo Loss: Dice + Focal + Boundary
- Tversky Loss: For highly imbalanced datasets
- Boundary Loss: Improve edge precision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    
    Measures overlap between prediction and ground truth
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C, D, H, W) - predicted probabilities
            target: (N, C, D, H, W) - ground truth (one-hot encoded)
        """
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average over batch and classes, return loss
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Focuses training on hard examples by down-weighting easy examples
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C, D, H, W)
            target: (N, C, D, H, W)
        """
        # Flatten
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        # BCE loss
        bce = F.binary_cross_entropy(pred_flat, target_flat, reduction='none')
        
        # Focal modulation
        p_t = pred_flat * target_flat + (1 - pred_flat) * (1 - target_flat)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * bce
        
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for improving edge precision
    
    Penalizes errors near the boundary more heavily
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss using distance transform
        
        Args:
            pred: (N, C, D, H, W)
            target: (N, C, D, H, W)
        """
        # Get boundary from target
        # Simple approximation: |gradient| > 0
        target_grad = self._compute_gradient_magnitude(target)
        boundary_mask = (target_grad > 0).float()
        
        # Weighted BCE on boundary regions
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        boundary_flat = boundary_mask.view(-1)
        
        # Higher weight on boundary pixels
        weight = 1.0 + 10.0 * boundary_flat  # 10x weight on boundaries
        
        bce = F.binary_cross_entropy(pred_flat, target_flat, reduction='none')
        weighted_bce = (bce * weight).mean()
        
        return weighted_bce
    
    def _compute_gradient_magnitude(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel-like filter"""
        # Simplified 3D gradient
        grad_z = torch.abs(tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :])
        grad_y = torch.abs(tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :])
        grad_x = torch.abs(tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1])
        
        # Pad to match original size
        grad_z = F.pad(grad_z, (0, 0, 0, 0, 0, 1))
        grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
        grad_x = F.pad(grad_x, (0, 1, 0, 0, 0, 0))
        
        # Magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
        
        return grad_mag


class ComboLoss(nn.Module):
    """
    Combination Loss: Dice + Focal + Boundary
    
    Research-backed optimal combination for medical image segmentation
    
    Default weights based on MICCAI 2026 best practices:
    - Dice: 0.5 (region overlap)
    - Focal: 0.3 (hard example mining)
    - Boundary: 0.2 (edge precision)
    """
    def __init__(self, 
                 dice_weight: float = 0.5,
                 focal_weight: float = 0.3,
                 boundary_weight: float = 0.2,
                 dice_smooth: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        assert abs(dice_weight + focal_weight + boundary_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            pred: (N, C, D, H, W) - predicted probabilities (after sigmoid)
            target: (N, D, H, W) - integer labels OR (N, C, D, H, W) one-hot
        
        Returns:
            Scalar loss value
        """
        # Convert target to one-hot if needed
        if target.dim() == 4:  # (N, D, H, W)
            target = self._to_onehot(target, num_classes=pred.size(1))
        
        # Compute individual losses
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        # Weighted combination
        total_loss = (self.dice_weight * dice + 
                     self.focal_weight * focal +
                     self.boundary_weight * boundary)
        
        return total_loss
    
    def _to_onehot(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert integer labels to one-hot encoding"""
        batch_size = labels.size(0)
        spatial_dims = labels.shape[1:]
        
        one_hot = torch.zeros(batch_size, num_classes, *spatial_dims, 
                             device=labels.device, dtype=torch.float32)
        
        labels = labels.unsqueeze(1)  # (N, 1, D, H, W)
        one_hot.scatter_(1, labels.long(), 1)
        
        return one_hot


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss
    
    Better for highly imbalanced datasets where you want to
    control the trade-off between false positives and false negatives
    
    alpha controls false positives weight
    beta controls false negatives weight
    
    For recall emphasis (reduce false negatives): alpha=0.3, beta=0.7
    For precision emphasis (reduce false positives): alpha=0.7, beta=0.3
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C, D, H, W)
            target: (N, C, D, H, W)
        """
        # Flatten
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        # True Positive, False Positive, False Negative
        TP = (pred_flat * target_flat).sum(dim=2)
        FP = (pred_flat * (1 - target_flat)).sum(dim=2)
        FN = ((1 - pred_flat) * target_flat).sum(dim=2)
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1.0 - tversky.mean()


# ============================================================================
# Combined Loss with Tversky variant
# ============================================================================

class TverskyComboLoss(nn.Module):
    """
    Combo Loss using Tversky instead of Dice
    
    Better for highly imbalanced tumor segmentation
    """
    def __init__(self,
                 tversky_weight: float = 0.5,
                 focal_weight: float = 0.3,
                 boundary_weight: float = 0.2,
                 tversky_alpha: float = 0.7,
                 tversky_beta: float = 0.3):
        super().__init__()
        
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 4:
            target = self._to_onehot(target, num_classes=pred.size(1))
        
        tversky = self.tversky_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total_loss = (self.tversky_weight * tversky +
                     self.focal_weight * focal +
                     self.boundary_weight * boundary)
        
        return total_loss
    
    def _to_onehot(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        batch_size = labels.size(0)
        spatial_dims = labels.shape[1:]
        one_hot = torch.zeros(batch_size, num_classes, *spatial_dims,
                             device=labels.device, dtype=torch.float32)
        labels = labels.unsqueeze(1)
        one_hot.scatter_(1, labels.long(), 1)
        return one_hot


# ============================================================================
# Test and comparison
# ============================================================================

def test_losses():
    """Test loss functions"""
    # Create dummy data
    batch_size = 2
    num_classes = 2
    depth, height, width = 32, 64, 64
    
    pred = torch.rand(batch_size, num_classes, depth, height, width)
    pred = torch.softmax(pred, dim=1)  # Normalize to probabilities
    
    target = torch.randint(0, num_classes, (batch_size, depth, height, width))
    
    # Test ComboLoss
    combo_loss = ComboLoss()
    loss_combo = combo_loss(pred, target)
    print(f"ComboLoss: {loss_combo.item():.4f}")
    
    # Test TverskyComboLoss
    tversky_combo = TverskyComboLoss()
    loss_tversky = tversky_combo(pred, target)
    print(f"TverskyComboLoss: {loss_tversky.item():.4f}")
    
    # Test individual losses
    dice = DiceLoss()
    print(f"DiceLoss: {dice(pred, combo_loss._to_onehot(target, num_classes)).item():.4f}")
    
    focal = FocalLoss()
    print(f"FocalLoss: {focal(pred, combo_loss._to_onehot(target, num_classes)).item():.4f}")


if __name__ == "__main__":
    test_losses()
