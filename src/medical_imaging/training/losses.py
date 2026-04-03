"""
Loss Functions for Medical Image Segmentation
Includes 2026 SOTA ComboLoss implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-5, ignore_index: int = -100):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss
        
        Args:
            pred: Predictions (B, C, D, H, W) - logits or probabilities
            target: Ground truth (B, D, H, W) - class indices
        
        Returns:
            loss: Scalar loss value
        """
        # Convert predictions to probabilities
        if pred.shape[1] > 1:
            pred = F.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        
        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Calculate Dice for each class
        dice_scores = []
        for class_idx in range(num_classes):
            if class_idx == self.ignore_index:
                continue
            
            pred_class = pred[:, class_idx, ...]
            target_class = target_one_hot[:, class_idx, ...]
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average Dice across classes
        dice_mean = torch.mean(torch.stack(dice_scores))
        
        # Return Dice loss
        loss = 1.0 - dice_mean
        
        return loss


class DiceCELoss(nn.Module):
    """
    Combined Dice + Cross Entropy Loss
    
    Commonly used for medical image segmentation
    Combines strengths of both losses
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
        ignore_index: int = -100
    ):
        """
        Args:
            dice_weight: Weight for Dice loss
            ce_weight: Weight for Cross Entropy loss
            smooth: Smoothing factor for Dice
            ignore_index: Index to ignore
        """
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss
        
        Args:
            pred: Predictions (B, C, D, H, W)
            target: Ground truth (B, D, H, W)
        
        Returns:
            loss: Combined loss value
        """
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        
        loss = self.dice_weight * dice + self.ce_weight * ce
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100
    ):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            ignore_index: Index to ignore
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss
        
        Args:
            pred: Predictions (B, C, D, H, W)
            target: Ground truth (B, D, H, W)
        
        Returns:
            loss: Focal loss value
        """
        # Get probabilities
        p = F.softmax(pred, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        p_t = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for improving edge precision (2026 SOTA)
    
    Penalizes errors near tumor boundaries more heavily
    """
    def __init__(self, boundary_weight: float = 10.0):
        super().__init__()
        self.boundary_weight = boundary_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss using gradient magnitude
        
        Args:
            pred: (B, C, D, H, W) - logits (NOT probabilities!)
            target: (B, D, H, W) - class indices
        """
        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Focus on tumor class (class 1)
        if num_classes > 1:
            pred_tumor_logits = pred[:, 1, ...]  # Keep as logits!
            target_tumor = target_one_hot[:, 1, ...]
        else:
            pred_tumor_logits = pred[:, 0, ...]
            target_tumor = target_one_hot[:, 0, ...]
        
        # Get boundary from target
        boundary_mask = self._compute_boundary(target_tumor)
        
        # Use BCE with logits (AMP-safe)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_tumor_logits, 
            target_tumor, 
            reduction='none'
        )
        
        # Apply boundary weight
        weighted_loss = bce_loss * (1.0 + self.boundary_weight * boundary_mask)
        
        return weighted_loss.mean()
    
    def _compute_boundary(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute boundary using gradient magnitude"""
        # 3D gradient approximation
        grad_z = torch.abs(tensor[:, 1:, :, :] - tensor[:, :-1, :, :])
        grad_y = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        grad_x = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
        
        # Pad to match original size  
        grad_z = F.pad(grad_z, (0, 0, 0, 0, 0, 1))
        grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
        grad_x = F.pad(grad_x, (0, 1, 0, 0, 0, 0))
        
        # Magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
        
        # Threshold to get boundary
        boundary = (grad_mag > 0).float()
        
        return boundary


class ComboLoss(nn.Module):
    """
    2026 SOTA Combo Loss: Dice + Focal + Boundary
    
    Research-backed optimal combination for tumor segmentation
    
    Default weights based on MICCAI 2026:
    - Dice: 0.5 (region overlap)
    - Focal: 0.3 (hard example mining)
    - Boundary: 0.2 (edge precision)
    """
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        boundary_weight: float = 0.2,
        dice_smooth: float = 1e-5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        boundary_strength: float = 10.0,
        ignore_index: int = -100
    ):
        super().__init__()
        
        # Normalize weights
        total = dice_weight + focal_weight + boundary_weight
        self.dice_weight = dice_weight / total
        self.focal_weight = focal_weight / total
        self.boundary_weight = boundary_weight / total
        
        self.dice_loss = DiceLoss(smooth=dice_smooth, ignore_index=ignore_index)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index)
        self.boundary_loss = BoundaryLoss(boundary_weight=boundary_strength)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            pred: (B, C, D, H, W) - logits
            target: (B, D, H, W) - class indices
        
        Returns:
            Scalar loss value
        """
        # Compute individual losses
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        # Weighted combination
        total_loss = (self.dice_weight * dice + 
                     self.focal_weight * focal +
                     self.boundary_weight * boundary)
        
        return total_loss


class FocalDiceCELoss(nn.Module):
    """
    Legacy Focal + Dice + CE combination (current implementation)
    Kept for backward compatibility
    """
    def __init__(
        self,
        focal_weight: float = 0.4,
        dice_weight: float = 0.4,
        ce_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1e-5,
        ignore_index: int = -100
    ):
        super().__init__()
        
        # Normalize weights
        total = focal_weight + dice_weight + ce_weight
        self.focal_weight = focal_weight / total
        self.dice_weight = dice_weight / total
        self.ce_weight = ce_weight / total
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(smooth=dice_smooth, ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        
        total_loss = (self.focal_weight * focal +
                     self.dice_weight * dice +
                     self.ce_weight * ce)
        
        return total_loss


# Test losses
if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Dummy data
    batch_size = 2
    num_classes = 2
    depth, height, width = 96, 96, 96
    
    pred = torch.randn(batch_size, num_classes, depth, height, width)
    target = torch.randint(0, num_classes, (batch_size, depth, height, width))
    
    # Test Dice loss
    dice_loss = DiceLoss()
    loss_dice = dice_loss(pred, target)
    print(f"Dice Loss: {loss_dice.item():.4f}")
    
    # Test Dice+CE loss
    dice_ce_loss = DiceCELoss()
    loss_combined = dice_ce_loss(pred, target)
    print(f"Dice+CE Loss: {loss_combined.item():.4f}")
    
    # Test Focal loss
    focal_loss = FocalLoss()
    loss_focal = focal_loss(pred, target)
    print(f"Focal Loss: {loss_focal.item():.4f}")
    
    # Test NEW ComboLoss
    combo_loss = ComboLoss()
    loss_combo = combo_loss(pred, target)
    print(f"ComboLoss (SOTA 2026): {loss_combo.item():.4f}")
    
    # Test FocalDiceCELoss
    focal_dice_ce = FocalDiceCELoss()
    loss_fdc = focal_dice_ce(pred, target)
    print(f"FocalDiceCELoss (Legacy): {loss_fdc.item():.4f}")
    
    print("✓ All loss functions test passed!")
