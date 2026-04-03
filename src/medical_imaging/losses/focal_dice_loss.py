"""
Focal Loss and Combined Loss Functions
Implements Focal Loss for class imbalance (Lin et al. 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t: predicted probability for the true class
    - α: class weight (balances positive/negative examples)
    - γ: focusing parameter (reduces weight of easy examples)
    
    Args:
        alpha: Weighting factor in [0, 1] for class 1 (tumor)
               alpha=0.25 means tumor class gets 0.25 weight
        gamma: Focusing parameter >= 0. gamma=2 is default
        reduction: 'mean', 'sum', or 'none'
        
    Usage:
        For CT tumor detection with severe class imbalance:
        - Background pixels >> Tumor pixels
        - Use alpha=0.25, gamma=2.0 to focus on hard tumor examples
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, *) predictions (logits)
            targets: (B, *) ground truth (class indices)
                    
        Returns:
            Focal loss value
        """
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get batch size and flatten spatial dimensions
        batch_size = inputs.shape[0]
        num_classes = inputs.shape[1]
        
        # Flatten: (B, C, *) -> (B*N, C) where N = product of spatial dims
        probs_flat = probs.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes) if len(inputs.shape) == 5 else \
                     probs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes) if len(inputs.shape) == 4 else \
                     probs.view(-1, num_classes)
        
        # Flatten targets: (B, *) -> (B*N,)
        targets_flat = targets.view(-1).long()
        
        # Get probability of true class
        targets_one_hot = F.one_hot(targets_flat, num_classes=num_classes).float()
        p_t = (probs_flat * targets_one_hot).sum(dim=1)  # (B*N,)
        
        # Focal weight: (1 - p_t)^gamma
        # This focuses on hard examples (low p_t -> high weight)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Cross entropy: -log(p_t)
        ce = -torch.log(p_t + 1e-8)
        
        # Alpha weighting for class balance
        if self.alpha is not None:
            # alpha for positive class (tumor=1), (1-alpha) for negative class (background=0)
            alpha_t = torch.where(
                targets_flat == 1,
                torch.tensor(self.alpha, device=inputs.device),
                torch.tensor(1 - self.alpha, device=inputs.device)
            )
            focal_loss = alpha_t * focal_weight * ce
        else:
            focal_loss = focal_weight * ce
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    
    Args:
        smooth: Smoothing constant to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, *) predictions (logits)
            targets: (B, C, *) or (B, *) ground truth
            
        Returns:
            Dice loss value
        """
        # Get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot if needed
        if inputs.dim() > targets.dim():
            num_classes = inputs.shape[1]
            targets = F.one_hot(
                targets.long(),
                num_classes=num_classes
            ).movedim(-1, 1).float()
        
        # Flatten spatial dimensions
        inputs = inputs.flatten(2)  # (B, C, N)
        targets = targets.flatten(2)  # (B, C, N)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum(dim=2)  # (B, C)
        union = inputs.sum(dim=2) + targets.sum(dim=2)  # (B, C)
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss
        dice_loss = 1.0 - dice
        
        # Reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FocalDiceCELoss(nn.Module):
    """
    Combined Loss: Focal + Dice + CrossEntropy
    
    This addresses multiple challenges in CT tumor detection:
    1. Focal Loss: Handles class imbalance (background >> tumor)
    2. Dice Loss: Optimizes boundary precision
    3. CE Loss: Provides stable gradients
    
    Total Loss = w_focal * FL + w_dice * DL + w_ce * CE
    
    Args:
        focal_weight: Weight for focal loss (default: 0.4)
        dice_weight: Weight for dice loss (default: 0.4)
        ce_weight: Weight for cross entropy (default: 0.2)
        focal_alpha: Alpha parameter for focal loss (default: 0.25)
        focal_gamma: Gamma parameter for focal loss (default: 2.0)
        dice_smooth: Smoothing constant for dice (default: 1e-5)
        
    Recommended weights for CT tumor detection:
    - focal_weight=0.4: Strong focus on hard examples
    - dice_weight=0.4: Maintain boundary accuracy
    - ce_weight=0.2: Provide stable gradients
    """
    
    def __init__(
        self,
        focal_weight: float = 0.4,
        dice_weight: float = 0.4,
        ce_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1e-5
    ):
        super().__init__()
        
        # Validate weights
        total_weight = focal_weight + dice_weight + ce_weight
        assert abs(total_weight - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {total_weight}"
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # Loss components
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean'
        )
        self.dice_loss = DiceLoss(
            smooth=dice_smooth,
            reduction='mean'
        )
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, *) predictions (logits)
            targets: (B, *) ground truth (class indices)
            
        Returns:
            Combined loss value
        """
        # Calculate individual losses
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        
        # Weighted combination
        total_loss = (
            self.focal_weight * focal +
            self.dice_weight * dice +
            self.ce_weight * ce
        )
        
        return total_loss
    
    def get_component_losses(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> dict:
        """
        Return individual loss components for monitoring
        
        Useful for TensorBoard logging to see which component
        is contributing most to the total loss
        
        Returns:
            Dictionary with keys: 'focal', 'dice', 'ce', 'total'
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        total = (
            self.focal_weight * focal +
            self.dice_weight * dice +
            self.ce_weight * ce
        )
        
        return {
            'focal': focal.item(),
            'dice': dice.item(),
            'ce': ce.item(),
            'total': total.item()
        }


# Test functions
if __name__ == "__main__":
    print("Testing Focal Loss implementations...")
    
    # Create dummy data
    batch_size = 2
    num_classes = 2
    spatial_size = (96, 96, 96)
    
    # Predictions (logits)
    inputs = torch.randn(batch_size, num_classes, *spatial_size)
    
    # Ground truth (class indices)
    targets = torch.randint(0, num_classes, (batch_size, *spatial_size))
    
    print(f"\nInput shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Test Focal Loss
    print("\n1. Testing Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(inputs, targets)
    print(f"   Focal Loss: {loss.item():.4f}")
    
    # Test Dice Loss
    print("\n2. Testing Dice Loss...")
    dice_loss = DiceLoss(smooth=1e-5)
    loss = dice_loss(inputs, targets)
    print(f"   Dice Loss: {loss.item():.4f}")
    
    # Test Combined Loss
    print("\n3. Testing FocalDiceCE Loss...")
    combined_loss = FocalDiceCELoss(
        focal_weight=0.4,
        dice_weight=0.4,
        ce_weight=0.2,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    loss = combined_loss(inputs, targets)
    components = combined_loss.get_component_losses(inputs, targets)
    
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   Components:")
    print(f"     - Focal: {components['focal']:.4f}")
    print(f"     - Dice: {components['dice']:.4f}")
    print(f"     - CE: {components['ce']:.4f}")
    
    # Test backward pass
    print("\n4. Testing backward pass...")
    loss.backward()
    print("   ✓ Gradient computation successful")
    
    print("\n✓ All tests passed!")
