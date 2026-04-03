"""
Unit Tests for Focal Loss Implementation
Tests mathematical correctness and integration
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from medical_imaging.losses import FocalLoss, DiceLoss, FocalDiceCELoss


class TestFocalLoss:
    """Test Focal Loss implementation"""
    
    def test_focal_loss_shape(self):
        """Test output shape"""
        batch_size = 2
        num_classes = 2
        spatial = (32, 32, 32)
        
        inputs = torch.randn(batch_size, num_classes, *spatial)
        targets = torch.randint(0, num_classes, (batch_size, *spatial))
        
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        loss = focal_loss(inputs, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        print(f"✓ Focal Loss output shape: {loss.shape}")
    
    def test_focal_loss_focuses_on_hard_examples(self):
        """Test that focal loss assigns higher weight to hard examples"""
        batch_size = 1
        num_classes = 2
       
        # Easy example: high confidence, correct prediction (class 0)
        easy_inputs = torch.tensor([[[[10.0, 10.0], [10.0, 10.0]],   # Class 0: high
                                      [[-10.0, -10.0], [-10.0, -10.0]]]])  # Class 1: low
        easy_targets = torch.tensor([[[0, 0], [0, 0]]])  # True class: 0
        
        # Hard example: low confidence, uncertain prediction
        hard_inputs = torch.tensor([[[[0.1, 0.1], [0.1, 0.1]],     # Class 0: ~50%
                                      [[0.1, 0.1], [0.1, 0.1]]]])    # Class 1: ~50%
        hard_targets = torch.tensor([[[0, 0], [0, 0]]])  # True class: 0
        
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        easy_loss = focal_loss(easy_inputs, easy_targets)
        hard_loss = focal_loss(hard_inputs, hard_targets)
        
        # Hard examples should have higher loss
        # Easy examples with p_t ≈ 1.0 should have (1-p_t)^2 ≈ 0 -> loss ≈ 0
        # Hard examples with p_t ≈ 0.5 should have (1-p_t)^2 = 0.25 -> higher loss
        assert hard_loss > easy_loss
        assert easy_loss < 0.01  # Easy loss should be very small
        assert hard_loss > 0.1   # Hard loss should be significant
        print(f"[OK] Easy loss: {easy_loss:.6f}, Hard loss: {hard_loss:.6f}")
    
    def test_focal_loss_backward(self):
        """Test gradient computation"""
        inputs = torch.randn(1, 2, 16, 16, 16, requires_grad=True)
        targets = torch.randint(0, 2, (1, 16, 16, 16))
        
        focal_loss = FocalLoss()
        loss = focal_loss(inputs, targets)
        loss.backward()
        
        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()
        print(f"✓ Gradient shape: {inputs.grad.shape}")


class TestDiceLoss:
    """Test Dice Loss implementation"""
    
    def test_dice_loss_perfect_prediction(self):
        """Perfect prediction should have loss close to 0"""
        batch_size = 1
        num_classes = 2
        size = 32
        
        # Perfect prediction
        inputs = torch.zeros(batch_size, num_classes, size, size, size)
        inputs[:, 1] = 10.0  # High confidence for class 1
        targets = torch.ones(batch_size, size, size, size, dtype=torch.long)
        
        dice_loss = DiceLoss(smooth=1e-5)
        loss = dice_loss(inputs, targets)
        
        assert loss < 0.1  # Should be very small
        print(f"✓ Perfect prediction Dice Loss: {loss:.6f}")
    
    def test_dice_loss_random_prediction(self):
        """Random prediction should have high loss"""
        batch_size = 1
        num_classes = 2
        size = 16
        
        # Random prediction
        inputs = torch.randn(batch_size, num_classes, size, size, size)
        targets = torch.randint(0, num_classes, (batch_size, size, size, size))
        
        dice_loss = DiceLoss()
        loss = dice_loss(inputs, targets)
        
        assert loss > 0.3  # Should be relatively high
        print(f"✓ Random prediction Dice Loss: {loss:.4f}")


class TestFocalDiceCELoss:
    """Test combined loss function"""
    
    def test_combined_loss_weights_sum_to_one(self):
        """Weights should sum to 1.0"""
        # Valid weights
        loss_fn = FocalDiceCELoss(
            focal_weight=0.4,
            dice_weight=0.4,
            ce_weight=0.2
        )
        total_weight = loss_fn.focal_weight + loss_fn.dice_weight + loss_fn.ce_weight
        assert abs(total_weight - 1.0) < 1e-6
        print(f"✓ Weights sum: {total_weight}")
        
        # Invalid weights should raise error
        with pytest.raises(AssertionError):
            FocalDiceCELoss(
                focal_weight=0.5,
                dice_weight=0.5,
                ce_weight=0.5  # Sum > 1.0
            )
        print("✓ Invalid weights rejected")
    
    def test_combined_loss_computation(self):
        """Test combined loss computation"""
        batch_size = 2
        num_classes = 2
        size = 32
        
        inputs = torch.randn(batch_size, num_classes, size, size, size)
        targets = torch.randint(0, num_classes, (batch_size, size, size, size))
        
        combined_loss = FocalDiceCELoss(
            focal_weight=0.4,
            dice_weight=0.4,
            ce_weight=0.2,
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        loss = combined_loss(inputs, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        print(f"✓ Combined loss: {loss:.4f}")
    
    def test_get_component_losses(self):
        """Test component loss monitoring"""
        batch_size = 1
        num_classes = 2
        size = 16
        
        inputs = torch.randn(batch_size, num_classes, size, size, size)
        targets = torch.randint(0, num_classes, (batch_size, size, size, size))
        
        combined_loss = FocalDiceCELoss()
        components = combined_loss.get_component_losses(inputs, targets)
        
        assert 'focal' in components
        assert 'dice' in components
        assert 'ce' in components
        assert 'total' in components
        
        print(f"✓ Component losses:")
        print(f"  - Focal: {components['focal']:.4f}")
        print(f"  - Dice: {components['dice']:.4f}")
        print(f"  - CE: {components['ce']:.4f}")
        print(f"  - Total: {components['total']:.4f}")
    
    def test_combined_loss_backward(self):
        """Test gradient flow through combined loss"""
        inputs = torch.randn(1, 2, 16, 16, 16, requires_grad=True)
        targets = torch.randint(0, 2, (1, 16, 16, 16))
        
        combined_loss = FocalDiceCELoss()
        loss = combined_loss(inputs, targets)
        loss.backward()
        
        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()
        print(f"✓ Gradient computation successful, mean grad: {inputs.grad.mean():.6f}")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("Testing Focal Loss Implementation")
    print("=" * 70)
    
    # Focal Loss Tests
    print("\n1. Focal Loss Tests")
    print("-" * 70)
    test_focal = TestFocalLoss()
    test_focal.test_focal_loss_shape()
    test_focal.test_focal_loss_focuses_on_hard_examples()
    test_focal.test_focal_loss_backward()
    
    # Dice Loss Tests
    print("\n2. Dice Loss Tests")
    print("-" * 70)
    test_dice = TestDiceLoss()
    test_dice.test_dice_loss_perfect_prediction()
    test_dice.test_dice_loss_random_prediction()
    
    # Combined Loss Tests
    print("\n3. Combined Loss (FocalDiceCE) Tests")
    print("-" * 70)
    test_combined = TestFocalDiceCELoss()
    test_combined.test_combined_loss_weights_sum_to_one()
    test_combined.test_combined_loss_computation()
    test_combined.test_get_component_losses()
    test_combined.test_combined_loss_backward()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
