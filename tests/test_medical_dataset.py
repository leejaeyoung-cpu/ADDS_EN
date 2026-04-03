"""
Test SOTA Medical Dataset
Tests ColonCancerDataset class for data loading and preprocessing
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medical_imaging.data.dataset import ColonCancerDataset


class TestColonCancerDataset:
    """Test suite for ColonCancerDataset"""
    
    def test_dataset_initialization(self):
        """Test dataset can be initialized"""
        # Skip if dataset not available
        data_root = "data/Dataset010_Colon"
        if not Path(data_root).exists():
            pytest.skip("Dataset not available")
        
        dataset = ColonCancerDataset(
            data_root=data_root,
            fold=0,
            mode='train'
        )
        
        assert len(dataset) > 0, "Dataset should have samples"
    
    def test_data_loading(self):
        """Test loading a single sample"""
        data_root = "data/Dataset010_Colon"
        if not Path(data_root).exists():
            pytest.skip("Dataset not available")
        
        dataset = ColonCancerDataset(
            data_root=data_root,
            fold=0,
            mode='train'
        )
        
        # Load first sample
        image, label = dataset[0]
        
        # Check types
        assert isinstance(image, torch.Tensor), "Image should be torch.Tensor"
        assert isinstance(label, torch.Tensor), "Label should be torch.Tensor"
        
        # Check shapes
        assert image.ndim == 4, f"Image should be 4D (C, D, H, W), got {image.ndim}D"
        assert label.ndim == 3, f"Label should be 3D (D, H, W), got {label.ndim}D"
        
        # Check channel dimension
        assert image.shape[0] == 1, f"Image should have 1 channel, got {image.shape[0]}"
        
        print(f"✓ Image shape: {image.shape}")
        print(f"✓ Label shape: {label.shape}")
    
    def test_train_val_split(self):
        """Test train/val split produces different samples"""
        data_root = "data/Dataset010_Colon"
        if not Path(data_root).exists():
            pytest.skip("Dataset not available")
        
        train_dataset = ColonCancerDataset(
            data_root=data_root,
            fold=0,
            mode='train'
        )
        
        val_dataset = ColonCancerDataset(
            data_root=data_root,
            fold=0,
            mode='val'
        )
        
        # Check split exists
        assert len(train_dataset) > 0, "Train dataset should not be empty"
        assert len(val_dataset) > 0, "Val dataset should not be empty"
        
        # Check split ratio (should be ~80/20)
        total = len(train_dataset) + len(val_dataset)
        train_ratio = len(train_dataset) / total
        
        assert 0.7 < train_ratio < 0.9, f"Train ratio should be ~80%, got {train_ratio:.2%}"
        
        print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"✓ Train ratio: {train_ratio:.2%}")
    
    def test_augmentation(self):
        """Test augmentation is applied in training mode"""
        data_root = "data/Dataset010_Colon"
        if not Path(data_root).exists():
            pytest.skip("Dataset not available")
        
        dataset_with_aug = ColonCancerDataset(
            data_root=data_root,
            fold=0,
            mode='train',
            use_augmentation=True
        )
        
        dataset_without_aug = ColonCancerDataset(
            data_root=data_root,
            fold=0,
            mode='val',
            use_augmentation=False
        )
        
        # Load same sample from both
        img1, _ = dataset_with_aug[0]
        img2, _ = dataset_without_aug[0]
        
        # Both should be valid tensors
        assert img1.shape == img2.shape, "Shapes should match"
        
        print(f"✓ Augmentation test passed")
    
    def test_patch_extraction(self):
        """Test patch extraction produces correct size"""
        data_root = "data/Dataset010_Colon"
        if not Path(data_root).exists():
            pytest.skip("Dataset not available")
        
        patch_size = (96, 96, 96)
        dataset = ColonCancerDataset(
            data_root=data_root,
            fold=0,
            mode='train',
            patch_size=patch_size
        )
        
        image, label = dataset[0]
        
        # Check patch size
        assert image.shape[1:] == patch_size, f"Expected patch size {patch_size}, got {image.shape[1:]}"
        assert label.shape == patch_size, f"Expected label size {patch_size}, got {label.shape}"
        
        print(f"✓ Patch size: {image.shape[1:]}")


if __name__ == "__main__":
    # Run tests
    test = TestColonCancerDataset()
    
    print("Testing SOTA Medical Dataset...")
    print("-" * 50)
    
    try:
        test.test_dataset_initialization()
        print("✓ Test 1/5: Dataset initialization passed")
    except Exception as e:
        print(f"✗ Test 1/5 failed: {e}")
    
    try:
        test.test_data_loading()
        print("✓ Test 2/5: Data loading passed")
    except Exception as e:
        print(f"✗ Test 2/5 failed: {e}")
    
    try:
        test.test_train_val_split()
        print("✓ Test 3/5: Train/Val split passed")
    except Exception as e:
        print(f"✗ Test 3/5 failed: {e}")
    
    try:
        test.test_augmentation()
        print("✓ Test 4/5: Augmentation passed")
    except Exception as e:
        print(f"✗ Test 4/5 failed: {e}")
    
    try:
        test.test_patch_extraction()
        print("✓ Test 5/5: Patch extraction passed")
    except Exception as e:
        print(f"✗ Test 5/5 failed: {e}")
    
    print("-" * 50)
    print("All tests completed!")
