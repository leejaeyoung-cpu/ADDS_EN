"""
Unit Tests for 3D Context Preprocessing Functions
Tests contextual slicing, channel stacking, and tumor-focused sampling
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from medical_imaging.data.preprocessing_pipeline import MedicalPreprocessor


class Test3DContextSlicing:
    """Test 3D context extraction methods"""
    
    def setup_method(self):
        """Setup test data"""
        self.preprocessor = MedicalPreprocessor()
        # Create dummy 3D volume (100 slices, 512x512)
        self.volume = np.random.randn(100, 512, 512).astype(np.float32)
    
    def test_get_contextual_slices_middle(self):
        """Test context extraction from middle of volume"""
        slice_idx = 50
        context_size = 5
        
        context = self.preprocessor.get_contextual_slices(
            self.volume, slice_idx, context_size
        )
        
        # Should return 5 slices: [48, 49, 50, 51, 52]
        assert context.shape == (5, 512, 512)
        
        # Verify it's the correct slices
        assert np.array_equal(context[2], self.volume[50])  # Middle slice
        assert np.array_equal(context[0], self.volume[48])  # First slice
        assert np.array_equal(context[4], self.volume[52])  # Last slice
        
        print(f"✓ Middle context extraction: shape {context.shape}")
    
    def test_get_contextual_slices_boundary_start(self):
        """Test context extraction at volume start (needs padding)"""
        slice_idx = 1  # Near start
        context_size = 5
        
        context = self.preprocessor.get_contextual_slices(
            self.volume, slice_idx, context_size, pad_mode='edge'
        )
        
        assert context.shape == (5, 512, 512)
        
        # Should use edge padding for negative indices
        # slice_idx=1, context=5 -> slices [-1, 0, 1, 2, 3]
        # -1 and 0 -> volume[0] (edge padding)
        assert np.array_equal(context[0], self.volume[0])  # Padded
        assert np.array_equal(context[2], self.volume[1])  # Center
        
        print(f"✓ Start boundary context: shape {context.shape}")
    
    def test_get_contextual_slices_boundary_end(self):
        """Test context extraction at volume end (needs padding)"""
        slice_idx = 98  # Near end
        context_size = 5
        
        context = self.preprocessor.get_contextual_slices(
            self.volume, slice_idx, context_size, pad_mode='edge'
        )
        
        assert context.shape == (5, 512, 512)
        
        # slice_idx=98, context=5 -> slices [96, 97, 98, 99, 100]
        # 100 exceeds depth (100), should pad with volume[-1]
        assert np.array_equal(context[2], self.volume[98])  # Center
        assert np.array_equal(context[4], self.volume[99])  # Last valid
        
        print(f"✓ End boundary context: shape {context.shape}")
    
    def test_get_contextual_slices_different_sizes(self):
        """Test different context sizes"""
        slice_idx = 50
        
        for context_size in [3, 5, 7, 9]:
            context = self.preprocessor.get_contextual_slices(
                self.volume, slice_idx, context_size
            )
            assert context.shape == (context_size, 512, 512)
            # Verify center slice
            center_idx = context_size // 2
            assert np.array_equal(context[center_idx], self.volume[slice_idx])
        
        print(f"✓ Variable context sizes: [3, 5, 7, 9]")
    
    def test_stack_as_channels(self):
        """Test 2.5D channel stacking"""
        slice_idx = 50
        n_slices = 5
        
        stacked = self.preprocessor.stack_as_channels(
            self.volume, slice_idx, n_slices
        )
        
        # Should have same shape as get_contextual_slices
        assert stacked.shape == (5, 512, 512)
        
        # Can be used as channels: (C, H, W)
        # For 2D model: batch = stacked.unsqueeze(0) -> (1, 5, 512, 512)
        
        print(f"✓ Channel stacking for 2.5D: shape {stacked.shape}")


class TestTumorFocusedSampling:
    """Test tumor-focused patch extraction"""
    
    def setup_method(self):
        """Setup test data"""
        self.preprocessor = MedicalPreprocessor()
        # Create volume with synthetic tumor in center
        self.volume = np.zeros((100, 256, 256), dtype=np.float32)
        
        # Add synthetic tumor at center
        tumor_center = (50, 128, 128)
        tumor_radius = 15
        z, y, x = np.ogrid[:100, :256, :256]
        tumor_mask = (
            (z - tumor_center[0])**2 + 
            (y - tumor_center[1])**2 + 
            (x - tumor_center[2])**2 <= tumor_radius**2
        )
        self.volume[tumor_mask] = 1.0
        self.tumor_mask = tumor_mask.astype(np.float32)
    
    def test_extract_tumor_focused_patches_with_mask(self):
        """Test tumor-focused sampling with tumor mask"""
        patch_size = (32, 32, 32)
        n_patches = 4
        
        patches = self.preprocessor.extract_tumor_focused_patches(
            self.volume,
            patch_size=patch_size,
            tumor_mask=self.tumor_mask,
            n_patches=n_patches
        )
        
        assert len(patches) == n_patches
        
        # All patches should have correct size
        for patch in patches:
            assert patch.shape == patch_size
        
        # Patches should contain tumor voxels (value 1.0)
        tumor_counts = [np.sum(patch > 0.5) for patch in patches]
        avg_tumor_voxels = np.mean(tumor_counts)
        
        assert avg_tumor_voxels > 0, "Patches should contain tumor voxels"
        
        print(f"✓ Tumor-focused patches: {n_patches} patches extracted")
        print(f"  Average tumor voxels per patch: {avg_tumor_voxels:.1f}")
    
    def test_extract_random_patches_without_mask(self):
        """Test random sampling without tumor mask"""
        patch_size = (32, 32, 32)
        n_patches = 4
        
        patches = self.preprocessor.extract_tumor_focused_patches(
            self.volume,
            patch_size=patch_size,
            tumor_mask=None,  # No mask -> random sampling
            n_patches=n_patches
        )
        
        assert len(patches) == n_patches
        
        for patch in patches:
            assert patch.shape == patch_size
        
        print(f"✓ Random patches (no mask): {n_patches} patches extracted")
    
    def test_padding_for_boundary_patches(self):
        """Test padding when patch extends beyond volume"""
        # Small volume to force padding
        small_volume = np.random.randn(10, 20, 20).astype(np.float32)
        patch_size = (16, 16, 16)  # Larger than volume dimensions
        
        patches = self.preprocessor.extract_tumor_focused_patches(
            small_volume,
            patch_size=patch_size,
            tumor_mask=None,
            n_patches=2
        )
        
        for patch in patches:
            assert patch.shape == patch_size, "Should be padded to correct size"
        
        print(f"✓ Boundary padding: patches padded to {patch_size}")


class Test3DContextIntegration:
    """Integration tests for 3D context in training pipeline"""
    
    def test_context_for_dataset_integration(self):
        """Test how context slicing integrates with dataset"""
        preprocessor = MedicalPreprocessor()
        
        # Simulate CT volume from dataset
        volume = np.random.randn(150, 512, 512).astype(np.float32)
        
        # Simulate extracting training samples
        training_samples = []
        for slice_idx in [30, 60, 90, 120]:
            # Extract 5-slice context
            context = preprocessor.get_contextual_slices(
                volume, slice_idx, context_size=5
            )
            training_samples.append(context)
        
        # All samples should have consistent shape
        for sample in training_samples:
            assert sample.shape == (5, 512, 512)
        
        print(f"✓ Dataset integration: {len(training_samples)} samples extracted")
        print(f"  Sample shape: {training_samples[0].shape}")


def run_all_tests():
    """Run all preprocessing tests"""
    print("=" * 70)
    print("Testing 3D Context Preprocessing")
    print("=" * 70)
    
    # Context Slicing Tests
    print("\n1. 3D Context Slicing Tests")
    print("-" * 70)
    test_context = Test3DContextSlicing()
    test_context.setup_method()
    test_context.test_get_contextual_slices_middle()
    test_context.test_get_contextual_slices_boundary_start()
    test_context.test_get_contextual_slices_boundary_end()
    test_context.test_get_contextual_slices_different_sizes()
    test_context.test_stack_as_channels()
    
    # Tumor-Focused Sampling Tests
    print("\n2. Tumor-Focused Sampling Tests")
    print("-" * 70)
    test_tumor = TestTumorFocusedSampling()
    test_tumor.setup_method()
    test_tumor.test_extract_tumor_focused_patches_with_mask()
    test_tumor.test_extract_random_patches_without_mask()
    test_tumor.test_padding_for_boundary_patches()
    
    # Integration Tests
    print("\n3. Integration Tests")
    print("-" * 70)
    test_integration = Test3DContextIntegration()
    test_integration.test_context_for_dataset_integration()
    
    print("\n" + "=" * 70)
    print("✅ ALL PREPROCESSING TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
