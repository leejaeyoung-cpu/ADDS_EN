"""
데이터 검증 스크립트 - 1채널 Transfer Learning
Low Dice Score (0.0069) 원인 분석
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.medical_imaging.data.dataset_1ch import ColonCancerDataset1Channel

def verify_dataset():
    print("="*60)
    print("1-Channel Dataset Verification")
    print("="*60)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    dataset = ColonCancerDataset1Channel(
        data_root='data/medical_decathlon/Task10_Colon',
        fold=0,
        mode='train',
        tumor_focused_sampling=False  # Disable for verification
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check multiple samples
    print("\n[2] Checking samples...")
    
    tumor_samples = 0
    no_tumor_samples = 0
    
    for idx in range(min(10, len(dataset))):
        image, label = dataset[idx]
        
        # Check shapes
        assert image.shape[0] == 1, f"Expected 1 channel, got {image.shape[0]}"
        assert len(label.shape) == 3, f"Expected 3D label, got {label.shape}"
        
        # Check label content
        unique_labels = torch.unique(label)
        has_tumor = (unique_labels > 0).any().item()
        
        if has_tumor:
            tumor_samples += 1
            tumor_ratio = (label > 0).sum().float() / label.numel()
            print(f"Sample {idx}: HAS TUMOR - {tumor_ratio.item()*100:.2f}% tumor voxels")
            print(f"  Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  Label shape: {label.shape}, classes: {unique_labels.tolist()}")
        else:
            no_tumor_samples += 1
    
    print(f"\n[3] Summary of first 10 samples:")
    print(f"  Tumor samples: {tumor_samples}")
    print(f"  No tumor samples: {no_tumor_samples}")
    
    if tumor_samples == 0:
        print("\n*** WARNING: No tumor found in first 10 samples! ***")
        print("This could explain the low Dice score.")
    
    # Detailed check of one tumor sample
    print("\n[4] Detailed check of sample with tumor...")
    
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        if (label > 0).any():
            print(f"\nAnalyzing tumor sample {idx}:")
            
            # Image statistics
            print(f"Image stats:")
            print(f"  Mean: {image.mean():.4f}")
            print(f"  Std: {image.std():.4f}")
            print(f"  Min: {image.min():.4f}")
            print(f"  Max: {image.max():.4f}")
            
            # Label statistics
            tumor_voxels = (label > 0).sum()
            total_voxels = label.numel()
            tumor_ratio = tumor_voxels.float() / total_voxels
            
            print(f"\nLabel stats:")
            print(f"  Tumor voxels: {tumor_voxels}")
            print(f"  Total voxels: {total_voxels}")
            print(f"  Tumor ratio: {tumor_ratio.item()*100:.4f}%")
            print(f"  Unique classes: {torch.unique(label).tolist()}")
            
            # Check if tumor is visible in image
            tumor_mask = (label > 0).numpy()
            image_np = image[0].numpy()  # Remove channel dim
            
            tumor_region_intensity = image_np[tumor_mask]
            background_intensity = image_np[~tumor_mask]
            
            print(f"\nIntensity comparison:")
            print(f"  Tumor region mean: {tumor_region_intensity.mean():.4f}")
            print(f"  Background mean: {background_intensity.mean():.4f}")
            print(f"  Contrast: {abs(tumor_region_intensity.mean() - background_intensity.mean()):.4f}")
            
            if abs(tumor_region_intensity.mean() - background_intensity.mean()) < 0.1:
                print("\n*** WARNING: Very low contrast between tumor and background! ***")
                print("Windowing might not be optimal for tumor detection.")
            
            break
    
    print("\n" + "="*60)
    print("Verification Complete")
    print("="*60)


if __name__ == "__main__":
    verify_dataset()
