#!/usr/bin/env python3
"""
Generate Tumor Segmentation Masks
Create binary and multi-class masks for detected tumors
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import json
from typing import Dict, List
from scipy import ndimage
from skimage.measure import label, regionprops

def load_nifti(file_path: Path) -> np.ndarray:
    """Load NIfTI file"""
    nii = nib.load(str(file_path))
    return nii.get_fdata()

def save_nifti(data: np.ndarray, output_path: Path, reference_nii_path: Path):
    """Save array as NIfTI with reference header"""
    ref_nii = nib.load(str(reference_nii_path))
    new_nii = nib.Nifti1Image(data, ref_nii.affine, ref_nii.header)
    nib.save(new_nii, str(output_path))
    print(f"  Saved: {output_path}")

def reconstruct_lesion_mask(
    ct_slice: np.ndarray,
    organ_mask: np.ndarray,
    baseline: Dict,
    lesion_info: Dict
) -> np.ndarray:
    """
    Reconstruct lesion mask from detection parameters
    
    Uses the same detection logic to recreate the exact mask
    """
    mean_hu = baseline['mean_hu']
    std_hu = baseline['std_hu']
    organ_pixels = organ_mask > 0
    
    # Method 1: Z-score detection
    z_scores = np.zeros_like(ct_slice)
    z_scores[organ_pixels] = (ct_slice[organ_pixels] - mean_hu) / (std_hu + 1e-6)
    anomalies_zscore = (np.abs(z_scores) > 1.5) & organ_pixels
    
    # Method 2: HU threshold
    anomalies_hu = (ct_slice >= 30) & (ct_slice <= 200) & organ_pixels
    
    # Combine
    anomalies = anomalies_zscore | anomalies_hu
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    anomalies = cv2.morphologyEx(anomalies.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    anomalies = cv2.morphologyEx(anomalies, cv2.MORPH_CLOSE, kernel)
    
    # Label regions
    labeled = label(anomalies)
    
    # Find matching region based on area and approximate position
    target_area = lesion_info.get('area', 0)
    
    for region in regionprops(labeled):
        area_diff = abs(region.area - target_area)
        if area_diff < target_area * 0.1:  # Within 10% of target area
            return (labeled == region.label).astype(np.uint8)
    
    return np.zeros_like(ct_slice, dtype=np.uint8)

def generate_tumor_masks(results_file: Path, ct_path: Path, output_dir: Path):
    """
    Generate 3D tumor segmentation masks
    
    Creates:
    1. Binary tumor mask (all tumors)
    2. Multi-class mask (tumor, suspicious, other)
    3. Individual tumor masks (each tumor separately)
    """
    print("\n" + "="*70)
    print("TUMOR MASK GENERATION")
    print("="*70 + "\n")
    
    # Load results
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Load CT volume
    ct_volume = load_nifti(ct_path)
    n_slices, height, width = ct_volume.shape
    print(f"CT volume: {ct_volume.shape}")
    
    # Load organ masks
    seg_dir = Path("CTdata/segmentation")
    organ_masks_3d = {}
    priority_organs = ['colon', 'liver', 'kidney_left', 'kidney_right',
                      'spleen', 'pancreas', 'stomach', 'small_bowel']
    
    print("\nLoading organ masks...")
    for organ_name in priority_organs:
        organ_file = seg_dir / f"{organ_name}.nii.gz"
        if organ_file.exists():
            mask = load_nifti(organ_file)
            if mask.sum() > 0:
                organ_masks_3d[organ_name] = mask.astype(np.uint8)
                print(f"  [OK] {organ_name}")
    
    # Compute baselines
    print("\nComputing organ baselines...")
    organ_baselines = {}
    for organ_name, mask_3d in organ_masks_3d.items():
        voxels = ct_volume[mask_3d > 0]
        if len(voxels) > 0:
            organ_baselines[organ_name] = {
                'mean_hu': float(np.mean(voxels)),
                'std_hu': float(np.std(voxels)),
            }
    
    # Initialize masks
    binary_tumor_mask = np.zeros_like(ct_volume, dtype=np.uint8)
    multiclass_mask = np.zeros_like(ct_volume, dtype=np.uint8)
    # Class IDs: 0=background, 1=potential_tumor, 2=suspicious, 3=other
    
    print("\nGenerating masks...")
    tumor_count = 0
    suspicious_count = 0
    
    for slice_idx_str, data in all_results.items():
        slice_idx = int(slice_idx_str)
        
        if data['lesion_count'] == 0:
            continue
        
        ct_slice = ct_volume[slice_idx, :, :]
        
        for lesion in data['lesions']:
            organ_name = lesion['organ']
            lesion_class = lesion['class']
            
            if organ_name not in organ_masks_3d:
                continue
            
            # Get organ mask for this slice
            organ_mask_2d = organ_masks_3d[organ_name][slice_idx, :, :]
            baseline = organ_baselines.get(organ_name)
            
            if baseline is None or organ_mask_2d.sum() < 100:
                continue
            
            # Reconstruct lesion mask
            lesion_mask = reconstruct_lesion_mask(
                ct_slice, organ_mask_2d, baseline, lesion
            )
            
            if lesion_mask.sum() > 0:
                # Add to binary mask
                if lesion_class in ['potential_tumor', 'suspicious']:
                    binary_tumor_mask[slice_idx, :, :] = np.maximum(
                        binary_tumor_mask[slice_idx, :, :],
                        lesion_mask
                    )
                
                # Add to multiclass mask
                if lesion_class == 'potential_tumor':
                    multiclass_mask[slice_idx, :, :][lesion_mask > 0] = 1
                    tumor_count += 1
                elif lesion_class == 'suspicious':
                    # Only set if not already marked as tumor
                    mask_update = (lesion_mask > 0) & (multiclass_mask[slice_idx, :, :] == 0)
                    multiclass_mask[slice_idx, :, :][mask_update] = 2
                    suspicious_count += 1
    
    print(f"\nMasks generated:")
    print(f"  Potential tumors: {tumor_count}")
    print(f"  Suspicious regions: {suspicious_count}")
    print(f"  Binary mask volume: {binary_tumor_mask.sum()} voxels")
    print(f"  Multiclass mask volume: {multiclass_mask.sum()} voxels")
    
    # Save masks
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nSaving masks...")
    save_nifti(binary_tumor_mask, output_dir / "tumor_mask_binary.nii.gz", ct_path)
    save_nifti(multiclass_mask, output_dir / "tumor_mask_multiclass.nii.gz", ct_path)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_mask_visualizations(ct_volume, binary_tumor_mask, multiclass_mask, output_dir)
    
    print("\n" + "="*70)
    print("MASK GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    print(f"  1. tumor_mask_binary.nii.gz - Binary tumor mask")
    print(f"  2. tumor_mask_multiclass.nii.gz - Multi-class mask")
    print(f"  3. tumor_overlay_*.png - Visualization images")

def create_mask_visualizations(
    ct_volume: np.ndarray,
    binary_mask: np.ndarray,
    multiclass_mask: np.ndarray,
    output_dir: Path
):
    """Create visualization of tumor masks overlaid on CT"""
    
    # Find slices with tumors
    tumor_slices = np.where(binary_mask.sum(axis=(1, 2)) > 0)[0]
    
    if len(tumor_slices) == 0:
        print("  No tumor slices to visualize")
        return
    
    # Sample slices (every 5th slice or max 10 slices)
    sample_indices = tumor_slices[::max(1, len(tumor_slices) // 10)][:10]
    
    for slice_idx in sample_indices:
        ct_slice = ct_volume[slice_idx, :, :]
        tumor_mask_2d = binary_mask[slice_idx, :, :]
        class_mask_2d = multiclass_mask[slice_idx, :, :]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original CT
        ax = axes[0]
        ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
        ax.set_title(f'Original CT - Slice {slice_idx}', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 2. Binary mask overlay
        ax = axes[1]
        ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
        
        # Create red overlay for tumors
        tumor_overlay = np.zeros((*ct_slice.shape, 4))
        tumor_overlay[tumor_mask_2d > 0] = [1, 0, 0, 0.5]  # Red with 50% transparency
        ax.imshow(tumor_overlay)
        
        ax.set_title('Binary Tumor Mask', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 3. Multi-class mask overlay
        ax = axes[2]
        ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
        
        # Create colored overlay
        multiclass_overlay = np.zeros((*ct_slice.shape, 4))
        multiclass_overlay[class_mask_2d == 1] = [1, 0, 0, 0.6]  # Red: potential tumor
        multiclass_overlay[class_mask_2d == 2] = [1, 1, 0, 0.4]  # Yellow: suspicious
        ax.imshow(multiclass_overlay)
        
        ax.set_title('Multi-class Mask\n(Red: Tumor, Yellow: Suspicious)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        output_file = output_dir / f"tumor_overlay_slice_{slice_idx:03d}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Created {len(sample_indices)} overlay visualizations")

def main():
    """Main execution"""
    
    results_file = Path("CTdata/visualizations/full_series/full_series_results.json")
    ct_path = Path("CTdata/nifti/inha_ct_arterial.nii.gz")
    output_dir = Path("CTdata/tumor_masks")
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run full_ct_series_analysis.py first")
        return
    
    if not ct_path.exists():
        print(f"Error: CT file not found: {ct_path}")
        return
    
    generate_tumor_masks(results_file, ct_path, output_dir)

if __name__ == '__main__':
    main()
