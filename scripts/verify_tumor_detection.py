#!/usr/bin/env python3
"""
Verify Tumor Detection - Analyze HU values and visual appearance
Check if detected regions are actually brighter than surroundings
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_nifti(file_path: Path) -> np.ndarray:
    """Load NIfTI file"""
    nii = nib.load(str(file_path))
    return nii.get_fdata()

def analyze_slice_hu_distribution(ct_slice, mask, organ_mask):
    """Analyze HU distribution in detected regions vs normal tissue"""
    
    # HU values in detected tumor regions
    tumor_hu = ct_slice[mask > 0]
    
    # HU values in normal organ tissue (excluding tumor)
    normal_hu = ct_slice[(organ_mask > 0) & (mask == 0)]
    
    # Background (outside organ)
    background_hu = ct_slice[(organ_mask == 0) & (mask == 0)]
    
    return {
        'tumor_hu': tumor_hu,
        'normal_hu': normal_hu,
        'background_hu': background_hu,
        'tumor_mean': np.mean(tumor_hu) if len(tumor_hu) > 0 else 0,
        'normal_mean': np.mean(normal_hu) if len(normal_hu) > 0 else 0,
        'tumor_std': np.std(tumor_hu) if len(tumor_hu) > 0 else 0,
        'normal_std': np.std(normal_hu) if len(normal_hu) > 0 else 0,
    }

def create_detailed_analysis(slice_idx=66):
    """Create detailed analysis of a specific slice"""
    
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS - Slice {slice_idx}")
    print(f"{'='*70}\n")
    
    # Load data
    ct_path = Path("CTdata/nifti/inha_ct_arterial.nii.gz")
    binary_mask_path = Path("CTdata/tumor_masks/tumor_mask_binary.nii.gz")
    results_file = Path("CTdata/visualizations/full_series/full_series_results.json")
    
    ct_volume = load_nifti(ct_path)
    binary_mask = load_nifti(binary_mask_path)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    ct_slice = ct_volume[slice_idx, :, :]
    mask_slice = binary_mask[slice_idx, :, :]
    
    # Load organ mask
    seg_dir = Path("CTdata/segmentation")
    colon_mask = load_nifti(seg_dir / "colon.nii.gz")[slice_idx, :, :]
    small_bowel_mask = load_nifti(seg_dir / "small_bowel.nii.gz")[slice_idx, :, :]
    combined_organ = (colon_mask > 0) | (small_bowel_mask > 0)
    
    # Analyze HU distribution
    stats = analyze_slice_hu_distribution(ct_slice, mask_slice, combined_organ)
    
    print(f"[HU STATISTICS]")
    print(f"  Detected 'tumor' regions:")
    print(f"    Mean HU: {stats['tumor_mean']:.1f} ± {stats['tumor_std']:.1f}")
    print(f"    Range: {np.min(stats['tumor_hu']):.1f} to {np.max(stats['tumor_hu']):.1f}")
    print(f"    Pixels: {len(stats['tumor_hu'])}")
    print(f"\n  Normal organ tissue:")
    print(f"    Mean HU: {stats['normal_mean']:.1f} ± {stats['normal_std']:.1f}")
    print(f"    Range: {np.min(stats['normal_hu']):.1f} to {np.max(stats['normal_hu']):.1f}")
    print(f"    Pixels: {len(stats['normal_hu'])}")
    print(f"\n  Difference: {stats['tumor_mean'] - stats['normal_mean']:.1f} HU")
    
    # Check if tumor is actually brighter
    is_brighter = stats['tumor_mean'] > stats['normal_mean']
    print(f"\n  Tumor brighter than normal? {'YES' if is_brighter else 'NO ⚠️'}")
    
    # Create visualization with multiple window settings
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Different window/level settings
    windows = [
        ("Soft Tissue", 40, 400),
        ("Lung", -600, 1500),
        ("Bone", 400, 1800),
        ("Wide", 50, 2000),
    ]
    
    for col, (name, center, width) in enumerate(windows):
        vmin = center - width / 2
        vmax = center + width / 2
        
        # Top row: Original CT
        ax = axes[0, col]
        ax.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{name} Window\nC={center}, W={width}', fontsize=10)
        ax.axis('off')
        
        # Bottom row: With tumor overlay
        ax = axes[1, col]
        ax.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax)
        overlay = np.zeros((*ct_slice.shape, 4))
        overlay[mask_slice > 0] = [1, 0, 0, 0.5]
        ax.imshow(overlay)
        ax.set_title(f'With Tumor Overlay', fontsize=10)
        ax.axis('off')
    
    plt.suptitle(
        f'Slice {slice_idx} - Multiple Window Settings\n'
        f'Tumor Mean: {stats["tumor_mean"]:.1f} HU vs Normal: {stats["normal_mean"]:.1f} HU',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    output_file = Path("CTdata/tumor_masks/verification_slice_66_windows.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] {output_file}")
    plt.close()
    
    # Create HU histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(stats['tumor_hu'], bins=50, alpha=0.6, color='red', label='Detected Tumor', density=True)
    ax.hist(stats['normal_hu'], bins=50, alpha=0.6, color='blue', label='Normal Tissue', density=True)
    
    ax.axvline(stats['tumor_mean'], color='red', linestyle='--', linewidth=2, 
               label=f'Tumor Mean: {stats["tumor_mean"]:.1f}')
    ax.axvline(stats['normal_mean'], color='blue', linestyle='--', linewidth=2,
               label=f'Normal Mean: {stats["normal_mean"]:.1f}')
    
    ax.set_xlabel('HU Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Slice {slice_idx} - HU Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    output_file = Path("CTdata/tumor_masks/verification_slice_66_histogram.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {output_file}")
    plt.close()
    
    # Print lesion details from JSON
    print(f"\n[LESION DETAILS FROM JSON]")
    slice_data = results[str(slice_idx)]
    for i, lesion in enumerate(slice_data['lesions'], 1):
        if lesion['class'] in ['potential_tumor', 'suspicious']:
            print(f"\n  Lesion {i}:")
            print(f"    Class: {lesion['class']}")
            print(f"    HU: {lesion['mean_hu']:.1f}")
            print(f"    Size: {lesion['area']:.0f} pixels")
            print(f"    Organ: {lesion['organ']}")
            print(f"    Z-score: {lesion['z_score']:.2f}")
    
    return stats

def main():
    """Main analysis"""
    stats = create_detailed_analysis(slice_idx=66)
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}\n")
    
    diff = stats['tumor_mean'] - stats['normal_mean']
    
    if diff > 10:
        print("[OK] Detected regions ARE significantly brighter than normal tissue")
        print(f"   Difference: +{diff:.1f} HU")
    elif diff > 0:
        print("[WARNING] Detected regions are slightly brighter, but difference is small")
        print(f"   Difference: +{diff:.1f} HU")
        print("   -> May be normal anatomical variation")
    else:
        print("[ERROR] Detected regions are NOT brighter than normal tissue")
        print(f"   Difference: {diff:.1f} HU")
        print("   -> This is a FALSE POSITIVE")
        print("\n   LIKELY CAUSE:")
        print("   - Detection threshold is too low (currently 30 HU)")
        print("   - Z-score threshold is too relaxed (currently 1.5)")
        print("   - Need to increase thresholds to reduce false positives")

if __name__ == '__main__':
    main()
