#!/usr/bin/env python3
"""
AGGRESSIVE Tumor Detection - Lowered Thresholds
For confirmed cancer case - very sensitive detection
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from scipy import ndimage
from skimage.measure import label, regionprops
import json

def load_nifti(file_path: Path) -> np.ndarray:
    nii = nib.load(str(file_path))
    return nii.get_fdata()

def aggressive_lesion_detection(ct_slice: np.ndarray, organ_mask: np.ndarray, baseline: dict, organ_name: str):
    """
    VERY SENSITIVE detection for confirmed cancer case
    
    Lowered thresholds:
    - Z-score: 3.0 → 1.5 (much more sensitive)
    - Min size: 100 → 20 pixels
    - HU cutoff: -500 → 0 (include all soft tissue)
    """
    if organ_mask.sum() == 0 or baseline is None:
        return []
    
    mean_hu = baseline['mean_hu']
    std_hu = baseline['std_hu']
    
    # Calculate z-scores
    z_scores = np.zeros_like(ct_slice)
    organ_pixels = organ_mask > 0
    z_scores[organ_pixels] = (ct_slice[organ_pixels] - mean_hu) / (std_hu + 1e-6)
    
    # AGGRESSIVE thresholds
    z_threshold = 1.5  # Lowered from 3.0
    
    # Detect ANY deviation
    anomalies = (np.abs(z_scores) > z_threshold) & organ_pixels
    
    # Also detect bright regions (potential tumors)
    bright_regions = (ct_slice > 40) & organ_pixels  # Soft tissue density
    
    # Combine
    all_candidates = anomalies | bright_regions
    
    # Minimal cleanup
    kernel = np.ones((2, 2), np.uint8)
    all_candidates = cv2.morphologyEx(all_candidates.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Label
    labeled = label(all_candidates)
    
    lesions = []
    for region in regionprops(labeled):
        # VERY permissive size filter
        if region.area < 20:  # Lowered from 100
            continue
        
        lesion_mask = (labeled == region.label).astype(np.uint8)
        lesion_pixels = ct_slice[lesion_mask > 0]
        
        mean_hu_lesion = float(np.mean(lesion_pixels))
        
        # Filter out ONLY extreme gas
        if mean_hu_lesion < 0:  # Changed from -500
            continue
        
        lesions.append({
            'area': float(region.area),
            'mean_hu': mean_hu_lesion,
            'z_score': (mean_hu_lesion - mean_hu) / (std_hu + 1e-6),
            'centroid': region.centroid,
            'bbox': region.bbox,
            'organ': organ_name
        })
    
    return lesions

def create_detailed_visualization(ct_slice, organ_mask, lesions, slice_idx, output_dir):
    """Create detailed visualization with ALL candidates"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. Original CT
    ax = axes[0, 0]
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    ax.set_title(f'Original CT - Slice {slice_idx}', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 2. Organ mask
    ax = axes[0, 1]
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    ax.imshow(organ_mask, alpha=0.3, cmap='Greens')
    ax.set_title('Organ Boundary', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 3. All candidates
    ax = axes[1, 0]
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    for i, lesion in enumerate(lesions):
        cy, cx = lesion['centroid']
        hu = lesion['mean_hu']
        
        # Color by HU
        if hu > 60:
            color = 'red'  # High HU - likely tumor
            marker = 'o'
            size = 200
        elif hu > 40:
            color = 'orange'  # Moderate HU - possible tumor
            marker = 's'
            size = 150
        else:
            color = 'yellow'  # Low HU - uncertain
            marker = '^'
            size = 100
        
        ax.scatter(cx, cy, c=color, marker=marker, s=size, edgecolors='white', linewidths=2)
        ax.text(cx + 10, cy, f"{i+1}\nHU:{hu:.0f}", color='white', fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_title(f'Detected Candidates ({len(lesions)} found)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"AGGRESSIVE DETECTION RESULTS\n"
    summary_text += "="*40 + "\n\n"
    summary_text += f"Slice: {slice_idx}\n"
    summary_text += f"Candidates: {len(lesions)}\n\n"
    
    summary_text += "TOP CANDIDATES:\n"
    summary_text += "-"*40 + "\n"
    
    # Sort by HU (highest first)
    sorted_lesions = sorted(lesions, key=lambda x: -x['mean_hu'])
    
    for i, lesion in enumerate(sorted_lesions[:10], 1):
        summary_text += f"\n{i}. HU: {lesion['mean_hu']:.1f}"
        summary_text += f" | Size: {lesion['area']:.0f}px"
        summary_text += f" | Z: {lesion['z_score']:.2f}"
        
        # Flag suspicious ones
        if lesion['mean_hu'] > 60 and lesion['area'] > 50:
            summary_text += " ⚠️ HIGH PRIORITY"
        elif lesion['mean_hu'] > 40 and lesion['area'] > 100:
            summary_text += " ⚠️ REVIEW"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    output_file = output_dir / f'aggressive_detection_slice_{slice_idx:03d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    """Aggressive re-detection for confirmed cancer case"""
    
    print("\n" + "="*70)
    print("AGGRESSIVE TUMOR DETECTION - CONFIRMED CANCER CASE")
    print("="*70 + "\n")
    print("Using VERY SENSITIVE thresholds:")
    print("  - Z-score: 1.5 (was 3.0)")
    print("  - Min size: 20px (was 100px)")
    print("  - HU cutoff: 0 (was -500)")
    print()
    
    # Load CT
    ct_path = Path("CTdata/nifti/inha_ct_arterial.nii.gz")
    ct_volume = load_nifti(ct_path)
    n_slices = ct_volume.shape[0]
    
    # Load colon mask
    colon_file = Path("CTdata/segmentation/colon.nii.gz")
    colon_mask_3d = load_nifti(colon_file).astype(np.uint8)
    
    # Compute baseline
    colon_voxels = ct_volume[colon_mask_3d > 0]
    baseline = {
        'mean_hu': float(np.mean(colon_voxels)),
        'std_hu': float(np.std(colon_voxels))
    }
    
    print(f"Colon baseline: {baseline['mean_hu']:.1f} ± {baseline['std_hu']:.1f} HU\n")
    
    # Process all slices
    print(f"Processing {n_slices} slices...\n")
    output_dir = Path("CTdata/visualizations/aggressive_detection")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_candidates = {}
    high_priority = []
    
    for slice_idx in range(n_slices):
        ct_slice = ct_volume[slice_idx, :, :]
        colon_mask_2d = colon_mask_3d[slice_idx, :, :]
        
        if colon_mask_2d.sum() < 100:
            continue
        
        lesions = aggressive_lesion_detection(ct_slice, colon_mask_2d, baseline, 'colon')
        
        if lesions:
            all_candidates[slice_idx] = lesions
            
            # Identify high priority
            for lesion in lesions:
                if lesion['mean_hu'] > 40 and lesion['area'] > 50:
                    high_priority.append({
                        'slice': slice_idx,
                        'lesion': lesion
                    })
            
            print(f"  Slice {slice_idx}: {len(lesions)} candidates")
            
            # Visualize if has candidates
            create_detailed_visualization(ct_slice, colon_mask_2d, lesions, slice_idx, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    
    total_candidates = sum(len(v) for v in all_candidates.values())
    print(f"\nTotal candidates: {total_candidates}")
    print(f"Slices with candidates: {len(all_candidates)}")
    print(f"HIGH PRIORITY candidates: {len(high_priority)}")
    
    if high_priority:
        print("\n" + "="*70)
        print("HIGH PRIORITY CANDIDATES (HU > 40, Size > 50px)")
        print("="*70)
        
        # Sort by HU
        high_priority.sort(key=lambda x: -x['lesion']['mean_hu'])
        
        for i, item in enumerate(high_priority[:20], 1):
            lesion = item['lesion']
            print(f"\n{i}. Slice {item['slice']}")
            print(f"   HU: {lesion['mean_hu']:.1f}")
            print(f"   Size: {lesion['area']:.0f} pixels (~{lesion['area'] * 0.01:.1f} cm²)")
            print(f"   Z-score: {lesion['z_score']:.2f}")
            print(f"   Location: {lesion['centroid']}")
    
    # Save results
    results_file = output_dir / "aggressive_detection_results.json"
    
    json_results = {}
    for slice_idx, lesions in all_candidates.items():
        json_results[str(slice_idx)] = [
            {
                'mean_hu': l['mean_hu'],
                'area': l['area'],
                'z_score': l['z_score'],
                'centroid': [float(c) for c in l['centroid']]
            }
            for l in lesions
        ]
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_dir}")
    print(f"JSON data: {results_file}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review high-priority candidates manually")
    print("2. Get tumor location from 교수님")
    print("3. Compare system findings with ground truth")
    print("4. Adjust thresholds if needed")

if __name__ == '__main__':
    main()
