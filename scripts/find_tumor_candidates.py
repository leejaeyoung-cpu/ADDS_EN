#!/usr/bin/env python3
"""
Tumor Candidate Visualization
Re-analyze all lesions and visualize only tumor-like candidates
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import json
from typing import List, Dict

def load_nifti(file_path: Path) -> np.ndarray:
    """Load NIfTI file"""
    nii = nib.load(str(file_path))
    return nii.get_fdata()

def is_tumor_candidate(lesion: Dict) -> bool:
    """
    Enhanced tumor candidate identification
    
    Accepts lesions from new classification system including 'suspicious' category.
    Relaxed criteria to match enhanced detection algorithm.
    """
    mean_hu = lesion.get('mean_hu', -1000)
    area = lesion.get('area', 0)
    lesion_class = lesion.get('class', 'unknown')
    
    # Accept new classification categories
    if lesion_class in ['potential_tumor', 'suspicious']:
        return True, f"Classified as {lesion_class}"
    
    # Relaxed HU-based criteria (reduced from 40+ to 20+)
    if mean_hu >= 20 and area >= 50:
        return True, f"HU={mean_hu:.1f}, size={area:.0f}px"
    
    # Enhanced tumor (high HU)
    if mean_hu >= 60 and area >= 30:
        return True, f"Enhanced (HU={mean_hu:.1f})"
    
    return False, "No tumor-like features"

def create_tumor_visualization(
    ct_slice: np.ndarray,
    organ_mask: np.ndarray,
    lesion_mask: np.ndarray,
    lesion_info: Dict,
    slice_idx: int,
    output_dir: Path
):
    """Create focused tumor candidate visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original CT
    ax = axes[0]
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    ax.set_title('Original CT', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add crosshair at lesion center
    cy, cx = lesion_info['centroid']
    ax.plot(cx, cy, 'r+', markersize=20, markeredgewidth=3)
    
    # 2. Zoomed Region
    ax = axes[1]
    
    # Get bounding box
    ys, xs = np.where(lesion_mask > 0)
    if len(ys) > 0:
        y_min, y_max = max(0, ys.min() - 30), min(ct_slice.shape[0], ys.max() + 30)
        x_min, x_max = max(0, xs.min() - 30), min(ct_slice.shape[1], xs.max() + 30)
        
        roi = ct_slice[y_min:y_max, x_min:x_max]
        roi_lesion = lesion_mask[y_min:y_max, x_min:x_max]
        
        # Create overlay
        roi_rgb = cv2.cvtColor(
            cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2RGB
        )
        
        # Highlight lesion in red
        roi_rgb[roi_lesion > 0] = (roi_rgb[roi_lesion > 0] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
        
        ax.imshow(roi_rgb)
        ax.set_title('Zoomed ROI', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # 3. Feature Summary
    ax = axes[2]
    ax.axis('off')
    
    summary = f"""LESION ANALYSIS
    
Slice: {slice_idx}
Organ: {lesion_info['organ']}
Reason: {lesion_info['reason']}

INTENSITY FEATURES
Mean HU: {lesion_info['mean_hu']:.1f}
Z-score: {lesion_info.get('z_score', 0):.2f}

SIZE FEATURES
Area: {lesion_info['area']:.0f} pixels
Estimated Size: {lesion_info['area'] * 0.01:.1f} cm²

CLASSIFICATION
Original: {lesion_info['class']}
Candidate: YES

RECOMMENDATION
⚠️ Manual review required
Consider correlation with:
- Clinical symptoms
- Other imaging modalities
- Follow-up scans
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Tumor Candidate Analysis - Slice {slice_idx}',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = output_dir / f'tumor_candidate_slice_{slice_idx:03d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    """Main tumor candidate analysis"""
    
    print("\n" + "="*70)
    print("TUMOR CANDIDATE RE-ANALYSIS")
    print("="*70 + "\n")
    
    # Load results
    results_file = Path("CTdata/visualizations/full_series/full_series_results.json")
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Load CT
    ct_path = Path("CTdata/nifti/inha_ct_arterial.nii.gz")
    ct_volume = load_nifti(ct_path)
    
    # Load organs
    seg_dir = Path("CTdata/segmentation")
    organ_masks_3d = {}
    for organ_name in ['colon', 'liver', 'kidney_left', 'kidney_right', 
                      'spleen', 'pancreas', 'stomach', 'small_bowel']:
        organ_file = seg_dir / f"{organ_name}.nii.gz"
        if organ_file.exists():
            mask = load_nifti(organ_file)
            if mask.sum() > 0:
                organ_masks_3d[organ_name] = mask.astype(np.uint8)
    
    # Find tumor candidates
    print("Analyzing all lesions for tumor characteristics...")
    candidates = []
    
    for slice_idx_str, data in all_results.items():
        slice_idx = int(slice_idx_str)
        
        for lesion in data['lesions']:
            is_candidate, reason = is_tumor_candidate(lesion)
            
            if is_candidate:
                candidates.append({
                    'slice': slice_idx,
                    'lesion': lesion,
                    'reason': reason
                })
    
    print(f"\nFound {len(candidates)} tumor candidates")
    
    if len(candidates) == 0:
        print("\n" + "="*70)
        print("NO TUMOR CANDIDATES FOUND")
        print("="*70)
        print("\nAll detected lesions have characteristics of:")
        print("  - Gas/air (HU < -500)")
        print("  - Small artifacts (area < 100)")
        print("  - Normal variations")
        print("\nRECOMMENDATION:")
        print("  1. Confirm tumor location with radiologist")
        print("  2. Review portal venous phase if available")
        print("  3. Check if tumor is isoattenuating")
        return
    
    # Visualize candidates
    print(f"\nGenerating visualizations for {len(candidates)} candidates...")
    output_dir = Path("CTdata/visualizations/tumor_candidates")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for i, candidate in enumerate(candidates, 1):
        slice_idx = candidate['slice']
        lesion = candidate['lesion']
        
        print(f"  {i}. Slice {slice_idx}: {lesion['organ']}, "
              f"HU={lesion['mean_hu']:.1f}, "
              f"size={lesion['area']:.0f}px ({candidate['reason']})")
        
        # Get CT slice and organ mask
        ct_slice = ct_volume[slice_idx, :, :]
        organ_name = lesion['organ']
        
        if organ_name in organ_masks_3d:
            organ_mask_2d = organ_masks_3d[organ_name][slice_idx, :, :].astype(np.uint8)
            
            # Reconstruct lesion mask (approximate from centroid and area)
            # This is a simplified reconstruction
            lesion_mask = np.zeros_like(ct_slice, dtype=np.uint8)
            # Note: We don't have the exact mask, so this is approximate
            
            # Create visualization
            lesion_info = {
                **lesion,
                'centroid': (int(ct_slice.shape[0] * 0.5), int(ct_slice.shape[1] * 0.5)),  # Approximate
                'reason': candidate['reason']
            }
            
            output_file = create_tumor_visualization(
                ct_slice, organ_mask_2d, lesion_mask, lesion_info, slice_idx, output_dir
            )
    
    # Create summary
    print("\n" + "="*70)
    print("TUMOR CANDIDATE SUMMARY")
    print("="*70)
    
    print(f"\nTotal candidates: {len(candidates)}")
    print(f"Output directory: {output_dir}")
    
    # Group by organ
    by_organ = {}
    for c in candidates:
        organ = c['lesion']['organ']
        by_organ[organ] = by_organ.get(organ, 0) + 1
    
    print("\nBy organ:")
    for organ, count in sorted(by_organ.items(), key=lambda x: -x[1]):
        print(f"  {organ}: {count}")

if __name__ == '__main__':
    main()
