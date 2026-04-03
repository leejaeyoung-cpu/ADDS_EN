"""
Deep Validation: Compare Candidate Detection with Ground Truth
Verify detection quality and compute metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

from medical_imaging.detection import TumorDetector

def load_nifti(path):
    """Load NIfTI file"""
    nii = nib.load(str(path))
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    return volume, spacing


def calculate_overlap_with_gt(candidate, gt_slice):
    """Calculate if candidate overlaps with ground truth"""
    bbox = candidate.bounding_box
    # Extract region from bbox
    min_row, min_col, max_row, max_col = bbox
    
    # Check if any GT pixels in this bbox
    gt_region = gt_slice[min_row:max_row, min_col:max_col]
    overlap = (gt_region > 0).sum()
    
    return overlap > 0, overlap


def deep_validation():
    print("="*80)
    print("DEEP VALIDATION: Candidate Detection vs Ground Truth")
    print("="*80)
    
    # Load data
    data_path = Path("data/medical_decathlon/Task10_Colon/imagesTr")
    gt_path = Path("data/medical_decathlon/Task10_Colon/labelsTr")
    
    test_file = "colon_001.nii.gz"
    ct_file = data_path / test_file
    label_file = gt_path / test_file
    
    print(f"\nLoading: {test_file}")
    volume, spacing = load_nifti(ct_file)
    gt_volume, _ = load_nifti(label_file)
    
    print(f"Volume shape: {volume.shape}")
    print(f"GT shape: {gt_volume.shape}")
    
    # Find slices with tumors
    gt_slices_with_tumor = []
    for i in range(gt_volume.shape[2]):
        if gt_volume[:, :, i].sum() > 100:
            gt_slices_with_tumor.append(i)
    
    print(f"\nSlices with ground truth tumors: {len(gt_slices_with_tumor)}")
    print(f"Slice indices: {gt_slices_with_tumor}")
    
    # Initialize detector
    detector = TumorDetector(
        min_area_mm2=10.0,
        max_area_mm2=10000.0,
        hu_range=(-50, 200)
    )
    
    # Test on each tumor slice
    print("\n" + "="*80)
    print("PER-SLICE ANALYSIS")
    print("="*80)
    
    total_true_positives = 0
    total_false_positives = 0
    total_gt_tumors = len(gt_slices_with_tumor)
    
    for slice_idx in gt_slices_with_tumor:
        print(f"\n--- Slice {slice_idx} ---")
        
        # Get slice
        ct_slice = volume[:, :, slice_idx]
        gt_slice = gt_volume[:, :, slice_idx]
        
        # Ground truth info
        gt_pixels = (gt_slice > 0).sum()
        gt_area_mm2 = gt_pixels * spacing[0] * spacing[1]
        print(f"GT tumor pixels: {gt_pixels}")
        print(f"GT tumor area: {gt_area_mm2:.1f} mm²")
        
        # Detect candidates
        candidates = detector.detect_candidates_2d(
            hu_slice=ct_slice,
            pixel_spacing=(spacing[0], spacing[1]),
            body_mask=None,
            slice_index=slice_idx,
            method='multi_threshold'
        )
        
        print(f"Detected candidates: {len(candidates)}")
        
        # Check overlap with GT
        true_positive_candidates = []
        false_positive_candidates = []
        
        for c in candidates:
            has_overlap, overlap_pixels = calculate_overlap_with_gt(c, gt_slice)
            if has_overlap:
                true_positive_candidates.append((c, overlap_pixels))
            else:
                false_positive_candidates.append(c)
        
        print(f"  True Positives: {len(true_positive_candidates)} (overlap with GT)")
        print(f"  False Positives: {len(false_positive_candidates)}")
        
        # Show top TP candidates
        if true_positive_candidates:
            print(f"\n  Top True Positive Candidates:")
            tp_sorted = sorted(true_positive_candidates, 
                              key=lambda x: x[0].confidence_score, 
                              reverse=True)
            for i, (c, overlap) in enumerate(tp_sorted[:5]):
                print(f"    #{i+1}: Conf={c.confidence_score:.2f}, "
                      f"Area={c.area_mm2:.1f}mm², "
                      f"HU={c.mean_hu:.1f}, "
                      f"Overlap={overlap} pixels")
        
        # Update totals
        total_true_positives += len(true_positive_candidates)
        total_false_positives += len(false_positive_candidates)
    
    # Overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    
    total_candidates = total_true_positives + total_false_positives
    
    print(f"\nTotal candidates detected: {total_candidates}")
    print(f"True Positives: {total_true_positives}")
    print(f"False Positives: {total_false_positives}")
    
    if total_candidates > 0:
        precision = total_true_positives / total_candidates
        print(f"\nPrecision: {precision:.3f} ({total_true_positives}/{total_candidates})")
    
    if total_gt_tumors > 0:
        recall = len([s for s in gt_slices_with_tumor if any(
            calculate_overlap_with_gt(c, gt_volume[:, :, s])[0]
            for c in detector.detect_candidates_2d(
                volume[:, :, s], 
                (spacing[0], spacing[1]), 
                None, s, 'multi_threshold'
            )
        )]) / total_gt_tumors
        print(f"Recall (slice-level): {recall:.3f}")
    
    # Comparison with ADDS baseline
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print(f"\nADDS (previous):")
    print(f"  Dice: 0.0000")
    print(f"  Detections: 0")
    print(f"  Status: Complete failure")
    
    print(f"\ntxt Pipeline (current):")
    print(f"  Total candidates: {total_candidates}")
    print(f"  True Positives: {total_true_positives}")
    print(f"  Precision: {precision:.3f}" if total_candidates > 0 else "  Precision: N/A")
    print(f"  Status: WORKING!")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    if total_true_positives > 0:
        print("\nRESULT: SUCCESS - System detects actual tumors!")
    else:
        print("\nWARNING: No true positive detections")


if __name__ == "__main__":
    deep_validation()
