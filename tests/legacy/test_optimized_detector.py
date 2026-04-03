"""
Test Optimized Detector
Compare baseline vs optimized performance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd

from medical_imaging.detection import (
    TumorDetector,
    OptimizedColonDetector,
    create_optimized_detector
)


def load_nifti(path):
    """Load NIfTI file"""
    nii = nib.load(str(path))
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    return volume, spacing


def calculate_metrics(candidates, gt_slice):
    """Calculate TP/FP"""
    tp = 0
    fp = 0
    
    for c in candidates:
        bbox = c.bounding_box
        min_row, min_col, max_row, max_col = bbox
        gt_region = gt_slice[min_row:max_row, min_col:max_col]
        
        if (gt_region > 0).sum() > 0:
            tp += 1
        else:
            fp += 1
    
    return tp, fp


def test_detectors():
    print("="*80)
    print("OPTIMIZED DETECTOR TEST")
    print("Baseline vs Optimized Comparison")
    print("="*80)
    
    # Load data
    data_dir = Path("data/medical_decathlon/Task10_Colon")
    ct_path = data_dir / "imagesTr" / "colon_001.nii.gz"
    gt_path = data_dir / "labelsTr" / "colon_001.nii.gz"
    
    volume, spacing = load_nifti(ct_path)
    gt_volume, _ = load_nifti(gt_path)
    
    print(f"\nTest case: colon_001")
    print(f"Volume shape: {volume.shape}")
    
    # Find tumor slices
    tumor_slices = [i for i in range(gt_volume.shape[2]) 
                    if gt_volume[:, :, i].sum() > 100]
    print(f"Tumor slices: {tumor_slices}")
    
    # Create detectors
    baseline = TumorDetector(
        min_area_mm2=10.0,
        max_area_mm2=10000.0,
        hu_range=(-50, 200)
    )
    
    optimized = create_optimized_detector(
        fp_filtering=True,
        colon_scoring=True
    )
    
    print("\n" + "="*80)
    print("DETECTOR CONFIGURATIONS")
    print("="*80)
    
    print("\nBaseline:")
    print(f"  min_area: {baseline.min_area_mm2} mm²")
    print(f"  max_area: {baseline.max_area_mm2} mm²")
    print(f"  hu_range: {baseline.hu_range}")
    print(f"  FP filtering: No")
    print(f"  Colon scoring: No")
    
    print("\nOptimized:")
    print(f"  min_area: {optimized.min_area_mm2} mm²")
    print(f"  max_area: {optimized.max_area_mm2} mm²")
    print(f"  hu_range: {optimized.hu_range}")
    print(f"  FP filtering: Yes")
    print(f"  Colon scoring: Yes")
    
    # Test on each tumor slice
    print("\n" + "="*80)
    print("PER-SLICE RESULTS")
    print("="*80)
    
    results = []
    
    for z in tumor_slices:
        ct_slice = volume[:, :, z]
        gt_slice = gt_volume[:, :, z]
        pixel_spacing = (spacing[0], spacing[1])
        
        # Baseline detection
        cands_baseline = baseline.detect_candidates_2d(
            ct_slice, pixel_spacing, None, z, 'multi_threshold'
        )
        tp_b, fp_b = calculate_metrics(cands_baseline, gt_slice)
        
        # Optimized detection
        cands_optimized = optimized.detect_candidates_2d(
            ct_slice, pixel_spacing, None, z, 'multi_threshold'
        )
        tp_o, fp_o = calculate_metrics(cands_optimized, gt_slice)
        
        # Confidence stats
        conf_b = [c.confidence_score for c in cands_baseline]
        conf_o = [c.confidence_score for c in cands_optimized]
        
        max_conf_b = max(conf_b) if conf_b else 0
        max_conf_o = max(conf_o) if conf_o else 0
        avg_conf_b = np.mean(conf_b) if conf_b else 0
        avg_conf_o = np.mean(conf_o) if conf_o else 0
        
        print(f"\nSlice {z}:")
        print(f"  Baseline:  {len(cands_baseline):2d} candidates "
              f"(TP={tp_b}, FP={fp_b:2d}) | "
              f"Conf: max={max_conf_b:.2f}, avg={avg_conf_b:.2f}")
        print(f"  Optimized: {len(cands_optimized):2d} candidates "
              f"(TP={tp_o}, FP={fp_o:2d}) | "
              f"Conf: max={max_conf_o:.2f}, avg={avg_conf_o:.2f}")
        
        if len(cands_optimized) < len(cands_baseline):
            reduction = len(cands_baseline) - len(cands_optimized)
            print(f"  Reduction: {reduction} candidates ({reduction/len(cands_baseline)*100:.1f}%)")
        
        results.append({
            'slice': z,
            'baseline_total': len(cands_baseline),
            'baseline_tp': tp_b,
            'baseline_fp': fp_b,
            'baseline_max_conf': max_conf_b,
            'baseline_avg_conf': avg_conf_b,
            'optimized_total': len(cands_optimized),
            'optimized_tp': tp_o,
            'optimized_fp': fp_o,
            'optimized_max_conf': max_conf_o,
            'optimized_avg_conf': avg_conf_o
        })
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL COMPARISON")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    print("\nBaseline:")
    print(f"  Total candidates: {df['baseline_total'].sum()}")
    print(f"  True Positives: {df['baseline_tp'].sum()}")
    print(f"  False Positives: {df['baseline_fp'].sum()}")
    if df['baseline_total'].sum() > 0:
        prec_b = df['baseline_tp'].sum() / df['baseline_total'].sum()
        print(f"  Precision: {prec_b:.3f}")
    print(f"  Avg max confidence: {df['baseline_max_conf'].mean():.3f}")
    print(f"  Avg avg confidence: {df['baseline_avg_conf'].mean():.3f}")
    
    print("\nOptimized:")
    print(f"  Total candidates: {df['optimized_total'].sum()}")
    print(f"  True Positives: {df['optimized_tp'].sum()}")
    print(f"  False Positives: {df['optimized_fp'].sum()}")
    if df['optimized_total'].sum() > 0:
        prec_o = df['optimized_tp'].sum() / df['optimized_total'].sum()
        print(f"  Precision: {prec_o:.3f}")
    print(f"  Avg max confidence: {df['optimized_max_conf'].mean():.3f}")
    print(f"  Avg avg confidence: {df['optimized_avg_conf'].mean():.3f}")
    
    # Improvements
    print("\nImprovement:")
    total_reduction = df['baseline_total'].sum() - df['optimized_total'].sum()
    fp_reduction = df['baseline_fp'].sum() - df['optimized_fp'].sum()
    
    print(f"  Candidate reduction: {total_reduction} ({total_reduction/df['baseline_total'].sum()*100:.1f}%)")
    print(f"  FP reduction: {fp_reduction} ({fp_reduction/df['baseline_fp'].sum()*100:.1f}%)")
    
    if df['baseline_total'].sum() > 0 and df['optimized_total'].sum() > 0:
        prec_improvement = (prec_o - prec_b) / prec_b * 100
        print(f"  Precision improvement: {prec_improvement:+.1f}%")
    
    conf_improvement = (df['optimized_max_conf'].mean() - df['baseline_max_conf'].mean())
    print(f"  Max confidence improvement: {conf_improvement:+.3f}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_detectors()
