"""
Comprehensive Validation Framework
====================================
Detailed validation with pixel-level metrics, parameter optimization,
and multi-case testing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict
import json

from medical_imaging.detection import TumorDetector, TumorCandidate


class ComprehensiveValidator:
    """Comprehensive validation framework"""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def load_nifti(self, path):
        """Load NIfTI file"""
        nii = nib.load(str(path))
        volume = nii.get_fdata()
        spacing = nii.header.get_zooms()
        return volume, spacing
    
    def compute_pixel_metrics(self, candidates: List[TumorCandidate], 
                               gt_slice: np.ndarray,
                               image_shape: Tuple[int, int]) -> Dict:
        """Compute pixel-level metrics"""
        
        # Create prediction mask from candidates
        pred_mask = np.zeros(image_shape, dtype=bool)
        for c in candidates:
            bbox = c.bounding_box
            min_row, min_col, max_row, max_col = bbox
            pred_mask[min_row:max_row, min_col:max_col] = True
        
        gt_mask = (gt_slice > 0)
        
        # Compute metrics
        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        pred_pixels = pred_mask.sum()
        gt_pixels = gt_mask.sum()
        
        # Dice coefficient
        dice = 2 * intersection / (pred_pixels + gt_pixels) if (pred_pixels + gt_pixels) > 0 else 0.0
        
        # IoU (Jaccard)
        iou = intersection / union if union > 0 else 0.0
        
        # Precision & Recall (pixel-level)
        precision = intersection / pred_pixels if pred_pixels > 0 else 0.0
        recall = intersection / gt_pixels if gt_pixels > 0 else 0.0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'intersection': int(intersection),
            'union': int(union),
            'pred_pixels': int(pred_pixels),
            'gt_pixels': int(gt_pixels)
        }
    
    def validate_case(self, case_name: str, detector: TumorDetector) -> Dict:
        """Validate a single case"""
        
        print(f"\n{'='*80}")
        print(f"Validating: {case_name}")
        print(f"{'='*80}")
        
        # Load data
        ct_path = self.data_dir / "imagesTr" / f"{case_name}.nii.gz"
        gt_path = self.data_dir / "labelsTr" / f"{case_name}.nii.gz"
        
        if not ct_path.exists() or not gt_path.exists():
            print(f"WARNING: Files not found for {case_name}")
            return None
        
        volume, spacing = self.load_nifti(ct_path)
        gt_volume, _ = self.load_nifti(gt_path)
        
        print(f"Volume shape: {volume.shape}")
        print(f"Spacing: {spacing}")
        
        # Find tumor slices
        tumor_slices = []
        for z in range(gt_volume.shape[2]):
            if gt_volume[:, :, z].sum() > 100:
                tumor_slices.append(z)
        
        print(f"Tumor slices: {len(tumor_slices)} - {tumor_slices}")
        
        if len(tumor_slices) == 0:
            print("WARNING: No tumor slices found")
            return None
        
        # Per-slice validation
        slice_results = []
        
        for z in tumor_slices:
            ct_slice = volume[:, :, z]
            gt_slice = gt_volume[:, :, z]
            
            # Detect candidates
            candidates = detector.detect_candidates_2d(
                hu_slice=ct_slice,
                pixel_spacing=(spacing[0], spacing[1]),
                body_mask=None,
                slice_index=z,
                method='multi_threshold'
            )
            
            # Compute metrics
            metrics = self.compute_pixel_metrics(
                candidates, gt_slice, ct_slice.shape
            )
            
            # Count true/false positives
            true_positives = 0
            false_positives = 0
            
            for c in candidates:
                bbox = c.bounding_box
                min_row, min_col, max_row, max_col = bbox
                gt_region = gt_slice[min_row:max_row, min_col:max_col]
                if (gt_region > 0).sum() > 0:
                    true_positives += 1
                else:
                    false_positives += 1
            
            slice_result = {
                'case': case_name,
                'slice': z,
                'total_candidates': len(candidates),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'high_confidence': sum(1 for c in candidates if c.confidence_score > 0.5),
                'max_confidence': max([c.confidence_score for c in candidates]) if candidates else 0.0,
                'avg_confidence': np.mean([c.confidence_score for c in candidates]) if candidates else 0.0,
                **metrics
            }
            
            slice_results.append(slice_result)
            
            print(f"  Slice {z}: {len(candidates)} candidates, "
                  f"Dice={metrics['dice']:.3f}, "
                  f"IoU={metrics['iou']:.3f}, "
                  f"TP={true_positives}, FP={false_positives}")
        
        # Aggregate case results
        case_result = {
            'case': case_name,
            'num_slices': len(tumor_slices),
            'avg_dice': np.mean([r['dice'] for r in slice_results]),
            'avg_iou': np.mean([r['iou'] for r in slice_results]),
            'avg_precision': np.mean([r['precision'] for r in slice_results]),
            'avg_recall': np.mean([r['recall'] for r in slice_results]),
            'total_candidates': sum(r['total_candidates'] for r in slice_results),
            'total_tp': sum(r['true_positives'] for r in slice_results),
            'total_fp': sum(r['false_positives'] for r in slice_results),
            'avg_max_conf': np.mean([r['max_confidence'] for r in slice_results]),
            'slice_results': slice_results
        }
        
        print(f"\nCase Summary:")
        print(f"  Avg Dice: {case_result['avg_dice']:.3f}")
        print(f"  Avg IoU: {case_result['avg_iou']:.3f}")
        print(f"  Total: {case_result['total_candidates']} candidates "
              f"({case_result['total_tp']} TP, {case_result['total_fp']} FP)")
        
        return case_result
    
    def parameter_sweep(self, case_name: str):
        """Test different parameter combinations"""
        
        print(f"\n{'='*80}")
        print(f"Parameter Sweep: {case_name}")
        print(f"{'='*80}")
        
        # Parameter combinations to test
        param_combinations = [
            # (min_area_mm2, max_area_mm2, hu_min, hu_max)
            (10.0, 10000.0, -50, 200),    # Default
            (10.0, 10000.0, 80, 160),     # Colon tumor HU
            (20.0, 5000.0, -50, 200),     # Smaller range
            (5.0, 10000.0, -50, 200),     # More sensitive
            (10.0, 10000.0, 60, 140),     # Tighter HU
        ]
        
        results = []
        
        for i, (min_area, max_area, hu_min, hu_max) in enumerate(param_combinations):
            print(f"\nTest {i+1}: min_area={min_area}, max_area={max_area}, "
                  f"hu_range=({hu_min}, {hu_max})")
            
            detector = TumorDetector(
                min_area_mm2=min_area,
                max_area_mm2=max_area,
                hu_range=(hu_min, hu_max)
            )
            
            case_result = self.validate_case(case_name, detector)
            
            if case_result:
                case_result['params'] = {
                    'min_area_mm2': min_area,
                    'max_area_mm2': max_area,
                    'hu_min': hu_min,
                    'hu_max': hu_max
                }
                results.append(case_result)
        
        # Find best parameters
        if results:
            best = max(results, key=lambda x: x['avg_dice'])
            print(f"\n{'='*80}")
            print(f"Best Parameters:")
            print(f"{'='*80}")
            print(f"Params: {best['params']}")
            print(f"Dice: {best['avg_dice']:.3f}")
            print(f"IoU: {best['avg_iou']:.3f}")
            print(f"Precision: {best['avg_precision']:.3f}")
            print(f"Recall: {best['avg_recall']:.3f}")
        
        return results
    
    def multi_case_validation(self, case_names: List[str], detector: TumorDetector):
        """Validate on multiple cases"""
        
        print(f"\n{'='*80}")
        print(f"Multi-Case Validation: {len(case_names)} cases")
        print(f"{'='*80}")
        
        all_results = []
        
        for case_name in case_names:
            result = self.validate_case(case_name, detector)
            if result:
                all_results.append(result)
        
        # Aggregate statistics
        if all_results:
            print(f"\n{'='*80}")
            print(f"OVERALL STATISTICS")
            print(f"{'='*80}")
            
            avg_dice = np.mean([r['avg_dice'] for r in all_results])
            avg_iou = np.mean([r['avg_iou'] for r in all_results])
            avg_precision = np.mean([r['avg_precision'] for r in all_results])
            avg_recall = np.mean([r['avg_recall'] for r in all_results])
            total_candidates = sum(r['total_candidates'] for r in all_results)
            total_tp = sum(r['total_tp'] for r in all_results)
            total_fp = sum(r['total_fp'] for r in all_results)
            
            print(f"\nCases tested: {len(all_results)}")
            print(f"\nPixel-level Metrics:")
            print(f"  Average Dice: {avg_dice:.3f}")
            print(f"  Average IoU: {avg_iou:.3f}")
            print(f"  Average Precision: {avg_precision:.3f}")
            print(f"  Average Recall: {avg_recall:.3f}")
            
            print(f"\nCandidate-level Metrics:")
            print(f"  Total candidates: {total_candidates}")
            print(f"  True Positives: {total_tp}")
            print(f"  False Positives: {total_fp}")
            if total_candidates > 0:
                print(f"  Candidate Precision: {total_tp/total_candidates:.3f}")
            
            # Save results
            results_file = self.output_dir / "multi_case_results.json"
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {results_file}")
            
            return all_results
        
        return None


def main():
    print("="*80)
    print("COMPREHENSIVE VALIDATION FRAMEWORK")
    print("="*80)
    
    # Setup
    data_dir = Path("data/medical_decathlon/Task10_Colon")
    output_dir = Path("results/comprehensive_validation")
    
    validator = ComprehensiveValidator(data_dir, output_dir)
    
    # Default detector
    detector = TumorDetector(
        min_area_mm2=10.0,
        max_area_mm2=10000.0,
        hu_range=(-50, 200)
    )
    
    # Test cases
    test_cases = [
        "colon_001",
        "colon_002", 
        "colon_003",
        "colon_004",
        "colon_005"
    ]
    
    print(f"\nTest configuration:")
    print(f"  Cases: {test_cases}")
    print(f"  Detector: min_area={detector.min_area_mm2}, "
          f"hu_range={detector.hu_range}")
    
    # 1. Single case detailed validation
    print("\n" + "="*80)
    print("PHASE 1: Detailed Single Case Validation")
    print("="*80)
    
    single_result = validator.validate_case("colon_001", detector)
    
    # 2. Parameter sweep
    print("\n" + "="*80)
    print("PHASE 2: Parameter Optimization")
    print("="*80)
    
    param_results = validator.parameter_sweep("colon_001")
    
    # 3. Multi-case validation
    print("\n" + "="*80)
    print("PHASE 3: Multi-Case Validation")
    print("="*80)
    
    # Only validate cases that exist
    available_cases = []
    for case in test_cases:
        ct_path = data_dir / "imagesTr" / f"{case}.nii.gz"
        if ct_path.exists():
            available_cases.append(case)
    
    print(f"Available cases: {available_cases}")
    
    if available_cases:
        multi_results = validator.multi_case_validation(available_cases, detector)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
