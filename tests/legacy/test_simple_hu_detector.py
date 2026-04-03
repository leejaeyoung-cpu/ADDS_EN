"""
Test Simple HU Detector on Medical Decathlon Data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import nibabel as nib
from src.medical_imaging.detection.simple_hu_detector import SimpleHUDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_detector():
    """Test simple HU detector on Medical Decathlon data"""
    
    data_root = Path("data/medical_decathlon/Task10_Colon")
    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"
    
    image_files = sorted(images_dir.glob("*.nii.gz"))
    if not image_files:
        print(f"[ERROR] No data found at {images_dir}")
        return
    
    num_test = min(3, len(image_files))
    
    print("="*80)
    print("SIMPLE HU-BASED TUMOR DETECTION TEST")
    print("="*80)
    print(f"\nTesting {num_test} cases")
    print(f"Method: Pure HU threshold (40-150 HU)")
    print(f"No organ segmentation required\n")
    
    detector = SimpleHUDetector(
        tumor_hu_min=40.0,
        tumor_hu_max=150.0,
        min_volume_mm3=50.0
    )
    
    results_summary = []
    
    for i, image_path in enumerate(image_files[:num_test]):
        case_name = image_path.stem.replace('.nii', '')
        label_path = labels_dir / image_path.name
        
        print(f"\n{'='*80}")
        print(f"Case {i+1}/{num_test}: {case_name}")
        print(f"{'='*80}")
        
        # Load image
        nifti_img = nib.load(image_path)
        ct_volume = nifti_img.get_fdata()
        spacing = nifti_img.header.get_zooms()
        
        print(f"Volume: {ct_volume.shape}, Spacing: {spacing} mm")
        print(f"HU range: [{ct_volume.min():.1f}, {ct_volume.max():.1f}]")
        
        # Load ground truth
        gt_volume = None
        has_tumor = False
        tumor_voxels = 0
        
        if label_path.exists():
            gt_nifti = nib.load(label_path)
            gt_volume = gt_nifti.get_fdata()
            has_tumor = (gt_volume > 0).any()
            tumor_voxels = (gt_volume > 0).sum()
            print(f"Ground truth: {'TUMOR (%d voxels)' % tumor_voxels if has_tumor else 'NORMAL'}")
        
        # Run detection
        print(f"\n[*] Running simple HU detection...")
        lesions = detector.detect(ct_volume, spacing=spacing)
        
        # Analyze results
        print(f"\n[RESULTS]:")
        print(f"  Detected: {len(lesions)} lesions")
        
        if len(lesions) > 0:
            print(f"\n  Lesion details:")
            for j, lesion in enumerate(lesions):
                print(f"    #{j+1}:")
                print(f"      Volume: {lesion.volume_voxels} voxels")
                print(f"      HU: {lesion.mean_hu:.1f} (max: {lesion.max_hu:.1f})")
                print(f"      Location: z={lesion.centroid[0]}, y={lesion.centroid[1]}, x={lesion.centroid[2]}")
                
                # Check overlap with ground truth
                if gt_volume is not None and has_tumor:
                    overlap = (lesion.mask & (gt_volume > 0)).sum()
                    if overlap > 0:
                        recall = overlap / tumor_voxels if tumor_voxels > 0 else 0
                        precision = overlap / lesion.volume_voxels if lesion.volume_voxels > 0 else 0
                        dice = 2 * overlap / (lesion.volume_voxels + tumor_voxels)
                        
                        print(f"      Overlap: {overlap} voxels")
                        print(f"      Dice: {dice:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
                        
                        if dice > 0.1:
                            print(f"      [***] LIKELY TRUE POSITIVE!")
        
        # Summary
        print(f"\n  SUMMARY:")
        if has_tumor and len(lesions) > 0:
            # Check if any lesion overlaps with GT
            best_overlap = 0
            if gt_volume is not None:
                for lesion in lesions:
                    overlap = (lesion.mask & (gt_volume > 0)).sum()
                    if overlap > best_overlap:
                        best_overlap = overlap
            
            if best_overlap > 0:
                print(f"    [OK] TRUE POSITIVE (overlap: {best_overlap} voxels)")
                results_summary.append("TP")
            else:
                print(f"    [!!] FALSE POSITIVE (no overlap with GT)")
                results_summary.append("FP")
        elif has_tumor and len(lesions) == 0:
            print(f"    [!!] FALSE NEGATIVE (missed tumor)")
            results_summary.append("FN")
        elif not has_tumor and len(lesions) > 0:
            print(f"    [!!] FALSE POSITIVE (detected in normal case)")
            results_summary.append("FP")
        else:
            print(f"    [OK] TRUE NEGATIVE (correctly detected nothing)")
            results_summary.append("TN")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Results: {results_summary}")
    print(f"TP: {results_summary.count('TP')}, FP: {results_summary.count('FP')}")
    print(f"FN: {results_summary.count('FN')}, TN: {results_summary.count('TN')}")
    
    if results_summary.count('TP') > 0:
        print(f"\n[SUCCESS] Detected tumors successfully!")
    else:
        print(f"\n[INFO] No true positives - may need parameter tuning")


if __name__ == "__main__":
    test_simple_detector()
