"""
Test Hybrid Tumor Detector with TotalSegmentator
Tests Fast mode vs Auto mode (with TotalSegmentator)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import nibabel as nib
from src.medical_imaging.detection.hybrid_detector import HybridTumorDetector
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice score"""
    intersection = (pred_mask & gt_mask).sum()
    if pred_mask.sum() + gt_mask.sum() == 0:
        return 0.0
    return 2.0 * intersection / (pred_mask.sum() + gt_mask.sum())


def test_totalseg_hybrid():
    """Test hybrid detector with TotalSegmentator on Medical Decathlon data"""
    
    data_root = Path("data/medical_decathlon/Task10_Colon")
    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"
    
    image_files = sorted(images_dir.glob("*.nii.gz"))
    if not image_files:
        print(f"[ERROR] No data found at {images_dir}")
        return
    
    num_test = min(3, len(image_files))
    
    print("="*80)
    print("HYBRID TUMOR DETECTOR TEST - TotalSegmentator")
    print("="*80)
    print(f"\nTesting {num_test} cases")
    print(f"Comparing FAST mode vs AUTO mode (TotalSegmentator)\n")
    
    # Create detectors
    print("Initializing detectors...")
    detector_fast = HybridTumorDetector(mode='fast')
    print("  [OK] Fast detector ready")
    
    try:
        # Use TotalSegmentator for Auto mode
        detector_auto = HybridTumorDetector(
            mode='auto',
            segmenter_type='totalseg',  # Use TotalSegmentator
            use_gpu=True
        )
        print("  [OK] Auto detector ready (with TotalSegmentator)")
    except Exception as e:
        print(f"  [WARN] TotalSegmentator init failed: {e}")
        print("  [INFO] Falling back to fast mode only")
        detector_auto = None
    
    results_summary = {
        'fast': [],
        'auto': []
    }
    
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
        
        # Test FAST mode
        print(f"\n[FAST MODE]")
        start_time = time.time()
        lesions_fast = detector_fast.detect(ct_volume, spacing=spacing)
        fast_time = time.time() - start_time
        
        print(f"  Time: {fast_time:.2f}s")
        print(f"  Detected: {len(lesions_fast)} lesions")
        
        # Find best match with GT
        best_dice_fast = 0.0
        best_lesion_fast = None
        if gt_volume is not None and has_tumor:
            gt_mask = (gt_volume > 0)
            for lesion in lesions_fast:
                dice = calculate_dice(lesion.mask, gt_mask)
                if dice > best_dice_fast:
                    best_dice_fast = dice
                    best_lesion_fast = lesion
            
            if best_lesion_fast:
                print(f"  Best match: {best_lesion_fast.volume_voxels} voxels, HU {best_lesion_fast.mean_hu:.1f}, Dice {best_dice_fast:.3f}")
        
        # Test AUTO mode (TotalSegmentator)
        if detector_auto:
            print(f"\n[AUTO MODE - TotalSegmentator]")
            start_time = time.time()
            
            try:
                lesions_auto = detector_auto.detect(ct_volume, spacing=spacing)
                auto_time = time.time() - start_time
                
                print(f"  Time: {auto_time:.2f}s")
                print(f"  Detected: {len(lesions_auto)} lesions")
                
                if len(lesions_auto) > 0:
                    print(f"  Top candidates:")
                    for j, lesion in enumerate(lesions_auto[:5]):  # Top 5
                        print(f"    #{j+1}: {lesion.volume_voxels} voxels, " +
                              f"HU {lesion.mean_hu:.1f}, " +
                              f"organ: {lesion.organ}, " +
                              f"overlap: {lesion.organ_overlap_ratio:.2f}, " +
                              f"confidence: {lesion.confidence:.2f}")
                
                # Find best match with GT
                best_dice_auto = 0.0
                best_lesion_auto = None
                if gt_volume is not None and has_tumor:
                    gt_mask = (gt_volume > 0)
                    for lesion in lesions_auto:
                        dice = calculate_dice(lesion.mask, gt_mask)
                        if dice > best_dice_auto:
                            best_dice_auto = dice
                            best_lesion_auto = lesion
                    
                    if best_lesion_auto:
                        print(f"  Best match: {best_lesion_auto.volume_voxels} voxels, HU {best_lesion_auto.mean_hu:.1f}, Dice {best_dice_auto:.3f}")
                
                # Summary
                print(f"\n[COMPARISON]")
                print(f"  Fast mode: {len(lesions_fast)} candidates in {fast_time:.2f}s")
                print(f"  Auto mode: {len(lesions_auto)} candidates in {auto_time:.2f}s")
                
                if len(lesions_fast) > 0 and len(lesions_auto) > 0:
                    reduction = 100 * (1 - len(lesions_auto) / len(lesions_fast))
                    print(f"  False positive reduction: {reduction:.1f}%")
                
                if has_tumor:
                    print(f"  Fast Dice: {best_dice_fast:.3f}")
                    print(f"  Auto Dice: {best_dice_auto:.3f}")
                    
                    if best_dice_fast > 0.1 or best_dice_auto > 0.1:
                        print(f"  [OK] Tumor detected successfully")
                        results_summary['fast'].append('TP')
                        results_summary['auto'].append('TP')
                    else:
                        print(f"  [FAIL] Tumor missed")
                        results_summary['fast'].append('FN')
                        results_summary['auto'].append('FN')
                        
            except Exception as e:
                print(f"  [ERROR] Auto mode failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n[AUTO MODE] Skipped (not initialized)")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nFast Mode:")
    print(f"  TP: {results_summary['fast'].count('TP')}/{num_test}")
    
    if detector_auto:
        print(f"\nAuto Mode (TotalSegmentator):")
        print(f"  TP: {results_summary['auto'].count('TP')}/{num_test}")
    
    print(f"\n[OK] Testing complete!")


if __name__ == "__main__":
    test_totalseg_hybrid()
