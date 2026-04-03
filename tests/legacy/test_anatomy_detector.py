"""
Test Anatomy-Based Tumor Detector on Medical Decathlon Colon Data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import nibabel as nib
from src.medical_imaging.detection.anatomy_based_detector import (
    AnatomyBasedTumorDetector,
    quick_detect_colon_tumors
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_on_medical_decathlon():
    """Test on actual Medical Decathlon data"""
    
    data_root = Path("data/medical_decathlon/Task10_Colon")
    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"
    
    # Get first case
    image_files = sorted(images_dir.glob("*.nii.gz"))
    if not image_files:
        print("❌ No Medical Decathlon data found!")
        print(f"Expected location: {images_dir}")
        return
    
    # Test on first 3 cases
    num_test_cases = min(3, len(image_files))
    
    print("="*80)
    print("ANATOMY-BASED TUMOR DETECTION TEST")
    print("="*80)
    print(f"\nTesting on {num_test_cases} cases from Medical Decathlon")
    print(f"Data: {data_root}\n")
    
    for i, image_path in enumerate(image_files[:num_test_cases]):
        case_name = image_path.stem.replace('.nii', '')
        label_path = labels_dir / image_path.name
        
        print(f"\n{'='*80}")
        print(f"Case {i+1}/{num_test_cases}: {case_name}")
        print(f"{'='*80}")
        
        # Load image
        print(f"Loading {image_path.name}...")
        nifti_img = nib.load(image_path)
        ct_volume = nifti_img.get_fdata()
        spacing = nifti_img.header.get_zooms()
        
        print(f"Volume shape: {ct_volume.shape}")
        print(f"Voxel spacing: {spacing} mm")
        print(f"HU range: [{ct_volume.min():.1f}, {ct_volume.max():.1f}]")
        
        # Load ground truth if available
        gt_volume = None
        has_tumor_gt = False
        if label_path.exists():
            gt_nifti = nib.load(label_path)
            gt_volume = gt_nifti.get_fdata()
            has_tumor_gt = (gt_volume > 0).any()
            tumor_voxels = (gt_volume > 0).sum()
            print(f"Ground truth: {'TUMOR' if has_tumor_gt else 'NORMAL'}")
            if has_tumor_gt:
                print(f"  Tumor volume: {tumor_voxels} voxels")
        
        # Run detection
        print(f"\n[*] Running anatomy-based detection...")
        try:
            detector = AnatomyBasedTumorDetector(device="gpu", fast_mode=True)
            results = detector.detect(ct_volume, spacing=spacing)
            
            # Display results
            print(f"\n[DETECTION RESULTS]:")
            
            total_candidates = 0
            tumor_candidates = 0
            
            for organ_name, candidates in results.items():
                print(f"\n  {organ_name.upper()}: {len(candidates)} candidates")
                
                for j, candidate in enumerate(candidates):
                    total_candidates += 1
                    
                    # Only detailed output for tumor-likely candidates
                    if candidate.classification in ['tumor_likely', 'tumor_probable', 'mixed_density']:
                        tumor_candidates += 1
                        print(f"    [*] Candidate #{j+1} [{candidate.classification.upper()}]")
                        print(f"       Volume: {candidate.volume_voxels} voxels")
                        print(f"       HU: {candidate.mean_hu:.1f} ± {candidate.std_hu:.1f}")
                        print(f"       Range: [{candidate.min_hu:.1f}, {candidate.max_hu:.1f}] HU")
                        print(f"       Confidence: {candidate.confidence:.2%}")
                        print(f"       Location: z={candidate.centroid[0]}, y={candidate.centroid[1]}, x={candidate.centroid[2]}")
                        
                        # Check overlap with ground truth
                        if gt_volume is not None:
                            overlap = (candidate.mask & (gt_volume > 0)).sum()
                            recall = overlap / tumor_voxels if tumor_voxels > 0 else 0
                            precision = overlap / candidate.volume_voxels if candidate.volume_voxels > 0 else 0
                            dice = 2 * overlap / (candidate.volume_voxels + tumor_voxels) if (candidate.volume_voxels + tumor_voxels) > 0 else 0
                            
                            print(f"       GT Overlap: {overlap} voxels")
                            print(f"       Dice: {dice:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            print(f"\n  SUMMARY:")
            print(f"    Total candidates: {total_candidates}")
            print(f"    Tumor-likely: {tumor_candidates}")
            print(f"    Ground truth: {'TUMOR' if has_tumor_gt else 'NORMAL'}")
            
            if has_tumor_gt and tumor_candidates > 0:
                print(f"    [OK] True Positive (detected tumor in positive case)")
            elif has_tumor_gt and tumor_candidates == 0:
                print(f"    [!!] False Negative (missed tumor in positive case)")
            elif not has_tumor_gt and tumor_candidates > 0:
                print(f"    [!!] False Positive (detected tumor in negative case)")
            elif not has_tumor_gt and tumor_candidates == 0:
                print(f"    [OK] True Negative (no detection in negative case)")
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"TEST COMPLETE")
    print(f"{'='*80}\n")


def quick_test():
    """Quick synthetic test"""
    print("\n[Quick Synthetic Test]\n")
    
    # Create synthetic data
    ct_volume = np.random.randn(64, 128, 128) * 15 + 35  # Soft tissue
    
    # Add simulated tumor
    ct_volume[30:35, 60:70, 60:70] = 75  # Bright lesion
    
    print(f"Synthetic volume: {ct_volume.shape}")
    print(f"HU range: [{ct_volume.min():.1f}, {ct_volume.max():.1f}]")
    
    # Quick detection (will fallback if TotalSegmentator not available)
    try:
        candidates = quick_detect_colon_tumors(ct_volume, spacing=(5.0, 1.0, 1.0))
        print(f"\nDetected {len(candidates)} candidates")
        for i, c in enumerate(candidates):
            print(f"  {i+1}. {c.classification} @ HU={c.mean_hu:.1f}, conf={c.confidence:.2f}")
    except Exception as e:
        print(f"[WARNING] Quick test failed (expected without TotalSegmentator): {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Anatomy-Based Tumor Detector")
    parser.add_argument("--quick", action="store_true", help="Run quick synthetic test only")
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        test_on_medical_decathlon()
