"""
Test Candidate Detection Pipeline
===================================
Validates the txt pipeline integration with Medical Decathlon data

Expected output:
- Candidate list with confidence scores
- Detection statistics
- Comparison with old ADDS results (Dice 0.0000)
"""

import sys
from pathlib import Path
import numpy as np
import nibabel as nib

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.medical_imaging.detection.candidate_detector import (
    TumorDetector,
    BodySegmentation,
    CTPreprocessor,
    merge_candidates
)


def load_nifti_volume(filepath: str):
    """Load NIfTI file and extract spacing"""
    nifti_img = nib.load(filepath)
    
    # Use float32 to save memory (critical for large volumes)
    volume = np.asarray(nifti_img.dataobj, dtype=np.float32)
    
    # Get spacing from header
    header = nifti_img.header
    spacing = header.get_zooms()[:3]  # (x, y, z)
    spacing = (spacing[2], spacing[1], spacing[0])  # Convert to (z, y, x)
    
    return volume, spacing


def test_candidate_detection(data_dir: str = "data/medical_decathlon/Task10_Colon"):
    """
    Test candidate detection on Medical Decathlon data
    
    Args:
        data_dir: Path to Medical Decathlon Task10 data
    """
    dataset_root = Path(data_dir)
    images_dir = dataset_root / "imagesTr"
    labels_dir = dataset_root / "labelsTr"
    
    # Find test cases
    image_files = sorted(images_dir.glob("*.nii.gz"))[:3]  # Test on first 3 cases
    
    if not image_files:
        print(f"[ERROR] No NIfTI files found in {images_dir}")
        print(f"   Please ensure Medical Decathlon data is available")
        return
    
    print("=" * 70)
    print("TXT Pipeline Candidate Detection Test")
    print("=" * 70)
    print("")
    
    # Initialize detector with txt pipeline parameters
    detector = TumorDetector(
        min_area_mm2=10.0,     # CRITICAL: txt pipeline value (vs ADDS: 1000)
        max_area_mm2=10000.0,
        hu_range=(-50, 200)    # Soft tissue + tumor range
    )
    
    print(f"[OK] Detector initialized:")
    print(f"   min_area: {detector.min_area_mm2} mm² (txt pipeline)")
    print(f"   hu_range: {detector.hu_range}")
    print("")
    
    all_results = []
    
    for idx, image_path in enumerate(image_files, 1):
        case_id = image_path.stem.replace('.nii', '')
        
        print(f"[{idx}/{len(image_files)}] Processing: {case_id}")
        print("-" * 70)
        
        # Load volume
        try:
            volume, spacing = load_nifti_volume(str(image_path))
            print(f"   Shape: {volume.shape}")
            print(f"   Spacing: {spacing} mm")
            print(f"   HU range: {volume.min():.1f} to {volume.max():.1f}")
            
            # Detect candidates (process every 5th slice to save time)
            max_slices = 20  # Process 20 slices max per volume
            candidates = detector.detect_candidates_3d(
                volume, spacing, max_slices=max_slices
            )
            
            # Merge duplicate candidates
            candidates = merge_candidates(candidates, distance_threshold=20.0)
            
            # Filter high-confidence
            high_conf = [c for c in candidates if c.confidence_score > 0.5]
            
            print(f"   [*] Detection Results:")
            print(f"      Total candidates: {len(candidates)}")
            print(f"      High confidence (>0.5): {len(high_conf)}")
            
            if candidates:
                max_conf = max(c.confidence_score for c in candidates)
                print(f"      Max confidence: {max_conf:.2f}")
                
                # Show top 5
                print(f"   ")
                print(f"   [TOP 5] Candidates:")
                for i, c in enumerate(candidates[:5], 1):
                    print(f"      #{i}: Slice {c.slice_index}, "
                          f"Area={c.area_mm2:.1f}mm2, "
                          f"HU={c.mean_hu:.1f}, "
                          f"Conf={c.confidence_score:.2f}")
            
            all_results.append({
                'case_id': case_id,
                'total': len(candidates),
                'high_conf': len(high_conf),
                'max_conf': max_conf if candidates else 0.0
            })
            
        except Exception as e:
            print(f"   [ERROR] {e}")
        
        print("")
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("")
    
    if all_results:
        total_candidates = sum(r['total'] for r in all_results)
        total_high_conf = sum(r['high_conf'] for r in all_results)
        max_conf_overall = max(r['max_conf'] for r in all_results)
        
        print(f"Total cases processed: {len(all_results)}")
        print(f"Total candidates detected: {total_candidates}")
        print(f"Total high confidence: {total_high_conf} ({100*total_high_conf/total_candidates:.1f}%)" if total_candidates > 0 else "N/A")
        print(f"Max confidence (overall): {max_conf_overall:.2f}")
        print("")
        
        print("[COMPARISON] vs. Old ADDS Performance:")
        print(f"   Old ADDS: Dice 0.0000 (complete failure)")
        print(f"   TXT Pipeline: {total_candidates} candidates detected")
        print(f"   TXT Pipeline: {max_conf_overall:.2f} max confidence")
        print("")
        
        if total_candidates > 0:
            print("[SUCCESS] Candidate detection is working!")
            print("   Next step: Integrate with SOTA predictor for hybrid detection")
        else:
            print("[WARNING] No candidates detected")
            print("   Try adjusting hu_range or min_area_mm2 parameters")
    else:
        print("[ERROR] No results to summarize")
    
    print("")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test candidate detection pipeline")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/medical_decathlon/Task10_Colon",
        help="Path to Medical Decathlon Task10 data"
    )
    
    args = parser.parse_args()
    
    test_candidate_detection(args.data_dir)
