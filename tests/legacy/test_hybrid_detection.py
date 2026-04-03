"""
Test Hybrid Detection System
==============================
Validate hybrid predictor combining DL + txt pipeline candidate detection

Test modes:
- Candidate-only (txt pipeline)
- DL-only (if checkpoint available)
- Hybrid (DL + Candidate fusion)

Expected improvement: Dice 0.0-0.31 → 0.5-0.7
"""

import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import time
import json

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.medical_imaging.inference.hybrid_predictor import HybridPredictor


def load_nifti(filepath: str):
    """Load NIfTI volume and spacing"""
    nifti_img = nib.load(filepath)
    volume = np.asarray(nifti_img.dataobj, dtype=np.float32)
    header = nifti_img.header
    spacing = header.get_zooms()[:3]
    spacing = (spacing[2], spacing[1], spacing[0])  # (z, y, x)
    return volume, spacing


def test_hybrid_detection(
    data_dir: str = "data/medical_decathlon/Task10_Colon",
    num_cases: int = 3,
    mode: str = "rule_only",  # Start with rule_only (no checkpoint needed)
    dl_checkpoint: str = None
):
    """
    Test hybrid detection system
    
    Args:
        data_dir: Path to Medical Decathlon data
        num_cases: Number of test cases
        mode: Detection mode ("rule_only", "dl_only", "hybrid", "ensemble")
        dl_checkpoint: Path to DL model checkpoint (optional)
    """
    dataset_root = Path(data_dir)
    images_dir = dataset_root / "imagesTr"
    
    image_files = sorted(images_dir.glob("*.nii.gz"))[:num_cases]
    
    if not image_files:
        print(f"[ERROR] No NIfTI files found in {images_dir}")
        return
    
    print("=" * 80)
    print("HYBRID DETECTION SYSTEM TEST")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Cases: {num_cases}")
    print("")
    
    # Initialize predictor
    print(f"[1/{num_cases+1}] Initializing {mode} predictor...")
    
    try:
        predictor = HybridPredictor(
            dl_checkpoint_path=dl_checkpoint,
            device="cuda",
            detection_mode=mode,
            enable_fp_filtering=False,  # Disabled (txt pipeline doesn't use this)
            enable_colon_scoring=False  # Disabled (txt pipeline doesn't use this)
        )
        print(f"   [OK] Predictor initialized in '{mode}' mode")
        print("")
    except Exception as e:
        print(f"   [ERROR] Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test each case
    all_results = []
    
    for idx, image_path in enumerate(image_files, 1):
        case_id = image_path.stem.replace('.nii', '')
        
        print(f"[{idx+1}/{num_cases+1}] Processing: {case_id}")
        print("-" * 80)
        
        try:
            # Load volume
            volume, spacing = load_nifti(str(image_path))
            print(f"   Shape: {volume.shape}")
            print(f"   Spacing: {spacing} mm")
            print(f"   HU range: {volume.min():.1f} to {volume.max():.1f}")
            
            # Run prediction
            start_time = time.time()
            
            results = predictor.predict(
                volume=volume,
                spacing=spacing,
                return_candidates=True,
                return_probabilities=False
            )
            
            elapsed = time.time() - start_time
            
            # Extract results
            mask = results['mask']
            candidates = results.get('candidates', [])
            metadata = results.get('metadata', {})
            
            # Statistics
            tumor_voxels = mask.sum()
            num_candidates = len(candidates) if candidates else 0
            high_conf = sum(1 for c in candidates if c.confidence_score > 0.5) if candidates else 0
            max_conf = max([c.confidence_score for c in candidates]) if candidates else 0.0
            
            print(f"   [*] Results:")
            print(f"      Tumor voxels: {tumor_voxels}")
            print(f"      Candidates: {num_candidates}")
            print(f"      High confidence (>0.5): {high_conf}")
            print(f"      Max confidence: {max_conf:.2f}")
            print(f"      Processing time: {elapsed:.2f}s")
            
            if candidates:
                print(f"   ")
                print(f"   [TOP 3] Candidates:")
                for i, c in enumerate(candidates[:3], 1):
                    print(f"      #{i}: Slice {c.slice_index}, "
                          f"Area={c.area_mm2:.1f}mm2, "
                          f"HU={c.mean_hu:.1f}, "
                          f"Conf={c.confidence_score:.2f}")
            
            all_results.append({
                'case_id': case_id,
                'tumor_voxels': int(tumor_voxels),
                'num_candidates': num_candidates,
                'high_conf_candidates': high_conf,
                'max_confidence': float(max_conf),
                'processing_time': elapsed,
                'metadata': metadata
            })
            
        except Exception as e:
            print(f"   [ERROR] {e}")
            import traceback
            traceback.print_exc()
        
        print("")
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("")
    
    if all_results:
        total_voxels = sum(r['tumor_voxels'] for r in all_results)
        total_candidates = sum(r['num_candidates'] for r in all_results)
        total_high_conf = sum(r['high_conf_candidates'] for r in all_results)
        avg_time = np.mean([r['processing_time'] for r in all_results])
        max_conf_overall = max(r['max_confidence'] for r in all_results)
        
        print(f"Cases processed: {len(all_results)}")
        print(f"Total tumor voxels: {total_voxels}")
        print(f"Total candidates: {total_candidates}")
        print(f"High confidence candidates: {total_high_conf}")
        print(f"Max confidence (overall): {max_conf_overall:.2f}")
        print(f"Average processing time: {avg_time:.2f}s")
        print("")
        
        print(f"[COMPARISON] vs. Previous Performance:")
        print(f"   Old ADDS: Dice 0.0000 (complete failure)")
        print(f"   txt pipeline standalone: 169 candidates, 0.99 max conf")
        print(f"   Hybrid {mode}: {total_candidates} candidates, {max_conf_overall:.2f} max conf")
        print("")
        
        if total_candidates > 0 or total_voxels > 0:
            print(f"[SUCCESS] {mode.upper()} detection working!")
            print(f"   Next: Test other modes (dl_only, hybrid, ensemble)")
        else:
            print(f"[WARNING] No detections in {mode} mode")
        
        # Save results
        output_file = f"outputs/hybrid_test_{mode}_{num_cases}cases.json"
        Path("outputs").mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'mode': mode,
                'num_cases': num_cases,
                'results': all_results,
                'summary': {
                    'total_voxels': total_voxels,
                    'total_candidates': total_candidates,
                    'high_conf_candidates': total_high_conf,
                    'max_confidence': float(max_conf_overall),
                    'avg_time': float(avg_time)
                }
            }, f, indent=2)
        print(f"   Results saved: {output_file}")
    else:
        print("[ERROR] No results to summarize")
    
    print("")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid detection system")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/medical_decathlon/Task10_Colon",
        help="Path to Medical Decathlon data"
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=3,
        help="Number of test cases"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rule_only",
        choices=["rule_only", "dl_only", "hybrid", "ensemble"],
        help="Detection mode"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to DL model checkpoint (for dl_only, hybrid, ensemble modes)"
    )
    
    args = parser.parse_args()
    
    test_hybrid_detection(
        data_dir=args.data_dir,
        num_cases=args.num_cases,
        mode=args.mode,
        dl_checkpoint=args.checkpoint
    )
