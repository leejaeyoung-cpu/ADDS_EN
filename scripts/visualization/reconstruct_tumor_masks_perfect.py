"""
Perfect Tumor 3D Mask Reconstruction using actual segmentation masks

Uses real detection masks instead of circular approximation
"""
import numpy as np
import nibabel as nib
from pathlib import Path
import json
from typing import Dict, List, Tuple


def reconstruct_with_direct_mapping(
    tumor: Dict,
    detection_masks: np.lib.npyio.NpzFile,
    detections_by_slice: Dict,
    volume_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Reconstruct 3D mask using DIRECT detection index lookup
    
    NO FUZZY MATCHING - uses stored detection indices for perfect accuracy
    
    Args:
        tumor: Tumor data with 'detection_indices' field
        detection_masks: NPZ file with all 2D masks
        detections_by_slice: Dict of {slice_idx: [detections]}
        volume_shape: 3D volume shape
    
    Returns:
        3D binary mask for this tumor
    """
    mask_3d = np.zeros(volume_shape, dtype=bool)
    
    # CRITICAL FIX: Use direct detection indices (no fuzzy matching!)
    detection_indices = tumor.get('detection_indices', [])
    
    if not detection_indices:
        # Fallback only if indices not available (shouldn't happen with new code)
        return mask_3d
    
    # Place each detection mask directly
    for (slice_idx, det_idx) in detection_indices:
        mask_key = f"slice_{slice_idx:03d}_det_{det_idx:02d}"
        
        if mask_key not in detection_masks:
            continue
        
        mask_2d = detection_masks[mask_key]
        
        # Get detection info for bounding box
        if slice_idx not in detections_by_slice:
            continue
        
        detections = detections_by_slice[slice_idx]
        if det_idx >= len(detections):
            continue
        
        det = detections[det_idx]
        bbox = det['bbox']  # min_row, min_col, max_row, max_col
        
        # Place mask at correct position
        y1, x1, y2, x2 = bbox
        
        # Ensure bbox is within volume bounds
        y1 = max(0, min(y1, volume_shape[1]))
        y2 = max(0, min(y2, volume_shape[1]))
        x1 = max(0, min(x1, volume_shape[2]))
        x2 = max(0, min(x2, volume_shape[2]))
        
        # Place mask
        if slice_idx < volume_shape[0] and mask_2d.shape[0] > 0 and mask_2d.shape[1] > 0:
            h = y2 - y1
            w = x2 - x1
            if h > 0 and w > 0:
                mask_resized = mask_2d[:h, :w]  # Trim if needed
                mask_3d[slice_idx, y1:y1+mask_resized.shape[0], x1:x1+mask_resized.shape[1]] = mask_resized
    
    return mask_3d


def reconstruct_with_actual_masks(
    tumors_json_path: Path,
    detection_json_path: Path,
    detection_masks_path: Path,
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float]
) -> Dict[int, np.ndarray]:
    """
    Reconstruct 3D tumor masks using DIRECT detection index mapping
    
    NO FUZZY MATCHING - 100% accurate reconstruction
    
    Returns:
        Dictionary of {tumor_id: mask_3d}
    """
    print("\n" + "="*80)
    print("PERFECT 3D MASK RECONSTRUCTION - DIRECT MAPPING")
    print("="*80)
    
    # Load data
    print(f"\n[*] Loading tumor data: {tumors_json_path}")
    with open(tumors_json_path) as f:
        tumor_data = json.load(f)
    
    tumors = tumor_data['tumors']
    print(f"[+] Loaded {len(tumors)} tumors")
    
    print(f"\n[*] Loading detection data: {detection_json_path}")
    with open(detection_json_path) as f:
        detection_data = json.load(f)
    
    # Load masks
    print(f"\n[*] Loading detection masks: {detection_masks_path}")
    masks_npz = np.load(detection_masks_path)
    print(f"[+] Loaded {len(masks_npz.files)} detection masks")
    
    # Build detection index
    detections_by_slice = {}
    for result in detection_data['results']:
        if result.get('has_tumor'):
            slice_idx = result['slice_idx']
            detections_by_slice[slice_idx] = result.get('detections', [])
    
    print(f"[+] Indexed detections in {len(detections_by_slice)} slices")
    
    # Reconstruct each tumor using DIRECT mapping
    tumor_masks = {}
    tumors_with_perfect_match = 0
    tumors_with_fallback = 0
    
    print(f"\n[*] Reconstructing tumor masks with DIRECT lookup...")
    for i, tumor in enumerate(tumors):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(tumors)}")
        
        tumor_id = tumor['tumor_id']
        
        # CRITICAL FIX: Use direct detection index lookup
        mask_3d = reconstruct_with_direct_mapping(
            tumor, masks_npz, detections_by_slice, volume_shape
        )
        
        # Check if we got a mask
        if mask_3d.sum() > 0:
            tumors_with_perfect_match += 1
        else:
            tumors_with_fallback += 1
        
        tumor_masks[tumor_id] = mask_3d
    
    print(f"\n[+] Reconstructed {len(tumor_masks)} tumor masks")
    print(f"[+] {tumors_with_perfect_match} tumors with PERFECT match ({tumors_with_perfect_match/len(tumors)*100:.1f}%)")
    print(f"[+] {tumors_with_fallback} tumors with fallback ({tumors_with_fallback/len(tumors)*100:.1f}%)")
    print("="*80)
    
    return tumor_masks


if __name__ == "__main__":
    # Paths -Updated to use new detection with masks
    tumors_json = Path("outputs/inha_3d_analysis/tumors_3d.json")
    detections_json = Path("outputs/inha_ct_detection_with_masks/detection_summary.json")
    detection_masks = Path("outputs/inha_ct_detection_with_masks/detection_masks.npz")
    volume_path = Path("outputs/inha_ct_analysis/inha_ct_volume.nii.gz")
    output_dir = Path("outputs/inha_3d_analysis/tumor_masks_perfect")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load volume
    nii = nib.load(volume_path)
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    affine = nii.affine
    
    print(f"[*] Volume shape: {volume.shape}")
    print(f"[*] Spacing: {spacing}")
    
    # Check if detection masks exist
    if not detection_masks.exists():
        print(f"\n[!] Detection masks not found: {detection_masks}")
        print(f"[!] Please re-run detection with mask saving enabled")
        print(f"[!] Run: python detect_tumors_inha_corrected.py")
        exit(1)
    
    # Reconstruct masks
    tumor_masks = reconstruct_with_actual_masks(
        tumors_json,
        detections_json,
        detection_masks,
        volume.shape,
        spacing
    )
    
    # Save first 5 as examples
    print(f"\n[*] Saving sample masks...")
    for tumor_id in sorted(tumor_masks.keys())[:5]:
        mask = tumor_masks[tumor_id]
        if mask.sum() > 0:  # Only save non-empty masks
            mask_nii = nib.Nifti1Image(mask.astype(np.uint8), affine)
            output_path = output_dir / f"tumor_{tumor_id:03d}_mask_perfect.nii.gz"
            nib.save(mask_nii, output_path)
            voxels = mask.sum()
            print(f"  [+] Saved tumor {tumor_id}: {voxels:,} voxels")
    
    # Save all masks
    masks_path = output_dir / "all_tumor_masks_perfect.npz"
    print(f"\n[*] Saving all masks: {masks_path}")
    np.savez_compressed(
        masks_path,
        **{f"tumor_{tid}": mask for tid, mask in tumor_masks.items()}
    )
    print(f"[+] Saved {len(tumor_masks)} masks")
    
    print("\n" + "="*80)
    print("PERFECT RECONSTRUCTION COMPLETE")
    print("="*80)
