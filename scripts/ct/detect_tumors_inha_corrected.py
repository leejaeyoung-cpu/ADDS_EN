"""
Inha Hospital CT Tumor Detection with Corrected Axial Volume
==============================================================
Re-run tumor detection using properly reconstructed axial volume
Fixes orientation issues from previous detection (2026-01-28)
"""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import json
from datetime import datetime

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.medical_imaging.detection.candidate_detector import TumorDetector


def load_nifti_volume(filepath):
    """Load NIfTI volume"""
    try:
        nii = nib.load(filepath)
        volume = nii.get_fdata()
        spacing = nii.header.get_zooms()
        
        print(f"[OK] Loaded volume: {volume.shape}")
        print(f"[OK] Spacing: {spacing}")
        print(f"[OK] HU range: [{volume.min():.1f}, {volume.max():.1f}]")
        
        return volume, spacing, nii.affine
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None, None, None


def is_in_abdomen_region(centroid, image_shape):
    """
    Check if candidate is in abdominal region (colon location)
    
    Args:
        centroid: (y, x) position
        image_shape: (height, width)
    
    Returns:
        bool: True if in abdominal region
    """
    y, x = centroid
    h, w = image_shape
    
    # Relaxed region for colon (10-90% to catch more candidates)
    # This excludes only extreme edges
    x_min, x_max = w * 0.1, w * 0.9
    y_min, y_max = h * 0.1, h * 0.9
    
    return (x_min < x < x_max) and (y_min < y < y_max)


def detect_tumors_in_slice(slice_data, detector, pixel_spacing, slice_idx, organ_mask=None):
    """
    Detect tumors in a single axial slice with anatomical filtering
    
    Args:
        organ_mask: Optional 2D mask for specific organ (e.g., colon)
    
    Returns:
        candidates: List of tumor candidates (filtered for organ region if provided)
    """
    try:
        # STEP 0: Create body mask to exclude background (air/table)
        # Background has HU < -200 (air is ~-1000)
        body_mask = slice_data > -200
        
        # Detect all candidates
        all_candidates = detector.detect_candidates_2d(
            hu_slice=slice_data,
            pixel_spacing=pixel_spacing[:2],  # X, Y spacing
            slice_index=slice_idx,
            method='multi_threshold'
        )
        
        # === ANATOMICAL FILTERING FOR COLON ===
        
        # FILTER 0: Body mask - exclude candidates outside patient body
        image_shape = slice_data.shape
        candidates_in_body = []
        for c in all_candidates:
            y, x = int(c.centroid[0]), int(c.centroid[1])
            # Check if centroid is within body
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                if body_mask[y, x]:
                    candidates_in_body.append(c)
        
        # FILTER 1: Organ mask (if provided, e.g., colon)
        if organ_mask is not None:
            candidates_in_organ = []
            for c in candidates_in_body:
                y, x = int(c.centroid[0]), int(c.centroid[1])
                if 0 <= y < organ_mask.shape[0] and 0 <= x < organ_mask.shape[1]:
                    if organ_mask[y, x]:
                        candidates_in_organ.append(c)
            candidates_filtered_region = candidates_in_organ
        else:
            # Fallback: use anatomical region filter
            candidates_filtered_region = [
                c for c in candidates_in_body
                if is_in_abdomen_region(c.centroid, image_shape)
            ]
        
        # FILTER 2: Relaxed HU range for colon tumors (20-200 HU)
        # Still excludes: air (-1000), lungs (-500 to -200), fat (-100 to -50)
        candidates_colon_hu = [
            c for c in candidates_filtered_region
            if 20 <= c.mean_hu <= 200
        ]
        
        # FILTER 3: Relaxed size constraints (10-8000 mm²)
        candidates_filtered = [
            c for c in candidates_colon_hu
            if 10 <= c.area_mm2 <= 8000
        ]
        
        return candidates_filtered
        
    except Exception as e:
        print(f"[ERROR] Detection failed for slice {slice_idx}: {e}")
        return []


def visualize_detection(slice_data, candidates, slice_idx, output_path, z_position_mm=None):
    """Create visualization for a single slice with tumor detections"""
    
    # Filter high-confidence candidates
    high_conf = [c for c in candidates if c.confidence_score > 0.7]
    has_tumor = len(high_conf) > 0
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Normalize for visualization (soft tissue window)
    ct_display = np.clip(slice_data, -150, 250)
    ct_display = (ct_display + 150) / 400
    
    # 1. Original slice
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ct_display, cmap='gray', origin='lower')
    title = f'Axial Slice {slice_idx}/118'
    if z_position_mm is not None:
        title += f'\nZ = {z_position_mm:.1f} mm'
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Add HU statistics
    stats_text = f'HU: [{slice_data.min():.0f}, {slice_data.max():.0f}]\n'
    stats_text += f'Mean: {slice_data.mean():.0f}'
    ax1.text(10, 30, stats_text, color='yellow', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # 2. All candidates
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_display, cmap='gray', origin='lower')
    ax2.set_title(f'All Candidates (n={len(candidates)})', fontsize=12, fontweight='bold')
    
    for candidate in candidates[:50]:  # Limit to 50 for visibility
        x, y = candidate.centroid
        radius = max(5, min(30, np.sqrt(candidate.area_pixels / np.pi)))
        conf = candidate.confidence_score
        
        color = 'yellow' if conf > 0.7 else 'orange'
        alpha = 0.3 + 0.4 * conf
        
        circle = plt.Circle((x, y), radius, color=color, fill=False,
                          linewidth=2, alpha=alpha)
        ax2.add_patch(circle)
    ax2.axis('off')
    
    # 3. High-confidence detections
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(ct_display, cmap='gray', origin='lower')
    
    if has_tumor:
        for candidate in high_conf:
            x, y = candidate.centroid
            radius = max(5, min(30, np.sqrt(candidate.area_pixels / np.pi)))
            
            # Draw circle
            circle = plt.Circle((x, y), radius, color='red', fill=False,
                              linewidth=3, alpha=0.9)
            ax3.add_patch(circle)
            
            # Add confidence label
            ax3.text(x, y - radius - 5, f'{candidate.confidence_score:.1%}',
                    color='red', fontsize=9, fontweight='bold',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        ax3.set_title(f'TUMOR DETECTED (n={len(high_conf)})',
                     fontsize=12, fontweight='bold', color='red')
    else:
        ax3.set_title('NO TUMOR', fontsize=12, fontweight='bold', color='green')
    
    ax3.axis('off')
    
    # Overall title
    status = "TUMOR DETECTED" if has_tumor else "NO TUMOR"
    status_color = "red" if has_tumor else "green"
    plt.suptitle(f'Slice {slice_idx} - {status}',
                fontsize=14, fontweight='bold', color=status_color)
    
    # Save
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return {
        'slice_idx': slice_idx,
        'z_position_mm': z_position_mm,
        'has_tumor': has_tumor,
        'total_candidates': len(candidates),
        'high_conf_candidates': len(high_conf),
        'max_confidence': max([c.confidence_score for c in candidates]) if candidates else 0,
        'detections': [
            {
                'centroid': c.centroid,
                'area_pixels': c.area_pixels,
                'mean_hu': c.mean_hu,
                'min_hu': c.min_hu,
                'max_hu': c.max_hu,
                'confidence': c.confidence_score,
                'bbox': c.bounding_box,  # For mask placement
                'mask_2d': c.mask_2d  # Actual segmentation mask (will be saved separately)
            }
            for c in high_conf
        ]
    }


def run_detection(volume_path, output_dir, organ_mask_path=None):
    """
    Run tumor detection on NIfTI volume (can be called from Backend API)
    
    Args:
        volume_path: Path to NIfTI volume file
        output_dir: Directory to save detection results
        organ_mask_path: Optional path to organ mask (e.g., colon)
    
    Returns:
        dict: Detection results with tumor count and image paths
    """
    
    volume_path = Path(volume_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] Input volume: {volume_path}")
    print(f"[*] Output directory: {output_dir}")
    
    # Load organ mask if provided
    organ_mask_volume = None
    if organ_mask_path:
        print(f"[*] Loading organ mask: {organ_mask_path}")
        mask_nii = nib.load(organ_mask_path)
        organ_mask_volume = mask_nii.get_fdata().astype(bool)
        print(f"[*] Organ mask shape: {organ_mask_volume.shape}")
        organ_slices = np.where(organ_mask_volume.sum(axis=(1,2)) > 0)[0]
        if len(organ_slices) > 0:
            print(f"[*] Organ present in slices: {organ_slices[0]} to {organ_slices[-1]}")
    
    # Load volume
    print(f"\n[*] Loading NIfTI volume...")
    volume, spacing, affine = load_nifti_volume(volume_path)
    
    if volume is None:
        print("[ERROR] Failed to load volume!")
        return {'status': 'error', 'message': 'Failed to load volume'}
    
    n_slices = volume.shape[0]
    print(f"\n[*] Volume shape: {volume.shape}")
    print(f"[*] Number of axial slices: {n_slices}")
    print(f"[*] Voxel spacing: {spacing}")
    
    # Initialize detector
    print(f"\n[*] Initializing tumor detector...")
    detector = TumorDetector()
    
    # Process slices
    print(f"\n[*] Processing {n_slices} axial slices...")
    results = []
    tumor_count = 0
    
    for slice_idx in tqdm(range(n_slices), desc="Detecting tumors"):
        # Extract axial slice
        slice_data = volume[slice_idx, :, :]
        
        # Get organ mask for this slice (if available)
        slice_organ_mask = None
        if organ_mask_volume is not None:
            slice_organ_mask = organ_mask_volume[slice_idx, :, :]
        
        # Calculate Z position
        z_position_mm = float(slice_idx * spacing[2])
        
        # Detect tumors using corrected filter
        candidates = detect_tumors_in_slice(
            slice_data=slice_data,
            detector=detector,
            pixel_spacing=spacing,
            slice_idx=slice_idx,
            organ_mask=slice_organ_mask
        )
        
        # Visualize if tumors detected or every 10th slice
        if len([c for c in candidates if c.confidence_score > 0.7]) > 0 or slice_idx % 10 == 0:
            output_path = output_dir / f"slice_{slice_idx:03d}_detection.png"
            
            result = visualize_detection(
                slice_data=slice_data,
                candidates=candidates,
                slice_idx=slice_idx,
                output_path=output_path,
                z_position_mm=z_position_mm
            )
            
            results.append(result)
            
            if result['has_tumor']:
                tumor_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total slices processed: {n_slices}")
    print(f"Tumor-positive slices: {tumor_count}")
    print(f"Detection rate: {tumor_count/n_slices*100:.1f}%")
    
    # Get top 5 by confidence
    tumor_results = [r for r in results if r['has_tumor']]
    top_results = sorted(tumor_results, key=lambda x: x.get('max_confidence', 0), reverse=True)[:5]
    
    print(f"\nTop 5 detections:")
    for i, r in enumerate(top_results, 1):
        print(f"  {i}. Slice {r['slice_idx']:03d}: {r['high_conf_candidates']} lesions, "
              f"confidence={r['max_confidence']:.2f}")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Save detailed results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_volume': str(volume_path),
        'volume_shape': [int(x) for x in volume.shape],
        'voxel_spacing_mm': [float(x) for x in spacing],
        'orientation': 'AXIAL (corrected)',
        'total_slices': int(n_slices),
        'processed_slices': n_slices, # All slices are processed now
        'visualized_slices': len(results),
        'tumor_detected_slices': tumor_count,
        'detection_rate': f"{tumor_count/n_slices*100:.2f}%",
        'results': results
    }
    
    # Extract masks for separate storage (JSON can't serialize numpy arrays)
    all_masks = {}
    for result in results:
        if result.get('has_tumor'):
            slice_idx = result['slice_idx']
            for det_idx, det in enumerate(result.get('detections', [])):
                if 'mask_2d' in det and det['mask_2d'] is not None:
                    mask_key = f"slice_{slice_idx:03d}_det_{det_idx:02d}"
                    all_masks[mask_key] = det['mask_2d']
                    # Remove from JSON (can't serialize numpy)
                    det.pop('mask_2d')
    
    summary_path = output_dir / "detection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save masks separately as NPZ
    if all_masks:
        masks_path = output_dir / "detection_masks.npz"
        np.savez_compressed(masks_path, **all_masks)
        print(f"\n[+] Saved {len(all_masks)} detection masks: {masks_path}")
    
    print(f"[+] Summary saved to: {summary_path}")
    print(f"[+] Visualizations saved to: {output_dir}")
    
    # List tumor-positive slices
    if tumor_count > 0:
        print(f"\n[*] Tumor-positive slices ({tumor_count}):")
        for result in sorted([r for r in results if r['has_tumor']], 
                            key=lambda x: x['max_confidence'], reverse=True):
            print(f"  - Slice {result['slice_idx']:3d} (Z={result['z_position_mm']:6.1f}mm): "
                  f"{result['high_conf_candidates']} lesions, "
                  f"max conf={result['max_confidence']:.1%}")
    
    print(f"\n[OK] Detection complete!")
    print("="*80)
    
    # Return results for Backend API
    return {
        'status': 'success',
        'tumor_count': tumor_count,
        'total_slices': n_slices,
        'detection_rate': f"{tumor_count/n_slices*100:.1f}%",
        'results': results,
        'output_dir': str(output_dir)
    }



if __name__ == "__main__":
    main()
