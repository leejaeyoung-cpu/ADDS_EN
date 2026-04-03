"""
Step 1: 3D Connected Component Analysis

Convert 2D slice-by-slice detections into true 3D tumor segments
"""
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from skimage import measure
from typing import List, Dict, Tuple
import json

def create_3d_mask_from_2d_detections(
    detection_results: List[Dict],
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float]
) -> np.ndarray:
    """
    Create 3D binary mask from 2D detections
    
    Args:
        detection_results: List of slice results with detections
        volume_shape: (D, H, W) shape of volume
        spacing: (x, y, z) voxel spacing in mm
    
    Returns:
        3D binary mask where detected regions = 1
    """
    
    # Initialize empty 3D mask
    mask_3d = np.zeros(volume_shape, dtype=np.uint8)
    
    # Fill in detections from each slice
    for result in detection_results:
        if not result.get('has_tumor'):
            continue
        
        slice_idx = result['slice_idx']
        
        # Get detections for this slice
        for det in result.get('detections', []):
            centroid = det['centroid']  # (y, x)
            y, x = int(centroid[0]), int(centroid[1])
            
            # Estimate radius from area
            area_pixels = det.get('area_pixels', 10)
            radius = int(np.sqrt(area_pixels / np.pi))
            
            # Draw filled circle
            yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
            circle = (yy**2 + xx**2 <= radius**2)
            
            y_min = max(0, y - radius)
            y_max = min(volume_shape[1], y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(volume_shape[2], x + radius + 1)
            
            circle_crop = circle[
                max(0, radius - y):radius + (y_max - y_min) - radius,
                max(0, radius - x):radius + (x_max - x_min) - radius
            ]
            
            mask_3d[slice_idx, y_min:y_max, x_min:x_max] |= circle_crop
    
    return mask_3d


def segment_tumors_3d(
    mask_3d: np.ndarray,
    spacing: Tuple[float, float, float],
    min_volume_mm3: float = 50.0,
    max_volume_mm3: float = 50000.0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Perform 3D connected component analysis on tumor mask
    
    Args:
        mask_3d: Binary 3D mask
        spacing: Voxel spacing (x, y, z) in mm
        min_volume_mm3: Minimum tumor volume
        max_volume_mm3: Maximum tumor volume
    
    Returns:
        labeled_volume: 3D volume with each tumor labeled uniquely
        tumor_properties: List of tumor properties
    """
    
    print(f"\n[*] Performing 3D connected component analysis...")
    
    # 3D morphological closing to connect nearby regions
    struct = ndimage.generate_binary_structure(3, 2)  # 26-connectivity
    mask_closed = ndimage.binary_closing(mask_3d, structure=struct, iterations=2)
    
    # Label connected components
    labeled_volume, num_tumors = ndimage.label(mask_closed, structure=struct)
    
    print(f"[*] Found {num_tumors} initial 3D components")
    
    # Calculate voxel volume
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    # Analyze each tumor
    tumor_props = []
    valid_labels = []
    
    for label_id in range(1, num_tumors + 1):
        tumor_mask = (labeled_volume == label_id)
        voxel_count = tumor_mask.sum()
        volume_mm3 = voxel_count * voxel_volume_mm3
        
        # Filter by size
        if volume_mm3 < min_volume_mm3 or volume_mm3 > max_volume_mm3:
            labeled_volume[tumor_mask] = 0  # Remove
            continue
        
        # Get bounding box and centroid
        coords = np.where(tumor_mask)
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        centroid = (
            coords[2].mean() * spacing[0],  # x in mm
            coords[1].mean() * spacing[1],  # y in mm
            coords[0].mean() * spacing[2]   # z in mm
        )
        
        # Calculate diameter (approximate as sphere)
        radius_mm = (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
        diameter_mm = 2 * radius_mm
        
        # Calculate shape features
        props = measure.regionprops(tumor_mask.astype(int))[0]
        extent = props.extent  # fraction of bounding box filled
        
        tumor_props.append({
            'label': label_id,
            'volume_mm3': float(volume_mm3),
            'diameter_mm': float(diameter_mm),
            'centroid_mm': centroid,
            'bounding_box': {
                'z': (int(z_min), int(z_max)),
                'y': (int(y_min), int(y_max)),
                'x': (int(x_min), int(x_max))
            },
            'extent': float(extent),
            'voxel_count': int(voxel_count)
        })
        
        valid_labels.append(label_id)
    
    print(f"[+] After filtering: {len(tumor_props)} valid tumors")
    
    # Relabel to consecutive numbers
    new_labeled = np.zeros_like(labeled_volume)
    for new_label, old_label in enumerate(valid_labels, 1):
        new_labeled[labeled_volume == old_label] = new_label
        # Update label in properties
        for prop in tumor_props:
            if prop['label'] == old_label:
                prop['label'] = new_label
    
    return new_labeled, tumor_props


if __name__ == "__main__":
    print("="*80)
    print("STEP 1: 3D CONNECTED COMPONENT ANALYSIS")
    print("="*80)
    
    # Load detection results
    results_path = Path("outputs/inha_ct_detection/strict_test/detection_summary.json")
    
    print(f"\n[*] Loading detection results: {results_path}")
    with open(results_path) as f:
        data = json.load(f)
    
    results = data['results']
    spacing = tuple(data['voxel_spacing_mm'])
    volume_shape = tuple(data['volume_shape'])
    
    print(f"[*] Volume shape: {volume_shape}")
    print(f"[*] Spacing: {spacing}")
    
    # Step 1: Create 3D mask from 2D detections
    print(f"\n[*] Creating 3D mask from 2D detections...")
    mask_3d = create_3d_mask_from_2d_detections(results, volume_shape, spacing)
    
    print(f"[+] 3D mask created. Non-zero voxels: {mask_3d.sum():,}")
    
    # Step 2: 3D connected component analysis
    labeled_tumors, tumor_props = segment_tumors_3d(
        mask_3d, 
        spacing,
        min_volume_mm3=50.0,
        max_volume_mm3=50000.0
    )
    
    # Save results
    output_dir = Path("outputs/inha_ct_detection/3d_segmentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save labeled volume
    nifti_path = Path("outputs/inha_ct_analysis/inha_ct_volume.nii.gz")
    nii = nib.load(nifti_path)
    
    labeled_nii = nib.Nifti1Image(labeled_tumors.astype(np.uint16), nii.affine)
    labeled_path = output_dir / "tumors_3d_labeled.nii.gz"
    nib.save(labeled_nii, labeled_path)
    
    print(f"\n[+] Labeled volume saved: {labeled_path}")
    
    # Save tumor properties
    props_path = output_dir / "tumor_properties_3d.json"
    with open(props_path, 'w') as f:
        json.dump({
            'tumor_count': len(tumor_props),
            'tumors': tumor_props
        }, f, indent=2)
    
    print(f"[+] Tumor properties saved: {props_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total 3D tumors: {len(tumor_props)}")
    
    if tumor_props:
        sorted_tumors = sorted(tumor_props, key=lambda x: x['volume_mm3'], reverse=True)
        print(f"\nTop 5 largest tumors:")
        for i, t in enumerate(sorted_tumors[:5], 1):
            print(f"  {i}. Tumor #{t['label']}: {t['diameter_mm']:.1f}mm diameter, "
                  f"{t['volume_mm3']:.1f}mm³")
        
        total_volume = sum(t['volume_mm3'] for t in tumor_props)
        print(f"\nTotal tumor volume: {total_volume/1000:.2f} mL")
    
    print(f"\n{'='*80}")
    print("✓ STEP 1 COMPLETE")
    print(f"{'='*80}")
