"""
Step 3 OPTIMIZED V2: Tiered Processing Based on Tumor Size

Strategy: Skip expensive operations for small tumors, preserve original algorithm for large ones
Expected speedup: 20-30%
"""
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from skimage import measure, morphology
from typing import List, Dict, Tuple
import json
import time


def ellipsoid_surface_area(mask_3d: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    """
    Fast approximation for very small tumors using ellipsoid fit
    """
    coords = np.argwhere(mask_3d)
    
    if len(coords) < 3:
        # Fallback to spherical
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        volume_mm3 = mask_3d.sum() * voxel_volume
        radius = (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
        return 4 * np.pi * (radius ** 2)
    
    # Fit ellipsoid using PCA
    coords_centered = coords - coords.mean(axis=0)
    cov = np.cov(coords_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Semi-axes of ellipsoid (with spacing correction)
    a = np.sqrt(eigenvalues[0]) * spacing[0]
    b = np.sqrt(eigenvalues[1]) * spacing[1]
    c = np.sqrt(eigenvalues[2]) * spacing[2]
    
    # Approximate ellipsoid surface area (Knud Thomsen's formula)
    p = 1.6075
    sa = 4 * np.pi * (((a*b)**p + (a*c)**p + (b*c)**p) / 3) ** (1/p)
    
    return sa


def calculate_surface_area_tiered(
    mask_3d: np.ndarray,
    spacing: Tuple[float, float, float],
    volume_mm3: float = None
) -> float:
    """
    TIERED: Choose processing method based on tumor size
    
    Tier 1 (<100 voxels): Ellipsoid approximation (FAST)
    Tier 2 (100-500): No smoothing, direct marching cubes
    Tier 3 (500-2000): Light smoothing (erosion only)
    Tier 4 (2000+): Full smoothing (original algorithm)
    """
    voxel_count = mask_3d.sum()
    
    try:
        # TIER 1: Very small tumors - Use ellipsoid approximation
        if voxel_count < 100:
            return ellipsoid_surface_area(mask_3d, spacing)
        
        # TIER 2: Small tumors - Skip smoothing
        elif voxel_count < 500:
            verts, faces, normals, values = measure.marching_cubes(
                mask_3d.astype(float),
                level=0.5,
                spacing=spacing,
                allow_degenerate=False
            )
            surface_area = measure.mesh_surface_area(verts, faces)
        
        # TIER 3: Medium tumors - Light smoothing (erosion only)
        elif voxel_count < 2000:
            struct = ndimage.generate_binary_structure(3, 1)
            mask_smoothed = ndimage.binary_erosion(mask_3d, structure=struct)
            
            verts, faces, normals, values = measure.marching_cubes(
                mask_smoothed.astype(float),
                level=0.5,
                spacing=spacing,
                allow_degenerate=False
            )
            surface_area = measure.mesh_surface_area(verts, faces)
        
        # TIER 4: Large tumors - Full smoothing (ORIGINAL)
        else:
            # Use original morphological smoothing
            mask_smoothed = morphology.binary_closing(
                mask_3d, morphology.ball(1)
            )
            mask_smoothed = morphology.binary_opening(
                mask_smoothed, morphology.ball(1)
            )
            
            verts, faces, normals, values = measure.marching_cubes(
                mask_smoothed.astype(float),
                level=0.5,
                spacing=spacing,
                allow_degenerate=False
            )
            surface_area = measure.mesh_surface_area(verts, faces)
        
        # Sanity check
        if volume_mm3 is not None and volume_mm3 > 0:
            radius = (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
            expected_sa = 4 * np.pi * (radius ** 2)
            if surface_area > 10 * expected_sa:
                surface_area = expected_sa
        
        return surface_area
        
    except Exception as e:
        # Fallback
        struct = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(mask_3d, structure=struct)
        surface_voxels = mask_3d & ~eroded
        voxel_face_area = spacing[0] * spacing[1]
        surface_area = surface_voxels.sum() * voxel_face_area * 6
        
        if volume_mm3 is not None and volume_mm3 > 0:
            radius = (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
            expected_sa = 4 * np.pi * (radius ** 2)
            if surface_area > 10 * expected_sa:
                surface_area = expected_sa
        
        return surface_area


def calculate_sphericity(volume_mm3: float, surface_area_mm2: float) -> float:
    """Sphericity = (π^(1/3) * (6*V)^(2/3)) / A"""
    if surface_area_mm2 <= 0 or volume_mm3 <= 0:
        return 0.0
    numerator = (np.pi ** (1/3)) * ((6 * volume_mm3) ** (2/3))
    sphericity = numerator / surface_area_mm2
    return min(sphericity, 1.0)


def calculate_compactness(volume_mm3: float, surface_area_mm2: float) -> float:
    """Compactness = 36π * V² / A³"""
    if surface_area_mm2 <= 0 or volume_mm3 <= 0:
        return 0.0
    numerator = 36 * np.pi * (volume_mm3 ** 2)
    denominator = surface_area_mm2 ** 3
    compactness = numerator / denominator
    return min(compactness, 1.0)


def calculate_elongation(mask_3d: np.ndarray) -> float:
    """Elongation = λ2 / λ1 from PCA"""
    coords = np.argwhere(mask_3d)
    if len(coords) < 3:
        return 1.0
    coords_centered = coords - coords.mean(axis=0)
    cov = np.cov(coords_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    if eigenvalues[0] > 0:
        return float(eigenvalues[1] / eigenvalues[0])
    return 1.0


def calculate_flatness(mask_3d: np.ndarray) -> float:
    """Flatness = λ3 / λ1 from PCA"""
    coords = np.argwhere(mask_3d)
    if len(coords) < 3:
        return 1.0
    coords_centered = coords - coords.mean(axis=0)
    cov = np.cov(coords_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    if eigenvalues[0] > 0:
        return float(eigenvalues[2] / eigenvalues[0])
    return 1.0


def calculate_tumor_to_colon_distance(
    tumor_mask: np.ndarray,
    colon_mask: np.ndarray,
    spacing: Tuple[float, float, float]
) -> Dict:
    """Calculate distance from tumor to colon wall"""
    overlap = tumor_mask & colon_mask
    overlap_voxels = overlap.sum()
    tumor_voxels = tumor_mask.sum()
    
    if tumor_voxels == 0:
        return {
            'min_distance_mm': 0.0,
            'mean_distance_mm': 0.0,
            'overlap_fraction': 0.0,
            'inside_colon': True
        }
    
    overlap_fraction = overlap_voxels / tumor_voxels
    
    if overlap_fraction > 0.1:
        return {
            'min_distance_mm': 0.0,
            'mean_distance_mm': 0.0,
            'overlap_fraction': float(overlap_fraction),
            'inside_colon': True
        }
    
    colon_distance_map = ndimage.distance_transform_edt(~colon_mask, sampling=spacing)
    tumor_coords = np.argwhere(tumor_mask)
    tumor_distances = [colon_distance_map[tuple(coord)] for coord in tumor_coords]
    
    if len(tumor_distances) > 0:
        min_dist = float(np.min(tumor_distances))
        mean_dist = float(np.mean(tumor_distances))
    else:
        min_dist = 0.0
        mean_dist = 0.0
    
    return {
        'min_distance_mm': min_dist,
        'mean_distance_mm': mean_dist,
        'overlap_fraction': 0.0,
        'inside_colon': False
    }


def enhance_tumor_measurements_optimized_v2(
    tumors_json_path: Path,
    tumor_masks_path: Path,
    colon_mask_path: Path,
    output_path: Path,
    spacing: Tuple[float, float, float]
):
    """Enhanced measurements with TIERED processing"""
    print("="*80)
    print("STEP 3 OPTIMIZED V2: Tiered Processing")
    print("STRATEGY: Size-based conditional processing")
    print("="*80)
    
    # Load data
    print(f"\n[*] Loading tumor data: {tumors_json_path}")
    with open(tumors_json_path) as f:
        tumor_data = json.load(f)
    
    tumors = tumor_data['tumors']
    print(f"[+] Found {len(tumors)} tumors")
    
    print(f"\n[*] Loading PERFECT tumor masks: {tumor_masks_path}")
    masks_npz = np.load(tumor_masks_path)
    tumor_masks = {int(k.split('_')[1]): v for k, v in masks_npz.items()}
    print(f"[+] Loaded {len(tumor_masks)} masks")
    
    print(f"\n[*] Loading colon mask: {colon_mask_path}")
    colon_nii = nib.load(colon_mask_path)
    colon_mask = colon_nii.get_fdata().astype(bool)
    print(f"[+] Colon mask loaded")
    
    # Process tumors
    enhanced_tumors = []
    tier_counts = {'tier1': 0, 'tier2': 0, 'tier3': 0, 'tier4': 0}
    
    print(f"\n[*] Computing TIERED measurements...")
    start_time = time.time()
    
    for idx, tumor in enumerate(tumors):
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{len(tumors)}")
        
        tumor_id = tumor['tumor_id']
        
        try:
            tumor_mask = tumor_masks.get(tumor_id)
            
            if tumor_mask is None or tumor_mask.sum() == 0:
                # Fallback
                volume_mm3 = tumor['volume_mm3']
                radius = (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
                surface_area = 4 * np.pi * (radius ** 2)
                sphericity = 1.0
                compactness = 1.0
                elongation = 1.0
                flatness = 1.0
                distance_metrics = {
                    'min_distance_mm': 0.0,
                    'mean_distance_mm': 0.0,
                    'overlap_fraction': 0.0,
                    'inside_colon': True
                }
            else:
                # Track tier usage
                voxel_count = tumor_mask.sum()
                if voxel_count < 100:
                    tier_counts['tier1'] += 1
                elif voxel_count < 500:
                    tier_counts['tier2'] += 1
                elif voxel_count < 2000:
                    tier_counts['tier3'] += 1
                else:
                    tier_counts['tier4'] += 1
                
                # TIERED measurements
                volume_mm3 = tumor['volume_mm3']
                surface_area = calculate_surface_area_tiered(tumor_mask, spacing, volume_mm3)
                sphericity = calculate_sphericity(volume_mm3, surface_area)
                compactness = calculate_compactness(volume_mm3, surface_area)
                elongation = calculate_elongation(tumor_mask)
                flatness = calculate_flatness(tumor_mask)
                distance_metrics = calculate_tumor_to_colon_distance(
                    tumor_mask, colon_mask, spacing
                )
            
            enhanced_tumor = tumor.copy()
            enhanced_tumor['surface_area_mm2'] = float(surface_area)
            enhanced_tumor['sphericity'] = float(sphericity)
            enhanced_tumor['compactness'] = float(compactness)
            enhanced_tumor['elongation'] = float(elongation)
            enhanced_tumor['flatness'] = float(flatness)
            enhanced_tumor['distance_to_colon'] = distance_metrics
            
            enhanced_tumors.append(enhanced_tumor)
            
        except Exception as e:
            print(f"  [!] Error processing tumor {tumor_id}: {e}")
            enhanced_tumors.append(tumor)
    
    elapsed = time.time() - start_time
    
    # Save results
    output_data = {
        'summary': tumor_data['summary'],
        'tumors': enhanced_tumors,
        'processing': {
            'optimized': 'v2_tiered',
            'time_seconds': elapsed,
            'time_per_tumor': elapsed / len(tumors) if tumors else 0,
            'tier_distribution': tier_counts
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[+] Enhanced measurements saved: {output_path}")
    print(f"[+] Processing time: {elapsed:.1f}s ({elapsed/len(tumors):.2f}s per tumor)")
    
    # Calculate speedup
    original_time_per_tumor = 3.49
    speedup = (original_time_per_tumor * len(tumors)) / elapsed
    print(f"[+] Speedup vs original: {speedup:.2f}x")
    
    # Tier distribution
    print(f"\n[*] Tier Distribution:")
    print(f"  Tier 1 (<100 voxels, ellipsoid): {tier_counts['tier1']}")
    print(f"  Tier 2 (100-500, no smoothing): {tier_counts['tier2']}")
    print(f"  Tier 3 (500-2000, light smoothing): {tier_counts['tier3']}")
    print(f"  Tier 4 (2000+, full smoothing): {tier_counts['tier4']}")
    
    # Statistics
    print("\n" + "="*80)
    print("SHAPE ANALYSIS SUMMARY")
    print("="*80)
    
    sphericities = [t['sphericity'] for t in enhanced_tumors]
    elongations = [t['elongation'] for t in enhanced_tumors]
    flatnesses = [t['flatness'] for t in enhanced_tumors]
    surface_areas = [t['surface_area_mm2'] for t in enhanced_tumors]
    
    inside_colon = sum(1 for t in enhanced_tumors if t['distance_to_colon']['inside_colon'])
    outside_colon = len(enhanced_tumors) - inside_colon
    
    print(f"Sphericity: {np.min(sphericities):.3f} - {np.max(sphericities):.3f} (mean: {np.mean(sphericities):.3f})")
    print(f"Elongation: {np.min(elongations):.3f} - {np.max(elongations):.3f} (mean: {np.mean(elongations):.3f})")
    print(f"Flatness: {np.min(flatnesses):.3f} - {np.max(flatnesses):.3f} (mean: {np.mean(flatnesses):.3f})")
    print(f"Surface area: {np.min(surface_areas):.1f} - {np.max(surface_areas):.1f} mm²")
    print(f"\nInside colon: {inside_colon} ({inside_colon/len(enhanced_tumors)*100:.1f}%)")
    print(f"Outside colon: {outside_colon} ({outside_colon/len(enhanced_tumors)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("OPTIMIZED V2 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    tumors_json = Path("outputs/inha_3d_analysis/tumors_3d.json")
    tumor_masks_npz = Path("outputs/inha_3d_analysis/tumor_masks_perfect/all_tumor_masks_perfect.npz")
    colon_mask = Path("outputs/inha_ct_detection/3d_segmentation/colon_mask_3d_conservative.nii.gz")
    volume_path = Path("outputs/inha_ct_analysis/inha_ct_volume.nii.gz")
    output_path = Path("outputs/inha_3d_analysis/tumors_3d_enhanced_optimized_v2.json")
    
    nii = nib.load(volume_path)
    spacing = nii.header.get_zooms()
    
    enhance_tumor_measurements_optimized_v2(
        tumors_json, tumor_masks_npz, colon_mask, output_path, spacing
    )
