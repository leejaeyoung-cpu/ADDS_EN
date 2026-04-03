"""
3D Tumor Clustering and Size Measurement

Groups 2D detections across slices into 3D tumor candidates
and measures their actual 3D size (volume, diameter)
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import cdist

@dataclass
class Tumor3D:
    """3D tumor candidate"""
    tumor_id: int
    slices: List[int]  # Slice indices
    centroids_2d: List[Tuple[float, float]]  # (y, x) for each slice
    areas_mm2: List[float]  # Area in each slice
    mean_hu: float
    confidence: float
    
    # CRITICAL FIX: Store detection indices for perfect reconstruction
    detection_indices: List[Tuple[int, int]] = None  # [(slice_idx, det_idx), ...]
    
    # 3D measurements  
    volume_mm3: float = 0.0
    max_diameter_mm: float = 0.0
    z_range_mm: float = 0.0
    
    def __repr__(self):
        return (f"Tumor3D(id={self.tumor_id}, slices={len(self.slices)}, "
                f"volume={self.volume_mm3:.1f}mm³, diameter={self.max_diameter_mm:.1f}mm)")


def cluster_detections_3d(
    detection_results: List[Dict],
    spacing: Tuple[float, float, float],
    max_distance_mm: float = 20.0
) -> List[Tumor3D]:
    """
    Cluster 2D detections into 3D tumor candidates
    
    Args:
        detection_results: List of slice detection results
        spacing: Voxel spacing (x, y, z) in mm
        max_distance_mm: Maximum distance to consider same tumor
    
    Returns:
        List of 3D tumor candidates
    """
    
    # Extract all detections with slice info
    all_detections = []
    detection_index_map = {}  # Maps i -> (slice_idx, det_idx)
    
    for result in detection_results:
        if not result.get('has_tumor'):
            continue
        
        slice_idx = result['slice_idx']
        for det_idx, detection in enumerate(result.get('detections', [])):
            i = len(all_detections)  # Current index
            all_detections.append({
                'slice_idx': slice_idx,
                'centroid': detection['centroid'],
                'area_mm2': detection['area_pixels'] * spacing[0] * spacing[1],  # Convert to mm²
                'mean_hu': detection['mean_hu'],
                'confidence': detection['confidence']
            })
            # CRITICAL FIX: Store detection source
            detection_index_map[i] = (slice_idx, det_idx)
    
    if len(all_detections) == 0:
        return []
    
    # Convert to 3D coordinates (mm)
    coords_3d = []
    for det in all_detections:
        z = det['slice_idx'] * spacing[2]
        y = det['centroid'][0] * spacing[1]
        x = det['centroid'][1] * spacing[0]
        coords_3d.append([x, y, z])
    
    coords_3d = np.array(coords_3d)
    
    # Simple clustering: group by proximity
    n_detections = len(all_detections)
    clusters = []
    used = set()
    
    for i in range(n_detections):
        if i in used:
            continue
        
        # Start new cluster
        cluster = [i]
        used.add(i)
        
        # Find nearby detections
        for j in range(i + 1, n_detections):
            if j in used:
                continue
            
            # Check if close enough
            dist = np.linalg.norm(coords_3d[i] - coords_3d[j])
            if dist < max_distance_mm:
                cluster.append(j)
                used.add(j)
        
        clusters.append(cluster)
    
    # Create 3D tumor objects
    tumors_3d = []
    for tumor_id, cluster in enumerate(clusters):
        slices = [all_detections[i]['slice_idx'] for i in cluster]
        centroids = [all_detections[i]['centroid'] for i in cluster]
        areas = [all_detections[i]['area_mm2'] for i in cluster]
        mean_hu = np.mean([all_detections[i]['mean_hu'] for i in cluster])
        confidence = np.max([all_detections[i]['confidence'] for i in cluster])
        
        # CRITICAL FIX: Collect detection indices
        detection_indices = [detection_index_map[i] for i in cluster]
        
        tumor = Tumor3D(
            tumor_id=tumor_id,
            slices=slices,
            centroids_2d=centroids,
            areas_mm2=areas,
            mean_hu=mean_hu,
            confidence=confidence,
            detection_indices=detection_indices  # Store for perfect reconstruction!
        )
        
        # Calculate 3D measurements
        tumor = calculate_3d_size(tumor, spacing)
        
        tumors_3d.append(tumor)
    
    return tumors_3d


def calculate_3d_size(tumor: Tumor3D, spacing: Tuple[float, float, float]) -> Tumor3D:
    """
    Calculate 3D size measurements for a tumor
    
    Args:
        tumor: Tumor3D object
        spacing: Voxel spacing (x, y, z) in mm
    
    Returns:
        Tumor3D with updated measurements
    """
    
    # Volume: sum of slice areas × slice thickness
    slice_thickness = spacing[2]
    tumor.volume_mm3 = sum(tumor.areas_mm2) * slice_thickness
    
    # Z range
    if len(tumor.slices) > 0:
        z_min = min(tumor.slices) * slice_thickness
        z_max = max(tumor.slices) * slice_thickness
        tumor.z_range_mm = z_max - z_min
    
    # Maximum diameter: use  volume to estimate spherical diameter
    # V = 4/3 * π * r³ → r = (3V / 4π)^(1/3)
    if tumor.volume_mm3 > 0:
        radius = (3 * tumor.volume_mm3 / (4 * np.pi)) ** (1/3)
        tumor.max_diameter_mm = 2 * radius
    
    return tumor


def format_tumor_summary(tumors_3d: List[Tumor3D], total_detections: int, total_slices: int) -> Dict:
    """
    Format tumor summary for display
    
    Args:
        tumors_3d: List of 3D tumors
        total_detections: Total 2D detections
        total_slices: Total slices with detections
    
    Returns:
        Summary dictionary
    """
    
    if len(tumors_3d) == 0:
        return {
            'tumor_count': 0,
            'total_detections': total_detections,
            'tumor_slices': total_slices,
            'largest_tumor_mm': 0.0,
            'total_volume_ml': 0.0,
            'summary_text': f"검출된 종양 없음 ({total_detections} detections analyzed)"
        }
    
    # Sort by volume
    tumors_sorted = sorted(tumors_3d, key=lambda t: t.volume_mm3, reverse=True)
    
    largest_tumor = tumors_sorted[0]
    total_volume_mm3 = sum(t.volume_mm3 for t in tumors_3d)
    total_volume_ml = total_volume_mm3 / 1000.0
    
    summary_text = (
        f"{len(tumors_3d)}개 종양 후보 검출 "
        f"({total_detections} detections across {total_slices} slices)"
    )
    
    return {
        'tumor_count': len(tumors_3d),
        'total_detections': total_detections,
        'tumor_slices': total_slices,
        'largest_tumor_mm': largest_tumor.max_diameter_mm,
        'total_volume_ml': total_volume_ml,
        'summary_text': summary_text,
        'tumors': [
            {
                'id': t.tumor_id,
                'volume_mm3': round(t.volume_mm3, 1),
                'diameter_mm': round(t.max_diameter_mm, 1),
                'z_range_mm': round(t.z_range_mm, 1),
                'slices': len(t.slices),
                'confidence': round(t.confidence, 2)
            }
            for t in tumors_sorted[:10]  # Top 10
        ]
    }


if __name__ == "__main__":
    print("="*80)
    print("TESTING 3D TUMOR CLUSTERING")
    print("="*80)
    
    # Load detection results
    import json
    results_path = Path("outputs/inha_ct_detection/strict_test/detection_summary.json")
    
    if not results_path.exists():
        print(f"ERROR: {results_path} not found!")
        exit(1)
    
    print(f"\n[*] Loading: {results_path}")
    with open(results_path) as f:
        data = json.load(f)
    
    results = data['results']
    spacing = tuple(data['voxel_spacing_mm'])
    
    print(f"[*] Spacing: {spacing}")
    print(f"[*] Total slice results: {len(results)}")
    
    # Cluster
    print(f"\n[*] Clustering detections into 3D tumors...")
    tumors_3d = cluster_detections_3d(results, spacing, max_distance_mm=20.0)
    
    print(f"\n[+] Found {len(tumors_3d)} 3D tumor candidates")
    
    # Summary
    total_detections = sum(len(r.get('candidates', [])) for r in results if r.get('has_tumor'))
    tumor_slices = sum(1 for r in results if r.get('has_tumor'))
    
    summary = format_tumor_summary(tumors_3d, total_detections, tumor_slices)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"종양 후보: {summary['tumor_count']}개")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Tumor-positive slices: {summary['tumor_slices']}")
    print(f"Largest tumor: {summary['largest_tumor_mm']:.1f} mm")
    print(f"Total volume: {summary['total_volume_ml']:.2f} mL")
    print(f"\n{summary['summary_text']}")
    
    print(f"\n{'='*80}")
    print("TOP 10 TUMORS")
    print(f"{'='*80}")
    if 'tumors' in summary:
        for t in summary['tumors']:
            print(f"  Tumor #{t['id']}: {t['diameter_mm']:.1f}mm diameter, "
                  f"{t['volume_mm3']:.1f}mm³, {t['slices']} slices, "
                  f"conf={t['confidence']:.0%}")
    else:
        print("  (No tumors detected)")
    
    print(f"\n{'='*80}")
