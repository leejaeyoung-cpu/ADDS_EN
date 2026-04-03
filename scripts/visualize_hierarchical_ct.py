#!/usr/bin/env python3
"""
Hierarchical CT Visualization
Visualize organ segmentation + lesion detection
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Dict, List

# Organ color scheme
ORGAN_COLORS = {
    'liver': (255, 100, 100),          # Red
    'kidney_right': (100, 100, 255),   # Blue
    'kidney_left': (80, 80, 235),      # Dark Blue
    'spleen': (255, 150, 100),         # Orange
    'pancreas': (200, 100, 255),       # Purple
    'colon': (100, 255, 100),          # Green
    'small_bowel': (120, 255, 120),    # Light Green
    'stomach': (255, 255, 100),        # Yellow
    'gallbladder': (255, 200, 100),    # Gold
    'adrenal_gland_right': (255, 100, 255),  # Pink
    'adrenal_gland_left': (235, 80, 235),    # Dark Pink
}

def load_nifti(file_path: Path) -> np.ndarray:
    """Load NIfTI file"""
    nii = nib.load(str(file_path))
    return nii.get_fdata()

def create_organ_overlay(ct_slice: np.ndarray, organ_masks: Dict[str, np.ndarray], alpha: float = 0.4) -> np.ndarray:
    """
    Create color-coded organ overlay
    
    Args:
        ct_slice: 2D CT slice (grayscale)
        organ_masks: Dictionary of organ name -> 2D mask
        alpha: Transparency (0-1)
    
    Returns:
        RGB overlay image
    """
    # Normalize CT to 0-255
    ct_norm = cv2.normalize(ct_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert to RGB
    overlay = cv2.cvtColor(ct_norm, cv2.COLOR_GRAY2RGB)
    
    # Add each organ with its color
    for organ_name, mask in organ_masks.items():
        if organ_name in ORGAN_COLORS:
            color = ORGAN_COLORS[organ_name]
            organ_pixels = mask > 0
            if organ_pixels.sum() > 0:
                overlay[organ_pixels] = (
                    overlay[organ_pixels] * (1 - alpha) + 
                    np.array(color) * alpha
                ).astype(np.uint8)
    
    return overlay

def analyze_organ_statistics(ct_volume: np.ndarray, organ_mask: np.ndarray, organ_name: str) -> Dict:
    """
    Analyze organ-specific statistics
    
    Returns baseline for anomaly detection
    """
    organ_voxels = ct_volume[organ_mask > 0]
    
    if len(organ_voxels) == 0:
        return None
    
    stats = {
        'name': organ_name,
        'volume_mm3': np.sum(organ_mask),  # Approximation
        'mean_hu': float(np.mean(organ_voxels)),
        'std_hu': float(np.std(organ_voxels)),
        'median_hu': float(np.median(organ_voxels)),
        'min_hu': float(np.min(organ_voxels)),
        'max_hu': float(np.max(organ_voxels)),
        'q25_hu': float(np.percentile(organ_voxels, 25)),
        'q75_hu': float(np.percentile(organ_voxels, 75)),
    }
    
    return stats

def detect_organ_anomalies(ct_slice: np.ndarray, organ_mask: np.ndarray, baseline: Dict) -> np.ndarray:
    """
    Detect anomalous regions within an organ
    
    Uses z-score based outlier detection
    """
    if baseline is None:
        return np.zeros_like(organ_mask)
    
    # Calculate z-scores
    z_scores = np.zeros_like(ct_slice)
    organ_pixels = organ_mask > 0
    z_scores[organ_pixels] = (ct_slice[organ_pixels] - baseline['mean_hu']) / (baseline['std_hu'] + 1e-6)
    
    # Detect high-intensity anomalies (e.g., enhanced tumors)
    high_anomalies = (z_scores > 3.0) & organ_pixels
    
    # Detect low-intensity anomalies (e.g., necrosis, cysts)
    low_anomalies = (z_scores < -3.0) & organ_pixels
    
    # Combine
    anomalies = high_anomalies | low_anomalies
    
    # Clean up small artifacts (morphological opening)
    kernel = np.ones((3, 3), np.uint8)
    anomalies = cv2.morphologyEx(anomalies.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    return anomalies

def visualize_hierarchical_analysis(
    ct_slice: np.ndarray,
    organ_masks: Dict[str, np.ndarray],
    organ_stats: Dict[str, Dict],
    slice_idx: int,
    output_dir: Path
):
    """
    Create comprehensive hierarchical visualization
    
    Layout:
    ┌──────────────┬──────────────┬──────────────┐
    │  Original CT │ Organ Overlay│ Anomaly Map  │
    └──────────────┴──────────────┴──────────────┘
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original CT
    ax = axes[0]
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    ax.set_title('Original CT', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 2. Organ Overlay
    ax = axes[1]
    organ_overlay = create_organ_overlay(ct_slice, organ_masks, alpha=0.5)
    ax.imshow(organ_overlay)
    ax.set_title('Organ Segmentation', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = []
    for organ_name in organ_masks.keys():
        if organ_name in ORGAN_COLORS and organ_masks[organ_name].sum() > 0:
            color = np.array(ORGAN_COLORS[organ_name]) / 255.0
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color, markersize=10,
                                            label=organ_name.replace('_', ' ').title()))
    
    if legend_elements:
        ax.legend(handles=legend_elements[:5], loc='upper right', fontsize=8)  # Show first 5
    
    # 3. Anomaly Detection
    ax = axes[2]
    anomaly_map = np.zeros_like(ct_slice)
    
    for organ_name, mask in organ_masks.items():
        if mask.sum() > 0 and organ_name in organ_stats:
            anomalies = detect_organ_anomalies(ct_slice, mask, organ_stats[organ_name])
            anomaly_map = np.maximum(anomaly_map, anomalies)
    
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    ax.imshow(anomaly_map, alpha=0.7, cmap='hot')
    ax.set_title('Anomaly Detection', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle(f'Hierarchical CT Analysis - Slice {slice_idx}', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'hierarchical_analysis_slice_{slice_idx:03d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    """Main visualization script"""
    
    print("\n" + "="*70)
    print("Hierarchical CT Visualization")
    print("="*70 + "\n")
    
    # Load CT volume
    print("Loading CT volume...")
    ct_path = Path("CTdata/nifti/inha_ct_arterial.nii.gz")
    ct_volume = load_nifti(ct_path)
    print(f"  CT shape: {ct_volume.shape}")
    
    # Load individual organ segmentations
    print("\nLoading organ segmentations...")
    seg_dir = Path("CTdata/segmentation")
    
    # Priority organs for cancer detection
    priority_organs = [
        'liver', 'kidney_left', 'kidney_right', 'spleen', 'pancreas',
        'colon', 'stomach', 'small_bowel', 'gallbladder'
    ]
    
    # Load all organ files
    organ_masks_3d = {}
    organ_files = list(seg_dir.glob("*.nii.gz"))
    
    print(f"  Found {len(organ_files)} organ files")
    
    for organ_file in organ_files:
        organ_name = organ_file.stem.replace('.nii', '')
        
        # Only load priority organs
        if organ_name in priority_organs:
            mask_3d = load_nifti(organ_file)
            
            # Ensure same shape as CT
            if mask_3d.shape == ct_volume.shape:
                organ_masks_3d[organ_name] = mask_3d.astype(np.uint8)
                print(f"    [OK] {organ_name}: {np.sum(mask_3d > 0)} voxels")
    
    print(f"\n  Loaded {len(organ_masks_3d)} priority organs")
    
    # Analyze each organ (3D)
    print("\nAnalyzing organ statistics...")
    organ_stats = {}
    
    for organ_name, mask_3d in organ_masks_3d.items():
        stats = analyze_organ_statistics(ct_volume, mask_3d, organ_name)
        
        if stats:
            organ_stats[organ_name] = stats
            print(f"  {organ_name}: {stats['volume_mm3']} voxels, "
                  f"HU = {stats['mean_hu']:.1f} ± {stats['std_hu']:.1f}")
    
    # Visualize key slices
    print("\nGenerating visualizations...")
    output_dir = Path("CTdata/visualizations/hierarchical")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Select representative slices
    n_slices = ct_volume.shape[0]
    slice_indices = [
        n_slices // 4,      # Upper abdomen
        n_slices // 2,      # Mid abdomen
        3 * n_slices // 4,  # Lower abdomen
    ]
    
    for slice_idx in slice_indices:
        print(f"  Processing slice {slice_idx}...")
        
        ct_slice = ct_volume[slice_idx, :, :]
        
        # Extract organ masks for this slice
        organs_2d = {}
        for organ_name, mask_3d in organ_masks_3d.items():
            mask_2d = mask_3d[slice_idx, :, :].astype(np.uint8)
            if mask_2d.sum() > 100:  # Only include if substantial
                organs_2d[organ_name] = mask_2d
        
        # Visualize
        output_file = visualize_hierarchical_analysis(
            ct_slice,
            organs_2d,
            organ_stats,
            slice_idx,
            output_dir
        )
        
        print(f"    Saved: {output_file.name}")
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total organs detected: {len(organ_stats)}")
    print("\nNext: Review visualizations for anomalies")

if __name__ == '__main__':
    main()
