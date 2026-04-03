"""
Universal CT Visualization with Automatic Orientation Detection

Supports axial, coronal, and sagittal orientations with automatic detection
and proper axial slice generation for clinical visualization.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches
from typing import Tuple, Dict


def detect_volume_orientation(volume_shape: Tuple[int, int, int], 
                               voxel_spacing: Tuple[float, float, float]) -> Tuple[str, int]:
    """
    Detect whether volume is axial, coronal, or sagittal based on dimensions and spacing
    
    Args:
        volume_shape: Shape of the volume (D1, D2, D3)
        voxel_spacing: Voxel spacing in mm (S1, S2, S3)
    
    Returns:
        Tuple of (orientation_name, slice_axis_for_axial_view)
    """
    print("\n" + "="*80)
    print("ORIENTATION DETECTION")
    print("="*80)
    print(f"Volume shape: {volume_shape}")
    print(f"Voxel spacing: {voxel_spacing}")
    
    # If any dimension has spacing > 100mm, likely coronal or sagittal
    # Standard axial CT has 1-5mm slice thickness
    max_spacing = max(voxel_spacing)
    min_spacing = min(voxel_spacing)
    
    print(f"Max spacing: {max_spacing:.2f}mm")
    print(f"Min spacing: {min_spacing:.2f}mm")
    
    if max_spacing > 100:
        # Find which axis has the large spacing
        spacing_idx = voxel_spacing.index(max_spacing)
        print(f"Large spacing detected on axis {spacing_idx}")
        
        # Coronal: large spacing in anterior-posterior (axis 0)
        # Need to slice along axis 1 (superior-inferior) for axial views
        if spacing_idx == 0:
            print("[!] Detected: CORONAL orientation")
            print("  -> Will slice along axis 1 for axial views")
            return 'coronal', 1
        
        # Sagittal: large spacing in left-right (axis 2)
        # Need to slice along axis 1 for axial views
        elif spacing_idx == 2:
            print("[!] Detected: SAGITTAL orientation")
            print("  -> Will slice along axis 1 for axial views")
            return 'sagittal', 1
        
        else:
            print("[?] Detected: UNKNOWN orientation (large spacing on axis 1)")
            print("  -> Will use axis 0 by default")
            return 'unknown', 0
    
    else:
        # Standard axial acquisition
        print("[!] Detected: AXIAL orientation")
        print("  -> Will slice along axis 0")
        return 'axial', 0


def extract_slice(volume: np.ndarray, slice_axis: int, slice_idx: int) -> np.ndarray:
    """
    Extract a 2D slice from 3D volume along specified axis
    
    Args:
        volume: 3D numpy array
        slice_axis: Axis to slice along (0, 1, or 2)
        slice_idx: Index of the slice
    
    Returns:
        2D numpy array
    """
    if slice_axis == 0:
        return volume[slice_idx, :, :]
    elif slice_axis == 1:
        return volume[:, slice_idx, :]
    else:  # axis == 2
        return volume[:, :, slice_idx]


def visualize_multi_organ_universal(volume_path: str, seg_path: str, output_dir: str):
    """
    Universal multi-organ visualization with automatic orientation detection
    
    Args:
        volume_path: Path to CT volume NIfTI file
        seg_path: Path to segmentation NIfTI file
        output_dir: Directory to save outputs
    """
    print("\n" + "="*80)
    print("UNIVERSAL MULTI-ORGAN CT VISUALIZATION")
    print("="*80)
    
    # Load volume
    print(f"\nLoading volume: {volume_path}")
    volume_nifti = nib.load(volume_path)
    volume = volume_nifti.get_fdata().astype(np.float32)
    voxel_spacing = volume_nifti.header.get_zooms()
    
    print(f"Volume shape: {volume.shape}")
    print(f"Voxel spacing: {voxel_spacing}")
    
    # Load segmentation
    print(f"\nLoading segmentation: {seg_path}")
    seg_nifti = nib.load(seg_path)
    segmentation = seg_nifti.get_fdata().astype(np.uint8)
    
    # Detect orientation
    orientation, slice_axis = detect_volume_orientation(volume.shape, voxel_spacing)
    
    # TotalSegmentator organ labels
    organs = {
        'LIVER': 16,
        'KIDNEY_LEFT': 3,
        'KIDNEY_RIGHT': 2,
        'COLON': 14,
        'STOMACH': 17,
        'SPLEEN': 1,
    }
    
    # Count voxels
    print("\n" + "="*80)
    print("ORGAN STATISTICS")
    print("="*80)
    for organ_name, label in organs.items():
        organ_mask = (segmentation == label)
        voxels = np.sum(organ_mask)
        volume_cm3 = voxels / 1000
        print(f"  {organ_name:15s}: {voxels:,} voxels ({volume_cm3:.1f} cm³)")
    
    # Get middle slice
    n_slices = volume.shape[slice_axis]
    middle_slice = n_slices // 2
    
    print(f"\n{orientation.upper()} orientation detected")
    print(f"Total slices along axis {slice_axis}: {n_slices}")
    print(f"Using middle slice: {middle_slice}")
    
    # Extract slices
    ct_slice = extract_slice(volume, slice_axis, middle_slice)
    seg_slice = extract_slice(segmentation, slice_axis, middle_slice)
    
    print(f"Extracted slice shape: {ct_slice.shape}")
    
    # Color mapping
    colors = {
        'LIVER': [0.8, 0.4, 0.0],
        'KIDNEY_LEFT': [1.0, 0.0, 0.0],
        'KIDNEY_RIGHT': [1.0, 0.0, 0.0],
        'COLON': [0.0, 0.8, 0.0],
        'STOMACH': [0.0, 0.5, 1.0],
        'SPLEEN': [0.8, 0.0, 0.8],
    }
    
    # === Visualization 1: Individual Organs ===
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION 1: Individual Organs")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Multi-Organ CT Analysis - {orientation.upper()} → Axial View (Slice {middle_slice}/{n_slices})', 
                 fontsize=16, fontweight='bold')
    
    for idx, (organ_name, label) in enumerate(organs.items()):
        ax = axes[idx // 3, idx % 3]
        
        # Show CT
        ax.imshow(ct_slice, cmap='gray', aspect='auto', origin='lower')
        
        # Overlay organ
        organ_mask = (seg_slice == label).astype(np.uint8)
        
        if np.sum(organ_mask) > 0:
            overlay = np.zeros((*organ_mask.shape, 4))
            overlay[organ_mask > 0] = [*colors[organ_name], 0.6]
            ax.imshow(overlay, aspect='auto', origin='lower')
            
            voxels = np.sum(organ_mask)
            ax.set_title(f'{organ_name}\n{voxels:,} pixels in this slice', 
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{organ_name}\nNot visible in this slice', 
                        fontsize=12)
        
        ax.axis('off')
    
    plt.tight_layout()
    
    output_path1 = Path(output_dir) / 'multi_organ_corrected.png'
    output_path1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path1}")
    plt.close()
    
    # === Visualization 2: All Organs Overlay ===
    print("\nGENERATING VISUALIZATION 2: All Organs Overlay")
    
    fig2, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig2.suptitle(f'All Organs Overlay - {orientation.upper()} → Axial View (Slice {middle_slice}/{n_slices})', 
                  fontsize=16, fontweight='bold')
    
    ax.imshow(ct_slice, cmap='gray', aspect='auto', origin='lower')
    
    for organ_name, label in organs.items():
        organ_mask = (seg_slice == label).astype(np.uint8)
        if np.sum(organ_mask) > 0:
            overlay = np.zeros((*organ_mask.shape, 4))
            overlay[organ_mask > 0] = [*colors[organ_name], 0.5]
            ax.imshow(overlay, aspect='auto', origin='lower')
    
    patches = [mpatches.Patch(color=colors[name], alpha=0.5, label=name) 
               for name in organs.keys()]
    ax.legend(handles=patches, loc='upper right', fontsize=11)
    ax.axis('off')
    
    plt.tight_layout()
    
    output_path2 = Path(output_dir) / 'all_organs_corrected.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path2}")
    plt.close()
    
    # === Visualization 3: Progression ===
    print("\nGENERATING VISUALIZATION 3: Organ Progression")
    
    slices_to_show = [
        n_slices // 4,
        n_slices // 2,
        3 * n_slices // 4
    ]
    
    fig3, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig3.suptitle(f'CT Progression - {orientation.upper()} → Axial Slices', 
                  fontsize=16, fontweight='bold')
    
    colors_simple = {
        14: [0.0, 1.0, 0.0],  # Colon
        16: [1.0, 0.5, 0.0],  # Liver
        3: [1.0, 0.0, 0.0],   # Kidney L
        2: [1.0, 0.0, 0.0],   # Kidney R
        17: [0.0, 0.5, 1.0],  # Stomach
    }
    
    for row, slice_idx in enumerate(slices_to_show):
        # Get slices
        ct = extract_slice(volume, slice_axis, slice_idx)
        seg = extract_slice(segmentation, slice_axis, slice_idx)
        
        # Original
        ax1 = axes[row, 0]
        ax1.imshow(ct, cmap='gray', aspect='auto', origin='lower')
        ax1.set_title(f'Slice {slice_idx}\nOriginal CT', fontsize=10)
        ax1.axis('off')
        
        # With overlay
        ax2 = axes[row, 1]
        ax2.imshow(ct, cmap='gray', aspect='auto', origin='lower')
        for label, color in colors_simple.items():
            organ_mask = (seg == label)
            if np.sum(organ_mask) > 0:
                overlay = np.zeros((*organ_mask.shape, 4))
                overlay[organ_mask > 0] = [*color, 0.5]
                ax2.imshow(overlay, aspect='auto', origin='lower')
        ax2.set_title(f'Slice {slice_idx}\nOrgans Overlay', fontsize=10)
        ax2.axis('off')
        
        # Organs only
        ax3 = axes[row, 2]
        organs_only = np.zeros((*seg.shape, 3))
        for label, color in colors_simple.items():
            organ_mask = (seg == label)
            if np.sum(organ_mask) > 0:
                organs_only[organ_mask > 0] = color
        ax3.imshow(organs_only, aspect='auto', origin='lower')
        ax3.set_title(f'Slice {slice_idx}\nOrgans Only', fontsize=10)
        ax3.axis('off')
    
    organ_names = {14: 'Colon', 16: 'Liver', 3: 'Kidney L', 2: 'Kidney R', 17: 'Stomach'}
    patches = [mpatches.Patch(color=colors_simple[label], label=organ_names[label]) 
               for label in colors_simple.keys()]
    fig3.legend(handles=patches, loc='lower center', ncol=5, fontsize=10)
    
    plt.tight_layout()
    
    output_path3 = Path(output_dir) / 'organ_progression_corrected.png'
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path3}")
    plt.close()
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Generated 3 corrected visualizations:")
    print(f"  1. {output_path1.name}")
    print(f"  2. {output_path2.name}")
    print(f"  3. {output_path3.name}")
    
    return str(output_path1), str(output_path2), str(output_path3)


if __name__ == "__main__":
    # Default paths
    volume_path = "outputs/ct_pipeline_test/reconstructed_volume.nii.gz"
    seg_path = "outputs/ct_pipeline_test/segmentation_resampled.nii.gz"
    output_dir = "outputs/medical_decathlon_analysis"
    
    print("Starting Universal CT Visualization...")
    paths = visualize_multi_organ_universal(volume_path, seg_path, output_dir)
    
    print(f"\n[OK] All visualizations generated successfully!")
    print(f"\nOutput files:")
    for i, p in enumerate(paths, 1):
        print(f"  {i}. {p}")
