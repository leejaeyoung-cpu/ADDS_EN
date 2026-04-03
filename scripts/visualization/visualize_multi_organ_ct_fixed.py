"""
Fixed Multi-Organ CT Visualization with Correct Orientation
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

def visualize_multi_organ_ct_fixed():
    """Visualize CT with correct orientation"""
    
    print("="*80)
    print("FIXED MULTI-ORGAN CT VISUALIZATION")
    print("="*80)
    
    # Load volume
    volume_path = Path("outputs/ct_pipeline_test/reconstructed_volume.nii.gz")
    print(f"Loading volume: {volume_path}")
    volume_nifti = nib.load(volume_path)
    volume = volume_nifti.get_fdata().astype(np.float32)
    print(f"Volume shape: {volume.shape}")
    
    # Load segmentation
    seg_path = Path("outputs/ct_pipeline_test/segmentation_resampled.nii.gz")
    print(f"Loading segmentation: {seg_path}")
    seg_nifti = nib.load(seg_path)
    segmentation = seg_nifti.get_fdata().astype(np.uint8)
    
    # TotalSegmentator labels (key organs)
    organs = {
        'liver': 16,
        'kidney_left': 3,
        'kidney_right': 2,
        'colon': 14,
        'stomach': 17,
        'spleen': 1,
    }
    
    # Count voxels for each organ
    print("\nOrgan Statistics:")
    for organ_name, label in organs.items():
        organ_mask = (segmentation == label)
        voxels = np.sum(organ_mask)
        volume_cm3 = voxels / 1000
        print(f"  {organ_name:15s}: {voxels:,} voxels ({volume_cm3:.1f} cm³)")
    
    # Select middle slice - FIX: Use correct axis
    depth = volume.shape[0]
    slice_idx = depth // 2
    
    # Get slice with CORRECT orientation
    ct_slice = volume[slice_idx, :, :].T  # Transpose to fix orientation
    seg_slice = segmentation[slice_idx, :, :].T
    
    # Create visualization with multiple organs
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Multi-Organ CT Analysis - Axial Slice {slice_idx}/{depth}', 
                 fontsize=16, fontweight='bold')
    
    # Define colors for organs (RGB)
    colors = {
        'liver': [0.8, 0.4, 0.0],      # Orange
        'kidney_left': [1.0, 0.0, 0.0], # Red
        'kidney_right': [1.0, 0.0, 0.0],# Red
        'colon': [0.0, 0.8, 0.0],       # Green
        'stomach': [0.0, 0.5, 1.0],     # Blue
        'spleen': [0.8, 0.0, 0.8],      # Purple
    }
    
    # Individual organ views
    for idx, (organ_name, label) in enumerate(organs.items()):
        ax = axes[idx // 3, idx % 3]
        
        # Show CT with correct orientation
        ax.imshow(ct_slice, cmap='gray', origin='lower')
        
        # Overlay organ
        organ_slice_mask = (seg_slice == label).astype(np.uint8)
        
        if np.sum(organ_slice_mask) > 0:
            overlay = np.zeros((*organ_slice_mask.shape, 4))
            overlay[organ_slice_mask > 0] = [*colors[organ_name], 0.5]
            ax.imshow(overlay, origin='lower')
            
            voxels = np.sum(organ_slice_mask)
            ax.set_title(f'{organ_name.upper()}\n{voxels:,} pixels in this slice', 
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{organ_name.upper()}\nNot visible in this slice', 
                        fontsize=12)
        
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = Path("outputs/medical_decathlon_analysis/multi_organ_visualization_fixed.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved: {output_path}")
    
    # Create all-organs overlay
    fig2, ax = plt.subplots(1, 1, figsize=(12, 12))
    fig2.suptitle(f'All Organs Overlay - Axial Slice {slice_idx}/{depth}', 
                  fontsize=16, fontweight='bold')
    
    ax.imshow(ct_slice, cmap='gray', origin='lower')
    
    # Overlay all organs
    for organ_name, label in organs.items():
        organ_slice_mask = (seg_slice == label).astype(np.uint8)
        if np.sum(organ_slice_mask) > 0:
            overlay = np.zeros((*organ_slice_mask.shape, 4))
            overlay[organ_slice_mask > 0] = [*colors[organ_name], 0.4]
            ax.imshow(overlay, origin='lower')
    
    # Create legend
    patches = [mpatches.Patch(color=colors[name], alpha=0.4, label=name.upper()) 
               for name in organs.keys()]
    ax.legend(handles=patches, loc='upper right', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    
    output_path2 = Path("outputs/medical_decathlon_analysis/all_organs_overlay_fixed.png")
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path2}")
    
    return output_path, output_path2


def create_organ_comparison_fixed():
    """Create side-by-side organ comparison with correct orientation"""
    
    volume_path = Path("outputs/ct_pipeline_test/reconstructed_volume.nii.gz")
    volume = nib.load(volume_path).get_fdata().astype(np.float32)
    
    seg_path = Path("outputs/ct_pipeline_test/segmentation_resampled.nii.gz")
    segmentation = nib.load(seg_path).get_fdata().astype(np.uint8)
    
    # Select 3 different slices
    depth = volume.shape[0]
    slices = [depth // 4, depth // 2, 3 * depth // 4]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('CT Progression - Lower to Upper Abdomen', 
                 fontsize=16, fontweight='bold')
    
    colors = {
        14: [0.0, 1.0, 0.0],  # Colon - Green
        16: [1.0, 0.5, 0.0],  # Liver - Orange
        3: [1.0, 0.0, 0.0],   # Kidney L - Red
        2: [1.0, 0.0, 0.0],   # Kidney R - Red
    }
    
    for row, slice_idx in enumerate(slices):
        # Original CT - FIXED orientation
        ax1 = axes[row, 0]
        ct_slice = volume[slice_idx, :, :].T
        ax1.imshow(ct_slice, cmap='gray', origin='lower')
        ax1.set_title(f'Slice {slice_idx}\nOriginal CT', fontsize=10)
        ax1.axis('off')
        
        # With organs
        ax2 = axes[row, 1]
        ax2.imshow(ct_slice, cmap='gray', origin='lower')
        seg_slice = segmentation[slice_idx, :, :].T
        for label, color in colors.items():
            organ_slice = (seg_slice == label)
            if np.sum(organ_slice) > 0:
                overlay = np.zeros((*organ_slice.shape, 4))
                overlay[organ_slice > 0] = [*color, 0.4]
                ax2.imshow(overlay, origin='lower')
        ax2.set_title(f'Slice {slice_idx}\nOrgans Overlay', fontsize=10)
        ax2.axis('off')
        
        # Organs only
        ax3 = axes[row, 2]
        organs_only = np.zeros((*seg_slice.shape, 3))
        for label, color in colors.items():
            organ_slice = (seg_slice == label)
            if np.sum(organ_slice) > 0:
                organs_only[organ_slice > 0] = color
        ax3.imshow(organs_only, origin='lower')
        ax3.set_title(f'Slice {slice_idx}\nOrgans Only', fontsize=10)
        ax3.axis('off')
    
    # Legend
    organ_names = {14: 'Colon', 16: 'Liver', 3: 'Kidney L', 2: 'Kidney R'}
    patches = [mpatches.Patch(color=colors[label], label=organ_names[label]) 
               for label in colors.keys()]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=11)
    
    plt.tight_layout()
    
    output_path = Path("outputs/medical_decathlon_analysis/organ_progression_fixed.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved progression: {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("\n1. Creating multi-organ visualization (FIXED)...")
    path1, path2 = visualize_multi_organ_ct_fixed()
    
    print("\n2. Creating organ progression (FIXED)...")
    path3 = create_organ_comparison_fixed()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {path1}")
    print(f"  2. {path2}")
    print(f"  3. {path3}")
