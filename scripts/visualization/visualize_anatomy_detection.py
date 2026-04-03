"""
Visualize Anatomy-Based Detection with TotalSegmentator
Uses organ segmentation to constrain detection to body regions
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.medical_imaging.detection.anatomy_based_detector import quick_detect_colon_tumors


def visualize_anatomy_based_results(
    ct_volume,
    lesions,
    output_path="outputs/anatomy_based_detection.png",
    slice_idx=None
):
    """Create visualization of anatomy-based detection"""
    
    # Auto-select slice with lesions
    if slice_idx is None and lesions:
        # Get middle z of all lesions
        z_coords = [lesion.centroid[2] for lesion in lesions]
        slice_idx = int(np.median(z_coords))
    elif slice_idx is None:
        slice_idx = ct_volume.shape[2] // 2
    
    slice_idx = max(0, min(slice_idx, ct_volume.shape[2] - 1))
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Get CT slice
    ct_slice = ct_volume[:, :, slice_idx]
    ct_display = np.clip(ct_slice, -160, 240)
    ct_display = (ct_display + 160) / 400
    
    # 1. Original CT
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ct_display.T, cmap='gray', origin='lower')
    ax1.set_title(f'Original CT Slice (z={slice_idx})', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. All lesions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_display.T, cmap='gray', origin='lower')
    ax2.set_title(f'Detected Lesions (n={len(lesions)})', fontsize=14, fontweight='bold')
    
    # Draw lesions on this slice
    lesions_on_slice = [l for l in lesions if abs(l.centroid[0] - slice_idx) < 3]
    for lesion in lesions_on_slice:
        z, y, x = lesion.centroid
        # Get bounding box size
        bbox = lesion.bbox
        width = bbox['x_max'] - bbox['x_min']
        height = bbox['y_max'] - bbox['y_min']
        
        # Color by classification
        if lesion.classification == 'tumor_likely':
            color = 'red'
            linewidth = 3
        elif lesion.classification == 'tumor_possible':
            color = 'orange'
            linewidth = 2
        else:
            color = 'yellow'
            linewidth = 1
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox['x_min'], bbox['y_min']), width, height,
                        fill=False, edgecolor=color, linewidth=linewidth, alpha=0.8)
        ax2.add_patch(rect)
    ax2.axis('off')
    
    # 3. High-confidence only
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(ct_display.T, cmap='gray', origin='lower')
    
    high_conf = [l for l in lesions if l.confidence and l.confidence > 0.7]
    ax3.set_title(f'High Confidence (>70%, n={len(high_conf)})', fontsize=14, fontweight='bold')
    
    for lesion in [l for l in high_conf if abs(l.centroid[0] - slice_idx) < 3]:
        z, y, x = lesion.centroid
        bbox = lesion.bbox
        width = bbox['x_max'] - bbox['x_min']
        height = bbox['y_max'] - bbox['y_min']
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox['x_min'], bbox['y_min']), width, height,
                        fill=False, edgecolor='red', linewidth=3, alpha=0.9)
        ax3.add_patch(rect)
        
        # Label
        ax3.text(x, y, f"{lesion.confidence:.0%}",
                color='red', fontsize=10, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax3.axis('off')
    
    # 4. 3D scatter
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    if lesions:
        positions = np.array([l.centroid for l in lesions])
        confidences = np.array([l.confidence if l.confidence else 0.5 for l in lesions])
        
        scatter = ax4.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                            c=confidences, cmap='hot', s=100, alpha=0.7)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z (Slice)')
        ax4.set_title('Lesion 3D Distribution', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Confidence', shrink=0.6)
    
    # 5. Statistics
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    tumor_likely = len([l for l in lesions if l.classification == 'tumor_likely'])
    tumor_possible = len([l for l in lesions if l.classification == 'tumor_possible'])
    
    if lesions:
        stats_text = f"""
    Anatomy-Based Detection
    {'='*40}
    Total Lesions: {len(lesions)}
    Tumor Likely: {tumor_likely}
    Tumor Possible: {tumor_possible}
    
    Volume Range: {min([l.volume_voxels for l in lesions])}-{max([l.volume_voxels for l in lesions])} voxels
    HU Range: {min([l.mean_hu for l in lesions]):.1f}-{max([l.mean_hu for l in lesions]):.1f}
    
    Detections in colon organ
    Constrained to body region
    """
    else:
        stats_text = "\n\nNo lesions detected"
    
    ax5.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 6. HU distribution
    ax6 = fig.add_subplot(gs[1, 2])
    if lesions:
        hu_values = [l.mean_hu for l in lesions]
        ax6.hist(hu_values, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax6.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Normal Colon Wall')
        ax6.set_xlabel('Mean HU Value', fontsize=12)
        ax6.set_ylabel('Number of Lesions', fontsize=12)
        ax6.set_title('HU Distribution', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Anatomy-Based Detection - Organ-Constrained Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[+] Visualization saved to: {output_path}")
    
    return output_path


def main():
    """Run anatomy-based detection on colon_001"""
    
    case_name = "colon_001"
    image_path = f"data/medical_decathlon/Task10_Colon/imagesTr/{case_name}.nii.gz"
    output_path = f"outputs/anatomy_based_{case_name}.png"
    
    print(f"\n[*] Anatomy-Based Detection on {case_name}")
    print("="*60)
    print("[*] Using TotalSegmentator for organ segmentation")
    print("[*] Detecting anomalies within colon region only")
    print()
    
    # Load volume
    print(f"[*] Loading: {image_path}")
    nii = nib.load(image_path)
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    print(f"   Volume: {volume.shape}, Spacing: {spacing}")
    
    # Run detection
    print(f"\n[*] Running anatomy-based detection...")
    print("   This may take a few minutes (organ segmentation + analysis)")
    
    try:
        lesions = quick_detect_colon_tumors(
            ct_volume=volume,
            spacing=spacing,
            device="gpu"
        )
        
        print(f"\n[*] Detection Results:")
        print(f"   Total lesions: {len(lesions)}")
        
        if lesions:
            tumor_likely = [l for l in lesions if l.classification == 'tumor_likely']
            tumor_possible = [l for l in lesions if l.classification == 'tumor_possible']
            
            print(f"   Tumor likely: {len(tumor_likely)}")
            print(f"   Tumor possible: {len(tumor_possible)}")
            
            print(f"\n   Top candidates:")
            for i, lesion in enumerate(sorted(lesions, key=lambda x: x.confidence or 0, reverse=True)[:5]):
                conf_str = f"{lesion.confidence:.2f}" if lesion.confidence else "N/A"
                print(f"     #{i+1}: {lesion.volume_voxels} voxels, HU {lesion.mean_hu:.1f}, " +
                     f"class: {lesion.classification}, conf: {conf_str}")
        
        # Visualize
        print(f"\n[*] Creating visualization...")
        viz_path = visualize_anatomy_based_results(
            ct_volume=volume,
            lesions=lesions,
            output_path=output_path
        )
        
        print(f"\n[+] Done! Visualization at: {viz_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Detection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
