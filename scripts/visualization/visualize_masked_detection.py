"""
Visualize Anatomy-Based Detection with Red Mask Overlay
Shows lesion candidates with filled red masks
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


def visualize_with_mask_overlay(
    ct_volume,
    lesions,
    output_path="outputs/anatomy_masked_detection.png",
    slice_idx=None
):
    """Create visualization with red mask overlay"""
    
    # Auto-select slice with most lesions
    if slice_idx is None and lesions:
        # Get all z coordinates where lesions exist
        z_ranges = []
        for lesion in lesions:
            z_mid = (lesion.bbox['z_min'] + lesion.bbox['z_max']) // 2
            z_ranges.append(z_mid)
        
        if z_ranges:
            slice_idx = int(np.median(z_ranges))
            print(f"[DEBUG] Auto-selected slice {slice_idx} (median of lesion positions)")
        else:
            slice_idx = ct_volume.shape[2] // 2
    elif slice_idx is None:
        slice_idx = ct_volume.shape[2] // 2
    
    slice_idx = max(0, min(slice_idx, ct_volume.shape[2] - 1))
    
    print(f"[DEBUG] Visualizing slice {slice_idx}/{ct_volume.shape[2]}")
    print(f"[DEBUG] Lesion z-ranges:")
    for i, lesion in enumerate(lesions[:5]):  # Show first 5
        print(f"  Lesion {i+1}: z={lesion.bbox['z_min']}-{lesion.bbox['z_max']}, center={lesion.centroid[0]}")
    
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
    
    # 2. CT with red filled mask overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_display.T, cmap='gray', origin='lower')
    
    # Draw filled red regions for each lesion on this slice
    lesion_count = 0
    for lesion in lesions:
        z_min = lesion.bbox['z_min']
        z_max = lesion.bbox['z_max']
        
        # Check if lesion is on this slice
        if z_min <= slice_idx <= z_max:
            # Get the 2D slice of the lesion mask
            try:
                lesion_slice = lesion.mask[:, :, slice_idx]
                if lesion_slice.sum() > 0:
                    # Create red overlay for this lesion
                    from matplotlib.patches import Polygon
                    from skimage import measure
                    
                    # Find contours of the lesion
                    contours = measure.find_contours(lesion_slice.T, 0.5)
                    for contour in contours:
                        # Fill the contour with semi-transparent red
                        poly = Polygon(contour, facecolor='red', alpha=0.4, edgecolor='red', linewidth=2)
                        ax2.add_patch(poly)
                    lesion_count += 1
            except Exception as e:
                # If indexing fails, draw bbox instead
                x_min, x_max = lesion.bbox['x_min'], lesion.bbox['x_max']
                y_min, y_max = lesion.bbox['y_min'], lesion.bbox['y_max']
                from matplotlib.patches import Rectangle
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               fill=True, facecolor='red', alpha=0.3, 
                               edgecolor='red', linewidth=2)
                ax2.add_patch(rect)
                lesion_count += 1
    
    ax2.set_title(f'All Lesions - Red Mask ({lesion_count} on this slice)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Tumor-likely only with red mask
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(ct_display.T, cmap='gray', origin='lower')
    
    tumor_lesions = [l for l in lesions if 'tumor' in (l.classification or '').lower()]
    tumor_count = 0
    
    for lesion in tumor_lesions:
        z_min = lesion.bbox['z_min']
        z_max = lesion.bbox['z_max']
        
        if z_min <= slice_idx <= z_max:
            try:
                lesion_slice = lesion.mask[:, :, slice_idx]
                if lesion_slice.sum() > 0:
                    from matplotlib.patches import Polygon
                    from skimage import measure
                    contours = measure.find_contours(lesion_slice.T, 0.5)
                    for contour in contours:
                        poly = Polygon(contour, facecolor='red', alpha=0.6, edgecolor='yellow', linewidth=3)
                        ax3.add_patch(poly)
                    tumor_count += 1
            except:
                x_min, x_max = lesion.bbox['x_min'], lesion.bbox['x_max']
                y_min, y_max = lesion.bbox['y_min'], lesion.bbox['y_max']
                from matplotlib.patches import Rectangle
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               fill=True, facecolor='red', alpha=0.5,
                               edgecolor='yellow', linewidth=3)
                ax3.add_patch(rect)
                tumor_count += 1
    
    ax3.set_title(f'Tumor Only - Red Mask ({tumor_count} on this slice)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 4. All lesion bounding boxes (easier to see)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(ct_display.T, cmap='gray', origin='lower')
    ax4.set_title('Lesion Bounding Boxes', fontsize=14, fontweight='bold')
    
    for lesion in lesions:
        z_min = lesion.bbox['z_min']
        z_max = lesion.bbox['z_max']
        
        if z_min <= slice_idx <= z_max:
            x_min, x_max = lesion.bbox['x_min'], lesion.bbox['x_max']
            y_min, y_max = lesion.bbox['y_min'], lesion.bbox['y_max']
            
            color = 'red' if 'tumor' in (lesion.classification or '').lower() else 'orange'
            from matplotlib.patches import Rectangle
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, edgecolor=color, linewidth=2)
            ax4.add_patch(rect)
            
            # Add label
            ax4.text(x_min, y_min - 3, f"{lesion.mean_hu:.0f}HU",
                    color=color, fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax4.axis('off')
    
    # 5. Statistics with lesion details
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    tumor_likely = len([l for l in lesions if l.classification == 'tumor_likely'])
    tumor_probable = len([l for l in lesions if l.classification == 'tumor_probable'])
    calcification = len([l for l in lesions if l.classification == 'calcification'])
    
    # Count lesions on this slice
    lesions_on_slice = sum(1 for l in lesions if l.bbox['z_min'] <= slice_idx <= l.bbox['z_max'])
    
    stats_text = f"""
    Red Mask Analysis
    {'='*40}
    Total Lesions: {len(lesions)}
    - Tumor Likely: {tumor_likely}
    - Tumor Probable: {tumor_probable}
    - Calcification: {calcification}
    
    Slice {slice_idx}/{ct_volume.shape[2]}
    Lesions on this slice: {lesions_on_slice}
    
    Red mask = detected regions
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.5))
    
    # 6. HU distribution of lesions
    ax6 = fig.add_subplot(gs[1, 2])
    if lesions:
        hu_values = [l.mean_hu for l in lesions]
        colors = ['red' if 'tumor' in (l.classification or '').lower() else 'orange' for l in lesions]
        ax6.scatter(range(len(lesions)), hu_values, c=colors, s=100, alpha=0.7, edgecolors='black')
        ax6.axhline(y=50, color='blue', linestyle='--', linewidth=2, label='Normal Colon Wall')
        ax6.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Tumor Threshold')
        ax6.set_xlabel('Lesion Index', fontsize=12)
        ax6.set_ylabel('Mean HU Value', fontsize=12)
        ax6.set_title('HU Values per Lesion', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Anatomy-Based Detection - Red Mask Overlay', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[+] Masked visualization saved to: {output_path}")
    
    return output_path


def main():
    """Run anatomy-based detection with red mask overlay"""
    
    case_name = "colon_001"
    image_path = f"data/medical_decathlon/Task10_Colon/imagesTr/{case_name}.nii.gz"
    output_path = f"outputs/anatomy_masked_{case_name}.png"
    
    print(f"\n[*] Creating Red Mask Visualization for {case_name}")
    print("="*60)
    
    # Load volume
    print(f"[*] Loading: {image_path}")
    nii = nib.load(image_path)
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    print(f"   Volume: {volume.shape}")
    
    # Run detection
    print(f"\n[*] Running anatomy-based detection with TotalSegmentator...")
    
    try:
        lesions = quick_detect_colon_tumors(
            ct_volume=volume,
            spacing=spacing,
            device="gpu"
        )
        
        print(f"\n[*] Detection complete: {len(lesions)} lesions")
        
        if lesions:
            tumor_count = len([l for l in lesions if 'tumor' in (l.classification or '').lower()])
            print(f"   Tumor candidates: {tumor_count}")
        
        # Create masked visualization
        print(f"\n[*] Creating red mask overlay...")
        viz_path = visualize_with_mask_overlay(
            ct_volume=volume,
            lesions=lesions,
            output_path=output_path
        )
        
        print(f"\n[+] Done! Red mask visualization at: {viz_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
