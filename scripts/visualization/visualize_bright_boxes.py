"""
Simplified visualization - use bounding boxes with bright colors
Since mask coordinates don't match volume dimensions, use bbox directly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from src.medical_imaging.detection.anatomy_based_detector import quick_detect_colon_tumors


def create_simple_bbox_visualization(ct_volume, lesions, output_path):
    """Create clear visualization with colorful bounding boxes"""
    
    # Find best slice - use x or y coordinates instead
    if lesions:
        # Try y coordinates
        y_coords = [(l.bbox['y_min'] + l.bbox['y_max']) // 2 for l in lesions]
        slice_idx = int(np.median(y_coords))
        # Clamp to valid range
        slice_idx = max(0, min(slice_idx, ct_volume.shape[1] // 2))  # Use middle-ish
    else:
        slice_idx = ct_volume.shape[1] // 2
    
    # Create 3 views: z=30 fixed slice
    slice_idx = 30
    
    fig  = plt.figure(figsize=(24, 8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # View 1: Axial (z-slice)
    ax1 = fig.add_subplot(gs[0, 0])
    ct_slice = ct_volume[:, :, slice_idx]
    ct_display = np.clip(ct_slice, -160, 240)
    ct_display = (ct_display + 160) / 400
    ax1.imshow(ct_display.T, cmap='gray', origin='lower')
    ax1.set_title(f'Axial View (z={slice_idx})', fontsize=16, fontweight='bold')
    
    # Draw lesions whose z-range includes this slice
    count = 0
    for lesion in lesions:
        if lesion.bbox['z_min'] <= slice_idx <= lesion.bbox['z_max']:
            x_min, x_max = lesion.bbox['x_min'], lesion.bbox['x_max']
            y_min, y_max = lesion.bbox['y_min'], lesion.bbox['y_max']
            
            color = 'red' if 'tumor' in (lesion.classification or '').lower() else 'cyan'
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=True, facecolor=color, alpha=0.4,
                           edgecolor=color, linewidth=3)
            ax1.add_patch(rect)
            
            # Label with HU value
            ax1.text(x_min + 2, y_min + 2, f"{int(lesion.mean_hu)} HU",
                    color='yellow', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2))
            count += 1
    
    ax1.text(10, 500, f'{count} lesions on this slice', 
            color='yellow', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='red', alpha=0.8, pad=5))
    ax1.axis('off')
    
    # View 2: All lesions list with boxes
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_display.T, cmap='gray', origin='lower')
    ax2.set_title(f'All {len(lesions)} Lesions - Bright Boxes', fontsize=16, fontweight='bold')
    
    # Draw all lesions (try different slice)
    slice2 = 35
    ct_slice2 = ct_volume[:, :, slice2]
    ct_display2 = np.clip(ct_slice2, -160, 240)
    ct_display2 = (ct_display2 + 160) / 400
    ax2.imshow(ct_display2.T, cmap='gray', origin='lower')
    
    for lesion in lesions:
        if lesion.bbox['z_min'] <= slice2 <= lesion.bbox['z_max']:
            x_min, x_max = lesion.bbox['x_min'], lesion.bbox['x_max']
            y_min, y_max = lesion.bbox['y_min'], lesion.bbox['y_max']
            
            # Bright neon colors
            if 'tumor' in (lesion.classification or '').lower():
                color = 'lime'
            elif lesion.classification == 'calcification':
                color = 'magenta'
            else:
                color = 'yellow'
            
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, edgecolor=color, linewidth=4)
            ax2.add_patch(rect)
    
    ax2.axis('off')
    
    # View 3: stats
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    ax3.set_title('Lesion Statistics', fontsize=16, fontweight='bold')
    
    tumor_lesions = [l for l in lesions if 'tumor' in (l.classification or '').lower()]
    calc_lesions = [l for l in lesions if l.classification == 'calcification']
    
    stats = f"""
DETECTION SUMMARY
{'='*50}

Total Lesions: {len(lesions)}

By Classification:
  • Tumor-related: {len(tumor_lesions)} (LIME boxes)
  • Calcification: {len(calc_lesions)} (MAGENTA boxes)
  • Other: {len(lesions) - len(tumor_lesions) - len(calc_lesions)}

Top 5 Lesions by HU:
"""
    
    sorted_lesions = sorted(lesions, key=lambda l: l.mean_hu, reverse=True)[:5]
    for i, l in enumerate(sorted_lesions):
        stats += f"\n {i+1}. {l.mean_hu:.0f} HU - {l.classification}"
        stats += f"\n    ({l.volume_voxels} voxels)"
    
    stats += f"\n\n{'='*50}\nViewing slices z=30 and z=35\n"
    
    ax3.text(0.1, 0.5, stats, fontsize=14, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Anatomy-Based Detection - Bright Bounding Boxes', 
                fontsize=18, fontweight='bold')
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[+] Saved: {output_path}")
    
    return output_path


if __name__ == "__main__":
    case_name = "colon_001"
    image_path = f"data/medical_decathlon/Task10_Colon/imagesTr/{case_name}.nii.gz"
    output_path = f"outputs/bright_boxes_{case_name}.png"
    
    print(f"\n[*] Creating BRIGHT BOX visualization for {case_name}")
    print("="*60)
    
    nii = nib.load(image_path)
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    print(f"   Volume: {volume.shape}")
    
    print(f"\n[*] Running detection...")
    lesions = quick_detect_colon_tumors(ct_volume=volume, spacing=spacing, device="gpu")
    
    print(f"\n[*] Found {len(lesions)} lesions")
    print(f"\n[*] Creating visualization with BRIGHT colored boxes...")
    
    viz_path = create_simple_bbox_visualization(volume, lesions, output_path)
    print(f"\n[+] Done! See: {viz_path}")
