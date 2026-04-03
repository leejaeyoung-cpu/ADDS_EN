"""
Analyze why table is still included in body mask
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label, binary_erosion


def analyze_table_connection(case_id, dataset_dir, output_dir):
    """Analyze if body and table are connected"""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CT
    ct_file = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
    ct_data = nib.load(ct_file).get_fdata()
    
    # Middle slice
    slice_idx = ct_data.shape[2] // 2
    ct_slice = ct_data[:, :, slice_idx]
    
    # Threshold
    thresh_mask = ct_slice > -500
    
    # Apply closing
    from scipy.ndimage import binary_closing
    closed_mask = binary_closing(thresh_mask, iterations=3)
    
    # Label connected components
    labeled, num_components = label(closed_mask)
    
    # Find component sizes
    component_sizes = []
    for i in range(1, num_components + 1):
        size = (labeled == i).sum()
        component_sizes.append((i, size))
    
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original CT
    axes[0, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'Original CT - Slice {slice_idx}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Threshold (HU > -500)
    axes[0, 1].imshow(thresh_mask.T, cmap='gray', origin='lower')
    axes[0, 1].set_title('HU > -500 Threshold\n(before closing)', fontsize=12)
    axes[0, 1].axis('off')
    
    # 3. After closing
    axes[0, 2].imshow(closed_mask.T, cmap='gray', origin='lower')
    axes[0, 2].set_title(f'After Closing (3 iterations)\n{num_components} components', fontsize=12)
    axes[0, 2].axis('off')
    
    # 4. Largest component only
    if num_components > 0:
        largest_comp = (labeled == component_sizes[0][0])
        axes[1, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[1, 0].imshow(largest_comp.T, cmap='Reds', alpha=0.5, origin='lower')
        axes[1, 0].set_title(f'Largest Component\nSize: {component_sizes[0][1]} pixels', fontsize=12, color='red')
        axes[1, 0].axis('off')
    
    # 5. All components colored differently
    axes[1, 1].imshow(ct_slice.T, cmap='gray', origin='lower', alpha=0.5)
    axes[1, 1].imshow(labeled.T, cmap='nipy_spectral', origin='lower', alpha=0.6)
    axes[1, 1].set_title(f'All {num_components} Components\n(different colors)', fontsize=12)
    axes[1, 1].axis('off')
    
    # 6. Bottom region highlight (where table should be)
    bottom_highlight = np.zeros_like(ct_slice)
    height = ct_slice.shape[1]
    y_cutoff = int(height * 0.1)
    bottom_highlight[:, :y_cutoff] = 1
    
    axes[1, 2].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[1, 2].imshow(bottom_highlight.T, cmap='Reds', alpha=0.3, origin='lower')
    axes[1, 2].set_title(f'Bottom 10% (to be removed)\nY < {y_cutoff}', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_file = output_dir / f"{case_id}_table_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"\n[{case_id} - Slice {slice_idx}]")
    print(f"  Total components after closing: {num_components}")
    print(f"  Component sizes (largest to smallest):")
    for idx, (comp_id, size) in enumerate(component_sizes[:5]):
        pct = size / ct_slice.size * 100
        print(f"    #{idx+1}: {size} pixels ({pct:.1f}%)")
    
    # Check if table is connected to body
    if num_components == 1:
        print(f"\n  [PROBLEM] Only 1 component - Body and table are CONNECTED!")
        print(f"  [SOLUTION] Need stronger separation before connected components")
    else:
        print(f"\n  [GOOD] {num_components} components - Body and table are separated")
    
    # Check bottom region
    if num_components > 0:
        largest_comp = (labeled == component_sizes[0][0])
        table_region_pixels = largest_comp[:, :y_cutoff].sum()
        total_pixels = largest_comp.sum()
        table_pct = table_region_pixels / total_pixels * 100 if total_pixels > 0 else 0
        
        print(f"\n  Bottom 10% region:")
        print(f"    Pixels in largest component: {table_region_pixels}")
        print(f"    Percentage of largest component: {table_pct:.1f}%")
        
        if table_pct > 5:
            print(f"    [WARNING] >5% in bottom region - table likely included")
    
    return output_file, num_components


if __name__ == "__main__":
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    output_dir = Path("f:/ADDS/outputs/debug_masks")
    
    print("="*60)
    print("TABLE CONNECTION ANALYSIS")
    print("="*60)
    
    for case_id in ["colon_000", "colon_001", "colon_002"]:
        output_file, num_comp = analyze_table_connection(case_id, dataset_dir, output_dir)
        print(f"  Saved: {output_file.name}")
        print()
