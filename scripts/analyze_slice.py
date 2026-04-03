"""
Analyze specific slice to debug body mask issues
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_slice(case_id, dataset_dir, slice_idx, output_file):
    """Detailed analysis of a specific slice"""
    dataset_dir = Path(dataset_dir)
    
    # Load data
    ct_file = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
    body_mask_file = dataset_dir / "body_masks" / f"{case_id}_body.nii.gz"
    
    ct_data = nib.load(ct_file).get_fdata()
    body_mask = nib.load(body_mask_file).get_fdata()
    
    # Extract slice
    ct_slice = ct_data[:, :, slice_idx]
    body_slice = body_mask[:, :, slice_idx]
    
    # Create detailed figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original CT
    axes[0, 0].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-1000, vmax=400)
    axes[0, 0].set_title(f'Original CT - Slice {slice_idx}\nFull HU range')
    axes[0, 0].axis('off')
    
    # 2. CT windowed (soft tissue)
    axes[0, 1].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-200, vmax=200)
    axes[0, 1].set_title('CT - Soft Tissue Window\nHU: -200 to 200')
    axes[0, 1].axis('off')
    
    # 3. HU threshold visualization
    hu_thresh = ct_slice > -500
    axes[0, 2].imshow(hu_thresh.T, cmap='gray', origin='lower')
    axes[0, 2].set_title('HU > -500 (before morphology)')
    axes[0, 2].axis('off')
    
    # 4. Body mask overlay
    axes[1, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[1, 0].imshow(body_slice.T, cmap='Reds', alpha=0.5, origin='lower')
    axes[1, 0].set_title('Body Mask Overlay (RED)')
    axes[1, 0].axis('off')
    
    # 5. Mask only
    axes[1, 1].imshow(body_slice.T, cmap='gray', origin='lower')
    axes[1, 1].set_title('Body Mask Only')
    axes[1, 1].axis('off')
    
    # 6. Edge detection
    from scipy.ndimage import binary_erosion
    edges = body_slice.astype(bool) & ~binary_erosion(body_slice.astype(bool))
    axes[1, 2].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[1, 2].imshow(edges.T, cmap='Reds', alpha=0.8, origin='lower')
    axes[1, 2].set_title('Mask Boundary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Statistics
    print(f"\n[Slice {slice_idx} Analysis]")
    print(f"  CT HU range: {ct_slice.min():.1f} to {ct_slice.max():.1f}")
    print(f"  Body mask coverage: {body_slice.sum() / body_slice.size * 100:.1f}%")
    print(f"  Pixels > -500 HU: {(ct_slice > -500).sum() / ct_slice.size * 100:.1f}%")
    print(f"  Image size: {ct_slice.shape}")
    print(f"  Mask sum: {body_slice.sum()} pixels")
    
    # Check for over-expansion
    coverage = body_slice.sum() / body_slice.size * 100
    if coverage > 60:
        print(f"  [WARNING] Mask coverage {coverage:.1f}% exceeds 60% - likely over-expanded")
    
    return {
        'slice': slice_idx,
        'coverage': coverage,
        'hu_min': ct_slice.min(),
        'hu_max': ct_slice.max()
    }


if __name__ == "__main__":
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    output_dir = Path("f:/ADDS/outputs/debug_masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze slice 30
    print("="*60)
    print("ANALYZING SLICE 30")
    print("="*60)
    
    output_file = output_dir / "colon_000_slice30_analysis.png"
    stats = analyze_slice("colon_000", dataset_dir, 30, output_file)
    
    print(f"\nSaved detailed analysis to: {output_file}")
