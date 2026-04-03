"""
Compare original CT body contour with body mask
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_erosion


def compare_body_shapes(case_id, dataset_dir, slice_idx, output_file):
    """Compare actual body shape with mask shape"""
    dataset_dir = Path(dataset_dir)
    
    # Load data
    ct_file = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
    body_mask_file = dataset_dir / "body_masks" / f"{case_id}_body.nii.gz"
    
    ct_data = nib.load(ct_file).get_fdata()
    body_mask = nib.load(body_mask_file).get_fdata()
    
    # Extract slice
    ct_slice = ct_data[:, :, slice_idx]
    mask_slice = body_mask[:, :, slice_idx]
    
    # Create "ground truth" body by simple HU threshold
    actual_body = ct_slice > -500
    
    # Get edges for comparison
    mask_edge = mask_slice.astype(bool) & ~binary_erosion(mask_slice.astype(bool))
    actual_edge = actual_body & ~binary_erosion(actual_body)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. Original CT with actual body edge
    axes[0, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[0, 0].contour(actual_body.T, colors='green', linewidths=2, levels=[0.5])
    axes[0, 0].set_title('Original CT with Actual Body Contour (Green)\nHU > -500 threshold')
    axes[0, 0].axis('off')
    
    # 2. Original CT with mask edge
    axes[0, 1].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[0, 1].contour(mask_slice.T, colors='red', linewidths=2, levels=[0.5])
    axes[0, 1].set_title('Original CT with Body Mask Contour (Red)\nAfter morphology + connected components')
    axes[0, 1].axis('off')
    
    # 3. Both overlaid
    axes[1, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[1, 0].contour(actual_body.T, colors='green', linewidths=2, levels=[0.5], alpha=0.7)
    axes[1, 0].contour(mask_slice.T, colors='red', linewidths=2, levels=[0.5], alpha=0.7)
    axes[1, 0].set_title('Comparison: Green=Actual, Red=Mask\nLook for shape distortion')
    axes[1, 0].axis('off')
    
    # 4. Difference map
    # Show where mask differs from actual body
    mask_bool = mask_slice.astype(bool)
    over_included = mask_bool & ~actual_body  # Mask has it, actual doesn't
    under_included = actual_body & ~mask_bool  # Actual has it, mask doesn't
    
    diff_map = np.zeros_like(ct_slice)
    diff_map[over_included] = 1  # Red: over-included
    diff_map[under_included] = -1  # Blue: under-included
    
    axes[1, 1].imshow(ct_slice.T, cmap='gray', origin='lower', alpha=0.5)
    axes[1, 1].imshow(diff_map.T, cmap='RdBu', origin='lower', alpha=0.5, vmin=-1, vmax=1)
    axes[1, 1].set_title('Difference Map\nRed=Over-included, Blue=Under-included')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate metrics
    total_pixels = ct_slice.size
    actual_pixels = actual_body.sum()
    mask_pixels = mask_bool.sum()
    overlap = (actual_body & mask_bool).sum()
    
    dice = 2 * overlap / (actual_pixels + mask_pixels) if (actual_pixels + mask_pixels) > 0 else 0
    iou = overlap / ((actual_body | mask_bool).sum()) if ((actual_body | mask_bool).sum()) > 0 else 0
    
    over_pct = over_included.sum() / total_pixels * 100
    under_pct = under_included.sum() / total_pixels * 100
    
    print(f"\n[Shape Comparison - Slice {slice_idx}]")
    print(f"  Actual body pixels: {actual_pixels} ({actual_pixels/total_pixels*100:.1f}%)")
    print(f"  Mask pixels: {mask_pixels} ({mask_pixels/total_pixels*100:.1f}%)")
    print(f"  Overlap: {overlap} pixels")
    print(f"  Dice Score: {dice:.3f}")
    print(f"  IoU: {iou:.3f}")
    print(f"  Over-included: {over_included.sum()} pixels ({over_pct:.2f}%)")
    print(f"  Under-included: {under_included.sum()} pixels ({under_pct:.2f}%)")
    
    if dice < 0.95:
        print(f"  [WARNING] Dice < 0.95 - Shape distortion detected!")
    
    return {
        'dice': dice,
        'iou': iou,
        'over_pct': over_pct,
        'under_pct': under_pct
    }


if __name__ == "__main__":
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    output_dir = Path("f:/ADDS/outputs/debug_masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("BODY SHAPE COMPARISON")
    print("="*60)
    
    # Analyze slice 30
    output_file = output_dir / "body_shape_comparison_slice30.png"
    stats = compare_body_shapes("colon_000", dataset_dir, 30, output_file)
    
    print(f"\nSaved comparison to: {output_file}")
    
    if stats['dice'] < 0.95:
        print("\n[ISSUE] Shape distortion confirmed!")
        print("Recommendation: Reduce morphological operations")
