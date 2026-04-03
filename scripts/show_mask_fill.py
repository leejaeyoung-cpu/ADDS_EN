"""
Show mask FILL not just contours
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def show_mask_fill(case_id, dataset_dir, output_dir):
    """Show actual mask fill, not just contours"""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ct_file = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
    body_mask_file = dataset_dir / "body_masks" / f"{case_id}_body.nii.gz"
    
    ct_data = nib.load(ct_file).get_fdata()
    body_mask = nib.load(body_mask_file).get_fdata()
    
    # Middle slice
    slice_idx = ct_data.shape[2] // 2
    
    ct_slice = ct_data[:, :, slice_idx]
    mask_slice = body_mask[:, :, slice_idx]
    
    # Actual body (HU > -500)
    actual_body = ct_slice > -500
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Actual Body (Green)
    # 1. CT only
    axes[0, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'Original CT - Slice {slice_idx}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Actual body FILL (green)
    axes[0, 1].imshow(ct_slice.T, cmap='gray', origin='lower', alpha=0.7)
    axes[0, 1].imshow(actual_body.T, cmap='Greens', alpha=0.5, origin='lower')
    axes[0, 1].set_title('GREEN = HU > -500 (FILL)\n테이블 포함, 내부 구멍 있음', fontsize=12, fontweight='bold', color='green')
    axes[0, 1].axis('off')
    
    # 3. Actual body only (no CT)
    axes[0, 2].imshow(actual_body.T, cmap='gray', origin='lower')
    axes[0, 2].set_title('GREEN mask only\n구멍들 = 내부 공기', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Body Mask (Red)
    # 4. CT only (repeated)
    axes[1, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[1, 0].set_title(f'Original CT - Slice {slice_idx}', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 5. Body mask FILL (red)
    axes[1, 1].imshow(ct_slice.T, cmap='gray', origin='lower', alpha=0.7)
    axes[1, 1].imshow(mask_slice.T, cmap='Reds', alpha=0.5, origin='lower')
    axes[1, 1].set_title('RED = Body Mask (FILL)\n테이블 제거, 내부 채워짐', fontsize=12, fontweight='bold', color='red')
    axes[1, 1].axis('off')
    
    # 6. Body mask only (no CT)
    axes[1, 2].imshow(mask_slice.T, cmap='gray', origin='lower')
    axes[1, 2].set_title('RED mask only\n내부가 채워져 있음', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_file = output_dir / f"{case_id}_mask_fill_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Stats
    actual_pixels = actual_body.sum()
    mask_pixels = mask_slice.sum()
    
    # Count holes in actual body
    from scipy.ndimage import label
    inverted_actual = ~actual_body
    labeled_holes, num_holes = label(inverted_actual)
    # Subtract 1 for background
    num_holes_actual = num_holes - 1 if num_holes > 0 else 0
    
    # Count holes in mask
    inverted_mask = ~mask_slice.astype(bool)
    labeled_mask_holes, num_mask_holes = label(inverted_mask)
    num_holes_mask = num_mask_holes - 1 if num_mask_holes > 0 else 0
    
    print(f"\n[{case_id} - Slice {slice_idx}]")
    print(f"  GREEN (HU>-500):")
    print(f"    - Pixels: {actual_pixels} ({actual_pixels/ct_slice.size*100:.1f}%)")
    print(f"    - 내부 구멍 개수: {num_holes_actual}")
    print(f"  RED (Body Mask):")
    print(f"    - Pixels: {mask_pixels} ({mask_slice.sum()/ct_slice.size*100:.1f}%)")
    print(f"    - 내부 구멍 개수: {num_holes_mask}")
    print(f"\n  결과: Body mask가 내부 구멍 {num_holes_actual - num_holes_mask}개를 채웠습니다")
    
    return output_file


if __name__ == "__main__":
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    output_dir = Path("f:/ADDS/outputs/debug_masks")
    
    print("="*60)
    print("MASK FILL COMPARISON (내부 채우기 확인)")
    print("="*60)
    
    # Show multiple cases
    for case_id in ["colon_000", "colon_001", "colon_002"]:
        output_file = show_mask_fill(case_id, dataset_dir, output_dir)
        print(f"  Saved: {output_file.name}\n")
