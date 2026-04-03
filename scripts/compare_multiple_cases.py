"""
Compare body masks across multiple cases
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_erosion


def compare_multiple_cases(dataset_dir, output_dir, num_cases=5):
    """Create side-by-side comparison of multiple cases"""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get case IDs
    images_dir = dataset_dir / "imagesTr"
    image_files = sorted(images_dir.glob("*_0000.nii.gz"))[:num_cases]
    case_ids = [f.name.replace("_0000.nii.gz", "") for f in image_files]
    
    # Create large comparison figure
    fig, axes = plt.subplots(num_cases, 4, figsize=(20, 5*num_cases))
    
    if num_cases == 1:
        axes = axes.reshape(1, -1)
    
    for idx, case_id in enumerate(case_ids):
        # Load data
        ct_file = images_dir / f"{case_id}_0000.nii.gz"
        body_mask_file = dataset_dir / "body_masks" / f"{case_id}_body.nii.gz"
        
        ct_data = nib.load(ct_file).get_fdata()
        body_mask = nib.load(body_mask_file).get_fdata()
        
        # Middle slice
        slice_idx = ct_data.shape[2] // 2
        
        ct_slice = ct_data[:, :, slice_idx]
        mask_slice = body_mask[:, :, slice_idx]
        
        # Actual body (HU > -500)
        actual_body = ct_slice > -500
        
        # Column 1: Original CT
        axes[idx, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 0].set_title(f'{case_id} - Original CT\nSlice {slice_idx}', fontsize=10)
        axes[idx, 0].axis('off')
        
        # Column 2: CT + Actual body (Green)
        axes[idx, 1].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 1].contour(actual_body.T, colors='green', linewidths=2, levels=[0.5])
        axes[idx, 1].set_title('Actual Body (HU>-500)\nGreen contour', fontsize=10)
        axes[idx, 1].axis('off')
        
        # Column 3: CT + Body mask (Red)
        axes[idx, 2].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 2].contour(mask_slice.T, colors='red', linewidths=2, levels=[0.5])
        axes[idx, 2].set_title('Body Mask\nRed contour', fontsize=10)
        axes[idx, 2].axis('off')
        
        # Column 4: Overlay (Green + Red)
        axes[idx, 3].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 3].contour(actual_body.T, colors='green', linewidths=2, levels=[0.5], linestyles='dashed', alpha=0.7)
        axes[idx, 3].contour(mask_slice.T, colors='red', linewidths=2, levels=[0.5], alpha=0.8)
        axes[idx, 3].set_title('Overlay\nGreen=Actual, Red=Mask', fontsize=10)
        axes[idx, 3].axis('off')
        
        # Calculate stats
        mask_bool = mask_slice.astype(bool)
        actual_pixels = actual_body.sum()
        mask_pixels = mask_bool.sum()
        overlap = (actual_body & mask_bool).sum()
        dice = 2 * overlap / (actual_pixels + mask_pixels) if (actual_pixels + mask_pixels) > 0 else 0
        
        coverage_actual = actual_pixels / ct_slice.size * 100
        coverage_mask = mask_pixels / ct_slice.size * 100
        
        print(f"{case_id} (slice {slice_idx}): Dice={dice:.3f}, Actual={coverage_actual:.1f}%, Mask={coverage_mask:.1f}%")
    
    plt.tight_layout()
    output_file = output_dir / "multi_case_comparison.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison to: {output_file}")
    return output_file


if __name__ == "__main__":
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    output_dir = Path("f:/ADDS/outputs/debug_masks")
    
    print("="*60)
    print("MULTI-CASE BODY MASK COMPARISON")
    print("="*60)
    
    output_file = compare_multiple_cases(dataset_dir, output_dir, num_cases=5)
