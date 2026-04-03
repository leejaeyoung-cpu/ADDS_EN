"""
Multi-slice body shape analysis
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_erosion


def analyze_multiple_slices(case_id, dataset_dir, output_dir):
    """Analyze body shape across multiple slices"""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ct_file = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
    body_mask_file = dataset_dir / "body_masks" / f"{case_id}_body.nii.gz"
    
    ct_data = nib.load(ct_file).get_fdata()
    body_mask = nib.load(body_mask_file).get_fdata()
    
    num_slices = ct_data.shape[2]
    
    # Select representative slices (evenly distributed)
    slice_indices = [
        int(num_slices * 0.2),   # 20%
        int(num_slices * 0.4),   # 40%
        int(num_slices * 0.5),   # 50% (middle)
        int(num_slices * 0.6),   # 60%
        int(num_slices * 0.8),   # 80%
    ]
    
    # Create comparison figure
    fig, axes = plt.subplots(5, 3, figsize=(18, 24))
    
    stats_list = []
    
    for idx, slice_num in enumerate(slice_indices):
        ct_slice = ct_data[:, :, slice_num]
        mask_slice = body_mask[:, :, slice_num]
        
        # Actual body (simple threshold)
        actual_body = ct_slice > -500
        
        # Column 1: CT with actual body contour (GREEN)
        axes[idx, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 0].contour(actual_body.T, colors='green', linewidths=3, levels=[0.5])
        axes[idx, 0].set_title(f'Slice {slice_num} - Actual Body (GREEN)', fontsize=12)
        axes[idx, 0].axis('off')
        
        # Column 2: CT with mask contour (RED)
        axes[idx, 1].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 1].contour(mask_slice.T, colors='red', linewidths=3, levels=[0.5])
        axes[idx, 1].set_title(f'Slice {slice_num} - Body Mask (RED)', fontsize=12)
        axes[idx, 1].axis('off')
        
        # Column 3: Both overlaid
        axes[idx, 2].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 2].contour(actual_body.T, colors='green', linewidths=2, levels=[0.5], alpha=0.8, linestyles='dashed')
        axes[idx, 2].contour(mask_slice.T, colors='red', linewidths=2, levels=[0.5], alpha=0.8)
        axes[idx, 2].set_title(f'Slice {slice_num} - Overlay\nGreen=Actual (dashed), Red=Mask', fontsize=12)
        axes[idx, 2].axis('off')
        
        # Calculate stats
        mask_bool = mask_slice.astype(bool)
        actual_pixels = actual_body.sum()
        mask_pixels = mask_bool.sum()
        overlap = (actual_body & mask_bool).sum()
        dice = 2 * overlap / (actual_pixels + mask_pixels) if (actual_pixels + mask_pixels) > 0 else 0
        
        stats_list.append({
            'slice': slice_num,
            'dice': dice,
            'actual_pct': actual_pixels / ct_slice.size * 100,
            'mask_pct': mask_pixels / ct_slice.size * 100
        })
    
    plt.tight_layout()
    output_file = output_dir / f"{case_id}_multislice_comparison.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Print stats
    print(f"\n{'='*60}")
    print(f"MULTI-SLICE BODY SHAPE ANALYSIS")
    print(f"{'='*60}")
    print(f"Case: {case_id}")
    print(f"Total slices: {num_slices}")
    print(f"{'='*60}\n")
    
    for stat in stats_list:
        print(f"Slice {stat['slice']:3d}: Dice={stat['dice']:.3f}, "
              f"Actual={stat['actual_pct']:5.1f}%, Mask={stat['mask_pct']:5.1f}%")
    
    avg_dice = np.mean([s['dice'] for s in stats_list])
    print(f"\nAverage Dice Score: {avg_dice:.3f}")
    
    if avg_dice < 0.95:
        print(f"[WARNING] Average Dice < 0.95 - Shape distortion confirmed!")
    else:
        print(f"[INFO] Average Dice >= 0.95 - Shapes match well numerically")
        print(f"[INFO] Visual inspection needed to confirm shape preservation")
    
    print(f"\nSaved multi-slice comparison to: {output_file}")
    
    return stats_list


if __name__ == "__main__":
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    output_dir = Path("f:/ADDS/outputs/debug_masks")
    
    stats = analyze_multiple_slices("colon_000", dataset_dir, output_dir)
