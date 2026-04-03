"""
Detailed analysis of colon_005 masking failure
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label


def analyze_case_005(dataset_dir, output_dir):
    """Analyze why colon_005 masking failed"""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    case_id = "colon_005"
    
    # Load data
    ct_file = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
    body_mask_file = dataset_dir / "body_masks" / f"{case_id}_body.nii.gz"
    colon_mask_file = dataset_dir / "colon_masks" / f"{case_id}_colon.nii.gz"
    
    ct_data = nib.load(ct_file).get_fdata()
    body_mask = nib.load(body_mask_file).get_fdata()
    colon_mask = nib.load(colon_mask_file).get_fdata()
    
    # Get middle slice
    mid_slice = ct_data.shape[2] // 2
    
    print(f"\n{'='*60}")
    print(f"COLON_005 DETAILED ANALYSIS")
    print(f"{'='*60}")
    print(f"CT shape: {ct_data.shape}")
    print(f"Middle slice: {mid_slice}")
    print(f"{'='*60}\n")
    
    # Check multiple slices
    slices_to_check = [
        int(ct_data.shape[2] * 0.2),
        int(ct_data.shape[2] * 0.4),
        mid_slice,
        int(ct_data.shape[2] * 0.6),
        int(ct_data.shape[2] * 0.8),
    ]
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    
    for idx, slice_idx in enumerate(slices_to_check):
        ct_slice = ct_data[:, :, slice_idx]
        body_slice = body_mask[:, :, slice_idx]
        colon_slice = colon_mask[:, :, slice_idx]
        
        # Actual body (HU > -500)
        actual_body = ct_slice > -500
        
        # Stats
        ct_pixels = ct_slice.size
        actual_pixels = actual_body.sum()
        body_pixels = body_slice.astype(bool).sum()
        colon_pixels = colon_slice.astype(bool).sum()
        colon_in_body = (colon_slice.astype(bool) & body_slice.astype(bool)).sum()
        
        # Check bottom region
        height = ct_slice.shape[1]
        y_15pct = int(height * 0.15)
        y_12pct = int(height * 0.12)
        
        body_in_bottom_15 = actual_body[:, :y_15pct].sum()
        body_in_bottom_12 = actual_body[:, :y_12pct].sum()
        
        print(f"Slice {slice_idx} ({idx*20+20}%):")
        print(f"  Actual body (HU>-500): {actual_pixels} ({actual_pixels/ct_pixels*100:.1f}%)")
        print(f"  Body mask: {body_pixels} ({body_pixels/ct_pixels*100:.1f}%)")
        print(f"  Colon mask: {colon_pixels} ({colon_pixels/ct_pixels*100:.1f}%)")
        print(f"  Colon in body: {colon_in_body} ({colon_in_body/colon_pixels*100 if colon_pixels > 0 else 0:.1f}%)")
        print(f"  Actual body in bottom 15%: {body_in_bottom_15} ({body_in_bottom_15/actual_pixels*100 if actual_pixels > 0 else 0:.1f}%)")
        print(f"  Actual body in bottom 12%: {body_in_bottom_12} ({body_in_bottom_12/actual_pixels*100 if actual_pixels > 0 else 0:.1f}%)")
        
        if body_in_bottom_15 > actual_pixels * 0.05:
            print(f"  [WARNING] >5% of body in bottom 15% region - TOO MUCH REMOVED!")
        
        if colon_pixels == 0:
            print(f"  [WARNING] No colon detected in this slice")
        
        print()
        
        # Visualizations
        # Column 1: Original CT
        axes[idx, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 0].set_title(f'Slice {slice_idx} - Original CT', fontsize=10)
        axes[idx, 0].axis('off')
        
        # Column 2: Actual body (HU > -500) with cutoff lines
        axes[idx, 1].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 1].imshow(actual_body.T, cmap='Greens', alpha=0.3, origin='lower')
        axes[idx, 1].axhline(y=y_15pct, color='red', linestyle='--', linewidth=2, label='15% cutoff')
        axes[idx, 1].axhline(y=y_12pct, color='orange', linestyle='--', linewidth=2, label='12% cutoff')
        axes[idx, 1].set_title(f'Actual Body + Cutoff Lines\nBody in bottom 15%: {body_in_bottom_15/actual_pixels*100 if actual_pixels > 0 else 0:.1f}%', fontsize=10)
        axes[idx, 1].axis('off')
        axes[idx, 1].legend(loc='upper right', fontsize=8)
        
        # Column 3: Body mask
        axes[idx, 2].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 2].imshow(body_slice.T, cmap='Reds', alpha=0.5, origin='lower')
        axes[idx, 2].set_title(f'Body Mask\nCoverage: {body_pixels/ct_pixels*100:.1f}%', fontsize=10)
        axes[idx, 2].axis('off')
        
        # Column 4: Colon mask
        axes[idx, 3].imshow(ct_slice.T, cmap='gray', origin='lower')
        if colon_pixels > 0:
            axes[idx, 3].imshow(colon_slice.T, cmap='Greens', alpha=0.5, origin='lower')
            axes[idx, 3].set_title(f'Colon Mask\n{colon_pixels} pixels', fontsize=10)
        else:
            axes[idx, 3].set_title(f'Colon Mask\nNONE DETECTED', fontsize=10, color='red')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    output_file = output_dir / "colon_005_detailed_analysis.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    # Check overall colon mask
    total_colon_pixels = colon_mask.sum()
    total_body_pixels = body_mask.sum()
    
    print(f"Total colon pixels (all slices): {total_colon_pixels}")
    print(f"Total body pixels (all slices): {total_body_pixels}")
    
    if total_colon_pixels == 0:
        print(f"[CRITICAL] TotalSegmentator found NO colon in entire volume!")
    
    print(f"\nSaved analysis to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    output_dir = Path("f:/ADDS/outputs/debug_masks")
    
    output_file = analyze_case_005(dataset_dir, output_dir)
