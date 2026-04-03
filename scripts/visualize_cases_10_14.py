"""
Visualize only cases 010-014
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_cases_10_14(dataset_dir, output_dir):
    """Create visualizations for cases 010-014"""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    case_ids = [f"colon_{i:03d}" for i in range(10, 15)]
    
    # Create summary figure
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    
    stats_list = []
    
    for idx, case_id in enumerate(case_ids):
        # Load data
        ct_file = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        body_mask_file = dataset_dir / "body_masks" / f"{case_id}_body.nii.gz"
        colon_mask_file = dataset_dir / "colon_masks" / f"{case_id}_colon.nii.gz"
        
        ct_data = nib.load(ct_file).get_fdata()
        body_mask = nib.load(body_mask_file).get_fdata()
        colon_mask = nib.load(colon_mask_file).get_fdata()
        
        # Middle slice
        mid_slice = ct_data.shape[2] // 2
        
        ct_slice = ct_data[:, :, mid_slice]
        body_slice = body_mask[:, :, mid_slice]
        colon_slice = colon_mask[:, :, mid_slice]
        
        # Stats
        body_pixels = body_slice.astype(bool).sum()
        colon_pixels = colon_slice.astype(bool).sum()
        body_coverage = body_pixels / ct_slice.size * 100
        
        stats_list.append({
            'case': case_id,
            'slice': mid_slice,
            'body_coverage': body_coverage,
            'colon_pixels': colon_pixels
        })
        
        # Column 1: Original CT
        axes[idx, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 0].set_title(f'{case_id} - Original CT\nSlice {mid_slice}', fontsize=10)
        axes[idx, 0].axis('off')
        
        # Column 2: Body mask overlay
        axes[idx, 1].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 1].imshow(body_slice.T, cmap='Reds', alpha=0.5, origin='lower')
        axes[idx, 1].set_title(f'Body Mask (Red)\n{body_coverage:.1f}% coverage', fontsize=10)
        axes[idx, 1].axis('off')
        
        # Column 3: Colon mask overlay
        axes[idx, 2].imshow(ct_slice.T, cmap='gray', origin='lower')
        if colon_pixels > 0:
            axes[idx, 2].imshow(colon_slice.T, cmap='Greens', alpha=0.5, origin='lower')
            axes[idx, 2].set_title(f'Colon Mask (Green)\n{colon_pixels} pixels', fontsize=10)
        else:
            axes[idx, 2].set_title(f'Colon Mask\nNone in this slice', fontsize=10)
        axes[idx, 2].axis('off')
        
        # Column 4: Combined
        axes[idx, 3].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[idx, 3].imshow(body_slice.T, cmap='Reds', alpha=0.3, origin='lower')
        if colon_pixels > 0:
            axes[idx, 3].imshow(colon_slice.T, cmap='Greens', alpha=0.5, origin='lower')
        axes[idx, 3].set_title(f'Combined\nRed=Body, Green=Colon', fontsize=10)
        axes[idx, 3].axis('off')
        
        print(f"{case_id} (slice {mid_slice}): Body={body_coverage:.1f}%, Colon={colon_pixels} pixels")
    
    plt.tight_layout()
    output_file = output_dir / "cases_010_014_summary.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved summary to: {output_file}")
    
    # Create individual visualizations
    for case_id in case_ids:
        ct_file = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        body_mask_file = dataset_dir / "body_masks" / f"{case_id}_body.nii.gz"
        colon_mask_file = dataset_dir / "colon_masks" / f"{case_id}_colon.nii.gz"
        
        ct_data = nib.load(ct_file).get_fdata()
        body_mask = nib.load(body_mask_file).get_fdata()
        colon_mask = nib.load(colon_mask_file).get_fdata()
        
        mid_slice = ct_data.shape[2] // 2
        
        ct_slice = ct_data[:, :, mid_slice]
        body_slice = body_mask[:, :, mid_slice]
        colon_slice = colon_mask[:, :, mid_slice]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Same as above for individual files
        axes[0].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[0].set_title(f'Original CT - Slice {mid_slice}')
        axes[0].axis('off')
        
        axes[1].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[1].imshow(body_slice.T, cmap='Reds', alpha=0.5, origin='lower')
        axes[1].set_title('Body Mask (Red)')
        axes[1].axis('off')
        
        axes[2].imshow(ct_slice.T, cmap='gray', origin='lower')
        if colon_slice.sum() > 0:
            axes[2].imshow(colon_slice.T, cmap='Greens', alpha=0.5, origin='lower')
        axes[2].set_title('Colon Mask (Green)')
        axes[2].axis('off')
        
        axes[3].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[3].imshow(body_slice.T, cmap='Reds', alpha=0.3, origin='lower')
        if colon_slice.sum() > 0:
            axes[3].imshow(colon_slice.T, cmap='Greens', alpha=0.5, origin='lower')
        axes[3].set_title('Combined (Red=Body, Green=Colon)')
        axes[3].axis('off')
        
        plt.tight_layout()
        individual_file = output_dir / f"{case_id}_visualization.png"
        plt.savefig(individual_file, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"  Created: {individual_file.name}")
    
    return stats_list


if __name__ == "__main__":
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    output_dir = Path("f:/ADDS/outputs/mask_visualizations_cases_10_14")
    
    print("="*60)
    print("VISUALIZING CASES 010-014")
    print("="*60)
    
    stats = visualize_cases_10_14(dataset_dir, output_dir)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Cases visualized: {len(stats)}")
