"""
Visualize Anatomical Masks
===========================
Create comparison images showing:
- Original CT
- Body mask overlay
- Colon mask overlay
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def visualize_case(case_id, dataset_dir, output_dir, slice_idx=None):
    """
    Create visualization for a single case
    
    Args:
        case_id: Case identifier (e.g., 'colon_000')
        dataset_dir: Path to dataset
        output_dir: Where to save visualization
        slice_idx: Which slice to visualize (None = middle slice)
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ct_file = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
    body_mask_file = dataset_dir / "body_masks" / f"{case_id}_body.nii.gz"
    colon_mask_file = dataset_dir / "colon_masks" / f"{case_id}_colon.nii.gz"
    
    ct_img = nib.load(ct_file)
    ct_data = ct_img.get_fdata()
    
    body_mask = nib.load(body_mask_file).get_fdata()
    colon_mask = nib.load(colon_mask_file).get_fdata()
    
    # Select middle slice if not specified
    if slice_idx is None:
        slice_idx = ct_data.shape[2] // 2
    
    # Extract slice
    ct_slice = ct_data[:, :, slice_idx]
    body_slice = body_mask[:, :, slice_idx]
    colon_slice = colon_mask[:, :, slice_idx]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. Original CT
    axes[0, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'{case_id} - Original CT\nSlice {slice_idx}')
    axes[0, 0].axis('off')
    
    # 2. Body mask overlay
    axes[0, 1].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[0, 1].imshow(body_slice.T, cmap='Reds', alpha=0.3, origin='lower')
    axes[0, 1].set_title('Body Mask (Red)')
    axes[0, 1].axis('off')
    
    # 3. Colon mask overlay
    axes[1, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[1, 0].imshow(colon_slice.T, cmap='Greens', alpha=0.5, origin='lower')
    axes[1, 0].set_title('Colon Mask (Green)')
    axes[1, 0].axis('off')
    
    # 4. Combined overlay
    axes[1, 1].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[1, 1].imshow(body_slice.T, cmap='Reds', alpha=0.2, origin='lower')
    axes[1, 1].imshow(colon_slice.T, cmap='Greens', alpha=0.5, origin='lower')
    axes[1, 1].set_title('Combined (Red=Body, Green=Colon)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f"{case_id}_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file.name}")
    
    # Calculate statistics
    body_coverage = body_slice.sum() / body_slice.size * 100
    colon_coverage = colon_slice.sum() / body_slice.sum() * 100 if body_slice.sum() > 0 else 0
    
    return {
        'case_id': case_id,
        'slice': slice_idx,
        'body_coverage_pct': body_coverage,
        'colon_in_body_pct': colon_coverage
    }


def create_summary_visualization(dataset_dir, output_dir, max_cases=5):
    """Create visualizations for multiple cases"""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    
    # Get case IDs
    body_masks_dir = dataset_dir / "body_masks"
    mask_files = sorted(body_masks_dir.glob("*_body.nii.gz"))[:max_cases]
    
    case_ids = [f.name.replace("_body.nii.gz", "") for f in mask_files]
    
    print(f"\n{'='*60}")
    print(f"VISUALIZING ANATOMICAL MASKS")
    print(f"{'='*60}")
    print(f"Cases: {len(case_ids)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Visualize each case
    stats = []
    for case_id in case_ids:
        print(f"[{case_id}]")
        stat = visualize_case(case_id, dataset_dir, output_dir)
        stats.append(stat)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    for stat in stats:
        print(f"{stat['case_id']}:")
        print(f"  Slice: {stat['slice']}")
        print(f"  Body coverage: {stat['body_coverage_pct']:.1f}%")
        print(f"  Colon in body: {stat['colon_in_body_pct']:.1f}%")
    print(f"{'='*60}\n")
    
    print(f"All visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="f:/ADDS/nnUNet_raw/Dataset010_Colon")
    parser.add_argument("--output", type=str, default="f:/ADDS/outputs/mask_visualizations")
    parser.add_argument("--max_cases", type=int, default=5)
    
    args = parser.parse_args()
    
    create_summary_visualization(args.dataset, args.output, args.max_cases)
