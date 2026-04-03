"""
Anatomical Mask Creator for nnU-Net Dataset
============================================
Creates body and colon masks using TotalSegmentator
"""
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np


def run_totalsegmentator(input_ct, output_dir):
    """Run TotalSegmentator on single CT scan"""
    print(f"  Running TotalSegmentator...")
    
    cmd = [
        "TotalSegmentator",
        "-i", str(input_ct),
        "-o", str(output_dir),
        "--fast",
        "--ml",
        "--device", "gpu"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  [OK] TotalSegmentator complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] TotalSegmentator failed: {e}")
        return False
    except FileNotFoundError:
        print("  [ERROR] TotalSegmentator not found!")
        print("     Install with: pip install TotalSegmentator")
        return False


def extract_colon_mask(segmentation_file):
    """
    Extract colon + small bowel from TotalSegmentator output
    
    TotalSegmentator label IDs:
    - 14: Colon
    - 15: Small bowel
    """
    seg_nii = nib.load(segmentation_file)
    seg_data = seg_nii.get_fdata()
    
    # Extract colon (14) and small bowel (15)
    colon_mask = (seg_data == 14) | (seg_data == 15)
    
    return colon_mask.astype(np.uint8), seg_nii.affine


def create_body_mask(ct_data):
    """
    Create body mask using HU thresholding + connected components
    
    Strategy:
    1. HU threshold (> -500)
    2. Morphological cleanup (closing only)
    3. Remove bottom region FIRST (table location)
    4. Connected components (select largest = patient body)
    5. Fill holes (preserves internal air in organs)
    6. Additional bottom cleanup
    7. Minimal smoothing
    """
    from scipy.ndimage import binary_closing, binary_erosion, binary_dilation, binary_fill_holes, label
    
    # Step 1: HU thresholding
    body_mask = ct_data > -500
    
    # Step 2: Morphological cleanup - closing only
    body_mask = binary_closing(body_mask, iterations=3)
    
    # Step 3: REMOVE BOTTOM REGION FIRST (before connected components)
    # This ensures table is separated even if connected in some slices
    height = body_mask.shape[1]
    y_cutoff = int(height * 0.10)  # Reduced from 15% to 10%
    body_mask[:, :y_cutoff, :] = 0
    
    # Also remove very bottom in X direction (table edges)
    width = body_mask.shape[0]
    x_cutoff = int(width * 0.05)  # Bottom 5% in X
    body_mask[:x_cutoff, :, :] = 0
    body_mask[-x_cutoff:, :, :] = 0
    
    # Step 4: Connected Components Analysis (3D)
    # Now table should be completely separated
    labeled_array, num_features = label(body_mask)
    
    if num_features > 0:
        # Find the largest connected component (patient body)
        component_sizes = []
        for i in range(1, num_features + 1):
            size = (labeled_array == i).sum()
            component_sizes.append((i, size))
        
        # Sort by size (descending)
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the largest component (patient body)
        largest_label = component_sizes[0][0]
        body_mask = (labeled_array == largest_label)
        
        # Optional: Also keep 2nd largest if it's >20% of largest
        if num_features > 1:
            second_largest_label = component_sizes[1][0]
            second_size = component_sizes[1][1]
            largest_size = component_sizes[0][1]
            
            if second_size > largest_size * 0.2:
                body_mask = body_mask | (labeled_array == second_largest_label)
    
    # Step 5: Fill holes (preserves internal air)
    body_mask = binary_fill_holes(body_mask)
    
    # Step 6: Additional bottom cleanup (safety measure)
    # Remove any remaining bottom region
    y_final_cutoff = int(height * 0.08)  # Reduced from 12% to 8%
    body_mask[:, :y_final_cutoff, :] = 0
    
    # Step 7: Minimal smoothing
    body_mask = binary_erosion(body_mask, iterations=1)
    body_mask = binary_dilation(body_mask, iterations=1)
    
    return body_mask.astype(np.uint8)


def process_single_case(case_id, dataset_dir, temp_dir, use_totalsegmentator=True):
    """Process single training case"""
    print(f"\n[{case_id}]")
    
    images_dir = dataset_dir / "imagesTr"
    colon_masks_dir = dataset_dir / "colon_masks"
    body_masks_dir = dataset_dir / "body_masks"
    
    # Load CT image
    ct_file = images_dir / f"{case_id}_0000.nii.gz"
    if not ct_file.exists():
        print(f"  [ERROR] CT file not found: {ct_file}")
        return False
    
    ct_nii = nib.load(ct_file)
    ct_data = ct_nii.get_fdata()
    affine = ct_nii.affine
    
    # 1. TotalSegmentator (if available)
    if use_totalsegmentator:
        # TotalSegmentator with --ml outputs to: output_path.nii (not a directory!)
        seg_output_file = temp_dir / f"{case_id}.nii"
        
        if run_totalsegmentator(ct_file, seg_output_file):
            # Check if output file exists
            if seg_output_file.exists():
                colon_mask, _ = extract_colon_mask(seg_output_file)
                
                colon_mask_nii = nib.Nifti1Image(colon_mask, affine)
                colon_mask_path = colon_masks_dir / f"{case_id}_colon.nii.gz"
                nib.save(colon_mask_nii, colon_mask_path)
                print(f"  [OK] Colon mask saved: {colon_mask_path.name}")
            else:
                print(f"  [WARNING] Segmentation file not found: {seg_output_file}")
        else:
            print(f"  [WARNING] TotalSegmentator failed, skipping colon mask")
    
    # 2. Body mask (always create)
    body_mask = create_body_mask(ct_data)
    body_mask_nii = nib.Nifti1Image(body_mask, affine)
    body_mask_path = body_masks_dir / f"{case_id}_body.nii.gz"
    nib.save(body_mask_nii, body_mask_path)
    print(f"  [OK] Body mask saved: {body_mask_path.name}")
    
    return True


def create_masks_for_dataset(dataset_dir, max_cases=None, use_totalsegmentator=True):
    """Create anatomical masks for all training cases"""
    dataset_dir = Path(dataset_dir)
    
    # Create output directories
    colon_masks_dir = dataset_dir / "colon_masks"
    body_masks_dir = dataset_dir / "body_masks"
    temp_dir = dataset_dir / "temp_segmentations"
    
    colon_masks_dir.mkdir(exist_ok=True)
    body_masks_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    
    # Get all training images
    images_dir = dataset_dir / "imagesTr"
    image_files = sorted(images_dir.glob("*_0000.nii.gz"))
    
    if max_cases:
        image_files = image_files[:max_cases]
    
    print(f"\n{'='*60}")
    print(f"CREATING ANATOMICAL MASKS")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_dir.name}")
    print(f"Cases to process: {len(image_files)}")
    print(f"TotalSegmentator: {'Yes' if use_totalsegmentator else 'No (body masks only)'}")
    print(f"{'='*60}")
    
    # Process each case
    success_count = 0
    for img_file in image_files:
        # Extract case ID: colon_000_0000.nii.gz -> colon_000
        case_id = img_file.name.replace("_0000.nii.gz", "")
        
        if process_single_case(case_id, dataset_dir, temp_dir, use_totalsegmentator):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {success_count}/{len(image_files)} cases")
    print(f"Output directories:")
    print(f"  - Body masks: {body_masks_dir}")
    if use_totalsegmentator:
        print(f"  - Colon masks: {colon_masks_dir}")
    print(f"{'='*60}")
    
    # Cleanup temp directory (optional)
    response = input("\nDelete temporary segmentation files? (y/n): ").strip().lower()
    if response == 'y':
        import shutil
        shutil.rmtree(temp_dir)
        print("[OK] Temp directory deleted")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create anatomical masks for nnU-Net dataset")
    parser.add_argument("--dataset", type=str, default="f:/ADDS/nnUNet_raw/Dataset010_Colon",
                       help="Path to nnU-Net dataset")
    parser.add_argument("--max_cases", type=int, default=None,
                       help="Maximum number of cases to process")
    parser.add_argument("--no_totalsegmentator", action="store_true",
                       help="Skip TotalSegmentator (body masks only)")
    
    args = parser.parse_args()
    
    create_masks_for_dataset(
        dataset_dir=args.dataset,
        max_cases=args.max_cases,
        use_totalsegmentator=not args.no_totalsegmentator
    )


if __name__ == "__main__":
    main()
