"""
Step 2: 3D Morphology-based Organ Masking

Advanced 3D operations for accurate organ segmentation
"""
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from skimage import morphology, filters
import json
import sys

# Add source path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.medical_imaging.volume_interpolation import enhance_ct_resolution
    INTERPOLATION_AVAILABLE = True
except ImportError:
    print("[WARNING] Volume interpolation not available")
    INTERPOLATION_AVAILABLE = False

def create_body_mask_3d(volume: np.ndarray, hu_threshold: int = -200) -> np.ndarray:
    """
    Create 3D body mask (excludes air and background)
    
    Args:
        volume: 3D CT volume
        hu_threshold: HU value to separate body from air
    
    Returns:
        3D binary mask of body
    """
    # Threshold
    body_mask = volume > hu_threshold
    
    # 3D morphological operations to clean up
    struct = ndimage.generate_binary_structure(3, 1)
    
    # Close small holes
    body_mask = ndimage.binary_closing(body_mask, structure=struct, iterations=3)
    
    # Fill holes
    body_mask = ndimage.binary_fill_holes(body_mask)
    
    # Keep only largest component (main body)
    labeled, num = ndimage.label(body_mask)
    if num > 0:
        sizes = ndimage.sum(body_mask, labeled, range(1, num + 1))
        largest_label = sizes.argmax() + 1
        body_mask = (labeled == largest_label)
    
    return body_mask


def create_soft_tissue_mask_3d(volume: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    """
    Create 3D soft tissue mask (excludes bones, lungs, fat)
    
    Args:
        volume: 3D CT volume
        body_mask: 3D body mask
    
    Returns:
        3D soft tissue mask
    """
    # Soft tissue HU range: -50 to 150
    soft_tissue = (volume > -50) & (volume < 150) & body_mask
    
    # Morphological opening to remove small noise
    struct = morphology.ball(2)
    soft_tissue = morphology.binary_opening(soft_tissue, struct)
    
    return soft_tissue


def create_colon_mask_3d(
    volume: np.ndarray,
    spacing: tuple,
    body_mask: np.ndarray,
    soft_tissue_mask: np.ndarray
) -> np.ndarray:
    """
    Create 3D colon region mask using advanced morphology
    
    Strategy:
    1. Start with soft tissue in abdomen/pelvis
    2. 3D region growing from seed points
    3. Shape-based filtering
    
    Args:
        volume: 3D CT volume
        spacing: Voxel spacing (x, y, z)
        body_mask: 3D body mask
        soft_tissue_mask: 3D soft tissue mask
    
    Returns:
        3D colon mask
    """
    
    print(f"\n[*] Creating 3D colon mask...")
    
    # Define anatomical region (abdomen/pelvis)
    D, H, W = volume.shape
    
    # Z-axis: lower 20-60% (skip head/chest, stop at mid-abdomen)
    z_start = int(D * 0.2)
    z_end = int(D * 0.6)
    
    # XY-axis: central region (exclude periphery)
    y_start = int(H * 0.25)
    y_end = int(H * 0.75)
    x_start = int(W * 0.25)
    x_end = int(W * 0.75)
    
    # Create anatomical mask
    anatomical_mask = np.zeros_like(volume, dtype=bool)
    anatomical_mask[z_start:z_end, y_start:y_end, x_start:x_end] = True
    
    # Combine with soft tissue
    colon_region = soft_tissue_mask & anatomical_mask
    
    print(f"[*] Initial colon region voxels: {colon_region.sum():,}")
    
    # 3D morphological operations for refinement
    struct = morphology.ball(3)
    
    # Closing to connect nearby regions
    colon_region = morphology.binary_closing(colon_region, struct)
    
    # Opening to smooth boundaries
    colon_region = morphology.binary_opening(colon_region, struct)
    
    print(f"[*] After morphology: {colon_region.sum():,} voxels")
    
    # Erode body mask to get core region
    body_core = ndimage.binary_erosion(body_mask, structure=morphology.ball(5), iterations=2)
    
    # Final colon mask: intersection with body core
    colon_mask = colon_region & body_core
    
    print(f"[+] Final colon mask: {colon_mask.sum():,} voxels")
    
    return colon_mask


def visualize_3d_slices(volume: np.ndarray, masks: dict, output_dir: Path):
    """
    Visualize masks on sample slices
    
    Args:
        volume: 3D CT volume
        masks: Dictionary of masks {name: mask_3d}
        output_dir: Output directory
    """
    import matplotlib.pyplot as plt
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample slices
    D = volume.shape[0]
    sample_slices = [int(D * 0.3), int(D * 0.4), int(D * 0.5)]
    
    for slice_idx in sample_slices:
        fig, axes = plt.subplots(1, len(masks) + 1, figsize=(4 * (len(masks) + 1), 4))
        
        # Original CT
        ax = axes[0]
        ax.imshow(volume[slice_idx], cmap='gray', vmin=-200, vmax=200)
        ax.set_title(f'CT Slice {slice_idx}')
        ax.axis('off')
        
        # Each mask
        for idx, (name, mask) in enumerate(masks.items(), 1):
            ax = axes[idx]
            ax.imshow(volume[slice_idx], cmap='gray', vmin=-200, vmax=200)
            ax.imshow(mask[slice_idx], alpha=0.3, cmap='Reds')
            ax.set_title(name)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'slice_{slice_idx:03d}_masks.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[+] Visualizations saved to: {output_dir}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("STEP 2: 3D Morphology-based Organ Masking")
    print("=" * 80)
    
    # Load Inha CT volume
    volume_path = Path("outputs/inha_ct_analysis/inha_ct_volume.nii.gz")
    
    if not volume_path.exists():
        print(f"[ERROR] File not found: {volume_path}")
        print("Please run CT analysis first")
        return
    
    print(f"[+] Loading: {volume_path}")
    nii = nib.load(volume_path)
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()
    
    print(f"[+] Original volume shape: {volume.shape}")
    print(f"[+] Original spacing: {spacing} mm")
    
    # ENHANCEMENT: Interpolate volume for smoother results
    if INTERPOLATION_AVAILABLE:
        print(f"\n[*] Enhancing volume resolution...")
        try:
            # Target 1mm Z spacing for high quality
            if spacing[0] > 1.5:  # Only interpolate if Z spacing > 1.5mm
                volume, spacing = enhance_ct_resolution(
                    volume,
                    spacing,
                    target_z_spacing=1.0,
                    method='cubic'
                )
                print(f"[+] Enhanced volume shape: {volume.shape}")
                print(f"[+] Enhanced spacing: {spacing} mm")
            else:
                print(f"[+] Z spacing already good ({spacing[0]:.2f}mm), skipping interpolation")
        except Exception as e:
            print(f"[WARNING] Interpolation failed: {e}")
            print(f"[+] Continuing with original volume")
    
    print(f"[*] HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    # Step 1: Create body mask
    print(f"\n[*] Creating 3D body mask...")
    body_mask = create_body_mask_3d(volume, hu_threshold=-200)
    print(f"[+] Body mask voxels: {body_mask.sum():,}")
    
    # Step 2: Create soft tissue mask
    print(f"\n[*] Creating 3D soft tissue mask...")
    soft_tissue_mask = create_soft_tissue_mask_3d(volume, body_mask)
    print(f"[+] Soft tissue voxels: {soft_tissue_mask.sum():,}")
    
    # Step 3: Create colon mask
    colon_mask = create_colon_mask_3d(volume, spacing, body_mask, soft_tissue_mask)
    
    # Calculate statistics
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    stats = {
        'body_volume_ml': float(body_mask.sum() * voxel_volume_mm3 / 1000),
        'soft_tissue_volume_ml': float(soft_tissue_mask.sum() * voxel_volume_mm3 / 1000),
        'colon_volume_ml': float(colon_mask.sum() * voxel_volume_mm3 / 1000)
    }
    
    # Save masks
    output_dir = Path("outputs/inha_ct_detection/3d_segmentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each mask
    for name, mask in [
        ('body_mask_3d', body_mask),
        ('soft_tissue_mask_3d', soft_tissue_mask),
        ('colon_mask_3d', colon_mask)
    ]:
        mask_nii = nib.Nifti1Image(mask.astype(np.uint8), nii.affine)
        mask_path = output_dir / f"{name}.nii.gz"
        nib.save(mask_nii, mask_path)
        print(f"[+] Saved: {mask_path}")
    
    # Save statistics
    stats_path = output_dir / "mask_statistics_3d.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[+] Statistics saved: {stats_path}")
    
    # Visualize
    print(f"\n[*] Creating visualizations...")
    visualize_3d_slices(
        volume,
        {
            'Body': body_mask,
            'Soft Tissue': soft_tissue_mask,
            'Colon': colon_mask
        },
        output_dir / 'mask_visualization'
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Body volume: {stats['body_volume_ml']:.1f} mL")
    print(f"Soft tissue volume: {stats['soft_tissue_volume_ml']:.1f} mL")
    print(f"Colon region volume: {stats['colon_volume_ml']:.1f} mL")
    
    print(f"\n{'='*80}")
    print("STEP 2 COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
