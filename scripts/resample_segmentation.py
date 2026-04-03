"""
Resample segmentation to match reconstructed CT volume
"""

import numpy as np
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
import logging
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def resample_segmentation_to_volume(seg_path: Path,
                                   volume_path: Path,
                                   output_path: Path):
    """
    Resample segmentation to match volume geometry
    
    Args:
        seg_path: Path to segmentation (original spacing)
        volume_path: Path to reconstructed volume (target geometry)
        output_path: Path to save resampled segmentation
    """
    logger.info("="*80)
    logger.info("RESAMPLING SEGMENTATION TO MATCH VOLUME")
    logger.info("="*80)
    
    # Load volume to get target geometry
    logger.info(f"Loading target volume from: {volume_path}")
    volume_nifti = nib.load(volume_path)
    target_shape = volume_nifti.shape
    target_affine = volume_nifti.affine
    target_header = volume_nifti.header
    
    logger.info(f"  Target shape: {target_shape}")
    logger.info(f"  Target spacing: {target_header.get_zooms()}")
    
    # Load segmentation
    logger.info(f"\nLoading segmentation from: {seg_path}")
    seg_nifti = nib.load(seg_path)
    seg_data = seg_nifti.get_fdata().astype(np.uint8)
    seg_spacing = seg_nifti.header.get_zooms()
    
    logger.info(f"  Original shape: {seg_data.shape}")
    logger.info(f"  Original spacing: {seg_spacing}")
    
    # Calculate zoom factors
    zoom_factors = [
        seg_spacing[i] / target_header.get_zooms()[i]
        for i in range(3)
    ]
    
    logger.info(f"\nZoom factors: {zoom_factors}")
    
    # Resample using nearest neighbor (for labels)
    logger.info("Resampling (nearest neighbor for labels)...")
    resampled_seg = ndimage.zoom(
        seg_data,
        zoom_factors,
        order=0,  # Nearest neighbor
        mode='nearest'
    )
    
    logger.info(f"Resampled shape: {resampled_seg.shape}")
    
    # Crop or pad to match exact target shape
    if resampled_seg.shape != target_shape:
        logger.info(f"Adjusting shape from {resampled_seg.shape} to {target_shape}")
        
        final_seg = np.zeros(target_shape, dtype=np.uint8)
        
        # Calculate slices for cropping/padding
        slices = []
        for i in range(3):
            if resampled_seg.shape[i] > target_shape[i]:
                # Crop
                start = (resampled_seg.shape[i] - target_shape[i]) // 2
                slices.append(slice(start, start + target_shape[i]))
            else:
                # Will pad
                slices.append(slice(None))
        
        # Get cropped data
        cropped = resampled_seg[slices[0], slices[1], slices[2]]
        
        # Pad if necessary
        pad_widths = []
        for i in range(3):
            if cropped.shape[i] < target_shape[i]:
                diff = target_shape[i] - cropped.shape[i]
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_widths.append((pad_before, pad_after))
            else:
                pad_widths.append((0, 0))
        
        if any(p != (0, 0) for p in pad_widths):
            final_seg = np.pad(cropped, pad_widths, mode='constant', constant_values=0)
        else:
            final_seg = cropped
        
        resampled_seg = final_seg
    
    logger.info(f"Final shape: {resampled_seg.shape}")
    
    # Verify labels
    unique_labels = np.unique(resampled_seg)
    logger.info(f"Labels in resampled seg: {unique_labels}")
    
    # Save with same geometry as volume
    logger.info(f"\nSaving to: {output_path}")
    resampled_nifti = nib.Nifti1Image(resampled_seg, target_affine, target_header)
    nib.save(resampled_nifti, output_path)
    
    logger.info("="*80)
    logger.info("RESAMPLING COMPLETE")
    logger.info("="*80)
    
    return resampled_seg


if __name__ == "__main__":
    seg_path = Path("F:/ADDS/CTdata/segmentation_remapped.nii.gz")
    volume_path = Path("F:/ADDS/outputs/ct_pipeline_test/reconstructed_volume.nii.gz")
    output_path = Path("F:/ADDS/outputs/ct_pipeline_test/segmentation_resampled.nii.gz")
    
    resampled_seg = resample_segmentation_to_volume(seg_path, volume_path, output_path)
    
    print(f"\n✅ Segmentation resampled successfully")
    print(f"   Shape: {resampled_seg.shape}")
    print(f"   Output: {output_path}")
