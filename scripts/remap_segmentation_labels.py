"""
Fix segmentation labels to be consecutive for nnU-Net
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def remap_labels_to_consecutive(segmentation_path: Path, output_path: Path):
    """
    Remap segmentation labels to consecutive integers
    
    Args:
        segmentation_path: Path to original segmentation
        output_path: Path to save remapped segmentation
    """
    logger.info(f"Loading segmentation from {segmentation_path}")
    
    # Load
    nifti = nib.load(segmentation_path)
    seg_data = nifti.get_fdata().astype(np.uint8)
    
    # Get unique labels
    unique_labels = np.unique(seg_data)
    logger.info(f"Original labels: {unique_labels}")
    
    # Create mapping: original_label -> consecutive_label
    label_mapping = {0: 0}  # Background stays 0
    
    consecutive_label = 1
    for orig_label in unique_labels:
        if orig_label > 0:  # Skip background
            label_mapping[orig_label] = consecutive_label
            consecutive_label += 1
    
    logger.info(f"Label mapping: {label_mapping}")
    
    # Apply mapping
    remapped = np.zeros_like(seg_data)
    for orig, new in label_mapping.items():
        remapped[seg_data == orig] = new
    
    # Verify
    new_labels = np.unique(remapped)
    logger.info(f"New labels: {new_labels}")
    
    # Save
    new_nifti = nib.Nifti1Image(remapped, nifti.affine, nifti.header)
    nib.save(new_nifti, output_path)
    
    logger.info(f"Saved remapped segmentation to {output_path}")
    
    return label_mapping, new_labels


if __name__ == "__main__":
    seg_path = Path("F:/ADDS/CTdata/segmentation.nii")
    output_path = Path("F:/ADDS/CTdata/segmentation_remapped.nii.gz")
    
    label_mapping, new_labels = remap_labels_to_consecutive(seg_path, output_path)
    
    print("\n" + "="*80)
    print("LABEL REMAPPING COMPLETE")
    print("="*80)
    print(f"Original labels: {len(label_mapping)} classes")
    print(f"New labels: {new_labels}")
    print(f"Output: {output_path}")
    print("="*80)
