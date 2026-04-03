#!/usr/bin/env python3
"""
DICOM to NIfTI Converter for TotalSegmentator
Prepares CT data for organ segmentation
"""

import pydicom
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List
import sys

def load_dicom_series(dicom_dir: Path, series_prefix: str = "200") -> tuple:
    """
    Load DICOM series
    
    Args:
        dicom_dir: Directory containing DICOM files
        series_prefix: Prefix for series number (e.g., "200" for arterial phase)
    
    Returns:
        volume: 3D numpy array (Z, Y, X)
        metadata: DICOM metadata
    """
    # Find all matching files
    dcm_files = sorted(dicom_dir.glob(f"{series_prefix}*.dcm"))
    
    if not dcm_files:
        print(f"No DICOM files found with prefix {series_prefix}")
        # Try all files
        dcm_files = sorted(dicom_dir.glob("*.dcm"))
    
    print(f"Found {len(dcm_files)} DICOM files")
    
    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")
    
    # Read first file for metadata
    first_dcm = pydicom.dcmread(dcm_files[0])
    
    # Read all slices and collect dimensions
    slices_data = []
    for dcm_file in dcm_files:
        dcm = pydicom.dcmread(dcm_file)
        image = dcm.pixel_array.astype(float)
        
        # Apply HU conversion
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            image = image * dcm.RescaleSlope + dcm.RescaleIntercept
        
        slices_data.append((image, dcm))
    
    # Find most common dimensions
    dimensions = {}
    for img, _ in slices_data:
        shape = img.shape
        dimensions[shape] = dimensions.get(shape, 0) + 1
    
    # Use most common dimension
    target_shape = max(dimensions.items(), key=lambda x: x[1])[0]
    rows, cols = target_shape
    n_slices = len([img for img, _ in slices_data if img.shape == target_shape])
    
    print(f"Target dimensions: {n_slices} slices x {rows} x {cols}")
    print(f"Filtered {len(dcm_files) - n_slices} slices with different dimensions")
    
    # Initialize volume
    volume = np.zeros((n_slices, rows, cols), dtype=np.float32)
    
    # Fill volume with matching slices only
    idx = 0
    for image, dcm in slices_data:
        if image.shape == target_shape:
            volume[idx] = image
            idx += 1
            if idx == 1:  # Use first valid slice for metadata
                first_dcm = dcm

    
    # Get spacing
    if hasattr(first_dcm, 'PixelSpacing'):
        pixel_spacing = first_dcm.PixelSpacing
    else:
        pixel_spacing = [1.0, 1.0]
    
    if hasattr(first_dcm, 'SliceThickness'):
        slice_thickness = first_dcm.SliceThickness
    else:
        slice_thickness = 1.0
    
    spacing = [slice_thickness, pixel_spacing[0], pixel_spacing[1]]
    
    metadata = {
        'spacing': spacing,
        'origin': [0, 0, 0],
        'direction': np.eye(3),
        'patient_id': getattr(first_dcm, 'PatientID', 'Unknown'),
        'series_description': getattr(first_dcm, 'SeriesDescription', 'Unknown')
    }
    
    return volume, metadata

def save_as_nifti(volume: np.ndarray, metadata: dict, output_path: Path):
    """
    Save volume as NIfTI file
    
    Args:
        volume: 3D numpy array
        metadata: Metadata dictionary
        output_path: Output NIfTI file path
    """
    # Create affine matrix
    affine = np.eye(4)
    affine[0, 0] = metadata['spacing'][2]  # X spacing
    affine[1, 1] = metadata['spacing'][1]  # Y spacing
    affine[2, 2] = metadata['spacing'][0]  # Z spacing
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume, affine)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_img, str(output_path))
    
    print(f"Saved NIfTI: {output_path}")
    print(f"  Shape: {volume.shape}")
    print(f"  Spacing: {metadata['spacing']}")
    print(f"  Min HU: {volume.min():.1f}")
    print(f"  Max HU: {volume.max():.1f}")

def main():
    """Convert Inha CT DICOM to NIfTI"""
    
    print("\n" + "="*70)
    print("DICOM to NIfTI Converter")
    print("="*70 + "\n")
    
    # Input/output paths
    dicom_dir = Path("CTdata/CTdcm")
    output_file = Path("CTdata/nifti/inha_ct_arterial.nii.gz")
    
    # Load DICOM series
    print("Loading DICOM series...")
    volume, metadata = load_dicom_series(dicom_dir, series_prefix="200")
    
    # Save as NIfTI
    print("\nSaving as NIfTI...")
    save_as_nifti(volume, metadata, output_file)
    
    print("\n" + "="*70)
    print("Conversion complete!")
    print("="*70)
    print(f"\nOutput: {output_file}")
    print(f"\nNext step:")
    print(f"  TotalSegmentator -i {output_file} -o CTdata/segmentation")

if __name__ == '__main__':
    main()
