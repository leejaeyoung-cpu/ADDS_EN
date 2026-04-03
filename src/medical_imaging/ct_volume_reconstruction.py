"""
Stage 1: 3D CT Volume Reconstruction and Preprocessing
Converts DICOM series to 3D volume with standardized spacing and orientation
"""

import numpy as np
import SimpleITK as sitk
import pydicom
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class CTVolumeReconstructor:
    """3D CT Volume Reconstruction from DICOM series"""
    
    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initialize reconstructor
        
        Args:
            target_spacing: Target isotropic spacing in mm (x, y, z)
        """
        self.target_spacing = target_spacing
        
    def load_dicom_series(self, dicom_folder: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load and sort DICOM series
        
        Args:
            dicom_folder: Path to folder containing DICOM files
            
        Returns:
            volume: 3D numpy array
            metadata: Dictionary with spacing, origin, direction
        """
        logger.info(f"Loading DICOM series from {dicom_folder}")
        
        # Use SimpleITK to read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_folder))
        
        if not dicom_names:
            raise ValueError(f"No DICOM files found in {dicom_folder}")
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Extract volume and metadata
        volume = sitk.GetArrayFromImage(image)  # Shape: (slices, height, width)
        
        metadata = {
            'spacing': image.GetSpacing(),  # (x, y, z)
            'origin': image.GetOrigin(),
            'direction': image.GetDirection(),
            'size': image.GetSize()
        }
        
        logger.info(f"Loaded volume: {volume.shape}, spacing: {metadata['spacing']}")
        
        return volume, metadata
    
    def apply_hounsfield_conversion(self, volume: np.ndarray, 
                                   intercept: float = -1024, 
                                   slope: float = 1.0) -> np.ndarray:
        """
        Convert pixel values to Hounsfield Units (HU)
        
        Args:
            volume: Raw pixel values
            intercept: Rescale intercept from DICOM
            slope: Rescale slope from DICOM
            
        Returns:
            volume_hu: Volume in Hounsfield Units
        """
        volume_hu = volume * slope + intercept
        return volume_hu
    
    def resample_to_isotropic(self, volume: np.ndarray, 
                              current_spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Resample volume to isotropic spacing
        
        Args:
            volume: Input volume (z, y, x)
            current_spacing: Current spacing (x, y, z)
            
        Returns:
            resampled_volume: Isotropic volume
        """
        # Calculate zoom factors
        # Note: spacing is (x, y, z) but volume is (z, y, x)
        zoom_factors = [
            current_spacing[2] / self.target_spacing[2],  # z
            current_spacing[1] / self.target_spacing[1],  # y
            current_spacing[0] / self.target_spacing[0]   # x
        ]
        
        logger.info(f"Resampling with zoom factors: {zoom_factors}")
        
        # Use scipy's zoom with linear interpolation
        resampled_volume = ndimage.zoom(volume, zoom_factors, order=1)
        
        logger.info(f"Resampled from {volume.shape} to {resampled_volume.shape}")
        
        return resampled_volume
    
    def normalize_intensity(self, volume: np.ndarray, 
                          window_min: float = -150, 
                          window_max: float = 250) -> np.ndarray:
        """
        Normalize intensity using abdominal window
        
        Args:
            volume: Input volume in HU
            window_min: Minimum HU for abdomen window
            window_max: Maximum HU for abdomen window
            
        Returns:
            normalized: Normalized volume [0, 1]
        """
        # Clip to window
        clipped = np.clip(volume, window_min, window_max)
        
        # Normalize to [0, 1]
        normalized = (clipped - window_min) / (window_max - window_min)
        
        return normalized.astype(np.float32)
    
    def reconstruct_3d_volume(self, dicom_folder: Path, 
                             normalize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Complete reconstruction pipeline
        
        Args:
            dicom_folder: Path to DICOM series
            normalize: Whether to normalize intensity
            
        Returns:
            volume: Reconstructed 3D volume
            metadata: Processing metadata
        """
        # Step 1: Load DICOM series
        volume, dicom_metadata = self.load_dicom_series(dicom_folder)
        
        # Step 2: Apply HU conversion (usually already done by SimpleITK)
        # But we'll keep the interface for manual conversion if needed
        
        # Step 3: Resample to isotropic
        current_spacing = dicom_metadata['spacing']
        volume_resampled = self.resample_to_isotropic(volume, current_spacing)
        
        # Step 4: Normalize intensity
        if normalize:
            volume_normalized = self.normalize_intensity(volume_resampled)
        else:
            volume_normalized = volume_resampled
        
        # Update metadata
        metadata = {
            'original_shape': volume.shape,
            'resampled_shape': volume_normalized.shape,
            'original_spacing': current_spacing,
            'target_spacing': self.target_spacing,
            'is_normalized': normalize,
            **dicom_metadata
        }
        
        logger.info(f"Reconstruction complete: {metadata['resampled_shape']}")
        
        return volume_normalized, metadata
    
    def save_as_nifti(self, volume: np.ndarray, output_path: Path, 
                     metadata: Optional[Dict] = None):
        """
        Save volume as NIfTI file
        
        Args:
            volume: 3D volume
            output_path: Output .nii.gz file path
            metadata: Optional metadata
        """
        image = sitk.GetImageFromArray(volume)
        
        if metadata:
            image.SetSpacing(self.target_spacing)
            if 'origin' in metadata:
                image.SetOrigin(metadata['origin'])
            if 'direction' in metadata:
                image.SetDirection(metadata['direction'])
        
        sitk.WriteImage(image, str(output_path), True)  # True = use compression
        logger.info(f"Saved NIfTI: {output_path}")


def reconstruct_patient_ct(dicom_folder: Path, 
                          output_dir: Optional[Path] = None) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function for single patient CT reconstruction
    
    Args:
        dicom_folder: Path to patient DICOM series
        output_dir: Optional output directory for NIfTI
        
    Returns:
        volume: Reconstructed 3D volume
        metadata: Metadata dictionary
    """
    reconstructor = CTVolumeReconstructor()
    volume, metadata = reconstructor.reconstruct_3d_volume(dicom_folder)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        nifti_path = output_dir / f"{dicom_folder.name}_reconstructed.nii.gz"
        reconstructor.save_as_nifti(volume, nifti_path, metadata)
    
    return volume, metadata


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ct_volume_reconstruction.py <dicom_folder>")
        sys.exit(1)
    
    dicom_folder = Path(sys.argv[1])
    output_dir = Path("outputs/reconstructed_volumes")
    
    logging.basicConfig(level=logging.INFO)
    
    volume, metadata = reconstruct_patient_ct(dicom_folder, output_dir)
    
    print(f"\n✓ Volume reconstructed: {volume.shape}")
    print(f"✓ Spacing: {metadata['target_spacing']}")
    print(f"✓ Normalized: {metadata['is_normalized']}")
