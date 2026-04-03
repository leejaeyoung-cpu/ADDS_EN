"""
Classical Volume Interpolation for Medical CT Scans

Provides fast, high-quality volume interpolation using classical methods
(trilinear, cubic) as an immediate improvement before deep learning models.
"""

import numpy as np
from scipy import ndimage
from pathlib import Path
import nibabel as nib
from typing import Tuple, Literal, Optional
import logging

logger = logging.getLogger(__name__)


class VolumeInterpolator:
    """Classical interpolation methods for CT volumes"""
    
    def __init__(self, method: Literal['linear', 'cubic', 'quintic'] = 'cubic'):
        """
        Initialize volume interpolator
        
        Args:
            method: Interpolation method
                - 'linear': Trilinear interpolation (fastest)
                - 'cubic': Cubic spline (best quality/speed balance)
                - 'quintic': 5th order spline (highest quality, slower)
        """
        self.method = method
        self.order_map = {
            'linear': 1,
            'cubic': 3,
            'quintic': 5
        }
        self.order = self.order_map[method]
    
    def interpolate_volume(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Optional[Tuple[float, float, float]] = None,
        interpolation_factor: Optional[float] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Interpolate CT volume to higher resolution
        
        Args:
            volume: Input volume (Z, Y, X)
            original_spacing: Original voxel spacing (z, y, x) in mm
            target_spacing: Desired voxel spacing (optional)
            interpolation_factor: Factor to reduce spacing by (optional)
                E.g., factor=2 means halve the spacing (double resolution)
        
        Returns:
            Tuple of (interpolated_volume, new_spacing)
        
        Example:
            # Double Z resolution
            vol_hires, new_spacing = interpolator.interpolate_volume(
                volume, 
                original_spacing=(3.0, 0.7, 0.7),
                target_spacing=(1.5, 0.7, 0.7)
            )
        """
        logger.info(f"Interpolating volume with {self.method} method")
        logger.info(f"Original shape: {volume.shape}, spacing: {original_spacing}")
        
        # Determine zoom factors
        if target_spacing is not None:
            # Calculate zoom from spacing ratio
            zoom_factors = tuple(
                orig / target 
                for orig, target in zip(original_spacing, target_spacing)
            )
            new_spacing = target_spacing
        elif interpolation_factor is not None:
            # Use uniform interpolation factor
            zoom_factors = (interpolation_factor, interpolation_factor, interpolation_factor)
            new_spacing = tuple(s / interpolation_factor for s in original_spacing)
        else:
            raise ValueError("Must provide either target_spacing or interpolation_factor")
        
        logger.info(f"Zoom factors: {zoom_factors}")
        
        # Perform interpolation
        interpolated_volume = ndimage.zoom(
            volume,
            zoom_factors,
            order=self.order,
            mode='nearest',  # Use nearest for boundaries
            prefilter=True   # Apply anti-aliasing filter
        )
        
        logger.info(f"Interpolated shape: {interpolated_volume.shape}, spacing: {new_spacing}")
        
        return interpolated_volume, new_spacing
    
    def interpolate_slices_only(
        self,
        volume: np.ndarray,
        z_factor: float = 2.0
    ) -> np.ndarray:
        """
        Interpolate only along Z-axis (between slices)
        
        Keeps X,Y resolution unchanged - optimized for CT scans where
        in-plane resolution is already high but slice spacing is large.
        
        Args:
            volume: Input volume (Z, Y, X)
            z_factor: Factor to increase slice count by
        
        Returns:
            Interpolated volume with more slices
        """
        logger.info(f"Interpolating Z-axis only with factor {z_factor}")
        
        zoom_factors = (z_factor, 1.0, 1.0)
        
        interpolated_volume = ndimage.zoom(
            volume,
            zoom_factors,
            order=self.order,
            mode='nearest',
            prefilter=True
        )
        
        logger.info(f"Z-interpolation: {volume.shape[0]} -> {interpolated_volume.shape[0]} slices")
        
        return interpolated_volume


def interpolate_nifti_file(
    input_path: Path,
    output_path: Path,
    method: str = 'cubic',
    z_factor: float = 2.0,
    preserve_dtype: bool = True
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load, interpolate, and save NIfTI file
    
    Args:
        input_path: Path to input .nii.gz file
        output_path: Path to save interpolated .nii.gz
        method: Interpolation method ('linear', 'cubic', 'quintic')
        z_factor: Z-axis interpolation factor
        preserve_dtype: Keep original data type after interpolation
    
    Returns:
        Tuple of (interpolated_data, new_spacing)
    """
    logger.info(f"Loading {input_path}")
    
    # Load NIfTI
    nii = nib.load(input_path)
    volume = nii.get_fdata()
    original_dtype = volume.dtype
    
    # Get spacing
    spacing = nii.header.get_zooms()[:3]
    
    # Interpolate
    interpolator = VolumeInterpolator(method=method)
    
    if z_factor != 1.0:
        # Z-only interpolation
        interpolated = interpolator.interpolate_slices_only(volume, z_factor=z_factor)
        new_spacing = (spacing[0] / z_factor, spacing[1], spacing[2])
    else:
        # Full 3D interpolation
        interpolated, new_spacing = interpolator.interpolate_volume(
            volume,
            original_spacing=spacing,
            interpolation_factor=2.0
        )
    
    # Preserve data type if requested
    if preserve_dtype:
        interpolated = interpolated.astype(original_dtype)
    
    # Create new NIfTI with updated spacing
    new_affine = nii.affine.copy()
    new_affine[0, 0] = new_affine[0, 0] / (interpolated.shape[2] / volume.shape[2])
    new_affine[1, 1] = new_affine[1, 1] / (interpolated.shape[1] / volume.shape[1])
    new_affine[2, 2] = new_affine[2, 2] / (interpolated.shape[0] / volume.shape[0])
    
    new_nii = nib.Nifti1Image(interpolated, new_affine, nii.header)
    
    # Update spacing in header
    new_nii.header.set_zooms(new_spacing)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new_nii, output_path)
    
    logger.info(f"Saved interpolated volume to {output_path}")
    logger.info(f"Spacing: {spacing} -> {new_spacing}")
    
    return interpolated, new_spacing


# Convenience function for CT pipeline integration
def enhance_ct_resolution(
    ct_volume: np.ndarray,
    spacing: Tuple[float, float, float],
    target_z_spacing: float = 1.0,
    method: str = 'cubic'
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Enhance CT volume resolution, primarily along Z-axis
    
    This is the main function to use in CT analysis pipelines.
    
    Args:
        ct_volume: Input CT volume (Z, Y, X)
        spacing: Current voxel spacing (z, y, x) in mm
        target_z_spacing: Desired Z spacing in mm
        method: Interpolation method
    
    Returns:
        Tuple of (enhanced_volume, new_spacing)
    
    Example:
        # In CT analysis pipeline
        ct_volume, spacing = load_ct_scan(...)
        
        # Enhance to 1mm Z spacing
        ct_enhanced, new_spacing = enhance_ct_resolution(
            ct_volume, spacing, target_z_spacing=1.0
        )
        
        # Continue with tumor detection on enhanced volume
        tumors = detect_tumors(ct_enhanced, ...)
    """
    interpolator = VolumeInterpolator(method=method)
    
    z_current, y_current, x_current = spacing
    
    # Calculate Z factor needed
    z_factor = z_current / target_z_spacing
    
    if z_factor <= 1.0:
        logger.info(f"Current Z spacing ({z_current}mm) already <= target ({target_z_spacing}mm)")
        return ct_volume, spacing
    
    logger.info(f"Enhancing Z resolution: {z_current}mm -> {target_z_spacing}mm (factor: {z_factor:.2f})")
    
    # Interpolate Z-axis only (preserve in-plane resolution)
    enhanced = interpolator.interpolate_slices_only(ct_volume, z_factor=z_factor)
    new_spacing = (target_z_spacing, y_current, x_current)
    
    return enhanced, new_spacing


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("Volume Interpolation Example")
    print("="*80)
    
    # Create synthetic volume
    test_volume = np.random.randn(50, 256, 256).astype(np.float32)
    test_spacing = (3.0, 1.0, 1.0)  # 3mm Z spacing, 1mm XY
    
    print(f"\nOriginal volume: {test_volume.shape}")
    print(f"Original spacing: {test_spacing}")
    
    # Enhance resolution
    enhanced, new_spacing = enhance_ct_resolution(
        test_volume,
        test_spacing,
        target_z_spacing=1.0,
        method='cubic'
    )
    
    print(f"\nEnhanced volume: {enhanced.shape}")
    print(f"Enhanced spacing: {new_spacing}")
    print(f"\nZ slices: {test_volume.shape[0]} -> {enhanced.shape[0]}")
    print(f"Improvement: {enhanced.shape[0] / test_volume.shape[0]:.2f}x")
