"""
ROI (Region of Interest) Extraction Module

Extracts focused regions from CT volumes based on organ segmentation,
enabling targeted tumor detection with reduced false positives.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ROIExtractor:
    """
    Extract regions of interest from CT volumes
    
    Useful for:
    - Focusing tumor detection on specific organs
    - Reducing computational cost
    - Decreasing false positives in irrelevant regions
    """
    
    def __init__(
        self,
        margin_mm: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        min_size: Optional[int] = None
    ):
        """
        Initialize ROI extractor
        
        Args:
            margin_mm: Safety margin in mm (z, y, x)
            min_size: Minimum ROI size in voxels (filters tiny regions)
        """
        self.margin_mm = margin_mm
        self.min_size = min_size or 1000  # Default: 1000 voxels
    
    def extract_roi(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None,
        return_bbox: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Extract ROI from volume using mask
        
        Args:
            volume: Input CT volume (D, H, W) or (C, D, H, W)
            mask: Binary mask indicating region (D, H, W)
            spacing: Voxel spacing in mm (z, y, x)
            return_bbox: Return bounding box info
        
        Returns:
            roi_volume: Cropped volume
            bbox_info: Bounding box information (if return_bbox=True)
        """
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        
        # Check mask validity
        if mask.sum() < self.min_size:
            logger.warning(f"Mask too small ({mask.sum()} voxels < {self.min_size})")
            logger.warning("Returning original volume")
            return volume, None
        
        # Get bounding box from mask
        bbox = self._compute_bbox(mask)
        
        # Apply safety margin
        bbox_with_margin = self._apply_margin(
            bbox, 
            volume.shape[-3:],  # Get spatial dimensions (D, H, W)
            spacing
        )
        
        # Extract ROI
        if volume.ndim == 3:
            # Single channel (D, H, W)
            roi = volume[
                bbox_with_margin['z_min']:bbox_with_margin['z_max'],
                bbox_with_margin['y_min']:bbox_with_margin['y_max'],
                bbox_with_margin['x_min']:bbox_with_margin['x_max']
            ]
        elif volume.ndim == 4:
            # Multi-channel (C, D, H, W)
            roi = volume[
                :,
                bbox_with_margin['z_min']:bbox_with_margin['z_max'],
                bbox_with_margin['y_min']:bbox_with_margin['y_max'],
                bbox_with_margin['x_min']:bbox_with_margin['x_max']
            ]
        else:
            raise ValueError(f"Unsupported volume dimensions: {volume.ndim}")
        
        logger.info(f"Extracted ROI: {volume.shape} -> {roi.shape}")
        
        bbox_info = bbox_with_margin if return_bbox else None
        
        return roi, bbox_info
    
    def map_to_original(
        self,
        roi_result: np.ndarray,
        bbox: Dict,
        original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Map ROI result back to original coordinate system
        
        Args:
            roi_result: Result from ROI processing (D', H', W')
            bbox: Bounding box used for extraction
            original_shape: Original volume shape (D, H, W)
        
        Returns:
            Full volume with ROI result placed back
        """
        # Create empty volume
        full_result = np.zeros(original_shape, dtype=roi_result.dtype)
        
        # Place ROI result back
        full_result[
            bbox['z_min']:bbox['z_max'],
            bbox['y_min']:bbox['y_max'],
            bbox['x_min']:bbox['x_max']
        ] = roi_result
        
        logger.info(f"Mapped ROI back to original: {roi_result.shape} -> {original_shape}")
        
        return full_result
    
    def extract_multiple_rois(
        self,
        volume: np.ndarray,
        masks: Dict[str, np.ndarray],
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Tuple[np.ndarray, Dict]]:
        """
        Extract multiple ROIs from different organs
        
        Args:
            volume: CT volume
            masks: Dictionary of organ_name -> mask
            spacing: Voxel spacing
        
        Returns:
            Dictionary of organ_name -> (roi, bbox)
        """
        rois = {}
        
        for organ_name, mask in masks.items():
            roi, bbox = self.extract_roi(volume, mask, spacing)
            rois[organ_name] = (roi, bbox)
        
        return rois
    
    def _compute_bbox(self, mask: np.ndarray) -> Dict[str, int]:
        """Compute tight bounding box from mask"""
        coords = np.where(mask > 0)
        
        if len(coords[0]) == 0:
            # Empty mask
            return {
                'z_min': 0, 'z_max': mask.shape[0],
                'y_min': 0, 'y_max': mask.shape[1],
                'x_min': 0, 'x_max': mask.shape[2]
            }
        
        return {
            'z_min': int(coords[0].min()),
            'z_max': int(coords[0].max() + 1),  # +1 for inclusive slicing
            'y_min': int(coords[1].min()),
            'y_max': int(coords[1].max() + 1),
            'x_min': int(coords[2].min()),
            'x_max': int(coords[2].max() + 1)
        }
    
    def _apply_margin(
        self,
        bbox: Dict[str, int],
        volume_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float]
    ) -> Dict[str, int]:
        """Apply safety margin to bounding box"""
        # Convert margin from mm to voxels
        margin_voxels = (
            int(self.margin_mm[0] / spacing[0]),
            int(self.margin_mm[1] / spacing[1]),
            int(self.margin_mm[2] / spacing[2])
        )
        
        # Apply margin with bounds checking
        return {
            'z_min': max(0, bbox['z_min'] - margin_voxels[0]),
            'z_max': min(volume_shape[0], bbox['z_max'] + margin_voxels[0]),
            'y_min': max(0, bbox['y_min'] - margin_voxels[1]),
            'y_max': min(volume_shape[1], bbox['y_max'] + margin_voxels[1]),
            'x_min': max(0, bbox['x_min'] - margin_voxels[2]),
            'x_max': min(volume_shape[2], bbox['x_max'] + margin_voxels[2])
        }
    
    def get_roi_coverage(
        self,
        bbox: Dict[str, int],
        volume_shape: Tuple[int, int, int]
    ) -> float:
        """
        Calculate what percentage of original volume the ROI covers
        
        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        roi_size = (
            (bbox['z_max'] - bbox['z_min']) *
            (bbox['y_max'] - bbox['y_min']) *
            (bbox['x_max'] - bbox['x_min'])
        )
        
        total_size = volume_shape[0] * volume_shape[1] * volume_shape[2]
        
        return roi_size / total_size


# Convenience function
def extract_organ_roi(
    ct_volume: np.ndarray,
    organ_mask: np.ndarray,
    margin_mm: float = 20.0,
    spacing: Optional[Tuple[float, float, float]] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Quick function to extract organ ROI
    
    Args:
        ct_volume: CT volume
        organ_mask: Binary organ mask
        margin_mm: Safety margin
        spacing: Voxel spacing
    
    Returns:
        (roi_volume, bbox_info)
    """
    extractor = ROIExtractor(margin_mm=(margin_mm, margin_mm, margin_mm))
    return extractor.extract_roi(ct_volume, organ_mask, spacing)


if __name__ == "__main__":
    # Test case
    print("Testing ROIExtractor...")
    
    # Create dummy volume and mask
    volume = np.random.randn(128, 256, 256).astype(np.float32)
    mask = np.zeros((128, 256, 256), dtype=bool)
    mask[40:80, 100:180, 100:180] = True  # Simulate organ region
    
    extractor = ROIExtractor(margin_mm=(10.0, 10.0, 10.0))
    
    # Extract ROI
    roi, bbox = extractor.extract_roi(volume, mask, spacing=(1.5, 1.0, 1.0))
    
    print(f"Original shape: {volume.shape}")
    print(f"ROI shape: {roi.shape}")
    print(f"Bounding box: {bbox}")
    
    # Test mapping back
    dummy_result = np.ones(roi.shape)
    full_result = extractor.map_to_original(dummy_result, bbox, volume.shape)
    
    print(f"Mapped back shape: {full_result.shape}")
    print(f"ROI coverage: {extractor.get_roi_coverage(bbox, volume.shape)*100:.1f}%")
    
    print("✓ ROIExtractor test passed!")
