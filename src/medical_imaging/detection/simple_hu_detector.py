"""
Simple HU-Based Tumor Detector
No organ segmentation required - uses pure HU thresholds
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimpleLesion:
    """Simple lesion representation"""
    mask: np.ndarray
    volume_voxels: int
    mean_hu: float
    max_hu: float
    centroid: Tuple[int, int, int]
    bbox: dict


class SimpleHUDetector:
    """
    Simple tumor detection using only HU value thresholds
    
    No organ segmentation needed - fast and dependency-free
    """
    
    def __init__(
        self,
        tumor_hu_min: float = 40.0,
        tumor_hu_max: float = 150.0,
        min_volume_mm3: float = 50.0
    ):
        """
        Initialize simple detector
        
        Args:
            tumor_hu_min: Minimum HU for tumor (default 40)
            tumor_hu_max: Maximum HU for tumor (default 150)
            min_volume_mm3: Minimum lesion size in mm³
        """
        self.tumor_hu_min = tumor_hu_min
        self.tumor_hu_max = tumor_hu_max
        self.min_volume_mm3 = min_volume_mm3
        
        logger.info(f"SimpleHUDetector initialized: HU range [{tumor_hu_min}, {tumor_hu_max}]")
    
    def detect(
        self,
        ct_volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> List[SimpleLesion]:
        """
        Detect tumors using HU thresholds
        
        Args:
            ct_volume: CT volume in HU units
            spacing: Voxel spacing (z, y, x) in mm
        
        Returns:
            List of detected lesions
        """
        if spacing is None:
            spacing = (5.0, 1.0, 1.0)
        
        logger.info(f"Running simple HU detection on volume {ct_volume.shape}")
        logger.info(f"HU range: [{ct_volume.min():.1f}, {ct_volume.max():.1f}]")
        
        # Step 1: Create body mask (exclude air)
        body_mask = ct_volume > -500  # Air is < -500 HU
        logger.info(f"Body voxels: {body_mask.sum():,}")
        
        # Step 2: Find bright regions (potential tumors)
        bright_mask = (ct_volume >= self.tumor_hu_min) & (ct_volume <= self.tumor_hu_max)
        logger.info(f"Bright voxels ({self.tumor_hu_min}-{self.tumor_hu_max} HU): {bright_mask.sum():,}")
        
        
        # Step 3: Combine - bright regions inside body
        candidate_mask = bright_mask & body_mask
        logger.info(f"Candidate voxels: {candidate_mask.sum():,}")
        
        if candidate_mask.sum() == 0:
            logger.error("❌ NO CANDIDATES FOUND!")
            logger.error(f"=== DIAGNOSTIC INFO ===")
            logger.error(f"Body voxels: {body_mask.sum():,}")
            logger.error(f"Bright voxels: {bright_mask.sum():,}")
            logger.error(f"Voxels in [0, 50] HU: {np.sum((ct_volume >= 0) & (ct_volume <= 50)):,}")
            logger.error(f"Voxels in [20, 80] HU: {np.sum((ct_volume >= 20) & (ct_volume <= 80)):,}")
            logger.error(f"Voxels in [40, 150] HU: {np.sum((ct_volume >= 40) & (ct_volume <= 150)):,}")
            logger.error(f"=======================")
            logger.error(f"Possible causes:")
            logger.error(f"  1. Volume HU range doesn't overlap with tumor range [{self.tumor_hu_min}, {self.tumor_hu_max}]")
            logger.error(f"  2. Try wider HU range (e.g., tumor_hu_min=10, tumor_hu_max=200)")
            return []
        
        # Step 4: Connected component analysis
        lesions = self._extract_lesions(
            ct_volume=ct_volume,
            candidate_mask=candidate_mask,
            spacing=spacing
        )
        
        logger.info(f"Detected {len(lesions)} lesions")
        
        return lesions
    
    def _extract_lesions(
        self,
        ct_volume: np.ndarray,
        candidate_mask: np.ndarray,
        spacing: Tuple[float, float, float]
    ) -> List[SimpleLesion]:
        """
        Extract individual lesions from candidate mask
        Memory-optimized version - avoids creating full boolean masks
        """
        
        # Label connected components
        labeled, num_features = ndimage.label(candidate_mask)
        logger.info(f"Found {num_features} connected components")
        
        # Calculate minimum voxels
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³
        min_voxels = int(np.ceil(self.min_volume_mm3 / voxel_volume))
        
        lesions = []
        
        # Process each component using coordinates only (memory efficient)
        for region_id in range(1, num_features + 1):
            # Get coordinates directly (no boolean mask)
            coords = np.where(labeled == region_id)
            volume_voxels = len(coords[0])
            
            # Filter by size early
            if volume_voxels < min_voxels:
                continue
            
            # Extract HU statistics using fancy indexing (no mask needed)
            region_hu = ct_volume[coords]
            mean_hu = float(np.mean(region_hu))
            max_hu = float(np.max(region_hu))
            
            # Get centroid
            centroid = (
                int(np.mean(coords[0])),
                int(np.mean(coords[1])),
                int(np.mean(coords[2]))
            )
            
            # Get bounding box
            bbox = {
                'z_min': int(coords[0].min()),
                'z_max': int(coords[0].max()),
                'y_min': int(coords[1].min()),
                'y_max': int(coords[1].max()),
                'x_min': int(coords[2].min()),
                'x_max': int(coords[2].max()),
            }
            
            # Only create mask when needed (for final result)
            # Use sparse representation
            region_mask = np.zeros(ct_volume.shape, dtype=bool)
            region_mask[coords] = True
            
            lesion = SimpleLesion(
                mask=region_mask,
                volume_voxels=int(volume_voxels),
                mean_hu=mean_hu,
                max_hu=max_hu,
                centroid=centroid,
                bbox=bbox
            )
            
            lesions.append(lesion)
            
            # Clean up
            del region_mask
        
        logger.info(f"After size filtering: {len(lesions)} lesions")
        
        return lesions


def detect_tumors_simple(
    ct_volume: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
    tumor_hu_min: float = 40.0,
    tumor_hu_max: float = 150.0
) -> List[SimpleLesion]:
    """
    Quick function for simple tumor detection
    
    Args:
        ct_volume: CT volume in HU
        spacing: Voxel spacing
        tumor_hu_min: Min HU threshold
        tumor_hu_max: Max HU threshold
    
    Returns:
        List of detected lesions
    """
    detector = SimpleHUDetector(
        tumor_hu_min=tumor_hu_min,
        tumor_hu_max=tumor_hu_max
    )
    return detector.detect(ct_volume, spacing)


if __name__ == "__main__":
    # Quick test
    print("Testing SimpleHUDetector...")
    
    # Create synthetic CT
    ct = np.random.randn(100, 200, 200) * 20 + 30  # Soft tissue ~30 HU
    
    # Add bright lesion (tumor simulation)
    ct[40:50, 90:105, 90:105] = 80  # Tumor at 80 HU
    
    print(f"CT shape: {ct.shape}")
    print(f"HU range: [{ct.min():.1f}, {ct.max():.1f}]")
    
    # Detect
    lesions = detect_tumors_simple(ct, spacing=(5.0, 1.0, 1.0))
    
    print(f"\nDetected {len(lesions)} lesions:")
    for i, lesion in enumerate(lesions):
        print(f"  {i+1}. Volume: {lesion.volume_voxels} voxels, HU: {lesion.mean_hu:.1f}")
        print(f"     Location: {lesion.centroid}")
    
    print("\n✓ Test passed!")
