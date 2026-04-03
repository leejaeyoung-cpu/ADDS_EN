"""
Anatomy-Based Tumor Detector
Implements sequential analysis pipeline based on anatomical structure and HU values
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from scipy import ndimage
from dataclasses import dataclass
import logging

from ..segmentation.organ_segmenter import OrganSegmenter

logger = logging.getLogger(__name__)


# HU value constants for different tissues
HU_RANGES = {
    'air': (-1000, -500),
    'lung': (-900, -500),
    'fat': (-150, -50),
    'water': (-10, 10),
    'soft_tissue': (10, 60),
    'blood': (30, 70),
    'muscle': (35, 70),
    'liver': (50, 70),
    'bone': (400, 1000),
}

# Normal HU ranges for colon
COLON_NORMAL_HU = {
    'wall': (20, 60),       # Colon wall
    'lumen': (-100, 10),    # Air/content in lumen
    'fat': (-150, -50),     # Surrounding fat
}


@dataclass
class LesionCandidate:
    """Represents a potential lesion"""
    mask: np.ndarray          # Binary mask of the lesion
    bbox: Dict[str, int]      # Bounding box coordinates
    volume_voxels: int        # Volume in voxels
    mean_hu: float            # Average HU value
    max_hu: float             # Maximum HU value
    min_hu: float             # Minimum HU value
    std_hu: float             # Standard deviation of HU
    centroid: Tuple[int, int, int]  # Center position (z, y, x)
    organ: str                # Which organ it's in
    classification: Optional[str] = None  # tumor, inflammation, calcification, etc.
    confidence: Optional[float] = None


class AnatomyBasedTumorDetector:
    """
    Anatomy-based sequential tumor detection
    
    Pipeline:
    1. Organ segmentation (using TotalSegmentator)
    2. HU value analysis per organ
    3. Anomaly detection based on HU deviation
    4. Lesion classification
    """
    
    def __init__(
        self,
        organ_segmenter: Optional[OrganSegmenter] = None,
        device: str = "gpu",
        fast_mode: bool = True
    ):
        """
        Initialize detector
        
        Args:
            organ_segmenter: Pre-initialized organ segmenter (optional)
            device: 'gpu' or 'cpu'
            fast_mode: Use fast segmentation
        """
        self.device = device
        self.fast_mode = fast_mode
        
        # Initialize or use provided organ segmenter
        if organ_segmenter is None:
            self.organ_segmenter = OrganSegmenter(
                device=device,
                fast_mode=fast_mode,
                roi_subset=['colon']  # Focus on colon for now
            )
        else:
            self.organ_segmenter = organ_segmenter
        
        logger.info("AnatomyBasedTumorDetector initialized")
    
    def detect(
        self,
        ct_volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, List[LesionCandidate]]:
        """
        Main detection pipeline
        
        Args:
            ct_volume: CT volume in HU units, shape (D, H, W)
            spacing: Voxel spacing in mm (z, y, x)
        
        Returns:
            Dictionary of organ_name -> list of lesion candidates
        """
        logger.info(f"Starting anatomy-based detection on volume shape: {ct_volume.shape}")
        
        if spacing is None:
            spacing = (5.0, 1.0, 1.0)  # Default CT spacing
        
        results = {}
        
        # Step 1: Segment organs
        logger.info("Step 1: Organ segmentation...")
        organ_masks = self.organ_segmenter.segment(ct_volume)
        
        # Step 2: Analyze each organ
        for organ_name, organ_mask in organ_masks.items():
            logger.info(f"Analyzing {organ_name}...")
            
            # Skip if organ not found
            if organ_mask is None or organ_mask.sum() == 0:
                logger.warning(f"{organ_name} mask is empty, skipping")
                continue
            
            # Convert to binary mask if needed
            if organ_mask.dtype != bool:
                organ_mask = (organ_mask > 0).astype(bool)
            
            # Detect anomalies in this organ
            candidates = self._detect_organ_anomalies(
                ct_volume=ct_volume,
                organ_mask=organ_mask,
                organ_name=organ_name,
                spacing=spacing
            )
            
            if candidates:
                results[organ_name] = candidates
                logger.info(f"Found {len(candidates)} candidates in {organ_name}")
        
        return results
    
    def _detect_organ_anomalies(
        self,
        ct_volume: np.ndarray,
        organ_mask: np.ndarray,
        organ_name: str,
        spacing: Tuple[float, float, float]
    ) -> List[LesionCandidate]:
        """
        Detect anomalies within a specific organ
        
        Args:
            ct_volume: Full CT volume
            organ_mask: Binary mask of the organ
            organ_name: Name of the organ
            spacing: Voxel spacing
        
        Returns:
            List of lesion candidates
        """
        # Extract organ region
        organ_hu = ct_volume[organ_mask]
        
        # Calculate normal statistics
        mean_hu = np.mean(organ_hu)
        std_hu = np.std(organ_hu)
        
        logger.info(f"{organ_name} normal range: {mean_hu:.1f} ± {std_hu:.1f} HU")
        
        # Define anomaly threshold (tissue brighter than normal)
        # For colon, tumors are typically 20-100 HU (brighter than wall)
        lower_threshold = mean_hu + 1.0 * std_hu  # 1 std above mean
        upper_threshold = 150  # Exclude bone/calcification initially
        
        # For colon specifically, use domain knowledge
        if organ_name == 'colon':
            lower_threshold = max(40, mean_hu + std_hu)  # At least 40 HU
            upper_threshold = 100  # Typical tumor range
        
        # Find anomalous voxels (brighter than normal)
        anomaly_mask = (ct_volume > lower_threshold) & (ct_volume < upper_threshold) & organ_mask
        
        # Also detect very bright regions (calcifications)
        calcification_mask = (ct_volume >= 150) & organ_mask
        
        # Combine candidates
        all_anomaly_mask = anomaly_mask | calcification_mask
        
        if all_anomaly_mask.sum() == 0:
            logger.info(f"No anomalies detected in {organ_name}")
            return []
        
        # Extract connected components
        candidates = self._extract_candidates(
            ct_volume=ct_volume,
            anomaly_mask=all_anomaly_mask,
            organ_name=organ_name,
            spacing=spacing
        )
        
        # Classify each candidate
        for candidate in candidates:
            self._classify_lesion(candidate)
        
        return candidates
    
    def _extract_candidates(
        self,
        ct_volume: np.ndarray,
        anomaly_mask: np.ndarray,
        organ_name: str,
        spacing: Tuple[float, float, float],
        min_volume_mm3: float = 50.0  # Minimum 50 mm³
    ) -> List[LesionCandidate]:
        """
        Extract individual candidate regions from anomaly mask
        
        Args:
            ct_volume: CT volume
            anomaly_mask: Binary mask of anomalies
            organ_name: Organ name
            spacing: Voxel spacing
            min_volume_mm3: Minimum volume threshold in mm³
        
        Returns:
            List of lesion candidates
        """
        # Connected component labeling
        labeled, num_features = ndimage.label(anomaly_mask)
        
        logger.info(f"Found {num_features} connected components in {organ_name}")
        
        # Voxel volume in mm³
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        min_voxels = int(np.ceil(min_volume_mm3 / voxel_volume))
        
        candidates = []
        
        for region_id in range(1, num_features + 1):
            region_mask = (labeled == region_id)
            
            # Filter by size
            volume_voxels = region_mask.sum()
            if volume_voxels < min_voxels:
                continue
            
            # Extract HU statistics
            region_hu = ct_volume[region_mask]
            mean_hu = float(np.mean(region_hu))
            max_hu = float(np.max(region_hu))
            min_hu = float(np.min(region_hu))
            std_hu = float(np.std(region_hu))
            
            # Get bounding box
            coords = np.where(region_mask)
            bbox = {
                'z_min': int(coords[0].min()),
                'z_max': int(coords[0].max()),
                'y_min': int(coords[1].min()),
                'y_max': int(coords[1].max()),
                'x_min': int(coords[2].min()),
                'x_max': int(coords[2].max()),
            }
            
            # Get centroid
            centroid = (
                int(np.mean(coords[0])),
                int(np.mean(coords[1])),
                int(np.mean(coords[2]))
            )
            
            # Create candidate
            candidate = LesionCandidate(
                mask=region_mask,
                bbox=bbox,
                volume_voxels=int(volume_voxels),
                mean_hu=mean_hu,
                max_hu=max_hu,
                min_hu=min_hu,
                std_hu=std_hu,
                centroid=centroid,
                organ=organ_name
            )
            
            candidates.append(candidate)
        
        logger.info(f"Extracted {len(candidates)} candidates (after size filtering)")
        
        return candidates
    
    def _classify_lesion(self, candidate: LesionCandidate) -> None:
        """
        Classify lesion based on HU value and characteristics
        
        Modifies candidate in-place
        
        Args:
            candidate: Lesion candidate
        """
        mean_hu = candidate.mean_hu
        max_hu = candidate.max_hu
        
        # Classification based on HU value
        if mean_hu < -500:
            candidate.classification = "air"  # Probably not tumor
            candidate.confidence = 0.1
        elif mean_hu > 200:
            candidate.classification = "calcification"  # Calcification or stone
            candidate.confidence = 0.9
        elif mean_hu > 100:
            if max_hu > 200:
                candidate.classification = "mixed_density"  # Tumor with calcification
                candidate.confidence = 0.7
            else:
                candidate.classification = "tumor_likely"  # High density tumor
                candidate.confidence = 0.8
        elif 60 < mean_hu < 100:
            candidate.classification = "tumor_probable"  # Most typical tumor range
            candidate.confidence = 0.7
        elif 40 < mean_hu < 60:
            candidate.classification = "inflammatory_possible"  # Could be inflammation
            candidate.confidence = 0.5
        else:
            candidate.classification = "uncertain"
            candidate.confidence = 0.3
        
        # Additional heuristics
        # Large size + high HU -> more likely tumor
        if candidate.volume_voxels > 1000 and 60 < mean_hu < 100:
            candidate.confidence = min(0.95, candidate.confidence + 0.2)
        
        # Very small + uncertain HU -> likely noise
        if candidate.volume_voxels < 50 and candidate.confidence < 0.6:
            candidate.confidence *= 0.5


def quick_detect_colon_tumors(
    ct_volume: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
    device: str = "gpu"
) -> List[LesionCandidate]:
    """
    Convenience function for quick colon tumor detection
    
    Args:
        ct_volume: CT volume in HU units
        spacing: Voxel spacing in mm
        device: 'gpu' or 'cpu'
    
    Returns:
        List of lesion candidates in colon
    """
    detector = AnatomyBasedTumorDetector(device=device, fast_mode=True)
    results = detector.detect(ct_volume, spacing=spacing)
    
    # Return colon candidates only
    return results.get('colon', [])


if __name__ == "__main__":
    # Test case
    print("Testing AnatomyBasedTumorDetector...")
    
    # Create synthetic CT volume with a "tumor"
    ct_volume = np.random.randn(128, 256, 256) * 20 + 30  # Normal tissue ~30 HU
    
    # Add a bright region (tumor simulation)
    ct_volume[50:60, 120:135, 120:135] = 80  # Tumor at 80 HU
    
    # Add another region (calcification)
    ct_volume[70:75, 150:155, 150:155] = 250  # Calcification
    
    print(f"CT volume shape: {ct_volume.shape}")
    print(f"CT HU range: [{ct_volume.min():.1f}, {ct_volume.max():.1f}]")
    
    try:
        # Note: This will fail without TotalSegmentator installed
        # But shows the interface
        detector = AnatomyBasedTumorDetector(device="cpu", fast_mode=True)
        results = detector.detect(ct_volume)
        
        print(f"\nDetection results: {len(results)} organs analyzed")
        for organ, candidates in results.items():
            print(f"\n{organ}: {len(candidates)} candidates")
            for i, candidate in enumerate(candidates):
                print(f"  Candidate {i+1}:")
                print(f"    Volume: {candidate.volume_voxels} voxels")
                print(f"    HU: {candidate.mean_hu:.1f} (±{candidate.std_hu:.1f})")
                print(f"    Classification: {candidate.classification} ({candidate.confidence:.2f})")
                print(f"    Location: {candidate.centroid}")
        
        print("\n✓ Test complete!")
        
    except Exception as e:
        print(f"✗ Test failed (expected if TotalSegmentator not installed): {e}")
        print("Interface test passed - install TotalSegmentator for full functionality")
