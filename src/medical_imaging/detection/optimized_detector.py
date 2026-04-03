"""
Optimized Tumor Detector with False Positive Filtering
========================================================
Enhanced version with colon-specific confidence and FP reduction
Based on comprehensive validation results
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, measure
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .candidate_detector import (
    TumorCandidate, 
    TumorDetector, 
    CTPreprocessor,
    BodySegmentation
)


class OptimizedColonDetector(TumorDetector):
    """
    Optimized detector for colon tumors with FP reduction
    
    Improvements over base detector:
    - Colon-specific confidence scoring
    - Organ-based filtering (liver exclusion)
    - Vessel detection filter
    - Extreme HU filtering
    """
    
    def __init__(self,
                 min_area_mm2: float = 20.0,  # Optimized from validation
                 max_area_mm2: float = 5000.0,
                 hu_range: Tuple[float, float] = (-50, 200),
                 enable_fp_filtering: bool = True,
                 enable_colon_scoring: bool = True):
        
        super().__init__(min_area_mm2, max_area_mm2, hu_range)
        
        self.enable_fp_filtering = enable_fp_filtering
        self.enable_colon_scoring = enable_colon_scoring
    
    def detect_candidates_2d(self,
                              hu_slice: np.ndarray,
                              pixel_spacing: Tuple[float, float],
                              body_mask: Optional[np.ndarray] = None,
                              slice_index: int = 0,
                              method: str = 'multi_threshold') -> List[TumorCandidate]:
        """Enhanced detection with FP filtering"""
        
        # 1. Base detection
        candidates = super().detect_candidates_2d(
            hu_slice, pixel_spacing, body_mask, slice_index, method
        )
        
        if not candidates:
            return candidates
        
        # 2. False positive filtering
        if self.enable_fp_filtering:
            candidates = self.filter_false_positives(
                candidates, hu_slice, body_mask
            )
        
        # 3. Colon-specific re-scoring
        if self.enable_colon_scoring:
            for c in candidates:
                c.confidence_score = self.calculate_colon_confidence(c, hu_slice)
        
        # 4. Sort by confidence
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return candidates
    
    def filter_false_positives(self,
                                 candidates: List[TumorCandidate],
                                 hu_slice: np.ndarray,
                                 body_mask: Optional[np.ndarray]) -> List[TumorCandidate]:
        """
        Filter out likely false positives
        
        Based on validation findings:
        - 87% FP rate (154/177)
        - Main sources: liver, vessels, bowel wall
        """
        
        filtered = []
        
        # Generate liver mask if available
        liver_mask = None
        if body_mask is not None:
            try:
                liver_mask = self._segment_liver_approximate(hu_slice, body_mask)
            except:
                liver_mask = None
        
        for c in candidates:
            # Filter 1: Vessel detection (small, highly circular)
            if self._is_likely_vessel(c):
                continue
            
            # Filter 2: Extreme HU values
            if self._has_extreme_hu(c):
                continue
            
            # Filter 3: Liver region (require higher confidence)
            if liver_mask is not None and self._is_in_liver(c, liver_mask):
                # Only keep if high baseline confidence
                if c.confidence_score < 0.5:
                    continue
            
            # Filter 4: Too small or too large
            if c.area_mm2 < 15 or c.area_mm2 > 4000:
                continue
            
            filtered.append(c)
        
        return filtered
    
    def _is_likely_vessel(self, candidate: TumorCandidate) -> bool:
        """Detect vessel cross-sections"""
        # Vessels are small, circular, with soft tissue HU
        is_small = candidate.area_mm2 < 40
        is_circular = candidate.circularity > 0.92
        is_soft_tissue = 30 <= candidate.mean_hu <= 60
        
        return is_small and is_circular and is_soft_tissue
    
    def _has_extreme_hu(self, candidate: TumorCandidate) -> bool:
        """Filter extreme HU values"""
        # Colon tumors typically in 80-160 HU range (but allow margin)
        return candidate.mean_hu < 10 or candidate.mean_hu > 180
    
    def _is_in_liver(self, 
                      candidate: TumorCandidate, 
                      liver_mask: np.ndarray) -> bool:
        """Check if candidate is in liver region"""
        bbox = candidate.bounding_box
        min_row, min_col, max_row, max_col = bbox
        
        # Check overlap with liver mask
        roi = liver_mask[min_row:max_row, min_col:max_col]
        overlap_ratio = roi.sum() / roi.size if roi.size > 0 else 0
        
        # If >50% in liver, consider it liver region
        return overlap_ratio > 0.5
    
    def _segment_liver_approximate(self,
                                     hu_slice: np.ndarray,
                                     body_mask: np.ndarray) -> np.ndarray:
        """
        Approximate liver segmentation
        Simple HU-based approach (not perfect but helps FP reduction)
        """
        # Liver HU range: 40-70 typically
        liver_hu_mask = (hu_slice >= 35) & (hu_slice <= 75)
        liver_mask = liver_hu_mask & body_mask.astype(bool)
        
        # Morphological operations
        liver_mask = morphology.binary_opening(liver_mask, morphology.disk(3))
        liver_mask = morphology.binary_closing(liver_mask, morphology.disk(10))
        
        # Keep largest component (likely liver)
        labels = measure.label(liver_mask)
        if labels.max() > 0:
            # Get top 2 largest regions (left and right lobe)
            sizes = np.bincount(labels.flat)[1:]
            if len(sizes) > 0:
                top_indices = np.argsort(sizes)[-2:] + 1
                liver_mask = np.isin(labels, top_indices)
        
        return liver_mask.astype(np.uint8)
    
    def calculate_colon_confidence(self,
                                     candidate: TumorCandidate,
                                     hu_slice: np.ndarray) -> float:
        """
        Colon-specific confidence scoring
        
        Based on validation insights:
        - Colon tumors: HU 100-140 (higher than soft tissue)
        - Larger lesions more likely tumors
        - Less strict shape requirements (irregular)
        """
        score = 0.0
        
        # 1. Size scoring (30%) - prefer larger lesions
        if candidate.area_mm2 > 200:
            score += 0.30
        elif candidate.area_mm2 > 100:
            score += 0.25
        elif candidate.area_mm2 > 50:
            score += 0.15
        else:
            score += 0.05
        
        # 2. HU scoring (35%) - colon tumor specific
        mean_hu = candidate.mean_hu
        if 100 <= mean_hu <= 140:
            # Optimal colon tumor HU
            score += 0.35
        elif 80 <= mean_hu <= 160:
            # Acceptable range
            score += 0.25
        elif 60 <= mean_hu <= 180:
            # Wider range (possible)
            score += 0.15
        else:
            # Outside typical range
            score += 0.05
        
        # 3. Shape scoring (20%) - less strict for irregular tumors
        circularity = candidate.circularity
        if 0.4 <= circularity <= 0.85:
            # Moderately irregular (typical for tumors)
            score += 0.20
        elif 0.25 <= circularity <= 0.95:
            # Wider range
            score += 0.12
        else:
            # Too irregular or too circular
            score += 0.05
        
        # 4. Solidity scoring (15%)
        if candidate.solidity > 0.75:
            score += 0.15
        elif candidate.solidity > 0.60:
            score += 0.10
        else:
            score += 0.05
        
        return min(score, 1.0)
    
    def detect_candidates_3d(self,
                              volume: np.ndarray,
                              spacing: Tuple[float, float, float],
                              max_slices: int = None) -> List[TumorCandidate]:
        """Enhanced 3D detection with optimization"""
        
        # Use base 3D detection
        candidates = super().detect_candidates_3d(volume, spacing, max_slices)
        
        # Additional 3D filtering (optional)
        # Could add continuity checks, 3D clustering, etc.
        
        return candidates


# Convenience function
def create_optimized_detector(
    fp_filtering: bool = True,
    colon_scoring: bool = True
) -> OptimizedColonDetector:
    """
    Create optimized detector with best parameters from validation
    
    Args:
        fp_filtering: Enable false positive filtering
        colon_scoring: Enable colon-specific confidence scoring
    
    Returns:
        Configured OptimizedColonDetector
    """
    return OptimizedColonDetector(
        min_area_mm2=20.0,        # Optimized value
        max_area_mm2=5000.0,      # Optimized value
        hu_range=(-50, 200),      # Best performing range
        enable_fp_filtering=fp_filtering,
        enable_colon_scoring=colon_scoring
    )
