"""
Candidate-based Tumor Detection Module
==========================================
OpenAI-style candidate detection with confidence scoring and rich feature extraction
Adapted from txt/tumor_detection_pipeline.py for ADDS integration

Author: ADDS Integration
Target: NIfTI (Medical Decathlon) + DICOM
"""

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure
from skimage.feature import peak_local_max
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. Data Classes
# ============================================================

@dataclass
class TumorCandidate:
    """Tumor candidate region data structure (OpenAI-style)"""
    centroid: Tuple[float, float]
    area_pixels: int
    area_mm2: float
    bounding_box: Tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    mean_hu: float
    max_hu: float
    min_hu: float
    eccentricity: float
    solidity: float
    perimeter: float
    circularity: float
    confidence_score: float
    slice_index: int = 0  # For 3D volumes
    mask_2d: Optional[np.ndarray] = None  # Actual segmentation mask from regionprops


# ============================================================
# 2. CT Preprocessing
# ============================================================

class CTPreprocessor:
    """CT image preprocessing"""
    
    # HU reference ranges
    HU_RANGES = {
        'air': (-1000, -500),
        'lung': (-500, -200),
        'fat': (-100, -50),
        'water': (-20, 20),
        'soft_tissue': (20, 80),
        'liver': (40, 60),
        'blood': (30, 45),
        'muscle': (35, 55),
        'bone': (300, 3000),
        'tumor_typical': (20, 100),
    }
    
    @staticmethod
    def apply_window(hu_image: np.ndarray, 
                     window_center: float, 
                     window_width: float) -> np.ndarray:
        """Apply window/level"""
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        windowed = np.clip(hu_image, min_val, max_val)
        # Normalize to 0-255
        normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return normalized
    
    @staticmethod
    def get_preset_window(preset: str) -> Tuple[float, float]:
        """Get predefined window settings"""
        presets = {
            'abdomen': (40, 400),
            'liver': (60, 150),
            'lung': (-600, 1500),
            'mediastinum': (50, 350),
            'bone': (300, 1500),
            'brain': (40, 80),
            'soft_tissue': (50, 400),
        }
        return presets.get(preset, (40, 400))
    
    @staticmethod
    def denoise(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """Denoise image"""
        if method == 'bilateral':
            return cv2.bilateralFilter(image.astype(np.float32), 9, 75, 75)
        elif method == 'gaussian':
            return ndimage.gaussian_filter(image, sigma=1.0)
        elif method == 'median':
            return ndimage.median_filter(image, size=3)
        return image


# ============================================================
# 3. Body Segmentation
# ============================================================

class BodySegmentation:
    """Body region segmentation"""
    
    @staticmethod
    def segment_body(hu_image: np.ndarray, threshold: float = -300) -> np.ndarray:
        """Create body region mask"""
        # Exclude air regions
        body_mask = hu_image > threshold
        
        # Morphological refinement
        body_mask = ndimage.binary_fill_holes(body_mask)
        body_mask = morphology.binary_closing(body_mask, morphology.disk(5))
        body_mask = morphology.binary_opening(body_mask, morphology.disk(3))
        
        # Keep only largest connected component
        labels = measure.label(body_mask)
        if labels.max() > 0:
            largest = np.argmax(np.bincount(labels.flat)[1:]) + 1
            body_mask = labels == largest
        
        return body_mask.astype(np.uint8)


# ============================================================
# 4. Tumor Detector (Core)
# ============================================================

class TumorDetector:
    """Tumor candidate detection (OpenAI style)"""
    
    def __init__(self, 
                 min_area_mm2: float = 10.0,  # CRITICAL: 10 not 1000!
                 max_area_mm2: float = 10000.0,
                 hu_range: Tuple[float, float] = (-50, 200)):
        self.min_area_mm2 = min_area_mm2
        self.max_area_mm2 = max_area_mm2
        self.hu_range = hu_range
    
    def detect_candidates_2d(self, 
                              hu_slice: np.ndarray,
                              pixel_spacing: Tuple[float, float],
                              body_mask: Optional[np.ndarray] = None,
                              slice_index: int = 0,
                              method: str = 'multi_threshold') -> List[TumorCandidate]:
        """Detect candidates in a single 2D slice"""
        
        if body_mask is None:
            body_mask = BodySegmentation.segment_body(hu_slice)
        
        if method == 'multi_threshold':
            return self._detect_multi_threshold(
                hu_slice, pixel_spacing, body_mask, slice_index
            )
        elif method == 'morphological':
            return self._detect_morphological(
                hu_slice, pixel_spacing, body_mask, slice_index
            )
        else:
            return self._detect_multi_threshold(
                hu_slice, pixel_spacing, body_mask, slice_index
            )
    
    def detect_candidates_3d(self,
                              volume: np.ndarray,
                              spacing: Tuple[float, float, float],
                              max_slices: int = None) -> List[TumorCandidate]:
        """
        Detect candidates in 3D volume (slice-by-slice)
        
        Args:
            volume: 3D numpy array (D, H, W)
            spacing: (spacing_z, spacing_y, spacing_x) in mm
            max_slices: Maximum number of slices to process (None = all)
        
        Returns:
            List of TumorCandidate objects
        """
        all_candidates = []
        depth = volume.shape[0]
        
        # Process subset of slices if specified
        if max_slices is not None:
            indices = np.linspace(0, depth - 1, min(max_slices, depth), dtype=int)
        else:
            indices = range(depth)
        
        pixel_spacing = (spacing[1], spacing[2])  # (y, x)
        
        for z in indices:
            slice_2d = volume[z, :, :]
            
            # Skip empty slices
            if slice_2d.max() - slice_2d.min() < 10:
                continue
            
            candidates = self.detect_candidates_2d(
                slice_2d, pixel_spacing, 
                body_mask=None, 
                slice_index=int(z),
                method='multi_threshold'
            )
            
            all_candidates.extend(candidates)
        
        # Sort by confidence
        all_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return all_candidates
    
    def _detect_multi_threshold(self, 
                                  hu_slice: np.ndarray,
                                  pixel_spacing: Tuple[float, float],
                                  body_mask: np.ndarray,
                                  slice_index: int) -> List[TumorCandidate]:
        """Multi-threshold based detection"""
        candidates = []
        
        # HU range mask
        hu_mask = (hu_slice >= self.hu_range[0]) & (hu_slice <= self.hu_range[1])
        combined_mask = hu_mask & body_mask.astype(bool)
        
        # Otsu threshold
        if combined_mask.any():
            masked_values = hu_slice[combined_mask]
            if len(np.unique(masked_values)) > 1:
                try:
                    thresh = filters.threshold_otsu(masked_values)
                    binary = (hu_slice > thresh) & combined_mask
                except:
                    binary = combined_mask
            else:
                binary = combined_mask
        else:
            return candidates
        
        # Morphological post-processing
        binary = morphology.binary_opening(binary, morphology.disk(2))
        binary = morphology.binary_closing(binary, morphology.disk(3))
        
        # Connected component analysis
        labels = measure.label(binary)
        regions = measure.regionprops(labels, intensity_image=hu_slice)
        
        # Filter candidate regions
        pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
        
        for region in regions:
            area_mm2 = region.area * pixel_area_mm2
            
            if self.min_area_mm2 <= area_mm2 <= self.max_area_mm2:
                candidate = self._create_candidate(
                    region, pixel_area_mm2, slice_index
                )
                candidates.append(candidate)
        
        return candidates
    
    def _detect_morphological(self,
                               hu_slice: np.ndarray,
                               pixel_spacing: Tuple[float, float],
                               body_mask: np.ndarray,
                               slice_index: int) -> List[TumorCandidate]:
        """Morphological feature-based detection"""
        candidates = []
        
        # Preprocessing
        enhanced = CTPreprocessor.denoise(hu_slice, method='bilateral')
        
        # Soft tissue region
        soft_tissue = (enhanced > 20) & (enhanced < 100) & body_mask.astype(bool)
        
        # Top-hat transform for bright lesions
        selem = morphology.disk(15)
        tophat = morphology.white_tophat(enhanced * soft_tissue, selem)
        
        # Threshold
        thresh = np.percentile(tophat[tophat > 0], 90) if tophat.max() > 0 else 0
        binary = tophat > thresh
        
        # Connected component analysis
        labels = measure.label(binary)
        regions = measure.regionprops(labels, intensity_image=hu_slice)
        
        pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
        
        for region in regions:
            area_mm2 = region.area * pixel_area_mm2
            if self.min_area_mm2 <= area_mm2 <= self.max_area_mm2:
                candidate = self._create_candidate(
                    region, pixel_area_mm2, slice_index
                )
                candidates.append(candidate)
        
        return candidates
    
    def _create_candidate(self, 
                           region, 
                           pixel_area_mm2: float,
                           slice_index: int) -> TumorCandidate:
        """Create TumorCandidate from RegionProperties"""
        # Calculate circularity
        if region.perimeter > 0:
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
        else:
            circularity = 0
        
        # Calculate confidence score
        confidence = self._calculate_confidence(region, circularity)
        
        # Extract actual segmentation mask from region
        # region.image contains the binary mask of this specific region
        mask_2d = region.image.copy() if hasattr(region, 'image') else None
        
        return TumorCandidate(
            centroid=region.centroid,
            area_pixels=region.area,
            area_mm2=region.area * pixel_area_mm2,
            bounding_box=region.bbox,
            mean_hu=region.mean_intensity,
            max_hu=region.max_intensity,
            min_hu=region.min_intensity,
            eccentricity=region.eccentricity,
            solidity=region.solidity,
            perimeter=region.perimeter,
            circularity=circularity,
            confidence_score=confidence,
            slice_index=slice_index,
            mask_2d=mask_2d  # Store actual mask!
        )
    
    def _calculate_confidence(self, region, circularity: float) -> float:
        """
        Calculate tumor candidate confidence score
        
        Scoring criteria (OpenAI style):
        - Circularity: 0.3 points (0.5-1.0 range)
        - Solidity: 0.3 points (>0.8)
        - Eccentricity: 0.2 points (<0.7)
        - HU value: 0.2 points (20-80 range)
        
        Returns: score ∈ [0, 1]
        """
        score = 0.0
        
        # Circularity score (tumors are relatively round)
        if 0.5 <= circularity <= 1.0:
            score += 0.3 * circularity
        
        # Solidity score (filled interior)
        if region.solidity > 0.8:
            score += 0.3
        elif region.solidity > 0.6:
            score += 0.2
        
        # Eccentricity score (not too elongated)
        if region.eccentricity < 0.7:
            score += 0.2
        elif region.eccentricity < 0.85:
            score += 0.1
        
        # HU value score (soft tissue range)
        mean_hu = region.mean_intensity
        if 20 <= mean_hu <= 80:
            score += 0.2
        elif 0 <= mean_hu <= 100:
            score += 0.1
        
        return min(score, 1.0)


# ============================================================
# 5. Utility Functions
# ============================================================

def merge_candidates(candidates: List[TumorCandidate],
                      distance_threshold: float = 20.0) -> List[TumorCandidate]:
    """Merge duplicate candidates"""
    if not candidates:
        return []
    
    merged = []
    used = set()
    
    for i, c1 in enumerate(candidates):
        if i in used:
            continue
        
        # Find nearby candidates
        to_merge = [c1]
        for j, c2 in enumerate(candidates[i+1:], i+1):
            if j in used:
                continue
            
            # Same slice check
            if c1.slice_index == c2.slice_index:
                dist = np.sqrt((c1.centroid[0] - c2.centroid[0])**2 + 
                               (c1.centroid[1] - c2.centroid[1])**2)
                
                if dist < distance_threshold:
                    to_merge.append(c2)
                    used.add(j)
        
        # Select highest confidence
        best = max(to_merge, key=lambda x: x.confidence_score)
        merged.append(best)
        used.add(i)
    
    return merged
