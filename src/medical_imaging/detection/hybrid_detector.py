"""
Hybrid Tumor Detector
Combines Simple HU detection with Organ Segmentation for optimal performance
"""

import numpy as np
from typing import List, Optional, Tuple
import logging
from dataclasses import dataclass

from src.medical_imaging.detection.simple_hu_detector import (
    SimpleHUDetector, SimpleLesion
)
from src.medical_imaging.segmentation.swin_segmenter import SwinUNETRSegmenter

logger = logging.getLogger(__name__)


@dataclass
class HybridLesion:
    """Enhanced lesion with organ information"""
    # From simple detection
    mask: np.ndarray
    volume_voxels: int
    mean_hu: float
    max_hu: float
    centroid: Tuple[int, int, int]
    bbox: dict
    
    # Enhanced info
    organ: str = "unknown"
    organ_overlap_ratio: float = 0.0
    confidence: float = 0.0


class HybridTumorDetector:
    """
    Hybrid tumor detection system
    
    Modes:
    - 'fast': HU threshold only (1-2 seconds)
    - 'accurate': HU + organ segmentation (10-30 seconds)  
    - 'auto': 2-stage pipeline (recommended)
    
    Example:
        detector = HybridTumorDetector(mode='auto')
        results = detector.detect(ct_volume, spacing)
    """
    
    def __init__(
        self,
        mode: str = 'auto',
        tumor_hu_min: float = 40.0,
        tumor_hu_max: float = 150.0,
        min_volume_mm3: float = 50.0,
        use_gpu: bool = True,
        pretrained_path: str = "models/pretrained/swin_unetr_pretrained.pt",
        segmenter_type: str = 'swin'  # 'swin' or 'totalseg'
    ):
        """
        Initialize hybrid detector
        
        Args:
            mode: 'fast', 'accurate', or 'auto'
            tumor_hu_min: Minimum HU for tumor
            tumor_hu_max: Maximum HU for tumor
            min_volume_mm3: Minimum lesion size
            use_gpu: Use GPU for organ segmentation
            pretrained_path: Path to pretrained Swin-UNETR
            segmenter_type: 'swin' (Swin-UNETR) or 'totalseg' (TotalSegmentator)
        """
        self.mode = mode
        self.segmenter_type = segmenter_type
        
        # Always create fast detector
        self.simple_detector = SimpleHUDetector(
            tumor_hu_min=tumor_hu_min,
            tumor_hu_max=tumor_hu_max,
            min_volume_mm3=min_volume_mm3
        )
        
        # Create organ segmenter if needed
        self.organ_segmenter = None
        if mode in ['accurate', 'auto']:
            try:
                device = 'cuda' if use_gpu else 'cpu'
                
                if segmenter_type == 'swin':
                    # Use Swin-UNETR
                    self.organ_segmenter = SwinUNETRSegmenter(
                        pretrained_path=pretrained_path,
                        device=device
                    )
                    logger.info(f"Swin-UNETR segmenter initialized (device: {device})")
                    
                elif segmenter_type == 'totalseg':
                    # Use TotalSegmentator
                    from src.medical_imaging.segmentation.organ_segmenter import OrganSegmenter
                    
                    self.organ_segmenter = OrganSegmenter(
                        device='gpu' if use_gpu else 'cpu',
                        fast_mode=True
                    )
                    logger.info(f"TotalSegmentator initialized")
                    
                else:
                    raise ValueError(f"Unknown segmenter_type: {segmenter_type}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize organ segmenter: {e}")
                logger.warning("Falling back to fast mode (HU only)")
                self.mode = 'fast'
                self.segmenter_type = 'none'
        
        logger.info(f"HybridTumorDetector initialized (mode: {self.mode}, segmenter: {self.segmenter_type})")
    
    def detect(
        self,
        ct_volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> List[HybridLesion]:
        """
        Detect tumors using selected mode
        
        Args:
            ct_volume: CT volume in HU units
            spacing: Voxel spacing (z, y, x) in mm
        
        Returns:
            List of detected lesions
        """
        if self.mode == 'fast':
            return self._fast_detect(ct_volume, spacing)
        elif self.mode == 'accurate':
            return self._accurate_detect(ct_volume, spacing)
        else:  # 'auto'
            return self._two_stage_detect(ct_volume, spacing)
    
    def _fast_detect(
        self,
        ct_volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]]
    ) -> List[HybridLesion]:
        """
        Fast mode: Simple HU detection only
        """
        logger.info("Running FAST detection (HU only)...")
        
        # Use simple detector
        simple_lesions = self.simple_detector.detect(ct_volume, spacing)
        
        # Convert to HybridLesion format
        hybrid_lesions = []
        for lesion in simple_lesions:
            hybrid = HybridLesion(
                mask=lesion.mask,
                volume_voxels=lesion.volume_voxels,
                mean_hu=lesion.mean_hu,
                max_hu=lesion.max_hu,
                centroid=lesion.centroid,
                bbox=lesion.bbox,
                organ="not_determined",
                organ_overlap_ratio=0.0,
                confidence=0.5  # Medium confidence (no organ filtering)
            )
            hybrid_lesions.append(hybrid)
        
        logger.info(f"Fast detection: {len(hybrid_lesions)} candidates")
        
        return hybrid_lesions
    
    def _accurate_detect(
        self,
        ct_volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]]
    ) -> List[HybridLesion]:
        """
        Accurate mode: Full pipeline with organ segmentation
        """
        logger.info("Running ACCURATE detection (HU + Organ Segmentation)...")
        
        # Run 2-stage pipeline (same as auto mode)
        return self._two_stage_detect(ct_volume, spacing)
    
    def _two_stage_detect(
        self,
        ct_volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]]
    ) -> List[HybridLesion]:
        """
        Two-stage pipeline (recommended)
        
        Stage 1: Fast HU detection (find all candidates)
        Stage 2: Organ-based filtering (remove false positives)
        """
        logger.info("="*60)
        logger.info("HYBRID DETECTION - 2-STAGE PIPELINE")
        logger.info("="*60)
        
        # Stage 1: Fast HU detection
        logger.info("\n[Stage 1] Fast HU Detection...")
        simple_lesions = self.simple_detector.detect(ct_volume, spacing)
        logger.info(f"  Found {len(simple_lesions)} candidates")
        
        if len(simple_lesions) == 0:
            logger.warning("  No candidates found!")
            return []
        
        # Stage 2: Organ segmentation and filtering
        logger.info("\n[Stage 2] Organ Segmentation and Filtering...")
        
        try:
            # Segment organs based on segmenter type
            if self.segmenter_type == 'swin':
                organ_masks = self.organ_segmenter.segment_for_tumor_detection(ct_volume)
            elif self.segmenter_type == 'totalseg':
                # TotalSegmentator - need to save as temp NIfTI
                import nibabel as nib
                import tempfile
                from pathlib import Path
                
                # Create temp NIfTI
                temp_dir = Path(tempfile.mkdtemp(prefix="hybrid_"))
                temp_nifti = temp_dir / "temp.nii.gz"
                
                nifti = nib.Nifti1Image(ct_volume, affine=np.eye(4))
                nib.save(nifti, temp_nifti)
                
                # Segment
                organ_dict = self.organ_segmenter.segment(temp_nifti)
                
                # Convert to masks
                organ_masks = {}
                for organ_name, mask_data in organ_dict.items():
                    if isinstance(mask_data, np.ndarray):
                        organ_masks[organ_name] = mask_data
                
                # Cleanup
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                raise ValueError(f"Unknown segmenter type: {self.segmenter_type}")
            
            logger.info(f"  Segmented {len(organ_masks)} organ regions")
            
            # Filter candidates by organ location
            filtered_lesions = self._filter_by_organs(
                simple_lesions,
                organ_masks,
                ct_volume
            )
            
            logger.info(f"  Filtered to {len(filtered_lesions)} high-confidence candidates")
            logger.info(f"  False positive reduction: {100*(1 - len(filtered_lesions)/len(simple_lesions)):.1f}%")
            
            return filtered_lesions
            
        except Exception as e:
            logger.error(f"  Organ segmentation failed: {e}")
            logger.warning("  Returning unfiltered results...")
            
            # Return simple results as fallback
            return [
                HybridLesion(
                    mask=l.mask, volume_voxels=l.volume_voxels,
                    mean_hu=l.mean_hu, max_hu=l.max_hu,
                    centroid=l.centroid, bbox=l.bbox,
                    organ="segmentation_failed",
                    organ_overlap_ratio=0.0,
                    confidence=0.3
                )
                for l in simple_lesions
            ]
    
    def _filter_by_organs(
        self,
        candidates: List[SimpleLesion],
        organ_masks: dict,
        ct_volume: np.ndarray
    ) -> List[HybridLesion]:
        """
        Filter candidates based on organ location
        
        Args:
            candidates: List of candidate lesions
            organ_masks: Dictionary of organ masks
            ct_volume: Original CT volume (for additional checks)
        
        Returns:
            Filtered list of HybridLesions
        """
        filtered = []
        
        # Use abdominal region if available
        if 'abdominal_region' in organ_masks:
            reference_mask = organ_masks['abdominal_region']
            reference_name = 'abdominal_region'
        else:
            # Combine all organ masks
            reference_mask = np.zeros_like(ct_volume, dtype=np.uint8)
            for mask in organ_masks.values():
                reference_mask = np.maximum(reference_mask, mask)
            reference_name = 'combined_organs'
        
        logger.info(f"  Using '{reference_name}' for filtering")
        logger.info(f"  Reference region: {reference_mask.sum()} voxels")
        
        for candidate in candidates:
            # Calculate overlap with organ region
            overlap = (candidate.mask & reference_mask).sum()
            overlap_ratio = overlap / candidate.volume_voxels if candidate.volume_voxels > 0 else 0
            
            # Determine if this is a high-confidence candidate
            # Criteria:
            # 1. Located inside abdominal organs (overlap > 30%)
            # 2. HU value in typical tumor range (60-100 HU = higher confidence)
            
            # Overlap score
            if overlap_ratio > 0.5:
                location_score = 1.0
            elif overlap_ratio > 0.3:
                location_score = 0.8
            elif overlap_ratio > 0.1:
                location_score = 0.5
            else:
                location_score = 0.2
            
            # HU score (typical tumor HU: 60-100)
            if 60 <= candidate.mean_hu <= 100:
                hu_score = 1.0
            elif 50 <= candidate.mean_hu <= 120:
                hu_score = 0.8
            elif 40 <= candidate.mean_hu <= 150:
                hu_score = 0.6
            else:
                hu_score = 0.3
            
            # Combined confidence
            confidence = (location_score * 0.6 + hu_score * 0.4)
            
            # Keep if confidence is reasonably high
            if confidence > 0.4:  # Threshold
                hybrid = HybridLesion(
                    mask=candidate.mask,
                    volume_voxels=candidate.volume_voxels,
                    mean_hu=candidate.mean_hu,
                    max_hu=candidate.max_hu,
                    centroid=candidate.centroid,
                    bbox=candidate.bbox,
                    organ=reference_name,
                    organ_overlap_ratio=overlap_ratio,
                    confidence=confidence
                )
                filtered.append(hybrid)
        
        # Sort by confidence (highest first)
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered


if __name__ == "__main__":
    print("Testing HybridTumorDetector...")
    
    # Quick synthetic test
    ct = np.random.randn(100, 200, 200) * 20 + 30
    ct[40:50, 90:105, 90:105] = 80  # Synthetic tumor
    
    spacing = (5.0, 1.0, 1.0)
    
    # Test fast mode
    print("\n1. Testing FAST mode...")
    detector_fast = HybridTumorDetector(mode='fast')
    results_fast = detector_fast.detect(ct, spacing)
    print(f"   Fast mode: {len(results_fast)} candidates")
    
    # Test auto mode (if GPU available)
    print("\n2. Testing AUTO mode...")
    try:
        detector_auto = HybridTumorDetector(mode='auto', use_gpu=False)
        results_auto = detector_auto.detect(ct, spacing)
        print(f"   Auto mode: {len(results_auto)} candidates")
    except Exception as e:
        print(f"   Auto mode failed (expected without real data): {e}")
    
    print("\n✓ Test complete!")
