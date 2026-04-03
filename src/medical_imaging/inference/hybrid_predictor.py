"""
Hybrid Predictor - DL + Rule-based Detection
Extensible architecture for combining deep learning and HU-based detection
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union, List, Dict
from pathlib import Path
import logging

from ..models.sota_model import SOTAModelWrapper
from ..detection.candidate_detector import (
    TumorDetector,
    TumorCandidate,
    merge_candidates
)
from .postprocess import postprocess_segmentation

logger = logging.getLogger(__name__)


class HybridPredictor:
    """
    Hybrid predictor combining deep learning and rule-based detection
    
    Features:
    - Multiple detection modes:
      - DL-only: Pure deep learning segmentation
      - Rule-only: Pure HU-based candidate detection
      - Hybrid: DL segmentation + candidate refinement
      - Ensemble: Vote-based fusion
      
    - Extensible architecture:
      - Easy to add new DL models
      - Easy to add new rule-based detectors
      - Pluggable fusion strategies
    """
    
    def __init__(
        self,
        dl_checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
        detection_mode: str = "hybrid",  # "dl_only", "rule_only", "hybrid", "ensemble"
        enable_fp_filtering: bool = True,
        enable_colon_scoring: bool = True
    ):
        """
        Args:
            dl_checkpoint_path: Path to DL model checkpoint (None for rule-only mode)
            device: Device (cuda/cpu)
            patch_size: Patch size for DL sliding window
            overlap: Overlap ratio for DL inference
            detection_mode: Detection mode
            enable_fp_filtering: Enable FP filtering in rule-based detector
            enable_colon_scoring: Enable colon-specific confidence
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.overlap = overlap
        self.detection_mode = detection_mode
        
        # Initialize DL model (if needed)
        self.dl_model = None
        if detection_mode in ["dl_only", "hybrid", "ensemble"] and dl_checkpoint_path:
            logger.info(f"Loading DL model from {dl_checkpoint_path}")
            self.dl_model = self._load_dl_model(dl_checkpoint_path)
            self.dl_model.eval()
            self.stride = tuple(int(p * (1 - overlap)) for p in patch_size)
        
        # Initialize rule-based detector (txt pipeline - proven 0.99 confidence)
        self.rule_detector = None
        if detection_mode in ["rule_only", "hybrid", "ensemble"]:
            logger.info("Initializing txt pipeline TumorDetector...")
            self.rule_detector = TumorDetector(
                min_area_mm2=10.0,  # CRITICAL: txt pipeline value (vs old: 1000!)
                max_area_mm2=10000.0,
                hu_range=(-50, 200 )  # Soft tissue + tumor range
            )
            logger.info("txt pipeline initialized: min_area=10.0mm2, HU=(-50, 200)")
        
        logger.info(f"HybridPredictor initialized in '{detection_mode}' mode")
    
    def _load_dl_model(self, checkpoint_path: str) -> nn.Module:
        """Load deep learning model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model wrapper
        model_wrapper = SOTAModelWrapper(
            model_type="swin_unetr",
            img_size=self.patch_size,
            in_channels=3,
            out_channels=2,
            use_pretrained=False,
            device=str(self.device)
        )
        
        # Load weights
        model_wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        
        return model_wrapper.model.to(self.device)
    
    def add_dl_model(self, model: nn.Module, name: str = "custom"):
        """
        Add a custom deep learning model
        
        Args:
            model: PyTorch model (should output logits or probabilities)
            name: Model name for reference
        """
        self.dl_model = model.to(self.device)
        self.dl_model.eval()
        logger.info(f"Added custom DL model: {name}")
    
    def predict(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        spacing: Optional[Tuple[float, float, float]] = None,
        return_candidates: bool = True,
        return_probabilities: bool = False
    ) -> Dict:
        """
        Predict with selected mode
        
        Args:
            volume: Input volume (D, H, W) or preprocessed
            spacing: Pixel spacing (mm) for rule-based detection
            return_candidates: Return TumorCandidate objects
            return_probabilities: Return DL probability maps
        
        Returns:
            results: Dictionary with:
                - 'mask': Binary segmentation (D, H, W)
                - 'candidates': List of TumorCandidate (if return_candidates=True)
                - 'probabilities': Probability maps (if return_probabilities=True)
                - 'metadata': Detection metadata
        """
        logger.info(f"Running prediction in '{self.detection_mode}' mode")
        
        if self.detection_mode == "dl_only":
            return self._predict_dl_only(volume, return_probabilities)
        
        elif self.detection_mode == "rule_only":
            return self._predict_rule_only(volume, spacing, return_candidates)
        
        elif self.detection_mode == "hybrid":
            return self._predict_hybrid(volume, spacing, return_candidates, return_probabilities)
        
        elif self.detection_mode == "ensemble":
            return self._predict_ensemble(volume, spacing, return_candidates, return_probabilities)
        
        else:
            raise ValueError(f"Unknown detection mode: {self.detection_mode}")
    
    def _predict_dl_only(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        return_probabilities: bool
    ) -> Dict:
        """Pure deep learning prediction"""
        
        if self.dl_model is None:
            raise ValueError("DL model not loaded. Cannot use dl_only mode.")
        
        logger.info("Running DL-only prediction...")
        
        # Convert to tensor
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).float()
        
        # Add dims
        if volume.ndim == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif volume.ndim == 4:
            volume = volume.unsqueeze(0)
        
        # Sliding window inference
        with torch.no_grad():
            probs = self._sliding_window_inference(volume)
        
        # Get binary mask
        probs_np = probs.cpu().numpy()
        mask = (probs_np[1] > 0.5).astype(np.uint8)
        
        # Post-processing
        mask, stats = postprocess_segmentation(
            mask,
            min_size=1000,
            max_size=50000,
            fill_holes_size=500,
            apply_morphology=True
        )
        
        results = {
            'mask': mask,
            'candidates': None,
            'metadata': {
                'mode': 'dl_only',
                'postprocess_stats': stats
            }
        }
        
        if return_probabilities:
            results['probabilities'] = probs_np
        
        return results
    
    def _predict_rule_only(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        return_candidates: bool
    ) -> Dict:
        """Pure rule-based prediction"""
        
        if self.rule_detector is None:
            raise ValueError("Rule detector not initialized.")
        
        logger.info("Running rule-only prediction...")
        
        # Detect candidates
        candidates = self.rule_detector.detect_candidates_3d(
            volume=volume,
            spacing=spacing
        )
        
        # Create mask from candidates
        mask = self._candidates_to_mask(candidates, volume.shape)
        
        results = {
            'mask': mask,
            'candidates': candidates if return_candidates else None,
            'metadata': {
                'mode': 'rule_only',
                'num_candidates': len(candidates),
                'high_confidence': sum(1 for c in candidates if c.confidence_score > 0.5)
            }
        }
        
        return results
    
    def _predict_hybrid(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        spacing: Tuple[float, float, float],
        return_candidates: bool,
        return_probabilities: bool
    ) -> Dict:
        """
        Hybrid prediction: DL segmentation + candidate refinement
        
        Workflow:
        1. DL model generates initial segmentation
        2. Extract candidate regions from DL mask
        3. Apply rule-based confidence scoring
        4. Filter candidates by confidence
        5. Refine final mask
        """
        
        if self.dl_model is None or self.rule_detector is None:
            raise ValueError("Both DL model and rule detector required for hybrid mode.")
        
        logger.info("Running hybrid prediction...")
        
        # Step 1: DL prediction
        dl_volume = volume.copy() if isinstance(volume, np.ndarray) else volume.clone()
        dl_results = self._predict_dl_only(dl_volume, return_probabilities=True)
        dl_mask = dl_results['mask']
        dl_probs = dl_results.get('probabilities', None)
        
        # Step 2: Extract candidates from DL mask
        candidates_from_dl = self._extract_candidates_from_mask(
            dl_mask,
            volume if isinstance(volume, np.ndarray) else volume.cpu().numpy(),
            spacing
        )
        
        logger.info(f"Extracted {len(candidates_from_dl)} candidates from DL mask")
        
        # Step 3: Re-score with rule-based confidence
        for candidate in candidates_from_dl:
            # Get slice
            z = candidate.slice_index
            hu_slice = volume[:, :, z] if isinstance(volume, np.ndarray) else volume.cpu().numpy()[:, :, z]
            
            # Recalculate confidence
            candidate.confidence_score = self.rule_detector.calculate_colon_confidence(
                candidate, hu_slice
            )
        
        # Step 4: Filter by confidence
        confidence_threshold = 0.3
        filtered_candidates = [
            c for c in candidates_from_dl if c.confidence_score >= confidence_threshold
        ]
        
        logger.info(f"Retained {len(filtered_candidates)} candidates after filtering (threshold={confidence_threshold})")
        
        # Step 5: Create refined mask
        refined_mask = self._candidates_to_mask(filtered_candidates, volume.shape if isinstance(volume, np.ndarray) else volume.shape[-3:])
        
        results = {
            'mask': refined_mask,
            'candidates': filtered_candidates if return_candidates else None,
            'metadata': {
                'mode': 'hybrid',
                'dl_candidates': len(candidates_from_dl),
                'filtered_candidates': len(filtered_candidates),
                'confidence_threshold': confidence_threshold
            }
        }
        
        if return_probabilities:
            results['probabilities'] = dl_probs
        
        return results
    
    def _predict_ensemble(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        spacing: Tuple[float, float, float],
        return_candidates: bool,
        return_probabilities: bool
    ) -> Dict:
        """
        Ensemble prediction: Vote-based fusion
        
        Workflow:
        1. Get DL prediction
        2. Get rule-based prediction
        3. Combine with voting/averaging
        """
        
        logger.info("Running ensemble prediction...")
        
        # DL prediction
        dl_results = self._predict_dl_only(volume, return_probabilities=True)
        dl_mask = dl_results['mask']
        
        # Rule prediction
        rule_results = self._predict_rule_only(
            volume if isinstance(volume, np.ndarray) else volume.cpu().numpy(),
            spacing,
            return_candidates=True
        )
        rule_mask = rule_results['mask']
        
        # Ensemble: Union of DL and high-confidence rule predictions
        rule_candidates = rule_results['candidates']
        high_conf_candidates = [c for c in rule_candidates if c.confidence_score > 0.7]
        high_conf_mask = self._candidates_to_mask(high_conf_candidates, volume.shape if isinstance(volume, np.ndarray) else volume.shape[-3:])
        
        # Combined mask (OR operation)
        ensemble_mask = np.logical_or(dl_mask, high_conf_mask).astype(np.uint8)
        
        results = {
            'mask': ensemble_mask,
            'candidates': rule_candidates if return_candidates else None,
            'metadata': {
                'mode': 'ensemble',
                'dl_voxels': dl_mask.sum(),
                'rule_voxels': high_conf_mask.sum(),
                'ensemble_voxels': ensemble_mask.sum()
            }
        }
        
        if return_probabilities:
            results['probabilities'] = dl_results.get('probabilities')
        
        return results
    
    def _sliding_window_inference(self, volume: torch.Tensor) -> torch.Tensor:
        """Sliding window inference with overlap"""
        _, _, D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride
        
        output = torch.zeros(2, D, H, W, device=self.device)
        count = torch.zeros(1, D, H, W, device=self.device)
        
        for d_start in range(0, max(D - pd + 1, 1), sd):
            d_end = min(d_start + pd, D)
            d_start = max(d_end - pd, 0)
            
            for h_start in range(0, max(H - ph + 1, 1), sh):
                h_end = min(h_start + ph, H)
                h_start = max(h_end - ph, 0)
                
                for w_start in range(0, max(W - pw + 1, 1), sw):
                    w_end = min(w_start + pw, W)
                    w_start = max(w_end - pw, 0)
                    
                    patch = volume[:, :, d_start:d_end, h_start:h_end, w_start:w_end].to(self.device)
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        patch_pred = self.dl_model(patch)
                        patch_pred = torch.softmax(patch_pred, dim=1)
                    
                    output[:, d_start:d_end, h_start:h_end, w_start:w_end] += patch_pred[0]
                    count[:, d_start:d_end, h_start:h_end, w_start:w_end] += 1
        
        output = output / (count + 1e-8)
        return output
    
    def _extract_candidates_from_mask(
        self,
        mask: np.ndarray,
        hu_volume: np.ndarray,
        spacing: Tuple[float, float, float]
    ) -> List[TumorCandidate]:
        """Extract candidates from binary mask"""
        from skimage import measure
        
        candidates = []
        
        # Process slice by slice
        for z in range(mask.shape[2]):
            mask_slice = mask[:, :, z]
            hu_slice = hu_volume[:, :, z]
            
            if mask_slice.sum() == 0:
                continue
            
            # Get regions
            props = measure.regionprops(measure.label(mask_slice), intensity_image=hu_slice)
            
            for region in props:
                # Create candidate
                centroid = region.centroid
                area_pixels = region.area
                area_mm2 = area_pixels * spacing[0] * spacing[1]
                
                bbox = region.bbox  # min_row, min_col, max_row, max_col
                mean_hu = region.mean_intensity
                max_hu = region.max_intensity
                min_hu = region.min_intensity
                
                # Calculate features
                perimeter = region.perimeter
                circularity = (4 * np.pi * area_pixels) / (perimeter ** 2 + 1e-8)
                
                candidate = TumorCandidate(
                    centroid=centroid,
                    area_pixels=area_pixels,
                    area_mm2=area_mm2,
                    bounding_box=bbox,
                    mean_hu=mean_hu,
                    max_hu=max_hu,
                    min_hu=min_hu,
                    eccentricity=region.eccentricity,
                    solidity=region.solidity,
                    perimeter=perimeter,
                    circularity=circularity,
                    confidence_score=0.5,  # Will be recalculated
                    slice_index=z
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def _candidates_to_mask(
        self,
        candidates: List[TumorCandidate],
        volume_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Convert candidates to binary mask"""
        mask = np.zeros(volume_shape, dtype=np.uint8)
        
        for c in candidates:
            z = c.slice_index
            bbox = c.bounding_box
            min_row, min_col, max_row, max_col = bbox
            
            # volume_shape is (H, W, D) from NIfTI
            if z < volume_shape[2]:
                mask[min_row:max_row, min_col:max_col, z] = 1
        
        return mask
    
    def switch_mode(self, new_mode: str):
        """Switch detection mode"""
        valid_modes = ["dl_only", "rule_only", "hybrid", "ensemble"]
        if new_mode not in valid_modes:
            raise ValueError(f"Invalid mode. Choose from: {valid_modes}")
        
        self.detection_mode = new_mode
        logger.info(f"Switched to '{new_mode}' mode")


# Convenience functions
def create_hybrid_predictor(
    dl_checkpoint: Optional[str] = None,
    mode: str = "hybrid",
    device: str = "cuda"
) -> HybridPredictor:
    """Create hybrid predictor with default settings"""
    return HybridPredictor(
        dl_checkpoint_path=dl_checkpoint,
        device=device,
        detection_mode=mode,
        enable_fp_filtering=True,
        enable_colon_scoring=True
    )
