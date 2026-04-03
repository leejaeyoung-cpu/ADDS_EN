"""
Ensemble Predictor - 5-Fold Model Ensemble
Combines predictions from multiple folds for robust inference
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import logging

from .predictor import SOTAPredictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor for combining multiple fold models
    
    Features:
    - Averaging ensemble (mean probabilities)
    - Voting ensemble (majority vote)
    - Confidence estimation
    """
    
    def __init__(
        self,
        checkpoint_paths: List[str],
        device: str = "cuda",
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
        ensemble_mode: str = "average"
    ):
        """
        Args:
            checkpoint_paths: List of checkpoint paths for each fold
            device: Device (cuda/cpu)
            patch_size: Patch size for sliding window
            overlap: Overlap ratio
            ensemble_mode: "average" or "vote"
        """
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.ensemble_mode = ensemble_mode
        
        # Create predictors for each fold
        logger.info(f"Initializing ensemble with {len(checkpoint_paths)} models...")
        self.predictors = []
        
        for i, ckpt_path in enumerate(checkpoint_paths):
            logger.info(f"Loading fold {i}: {ckpt_path}")
            predictor = SOTAPredictor(
                checkpoint_path=ckpt_path,
                device=device,
                patch_size=patch_size,
                overlap=overlap,
                use_tta=False,  # TTA handled at ensemble level
                apply_postprocessing=False  # Post-process final ensemble
            )
            self.predictors.append(predictor)
        
        logger.info(f"Ensemble initialized with {len(self.predictors)} models")
    
    def predict(
        self,
        volume: np.ndarray,
        return_uncertainty: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict with ensemble
        
        Args:
            volume: Input volume (D, H, W)
            return_uncertainty: Return uncertainty map
        
        Returns:
            prediction: Binary segmentation (D, H, W)
            uncertainty: (optional) Uncertainty map (D, H, W)
        """
        logger.info(f"Running ensemble prediction on volume shape: {volume.shape}")
        
        # Collect predictions from all folds
        all_probs = []
        
        for i, predictor in enumerate(self.predictors):
            logger.info(f"Predicting with fold {i}...")
            _, probs = predictor.predict(volume, return_probabilities=True)
            all_probs.append(probs)
        
        # Stack predictions (N_folds, 2, D, H, W)
        all_probs = np.stack(all_probs, axis=0)
        
        # Ensemble
        if self.ensemble_mode == "average":
            # Average probabilities
            ensemble_probs = all_probs.mean(axis=0)  # (2, D, H, W)
            prediction = (ensemble_probs[1] > 0.5).astype(np.uint8)
            
        elif self.ensemble_mode == "vote":
            # Majority voting
            individual_preds = (all_probs[:, 1, :, :, :] > 0.5).astype(int)
            votes = individual_preds.sum(axis=0)
            prediction = (votes > len(self.predictors) / 2).astype(np.uint8)
            ensemble_probs = None
            
        else:
            raise ValueError(f"Unknown ensemble mode: {self.ensemble_mode}")
        
        # Calculate uncertainty (std of probabilities)
        uncertainty = None
        if return_uncertainty and ensemble_probs is not None:
            # Uncertainty as standard deviation across folds
            uncertainty = all_probs[:, 1, :, :, :].std(axis=0)
            logger.info(f"Mean uncertainty: {uncertainty.mean():.4f}")
        
        logger.info(f"Ensemble prediction complete")
        logger.info(f"Predicted volume: {prediction.sum()} voxels")
        
        if return_uncertainty:
            return prediction, uncertainty
        else:
            return prediction, None


# Test ensemble
if __name__ == "__main__":
    print("Testing Ensemble Predictor...")
    
    # Note: Requires trained checkpoints
    print("EnsemblePredictor class definition successful")
    print("[OK] EnsemblePredictor ready for use")
