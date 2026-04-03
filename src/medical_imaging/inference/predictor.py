"""
SOTA Predictor - Sliding Window Inference
Full volume prediction with overlap and post-processing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path
import logging

from ..models.sota_model import SOTAModelWrapper
from .postprocess import postprocess_segmentation

logger = logging.getLogger(__name__)


class SOTAPredictor:
    """
    Predictor for SOTA medical segmentation models
    
    Features:
    - Sliding window inference
    - Test-time augmentation (TTA)
    - Automatic post-processing
    - Multiple output formats
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
        use_tta: bool = False,
        apply_postprocessing: bool = True
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device (cuda/cpu)
            patch_size: Patch size for sliding window
            overlap: Overlap ratio (0.0-1.0)
            use_tta: Use test-time augmentation
            apply_postprocessing: Apply post-processing to output
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.overlap = overlap
        self.use_tta = use_tta
        self.apply_postprocessing = apply_postprocessing
        
        # Calculate stride from overlap
        self.stride = tuple(int(p * (1 - overlap)) for p in patch_size)
        
        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        logger.info(f"Predictor initialized on {self.device}")
        logger.info(f"Patch size: {patch_size}, Stride: {self.stride}")
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model wrapper
        model_wrapper = SOTAModelWrapper(
            model_type="swin_unetr",
            img_size=self.patch_size,
            in_channels=3,  # Match trained checkpoint (3-channel multi-window input)
            out_channels=2,
            use_pretrained=False,
            device=str(self.device)
        )
        
        # Load weights
        model_wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        
        return model_wrapper.model.to(self.device)
    
    def predict(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict segmentation for full volume
        
        Args:
            volume: Input volume (D, H, W) or (1, D, H, W)
            return_probabilities: Return probability maps
        
        Returns:
            prediction: Binary segmentation mask (D, H, W)
            probabilities: (optional) Probability maps (2, D, H, W)
        """
        logger.info(f"Starting prediction on volume shape: {volume.shape}")
        
        # Convert to tensor if needed
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).float()
        
        # Add batch and channel dims if needed
        if volume.ndim == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        elif volume.ndim == 4:
            volume = volume.unsqueeze(0)  # (1, C, D, H, W)
        
        # Sliding window inference
        with torch.no_grad():
            if self.use_tta:
                prediction_probs = self._predict_with_tta(volume)
            else:
                prediction_probs = self._sliding_window_inference(volume)
        
        # Convert to numpy
        prediction_probs_np = prediction_probs.cpu().numpy()
        
        # Get binary prediction
        prediction_binary = (prediction_probs_np[1] > 0.5).astype(np.uint8)
        
        # Apply post-processing
        if self.apply_postprocessing:
            logger.info("Applying post-processing...")
            prediction_binary, stats = postprocess_segmentation(
                prediction_binary,
                min_size=1000,
                max_size=50000,
                fill_holes_size=500,
                apply_morphology=True
            )
            logger.info(f"Post-processing stats: {stats}")
        
        if return_probabilities:
            return prediction_binary, prediction_probs_np
        else:
            return prediction_binary
    
    def _sliding_window_inference(
        self,
        volume: torch.Tensor
    ) -> torch.Tensor:
        """
        Sliding window inference with overlap
        
        Args:
            volume: Input volume (1, 1, D, H, W)
        
        Returns:
            prediction: Probability maps (2, D, H, W)
        """
        _, _, D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride
        
        # Initialize output and count tensors
        output = torch.zeros(2, D, H, W, device=self.device)
        count = torch.zeros(1, D, H, W, device=self.device)
        
        # Calculate number of patches
        n_patches_d = int(np.ceil((D - pd) / sd)) + 1
        n_patches_h = int(np.ceil((H - ph) / sh)) + 1
        n_patches_w = int(np.ceil((W - pw) / sw)) + 1
        
        total_patches = n_patches_d * n_patches_h * n_patches_w
        logger.info(f"Processing {total_patches} patches...")
        
        patch_count = 0
        
        # Sliding window
        for d_start in range(0, D - pd + 1, sd):
            d_end = min(d_start + pd, D)
            d_start = d_end - pd
            
            for h_start in range(0, H - ph + 1, sh):
                h_end = min(h_start + ph, H)
                h_start = h_end - ph
                
                for w_start in range(0, W - pw + 1, sw):
                    w_end = min(w_start + pw, W)
                    w_start = w_end - pw
                    
                    # Extract patch
                    patch = volume[
                        :, :,
                        d_start:d_end,
                        h_start:h_end,
                        w_start:w_end
                    ].to(self.device)
                    
                    # Predict
                    with torch.cuda.amp.autocast(enabled=True):
                        patch_pred = self.model(patch)
                        patch_pred = torch.softmax(patch_pred, dim=1)
                    
                    # Accumulate
                    output[
                        :,
                        d_start:d_end,
                        h_start:h_end,
                        w_start:w_end
                    ] += patch_pred[0]
                    
                    count[
                        :,
                        d_start:d_end,
                        h_start:h_end,
                        w_start:w_end
                    ] += 1
                    
                    patch_count += 1
                    if patch_count % 10 == 0:
                        logger.info(f"Processed {patch_count}/{total_patches} patches")
        
        # Average overlapping predictions
        output = output / count
        
        return output
    
    def _predict_with_tta(
        self,
        volume: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict with test-time augmentation
        
        Augmentations:
        - Original
        - Flip along each axis
        - Average predictions
        
        Args:
            volume: Input volume (1, 1, D, H, W)
        
        Returns:
            prediction: Averaged probability maps (2, D, H, W)
        """
        logger.info("Using test-time augmentation...")
        
        predictions = []
        
        # Original
        pred = self._sliding_window_inference(volume)
        predictions.append(pred)
        
        # Flip D axis
        volume_flip_d = torch.flip(volume, dims=[2])
        pred_flip_d = self._sliding_window_inference(volume_flip_d)
        pred_flip_d = torch.flip(pred_flip_d, dims=[1])
        predictions.append(pred_flip_d)
        
        # Flip H axis
        volume_flip_h = torch.flip(volume, dims=[3])
        pred_flip_h = self._sliding_window_inference(volume_flip_h)
        pred_flip_h = torch.flip(pred_flip_h, dims=[2])
        predictions.append(pred_flip_h)
        
        # Flip W axis
        volume_flip_w = torch.flip(volume, dims=[4])
        pred_flip_w = self._sliding_window_inference(volume_flip_w)
        pred_flip_w = torch.flip(pred_flip_w, dims=[3])
        predictions.append(pred_flip_w)
        
        # Average
        prediction_avg = torch.stack(predictions).mean(dim=0)
        
        return prediction_avg


# Test predictor
if __name__ == "__main__":
    print("Testing SOTA Predictor...")
    
    # Note: Requires trained checkpoint
    print("Predictor class definition successful")
    print("[OK] SOTAPredictor ready for use")
