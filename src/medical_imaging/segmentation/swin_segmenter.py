"""
Swin-UNETR Organ Segmenter using Pretrained Weights
Uses pretrained 14-class model for organ segmentation
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SwinUNETRSegmenter:
    """
    Organ segmentation using pretrained Swin-UNETR
    
    Uses the original 14-class pretrained model for organ segmentation.
    Specifically optimized for colon segmentation.
    """
    
    # Organ class mapping (based on BTCV dataset or similar)
    # TODO: Verify exact class IDs from training data
    ORGAN_CLASSES = {
        'background': 0,
        'spleen': 1,
        'right_kidney': 2,
        'left_kidney': 3,
        'gallbladder': 4,
        'esophagus': 5,
        'liver': 6,
        'stomach': 7,
        'aorta': 8,
        'inferior_vena_cava': 9,
        'portal_vein_splenic_vein': 10,
        'pancreas': 11,
        'right_adrenal_gland': 12,
        'left_adrenal_gland': 13,
        # Note: colon might not be in standard 14 classes
        # The actual class ID needs verification
    }
    
    def __init__(
        self,
        pretrained_path: str = "models/pretrained/swin_unetr_pretrained.pt",
        num_classes: int = 14,
        device: str = "cuda"
    ):
        """
        Initialize organ segmenter
        
        Args:
            pretrained_path: Path to pretrained weights
            num_classes: Number of organ classes (14 by default)
            device: cuda or cpu
        """
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pretrained_path = pretrained_path
        
        logger.info(f"Initializing SwinUNETRSegmenter on {self.device}")
        
        # Create and load pretrained model
        self.model = self._load_pretrained_model()
        self.model.to(self.device)
        self.model.eval()  # Always in eval mode for segmentation
        
        logger.info("SwinUNETRSegmenter ready")
    
    def _load_pretrained_model(self) -> nn.Module:
        """Load pretrained 14-class model"""
        
        # Create Swin-UNETR with original 14 classes
        model = SwinUNETR(
            in_channels=1,
            out_channels=self.num_classes,
            feature_size=48,
            use_checkpoint=True,
            spatial_dims=3
        )
        
        logger.info("Model architecture created (14 classes)")
        
        # Load pretrained weights
        if Path(self.pretrained_path).exists():
            try:
                logger.info(f"Loading pretrained weights from {self.pretrained_path}")
                checkpoint = torch.load(self.pretrained_path, map_location='cpu', weights_only=False)
                
                # Extract state_dict
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    epoch = checkpoint.get('epoch', 'unknown')
                    logger.info(f"Loaded checkpoint from epoch {epoch}")
                else:
                    state_dict = checkpoint
                
                # Load weights
                model.load_state_dict(state_dict, strict=True)
                logger.info("✓ Pretrained weights loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
                raise RuntimeError(f"Cannot initialize segmenter without pretrained weights: {e}")
        else:
            raise FileNotFoundError(f"Pretrained weights not found: {self.pretrained_path}")
        
        return model
    
    def segment(
        self,
        ct_volume: np.ndarray,
        target_organs: list = None
    ) -> dict:
        """
        Segment organs in CT volume
        
        Args:
            ct_volume: CT volume (D, H, W) or (1, D, H, W) in HU units
            target_organs: List of organ names to segment (None = all)
        
        Returns:
            dict: {organ_name: binary_mask}
        """
        # Store original shape for unpadding
        original_shape = ct_volume.shape
        
        # Prepare input
        if ct_volume.ndim == 3:
            ct_volume = ct_volume[np.newaxis, ...]  # Add channel dim
        
        if ct_volume.ndim == 4:
            ct_volume = ct_volume[np.newaxis, ...]  # Add batch dim
        
        # Pad to multiple of 32 (Swin-UNETR requirement)
        padded_volume, pad_info = self._pad_to_multiple_of_32(ct_volume)
        
        # Convert to tensor
        x = torch.from_numpy(padded_volume).float().to(self.device)
        
        logger.info(f"Segmenting volume: {x.shape} (original: {original_shape}, padded: {padded_volume.shape})")
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(x)  # (B, num_classes, D, H, W)
            probs = torch.softmax(logits, dim=1)  # Convert to probabilities
        
        # Unpad results
        probs_unpadded = self._unpad(probs, pad_info)
        
        # Extract masks for each organ
        organ_masks = {}
        
        # If no target organs specified, segment all
        if target_organs is None:
            target_organs = list(self.ORGAN_CLASSES.keys())
        
        for organ_name in target_organs:
            if organ_name not in self.ORGAN_CLASSES:
                logger.warning(f"Unknown organ: {organ_name}")
                continue
            
            class_id = self.ORGAN_CLASSES[organ_name]
            
            # Extract probability for this class
            organ_prob = probs_unpadded[0, class_id].cpu().numpy()
            
            # Threshold to binary mask
            organ_mask = (organ_prob > 0.5).astype(np.uint8)
            
            # Only include if mask is non-empty
            if organ_mask.sum() > 0:
                organ_masks[organ_name] = organ_mask
                logger.debug(f"{organ_name}: {organ_mask.sum()} voxels")
        
        logger.info(f"Segmented {len(organ_masks)} organs")
        
        return organ_masks
    
    def _pad_to_multiple_of_32(self, volume: np.ndarray) -> tuple:
        """
        Pad volume to be divisible by 32 (Swin-UNETR requirement)
        
        Args:
            volume: Input volume (B, C, D, H, W)
        
        Returns:
            padded_volume: Padded volume
            pad_info: Padding information for unpadding
        """
        b, c, d, h, w = volume.shape
        
        # Calculate padding needed
        pad_d = (32 - d % 32) % 32
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            # No padding needed
            return volume, None
        
        logger.debug(f"Padding: D+{pad_d}, H+{pad_h}, W+{pad_w}")
        
        # Pad with air HU value (-1024)
        padded = np.pad(
            volume,
            ((0, 0), (0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=-1024
        )
        
        pad_info = {
            'pad_d': pad_d,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'original_shape': (d, h, w)
        }
        
        return padded, pad_info
    
    def _unpad(self, tensor: torch.Tensor, pad_info: dict) -> torch.Tensor:
        """
        Remove padding from tensor
        
        Args:
            tensor: Padded tensor (B, C, D, H, W)
            pad_info: Padding information
        
        Returns:
            Unpadded tensor
        """
        if pad_info is None:
            return tensor
        
        d_orig, h_orig, w_orig = pad_info['original_shape']
        
        # Crop to original size
        unpadded = tensor[:, :, :d_orig, :h_orig, :w_orig]
        
        return unpadded
    
    def segment_abdominal_organs(self, ct_volume: np.ndarray) -> dict:
        """
        Segment common abdominal organs
        
        Args:
            ct_volume: CT volume
        
        Returns:
            dict: {organ_name: binary_mask}
        """
        abdominal_organs = [
            'liver', 'spleen', 'pancreas', 'stomach',
            'left_kidney', 'right_kidney',
            'gallbladder'
        ]
        
        return self.segment(ct_volume, target_organs=abdominal_organs)
    
    def segment_for_tumor_detection(self, ct_volume: np.ndarray) -> dict:
        """
        Segment organs relevant for tumor detection
        
        This method attempts to find the colon or adjacent organs
        that can help localize colon tumors.
        
        Args:
            ct_volume: CT volume
        
        Returns:
            dict: Relevant organ masks
        """
        # Since colon might not be in the 14 classes,
        # we segment nearby organs that can help localize
        relevant_organs = [
            'liver', 'stomach', 'pancreas', 'spleen',
            'left_kidney', 'right_kidney'
        ]
        
        logger.info("Segmenting organs for tumor localization")
        masks = self.segment(ct_volume, target_organs=relevant_organs)
        
        # Create an "abdominal region" mask by combining organs
        if masks:
            combined = np.zeros_like(list(masks.values())[0])
            for mask in masks.values():
                combined = np.maximum(combined, mask)
            
            masks['abdominal_region'] = combined
            logger.info(f"Abdominal region: {combined.sum()} voxels")
        
        return masks


def test_segmenter():
    """Quick test of segmenter"""
    import nibabel as nib
    
    print("Testing SwinUNETRSegmenter...")
    
    # Load a test CT
    test_ct_path = "data/medical_decathlon/Task10_Colon/imagesTr/colon_001.nii.gz"
    
    if not Path(test_ct_path).exists():
        print(f"Test data not found: {test_ct_path}")
        return
    
    # Load CT
    nifti = nib.load(test_ct_path)
    ct_volume = nifti.get_fdata()
    
    print(f"CT volume: {ct_volume.shape}")
    
    # Create segmenter
    segmenter = SwinUNETRSegmenter(device='cuda')
    
    # Segment
    masks = segmenter.segment_for_tumor_detection(ct_volume)
    
    print(f"\nSegmented organs:")
    for organ, mask in masks.items():
        print(f"  {organ}: {mask.sum()} voxels")
    
    print("\n✓ Test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_segmenter()
