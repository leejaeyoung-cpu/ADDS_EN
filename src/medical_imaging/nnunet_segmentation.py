"""
Stage 2 & 3: nnU-Net Based Colon and Tumor Segmentation
Integrates nnU-Net v2 for automatic segmentation of colon and tumors
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import torch

try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    NNUNET_AVAILABLE = True
except ImportError:
    NNUNET_AVAILABLE = False
    print("WARNING: nnU-Net not available. Using fallback segmentation.")

import SimpleITK as sitk

logger = logging.getLogger(__name__)


class nnUNetSegmentationEngine:
    """
    nnU-Net v2 based segmentation engine for CT colon cancer detection
    
    Stages:
    - Stage 2: Colon segmentation
    - Stage 3: Tumor detection within colon region
    """
    
    def __init__(self, 
                 colon_model_dir: Optional[Path] = None,
                 tumor_model_dir: Optional[Path] = None,
                 device: str = 'cuda',
                 use_folds: Optional[List[int]] = None):
        """
        Initialize nnU-Net segmentation engine
        
        Args:
            colon_model_dir: Path to trained colon segmentation model
            tumor_model_dir: Path to trained tumor segmentation model
            device: 'cuda' or 'cpu'
            use_folds: Which folds to use for ensemble (default: all available)
        """
        self.colon_model_dir = colon_model_dir
        self.tumor_model_dir = tumor_model_dir
        self.device = device
        self.use_folds = use_folds
        
        # Check nnU-Net availability
        if not NNUNET_AVAILABLE:
            logger.warning("nnU-Net not available - will use fallback segmentation")
            self.colon_predictor = None
            self.tumor_predictor = None
        else:
            # Initialize predictors
            self.colon_predictor = None
            self.tumor_predictor = None
            
            if self.colon_model_dir and self.colon_model_dir.exists():
                logger.info(f"Loading colon segmentation model from {colon_model_dir}")
                self._init_colon_predictor()
            else:
                logger.warning("Colon model not found - will use fallback")
            
            if self.tumor_model_dir and self.tumor_model_dir.exists():
                logger.info(f"Loading tumor segmentation model from {tumor_model_dir}")
                self._init_tumor_predictor()
            else:
                logger.warning("Tumor model not found - will use fallback")
    
    def _init_colon_predictor(self):
        """Initialize colon segmentation predictor"""
        try:
            self.colon_predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device(self.device),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
            
            self.colon_predictor.initialize_from_trained_model_folder(
                str(self.colon_model_dir),
                use_folds=self.use_folds,
                checkpoint_name='checkpoint_final.pth'
            )
            
            logger.info("Colon predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize colon predictor: {e}")
            self.colon_predictor = None
    
    def _init_tumor_predictor(self):
        """Initialize tumor segmentation predictor"""
        try:
            self.tumor_predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device(self.device),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
            
            self.tumor_predictor.initialize_from_trained_model_folder(
                str(self.tumor_model_dir),
                use_folds=self.use_folds,
                checkpoint_name='checkpoint_final.pth'
            )
            
            logger.info("Tumor predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tumor predictor: {e}")
            self.tumor_predictor = None
    
    def segment_colon(self, volume: np.ndarray, 
                     spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Segment colon from CT volume
        
        Args:
            volume: 3D CT volume (Z, Y, X)
            spacing: Voxel spacing in mm (z, y, x)
            
        Returns:
            colon_mask: Binary mask of colon (0: background, 1: colon)
        """
        logger.info("Stage 2: Colon Segmentation")
        
        if self.colon_predictor is not None:
            # Use nnU-Net
            logger.info("Using nnU-Net for colon segmentation")
            try:
                # Convert to SimpleITK image
                image_sitk = sitk.GetImageFromArray(volume.astype(np.float32))
                image_sitk.SetSpacing(spacing[::-1])  # ITK uses (x, y, z)
                
                # Predict
                segmentation = self.colon_predictor.predict_single_npy_array(
                    volume[np.newaxis],  # Add channel dimension
                    {'spacing': spacing},
                    None,
                    None,
                    False
                )
                
                # Binarize (assume label 1 is colon)
                colon_mask = (segmentation == 1).astype(np.uint8)
                
                logger.info(f"Colon segmentation complete: {np.sum(colon_mask):,} voxels")
                return colon_mask
                
            except Exception as e:
                logger.error(f"nnU-Net colon segmentation failed: {e}")
                logger.info("Falling back to simple segmentation")
        
        # Fallback: Simple threshold-based segmentation
        return self._fallback_colon_segmentation(volume)
    
    def segment_tumor(self, volume: np.ndarray,
                     colon_mask: Optional[np.ndarray] = None,
                     spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> List[np.ndarray]:
        """
        Segment tumors within colon region
        
        Args:
            volume: 3D CT volume
            colon_mask: Optional colon mask to constrain search
            spacing: Voxel spacing in mm
            
        Returns:
            tumor_masks: List of binary tumor masks
        """
        logger.info("Stage 3: Tumor Segmentation")
        
        if self.tumor_predictor is not None:
            # Use nnU-Net
            logger.info("Using nnU-Net for tumor segmentation")
            try:
                # Optionally mask volume with colon
                if colon_mask is not None:
                    masked_volume = volume * colon_mask
                else:
                    masked_volume = volume
                
                # Predict
                segmentation = self.tumor_predictor.predict_single_npy_array(
                    masked_volume[np.newaxis],
                    {'spacing': spacing},
                    None,
                    None,
                    False
                )
                
                # Extract individual tumor instances
                tumor_masks = self._extract_tumor_instances(segmentation)
                
                logger.info(f"Detected {len(tumor_masks)} tumors")
                return tumor_masks
                
            except Exception as e:
                logger.error(f"nnU-Net tumor segmentation failed: {e}")
                logger.info("Falling back to simple detection")
        
        # Fallback: Simple detection
        return self._fallback_tumor_detection(volume, colon_mask)
    
    def _fallback_colon_segmentation(self, volume: np.ndarray) -> np.ndarray:
        """
        Fallback colon segmentation using intensity thresholding
        """
        logger.info("Using fallback colon segmentation")
        
        # Simple threshold-based approach
        # Colon typically has HU values between -100 and 100 in normalized volume
        colon_mask = np.logical_and(volume > 0.3, volume < 0.7).astype(np.uint8)
        
        # Morphological operations to clean up
        from scipy import ndimage
        
        # Remove small objects
        colon_mask = ndimage.binary_opening(colon_mask, iterations=2).astype(np.uint8)
        colon_mask = ndimage.binary_closing(colon_mask, iterations=2).astype(np.uint8)
        
        # Keep largest connected component
        labeled, num_features = ndimage.label(colon_mask)
        if num_features > 0:
            sizes = ndimage.sum(colon_mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            colon_mask = (labeled == largest_label).astype(np.uint8)
        
        logger.info(f"Fallback colon segmentation: {np.sum(colon_mask):,} voxels")
        return colon_mask
    
    def _fallback_tumor_detection(self, volume: np.ndarray,
                                  colon_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Fallback tumor detection using intensity and size criteria
        """
        logger.info("Using fallback tumor detection")
        
        # Apply colon mask if available
        if colon_mask is not None:
            search_volume = volume * colon_mask
        else:
            search_volume = volume
        
        # Threshold for potential tumors (higher intensity in CT)
        tumor_candidates = (search_volume > 0.65).astype(np.uint8)
        
        # Label connected components
        from scipy import ndimage
        labeled, num_features = ndimage.label(tumor_candidates)
        
        tumor_masks = []
        
        # Filter by size (tumors typically > 1000 voxels = 1 cm³)
        for label_id in range(1, num_features + 1):
            mask = (labeled == label_id).astype(np.uint8)
            volume_voxels = np.sum(mask)
            
            if 1000 < volume_voxels < 1000000:  # Between 1 cm³ and 1000 cm³
                tumor_masks.append(mask)
        
        logger.info(f"Fallback detection found {len(tumor_masks)} tumor candidates")
        return tumor_masks
    
    def _extract_tumor_instances(self, segmentation: np.ndarray) -> List[np.ndarray]:
        """
        Extract individual tumor instances from multi-label segmentation
        """
        from scipy import ndimage
        
        # Assume background is 0, tumors are non-zero
        tumor_region = (segmentation > 0).astype(np.uint8)
        
        # Label connected components
        labeled, num_features = ndimage.label(tumor_region)
        
        tumor_masks = []
        for label_id in range(1, num_features + 1):
            mask = (labeled == label_id).astype(np.uint8)
            if np.sum(mask) > 100:  # Minimum size threshold
                tumor_masks.append(mask)
        
        return tumor_masks
    
    def segment_all(self, volume: np.ndarray,
                   spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict:
        """
        Complete segmentation pipeline: colon + tumors
        
        Args:
            volume: 3D CT volume
            spacing: Voxel spacing
            
        Returns:
            result: Dictionary with colon_mask and tumor_masks
        """
        logger.info("Running complete segmentation pipeline")
        
        # Stage 2: Segment colon
        colon_mask = self.segment_colon(volume, spacing)
        
        # Stage 3: Segment tumors
        tumor_masks = self.segment_tumor(volume, colon_mask, spacing)
        
        result = {
            'colon_mask': colon_mask,
            'tumor_masks': tumor_masks,
            'num_tumors': len(tumor_masks),
            'colon_volume_mm3': float(np.sum(colon_mask)) * np.prod(spacing)
        }
        
        logger.info(f"Segmentation complete: {result['num_tumors']} tumors detected")
        return result


def demo_segmentation():
    """Demo of nnU-Net segmentation"""
    # Create synthetic volume
    volume = np.random.rand(100, 100, 100).astype(np.float32)
    
    # Initialize engine (will use fallback without trained models)
    engine = nnUNetSegmentationEngine(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run segmentation
    result = engine.segment_all(volume, spacing=(1.0, 1.0, 1.0))
    
    print(f"\nSegmentation Results:")
    print(f"  Colon volume: {result['colon_volume_mm3']:.1f} mm³")
    print(f"  Tumors detected: {result['num_tumors']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_segmentation()
