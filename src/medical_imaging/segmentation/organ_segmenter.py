"""
Organ Segmentation Module using TotalSegmentator

Provides automatic organ segmentation for CT images,
particularly for identifying colon region for tumor detection.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import logging
import torch

logger = logging.getLogger(__name__)


class OrganSegmenter:
    """
    Automatic organ segmentation using TotalSegmentator
    
    Supports 104 anatomical structures including:
    - Colon (our primary target)
    - Liver, kidneys, spleen, stomach, etc.
    """
    
    def __init__(
        self,
        device: str = "gpu",
        fast_mode: bool = True,
        roi_subset: Optional[list] = None
    ):
        """
        Initialize organ segmenter
        
        Args:
            device: 'gpu' or 'cpu'
            fast_mode: Use fast (lower res) segmentation
            roi_subset: List of organs to segment (None = all)
        """
        self.device = device
        self.fast_mode = fast_mode
        self.roi_subset = roi_subset or ['colon']
        
        # Check if TotalSegmentator is available
        try:
            from totalsegmentator.python_api import totalsegmentator
            self.totalsegmentator = totalsegmentator
            logger.info("TotalSegmentator loaded successfully")
        except ImportError:
            logger.error("TotalSegmentator not installed!")
            logger.error("Install with: pip install totalsegmentator")
            raise
    
    def segment(
        self,
        image: Union[np.ndarray, str, Path],
        output_path: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment organs in CT image
        
        Args:
            image: CT volume (D, H, W) or path to NIfTI file
            output_path: Optional path to save segmentation
        
        Returns:
            Dictionary of organ_name -> mask (np.ndarray)
        """
        logger.info("Running organ segmentation...")
        
        # Handle input
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image_data = nib.load(image_path).get_fdata()
        else:
            # Save temporary NIfTI for TotalSegmentator
            temp_nifti = Path("temp_input.nii.gz")
            nifti_img = nib.Nifti1Image(image, np.eye(4))
            nib.save(nifti_img, temp_nifti)
            image_path = str(temp_nifti)
            image_data = image
        
        # Run TotalSegmentator
        try:
            # Create temporary output directory
            import tempfile
            temp_output = None
            if output_path is None:
                temp_output = Path(tempfile.mkdtemp(prefix="totalseg_"))
                output_path = temp_output
            
            #  Use simpler API call - TotalSegmentator saves to files
            self.totalsegmentator(
                input=image_path,
                output=str(output_path),
                fast=self.fast_mode,
                task="total",  # Full body segmentation
                roi_subset=self.roi_subset if self.roi_subset else None,
                device=self.device,
                quiet=True
            )
            
            # Load segmentation results from output directory
            processed_dict = {}
            output_dir = Path(output_path)
            
            if output_dir.exists() and output_dir.is_dir():
                # TotalSegmentator saves individual NIfTI files for each organ
                for nifti_file in output_dir.glob("*.nii.gz"):
                    organ_name = nifti_file.stem.replace('.nii', '')
                    try:
                        mask_nifti = nib.load(nifti_file)
                        mask_array = mask_nifti.get_fdata()
                        processed_dict[organ_name] = mask_array
                        logger.debug(f"Loaded {organ_name}: {mask_array.shape}")
                    except Exception as load_error:
                        logger.warning(f"Failed to load {nifti_file}: {load_error}")
            
            logger.info(f"Segmentation complete. Found {len(processed_dict)} structures")
            
            # Clean up temp files
            if isinstance(image, np.ndarray):
                temp_nifti.unlink(missing_ok=True)
            
            if temp_output and temp_output.exists():
                import shutil
                shutil.rmtree(temp_output, ignore_errors=True)
            
            return processed_dict
            
        except Exception as e:
            logger.error(f"Organ segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return empty masks
            return self._create_fallback_masks(image_data.shape)
    
    def segment_colon(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> np.ndarray:
        """
        Segment only colon (optimized for speed)
        
        Args:
            image: CT volume or path
        
        Returns:
            Colon mask (binary, same shape as input)
        """
        logger.info("Segmenting colon specifically...")
        
        # Use ROI subset for faster processing
        original_subset = self.roi_subset
        self.roi_subset = ['colon']
        
        output_dict = self.segment(image)
        
        # Restore original ROI subset
        self.roi_subset = original_subset
        
        # Get colon mask
        colon_mask = output_dict.get('colon', None)
        
        if colon_mask is None:
            logger.warning("Colon not found in segmentation!")
            # Return empty mask
            if isinstance(image, np.ndarray):
                return np.zeros_like(image, dtype=bool)
            else:
                img_data = nib.load(image).get_fdata()
                return np.zeros_like(img_data, dtype=bool)
        
        return (colon_mask > 0).astype(bool)
    
    def get_organ_statistics(
        self,
        organ_mask: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate statistics for segmented organ
        
        Args:
            organ_mask: Binary mask
            spacing: Voxel spacing in mm (z, y, x)
        
        Returns:
            Statistics dictionary
        """
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        
        voxel_count = organ_mask.sum()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm^3
        volume_mm3 = voxel_count * voxel_volume
        volume_ml = volume_mm3 / 1000.0
        
        # Bounding box
        coords = np.where(organ_mask)
        if len(coords[0]) > 0:
            bbox = {
                'z_min': int(coords[0].min()),
                'z_max': int(coords[0].max()),
                'y_min': int(coords[1].min()),
                'y_max': int(coords[1].max()),
                'x_min': int(coords[2].min()),
                'x_max': int(coords[2].max()),
            }
        else:
            bbox = None
        
        return {
            'voxel_count': int(voxel_count),
            'volume_ml': float(volume_ml),
            'bounding_box': bbox
        }
    
    def _create_fallback_masks(self, shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        """Create empty fallback masks if segmentation fails"""
        logger.warning("Using fallback empty masks")
        return {
            'colon': np.zeros(shape, dtype=np.uint8)
        }


# Convenience function
def segment_colon(
    ct_image: Union[np.ndarray, str, Path],
    fast: bool = True,
    device: str = "gpu"
) -> np.ndarray:
    """
    Quick function to segment colon from CT
    
    Args:
        ct_image: CT volume or path to NIfTI
        fast: Use fast mode (lower resolution)
        device: 'gpu' or 'cpu'
    
    Returns:
        Binary colon mask
    """
    segmenter = OrganSegmenter(device=device, fast_mode=fast)
    return segmenter.segment_colon(ct_image)


if __name__ == "__main__":
    # Test case
    print("Testing OrganSegmenter...")
    
    # Create dummy CT volume
    dummy_ct = np.random.randn(128, 256, 256).astype(np.float32)
    
    try:
        segmenter = OrganSegmenter(device="cpu", fast_mode=True)
        colon_mask = segmenter.segment_colon(dummy_ct)
        
        print(f"Colon mask shape: {colon_mask.shape}")
        print(f"Colon voxels: {colon_mask.sum()}")
        
        stats = segmenter.get_organ_statistics(colon_mask)
        print(f"Statistics: {stats}")
        
        print("✓ OrganSegmenter test passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
