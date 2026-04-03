"""
Organ Segmentation Engine using TotalSegmentator.
Provides wrapper for multi-organ CT segmentation with 104 anatomical structures.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import logging

try:
    from totalsegmentator.python_api import totalsegmentator
except ImportError:
    totalsegmentator = None
    logging.warning("TotalSegmentator not installed. Run: pip install TotalSegmentator")


class OrganSegmentationEngine:
    """
    Wrapper for TotalSegmentator to perform multi-organ segmentation.
    
    Segments 104 anatomical structures including:
    - 27 organs (liver, kidneys, spleen, pancreas, colon, etc.)
    - 59 bones (vertebrae, ribs, pelvis, etc.)
    - 10 muscles
    - 8 vessels
    
    Attributes:
        device (str): CUDA device or 'cpu'
        fast_mode (bool): Use fast mode (lower accuracy, faster)
        task (str): Segmentation task ('total', 'organ', 'vessel', etc.)
    """
    
    # Organ groups for colon cancer analysis
    ABDOMINAL_ORGANS = [
        'liver', 'spleen', 'pancreas', 'kidney_right', 'kidney_left',
        'gallbladder', 'stomach', 'duodenum', 'small_bowel', 'colon',
        'urinary_bladder'
    ]
    
    LYMPH_NODE_REGIONS = [
        'lymph_node_right', 'lymph_node_left'
    ]
    
    MAJOR_VESSELS = [
        'aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein',
        'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right'
    ]
    
    def __init__(
        self,
        device: str = 'gpu' if torch.cuda.is_available() else 'cpu',
        fast_mode: bool = False,
        task: str = 'total'
    ):
        """
        Initialize Organ Segmentation Engine.
        
        Args:
            device: Device to run inference on ('gpu', 'cpu', or 'gpu:0', 'gpu:1', etc.)
            fast_mode: Use fast mode (3mm resolution, faster but less accurate)
            task: Segmentation task type
                - 'total': All 104 structures
                - 'organ': Major organs only
                - 'abdominal': Abdominal organs (for cancer analysis)
        """
        if totalsegmentator is None:
            raise ImportError(
                "TotalSegmentator is not installed. "
                "Install it with: pip install TotalSegmentator"
            )
        
        self.device = device
        self.fast_mode = fast_mode
        self.task = task
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized OrganSegmentationEngine (device={device}, fast={fast_mode})")
    
    def segment_from_file(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment organs from NIfTI file.
        
        Args:
            input_path: Path to input NIfTI file (.nii or .nii.gz)
            output_dir: Directory to save segmentation masks (optional)
            
        Returns:
            Dictionary mapping organ names to segmentation masks (numpy arrays)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        self.logger.info(f"Segmenting organs from: {input_path}")
        
        # Create temporary output directory if not provided
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_seg"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Run TotalSegmentator
        totalsegmentator(
            input=str(input_path),
            output=str(output_dir),
            fast=self.fast_mode,
            device=self.device,
            ml=True  # Multi-label output
        )
        
        # Load segmentation results
        masks = self._load_segmentation_masks(output_dir)
        
        self.logger.info(f"Segmented {len(masks)} structures")
        
        return masks
    
    def segment_from_array(
        self,
        ct_volume: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment organs from numpy array.
        
        Args:
            ct_volume: CT volume as numpy array (D, H, W)
            spacing: Voxel spacing in mm (z, y, x)
            output_dir: Directory to save temporary NIfTI and results
            
        Returns:
            Dictionary mapping organ names to segmentation masks
        """
        # Create temporary directory
        if output_dir is None:
            output_dir = Path("temp_segmentation")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save as temporary NIfTI
        temp_nifti = output_dir / "temp_ct.nii.gz"
        affine = np.eye(4)
        affine[0, 0] = spacing[2]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[0]
        
        nii_img = nib.Nifti1Image(ct_volume.astype(np.float32), affine)
        nib.save(nii_img, str(temp_nifti))
        
        # Run segmentation
        masks = self.segment_from_file(temp_nifti, output_dir / "results")
        
        # Clean up temporary file
        temp_nifti.unlink()
        
        return masks
    
    def _load_segmentation_masks(
        self,
        seg_dir: Path
    ) -> Dict[str, np.ndarray]:
        """
        Load all segmentation masks from output directory.
        
        Args:
            seg_dir: Directory containing segmentation NIfTI files
            
        Returns:
            Dictionary mapping organ names to masks
        """
        masks = {}
        
        # Check for multi-label output
        multilabel_file = seg_dir / "segmentations.nii.gz"
        if multilabel_file.exists():
            # Load multi-label segmentation
            seg_nii = nib.load(str(multilabel_file))
            seg_data = seg_nii.get_fdata().astype(np.uint8)
            
            # Split into individual organ masks
            unique_labels = np.unique(seg_data)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background
            
            for label_id in unique_labels:
                mask = (seg_data == label_id).astype(np.uint8)
                organ_name = self._get_organ_name(label_id)
                masks[organ_name] = mask
        else:
            # Load individual segmentation files
            for seg_file in seg_dir.glob("*.nii.gz"):
                organ_name = seg_file.stem.replace('.nii', '')
                seg_nii = nib.load(str(seg_file))
                mask = seg_nii.get_fdata().astype(np.uint8)
                masks[organ_name] = mask
        
        return masks
    
    def _get_organ_name(self, label_id: int) -> str:
        """
        Get organ name from label ID (TotalSegmentator label mapping).
        
        Args:
            label_id: Integer label ID
            
        Returns:
            Organ name string
        """
        # Simplified mapping (full mapping has 104 entries)
        label_map = {
            1: 'spleen',
            2: 'kidney_right',
            3: 'kidney_left',
            4: 'gallbladder',
            5: 'liver',
            6: 'stomach',
            7: 'pancreas',
            8: 'adrenal_gland_right',
            9: 'adrenal_gland_left',
            10: 'lung_upper_lobe_left',
            # ... (104 total labels)
        }
        
        return label_map.get(label_id, f"organ_{label_id}")
    
    def get_abdominal_organs(
        self,
        all_masks: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Filter for abdominal organs relevant to colon cancer analysis.
        
        Args:
            all_masks: Dictionary of all segmentation masks
            
        Returns:
            Filtered dictionary with only abdominal organs
        """
        abdominal_masks = {}
        
        for organ in self.ABDOMINAL_ORGANS:
            if organ in all_masks:
                abdominal_masks[organ] = all_masks[organ]
        
        return abdominal_masks
    
    def compute_organ_volumes(
        self,
        masks: Dict[str, np.ndarray],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        """
        Compute volumes of segmented organs in mL (cubic centimeters).
        
        Args:
            masks: Dictionary of organ masks
            spacing: Voxel spacing in mm (z, y, x)
            
        Returns:
            Dictionary mapping organ names to volumes in mL
        """
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        voxel_volume_ml = voxel_volume_mm3 / 1000.0  # Convert mm³ to mL
        
        volumes = {}
        for organ_name, mask in masks.items():
            num_voxels = np.sum(mask > 0)
            volume_ml = num_voxels * voxel_volume_ml
            volumes[organ_name] = volume_ml
        
        return volumes
    
    def create_combined_mask(
        self,
        masks: Dict[str, np.ndarray],
        organs: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Combine multiple organ masks into a single multi-label mask.
        
        Args:
            masks: Dictionary of individual organ masks
            organs: List of organs to include (None = all)
            
        Returns:
            Combined mask with unique label per organ
        """
        if organs is None:
            organs = list(masks.keys())
        
        # Get shape from first mask
        shape = next(iter(masks.values())).shape
        combined = np.zeros(shape, dtype=np.uint8)
        
        for label_id, organ_name in enumerate(organs, start=1):
            if organ_name in masks:
                combined[masks[organ_name] > 0] = label_id
        
        return combined
