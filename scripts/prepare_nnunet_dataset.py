"""
nnU-Net Dataset Preparation for CRC Detection
Converts CT data and segmentation to nnU-Net format
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import json
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class nnUNetDatasetPreparator:
    """
    Prepare CT CRC dataset for nnU-Net training
    """
    
    def __init__(self, 
                 dataset_id: int = 100,
                 dataset_name: str = "CRC_CT",
                 nnunet_raw_dir: Path = None):
        """
        Initialize dataset preparator
        
        Args:
            dataset_id: nnU-Net dataset ID (e.g., 100)
            dataset_name: Dataset name
            nnunet_raw_dir: Path to nnUNet_raw directory
        """
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        
        # Setup directories
        if nnunet_raw_dir is None:
            nnunet_raw_dir = Path("F:/ADDS/nnUNet_data/nnUNet_raw")
        
        self.nnunet_raw_dir = Path(nnunet_raw_dir)
        self.dataset_dir = self.nnunet_raw_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
        
        logger.info(f"Dataset directory: {self.dataset_dir}")
    
    def setup_directories(self):
        """Create nnU-Net directory structure"""
        logger.info("Setting up nnU-Net directory structure")
        
        dirs_to_create = [
            self.dataset_dir / "imagesTr",
            self.dataset_dir / "labelsTr",
            self.dataset_dir / "imagesTs"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {dir_path}")
    
    def convert_case(self, 
                    volume_path: Path,
                    segmentation_path: Path,
                    case_id: str = "001",
                    is_test: bool = False):
        """
        Convert single case to nnU-Net format
        
        Args:
            volume_path: Path to CT volume (NIfTI)
            segmentation_path: Path to segmentation (NIfTI)
            case_id: Case identifier (e.g., "001")
            is_test: If True, save to test set
        """
        logger.info(f"Converting case {case_id}")
        
        # Load volume
        if not volume_path.exists():
            logger.error(f"Volume not found: {volume_path}")
            return False
        
        # Output filenames (nnU-Net format)
        if is_test:
            image_output = self.dataset_dir / "imagesTs" / f"{self.dataset_name}_{case_id}_0000.nii.gz"
        else:
            image_output = self.dataset_dir / "imagesTr" / f"{self.dataset_name}_{case_id}_0000.nii.gz"
            label_output = self.dataset_dir / "labelsTr" / f"{self.dataset_name}_{case_id}.nii.gz"
        
        # Copy/convert image
        logger.info(f"Copying volume to {image_output}")
        shutil.copy(volume_path, image_output)
        
        # Copy/convert label (for training cases)
        if not is_test and segmentation_path and segmentation_path.exists():
            logger.info(f"Copying segmentation to {label_output}")
            shutil.copy(segmentation_path, label_output)
        
        logger.info(f"Case {case_id} converted successfully")
        return True
    
    def create_dataset_json(self, 
                           num_training: int = 1,
                           modality: str = "CT",
                           labels: dict = None):
        """
        Create dataset.json for nnU-Net
        
        Args:
            num_training: Number of training cases
            modality: Imaging modality
            labels: Label dictionary (e.g., {0: "background", 1: "tumor"})
        """
        if labels is None:
            # Default: multi-organ segmentation from existing data
            labels = {
                "background": 0,
                "tumor": 1
            }
        
        dataset_json = {
            "channel_names": {
                "0": modality
            },
            "labels": labels,
            "numTraining": num_training,
            "file_ending": ".nii.gz",
            "name": self.dataset_name,
            "description": "CT-based colorectal cancer detection",
            "reference": "ADDS Project - Inha University Hospital",
            "licence": "Internal Use Only",
            "release": "1.0"
        }
        
        json_path = self.dataset_dir / "dataset.json"
        with open(json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        logger.info(f"Created dataset.json at {json_path}")
        return json_path
    
    def prepare_from_existing_data(self, 
                                   ct_volume_path: Path,
                                   segmentation_path: Path):
        """
        Prepare dataset from existing CT and segmentation
        
        Args:
            ct_volume_path: Path to reconstructed CT volume
            segmentation_path: Path to segmentation mask
        """
        logger.info("="*80)
        logger.info("PREPARING nnU-Net DATASET")
        logger.info("="*80)
        
        # Setup directories
        self.setup_directories()
        
        # Convert case
        success = self.convert_case(
            volume_path=ct_volume_path,
            segmentation_path=segmentation_path,
            case_id="001",
            is_test=False
        )
        
        if not success:
            logger.error("Failed to convert case")
            return False
        
        # Create dataset.json
        # Analyze segmentation to determine labels
        if segmentation_path.exists():
            seg_nifti = nib.load(segmentation_path)
            seg_data = seg_nifti.get_fdata()
            unique_labels = np.unique(seg_data).astype(int)
            
            logger.info(f"Unique labels in segmentation: {unique_labels}")
            
            # Create label dictionary
            labels = {"background": 0}
            for label in unique_labels:
                if label > 0:
                    labels[f"organ_{label}"] = int(label)
            
            logger.info(f"Label dictionary: {labels}")
        else:
            labels = {"background": 0, "tumor": 1}
        
        self.create_dataset_json(num_training=1, labels=labels)
        
        logger.info("="*80)
        logger.info("DATASET PREPARATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Dataset location: {self.dataset_dir}")
        logger.info(f"Training images: {len(list((self.dataset_dir / 'imagesTr').glob('*.nii.gz')))}")
        logger.info(f"Training labels: {len(list((self.dataset_dir / 'labelsTr').glob('*.nii.gz')))}")
        
        return True


def prepare_inha_dataset():
    """Prepare Inha CT dataset for nnU-Net"""
    
    # Paths
    ct_volume = Path("F:/ADDS/outputs/ct_pipeline_test/reconstructed_volume.nii.gz")
    segmentation = Path("F:/ADDS/CTdata/segmentation.nii")
    
    if not ct_volume.exists():
        logger.error(f"CT volume not found: {ct_volume}")
        logger.info("Please run Stage 1 reconstruction first")
        return False
    
    if not segmentation.exists():
        logger.error(f"Segmentation not found: {segmentation}")
        return False
    
    # Initialize preparator
    preparator = nnUNetDatasetPreparator(
        dataset_id=100,
        dataset_name="CRC_CT",
        nnunet_raw_dir=Path("F:/ADDS/nnUNet_data/nnUNet_raw")
    )
    
    # Prepare dataset
    success = preparator.prepare_from_existing_data(
        ct_volume_path=ct_volume,
        segmentation_path=segmentation
    )
    
    if success:
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Set environment variables:")
        print(f"   $env:nnUNet_raw = 'F:\\ADDS\\nnUNet_data\\nnUNet_raw'")
        print(f"   $env:nnUNet_preprocessed = 'F:\\ADDS\\nnUNet_data\\nnUNet_preprocessed'")
        print(f"   $env:nnUNet_results = 'F:\\ADDS\\nnUNet_data\\nnUNet_results'")
        print()
        print("2. Plan and preprocess:")
        print("   nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity")
        print()
        print("3. Train (example - 2D network, fold 0):")
        print("   nnUNetv2_train 100 2d 0")
        print()
        print("NOTE: With only 1 case, this is a demo/test setup.")
        print("      For production, you need 20+ annotated cases.")
        print("="*80)
    
    return success


if __name__ == "__main__":
    prepare_inha_dataset()
