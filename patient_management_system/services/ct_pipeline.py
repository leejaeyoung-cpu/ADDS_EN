"""
CT Analysis Pipeline Service
Orchestrates the complete CT analysis workflow
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import sys
import numpy as np
import nibabel as nib
import json

# Add ADDS src to path
ADDS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ADDS_ROOT))
sys.path.insert(0, str(ADDS_ROOT / "src"))

# Import from existing ADDS system
from medical_imaging.ct_volume_reconstruction import CTVolumeReconstructor
from medical_imaging.detection.candidate_detector import TumorDetector
from medical_imaging.radiomics_extractor import RadiomicsExtractor

# Import database
patient_system_path = Path(__file__).parent.parent
sys.path.insert(0, str(patient_system_path))
from database.db import get_db_session
from database.models import CTAnalysis


class CTPipelineService:
    """
    CT Analysis Pipeline Orchestrator
    
    Runs complete analysis pipeline:
    1. Volume reconstruction from DICOM
    2. Tumor detection
    3. Radiomics feature extraction
    4. Tumor characterization
    """
    
    def __init__(self):
        """Initialize pipeline components"""
        self.volume_reconstructor = CTVolumeReconstructor()
        self.tumor_detector = TumorDetector()
        self.radiomics_extractor = RadiomicsExtractor()
    
    def run_full_pipeline(self, analysis_id: int, dicom_path: str):
        """
        Run complete CT analysis pipeline
        
        Args:
            analysis_id: Database analysis ID
            dicom_path: Path to DICOM file or directory
        """
        with get_db_session() as db:
            analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
            if not analysis:
                print(f"Analysis {analysis_id} not found")
                return
            
            try:
                start_time = datetime.now()
                
                # Step 1: Volume reconstruction
                self._update_status(db, analysis, "reconstructing_volume", 20)
                print(f"[Analysis {analysis_id}] Step 1: Reconstructing 3D volume...")
                
                dicom_dir = Path(dicom_path).parent
                volume, metadata = self.volume_reconstructor.reconstruct_3d_volume(str(dicom_dir))
                
                # Save volume
                volume_path = dicom_dir / "reconstructed_volume.nii.gz"
                nii_img = nib.Nifti1Image(volume, affine=np.eye(4))
                nib.save(nii_img, str(volume_path))
                analysis.volume_path = str(volume_path)
                db.commit()
                
                # Step 2: Tumor detection
                self._update_status(db, analysis, "detecting_tumors", 40)
                print(f"[Analysis {analysis_id}] Step 2: Detecting tumors...")
                
                # For 3D volume, detect on multiple slices
                tumor_candidates = []
                z_middle = volume.shape[0] // 2
                z_start = max(0, z_middle - 40)
                z_end = min(volume.shape[0], z_middle + 40)
                
                for z in range(z_start, z_end, 5):  # Sample every 5 slices
                    slice_2d = volume[z, :, :]
                    candidates = self.tumor_detector.detect_candidates(slice_2d, slice_index=z)
                    tumor_candidates.extend(candidates)
                
                print(f"[Analysis {analysis_id}] Found {len(tumor_candidates)} tumor candidates")
                
                # Get largest tumor for radiomics
                if tumor_candidates:
                    largest_tumor = max(tumor_candidates, key=lambda x: x.area_mm2)
                    tumor_slice_idx = largest_tumor.slice_index
                    
                    # Step 3: Radiomics extraction
                    self._update_status(db, analysis, "extracting_radiomics", 60)
                    print(f"[Analysis {analysis_id}] Step 3: Extracting radiomics features...")
                    
                    # Create simple mask for largest tumor
                    mask = np.zeros_like(volume, dtype=np.uint8)
                    tumor_slice = volume[tumor_slice_idx, :, :]
                    # Simple threshold-based mask
                    mask[tumor_slice_idx, :, :] = (tumor_slice > largest_tumor.mean_hu - 50) & \
                                                    (tumor_slice < largest_tumor.mean_hu + 50)
                    
                    radiomics = self.radiomics_extractor.extract_features(volume, mask)
                    analysis.radiomics_features = radiomics
                    
                    # Step 4: Tumor characterization
                    self._update_status(db, analysis, "characterizing_tumor", 80)
                    print(f"[Analysis {analysis_id}] Step 4: Characterizing tumor...")
                    
                    characteristics = self._characterize_tumor(largest_tumor, radiomics)
                    analysis.tumor_characteristics = characteristics
                    
                    # Create detection summary
                    detection_summary = {
                        "total_candidates": len(tumor_candidates),
                        "largest_tumor": {
                            "slice_index": tumor_slice_idx,
                            "area_mm2": float(largest_tumor.area_mm2),
                            "mean_hu": float(largest_tumor.mean_hu),
                            "confidence": float(largest_tumor.confidence_score)
                        },
                        "z_range_scanned": [z_start, z_end]
                    }
                    analysis.detection_summary = detection_summary
                else:
                    print(f"[Analysis {analysis_id}] No tumors detected")
                    analysis.radiomics_features = {}
                    analysis.tumor_characteristics = {"status": "no_tumor_detected"}
                    analysis.detection_summary = {"total_candidates": 0}
                
                # Complete
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                analysis.status = "completed"
                analysis.progress = 100
                analysis.current_step = "Analysis complete"
                analysis.completed_at = end_time
                analysis.processing_time_seconds = processing_time
                
                db.commit()
                print(f"[Analysis {analysis_id}] Pipeline completed in {processing_time:.1f}s")
                
            except Exception as e:
                print(f"[Analysis {analysis_id}] ERROR: {str(e)}")
                analysis.status = "failed"
                analysis.error_message = str(e)
                analysis_current_step = f"Failed at: {analysis.current_step}"
                db.commit()
    
    def _update_status(self, db, analysis, step: str, progress: int):
        """Update analysis status in database"""
        analysis.current_step = step
        analysis.progress = progress
        analysis.status = "processing"
        db.commit()
    
    def _characterize_tumor(self, tumor_candidate, radiomics: Dict) -> Dict:
        """
        Characterize tumor based on detection and radiomics
        
        Returns:
            Tumor characteristics dictionary
        """
        return {
            "morphology": {
                "area_mm2": float(tumor_candidate.area_mm2),
                "circularity": float(tumor_candidate.circularity),
                "eccentricity": float(tumor_candidate.eccentricity),
                "solidity": float(tumor_candidate.solidity)
            },
            "intensity": {
                "mean_hu": float(tumor_candidate.mean_hu),
                "max_hu": float(tumor_candidate.max_hu),
                "min_hu": float(tumor_candidate.min_hu)
            },
            "radiomics_summary": {
                "shape_sphericity": radiomics.get("shape_Sphericity", 0),
                "texture_entropy": radiomics.get("firstorder_Entropy", 0),
                "texture_contrast": radiomics.get("glcm_Contrast", 0)
            },
            "confidence_score": float(tumor_candidate.confidence_score),
            "location": {
                "slice_index": tumor_candidate.slice_index,
                "centroid_x": float(tumor_candidate.centroid[0]),
                "centroid_y": float(tumor_candidate.centroid[1])
            }
        }
