"""
Enhanced Patient Integration Service

Integrates CT analysis results with patient database and triggers ML updates.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from patient_management_system.database.db_enhanced import get_session
from patient_management_system.database.models_enhanced import (
    Patient, CTAnalysis, TumorMeasurement
)
from patient_management_system.services.metadata_extraction import CTMetadataExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientCTIntegration:
    """Integrates CT analysis results into patient database"""
    
    def __init__(self):
        self.db = get_session()
        self.metadata_extractor = CTMetadataExtractor()
    
    def process_ct_analysis(
        self,
        patient_id: str,
        ct_volume_path: Path,
        segmentation_path: Path,
        spacing: tuple,
        tumor_stats: Dict[str, Any]
    ) -> int:
        """
        Process CT analysis and save to database
        
        Args:
            patient_id: Patient identifier
            ct_volume_path: Path to CT volume
            segmentation_path: Path to segmentation
            spacing: Voxel spacing
            tumor_stats: Tumor statistics from analysis
        
        Returns:
            CT analysis ID
        """
        # Find or create patient
        patient = self.db.query(Patient).filter(Patient.patient_id == patient_id).first()
        
        if not patient:
            logger.info(f"Creating new patient: {patient_id}")
            patient = Patient(
                patient_id=patient_id,
                name=f"Patient-{patient_id}",
                birthdate=datetime.now(),  # TODO: Get from DICOM
                gender="M"  # TODO: Get from DICOM
            )
            self.db.add(patient)
            self.db.commit()
            self.db.refresh(patient)
        
        # Extract primary tumor info for staging
        primary_tumor = None
        if 'tumors' in tumor_stats and tumor_stats['tumors']:
            primary_tumor = tumor_stats['tumors'][0]
        
        # Create CT analysis
        analysis = CTAnalysis(
            patient_id=patient.id,
            status="completed",
            nifti_file=str(ct_volume_path),
            segmentation_file=str(segmentation_path),
            processing_time_seconds=0,  # TODO: Track actual time
            
            # Tumor metrics
            volume_ml=tumor_stats.get('total_volume_ml'),
            max_diameter_mm=tumor_stats.get('max_diameter_mm'),
            num_tumors=tumor_stats.get('num_tumors', 0),
            hu_mean=tumor_stats.get('hu_mean'),
            hu_std=tumor_stats.get('hu_std'),
            
            # Staging - NEW!
            tnm_stage=self._format_tnm(primary_tumor.get('tnm_stage')) if primary_tumor else None,
            overall_stage=primary_tumor.get('overall_stage') if primary_tumor else None,
            
            # Full data
            tumor_characteristics=tumor_stats,
            radiomics_features=primary_tumor.get('radiomics', {}) if primary_tumor else {}
        )
        
        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)
        
        logger.info(f"Created CT analysis #{analysis.id} for patient {patient_id}")
        
        # Save individual tumor measurements
        if 'tumors' in tumor_stats:
            for i, tumor in enumerate(tumor_stats['tumors']):
                measurement = TumorMeasurement(
                    ct_analysis_id=analysis.id,
                    tumor_index=i,
                    volume_ml=tumor.get('volume_ml'),
                    max_diameter_mm=tumor.get('max_diameter_mm'),
                    location_x=tumor.get('centroid', {}).get('x'),
                    location_y=tumor.get('centroid', {}).get('y'),
                    location_z=tumor.get('centroid', {}).get('z'),
                    hu_mean=tumor.get('hu_mean'),
                    hu_std=tumor.get('hu_std')
                )
                self.db.add(measurement)
            
            self.db.commit()
            logger.info(f"  Saved {len(tumor_stats['tumors'])} tumor measurements")
        
        return analysis.id
    
    def _format_tnm(self, tnm_dict: Dict) -> str:
        """
        Format TNM dictionary to string
        
        Args:
            tnm_dict: Dictionary with T, N, M values
        
        Returns:
            TNM string (e.g., "T3N1M0")
        """
        if not tnm_dict:
            return None
        return f"T{tnm_dict.get('T', 'X')}N{tnm_dict.get('N', 'X')}M{tnm_dict.get('M', 'X')}"
    
    def update_with_ai_results(
        self,
        analysis_id: int,
        adds_result: Dict[str, Any],
        openai_result: Optional[Dict[str, Any]] = None
    ):
        """Update analysis with AI recommendation results"""
        analysis = self.db.query(CTAnalysis).get(analysis_id)
        
        if analysis:
            analysis.adds_result = adds_result
            if openai_result:
                analysis.openai_result = openai_result
            
            self.db.commit()
            logger.info(f"Updated analysis #{analysis_id} with AI results")


# Convenience function
def save_ct_to_database(
    patient_id: str,
    ct_path: Path,
    seg_path: Path,
    spacing: tuple,
    stats: Dict[str, Any]
) -> int:
    """
    Convenience function to save CT analysis to database
    
    Usage in existing pipelines:
        from patient_management_system.services.patient_integration import save_ct_to_database
        
        analysis_id = save_ct_to_database(
            patient_id="PATIENT001",
            ct_path=ct_file,
            seg_path=seg_file,
            spacing=(1.0, 0.8, 0.8),
            stats=tumor_stats
        )
    """
    integration = PatientCTIntegration()
    return integration.process_ct_analysis(patient_id, ct_path, seg_path, spacing, stats)
