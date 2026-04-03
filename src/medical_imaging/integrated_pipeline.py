"""
Integrated Clinical Analysis Pipeline

Combines all 4 modules into a single end-to-end system:
1. Module 1: Organ Segmentation (TotalSegmentator)
2. Module 2: Tumor Detection (Swin-UNETR)
3. Module 3: TNM Staging Classification
4. Module 4: Prognosis Prediction

Input: CT DICOM/NIfTI volume
Output: Comprehensive clinical report with:
- Segmented organs (104 structures)
- Detected tumors with volumes
- TNM classification and overall stage
- Survival predictions and risk stratification
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import json
from datetime import datetime

# Import all modules
from src.medical_imaging.segmentation import OrganSegmentationEngine
from src.medical_imaging.detection import TumorDetectionEngine
from src.medical_imaging.staging import TNMStagingEngine, TNMStage
from src.medical_imaging.prognosis import PrognosisEngine, SurvivalPrediction


@dataclass
class ClinicalReport:
    """Complete clinical analysis report"""
    patient_id: str
    scan_date: str
    analysis_date: str
    
    # Module 1: Organ Segmentation
    organs_detected: List[str]
    organ_volumes: Dict[str, float]
    
    # Module 2: Tumor Detection
    tumors_detected: int
    tumor_locations: List[Dict[str, any]]
    primary_tumor_volume_cm3: float
    
    # Module 3: TNM Staging
    tnm_classification: Dict[str, str]
    cancer_stage: str
    stage_details: Dict[str, any]
    
    # Module 4: Prognosis
    survival_probabilities: Dict[str, float]
    risk_category: str
    median_survival_months: float
    
    # Metadata
    processing_time_seconds: float
    warnings: List[str]


class IntegratedClinicalPipeline:
    """
    End-to-end clinical analysis pipeline.
    
    Workflow:
    1. Load CT volume
    2. Segment organs (Module 1)
    3. Detect tumors (Module 2)
    4. Classify TNM stage (Module 3)
    5. Predict prognosis (Module 4)
    6. Generate comprehensive report
    """
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        organ_model_path: Optional[str] = None,
        tumor_model_path: Optional[str] = None,
        prognosis_model_path: Optional[str] = None
    ):
        """
        Initialize integrated pipeline.
        
        Args:
            device: Computation device
            organ_model_path: Path to organ segmentation model
            tumor_model_path: Path to tumor detection model
            prognosis_model_path: Path to prognosis prediction model
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing Integrated Clinical Pipeline...")
        
        # Initialize Module 1: Organ Segmentation
        self.organ_segmentation = OrganSegmentationEngine(
            model_path=organ_model_path,
            device=device
        )
        
        # Initialize Module 2: Tumor Detection
        self.tumor_detection = TumorDetectionEngine(
            model_path=tumor_model_path,
            device=device
        )
        
        # Initialize Module 3: TNM Staging
        self.tnm_staging = TNMStagingEngine(device=device)
        
        # Initialize Module 4: Prognosis Prediction
        self.prognosis_prediction = PrognosisEngine(
            model_path=prognosis_model_path,
            device=device
        )
        
        self.logger.info("Pipeline initialization complete")
    
    def analyze_patient(
        self,
        ct_volume: np.ndarray,
        spacing: Tuple[float, float, float],
        patient_id: str = "UNKNOWN",
        scan_date: str = "UNKNOWN",
        clinical_features: Optional[Dict[str, any]] = None
    ) -> ClinicalReport:
        """
        Perform complete clinical analysis.
        
        Args:
            ct_volume: CT image volume (D, H, W)
            spacing: Voxel spacing in mm (z, y, x)
            patient_id: Patient identifier
            scan_date: Date of CT scan
            clinical_features: Additional clinical data (age, gender, etc.)
            
        Returns:
            ClinicalReport with all analysis results
        """
        import time
        start_time = time.time()
        
        warnings = []
        
        self.logger.info(f"Starting analysis for patient {patient_id}")
        
        # ==================== Module 1: Organ Segmentation ====================
        self.logger.info("Step 1/4: Organ Segmentation")
        try:
            organ_masks = self.organ_segmentation.segment(ct_volume)
            organs_detected = list(organ_masks.keys())
            
            # Calculate organ volumes
            organ_volumes = {}
            for organ_name, mask in organ_masks.items():
                volume_mm3 = np.sum(mask) * np.prod(spacing)
                organ_volumes[organ_name] = volume_mm3 / 1000.0  # Convert to cm³
            
            self.logger.info(f"Detected {len(organs_detected)} organs")
        except Exception as e:
            self.logger.error(f"Organ segmentation failed: {e}")
            warnings.append(f"Organ segmentation error: {str(e)}")
            organ_masks = {}
            organs_detected = []
            organ_volumes = {}
        
        # ==================== Module 2: Tumor Detection ====================
        self.logger.info("Step 2/4: Tumor Detection")
        try:
            tumor_results = self.tumor_detection.detect_tumors(ct_volume, spacing)
            tumor_mask = tumor_results['combined_mask']
            tumor_regions = tumor_results['tumor_regions']
            
            tumors_detected = len(tumor_regions)
            tumor_locations = []
            
            for i, region in enumerate(tumor_regions):
                tumor_locations.append({
                    'id': i + 1,
                    'volume_cm3': region['volume_cm3'],
                    'centroid': region['centroid'],
                    'confidence': region.get('confidence', 0.0)
                })
            
            # Primary tumor (largest)
            if tumor_regions:
                primary_tumor_volume = max(r['volume_cm3'] for r in tumor_regions)
            else:
                primary_tumor_volume = 0.0
                warnings.append("No tumors detected")
            
            self.logger.info(f"Detected {tumors_detected} tumor(s)")
        except Exception as e:
            self.logger.error(f"Tumor detection failed: {e}")
            warnings.append(f"Tumor detection error: {str(e)}")
            tumor_mask = np.zeros_like(ct_volume, dtype=np.uint8)
            tumors_detected = 0
            tumor_locations = []
            primary_tumor_volume = 0.0
        
        # ==================== Module 3: TNM Staging ====================
        self.logger.info("Step 3/4: TNM Staging Classification")
        try:
            tnm_result = self.tnm_staging.classify_tnm(tumor_mask, organ_masks, ct_volume)
            
            tnm_classification = {
                'T': tnm_result.T,
                'N': tnm_result.N,
                'M': tnm_result.M
            }
            cancer_stage = tnm_result.stage
            stage_details = tnm_result.details
            
            self.logger.info(f"TNM Classification: {tnm_result.T} {tnm_result.N} {tnm_result.M} → Stage {cancer_stage}")
        except Exception as e:
            self.logger.error(f"TNM staging failed: {e}")
            warnings.append(f"TNM staging error: {str(e)}")
            tnm_classification = {'T': 'T0', 'N': 'N0', 'M': 'M0'}
            cancer_stage = "Unknown"
            stage_details = {}
        
        # ==================== Module 4: Prognosis Prediction ====================
        self.logger.info("Step 4/4: Prognosis Prediction")
        try:
            # Prepare clinical features
            if clinical_features is None:
                clinical_features = {}
            
            # Add TNM stage to clinical features
            clinical_features['tnm_stage'] = tnm_classification
            
            prognosis_result = self.prognosis_prediction.predict_survival(
                ct_volume,
                tumor_mask,
                clinical_features,
                spacing
            )
            
            survival_probabilities = prognosis_result.survival_probabilities
            risk_category = prognosis_result.risk_category
            median_survival = prognosis_result.median_survival_months
            
            self.logger.info(f"Prognosis: {risk_category} risk, Median survival: {median_survival:.1f} months")
        except Exception as e:
            self.logger.error(f"Prognosis prediction failed: {e}")
            warnings.append(f"Prognosis prediction error: {str(e)}")
            survival_probabilities = {'6mo': 0.0, '12mo': 0.0, '24mo': 0.0, '60mo': 0.0}
            risk_category = "Unknown"
            median_survival = 0.0
        
        # ==================== Generate Report ====================
        processing_time = time.time() - start_time
        
        report = ClinicalReport(
            patient_id=patient_id,
            scan_date=scan_date,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            organs_detected=organs_detected,
            organ_volumes=organ_volumes,
            tumors_detected=tumors_detected,
            tumor_locations=tumor_locations,
            primary_tumor_volume_cm3=primary_tumor_volume,
            tnm_classification=tnm_classification,
            cancer_stage=cancer_stage,
            stage_details=stage_details,
            survival_probabilities=survival_probabilities,
            risk_category=risk_category,
            median_survival_months=median_survival,
            processing_time_seconds=processing_time,
            warnings=warnings
        )
        
        self.logger.info(f"Analysis complete in {processing_time:.2f}s")
        
        return report
    
    def save_report(self, report: ClinicalReport, output_path: Path):
        """
        Save clinical report to JSON file.
        
        Args:
            report: ClinicalReport object
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Report saved to {output_path}")
    
    def generate_summary(self, report: ClinicalReport) -> str:
        """
        Generate human-readable summary of clinical report.
        
        Args:
            report: ClinicalReport object
            
        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("=" * 80)
        summary.append("CLINICAL ANALYSIS REPORT")
        summary.append("=" * 80)
        summary.append(f"Patient ID: {report.patient_id}")
        summary.append(f"Scan Date: {report.scan_date}")
        summary.append(f"Analysis Date: {report.analysis_date}")
        summary.append("")
        
        # Module 1: Organs
        summary.append("--- ORGAN SEGMENTATION ---")
        summary.append(f"Organs Detected: {len(report.organs_detected)}")
        if report.organs_detected:
            top_organs = sorted(
                report.organ_volumes.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            summary.append("Top 5 organs by volume:")
            for organ, volume in top_organs:
                summary.append(f"  - {organ}: {volume:.1f} cm³")
        summary.append("")
        
        # Module 2: Tumors
        summary.append("--- TUMOR DETECTION ---")
        summary.append(f"Tumors Detected: {report.tumors_detected}")
        if report.tumors_detected > 0:
            summary.append(f"Primary Tumor Volume: {report.primary_tumor_volume_cm3:.2f} cm³")
            for tumor in report.tumor_locations[:3]:  # Top 3
                summary.append(f"  Tumor {tumor['id']}: {tumor['volume_cm3']:.2f} cm³")
        summary.append("")
        
        # Module 3: TNM Staging
        summary.append("--- TNM CLASSIFICATION ---")
        tnm = report.tnm_classification
        summary.append(f"T Stage: {tnm['T']}")
        summary.append(f"N Stage: {tnm['N']}")
        summary.append(f"M Stage: {tnm['M']}")
        summary.append(f"Overall Cancer Stage: {report.cancer_stage}")
        summary.append("")
        
        # Module 4: Prognosis
        summary.append("--- PROGNOSIS PREDICTION ---")
        summary.append(f"Risk Category: {report.risk_category}")
        summary.append(f"Median Survival: {report.median_survival_months:.1f} months")
        summary.append("Survival Probabilities:")
        for timepoint, prob in report.survival_probabilities.items():
            summary.append(f"  {timepoint}: {prob:.1%}")
        summary.append("")
        
        # Warnings
        if report.warnings:
            summary.append("--- WARNINGS ---")
            for warning in report.warnings:
                summary.append(f"  ⚠️ {warning}")
            summary.append("")
        
        summary.append(f"Processing Time: {report.processing_time_seconds:.2f} seconds")
        summary.append("=" * 80)
        
        return "\n".join(summary)


def main():
    """Example usage"""
    # Create dummy CT volume
    ct_volume = np.random.randn(100, 256, 256) * 100 + 50  # HU values
    spacing = (2.0, 1.0, 1.0)  # mm
    
    # Patient info
    patient_id = "PATIENT001"
    scan_date = "2026-01-25"
    clinical_features = {
        'age': 65,
        'gender': 'male'
    }
    
    # Initialize pipeline
    pipeline = IntegratedClinicalPipeline()
    
    # Run analysis
    report = pipeline.analyze_patient(
        ct_volume,
        spacing,
        patient_id=patient_id,
        scan_date=scan_date,
        clinical_features=clinical_features
    )
    
    # Print summary
    print(pipeline.generate_summary(report))
    
    # Save report
    pipeline.save_report(report, Path("reports/patient001_analysis.json"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
