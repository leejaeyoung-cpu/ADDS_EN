"""
CDSS Orchestrator - Master Coordinator

Coordinates all CDSS components for end-to-end patient data processing:
1. Patient data ingestion
2. CT/cell image analysis
3. Metadata extraction  
4. Treatment recommendations
5. Outcome collection
6. Continuous learning
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from patient_management_system.database.db_enhanced import get_session
from patient_management_system.database.models_enhanced import (
    Patient, CTAnalysis, CellImage, Treatment, TreatmentOutcome
)
from patient_management_system.services.metadata_extraction import (
    CTMetadataExtractor, CellImageFeatureExtractor, MetadataAggregator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CDSSOrchestrator:
    """Orchestrates end-to-end CDSS workflow"""
    
    def __init__(self):
        self.db = get_session()
        self.ct_extractor = CTMetadataExtractor()
        self.cell_extractor = CellImageFeatureExtractor()
        self.metadata_aggregator = MetadataAggregator()
    
    def process_new_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a new patient from registration through analysis
        
        Args:
            patient_data: Dictionary with patient info
            
        Returns:
            Processing status and results
        """
        logger.info(f"Processing new patient: {patient_data.get('patient_id')}")
        
        try:
            # Step 1: Create patient record
            patient = self._create_patient_record(patient_data)
            
            # Step 2: If CT data provided, analyze it
            ct_analysis = None
            if 'ct_directory' in patient_data:
                ct_analysis = self._process_ct_scan(
                    patient.id,
                    patient_data['ct_directory']
                )
            
            # Step 3: If cell images provided, extract features
            cell_features = []
            if 'cell_images' in patient_data:
                cell_features = self._process_cell_images(
                    patient.id,
                    patient_data['cell_images']
                )
            
            # Step 4: Generate initial assessment
            assessment = self._generate_assessment(patient, ct_analysis, cell_features)
            
            return {
                'success': True,
                'patient_id': patient.patient_id,
                'assessment': assessment
            }
            
        except Exception as e:
            logger.error(f"Error processing patient: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_patient_record(self, patient_data: Dict[str, Any]) -> Patient:
        """Create patient database record"""
        patient = Patient(
            patient_id=patient_data['patient_id'],
            name=patient_data.get('name', ''),
            birthdate=patient_data.get('birthdate', datetime.now()),
            gender=patient_data.get('gender', 'M')
        )
        
        self.db.add(patient)
        self.db.commit()
        
        logger.info(f"Created patient record: {patient.patient_id}")
        return patient
    
    def _process_ct_scan(self, patient_id: int, ct_directory: str) -> CTAnalysis:
        """
        Process CT scan and extract metadata
        
        Args:
            patient_id: Patient database ID
            ct_directory: Path to DICOM directory
            
        Returns:
            CTAnalysis record
        """
        logger.info(f"Processing CT scan from: {ct_directory}")
        
        # Create CT analysis record
        ct_analysis = CTAnalysis(
            patient_id=patient_id,
            dicom_directory=ct_directory,
            status='processing',
            started_at=datetime.utcnow()
        )
        
        self.db.add(ct_analysis)
        self.db.commit()
        
        try:
            # TODO: Call actual CT tumor detection pipeline
            # For now, placeholder
            logger.info("  CT analysis pipeline integration pending")
            
            # Update status
            ct_analysis.status = 'completed'
            ct_analysis.completed_at = datetime.utcnow()
            self.db.commit()
            
        except Exception as e:
            logger.error(f"CT analysis failed: {e}")
            ct_analysis.status = 'failed'
            ct_analysis.error_message = str(e)
            self.db.commit()
        
        return ct_analysis
    
    def _process_cell_images(self, patient_id: int, image_paths: list) -> list:
        """Process cell images and extract features"""
        logger.info(f"Processing {len(image_paths)} cell images")
        
        features_list = []
        
        for image_path in image_paths:
            try:
                # Extract features
                features = self.cell_extractor.extract_morphology_features(Path(image_path))
                
                # Create database record
                cell_image = CellImage(
                    patient_id=patient_id,
                    image_path=str(image_path),
                    ai_features=features.get('cnn_features', []),
                    cell_density=features.get('cell_density', 0.0),
                    quality_score=0.85 if features.get('success') else 0.0
                )
                
                self.db.add(cell_image)
                features_list.append(features)
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
        
        self.db.commit()
        logger.info(f"  Processed {len(features_list)} images successfully")
        
        return features_list
    
    def _generate_assessment(
        self,
        patient: Patient,
        ct_analysis: Optional[CTAnalysis],
        cell_features: list
    ) -> Dict[str, Any]:
        """Generate comprehensive patient assessment"""
        
        assessment = {
            'patient_id': patient.patient_id,
            'assessment_date': datetime.now().isoformat(),
            'data_completeness': {}
        }
        
        # Data completeness
        assessment['data_completeness'] = {
            'ct_scan': ct_analysis is not None and ct_analysis.status == 'completed',
            'cell_images': len(cell_features) > 0,
            'physician_notes': False  # TODO: Check
        }
        
        # Analysis summary
        if ct_analysis:
            assessment['ct_summary'] = {
                'status': ct_analysis.status,
                'tumor_detected': ct_analysis.num_tumors > 0 if ct_analysis.num_tumors else False
            }
        
        if cell_features:
            assessment['cell_analysis'] = {
                'images_processed': len(cell_features),
                'features_extracted': all(f.get('success', False) for f in cell_features)
            }
        
        return assessment
    
    def update_patient_data(self, patient_id: str, update_type: str, data: Dict[str, Any]) -> bool:
        """
        Update existing patient data and trigger re-analysis if needed
        
        Args:
            patient_id: Patient identifier
            update_type: 'ct_scan', 'cell_images', 'physician_notes', 'treatment_outcome'
            data: Update data
            
        Returns:
            Success status
        """
        logger.info(f"Updating patient {patient_id}: {update_type}")
        
        patient = self.db.query(Patient).filter(Patient.patient_id == patient_id).first()
        
        if not patient:
            logger.error(f"Patient not found: {patient_id}")
            return False
        
        try:
            if update_type == 'ct_scan':
                self._process_ct_scan(patient.id, data['ct_directory'])
                
            elif update_type == 'cell_images':
                self._process_cell_images(patient.id, data['image_paths'])
                
            elif update_type == 'physician_notes':
                from patient_management_system.database.models_enhanced import PhysicianNote
                note = PhysicianNote(
                    patient_id=patient.id,
                    clinical_assessment=data.get('assessment', ''),
                    physician_name=data.get('physician', 'Unknown'),
                    severity_score=data.get('severity', 5)
                )
                self.db.add(note)
                self.db.commit()
                
            elif update_type == 'treatment_outcome':
                self._record_treatment_outcome(patient.id, data)
            
            logger.info(f"  Update completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False
    
    def _record_treatment_outcome(self, patient_id: int, outcome_data: Dict[str, Any]):
        """Record treatment outcome for learning"""
        
        # Find treatment
        treatment = (
            self.db.query(Treatment)
            .filter(Treatment.patient_id == patient_id)
            .order_by(Treatment.start_date.desc())
            .first()
        )
        
        if not treatment:
            logger.warning(f"No treatment found for patient {patient_id}")
            return
        
        # Create outcome record
        outcome = TreatmentOutcome(
            treatment_id=treatment.id,
            assessment_date=datetime.utcnow(),
            response_type=outcome_data.get('response_type', 'SD'),
            tumor_size_change_percent=outcome_data.get('tumor_change', 0.0),
            pfs_days=outcome_data.get('pfs_days', 0),
            qol_score=outcome_data.get('qol', 5.0)
        )
        
        self.db.add(outcome)
        self.db.commit()
        
        logger.info(f"  Recorded treatment outcome: {outcome.response_type}")
    
    def trigger_daily_learning(self) -> bool:
        """
        Trigger daily metadata aggregation and model training
        
        Returns:
            Success status
        """
        logger.info("Triggering daily learning cycle")
        
        try:
            # Import here to avoid circular dependency
            from patient_management_system.services.daily_ml_trainer import DailyMLTrainer
            
            # Create trainer
            trainer = DailyMLTrainer(model_dir=Path("models/pharmacodynamics"))
            
            # Run training
            trainer.run_daily_training()
            
            logger.info("Daily learning completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Daily learning failed: {e}")
            return False


if __name__ == "__main__":
    print("="*80)
    print("CDSS Orchestrator Test")
    print("="*80)
    
    orchestrator = CDSSOrchestrator()
    
    # Test patient creation
    test_patient = {
        'patient_id': 'TEST-ORCH-001',
        'name': 'Test Patient',
        'birthdate': datetime(1960, 1, 1),
        'gender': 'M'
    }
    
    result = orchestrator.process_new_patient(test_patient)
    print(f"\nPatient processing: {result['success']}")
    
    if result['success']:
        print(f"Assessment: {result['assessment']}")
