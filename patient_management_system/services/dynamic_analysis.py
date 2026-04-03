"""
Dynamic Analysis Update System

Automatically re-analyzes patient data when:
1. New CT images are uploaded
2. Physician notes are added/updated
3. Cell images are uploaded
4. Treatment outcomes are recorded

This ensures the system always has the latest analysis.
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from patient_management_system.database.db_enhanced import get_session
from patient_management_system.database.models_enhanced import (
    Patient, CTAnalysis, PhysicianNote, CellImage
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientDataWatcher(FileSystemEventHandler):
    """Watches for new patient data files and triggers analysis"""
    
    def __init__(self, base_dir: Path, analysis_trigger_callback):
        """
        Args:
            base_dir: Base directory to watch
            analysis_trigger_callback: Function to call when new data is detected
        """
        self.base_dir = Path(base_dir)
        self.analysis_trigger = analysis_trigger_callback
        self.debounce_time = 2  # seconds
        self.last_trigger: Dict[str, float] = {}
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check file type
        if self._is_ct_data(file_path):
            self._trigger_ct_analysis(file_path)
        elif self._is_cell_image(file_path):
            self._trigger_cell_image_processing(file_path)
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Handle metadata files
        if file_path.suffix in ['.json', '.txt']:
            self._check_metadata_update(file_path)
    
    def _is_ct_data(self, file_path: Path) -> bool:
        """Check if file is CT data"""
        return file_path.suffix.lower() in ['.dcm', '.nii', '.nii.gz']
    
    def _is_cell_image(self, file_path: Path) -> bool:
        """Check if file is cell image"""
        return file_path.suffix.lower() in ['.jpg', '.png', '.tif', '.tiff']
    
    def _trigger_ct_analysis(self, file_path: Path):
        """Trigger CT analysis for new CT data"""
        # Debounce
        key = f"ct_{file_path.parent}"
        current_time = time.time()
        
        if key in self.last_trigger:
            if current_time - self.last_trigger[key] < self.debounce_time:
                return
        
        self.last_trigger[key] = current_time
        
        logger.info(f"New CT data detected: {file_path}")
        
        # Extract patient ID from path
        patient_id = self._extract_patient_id(file_path)
        
        if patient_id:
            self.analysis_trigger('ct_analysis', patient_id, file_path)
    
    def _trigger_cell_image_processing(self, file_path: Path):
        """Trigger cell image processing"""
        logger.info(f"New cell image detected: {file_path}")
        
        patient_id = self._extract_patient_id(file_path)
        
        if patient_id:
            self.analysis_trigger('cell_image', patient_id, file_path)
    
    def _check_metadata_update(self, file_path: Path):
        """Check for metadata updates that require re-analysis"""
        if 'notes' in str(file_path).lower():
            logger.info(f"Physician notes updated: {file_path}")
            patient_id = self._extract_patient_id(file_path)
            
            if patient_id:
                self.analysis_trigger('notes_update', patient_id, file_path)
    
    def _extract_patient_id(self, file_path: Path) -> Optional[str]:
        """
        Extract patient ID from file path
        
        Assumes structure like: .../patients/PATIENT_ID/...
        """
        parts = file_path.parts
        
        try:
            if 'patients' in parts:
                idx = parts.index('patients')
                if idx + 1 < len(parts):
                    return parts[idx + 1]
        except Exception as e:
            logger.error(f"Could not extract patient ID: {e}")
        
        return None


class AnalysisTriggerService:
    """Service to handle analysis triggers"""
    
    def __init__(self):
        self.db = get_session()
    
    def trigger_analysis(self, analysis_type: str, patient_id: str, file_path: Path):
        """
        Trigger appropriate analysis based on type
        
        Args:
            analysis_type: Type of analysis to trigger
            patient_id: Patient identifier
            file_path: Path to new data file
        """
        logger.info(f"Triggering {analysis_type} for patient {patient_id}")
        
        try:
            if analysis_type == 'ct_analysis':
                self._trigger_ct_analysis(patient_id, file_path)
            elif analysis_type == 'cell_image':
                self._process_cell_image(patient_id, file_path)
            elif analysis_type == 'notes_update':
                self._handle_notes_update(patient_id, file_path)
        except Exception as e:
            logger.error(f"Error triggering analysis: {e}")
    
    def _trigger_ct_analysis(self, patient_id: str, file_path: Path):
        """Trigger CT analysis pipeline"""
        # Find or create patient
        patient = self.db.query(Patient).filter(Patient.patient_id == patient_id).first()
        
        if not patient:
            logger.warning(f"Patient {patient_id} not found - creating placeholder")
            # Auto-create patient entry
            patient = Patient(
                patient_id=patient_id,
                name=f"Auto-generated-{patient_id}",
                birthdate=datetime.now(),
                gender="M"  # Default
            )
            self.db.add(patient)
            self.db.commit()
        
        # Create new CT analysis entry
        new_analysis = CTAnalysis(
            patient_id=patient.id,
            status="pending",
            dicom_directory=str(file_path.parent)
        )
        
        self.db.add(new_analysis)
        self.db.commit()
        
        logger.info(f"Created CT analysis #{new_analysis.id} for patient {patient_id}")
        
        # Trigger actual CT analysis pipeline
        try:
            # Import orchestrator to handle processing
            from patient_management_system.services.cdss_orchestrator import CDSSOrchestrator
            orchestrator = CDSSOrchestrator()
            
            # Update with new CT data
            orchestrator.update_patient_data(
                patient_id,
                'ct_scan',
                {'ct_directory': str(file_path.parent)}
            )
            
            logger.info(f"  CT analysis pipeline triggered successfully")
        except Exception as e:
            logger.error(f"  Failed to trigger CT pipeline: {e}")
            new_analysis.status = "failed"
            new_analysis.error_message = str(e)
            self.db.commit()
    
    def _process_cell_image(self, patient_id: str, file_path: Path):
        """Process new cell image"""
        patient = self.db.query(Patient).filter(Patient.patient_id == patient_id).first()
        
        if patient:
            # Create cell image record
            cell_image = CellImage(
                patient_id=patient.id,
                image_path=str(file_path),
                image_type="auto-detected"
            )
            
            self.db.add(cell_image)
            self.db.commit()
            
            logger.info(f"Registered cell image for patient {patient_id}")
            
            # Trigger CNN feature extraction
            try:
                from patient_management_system.services.cdss_orchestrator import CDSSOrchestrator
                orchestrator = CDSSOrchestrator()
                
                orchestrator.update_patient_data(
                    patient_id,
                    'cell_images',
                    {'image_paths': [str(file_path)]}
                )
                
                logger.info(f"  Cell image feature extraction triggered")
            except Exception as e:
                logger.error(f"  Feature extraction failed: {e}")
    
    def _handle_notes_update(self, patient_id: str, file_path: Path):
        """Handle physician notes update with NLP parsing"""
        patient = self.db.query(Patient).filter(Patient.patient_id == patient_id).first()
        
        if patient:
            logger.info(f"Processing physician notes for patient {patient_id}")
            
            try:
                # Read notes file
                with open(file_path, 'r', encoding='utf-8') as f:
                    notes_content = f.read()
                
                # Parse notes with NLP
                from patient_management_system.services.nlp_parser import PhysicianNotesParser
                parser = PhysicianNotesParser()
                parsed_data = parser.parse(notes_content)
                
                # Save parsed notes to database
                from patient_management_system.database.models_enhanced import PhysicianNote
                
                note = PhysicianNote(
                    patient_id=patient.id,
                    clinical_assessment=notes_content,
                    physician_name="Auto-parsed",
                    severity_score=parsed_data['severity']['score'],
                    note_date=datetime.now()
                )
                
                self.db.add(note)
                self.db.commit()
                
                logger.info(f"  Severity: {parsed_data['severity']['level']} ({parsed_data['severity']['score']}/10)")
                logger.info(f"  Symptoms detected: {len(parsed_data['symptoms'])}")
                logger.info(f"  Tumor status: {parsed_data['tumor_status']['status']}")
                
                # Check if re-analysis needed
                if parsed_data['requires_reanalysis']:
                    logger.warning(f"  ⚠️ Notes indicate need for re-analysis!")
                    
                    # Get latest CT analysis
                    latest_analysis = (
                        self.db.query(CTAnalysis)
                        .filter(CTAnalysis.patient_id == patient.id)
                        .order_by(CTAnalysis.analysis_date.desc())
                        .first()
                    )
                    
                    if latest_analysis:
                        # Check age of analysis
                        if latest_analysis.analysis_date:
                            age_days = (datetime.now() - latest_analysis.analysis_date).days
                            
                            # Trigger if >3 days old OR tumor growth detected
                            tumor_growth = parsed_data['tumor_status']['status'] == 'growth'
                            
                            if age_days > 3 or tumor_growth:
                                logger.info(f"  🔄 Triggering CT re-analysis (age: {age_days} days, growth: {tumor_growth})")
                                self._trigger_ct_analysis(patient_id, Path(latest_analysis.dicom_directory))
                
            except Exception as e:
                logger.error(f"Failed to parse physician notes: {e}")


def start_file_watcher(watch_directory: Path):
    """
    Start file system watcher
    
    Args:
        watch_directory: Directory to monitor
    """
    watch_directory = Path(watch_directory)
    
    if not watch_directory.exists():
        logger.warning(f"Watch directory does not exist: {watch_directory}")
        watch_directory.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting file watcher on: {watch_directory}")
    
    # Create trigger service
    trigger_service = AnalysisTriggerService()
    
    # Create event handler
    event_handler = PatientDataWatcher(
        watch_directory,
        trigger_service.trigger_analysis
    )
    
    # Create observer
    observer = Observer()
    observer.schedule(event_handler, str(watch_directory), recursive=True)
    observer.start()
    
    logger.info("File watcher started successfully")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("File watcher stopped")
    
    observer.join()


if __name__ == "__main__":
    # Example usage
    watch_dir = Path("F:/ADDS/patient_data")
    
    print("=" * 80)
    print("CDSS Dynamic Analysis Watcher")
    print("=" * 80)
    print(f"\nWatching: {watch_dir}")
    print("\nThis service will automatically:")
    print("  - Detect new CT scans → Trigger analysis")
    print("  - Detect new cell images → Extract features")
    print("  - Detect notes updates → Consider re-analysis")
    print("\nPress Ctrl+C to stop")
    print("=" * 80 + "\n")
    
    start_file_watcher(watch_dir)
