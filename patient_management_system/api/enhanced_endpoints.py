"""
Enhanced Patient Management API

Adds new endpoints for:
- Physician notes
- Cell images
- Treatments
- Treatment outcomes
- Comprehensive patient profiles
- System metrics
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import shutil

from patient_management_system.database.db_enhanced import get_db
from patient_management_system.database.models_enhanced import (
    Patient, CTAnalysis, PhysicianNote, CellImage, 
    Treatment, TreatmentOutcome, SideEffect
)
from patient_management_system.database.schemas_enhanced import (
    PhysicianNoteCreate, PhysicianNoteResponse,
    CellImageCreate, CellImageResponse,
    TreatmentCreate, TreatmentResponse,
    TreatmentOutcomeCreate, TreatmentOutcomeResponse,
    SideEffectCreate, SideEffectResponse,
    PatientProfile, SystemMetrics
)

router = APIRouter(prefix="/api/v1/enhanced", tags=["Enhanced Patient Management"])


# ========== Physician Notes ==========
@router.post("/patients/{patient_id}/notes", response_model=PhysicianNoteResponse)
def add_physician_note(
    patient_id: int,
    note: PhysicianNoteCreate,
    db: Session = Depends(get_db)
):
    """Add physician note for a patient"""
    # Verify patient exists
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Create note
    db_note = PhysicianNote(
        patient_id=patient_id,
        **note.dict()
    )
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    
    return db_note


@router.get("/patients/{patient_id}/notes", response_model=List[PhysicianNoteResponse])
def get_physician_notes(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get all physician notes for a patient"""
    return db.query(PhysicianNote).filter(PhysicianNote.patient_id == patient_id).all()


# ========== Cell Images ==========
@router.post("/patients/{patient_id}/cell-images", response_model=CellImageResponse)
async def upload_cell_image(
    patient_id: int,
    file: UploadFile = File(...),
    image_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Upload cell image for a patient"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Save file
    upload_dir = Path(f"uploads/cell_images/{patient.patient_id}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create database record
    db_image = CellImage(
        patient_id=patient_id,
        image_path=str(file_path),
        image_type=image_type or "unknown"
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    
    return db_image


@router.get("/patients/{patient_id}/cell-images", response_model=List[CellImageResponse])
def get_cell_images(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get all cell images for a patient"""
    return db.query(CellImage).filter(CellImage.patient_id == patient_id).all()


# ========== Treatments ==========
@router.post("/patients/{patient_id}/treatments", response_model=TreatmentResponse)
def create_treatment(
    patient_id: int,
    treatment: TreatmentCreate,
    db: Session = Depends(get_db)
):
    """Create new treatment record"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Convert Pydantic models to dicts
    drug_cocktail = [drug.dict() for drug in treatment.drug_cocktail]
    
    db_treatment = Treatment(
        patient_id=patient_id,
        start_date=treatment.start_date,
        treatment_name=treatment.treatment_name,
        drug_cocktail=drug_cocktail,
        regimen=treatment.regimen,
        cycles_planned=treatment.cycles_planned,
        ct_analysis_id=treatment.ct_analysis_id
    )
    db.add(db_treatment)
    db.commit()
    db.refresh(db_treatment)
    
    return db_treatment


@router.get("/patients/{patient_id}/treatments", response_model=List[TreatmentResponse])
def get_treatments(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get all treatments for a patient"""
    return db.query(Treatment).filter(Treatment.patient_id == patient_id).all()


# ========== Treatment Outcomes ==========
@router.post("/treatments/{treatment_id}/outcomes", response_model=TreatmentOutcomeResponse)
def add_treatment_outcome(
    treatment_id: int,
    outcome: TreatmentOutcomeCreate,
    db: Session = Depends(get_db)
):
    """Add outcome to a treatment"""
    treatment = db.query(Treatment).filter(Treatment.id == treatment_id).first()
    if not treatment:
        raise HTTPException(status_code=404, detail="Treatment not found")
    
    db_outcome = TreatmentOutcome(
        treatment_id=treatment_id,
        **outcome.dict()
    )
    db.add(db_outcome)
    db.commit()
    db.refresh(db_outcome)
    
    return db_outcome


# ========== Comprehensive Patient Profile ==========
@router.get("/patients/{patient_id}/profile", response_model=PatientProfile)
def get_patient_profile(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get comprehensive patient profile with all metadata"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get latest CT
    latest_ct = (
        db.query(CTAnalysis)
        .filter(CTAnalysis.patient_id == patient_id)
        .order_by(CTAnalysis.analysis_date.desc())
        .first()
    )
    
    # Get all related data
    notes = db.query(PhysicianNote).filter(PhysicianNote.patient_id == patient_id).all()
    images = db.query(CellImage).filter(CellImage.patient_id == patient_id).all()
    treatments = db.query(Treatment).filter(Treatment.patient_id == patient_id).all()
    
    # Get all outcomes
    all_outcomes = []
    for t in treatments:
        outcomes = db.query(TreatmentOutcome).filter(TreatmentOutcome.treatment_id == t.id).all()
        all_outcomes.extend(outcomes)
    
    # Calculate age
    age = (datetime.now() - patient.birthdate).days // 365
    
    # Data completeness
    completeness_factors = [
        1 if latest_ct else 0,
        1 if notes else 0,
        1 if images else 0,
        1 if treatments else 0,
        1 if all_outcomes else 0
    ]
    completeness = sum(completeness_factors) / len(completeness_factors)
    
    return {
        "patient_id": patient.patient_id,
        "name": patient.name,
        "age": age,
        "gender": patient.gender,
        "latest_ct": latest_ct,
        "physician_notes": notes,
        "cell_images": images,
        "treatments": treatments,
        "treatment_outcomes": all_outcomes,
        "total_treatments": len(treatments),
        "has_ct_data": latest_ct is not None,
        "has_outcome_data": len(all_outcomes) > 0,
        "data_completeness_score": completeness
    }


# ========== System Metrics ==========
@router.get("/metrics/system", response_model=SystemMetrics)
def get_system_metrics(db: Session = Depends(get_db)):
    """Get overall system performance metrics"""
    from patient_management_system.database.models_enhanced import MLTrainingRun, MetadataSnapshot
    
    total_patients = db.query(Patient).count()
    total_analyses = db.query(CTAnalysis).count()
    total_treatments = db.query(Treatment).count()
    
    # Patients with outcomes
    patients_with_outcomes = db.query(Patient.id).join(Treatment).join(TreatmentOutcome).distinct().count()
    
    # Data completeness
    all_patients = db.query(Patient).all()
    if all_patients:
        completeness_scores = []
        for p in all_patients:
            has_ct = db.query(CTAnalysis).filter(CTAnalysis.patient_id == p.id).first() is not None
            has_treatment = db.query(Treatment).filter(Treatment.patient_id == p.id).first() is not None
            has_outcome = db.query(Treatment).filter(Treatment.patient_id == p.id).join(TreatmentOutcome).first() is not None
            score = sum([has_ct, has_treatment, has_outcome]) / 3
            completeness_scores.append(score)
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
    else:
        avg_completeness = 0.0
    
    # Model metrics
    deployed_model = (
        db.query(MLTrainingRun)
        .filter(MLTrainingRun.is_deployed == True)
        .order_by(MLTrainingRun.deployed_at.desc())
        .first()
    )
    
    model_version = None
    model_accuracy = None
    if deployed_model:
        model_version = f"v{deployed_model.id}"
        model_accuracy = deployed_model.val_metrics.get('val_accuracy') if deployed_model.val_metrics else None
    
    return {
        "total_patients": total_patients,
        "total_analyses": total_analyses,
        "total_treatments": total_treatments,
        "avg_data_completeness": avg_completeness,
        "patients_with_outcomes": patients_with_outcomes,
        "current_model_version": model_version,
        "model_accuracy": model_accuracy,
        "accuracy_improvement_percent": None,  # TODO: Calculate from history
        "metadata_usage_impact": None  # TODO: Calculate A/B test results
    }
