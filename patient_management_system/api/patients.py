"""
Patient Management API
CRUD operations for patients
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db
from database.models import Patient
from database.schemas import (
    PatientCreate,
    PatientUpdate,
    PatientResponse,
    PatientSearchResult
)

router = APIRouter()


def generate_patient_id(db: Session) -> str:
    """Generate unique patient ID in format P-YYYY-NNN"""
    year = datetime.now().year
    
    # Count patients created this year
    count = db.query(Patient).filter(
        Patient.patient_id.like(f"P-{year}-%")
    ).count()
    
    return f"P-{year}-{count + 1:03d}"


@router.post("/register", response_model=PatientResponse, status_code=201)
async def register_patient(
    patient: PatientCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new patient
    
    Returns:
        PatientResponse: Created patient information
    """
    # Generate patient ID
    patient_id = generate_patient_id(db)
    
    # Create patient record
    db_patient = Patient(
        patient_id=patient_id,
        name=patient.name,
        birthdate=patient.birthdate,
        gender=patient.gender.value,
        contact=patient.contact,
        address=patient.address
    )
    
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    
    return db_patient


@router.get("/search", response_model=PatientSearchResult)
async def search_patients(
    query: str = Query(..., min_length=1, description="Search by name or patient ID"),
    db: Session = Depends(get_db)
):
    """
    Search patients by name or patient ID
    
    Args:
        query: Search term
        
    Returns:
        PatientSearchResult: List of matching patients
    """
    # Search by patient_id or name
    patients = db.query(Patient).filter(
        (Patient.patient_id.contains(query)) |
        (Patient.name.contains(query))
    ).order_by(Patient.created_at.desc()).all()
    
    return {
        "patients": patients,
        "total": len(patients)
    }


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Get patient details by patient ID
    
    Args:
        patient_id: Patient identifier (e.g., P-2024-001)
        
    Returns:
        PatientResponse: Patient information
    """
    patient = db.query(Patient).filter(
        Patient.patient_id == patient_id
    ).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return patient


@router.get("/{patient_id}/history")
async def get_patient_history(
    patient_id: str,
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get analysis history for a patient
    
    Args:
        patient_id: Patient identifier
        limit: Maximum number of analyses to return
        
    Returns:
        AnalysisHistoryResult: List of CT analyses
    """
    from ..database.models import CTAnalysis
    
    # Get patient
    patient = db.query(Patient).filter(
        Patient.patient_id == patient_id
    ).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get analyses
    analyses = db.query(CTAnalysis).filter(
        CTAnalysis.patient_id == patient.id
    ).order_by(CTAnalysis.analysis_date.desc()).limit(limit).all()
    
    return {
        "analyses": [analysis.to_dict() for analysis in analyses],
        "total": len(analyses)
    }


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: str,
    patient_update: PatientUpdate,
    db: Session = Depends(get_db)
):
    """
    Update patient information
    
    Args:
        patient_id: Patient identifier
        patient_update: Fields to update
        
    Returns:
        PatientResponse: Updated patient information
    """
    patient = db.query(Patient).filter(
        Patient.patient_id == patient_id
    ).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Update fields
    if patient_update.name is not None:
        patient.name = patient_update.name
    if patient_update.contact is not None:
        patient.contact = patient_update.contact
    if patient_update.address is not None:
        patient.address = patient_update.address
    
    db.commit()
    db.refresh(patient)
    
    return patient


@router.delete("/{patient_id}", status_code=204)
async def delete_patient(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a patient and all associated analyses
    
    Args:
        patient_id: Patient identifier
    """
    patient = db.query(Patient).filter(
        Patient.patient_id == patient_id
    ).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    db.delete(patient)
    db.commit()
    
    return None


@router.get("/", response_model=List[PatientResponse])
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    List all patients with pagination
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List[PatientResponse]: List of patients
    """
    patients = db.query(Patient).order_by(
        Patient.created_at.desc()
    ).offset(skip).limit(limit).all()
    
    return patients
