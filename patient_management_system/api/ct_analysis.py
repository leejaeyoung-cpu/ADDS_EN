"""
CT Analysis API
Endpoints for CT image upload, processing, and status tracking
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional
from pathlib import Path
from datetime import datetime
import shutil
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db
from database.models import Patient, CTAnalysis
from database.schemas import (
    CTAnalysisCreate,
    CTAnalysisUpdate,
    CTAnalysisResponse,
    CTAnalysisDetailResponse,
    AnalysisProgress,
    ClinicalMetadata
)

router = APIRouter()

# Directory for uploads
UPLOAD_DIR = Path(__file__).parent.parent.parent / "outputs" / "patient_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/{patient_id}/upload")
async def upload_ct_dicom(
    patient_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Upload DICOM file for CT analysis
    
    Args:
        patient_id: Patient identifier (e.g., P-2024-001)
        file: DICOM file
        
    Returns:
        Analysis ID and status
    """
    # Find patient
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Validate file type
    if not file.filename.endswith(('.dcm', '.DCM', '.dicom', '.DICOM')):
        raise HTTPException(status_code=400, detail="Only DICOM files are supported")
    
    # Create analysis record
    analysis = CTAnalysis(
        patient_id=patient.id,
        status="pending",
        progress=0,
        current_step="Uploading DICOM files",
        started_at=datetime.now()
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    # Save DICOM file
    patient_dir = UPLOAD_DIR / patient_id / f"analysis_{analysis.id}"
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    dicom_path = patient_dir / file.filename
    with open(dicom_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Update analysis with file path
    analysis.dicom_path = str(dicom_path)
    db.commit()
    
    # Start background processing
    if background_tasks:
        from services.ct_pipeline import CTPipelineService
        pipeline = CTPipelineService()
        background_tasks.add_task(pipeline.run_full_pipeline, analysis.id, str(dicom_path))
    
    return {
        "analysis_id": analysis.id,
        "patient_id": patient_id,
        "status": "uploaded",
        "message": "DICOM file uploaded successfully. Processing will start shortly."
    }


@router.get("/analysis/{analysis_id}/status", response_model=AnalysisProgress)
async def get_analysis_status(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get CT analysis processing status
    
    Args:
        analysis_id: Analysis ID
        
    Returns:
        Current processing status and progress
    """
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {
        "status": analysis.status,
        "progress": analysis.progress,
        "current_step": analysis.current_step,
        "error_message": analysis.error_message
    }


@router.post("/analysis/{analysis_id}/metadata")
async def add_clinical_metadata(
    analysis_id: int,
    metadata: ClinicalMetadata,
    db: Session = Depends(get_db)
):
    """
    Add clinical metadata to CT analysis
    
    Args:
        analysis_id: Analysis ID
        metadata: Clinical metadata (TNM stage, MSI status, etc.)
        
    Returns:
        Updated analysis
    """
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Update metadata fields
    if metadata.tumor_location:
        analysis.tumor_location = metadata.tumor_location
    if metadata.tnm_stage:
        analysis.tnm_stage = metadata.tnm_stage
    if metadata.msi_status:
        analysis.msi_status = metadata.msi_status.value
    if metadata.kras_mutation:
        analysis.kras_mutation = metadata.kras_mutation.value
    if metadata.additional_notes:
        analysis.additional_notes = metadata.additional_notes
    
    db.commit()
    db.refresh(analysis)
    
    return {
        "analysis_id": analysis.id,
        "status": "metadata_updated",
        "metadata": {
            "tumor_location": analysis.tumor_location,
            "tnm_stage": analysis.tnm_stage,
            "msi_status": analysis.msi_status,
            "kras_mutation": analysis.kras_mutation
        }
    }


@router.get("/analysis/{analysis_id}", response_model=CTAnalysisDetailResponse)
async def get_analysis_details(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get complete CT analysis details including results
    
    Args:
        analysis_id: Analysis ID
        
    Returns:
        Complete analysis with radiomics, tumor characteristics, and inference results
    """
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis


@router.delete("/analysis/{analysis_id}")
async def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete CT analysis
    
    Args:
        analysis_id: Analysis ID
    """
    analysis = db.query(CTAnalysis).filter(CTAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Delete files
    if analysis.dicom_path:
        dicom_path = Path(analysis.dicom_path)
        if dicom_path.exists():
            # Delete entire analysis directory
            analysis_dir = dicom_path.parent
            shutil.rmtree(analysis_dir, ignore_errors=True)
    
    # Delete database record
    db.delete(analysis)
    db.commit()
    
    return {"status": "deleted", "analysis_id": analysis_id}
